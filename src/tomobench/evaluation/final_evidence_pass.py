"""Reproducible evidence pass for the final evidence pass.

This module is intentionally separate from the frozen Benchmark v2 analysis.  It
reuses the immutable 250-target production corpus and the source-backed forward
operator audit, then writes a versioned evidence bundle.  All model selection is
validation-only; the 38-target test set is read only after a configuration has
been frozen.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tomobench.config import load_settings
from tomobench.evaluation.benchmark_v2_final_analysis import (
    FEATURE_VARIANTS,
    ObservationPCAModel,
    ProductionCase,
    RayGeometry,
    build_feature_matrices,
    cell_centered_values_from_node_vector,
    fit_observation_pca_model,
    fit_target_pca,
    load_production_cases,
    solve_fixed_ray_case,
    reconstruct_from_pca,
    _node_shape,
    _prediction_metrics_from_arrays,
    _family_parameters,
    _read_json,
    _stable_seed,
)
from tomobench.evaluation.robustness_checks import (
    _load_geometry,
    direct_analytic_cell_truth,
)
from tomobench.evaluation.benchmark_v2_final_analysis import build_structure_masks
from tomobench.tomography.reference import build_layered_reference_velocity_grid
from tomobench.utils.paths import get_repo_root


REVISION_RELATIVE = Path(
    "outputs/generated/submission_readiness/final_evidence_pass_v2"
)
PRODUCTION_RELATIVE = Path(
    "outputs/generated/submission_readiness/benchmark_v2_production_250_v1"
)
OPERATOR_RELATIVE = Path(
    "outputs/generated/submission_readiness/robustness_checks/operator_consistency"
)
# Frozen seed label. `_stable_seed` hashes this text with SHA-256 to derive the
# generator seed, so the string is part of the experiment's definition rather than
# its documentation: changing it changes every noise draw and therefore every
# published noise-robustness value. It is kept verbatim for that reason alone.
FROZEN_NOISE_SEED_LABEL = "reviewer-round-noise"

CANONICAL_ENDPOINT_RECORD = "records/forward_solver_endpoint_contract.md"
FAMILY_ORDER = ("block_anomaly", "dyke_intrusion", "faulted", "layered", "salt_dome")
PUBLIC_METHOD_LABELS = {
    "realistic_full": "Full-input PCA-ridge",
    "travel_time_only": "Travel-time-only PCA-ridge",
    "no_travel_time": "No-travel-time PCA-ridge",
    "geometry_only": "Geometry-only PCA-ridge",
    "distance_only": "Distance-only PCA-ridge",
    "shuffled_travel_time": "Shuffled-travel-time PCA-ridge",
    "training_target_mean": "Training-target mean",
    "reference_ray": "Reference-ray baseline",
    "reference_prior": "Reference 1-D prior",
    "cell_native": "Cell-native PCA-ridge sensitivity",
}


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else get_repo_root() / value


def load_revision_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the centrally stored evidence-pass grid and seed contract."""

    config_path = _repo_path(path or "config/final_evidence_pass.json")
    payload = _read_json(config_path)
    required = (
        "pca_component_grid",
        "ridge_alpha_grid",
        "reference_damping_grid",
        "reference_smoothing_grid",
        "shuffled_time_seed_start",
        "shuffled_time_seed_count",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Evidence-pass configuration is missing keys: {missing}")
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json_local(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_csv_local(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            values: dict[str, Any] = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, (np.ndarray, list, tuple, dict)):
                    value = json.dumps(value, sort_keys=True, default=_json_default)
                elif isinstance(value, (np.integer, np.floating, np.bool_)):
                    value = value.item()
                values[field] = value
            writer.writerow(values)


def _read_csv_local(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rmse(true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(true)) ** 2)))


def _mae(true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(true))))


def _cell_prediction(case: ProductionCase, node_prediction: np.ndarray) -> np.ndarray:
    return cell_centered_values_from_node_vector(node_prediction, _node_shape(case.target_grid))


def _truth_cells(cases: Sequence[ProductionCase]) -> np.ndarray:
    return np.vstack([direct_analytic_cell_truth(case) for case in cases])


def _target_level_summary(values: Sequence[float], seed_label: str, bootstrap_resamples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty target-level vector.")
    rng = np.random.default_rng(_stable_seed(seed_label))
    sample = rng.choice(array, size=(bootstrap_resamples, array.size), replace=True)
    means = np.mean(sample, axis=1)
    return {
        "target_count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sd_population": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_unit": "unique geological target",
        "ci95_lower": float(np.percentile(means, 2.5)),
        "ci95_upper": float(np.percentile(means, 97.5)),
    }


def _family_stratified_ci(
    values: Sequence[float],
    families: Sequence[str],
    seed_label: str,
    bootstrap_resamples: int,
) -> tuple[float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for family, value in zip(families, values, strict=True):
        grouped[str(family)].append(float(value))
    rng = np.random.default_rng(_stable_seed(seed_label))
    family_arrays = [np.asarray(grouped[key], dtype=float) for key in sorted(grouped)]
    draws = [rng.choice(array, size=bootstrap_resamples, replace=True) for array in family_arrays]
    means = np.mean(np.vstack(draws), axis=0)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    delta_field: str = "paired_delta",
    tie_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty paired target vector.")
    deltas = np.asarray([float(row[delta_field]) for row in rows], dtype=float)
    families = [str(row["family"]) for row in rows]
    tolerance = float(tie_tolerance)
    summary = _target_level_summary(
        deltas,
        "paired:" + str(rows[0].get("comparison_id", "comparison")),
        bootstrap_resamples,
    )
    lower, upper = summary["ci95_lower"], summary["ci95_upper"]
    family_lower, family_upper = _family_stratified_ci(
        deltas, families, "paired-family:" + str(rows[0].get("comparison_id", "comparison")), bootstrap_resamples
    )
    return {
        "comparison_id": str(rows[0].get("comparison_id", "comparison")),
        "target_count": int(deltas.size),
        "delta_definition": "first method RMSE minus second method RMSE; negative favors the first method",
        "bootstrap_resamples": int(bootstrap_resamples),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "sd_population": float(np.std(deltas)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "family_stratified_ci95_lower": float(family_lower),
        "family_stratified_ci95_upper": float(family_upper),
        "wins": int(np.count_nonzero(deltas < -tolerance)),
        "losses": int(np.count_nonzero(deltas > tolerance)),
        "ties": int(np.count_nonzero(np.abs(deltas) <= tolerance)),
        "tie_tolerance": tolerance,
        "bootstrap_unit": "unique geological target",
        "ci_interpretation": "exploratory target-level bootstrap interval unless comparison_id is the primary endpoint",
    }


def _paired_metric_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    delta_field: str,
    metric_label: str,
    bootstrap_resamples: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    """Summarize a paired target-level metric with pooled and family-stratified CIs."""

    if not rows:
        raise ValueError("Cannot summarize an empty paired target vector.")
    deltas = np.asarray([float(row[delta_field]) for row in rows], dtype=float)
    families = [str(row["family"]) for row in rows]
    comparison_id = str(rows[0].get("comparison_id", "comparison"))
    target_summary = _target_level_summary(
        deltas,
        f"paired:{comparison_id}:{metric_label.lower()}",
        bootstrap_resamples,
    )
    family_lower, family_upper = _family_stratified_ci(
        deltas,
        families,
        f"paired-family:{comparison_id}:{metric_label.lower()}",
        bootstrap_resamples,
    )
    tolerance = float(tie_tolerance)
    return {
        "metric": f"direct-cell {metric_label}",
        "comparison_id": comparison_id,
        "target_count": int(deltas.size),
        "delta_definition": f"first method direct-cell {metric_label} minus second method direct-cell {metric_label}; negative favors the first method",
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_unit": "unique geological target",
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "sd_population": float(np.std(deltas)),
        "ci95_lower": float(target_summary["ci95_lower"]),
        "ci95_upper": float(target_summary["ci95_upper"]),
        "family_stratified_ci95_lower": float(family_lower),
        "family_stratified_ci95_upper": float(family_upper),
        "wins": int(np.count_nonzero(deltas < -tolerance)),
        "losses": int(np.count_nonzero(deltas > tolerance)),
        "ties": int(np.count_nonzero(np.abs(deltas) <= tolerance)),
        "tie_tolerance": tolerance,
        "ci_interpretation": "exploratory target-level bootstrap interval unless comparison_id is the primary endpoint",
    }


def _metric_row(
    case: ProductionCase,
    method_id: str,
    prediction_nodes: np.ndarray | None,
    prediction_cells: np.ndarray,
    split: str,
) -> dict[str, Any]:
    direct = direct_analytic_cell_truth(case)
    cell_metrics = _prediction_metrics_from_arrays(direct, prediction_cells)
    node_metrics: dict[str, Any] = {"rmse": None, "mae": None, "bias": None}
    if prediction_nodes is not None:
        node_metrics = _prediction_metrics_from_arrays(case.target_vector, prediction_nodes)
    return {
        "target_id": case.target_id,
        "target_hash": case.target_hash,
        "family": case.family,
        "split": split,
        "method_id": method_id,
        "method_label": PUBLIC_METHOD_LABELS.get(method_id, method_id),
        "node_rmse_km_per_s": node_metrics["rmse"],
        "node_mae_km_per_s": node_metrics["mae"],
        "node_bias_km_per_s": node_metrics["bias"],
        "direct_cell_rmse_km_per_s": cell_metrics["rmse"],
        "direct_cell_mae_km_per_s": cell_metrics["mae"],
        "direct_cell_bias_km_per_s": cell_metrics["bias"],
    }


def _candidate_metrics(
    cases: Sequence[ProductionCase],
    node_prediction: np.ndarray,
) -> tuple[float, float, float, float]:
    node_rows: list[float] = []
    node_mae: list[float] = []
    cell_rows: list[float] = []
    cell_mae: list[float] = []
    for case, prediction in zip(cases, node_prediction, strict=True):
        node = _prediction_metrics_from_arrays(case.target_vector, prediction)
        cell = _prediction_metrics_from_arrays(direct_analytic_cell_truth(case), _cell_prediction(case, prediction))
        node_rows.append(float(node["rmse"]))
        node_mae.append(float(node["mae"]))
        cell_rows.append(float(cell["rmse"]))
        cell_mae.append(float(cell["mae"]))
    return float(np.mean(node_rows)), float(np.mean(node_mae)), float(np.mean(cell_rows)), float(np.mean(cell_mae))


def _valid_components(pca: Any, grid: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(k) for k in grid if int(k) <= int(pca.rank))


def _write_prediction_npz(path: Path, cases: Sequence[ProductionCase], predictions: np.ndarray, field_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        target_ids=np.asarray([case.target_id for case in cases]),
        target_hashes=np.asarray([case.target_hash for case in cases]),
        **{field_name: np.asarray(predictions, dtype=float)},
    )


def _fit_variant_model(
    feature_train: np.ndarray,
    targets: np.ndarray,
    pca: Any,
    selected: Mapping[str, Any],
    bounds: tuple[float, float],
) -> ObservationPCAModel:
    alpha = selected.get("ridge_alpha")
    alpha_value = None if alpha in (None, "") else float(alpha)
    return fit_observation_pca_model(
        feature_train,
        targets,
        pca,
        int(selected["pca_component_count"]),
        str(selected["model_name"]),
        alpha_value,
        bounds,
    )


def _selection_key(row: Mapping[str, Any], *, primary: str) -> tuple[Any, ...]:
    alpha = row.get("ridge_alpha")
    alpha_key = 0.0 if alpha in (None, "") else float(alpha)
    if primary == "node":
        return (
            float(row["validation_node_rmse_mean_km_per_s"]),
            float(row["validation_node_mae_mean_km_per_s"]),
        ) + (
            0 if str(row["model_name"]) == "pca_ridge" else 1,
            int(row["pca_component_count"]),
            alpha_key,
        )
    return (
        float(row["validation_direct_cell_rmse_mean_km_per_s"]),
        float(row["validation_direct_cell_mae_mean_km_per_s"]),
        0 if str(row["model_name"]) == "pca_ridge" else 1,
        int(row["pca_component_count"]),
        alpha_key,
    )


def _write_pca_spectrum(pca: Any, path: Path, *, basis: str, grid: Sequence[int]) -> None:
    cumulative = np.cumsum(pca.explained_variance_ratio)
    rows = [
        {
            "component": index,
            "singular_value": float(pca.singular_values[index - 1]),
            "explained_variance_ratio": float(pca.explained_variance_ratio[index - 1]),
            "cumulative_explained_variance": float(cumulative[index - 1]),
            "basis": basis,
            "candidate_grid": list(grid),
        }
        for index in range(1, int(pca.rank) + 1)
    ]
    _write_csv_local(rows, path)


def run_node_native_analysis(
    cases: Sequence[ProductionCase],
    output_root: Path,
    config: Mapping[str, Any],
    settings: Any,
) -> dict[str, Any]:
    """Fit the complete node-native workflow and its frozen-input ablations."""

    train = [case for case in cases if case.split == "train"]
    validation = [case for case in cases if case.split == "validation"]
    test = [case for case in cases if case.split == "test"]
    bounds = tuple(float(value) for value in settings.velocity_model_generation.velocity_bounds_km_per_s)
    target_train = np.vstack([case.target_vector for case in train])
    pca = fit_target_pca(target_train)
    features = build_feature_matrices(cases)
    feature_train = features["realistic_full"][[case.split == "train" for case in cases]]
    feature_validation = features["realistic_full"][[case.split == "validation" for case in cases]]
    candidate_rows: list[dict[str, Any]] = []
    component_grid = _valid_components(pca, config["pca_component_grid"])
    for component_count in component_grid:
        linear = fit_observation_pca_model(feature_train, target_train, pca, component_count, "pca_linear", None, bounds)
        prediction = linear.predict(feature_validation)
        node_rmse, node_mae, cell_rmse, cell_mae = _candidate_metrics(validation, prediction)
        candidate_rows.append({
            "feature_variant": "realistic_full",
            "model_name": "pca_linear",
            "pca_component_count": component_count,
            "ridge_alpha": None,
            "validation_target_count": len(validation),
            "validation_node_rmse_mean_km_per_s": node_rmse,
            "validation_node_mae_mean_km_per_s": node_mae,
            "validation_direct_cell_rmse_mean_km_per_s": cell_rmse,
            "validation_direct_cell_mae_mean_km_per_s": cell_mae,
            "selection_metric": "validation mean node RMSE",
        })
        for alpha in config["ridge_alpha_grid"]:
            model = fit_observation_pca_model(feature_train, target_train, pca, component_count, "pca_ridge", float(alpha), bounds)
            prediction = model.predict(feature_validation)
            node_rmse, node_mae, cell_rmse, cell_mae = _candidate_metrics(validation, prediction)
            candidate_rows.append({
                "feature_variant": "realistic_full",
                "model_name": "pca_ridge",
                "pca_component_count": component_count,
                "ridge_alpha": float(alpha),
                "validation_target_count": len(validation),
                "validation_node_rmse_mean_km_per_s": node_rmse,
                "validation_node_mae_mean_km_per_s": node_mae,
                "validation_direct_cell_rmse_mean_km_per_s": cell_rmse,
                "validation_direct_cell_mae_mean_km_per_s": cell_mae,
                "selection_metric": "validation mean node RMSE",
            })
    selected_row = min(candidate_rows, key=lambda row: _selection_key(row, primary="node"))
    selected = {
        "feature_variant": "realistic_full",
        "model_name": str(selected_row["model_name"]),
        "pca_component_count": int(selected_row["pca_component_count"]),
        "ridge_alpha": selected_row["ridge_alpha"],
        "validation_selection_metric": "mean node RMSE over 37 validation targets; ties node MAE, ridge before linear, K, alpha_ridge",
        "test_evaluation_policy": "selected configuration evaluated once on frozen 38-target test set",
        "pca_component_grid": list(config["pca_component_grid"]),
        "ridge_alpha_grid": list(config["ridge_alpha_grid"]),
        "selected_at_grid_boundary": bool(
            int(selected_row["pca_component_count"]) in {min(component_grid), max(component_grid)}
            or float(selected_row["ridge_alpha"] or 0.0) in {float(min(config["ridge_alpha_grid"])), float(max(config["ridge_alpha_grid"]))}
        ),
        "global_optimum_claim": False,
    }
    out = output_root / "node_native"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv_local(candidate_rows, out / "validation_model_selection.csv")
    _write_json_local(selected, out / "selected_ml_configuration.json")
    _write_pca_spectrum(pca, out / "pca_spectrum.csv", basis="training node vectors only", grid=component_grid)

    models: dict[str, ObservationPCAModel] = {}
    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    split_indices = {
        "train": np.asarray([case.split == "train" for case in cases]),
        "validation": np.asarray([case.split == "validation" for case in cases]),
        "test": np.asarray([case.split == "test" for case in cases]),
    }
    for method in FEATURE_VARIANTS:
        model = _fit_variant_model(features[method][split_indices["train"]], target_train, pca, selected, bounds)
        models[method] = model
        predicted_all = model.predict(features[method])
        predictions[method] = predicted_all
        for index, case in enumerate(cases):
            if case.split == "train":
                continue
            metric_rows.append(_metric_row(case, method, predicted_all[index], _cell_prediction(case, predicted_all[index]), case.split))
        _write_prediction_npz(
            out / f"{method}_test_predictions.npz",
            test,
            predicted_all[split_indices["test"]],
            "predicted_node_velocity_km_per_s",
        )
    mean_prediction = np.mean(target_train, axis=0)
    mean_all = np.repeat(mean_prediction[None, :], len(cases), axis=0)
    predictions["training_target_mean"] = mean_all
    for index, case in enumerate(cases):
        if case.split != "train":
            metric_rows.append(_metric_row(case, "training_target_mean", mean_prediction, _cell_prediction(case, mean_prediction), case.split))
    _write_prediction_npz(out / "training_target_mean_test_predictions.npz", test, mean_all[split_indices["test"]], "predicted_node_velocity_km_per_s")
    _write_csv_local(metric_rows, out / "case_metrics.csv")

    summary_rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for method in (*FEATURE_VARIANTS.keys(), "training_target_mean"):
            group = [row for row in metric_rows if row["split"] == split and row["method_id"] == method]
            for metric in ("node_rmse_km_per_s", "node_mae_km_per_s", "direct_cell_rmse_km_per_s", "direct_cell_mae_km_per_s"):
                values = [float(row[metric]) for row in group]
                summary = _target_level_summary(values, f"summary:{split}:{method}:{metric}", int(config["bootstrap_resamples"]))
                summary.update({"split": split, "method_id": method, "method_label": PUBLIC_METHOD_LABELS.get(method, method), "metric": metric})
                summary_rows.append(summary)
    _write_csv_local(summary_rows, out / "summary.csv")
    return {
        "cases": list(cases),
        "output_dir": out,
        "pca": pca,
        "models": models,
        "predictions": predictions,
        "metric_rows": metric_rows,
        "summary_rows": summary_rows,
        "selected": selected,
        "train": train,
        "validation": validation,
        "test": test,
        "bounds": bounds,
        "features": features,
    }


def run_node_native_direct_cell_selection(
    node_native: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Select a node-native PCA-ridge model using the common direct-cell endpoint.

    The target PCA remains node-native and is fitted on the 175 training target
    vectors.  Predictions are converted to cell centers with the common
    eight-corner operator before validation scoring and configuration selection.
    The held-out test targets are evaluated only after that selection is frozen.
    """

    cases = list(node_native["cases"])
    train = list(node_native["train"])
    validation = list(node_native["validation"])
    test = list(node_native["test"])
    pca = node_native["pca"]
    bounds = tuple(float(value) for value in node_native["bounds"])
    features = node_native["features"]["realistic_full"]
    train_indices = np.asarray([case.split == "train" for case in cases])
    validation_indices = np.asarray([case.split == "validation" for case in cases])
    test_indices = np.asarray([case.split == "test" for case in cases])
    train_features = features[train_indices]
    validation_features = features[validation_indices]
    test_features = features[test_indices]
    target_train = np.vstack([case.target_vector for case in train])
    component_grid = _valid_components(pca, config["pca_component_grid"])
    candidate_rows: list[dict[str, Any]] = []
    for component_count in component_grid:
        for model_name, alphas in (("pca_linear", (None,)), ("pca_ridge", config["ridge_alpha_grid"])):
            for alpha in alphas:
                model = fit_observation_pca_model(
                    train_features,
                    target_train,
                    pca,
                    component_count,
                    model_name,
                    None if alpha is None else float(alpha),
                    bounds,
                )
                prediction = model.predict(validation_features)
                node_rmse, node_mae, cell_rmse, cell_mae = _candidate_metrics(validation, prediction)
                candidate_rows.append(
                    {
                        "feature_variant": "realistic_full",
                        "model_name": model_name,
                        "pca_component_count": component_count,
                        "ridge_alpha": None if alpha is None else float(alpha),
                        "validation_target_count": len(validation),
                        "validation_node_rmse_mean_km_per_s": node_rmse,
                        "validation_node_mae_mean_km_per_s": node_mae,
                        "validation_direct_cell_rmse_mean_km_per_s": cell_rmse,
                        "validation_direct_cell_mae_mean_km_per_s": cell_mae,
                        "selection_metric": "validation mean direct analytic cell-center all-cell RMSE",
                    }
                )
    selected_row = min(candidate_rows, key=lambda row: _selection_key(row, primary="direct"))
    selected = {
        "method_id": "node_native_direct_cell_selection",
        "feature_variant": "realistic_full",
        "model_name": str(selected_row["model_name"]),
        "pca_component_count": int(selected_row["pca_component_count"]),
        "ridge_alpha": selected_row["ridge_alpha"],
        "target_representation": "node-native PCA target reconstructed at nodes, then converted to direct cell-center values by eight-corner averaging",
        "feature_dimension": int(features.shape[1]),
        "observation_count": int(train[0].observations.shape[0]),
        "scalar_fields_per_observation": int(train[0].observations.shape[1]),
        "validation_selection_metric": "mean direct analytic cell-center all-cell RMSE over 37 validation targets; ties direct-cell MAE, ridge before linear, K, alpha_ridge",
        "test_evaluation_policy": "selected configuration evaluated once on frozen 38-target test set",
        "pca_component_grid": list(config["pca_component_grid"]),
        "ridge_alpha_grid": list(config["ridge_alpha_grid"]),
        "selected_at_grid_boundary": bool(
            int(selected_row["pca_component_count"]) in {min(component_grid), max(component_grid)}
            or (
                selected_row["ridge_alpha"] is not None
                and float(selected_row["ridge_alpha"]) in {
                    float(min(config["ridge_alpha_grid"])),
                    float(max(config["ridge_alpha_grid"])),
                }
            )
        ),
        "global_optimum_claim": False,
    }
    model = _fit_variant_model(train_features, target_train, pca, selected, bounds)
    validation_prediction = model.predict(validation_features)
    test_prediction = model.predict(test_features)
    validation_rows = [
        _metric_row(case, selected["method_id"], prediction, _cell_prediction(case, prediction), "validation")
        for case, prediction in zip(validation, validation_prediction, strict=True)
    ]
    test_rows = [
        _metric_row(case, selected["method_id"], prediction, _cell_prediction(case, prediction), "test")
        for case, prediction in zip(test, test_prediction, strict=True)
    ]
    out = output_root / "node_native_direct_cell_selection"
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    _write_csv_local(candidate_rows, out / "validation_model_selection.csv")
    _write_json_local(selected, out / "selected_configuration.json")
    _write_csv_local(test_rows, out / "test_metrics_by_target.csv")
    _write_csv_local(validation_rows + test_rows, out / "case_metrics.csv")
    test_cells = np.vstack([_cell_prediction(case, prediction) for case, prediction in zip(test, test_prediction, strict=True)])
    truth_cells = _truth_cells(test)
    np.savez_compressed(
        out / "predictions" / "test_predictions.npz",
        target_ids=np.asarray([case.target_id for case in test]),
        target_hashes=np.asarray([case.target_hash for case in test]),
        predicted_node_velocity_km_per_s=test_prediction,
        predicted_cell_velocity_km_per_s=test_cells,
        direct_analytic_cell_truth_km_per_s=truth_cells,
    )
    for case, node_prediction, cell_prediction, truth in zip(test, test_prediction, test_cells, truth_cells, strict=True):
        np.savez_compressed(
            out / "predictions" / f"{case.target_id}.npz",
            target_id=case.target_id,
            target_hash=case.target_hash,
            predicted_node_velocity_km_per_s=node_prediction,
            predicted_cell_velocity_km_per_s=cell_prediction,
            direct_analytic_cell_truth_km_per_s=truth,
        )
    selected_test_summary = {
        "rmse": _target_level_summary(
            [float(row["direct_cell_rmse_km_per_s"]) for row in test_rows],
            "node-native-direct-cell-test-rmse",
            int(config["bootstrap_resamples"]),
        ),
        "mae": _target_level_summary(
            [float(row["direct_cell_mae_km_per_s"]) for row in test_rows],
            "node-native-direct-cell-test-mae",
            int(config["bootstrap_resamples"]),
        ),
    }
    node_primary_rows = {
        str(row["target_id"]): row
        for row in node_native["metric_rows"]
        if row["split"] == "test" and row["method_id"] == "realistic_full"
    }
    direct_vs_primary = [
        {
            "comparison_id": "node_native_direct_cell_selection_vs_node_native_primary",
            "target_id": row["target_id"],
            "family": row["family"],
            "paired_delta": float(row["direct_cell_rmse_km_per_s"])
            - float(node_primary_rows[str(row["target_id"])] ["direct_cell_rmse_km_per_s"]),
            "delta_definition": "node-native direct-cell-selected RMSE minus node-native node-selected primary RMSE; negative favors direct-cell selection",
        }
        for row in test_rows
    ]
    _write_json_local(
        {
            "selected": selected,
            "selected_validation_row": selected_row,
            "test_summary": selected_test_summary,
            "paired_summary_vs_node_native_primary": _paired_summary(
                direct_vs_primary,
                bootstrap_resamples=int(config["bootstrap_resamples"]),
                tie_tolerance=float(config.get("tie_tolerance", 1.0e-12)),
            ),
            "test_selection": False,
        },
        out / "selection_summary.json",
    )
    _write_json_local(
        {
            "selected": selected,
            "rmse": selected_test_summary["rmse"],
            "mae": selected_test_summary["mae"],
            "test_selection": False,
        },
        out / "test_summary.json",
    )
    _write_csv_local(direct_vs_primary, out / "paired_deltas_vs_node_native_primary.csv")
    return {
        "cases": cases,
        "train": train,
        "validation": validation,
        "test": test,
        "pca": pca,
        "model": model,
        "selected": selected,
        "selected_validation_row": selected_row,
        "validation_prediction": validation_prediction,
        "test_prediction": test_prediction,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "output_dir": out,
    }


def _run_independent_pca_sensitivity(
    *,
    method_id: str,
    feature_matrix: np.ndarray,
    train: Sequence[ProductionCase],
    validation: Sequence[ProductionCase],
    test: Sequence[ProductionCase],
    pca_targets: np.ndarray,
    config: Mapping[str, Any],
    bounds: tuple[float, float],
    output_dir: Path,
    selection_domain: str,
) -> dict[str, Any]:
    pca = fit_target_pca(pca_targets)
    train_features = feature_matrix[: len(train)]
    validation_features = feature_matrix[len(train) : len(train) + len(validation)]
    test_features = feature_matrix[len(train) + len(validation) :]
    candidates: list[dict[str, Any]] = []
    for k in _valid_components(pca, config["pca_component_grid"]):
        linear = fit_observation_pca_model(train_features, pca_targets, pca, k, "pca_linear", None, bounds)
        prediction = linear.predict(validation_features)
        node_rmse, node_mae, cell_rmse, cell_mae = _candidate_metrics(validation, prediction)
        candidates.append({"method_id": method_id, "model_name": "pca_linear", "pca_component_count": k, "ridge_alpha": None, "validation_node_rmse_mean_km_per_s": node_rmse, "validation_node_mae_mean_km_per_s": node_mae, "validation_direct_cell_rmse_mean_km_per_s": cell_rmse, "validation_direct_cell_mae_mean_km_per_s": cell_mae, "selection_metric": selection_domain})
        for alpha in config["ridge_alpha_grid"]:
            model = fit_observation_pca_model(train_features, pca_targets, pca, k, "pca_ridge", float(alpha), bounds)
            prediction = model.predict(validation_features)
            node_rmse, node_mae, cell_rmse, cell_mae = _candidate_metrics(validation, prediction)
            candidates.append({"method_id": method_id, "model_name": "pca_ridge", "pca_component_count": k, "ridge_alpha": float(alpha), "validation_node_rmse_mean_km_per_s": node_rmse, "validation_node_mae_mean_km_per_s": node_mae, "validation_direct_cell_rmse_mean_km_per_s": cell_rmse, "validation_direct_cell_mae_mean_km_per_s": cell_mae, "selection_metric": selection_domain})
    selected_row = min(candidates, key=lambda row: _selection_key(row, primary="direct"))
    selected = {
        "method_id": method_id,
        "model_name": selected_row["model_name"],
        "pca_component_count": int(selected_row["pca_component_count"]),
        "ridge_alpha": selected_row["ridge_alpha"],
        "validation_selection_metric": selection_domain,
        "test_evaluation_policy": "selected configuration evaluated once on frozen 38-target test set",
        "pca_component_grid": list(config["pca_component_grid"]),
        "ridge_alpha_grid": list(config["ridge_alpha_grid"]),
        "selected_at_grid_boundary": bool(
            int(selected_row["pca_component_count"]) in {min(config["pca_component_grid"]), max(config["pca_component_grid"])}
            or float(selected_row["ridge_alpha"] or 0.0) in {float(min(config["ridge_alpha_grid"])), float(max(config["ridge_alpha_grid"]))}
        ),
        "global_optimum_claim": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_local(candidates, output_dir / "validation_model_selection.csv")
    _write_json_local(selected, output_dir / "selected_configuration.json")
    model = _fit_variant_model(train_features, pca_targets, pca, selected, bounds)
    validation_prediction = model.predict(validation_features)
    test_prediction = model.predict(test_features)
    rows: list[dict[str, Any]] = []
    for split, split_cases, split_prediction in (("validation", validation, validation_prediction), ("test", test, test_prediction)):
        for case, prediction in zip(split_cases, split_prediction, strict=True):
            rows.append(_metric_row(case, method_id, prediction, _cell_prediction(case, prediction), split))
    _write_csv_local(rows, output_dir / "case_metrics.csv")
    _write_prediction_npz(output_dir / "test_predictions.npz", test, test_prediction, "predicted_node_velocity_km_per_s")
    _write_pca_spectrum(pca, output_dir / "pca_spectrum.csv", basis=f"training targets for {method_id}", grid=_valid_components(pca, config["pca_component_grid"]))
    return {"pca": pca, "model": model, "selected": selected, "rows": rows, "test_prediction": test_prediction, "validation_prediction": validation_prediction}


def run_cell_native_sensitivity(
    train: Sequence[ProductionCase],
    validation: Sequence[ProductionCase],
    test: Sequence[ProductionCase],
    node_native_test_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    bounds: tuple[float, float],
    output_root: Path,
) -> dict[str, Any]:
    """Fit PCA on direct cell-center targets using training cases only."""

    targets_train = _truth_cells(train)
    targets_validation = _truth_cells(validation)
    targets_test = _truth_cells(test)
    pca = fit_target_pca(targets_train)
    full_features = build_feature_matrices([*train, *validation, *test])["realistic_full"]
    train_features = full_features[: len(train)]
    validation_features = full_features[len(train) : len(train) + len(validation)]
    test_features = full_features[-len(test) :]
    candidate_rows: list[dict[str, Any]] = []
    for k in _valid_components(pca, config["pca_component_grid"]):
        linear = fit_observation_pca_model(train_features, targets_train, pca, k, "pca_linear", None, bounds)
        prediction = linear.predict(validation_features)
        candidate_rows.append({"feature_variant": "realistic_full", "feature_dimension": int(full_features.shape[1]), "model_name": "pca_linear", "pca_component_count": k, "ridge_alpha": None, "validation_target_count": len(validation), "validation_direct_cell_rmse_mean_km_per_s": float(np.mean([_rmse(t, p) for t, p in zip(targets_validation, prediction, strict=True)])), "validation_direct_cell_mae_mean_km_per_s": float(np.mean([_mae(t, p) for t, p in zip(targets_validation, prediction, strict=True)])), "selection_metric": "validation mean direct analytic cell-center all-cell RMSE"})
        for alpha in config["ridge_alpha_grid"]:
            model = fit_observation_pca_model(train_features, targets_train, pca, k, "pca_ridge", float(alpha), bounds)
            prediction = model.predict(validation_features)
            candidate_rows.append({"feature_variant": "realistic_full", "feature_dimension": int(full_features.shape[1]), "model_name": "pca_ridge", "pca_component_count": k, "ridge_alpha": float(alpha), "validation_target_count": len(validation), "validation_direct_cell_rmse_mean_km_per_s": float(np.mean([_rmse(t, p) for t, p in zip(targets_validation, prediction, strict=True)])), "validation_direct_cell_mae_mean_km_per_s": float(np.mean([_mae(t, p) for t, p in zip(targets_validation, prediction, strict=True)])), "selection_metric": "validation mean direct analytic cell-center all-cell RMSE"})
    selected_row = min(candidate_rows, key=lambda row: (float(row["validation_direct_cell_rmse_mean_km_per_s"]), float(row["validation_direct_cell_mae_mean_km_per_s"]), 0 if row["model_name"] == "pca_ridge" else 1, int(row["pca_component_count"]), float(row["ridge_alpha"] or 0.0)))
    selected = {
        "method_id": "cell_native",
        "feature_variant": "realistic_full",
        "model_name": selected_row["model_name"],
        "pca_component_count": int(selected_row["pca_component_count"]),
        "ridge_alpha": selected_row["ridge_alpha"],
        "target_representation": "direct analytic velocity evaluated at target-cell centers",
        "pca_basis_source": "training direct cell-center target vectors only",
        "feature_dimension": int(full_features.shape[1]),
        "observation_count": int(train[0].observations.shape[0]),
        "scalar_fields_per_observation": int(train[0].observations.shape[1]),
        "feature_representation_contract": "full-input representation: 384 observations x 8 scalar fields = 3072 scalars per target",
        "validation_selection_metric": "validation mean direct analytic cell-center all-cell RMSE",
        "test_evaluation_policy": "selected configuration evaluated once on frozen 38-target test set",
        "pca_component_grid": list(config["pca_component_grid"]),
        "ridge_alpha_grid": list(config["ridge_alpha_grid"]),
        "selected_at_grid_boundary": bool(int(selected_row["pca_component_count"]) in {min(config["pca_component_grid"]), max(config["pca_component_grid"])} or float(selected_row["ridge_alpha"] or 0.0) in {float(min(config["ridge_alpha_grid"])), float(max(config["ridge_alpha_grid"]))}),
        "global_optimum_claim": False,
    }
    out = output_root / "cell_native"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv_local(candidate_rows, out / "validation_model_selection.csv")
    _write_json_local(selected, out / "selected_configuration.json")
    model = _fit_variant_model(train_features, targets_train, pca, selected, bounds)
    validation_prediction = model.predict(validation_features)
    test_prediction = model.predict(test_features)
    rows: list[dict[str, Any]] = []
    for split, split_cases, split_targets, split_prediction in (("validation", validation, targets_validation, validation_prediction), ("test", test, targets_test, test_prediction)):
        for case, truth, prediction in zip(split_cases, split_targets, split_prediction, strict=True):
            metrics = _prediction_metrics_from_arrays(truth, prediction)
            rows.append({"target_id": case.target_id, "target_hash": case.target_hash, "family": case.family, "split": split, "method_id": "cell_native", "method_label": PUBLIC_METHOD_LABELS["cell_native"], "direct_cell_rmse_km_per_s": metrics["rmse"], "direct_cell_mae_km_per_s": metrics["mae"], "direct_cell_bias_km_per_s": metrics["bias"]})
    _write_csv_local(rows, out / "case_metrics.csv")
    _write_prediction_npz(out / "test_predictions.npz", test, test_prediction, "predicted_cell_velocity_km_per_s")
    oracle_rows: list[dict[str, Any]] = []
    for k in _valid_components(pca, config["pca_component_grid"]):
        validation_oracle = reconstruct_from_pca(pca, targets_validation, k)
        test_oracle = reconstruct_from_pca(pca, targets_test, k)
        oracle_rows.append({"pca_component_count": k, "validation_oracle_direct_cell_rmse_km_per_s": float(np.mean([_rmse(t, p) for t, p in zip(targets_validation, validation_oracle, strict=True)])), "test_oracle_direct_cell_rmse_km_per_s": float(np.mean([_rmse(t, p) for t, p in zip(targets_test, test_oracle, strict=True)])), "validation_oracle_direct_cell_mae_km_per_s": float(np.mean([_mae(t, p) for t, p in zip(targets_validation, validation_oracle, strict=True)])), "test_oracle_direct_cell_mae_km_per_s": float(np.mean([_mae(t, p) for t, p in zip(targets_test, test_oracle, strict=True)])), "oracle_status": "training-only PCA projection diagnostic; not an achievable prediction"})
    _write_csv_local(oracle_rows, out / "training_only_cell_pca_oracle_curve.csv")
    node_by_id = {str(row["target_id"]): row for row in node_native_test_rows}
    cell_test_rows = {
        str(row["target_id"]): row for row in rows if row["split"] == "test"
    }
    paired_rows = _make_paired_rows(
        "cell_native_vs_node_native", cell_test_rows, node_by_id
    )
    paired_summary = _paired_summary(
        paired_rows,
        bootstrap_resamples=int(config["bootstrap_resamples"]),
        tie_tolerance=float(config.get("tie_tolerance", 1.0e-12)),
    )
    paired_summary["mae"] = _paired_metric_summary(
        paired_rows,
        delta_field="paired_delta_mae",
        metric_label="MAE",
        bootstrap_resamples=int(config["bootstrap_resamples"]),
        tie_tolerance=float(config.get("tie_tolerance", 1.0e-12)),
    )
    _write_csv_local(paired_rows, out / "paired_deltas_vs_node_native.csv")
    _write_json_local({"selected": selected, "test_summary": _target_level_summary([float(row["direct_cell_rmse_km_per_s"]) for row in rows if row["split"] == "test"], "cell-native-test", int(config["bootstrap_resamples"])), "node_native_test_summary": _target_level_summary([float(row["direct_cell_rmse_km_per_s"]) for row in node_native_test_rows], "node-native-test", int(config["bootstrap_resamples"])), "paired_summary": paired_summary, "test_selection": False}, out / "summary.json")
    _write_csv_local([row for row in rows if row["split"] == "test"], out / "test_metrics_by_target.csv")
    return {"pca": pca, "model": model, "selected": selected, "rows": rows, "test_prediction": test_prediction, "validation_prediction": validation_prediction, "oracle_rows": oracle_rows, "features": full_features, "paired_vs_node_native": paired_rows, "output_dir": out}


def run_reference_ray_analysis(
    cases: Sequence[ProductionCase],
    output_root: Path,
    config: Mapping[str, Any],
    settings: Any,
    operator_dir: Path,
) -> dict[str, Any]:
    """Run the expanded validation-controlled fixed reference-ray grid."""

    validation = [case for case in cases if case.split == "validation"]
    test = [case for case in cases if case.split == "test"]
    bounds = tuple(float(value) for value in settings.velocity_model_generation.velocity_bounds_km_per_s)
    with np.load(operator_dir / "operator_consistency_arrays.npz") as artifact:
        fsm_by_target = {str(target_id): np.asarray(times, dtype=float) for target_id, times in zip(artifact["target_ids"].tolist(), artifact["fsm_times_s"], strict=True)}
    geometry_dir = operator_dir / "reference_ray_geometry"
    cache: dict[str, tuple[RayGeometry, np.ndarray, np.ndarray]] = {}
    for case in [*validation, *test]:
        geometry = _load_geometry(geometry_dir / f"{case.target_id}.npz")
        reference_grid = build_layered_reference_velocity_grid(case.target_grid, settings)
        reference_cells = cell_centered_values_from_node_vector(np.asarray(reference_grid.p_velocity_km_per_s, dtype=float), _node_shape(case.target_grid))
        cache[case.target_id] = (geometry, reference_cells, direct_analytic_cell_truth(case))
    candidates: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for damping in config["reference_damping_grid"]:
        for smoothing in config["reference_smoothing_grid"]:
            solved_rows: list[dict[str, Any]] = []
            for case in validation:
                geometry, reference_cells, target_cells = cache[case.target_id]
                solved = solve_fixed_ray_case(case=case, geometry=geometry, reference_cells=reference_cells, target_cells=target_cells, damping=float(damping), smoothing=float(smoothing), cell_shape=tuple(value - 1 for value in _node_shape(case.target_grid)), velocity_bounds=bounds, reference_travel_times_s=fsm_by_target[case.target_id])
                solved_rows.append(solved)
                validation_rows.append({"target_id": case.target_id, "family": case.family, "damping": float(damping), "smoothing": float(smoothing), "alpha_damping": float(damping), "alpha_smooth": float(smoothing), "all_cell_rmse_km_per_s": solved["all_cell_rmse"], "all_cell_mae_km_per_s": solved["all_cell_mae"], "positive_coverage_rmse_km_per_s": solved["positive_coverage_rmse"], "travel_time_rmse_s": solved["travel_time_rmse"], "clipped_fraction": solved["clipped_fraction"], "pcg_iterations": solved["pcg_iterations"], "pcg_converged": solved["pcg_converged"]})
            candidates.append({"damping": float(damping), "smoothing": float(smoothing), "alpha_damping": float(damping), "alpha_smooth": float(smoothing), "validation_all_cell_rmse_mean_km_per_s": float(np.mean([row["all_cell_rmse"] for row in solved_rows])), "validation_all_cell_mae_mean_km_per_s": float(np.mean([row["all_cell_mae"] for row in solved_rows])), "validation_positive_coverage_rmse_mean_km_per_s": float(np.mean([row["positive_coverage_rmse"] for row in solved_rows])), "validation_travel_time_rmse_mean_s": float(np.mean([row["travel_time_rmse"] for row in solved_rows])), "all_pcg_converged": bool(all(row["pcg_converged"] for row in solved_rows)), "selection_metric": "validation mean direct analytic cell-center all-cell RMSE"})
    selected_row = min(candidates, key=lambda row: (float(row["validation_all_cell_rmse_mean_km_per_s"]), float(row["validation_positive_coverage_rmse_mean_km_per_s"]), float(row["validation_travel_time_rmse_mean_s"]), float(row["damping"]), float(row["smoothing"])))
    selected = {"method_id": "reference_ray", "selected_damping": selected_row["damping"], "selected_smoothing": selected_row["smoothing"], "alpha_damping": selected_row["alpha_damping"], "alpha_smooth": selected_row["alpha_smooth"], "damping_grid": list(config["reference_damping_grid"]), "smoothing_grid": list(config["reference_smoothing_grid"]), "selection_metric": "validation mean direct analytic cell-center all-cell RMSE", "selection_unit": "37 validation targets", "test_evaluation_policy": "selected configuration evaluated once on frozen 38-target test set", "residual_formula": "t_obs - t_FSM(s_0)", "sensitivity_operator": "G_ref retained from fixed rays traced through the layered reference model", "method_scope": "independent fixed reference-ray regularized path-operator baseline; not exact FSM Jacobian or validated FSM linearization", "damping_selected_at_grid_boundary": bool(float(selected_row["damping"]) in {float(min(config["reference_damping_grid"])), float(max(config["reference_damping_grid"]))}), "smoothing_selected_at_grid_boundary": bool(float(selected_row["smoothing"]) in {float(min(config["reference_smoothing_grid"])), float(max(config["reference_smoothing_grid"]))}), "global_optimum_claim": False}
    out = output_root / "reference_ray"
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    _write_csv_local(validation_rows, out / "validation_grid.csv")
    _write_csv_local(candidates, out / "validation_candidate_summary.csv")
    _write_json_local(selected, out / "selected_configuration.json")
    test_rows: list[dict[str, Any]] = []
    test_predictions: dict[str, np.ndarray] = {}
    for case in test:
        geometry, reference_cells, target_cells = cache[case.target_id]
        solved = solve_fixed_ray_case(case=case, geometry=geometry, reference_cells=reference_cells, target_cells=target_cells, damping=float(selected["selected_damping"]), smoothing=float(selected["selected_smoothing"]), cell_shape=tuple(value - 1 for value in _node_shape(case.target_grid)), velocity_bounds=bounds, reference_travel_times_s=fsm_by_target[case.target_id])
        prediction = np.asarray(solved["fixed_cells"], dtype=float)
        test_predictions[case.target_id] = prediction
        np.savez_compressed(out / "predictions" / f"{case.target_id}.npz", target_cells=target_cells, reference_cells=reference_cells, fixed_cells=prediction, coverage_km=geometry.coverage_km, fixed_predicted_times=solved["predicted_times"], fixed_residual_s=solved["residuals"])
        test_rows.append({"target_id": case.target_id, "target_hash": case.target_hash, "family": case.family, "split": "test", "method_id": "reference_ray", "method_label": PUBLIC_METHOD_LABELS["reference_ray"], "direct_cell_rmse_km_per_s": solved["all_cell_rmse"], "direct_cell_mae_km_per_s": solved["all_cell_mae"], "direct_cell_bias_km_per_s": float(np.mean(prediction - target_cells)), "positive_coverage_rmse_km_per_s": solved["positive_coverage_rmse"], "travel_time_rmse_s": solved["travel_time_rmse"], "pcg_iterations": solved["pcg_iterations"], "pcg_converged": solved["pcg_converged"]})
    _write_csv_local(test_rows, out / "test_metrics.csv")
    _write_json_local({"test_summary": _target_level_summary([float(row["direct_cell_rmse_km_per_s"]) for row in test_rows], "reference-ray-test", int(config["bootstrap_resamples"])), "coverage_source": "accumulated reference-model fixed-ray path length", "coverage_thresholds_km": list(config["coverage_thresholds_km"])}, out / "summary.json")
    return {"selected": selected, "test_rows": test_rows, "test_predictions": test_predictions, "cache": cache, "fsm_by_target": fsm_by_target, "output_dir": out}


def _make_paired_rows(
    comparison_id: str,
    first_rows: Mapping[str, Mapping[str, Any]],
    second_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build target-matched direct-cell rows with the requested first-minus-second sign."""

    rows: list[dict[str, Any]] = []
    for target_id in sorted(first_rows):
        if target_id not in second_rows:
            continue
        first = first_rows[target_id]
        second = second_rows[target_id]
        first_rmse = float(first["direct_cell_rmse_km_per_s"])
        second_rmse = float(second["direct_cell_rmse_km_per_s"])
        first_mae = float(first["direct_cell_mae_km_per_s"])
        second_mae = float(second["direct_cell_mae_km_per_s"])
        rows.append(
            {
                "comparison_id": comparison_id,
                "target_id": target_id,
                "family": first["family"],
                "first_direct_cell_rmse_km_per_s": first_rmse,
                "second_direct_cell_rmse_km_per_s": second_rmse,
                "paired_delta": first_rmse - second_rmse,
                "delta_definition": "first method direct-cell RMSE minus second method direct-cell RMSE; negative favors the first method",
                "first_direct_cell_mae_km_per_s": first_mae,
                "second_direct_cell_mae_km_per_s": second_mae,
                "paired_delta_mae": first_mae - second_mae,
                "mae_delta_definition": "first method direct-cell MAE minus second method direct-cell MAE; negative favors the first method",
            }
        )
    if len(rows) != len(first_rows) or len(rows) != len(second_rows):
        raise ValueError(f"{comparison_id} does not have a complete target-level pairing.")
    return rows


def finalize_direct_cell_comparisons(
    node_native: Mapping[str, Any],
    node_native_direct: Mapping[str, Any],
    cell_native: Mapping[str, Any],
    reference_ray: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Write the cell-native and direct-cell-selected-node paired comparisons."""

    bootstrap_resamples = int(config["bootstrap_resamples"])
    tie_tolerance = float(config.get("tie_tolerance", 1.0e-12))
    cell_rows = {
        str(row["target_id"]): row
        for row in cell_native["rows"]
        if row["split"] == "test"
    }
    reference_rows = {
        str(row["target_id"]): row for row in reference_ray["test_rows"]
    }
    node_primary_rows = {
        str(row["target_id"]): row
        for row in node_native["metric_rows"]
        if row["split"] == "test" and row["method_id"] == "realistic_full"
    }
    direct_selected_rows = {
        str(row["target_id"]): row for row in node_native_direct["test_rows"]
    }
    pair_specs = (
        (
            "cell_native_vs_reference_ray",
            cell_rows,
            reference_rows,
            cell_native["output_dir"] / "paired_deltas_vs_reference_ray.csv",
            cell_native["output_dir"] / "paired_summary_vs_reference_ray.json",
        ),
        (
            "cell_native_vs_node_native",
            cell_rows,
            node_primary_rows,
            cell_native["output_dir"] / "paired_deltas_vs_node_native.csv",
            cell_native["output_dir"] / "paired_summary_vs_node_native.json",
        ),
        (
            "node_native_direct_cell_selection_vs_reference_ray",
            direct_selected_rows,
            reference_rows,
            node_native_direct["output_dir"] / "paired_deltas_vs_reference_ray.csv",
            node_native_direct["output_dir"] / "paired_summary_vs_reference_ray.json",
        ),
    )
    summaries: dict[str, dict[str, Any]] = {}
    for comparison_id, first_rows, second_rows, csv_path, json_path in pair_specs:
        rows = _make_paired_rows(comparison_id, first_rows, second_rows)
        summary = _paired_summary(
            rows,
            bootstrap_resamples=bootstrap_resamples,
            tie_tolerance=tie_tolerance,
        )
        summary["mae"] = _paired_metric_summary(
            rows,
            delta_field="paired_delta_mae",
            metric_label="MAE",
            bootstrap_resamples=bootstrap_resamples,
            tie_tolerance=tie_tolerance,
        )
        _write_csv_local(rows, csv_path)
        _write_json_local(summary, json_path)
        summaries[comparison_id] = summary

    cell_summary = _read_json(cell_native["output_dir"] / "summary.json")
    cell_summary.update(
        {
            "paired_summary_vs_reference_ray": summaries["cell_native_vs_reference_ray"],
            "paired_summary_vs_node_native": summaries["cell_native_vs_node_native"],
            "test_selection": False,
        }
    )
    _write_json_local(cell_summary, cell_native["output_dir"] / "summary.json")
    direct_summary = _read_json(node_native_direct["output_dir"] / "selection_summary.json")
    direct_summary["paired_summary_vs_reference_ray"] = summaries[
        "node_native_direct_cell_selection_vs_reference_ray"
    ]
    direct_summary["test_selection"] = False
    _write_json_local(direct_summary, node_native_direct["output_dir"] / "selection_summary.json")


def run_multiseed_shuffle(
    cases: Sequence[ProductionCase],
    node_native: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Repeat whole-vector within-split time reassignment over 100 seeds.

    The inferential unit is the permutation seed: each seed contributes one
    mean paired direct-cell effect over the 38 test targets.  The 3,800
    target-by-seed rows are retained for audit and visualization but are not
    treated as 3,800 independent observations.
    """

    train = list(node_native["train"])
    test = list(node_native["test"])
    features = node_native["features"]["realistic_full"]
    target_train = np.vstack([case.target_vector for case in train])
    pca = node_native["pca"]
    selected = node_native["selected"]
    bounds = node_native["bounds"]
    full_predictions = node_native["predictions"]["realistic_full"]
    test_indices = np.asarray([case.split == "test" for case in cases])
    full_test = full_predictions[test_indices]
    rows: list[dict[str, Any]] = []
    seeds = [int(config["shuffled_time_seed_start"]) + index for index in range(int(config["shuffled_time_seed_count"]))]
    for seed in seeds:
        shuffled = features.copy()
        for split in ("train", "validation", "test"):
            indices = np.asarray([index for index, case in enumerate(cases) if case.split == split], dtype=int)
            permutation = np.random.default_rng(seed + _stable_seed(f"shuffle-split:{split}")).permutation(indices.size)
            source_indices = indices[permutation]
            shuffled[indices, 7 * 384 : 8 * 384] = features[source_indices, 7 * 384 : 8 * 384]
        model = _fit_variant_model(shuffled[[case.split == "train" for case in cases]], target_train, pca, selected, bounds)
        shuffled_prediction = model.predict(shuffled[test_indices])
        for case, full, shuffled_nodes in zip(test, full_test, shuffled_prediction, strict=True):
            truth = direct_analytic_cell_truth(case)
            full_rmse = _rmse(truth, _cell_prediction(case, full))
            shuffled_rmse = _rmse(truth, _cell_prediction(case, shuffled_nodes))
            rows.append({"shuffle_seed": seed, "target_id": case.target_id, "family": case.family, "full_input_direct_cell_rmse_km_per_s": full_rmse, "shuffled_direct_cell_rmse_km_per_s": shuffled_rmse, "paired_delta_km_per_s": full_rmse - shuffled_rmse, "paired_delta_definition": "full-input direct-cell RMSE minus shuffled-time direct-cell RMSE; negative favors full-input"})
    out = output_root / "shuffled_time"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv_local(rows, out / "multiseed_direct_cell_target_rows.csv")
    seed_summaries: list[dict[str, Any]] = []
    for seed in seeds:
        group = [row for row in rows if int(row["shuffle_seed"]) == seed]
        deltas = np.asarray([float(row["paired_delta_km_per_s"]) for row in group])
        seed_summaries.append({"shuffle_seed": seed, "target_count": len(group), "mean_paired_delta_km_per_s": float(np.mean(deltas)), "median_paired_delta_km_per_s": float(np.median(deltas)), "sd_paired_delta_km_per_s": float(np.std(deltas)), "p2_5_target_delta_km_per_s": float(np.percentile(deltas, 2.5)), "p97_5_target_delta_km_per_s": float(np.percentile(deltas, 97.5)), "summary_unit": "one permutation seed represented by the mean over 38 target-level direct-cell paired deltas", "target_bootstrap_applied": False})
    _write_csv_local(seed_summaries, out / "multiseed_direct_cell_seed_summary.csv")
    seed_effects = np.asarray([float(row["mean_paired_delta_km_per_s"]) for row in seed_summaries])
    overall = {"shuffle_seed_count": len(seeds), "shuffle_seeds": seeds, "permutation_scope": "whole 384-value travel-time vectors reassigned across cases within each split; coordinates and distances remain attached to the receiving case", "model_refit_policy": "refit training observation-to-PCA map for every seed using frozen Full-input PCA-ridge K/alpha_ridge", "test_selection": False, "target_count_per_seed": len(test), "target_row_count": len(rows), "inferential_unit": "100 permutation-seed mean effects", "pooled_target_rows_inferentially_used": False, "mean_seed_effect_km_per_s": float(np.mean(seed_effects)), "median_seed_effect_km_per_s": float(np.median(seed_effects)), "sd_seed_effect_km_per_s": float(np.std(seed_effects)), "p2_5_seed_effect_km_per_s": float(np.percentile(seed_effects, 2.5)), "p97_5_seed_effect_km_per_s": float(np.percentile(seed_effects, 97.5)), "proportion_seed_effects_below_zero": float(np.mean(seed_effects < 0.0)), "interpretation": "permutation sensitivity of the direct-cell case-level association diagnostic; the 100-seed distribution is not a target-bootstrap confidence interval", "row_level_note": "The 3,800 target-by-seed rows are dependent within seed and are retained for audit only."}
    _write_json_local(overall, out / "multiseed_direct_cell_overall_summary.json")
    _write_json_local(overall, out / "metadata.json")
    return {"rows": rows, "seed_summaries": seed_summaries, "seeds": seeds, "overall": overall}


def run_noise_analysis_revised(
    cases: Sequence[ProductionCase],
    node_native: Mapping[str, Any],
    travel_sensitivity: Mapping[str, Any],
    reference_ray: Mapping[str, Any],
    config: Mapping[str, Any],
    settings: Any,
    operator_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Regenerate matched timing-noise rows after the revised selections."""

    test = list(node_native["test"])
    bounds = node_native["bounds"]
    geometry_dir = operator_dir / "reference_ray_geometry"
    rows: list[dict[str, Any]] = []
    zero: dict[tuple[str, str], float] = {}
    levels = [float(value) for value in config["noise_levels_s"]]
    repetitions = int(config["noise_nonzero_repetitions"])
    with np.load(operator_dir / "operator_consistency_arrays.npz") as artifact:
        fsm_by_target = {str(target_id): np.asarray(times, dtype=float) for target_id, times in zip(artifact["target_ids"].tolist(), artifact["fsm_times_s"], strict=True)}
    for level in levels:
        seeds = [0] if level == 0.0 else list(range(1, repetitions + 1))
        for noise_seed in seeds:
            for case in test:
                rng = np.random.default_rng(_stable_seed(f"{FROZEN_NOISE_SEED_LABEL}:{int(round(level * 1000))}:{noise_seed}:{case.target_id}"))
                observations = case.observations.copy()
                if level > 0.0:
                    observations[:, 7] += rng.normal(0.0, level, size=observations.shape[0])
                noisy_case = replace(case, observations=observations)
                full_features = build_feature_matrices([noisy_case])["realistic_full"]
                travel_features = build_feature_matrices([noisy_case])["travel_time_only"]
                full_nodes = node_native["models"]["realistic_full"].predict(full_features)[0]
                travel_nodes = travel_sensitivity["model"].predict(travel_features)[0]
                geometry = _load_geometry(geometry_dir / f"{case.target_id}.npz")
                reference_grid = build_layered_reference_velocity_grid(case.target_grid, settings)
                reference_cells = cell_centered_values_from_node_vector(np.asarray(reference_grid.p_velocity_km_per_s, dtype=float), _node_shape(case.target_grid))
                target = direct_analytic_cell_truth(case)
                fixed = solve_fixed_ray_case(case=noisy_case, geometry=geometry, reference_cells=reference_cells, target_cells=target, damping=float(reference_ray["selected"]["selected_damping"]), smoothing=float(reference_ray["selected"]["selected_smoothing"]), cell_shape=tuple(value - 1 for value in _node_shape(case.target_grid)), velocity_bounds=bounds, reference_travel_times_s=fsm_by_target[case.target_id])
                predictions = {"realistic_full": _cell_prediction(case, full_nodes), "travel_time_only": _cell_prediction(case, travel_nodes), "reference_ray": fixed["fixed_cells"]}
                for method, prediction in predictions.items():
                    metrics = _prediction_metrics_from_arrays(target, prediction)
                    if level == 0.0:
                        zero[(case.target_id, method)] = float(metrics["rmse"])
                    rows.append({"target_id": case.target_id, "family": case.family, "method_id": method, "noise_std_s": level, "noise_seed": noise_seed, "direct_cell_rmse_km_per_s": metrics["rmse"], "direct_cell_mae_km_per_s": metrics["mae"], "direct_cell_bias_km_per_s": metrics["bias"], "degradation_from_zero_rmse_km_per_s": None})
    for row in rows:
        row["degradation_from_zero_rmse_km_per_s"] = float(row["direct_cell_rmse_km_per_s"]) - zero[(str(row["target_id"]), str(row["method_id"]))]
    out = output_root / "noise"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv_local(rows, out / "replication_metrics.csv")
    target_level_rows: list[dict[str, Any]] = []
    for target_id in sorted({str(row["target_id"]) for row in rows}):
        family = next(str(row["family"]) for row in rows if str(row["target_id"]) == target_id)
        for method in ("realistic_full", "travel_time_only", "reference_ray"):
            for level in levels:
                group = [row for row in rows if str(row["target_id"]) == target_id and row["method_id"] == method and float(row["noise_std_s"]) == level]
                target_level_rows.append({"target_id": target_id, "family": family, "method_id": method, "noise_std_s": level, "replication_count": len(group), "direct_cell_rmse_km_per_s": float(np.mean([float(row["direct_cell_rmse_km_per_s"]) for row in group])), "direct_cell_mae_km_per_s": float(np.mean([float(row["direct_cell_mae_km_per_s"]) for row in group])), "degradation_from_zero_rmse_km_per_s": float(np.mean([float(row["degradation_from_zero_rmse_km_per_s"]) for row in group]))})
    _write_csv_local(target_level_rows, out / "target_level_summary.csv")
    summary_rows: list[dict[str, Any]] = []
    for method in ("realistic_full", "travel_time_only", "reference_ray"):
        for level in levels:
            group = [row for row in target_level_rows if row["method_id"] == method and float(row["noise_std_s"]) == level]
            rmse = [float(row["direct_cell_rmse_km_per_s"]) for row in group]
            degradation = [float(row["degradation_from_zero_rmse_km_per_s"]) for row in group]
            rmse_summary = _target_level_summary(rmse, f"noise-rmse:{method}:{level}", int(config["bootstrap_resamples"]))
            deg_summary = _target_level_summary(degradation, f"noise-degradation:{method}:{level}", int(config["bootstrap_resamples"]))
            summary_rows.append({"method_id": method, "method_label": PUBLIC_METHOD_LABELS[method], "noise_std_s": level, "target_count": len(group), "replication_count_per_target": len([row for row in rows if row["method_id"] == method and float(row["noise_std_s"]) == level]) // len(group), "rmse_mean_km_per_s": rmse_summary["mean"], "rmse_sd_km_per_s": rmse_summary["sd_population"], "rmse_ci95_lower_km_per_s": rmse_summary["ci95_lower"], "rmse_ci95_upper_km_per_s": rmse_summary["ci95_upper"], "degradation_mean_km_per_s": deg_summary["mean"], "degradation_sd_km_per_s": deg_summary["sd_population"], "degradation_ci95_lower_km_per_s": deg_summary["ci95_lower"], "degradation_ci95_upper_km_per_s": deg_summary["ci95_upper"], "bootstrap_unit": "unique geological target after within-target repetition averaging"})
    _write_csv_local(summary_rows, out / "summary.csv")
    _write_json_local({"noise_levels_s": levels, "nonzero_repetitions": repetitions, "matched_noise": "same target/level/seed perturbation for Full-input PCA-ridge, Travel-time-only PCA-ridge, and Reference-ray baseline", "models_retrained": False, "test_selection": False, "reference_ray_residual": "t_obs_noisy - t_FSM(s_0)", "target_bootstrap_distinct_from_shuffle": True}, out / "metadata.json")
    return {"rows": rows, "target_level_rows": target_level_rows, "summary_rows": summary_rows}


def run_coverage_and_structure(
    node_native: Mapping[str, Any],
    reference_ray: Mapping[str, Any],
    output_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Regenerate direct-cell coverage and generator-mask diagnostics."""

    test = node_native["test"]
    full_predictions = node_native["predictions"]["realistic_full"]
    all_cases = node_native["cases"]
    test_indices = [all_cases.index(case) for case in test]
    full_test_nodes = full_predictions[test_indices]
    travel_model = node_native["models"]["travel_time_only"]
    raw_features = node_native["features"]["travel_time_only"]
    all_indices = {case.target_id: index for index, case in enumerate(all_cases)}
    travel_test_nodes = travel_model.predict(raw_features[[all_indices[case.target_id] for case in test]])
    out = output_root / "coverage_structure"
    out.mkdir(parents=True, exist_ok=True)
    coverage_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    methods = {
        "realistic_full": np.vstack([_cell_prediction(case, prediction) for case, prediction in zip(test, full_test_nodes, strict=True)]),
        "travel_time_only": np.vstack([_cell_prediction(case, prediction) for case, prediction in zip(test, travel_test_nodes, strict=True)]),
        "reference_ray": np.vstack([reference_ray["test_predictions"][case.target_id] for case in test]),
    }
    for case_index, case in enumerate(test):
        geometry, _, direct_truth = reference_ray["cache"][case.target_id]
        predictions = {method: values[case_index] for method, values in methods.items()}
        for threshold in config["coverage_thresholds_km"]:
            mask = geometry.coverage_km > 0.0 if float(threshold) == 0.0 else geometry.coverage_km >= float(threshold)
            for method, prediction in predictions.items():
                metrics = _prediction_metrics_from_arrays(direct_truth[mask], prediction[mask]) if np.any(mask) else {"rmse": None, "mae": None, "bias": None}
                coverage_rows.append({"target_id": case.target_id, "family": case.family, "method_id": method, "coverage_threshold_km": float(threshold), "cell_count": int(np.count_nonzero(mask)), "coverage_fraction": float(np.mean(mask)), "direct_cell_rmse_km_per_s": metrics["rmse"], "direct_cell_mae_km_per_s": metrics["mae"], "direct_cell_bias_km_per_s": metrics["bias"]})
        masks = build_structure_masks(case)
        for method, prediction in predictions.items():
            for scope, label_map in masks.items():
                for label, mask in label_map.items():
                    metrics = _prediction_metrics_from_arrays(direct_truth[mask], prediction[mask]) if np.any(mask) else {"rmse": None, "mae": None, "bias": None}
                    structure_rows.append({"target_id": case.target_id, "family": case.family, "method_id": method, "metric_scope": scope, "mask_label": label, "cell_count": int(np.count_nonzero(mask)), "direct_cell_rmse_km_per_s": metrics["rmse"], "direct_cell_mae_km_per_s": metrics["mae"], "direct_cell_bias_km_per_s": metrics["bias"]})
    _write_csv_local(coverage_rows, out / "coverage_metrics.csv")
    _write_csv_local(structure_rows, out / "structure_metrics.csv")
    coverage_summary: list[dict[str, Any]] = []
    for method in methods:
        for threshold in config["coverage_thresholds_km"]:
            selected = [row for row in coverage_rows if row["method_id"] == method and float(row["coverage_threshold_km"]) == float(threshold) and row["direct_cell_rmse_km_per_s"] not in (None, "")]
            values = [float(row["direct_cell_rmse_km_per_s"]) for row in selected]
            rmse_summary = _target_level_summary(values, f"coverage:{method}:{threshold}", int(config["bootstrap_resamples"])) if values else None
            coverage_summary.append({"method_id": method, "coverage_threshold_km": float(threshold), "target_count": len(values), "mean_direct_cell_rmse_km_per_s": rmse_summary["mean"] if rmse_summary else None, "mean_direct_cell_rmse_ci95_lower_km_per_s": rmse_summary["ci95_lower"] if rmse_summary else None, "mean_direct_cell_rmse_ci95_upper_km_per_s": rmse_summary["ci95_upper"] if rmse_summary else None, "mean_coverage_fraction": float(np.mean([float(row["coverage_fraction"]) for row in selected])) if selected else None, "bootstrap_resamples": int(config["bootstrap_resamples"]), "bootstrap_unit": "unique geological target"})
    _write_csv_local(coverage_summary, out / "coverage_summary.csv")
    _write_json_local({"coverage_source": "accumulated reference-model fixed-ray path length", "thresholds_km": list(config["coverage_thresholds_km"]), "positive_coverage_rule": "coverage_km > 0.0", "coverage_is_not_true_model_resolution": True}, out / "coverage_metadata.json")
    _write_json_local({"mask_source": "generator-defined analytic geometry sampled at target-cell centers", "predicted_localization_detector": False, "truth_definition": "direct analytic cell-center evaluation"}, out / "structure_metadata.json")
    return {"coverage_rows": coverage_rows, "coverage_summary": coverage_summary, "structure_rows": structure_rows, "methods": methods}


def _display_target_label(target_id: str) -> str:
    """Reader-facing form of an internal target identifier.

    ``salt_dome_target_26`` -> ``salt dome, target 26``. Used for figure text only;
    the stored identifier is unchanged in every record.
    """

    family, separator, number = target_id.rpartition("_target_")
    if not separator:
        return target_id.replace("_", " ")
    return f"{family.replace('_', ' ')}, target {number}"


def _figure_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300})


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")


def _relative_output_path(path: str | Path, output: Path) -> str:
    """Serialize generated artifact paths without machine-local prefixes."""

    return Path(path).resolve().relative_to(output.resolve()).as_posix()


def _relative_output_entry(entry: Any, output: Path) -> Any:
    """Convert nested figure manifest entries to output-relative paths."""

    if isinstance(entry, Mapping):
        return {key: _relative_output_entry(value, output) for key, value in entry.items()}
    return _relative_output_path(entry, output)


def _direct_field(case: ProductionCase, values: np.ndarray) -> np.ndarray:
    nx, ny, nz = (len(case.target_grid.x_coordinates_km) - 1, len(case.target_grid.y_coordinates_km) - 1, len(case.target_grid.z_coordinates_km) - 1)
    return np.asarray(values).reshape((nz, ny, nx))


def _plot_workflow(out: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    _figure_style()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    boxes = {
        "target": (0.2, 2.35, 2.0, 1.3, "Common synthetic\ntarget + acquisition"),
        "obs": (2.8, 2.35, 2.0, 1.3, "Common observed\ntravel times"),
        "full": (5.55, 4.0, 2.5, 1.25, "Full-input PCA-ridge\ncoordinates + distance + time"),
        "ray": (5.55, 1.0, 2.5, 1.25, "Reference-ray baseline\nfixed $G_{\\mathrm{ref}}$ + regularization"),
        "eval": (9.35, 2.35, 2.35, 1.3, "Direct analytic\ncell-center evaluation"),
    }
    for key, (x, y, w, h, label) in boxes.items():
        color = "#dbeafe" if key in {"target", "obs", "eval"} else "#fef3c7"
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=color, edgecolor="#1f2937", linewidth=1.2)
        ax.add_patch(patch); ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", linespacing=1.3)
    arrows = [((2.2, 3.0), (2.8, 3.0)), ((4.8, 3.0), (5.55, 4.55)), ((4.8, 3.0), (5.55, 1.65)), ((8.05, 4.55), (9.35, 3.25)), ((8.05, 1.65), (9.35, 2.75))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.1, color="#374151"))
    ax.text(6.8, 5.55, "Estimator branch", ha="center", va="center", color="#92400e")
    ax.text(6.8, 0.42, "Independent comparator branch", ha="center", va="center", color="#92400e")
    ax.set_title("Synthetic benchmark workflow: shared inputs, separate reconstruction branches, common endpoint", pad=10)
    path = out / "figure_1_workflow.png"; _save_figure(fig, path); plt.close(fig)
    return {"png": path.as_posix()}


def _plot_pca_spectrum(
    pca: Any,
    validation: Sequence[ProductionCase],
    out: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_style()
    cumulative = np.cumsum(pca.explained_variance_ratio)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    components = np.arange(1, len(cumulative) + 1)
    ax.plot(components, cumulative, "o-", color="#1d4ed8", label="Cumulative training-target variance")
    ax.axvline(1, color="#6b7280", linewidth=0.9, linestyle=":", label="Selected node-native predictive K=1")
    ax.set_xlabel("PCA components (K)"); ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0.0, 1.02); ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    targets = np.vstack([case.target_vector for case in validation])
    truth_cells = [cell_centered_values_from_node_vector(case.target_vector, _node_shape(case.target_grid)) for case in validation]
    oracle_values = []
    for component_count in range(1, int(pca.rank) + 1):
        reconstructed = reconstruct_from_pca(pca, targets, component_count)
        oracle_values.append(float(np.mean([_rmse(truth, _cell_prediction(case, prediction)) for case, truth, prediction in zip(validation, truth_cells, reconstructed, strict=True)])))
    ax2.plot(components, oracle_values, "s--", color="#b45309", label="Validation oracle")
    ax2.set_ylabel("Validation oracle eight-corner cell RMSE (km/s)")
    ax2.set_ylim(bottom=0.0)
    ax.set_title("Training-only PCA spectrum and validation oracle")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="center right")
    path = out / "figure_2_pca_spectrum.png"; _save_figure(fig, path); plt.close(fig)
    return {"png": path.as_posix()}


def _plot_primary_comparison(node_native: Mapping[str, Any], reference_ray: Mapping[str, Any], out: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_style()
    test_rows = [row for row in node_native["metric_rows"] if row["split"] == "test"]
    methods = ["realistic_full", "travel_time_only", "reference_ray"]
    values = {
        "realistic_full": [float(row["direct_cell_rmse_km_per_s"]) for row in test_rows if row["method_id"] == "realistic_full"],
        "travel_time_only": [float(row["direct_cell_rmse_km_per_s"]) for row in test_rows if row["method_id"] == "travel_time_only"],
        "reference_ray": [float(row["direct_cell_rmse_km_per_s"]) for row in reference_ray["test_rows"]],
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.boxplot(
        [values[method] for method in methods],
        labels=[PUBLIC_METHOD_LABELS[method] for method in methods],
        showmeans=False,
        flierprops={"marker": "o", "markerfacecolor": "none", "markeredgecolor": "#111827", "markersize": 6},
        medianprops={"color": "#f97316", "linewidth": 2.0},
        zorder=1,
    )
    rng = np.random.default_rng(_stable_seed("figure-3-jitter"))
    point_colors = {"realistic_full": "#2563eb", "travel_time_only": "#b45309", "reference_ray": "#047857"}
    for index, method in enumerate(methods, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(values[method]))
        ax.scatter(
            np.full(len(values[method]), index, dtype=float) + jitter,
            values[method],
            s=22,
            alpha=0.62,
            color=point_colors[method],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            index,
            float(np.mean(values[method])),
            marker="^",
            s=54,
            color="#15803d",
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
    ax.set_ylabel("Direct-cell RMSE (km/s)"); ax.set_title("Primary reconstruction comparison on held-out test targets")
    ax.grid(axis="y", alpha=0.25); plt.xticks(rotation=15, ha="right")
    path = out / "figure_3_primary_method_comparison.png"; _save_figure(fig, path); plt.close(fig)
    return {"png": path.as_posix()}


def _select_structural_cases(node_native: Mapping[str, Any], test: Sequence[ProductionCase]) -> list[dict[str, Any]]:
    rows = [row for row in node_native["metric_rows"] if row["split"] == "test" and row["method_id"] == "realistic_full"]
    ordered = sorted(rows, key=lambda row: (float(row["direct_cell_rmse_km_per_s"]), str(row["target_hash"])))
    indices = (len(ordered) // 4, (3 * len(ordered)) // 4)
    by_id = {case.target_id: case for case in test}
    return [{"selection_index_zero_based": index, "quantile_label": "lower quartile" if index == indices[0] else "upper quartile", "target_id": ordered[index]["target_id"], "target_hash": ordered[index]["target_hash"], "direct_cell_rmse_km_per_s": float(ordered[index]["direct_cell_rmse_km_per_s"]), "case_family": by_id[ordered[index]["target_id"]].family} for index in indices]


def _deterministic_structure_slice(case: ProductionCase) -> tuple[int, str, int]:
    """Choose a deterministic display slice without changing any test metric."""

    y_centers = np.asarray(case.target_grid.y_coordinates_km[:-1]) + 0.5 * np.diff(case.target_grid.y_coordinates_km)
    common_index = int(np.argmin(np.abs(y_centers - 50.0)))
    if case.family == "layered":
        return common_index, "common y=50 km slice (nearest cell center)", 0

    shape = (
        len(case.target_grid.z_coordinates_km) - 1,
        len(case.target_grid.y_coordinates_km) - 1,
        len(case.target_grid.x_coordinates_km) - 1,
    )
    masks = build_structure_masks(case)
    if "structure" in masks:
        body = masks["structure"]["body"].reshape(shape)
        scores = [int(np.count_nonzero(body[:, index, :])) for index in range(body.shape[1])]
        parameters = _family_parameters(case)
        center_y = float(parameters["center_km"][1])
        index = max(
            range(len(scores)),
            key=lambda value: (
                scores[value],
                -abs(float(y_centers[value]) - center_y),
                -float(y_centers[value]),
            ),
        )
        return (
            int(index),
            "analytic body slice maximizing body-cell count; ties nearest analytic body centre y then smallest y",
            int(scores[index]),
        )

    if "fault" in masks:
        positive = masks["fault"]["positive_side"].reshape(shape)
        scores = [
            int(np.count_nonzero(positive[:, index, 1:] != positive[:, index, :-1]))
            for index in range(positive.shape[1])
        ]
        parameters = _family_parameters(case)
        fault_y = float(parameters["fault_y_km"])
        index = max(
            range(len(scores)),
            key=lambda value: (
                scores[value],
                -abs(float(y_centers[value]) - fault_y),
                -float(y_centers[value]),
            ),
        )
        return (
            int(index),
            "fault-intersecting slice maximizing adjacent side changes; ties nearest fault y then smallest y",
            int(scores[index]),
        )

    return common_index, "common y=50 km slice (nearest cell center)", 0


def _plot_structural_fields(node_native: Mapping[str, Any], test: Sequence[ProductionCase], out: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_style()
    selected = _select_structural_cases(node_native, test)
    by_id = {case.target_id: case for case in test}
    all_cases = node_native["cases"]
    node_by_id = {case.target_id: node for case, node in zip(all_cases, node_native["predictions"]["realistic_full"], strict=True)}
    fields_by_case: list[tuple[dict[str, Any], ProductionCase, int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    velocity_values: list[np.ndarray] = []
    error_values: list[np.ndarray] = []
    for item in selected:
        case = by_id[item["target_id"]]
        true = _direct_field(case, direct_analytic_cell_truth(case))
        prediction = _direct_field(case, _cell_prediction(case, node_by_id[case.target_id]))
        error = prediction - true
        y_centers = np.asarray(case.target_grid.y_coordinates_km[:-1]) + 0.5 * np.diff(case.target_grid.y_coordinates_km)
        y_index, slice_rule, structure_cell_count = _deterministic_structure_slice(case)
        item["display_y_index"] = int(y_index)
        item["display_y_km"] = float(y_centers[y_index])
        item["display_slice_rule"] = slice_rule
        item["display_structure_cell_count"] = int(structure_cell_count)
        sliced = (true[:, y_index, :], prediction[:, y_index, :], error[:, y_index, :])
        fields_by_case.append((item, case, y_index, sliced))
        velocity_values.extend([sliced[0], sliced[1]])
        error_values.append(sliced[2])
    _write_json_local(
        {
            "population": "38 frozen test targets",
            "metric": "Full-input PCA-ridge all-cell direct-cell RMSE",
            "sort": "ascending direct-cell RMSE then target_hash",
            "selection_indices_zero_based": [9, 28],
            "slice_rule": "for each selected target, use the deterministic structure-intersecting slice for its family; body families maximize analytic body-cell count with ties nearest analytic body centre y then smallest y",
            "selected": selected,
        },
        out / "figure_4_selection.json",
    )
    velocity_min = float(np.min([np.min(values) for values in velocity_values]))
    velocity_max = float(np.max([np.max(values) for values in velocity_values]))
    error_limit = max(float(np.max([np.max(np.abs(values)) for values in error_values])), 1.0e-6)
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.4), constrained_layout=True)
    truth_image = None
    error_image = None
    panel_titles = ("Truth", "Full-input PCA-ridge prediction", "Prediction − truth")
    for row_index, (item, case, y_index, fields) in enumerate(fields_by_case):
        for column_index, (axis, field, title) in enumerate(zip(axes[row_index], fields, panel_titles, strict=True)):
            image = axis.imshow(
                field,
                origin="upper",
                aspect="auto",
                cmap="coolwarm" if column_index == 2 else "viridis",
                vmin=-error_limit if column_index == 2 else velocity_min,
                vmax=error_limit if column_index == 2 else velocity_max,
                extent=(0, 100, 30, 0),
            )
            if column_index == 0:
                truth_image = image
            if column_index == 2:
                error_image = image
            axis.set_title(title)
            axis.set_xlabel("x (km)")
            axis.set_ylabel("depth z (km)")
            panel_label = chr(97 + row_index * 3 + column_index)
            axis.text(0.02, 0.97, f"({panel_label})", transform=axis.transAxes, ha="left", va="top", color="white", fontsize=9, fontweight="bold", bbox={"facecolor": "#111827", "alpha": 0.55, "pad": 2.0})
            axes[row_index, 0].set_ylabel(f"{item['quantile_label'].capitalize()}\n depth z (km)")
        axes[row_index, 0].set_title(f"{panel_titles[0]}\n{_display_target_label(str(item['target_id']))}\nFull-input RMSE = {item['direct_cell_rmse_km_per_s']:.3f} km/s")
        axes[row_index, 1].set_title(f"{panel_titles[1]}\ny={float(np.asarray(case.target_grid.y_coordinates_km[:-1])[y_index] + 0.5 * np.diff(case.target_grid.y_coordinates_km)[y_index]):.2f} km")
        axes[row_index, 2].set_title(panel_titles[2])
    if truth_image is not None:
        fig.colorbar(truth_image, ax=axes[:, :2].ravel().tolist(), shrink=0.88, pad=0.02, label="P-wave velocity (km/s)")
    if error_image is not None:
        fig.colorbar(error_image, ax=axes[:, 2].ravel().tolist(), shrink=0.88, pad=0.08, label="Prediction − truth (km/s)")
    fig.suptitle("Structural-field comparison at deterministic quartile test targets", fontsize=12)
    path = out / "figure_4_structural_fields.png"
    _save_figure(fig, path)
    plt.close(fig)
    return {"png": path.as_posix()}


def _select_supplementary_structure_cases(test: Sequence[ProductionCase]) -> list[ProductionCase]:
    selected: list[ProductionCase] = []
    for family in FAMILY_ORDER:
        family_cases = [case for case in test if case.family == family]
        if not family_cases:
            raise ValueError(f"No test target is available for supplementary family {family}.")
        means = np.asarray([float(np.mean(direct_analytic_cell_truth(case))) for case in family_cases])
        family_median = float(np.median(means))
        selected.append(
            min(
                family_cases,
                key=lambda case: (
                    abs(float(np.mean(direct_analytic_cell_truth(case))) - family_median),
                    str(case.target_hash),
                ),
            )
        )
    return selected


def _supplementary_structure_slice(case: ProductionCase) -> tuple[int, str, str]:
    y_centers = np.asarray(case.target_grid.y_coordinates_km[:-1]) + 0.5 * np.diff(case.target_grid.y_coordinates_km)
    common_index = int(np.argmin(np.abs(y_centers - 50.0)))
    if case.family == "layered":
        return common_index, "common y=50 km slice (nearest cell center)", "layer boundaries are represented by the analytic layered field"
    masks = build_structure_masks(case)
    if "fault" in masks:
        positive = masks["fault"]["positive_side"].reshape(_direct_field(case, direct_analytic_cell_truth(case)).shape)
        scores = [int(np.count_nonzero(positive[:, index, 1:] != positive[:, index, :-1])) for index in range(positive.shape[1])]
        parameters = _family_parameters(case)
        target_y = float(parameters["fault_y_km"])
        index = max(range(len(scores)), key=lambda value: (scores[value], -abs(float(y_centers[value]) - target_y), -float(y_centers[value])))
        return int(index), "fault-intersecting x-z slice maximizing adjacent side changes; ties nearest fault y then smallest y", "fault side boundary overlay"
    body = masks["structure"]["body"].reshape(_direct_field(case, direct_analytic_cell_truth(case)).shape)
    scores = [int(np.count_nonzero(body[:, index, :])) for index in range(body.shape[1])]
    parameters = _family_parameters(case)
    center_y = float(parameters["center_km"][1])
    index = max(range(len(scores)), key=lambda value: (scores[value], -abs(float(y_centers[value]) - center_y), -float(y_centers[value])))
    return int(index), "structure-intersecting x-z slice maximizing analytic body-cell count; ties nearest center y then smallest y", "analytic body boundary overlay"


def _plot_supplementary_structure_slices(node_native: Mapping[str, Any], out: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_style()
    selected = _select_supplementary_structure_cases(node_native["test"])
    selections: list[dict[str, Any]] = []
    slices: list[tuple[np.ndarray, np.ndarray | None, np.ndarray, str]] = []
    for case in selected:
        field = _direct_field(case, direct_analytic_cell_truth(case))
        common_y_index = int(np.argmin(np.abs((np.asarray(case.target_grid.y_coordinates_km[:-1]) + 0.5 * np.diff(case.target_grid.y_coordinates_km)) - 50.0)))
        structure_y_index, rule, overlay_label = _supplementary_structure_slice(case)
        y_centers = np.asarray(case.target_grid.y_coordinates_km[:-1]) + 0.5 * np.diff(case.target_grid.y_coordinates_km)
        masks = build_structure_masks(case)
        overlay = None
        if "fault" in masks:
            overlay = masks["fault"]["positive_side"].reshape(field.shape)[:, structure_y_index, :]
        elif "structure" in masks:
            overlay = masks["structure"]["body"].reshape(field.shape)[:, structure_y_index, :]
        slices.append((field[:, common_y_index, :], overlay, field[:, structure_y_index, :], overlay_label))
        selections.append({"family": case.family, "target_id": case.target_id, "target_hash": case.target_hash, "common_y_index": common_y_index, "common_y_km": float(y_centers[common_y_index]), "structure_y_index": structure_y_index, "structure_y_km": float(y_centers[structure_y_index]), "structure_slice_rule": rule})
    _write_json_local({"selection_population": "38 frozen test targets", "selection_rule": "one representative target per family, closest to that family's median direct analytic target mean, ties by target_hash", "family_order": list(FAMILY_ORDER), "common_slice_rule": "x-z slice at target-grid cell center nearest y=50 km", "structure_slice_rule": "family-specific deterministic analytic-mask intersection rule recorded per target", "selected": selections}, out / "figure_S1_selection.json")
    velocity_min = float(np.min([np.min(values[0]) for values in slices]))
    velocity_max = float(np.max([np.max(values[0]) for values in slices]))
    fig, axes = plt.subplots(2, len(selected), figsize=(15.5, 6.1), constrained_layout=True)
    image = None
    for column, (case, slice_data, selection) in enumerate(zip(selected, slices, selections, strict=True)):
        common_field, _, structure_field, overlay_label = slice_data
        x_centers = np.asarray(case.target_grid.x_coordinates_km[:-1]) + 0.5 * np.diff(case.target_grid.x_coordinates_km)
        z_centers = np.asarray(case.target_grid.z_coordinates_km[:-1]) + 0.5 * np.diff(case.target_grid.z_coordinates_km)
        for row, field in enumerate((common_field, structure_field)):
            axis = axes[row, column]
            image = axis.imshow(field, origin="upper", aspect="auto", cmap="viridis", vmin=velocity_min, vmax=velocity_max, extent=(0, 100, 30, 0))
            if row == 1 and slice_data[1] is not None and case.family != "layered":
                # The mask and image both represent target-cell centres.  Passing
                # those coordinates explicitly keeps the outline on the displayed
                # cell-centre field instead of treating array indices as domain edges.
                axis.contour(x_centers, z_centers, slice_data[1].astype(float), levels=[0.5], colors="#111827", linewidths=0.8)
            axis.set_title(f"{case.family.replace('_', ' ')}\ntarget {str(case.target_id).rpartition('_target_')[2] or case.target_id}", fontsize=8)
            axis.set_xlabel("x (km)")
            if column == 0:
                axis.set_ylabel(("common y≈50 km\n" if row == 0 else "structure slice\n") + "depth z (km)")
            if row == 0:
                axis.text(0.02, 0.97, f"y={selection['common_y_km']:.2f} km", transform=axis.transAxes, ha="left", va="top", color="white", fontsize=7, bbox={"facecolor": "#111827", "alpha": 0.55, "pad": 1.5})
            else:
                axis.text(0.02, 0.97, f"y={selection['structure_y_km']:.2f} km", transform=axis.transAxes, ha="left", va="top", color="white", fontsize=7, bbox={"facecolor": "#111827", "alpha": 0.55, "pad": 1.5})
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02, label="P-wave velocity (km/s)")
    fig.suptitle("Deterministic family slices of direct analytic target fields", fontsize=12)
    path = out / "figure_S1_structure_intersecting_slices.png"
    _save_figure(fig, path)
    plt.close(fig)
    return {"png": path.as_posix(), "selection": (out / "figure_S1_selection.json").as_posix()}


def _plot_noise_coverage(noise: Mapping[str, Any], coverage: Mapping[str, Any], out: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_style()
    # Wider than the other figures on purpose: the two travel-time-only legend
    # entries name different estimators and must be readable at print scale.
    fig, (axis_noise, axis_coverage) = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    colors = {"realistic_full": "#1d4ed8", "travel_time_only": "#b45309", "reference_ray": "#047857"}
    labels = {method: PUBLIC_METHOD_LABELS[method] for method in colors}
    # The two panels come from different records and the identifier
    # travel_time_only denotes a different estimator in each: the
    # independently tuned sensitivity in records/noise/, and the frozen
    # Full-input ablation in records/coverage_structure/. A single lookup
    # gave both panels the same legend text, so one orange curve appeared
    # to continue across the figure. Name them separately.
    noise_labels = dict(labels)
    noise_labels["travel_time_only"] = (
        "Travel-time-only PCA-ridge (independently tuned sensitivity)")
    coverage_labels = dict(labels)
    coverage_labels["travel_time_only"] = (
        "Travel-time-only PCA-ridge (frozen Full-input configuration)")
    for method in colors:
        rows = [row for row in noise["summary_rows"] if row["method_id"] == method]
        rows = sorted(rows, key=lambda row: float(row["noise_std_s"]))
        x = [float(row["noise_std_s"]) for row in rows]
        y = [float(row["rmse_mean_km_per_s"]) for row in rows]
        lo = [float(row["rmse_ci95_lower_km_per_s"]) for row in rows]
        hi = [float(row["rmse_ci95_upper_km_per_s"]) for row in rows]
        axis_noise.plot(x, y, "o-", color=colors[method], label=noise_labels[method])
        axis_noise.fill_between(x, lo, hi, color=colors[method], alpha=0.12)
    axis_noise.set_xlabel("Timing-noise SD (s)"); axis_noise.set_ylabel("Direct-cell RMSE (km/s)"); axis_noise.set_title("Matched timing-noise sensitivity"); axis_noise.grid(alpha=0.25)
    axis_noise.margins(y=0.22)   # headroom so the legend does not cover the data
    axis_noise.tick_params(labelsize=9)
    axis_noise.legend(fontsize=9)
    # The thresholds are plotted at equally spaced positions rather than at their
    # own values. On a linear axis spanning 0 to 10 km the 0 and 0.1 km
    # thresholds land on top of each other, so two of the five were not
    # separable by eye. Categorical spacing costs the visual sense of interval
    # width, which carries nothing here: the thresholds are a chosen sweep, not
    # a measured variable.
    coverage_thresholds: list[float] = []
    for method in colors:
        rows = [row for row in coverage["coverage_summary"] if row["method_id"] == method]
        rows = sorted(rows, key=lambda row: float(row["coverage_threshold_km"]))
        coverage_thresholds = [float(row["coverage_threshold_km"]) for row in rows]
        x = list(range(len(rows)))
        y = [float(row["mean_direct_cell_rmse_km_per_s"]) for row in rows]
        lower = [float(row["mean_direct_cell_rmse_ci95_lower_km_per_s"]) for row in rows]
        upper = [float(row["mean_direct_cell_rmse_ci95_upper_km_per_s"]) for row in rows]
        axis_coverage.errorbar(x, y, yerr=[np.asarray(y) - np.asarray(lower), np.asarray(upper) - np.asarray(y)], fmt="o-", capsize=3, color=colors[method], label=coverage_labels[method])
    axis_coverage.set_xticks(list(range(len(coverage_thresholds))))
    axis_coverage.set_xticklabels(["%g" % value for value in coverage_thresholds])
    axis_coverage.set_xlabel("Minimum accumulated reference-ray coverage (km), equally spaced; 0 denotes >0 km"); axis_coverage.set_ylabel("Direct-cell RMSE (km/s)"); axis_coverage.set_title("Coverage-domain diagnostic"); axis_coverage.grid(alpha=0.25)
    axis_coverage.margins(y=0.16)   # headroom so the legend does not cover the data
    axis_coverage.tick_params(labelsize=9)
    axis_coverage.legend(fontsize=9)
    path = out / "figure_5_coverage_noise.png"; _save_figure(fig, path); plt.close(fig)
    return {"png": path.as_posix()}


def run_figures(
    node_native: Mapping[str, Any],
    cell_native: Mapping[str, Any],
    reference_ray: Mapping[str, Any],
    noise: Mapping[str, Any],
    coverage_structure: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    out = output_root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    workflow = _plot_workflow(out)
    pca = _plot_pca_spectrum(node_native["pca"], node_native["validation"], out)
    primary = _plot_primary_comparison(node_native, reference_ray, out)
    fields = _plot_structural_fields(node_native, node_native["test"], out)
    noise_coverage = _plot_noise_coverage(noise, coverage_structure, out)
    supplementary_structure = _plot_supplementary_structure_slices(node_native, out)
    return {"figure_1": workflow, "figure_2": pca, "figure_3": primary, "figure_4": fields, "figure_5": noise_coverage, "figure_S1": supplementary_structure}


def write_revision_tables(
    node_native: Mapping[str, Any],
    cell_native: Mapping[str, Any],
    travel_sensitivity: Mapping[str, Any],
    reference_ray: Mapping[str, Any],
    noise: Mapping[str, Any],
    coverage_structure: Mapping[str, Any],
    output_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Path]:
    out = output_root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    test_node = [row for row in node_native["metric_rows"] if row["split"] == "test"]
    node_methods = {
        method_id: {
            row["target_id"]: row
            for row in test_node
            if row["method_id"] == method_id
        }
        for method_id in (
            "realistic_full",
            "travel_time_only",
            "no_travel_time",
            "geometry_only",
            "distance_only",
            "shuffled_travel_time",
            "training_target_mean",
        )
    }
    travel_sensitivity_rows = [
        row for row in travel_sensitivity["rows"] if row["split"] == "test"
    ]
    travel_sensitivity_by_id = {
        row["target_id"]: row for row in travel_sensitivity_rows
    }
    reference_rows = {row["target_id"]: row for row in reference_ray["test_rows"]}
    cell_native_rows = {
        row["target_id"]: row for row in cell_native["rows"] if row["split"] == "test"
    }
    prior_rows: dict[str, dict[str, Any]] = {}
    for target_id, (_, reference_cells, truth_cells) in reference_ray["cache"].items():
        if target_id not in {row["target_id"] for row in reference_ray["test_rows"]}:
            continue
        prior_rows[target_id] = {
            "target_id": target_id,
            "family": next(
                case.family for case in node_native["test"] if case.target_id == target_id
            ),
            "direct_cell_rmse_km_per_s": _rmse(truth_cells, reference_cells),
            "direct_cell_mae_km_per_s": _mae(truth_cells, reference_cells),
            "direct_cell_bias_km_per_s": float(np.mean(reference_cells - truth_cells)),
        }

    methods = {
        "Full-input PCA-ridge": [
            float(row["direct_cell_rmse_km_per_s"])
            for row in node_methods["realistic_full"].values()
        ],
        "Travel-time-only PCA-ridge (frozen Full-input configuration)": [
            float(row["direct_cell_rmse_km_per_s"])
            for row in node_methods["travel_time_only"].values()
        ],
        "Travel-time-only PCA-ridge (independently tuned sensitivity)": [
            float(row["direct_cell_rmse_km_per_s"])
            for row in travel_sensitivity_by_id.values()
        ],
        "No-travel-time PCA-ridge": [
            float(row["direct_cell_rmse_km_per_s"])
            for row in node_methods["no_travel_time"].values()
        ],
        "Geometry-only PCA-ridge": [
            float(row["direct_cell_rmse_km_per_s"])
            for row in node_methods["geometry_only"].values()
        ],
        "Euclidean-distance-only PCA-ridge": [
            float(row["direct_cell_rmse_km_per_s"])
            for row in node_methods["distance_only"].values()
        ],
        "Shuffled-time control (PCA-ridge)": [
            float(row["direct_cell_rmse_km_per_s"])
            for row in node_methods["shuffled_travel_time"].values()
        ],
        "Training-target mean": [
            float(row["direct_cell_rmse_km_per_s"])
            for row in node_methods["training_target_mean"].values()
        ],
        "Reference 1-D prior": [
            float(row["direct_cell_rmse_km_per_s"]) for row in prior_rows.values()
        ],
        "Reference-ray baseline": [
            float(row["direct_cell_rmse_km_per_s"])
            for row in reference_rows.values()
        ],
        "Cell-native PCA-ridge sensitivity": [
            float(row["direct_cell_rmse_km_per_s"])
            for row in cell_native["rows"]
            if row["split"] == "test"
        ],
    }
    table2 = []
    for label, values in methods.items():
        summary = _target_level_summary(values, "table2:" + label, int(config["bootstrap_resamples"]))
        table2.append({"method": label, "comparison_role": "primary" if label in {"Full-input PCA-ridge", "Reference-ray baseline"} else "secondary/sensitivity", "target_count": len(values), "rmse_mean_km_per_s": summary["mean"], "rmse_median_km_per_s": summary["median"], "rmse_sd_km_per_s": summary["sd_population"], "ci95_lower_km_per_s": summary["ci95_lower"], "ci95_upper_km_per_s": summary["ci95_upper"], "bootstrap_unit": summary["bootstrap_unit"]})
    _write_csv_local(table2, out / "table_2_test_metrics.csv")
    comparison_specs = (
        ("full_vs_reference_ray", "Full-input PCA-ridge", "Reference-ray baseline", node_methods["realistic_full"], reference_rows, "primary confirmatory"),
        ("travel_time_only_vs_reference_ray", "Travel-time-only PCA-ridge (frozen Full-input configuration)", "Reference-ray baseline", node_methods["travel_time_only"], reference_rows, "secondary"),
        ("full_vs_no_time", "Full-input PCA-ridge", "No-travel-time PCA-ridge", node_methods["realistic_full"], node_methods["no_travel_time"], "secondary"),
        ("full_vs_geometry_only", "Full-input PCA-ridge", "Geometry-only PCA-ridge", node_methods["realistic_full"], node_methods["geometry_only"], "diagnostic"),
        ("full_vs_distance_only", "Full-input PCA-ridge", "Euclidean-distance-only PCA-ridge", node_methods["realistic_full"], node_methods["distance_only"], "diagnostic"),
        ("full_vs_shuffled_time", "Full-input PCA-ridge", "Shuffled-time control (PCA-ridge)", node_methods["realistic_full"], node_methods["shuffled_travel_time"], "secondary"),
        ("full_vs_training_target_mean", "Full-input PCA-ridge", "Training-target mean", node_methods["realistic_full"], node_methods["training_target_mean"], "secondary"),
        ("full_vs_travel_time_only", "Full-input PCA-ridge", "Travel-time-only PCA-ridge (frozen Full-input configuration)", node_methods["realistic_full"], node_methods["travel_time_only"], "secondary"),
        ("full_vs_reference_prior", "Full-input PCA-ridge", "Reference 1-D prior", node_methods["realistic_full"], prior_rows, "secondary"),
        ("travel_time_only_vs_training_target_mean", "Travel-time-only PCA-ridge (frozen Full-input configuration)", "Training-target mean", node_methods["travel_time_only"], node_methods["training_target_mean"], "secondary"),
        ("cell_native_vs_reference_ray", "Cell-native PCA-ridge sensitivity", "Reference-ray baseline", cell_native_rows, reference_rows, "secondary/diagnostic"),
        ("cell_native_vs_node_native", "Cell-native PCA-ridge sensitivity", "Full-input PCA-ridge", cell_native_rows, node_methods["realistic_full"], "secondary/diagnostic"),
    )
    primary_rows = []
    for comparison_id, first, second, first_map, second_map, status in comparison_specs:
        pair = _make_paired_rows(comparison_id, first_map, second_map)
        if len(pair) != 38:
            raise ValueError(f"{comparison_id} must contain exactly 38 paired test targets, found {len(pair)}.")
        summary = _paired_summary(pair, bootstrap_resamples=int(config["bootstrap_resamples"]), tie_tolerance=float(config.get("tie_tolerance", 1.0e-12)))
        mae_summary = _paired_metric_summary(
            pair,
            delta_field="paired_delta_mae",
            metric_label="MAE",
            bootstrap_resamples=int(config["bootstrap_resamples"]),
            tie_tolerance=float(config.get("tie_tolerance", 1.0e-12)),
        )
        summary.update(
            {
                "mae_delta_definition": mae_summary["delta_definition"],
                "mae_bootstrap_resamples": mae_summary["bootstrap_resamples"],
                "mae_mean_delta": mae_summary["mean_delta"],
                "mae_median_delta": mae_summary["median_delta"],
                "mae_sd_population": mae_summary["sd_population"],
                "mae_ci95_lower": mae_summary["ci95_lower"],
                "mae_ci95_upper": mae_summary["ci95_upper"],
                "mae_family_stratified_ci95_lower": mae_summary[
                    "family_stratified_ci95_lower"
                ],
                "mae_family_stratified_ci95_upper": mae_summary[
                    "family_stratified_ci95_upper"
                ],
                "mae_wins": mae_summary["wins"],
                "mae_losses": mae_summary["losses"],
                "mae_ties": mae_summary["ties"],
            }
        )
        summary.update({"first_method": first, "second_method": second, "endpoint": "direct analytic cell-center all-cell RMSE", "status": status})
        primary_rows.append(summary)
        _write_csv_local(pair, out / f"{comparison_id}_paired_deltas.csv")
    _write_csv_local(primary_rows, out / "table_3_paired_comparisons.csv")
    _write_csv_local([row for row in noise["summary_rows"]], out / "table_5_noise.csv")
    grouped_structure: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in coverage_structure["structure_rows"]:
        if row["method_id"] in {"realistic_full", "travel_time_only", "reference_ray"}:
            grouped_structure[(str(row["family"]), str(row["metric_scope"]), str(row["mask_label"]), str(row["method_id"]))].append(row)
    structure = []
    for (family, metric_scope, mask_label, method_id), rows in sorted(grouped_structure.items()):
        structure.append(
            {
                "family": family,
                "metric_scope": metric_scope,
                "mask_label": mask_label,
                "method_id": method_id,
                "target_count": len({str(row["target_id"]) for row in rows}),
                "mean_cell_count": float(np.mean([float(row["cell_count"]) for row in rows])),
                "mean_direct_cell_rmse_km_per_s": float(np.mean([float(row["direct_cell_rmse_km_per_s"]) for row in rows if row["direct_cell_rmse_km_per_s"] not in (None, "")])),
                "mean_direct_cell_mae_km_per_s": float(np.mean([float(row["direct_cell_mae_km_per_s"]) for row in rows if row["direct_cell_mae_km_per_s"] not in (None, "")])),
                "mean_direct_cell_bias_km_per_s": float(np.mean([float(row["direct_cell_bias_km_per_s"]) for row in rows if row["direct_cell_bias_km_per_s"] not in (None, "")])),
            }
        )
    _write_csv_local(structure, out / "table_4_structural_metrics.csv")
    return {"table_2": out / "table_2_test_metrics.csv", "table_3": out / "table_3_paired_comparisons.csv", "table_4": out / "table_4_structural_metrics.csv", "table_5": out / "table_5_noise.csv"}


def run_final_evidence_pass(
    *,
    production_dir: str | Path = PRODUCTION_RELATIVE,
    operator_dir: str | Path = OPERATOR_RELATIVE,
    output_dir: str | Path = REVISION_RELATIVE,
) -> Path:
    """Run the complete evidence-pass revision bundle without overwriting frozen evidence."""

    output = _repo_path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty revision directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = load_revision_config()
    settings = load_settings()
    cases = load_production_cases(_repo_path(production_dir))
    counts = {split: sum(case.split == split for case in cases) for split in ("train", "validation", "test")}
    if counts != {"train": 175, "validation": 37, "test": 38}:
        raise ValueError(f"Unexpected frozen split: {counts}")
    if len({case.target_hash for case in cases}) != 250:
        raise ValueError("Frozen target hashes are not unique.")
    _write_json_local({"artifact_type": "final_evidence_pass", "version": config["version"], "split_counts": counts, "target_count": len(cases), "target_hashes_reused_without_regeneration": True, "canonical_endpoint_record": CANONICAL_ENDPOINT_RECORD, "config_path": "config/final_evidence_pass.json"}, output / "run_metadata.json")
    train = [case for case in cases if case.split == "train"]
    validation = [case for case in cases if case.split == "validation"]
    test = [case for case in cases if case.split == "test"]
    node_native = run_node_native_analysis(cases, output / "evidence", config, settings)
    node_native_direct = run_node_native_direct_cell_selection(node_native, config, output / "evidence")
    travel_features = np.vstack([case.observations[:, 7].reshape(-1) for case in [*train, *validation, *test]])
    travel = _run_independent_pca_sensitivity(method_id="travel_time_only_sensitivity", feature_matrix=travel_features, train=train, validation=validation, test=test, pca_targets=np.vstack([case.target_vector for case in train]), config=config, bounds=node_native["bounds"], output_dir=output / "evidence" / "travel_time_only_sensitivity", selection_domain="secondary validation-only direct analytic cell-center all-cell RMSE")
    cell_native = run_cell_native_sensitivity(train, validation, test, [row for row in node_native["metric_rows"] if row["split"] == "test" and row["method_id"] == "realistic_full"], config, node_native["bounds"], output / "evidence")
    reference_ray = run_reference_ray_analysis(cases, output / "evidence", config, settings, _repo_path(operator_dir))
    finalize_direct_cell_comparisons(node_native, node_native_direct, cell_native, reference_ray, config)
    shuffle = run_multiseed_shuffle(cases, node_native, config, output / "evidence")
    noise = run_noise_analysis_revised(cases, node_native, travel, reference_ray, config, settings, _repo_path(operator_dir), output / "evidence")
    coverage_structure = run_coverage_and_structure(node_native, reference_ray, output / "evidence", config)
    figures = run_figures(node_native, cell_native, reference_ray, noise, coverage_structure, output / "submission")
    tables = write_revision_tables(node_native, cell_native, travel, reference_ray, noise, coverage_structure, output / "submission", config)
    _write_json_local({"figure_paths": {key: _relative_output_entry(value, output) for key, value in figures.items()}, "table_paths": {key: _relative_output_path(value, output) for key, value in tables.items()}, "figure_1_contract": "branched workflow with exact public branch labels and common Direct analytic cell-center evaluation", "figure_4_contract": "true/prediction/error fields for zero-based sorted direct-cell RMSE indices 9 and 28, ties by target_hash"}, output / "submission" / "figure_table_manifest.json")
    _write_json_local({"primary_endpoint": "Full-input PCA-ridge versus Reference-ray baseline", "secondary_endpoints": ["Node-native direct-cell validation selection", "Travel-time-only PCA-ridge sensitivity", "Cell-native PCA-ridge sensitivity", "100-seed shuffled-time sensitivity", "timing-noise and coverage diagnostics", "generator-defined structural diagnostics"], "multiplicity_statement": "Only the Full-input PCA-ridge versus Reference-ray baseline comparison is confirmatory; all other comparisons are secondary, diagnostic, or exploratory and are not treated as independent confirmatory tests.", "test_set_use": "one evaluation after validation-only selection for each analysis", "target_level_bootstrap": int(config["bootstrap_resamples"]), "shuffle_seed_count": int(config["shuffled_time_seed_count"]), "shuffle_target_row_count": len(shuffle["rows"]), "shuffle_inferential_unit": "100 seed-level mean effects; 3,800 target-by-seed rows are audit-only", "cell_native_feature_contract": "full-input representation: 384 observations x 8 scalar fields = 3072 scalars per target"}, output / "evidence" / "analysis_status.json")
    return output


def finish_evidence_round_bundle(
    *,
    output_dir: str | Path = REVISION_RELATIVE,
    operator_dir: str | Path = OPERATOR_RELATIVE,
) -> Path:
    """Finish a bundle whose expensive analyses already completed before packaging.

    This continuation is deliberately narrow: it reloads the persisted reference-ray,
    shuffle, and noise records, recomputes the small in-memory node model needed for
    field plotting, and then writes coverage, figures, tables, and manifests.  It
    never reruns the expanded reference-ray grid or the 100-seed sensitivity.
    """

    output = _repo_path(output_dir)
    evidence = output / "evidence"
    if not (evidence / "reference_ray" / "selected_configuration.json").is_file():
        raise FileNotFoundError("The expensive evidence-pass analyses are not complete.")
    config = load_revision_config()
    settings = load_settings()
    cases = load_production_cases(_repo_path(PRODUCTION_RELATIVE))
    node_native = run_node_native_analysis(cases, evidence, config, settings)
    node_native_direct = run_node_native_direct_cell_selection(node_native, config, evidence)
    validation = [case for case in cases if case.split == "validation"]
    test = [case for case in cases if case.split == "test"]
    reference_dir = evidence / "reference_ray"
    selected = _read_json(reference_dir / "selected_configuration.json")
    test_rows = _read_csv_local(reference_dir / "test_metrics.csv")
    operator = _repo_path(operator_dir)
    with np.load(operator / "operator_consistency_arrays.npz") as artifact:
        fsm_by_target = {str(target_id): np.asarray(times, dtype=float) for target_id, times in zip(artifact["target_ids"].tolist(), artifact["fsm_times_s"], strict=True)}
    cache: dict[str, tuple[RayGeometry, np.ndarray, np.ndarray]] = {}
    for case in [*validation, *test]:
        geometry = _load_geometry(operator / "reference_ray_geometry" / f"{case.target_id}.npz")
        reference_grid = build_layered_reference_velocity_grid(case.target_grid, settings)
        reference_cells = cell_centered_values_from_node_vector(np.asarray(reference_grid.p_velocity_km_per_s, dtype=float), _node_shape(case.target_grid))
        cache[case.target_id] = (geometry, reference_cells, direct_analytic_cell_truth(case))
    reference_predictions: dict[str, np.ndarray] = {}
    for case in test:
        with np.load(reference_dir / "predictions" / f"{case.target_id}.npz") as artifact:
            reference_predictions[case.target_id] = np.asarray(artifact["fixed_cells"], dtype=float)
    reference_ray = {"selected": selected, "test_rows": test_rows, "test_predictions": reference_predictions, "cache": cache, "fsm_by_target": fsm_by_target, "output_dir": reference_dir}
    cell_native = {"rows": _read_csv_local(evidence / "cell_native" / "case_metrics.csv"), "selected": _read_json(evidence / "cell_native" / "selected_configuration.json"), "output_dir": evidence / "cell_native"}
    travel_sensitivity = {"rows": _read_csv_local(evidence / "travel_time_only_sensitivity" / "case_metrics.csv"), "selected": _read_json(evidence / "travel_time_only_sensitivity" / "selected_configuration.json")}
    noise = {"summary_rows": _read_csv_local(evidence / "noise" / "summary.csv")}
    coverage_structure = run_coverage_and_structure(node_native, reference_ray, evidence, config)
    finalize_direct_cell_comparisons(node_native, node_native_direct, cell_native, reference_ray, config)
    figures = run_figures(node_native, cell_native, reference_ray, noise, coverage_structure, output / "submission")
    tables = write_revision_tables(node_native, cell_native, travel_sensitivity, reference_ray, noise, coverage_structure, output / "submission", config)
    _write_json_local({"figure_paths": {key: _relative_output_entry(value, output) for key, value in figures.items()}, "table_paths": {key: _relative_output_path(value, output) for key, value in tables.items()}, "figure_1_contract": "branched workflow with exact public branch labels and common Direct analytic cell-center evaluation", "figure_4_contract": "true/prediction/error fields for zero-based sorted direct-cell RMSE indices 9 and 28, ties by target_hash"}, output / "submission" / "figure_table_manifest.json")
    return output


__all__ = ["CANONICAL_ENDPOINT_RECORD", "REVISION_RELATIVE", "load_revision_config", "run_final_evidence_pass"]
