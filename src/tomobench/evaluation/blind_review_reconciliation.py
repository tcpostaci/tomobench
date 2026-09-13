"""Bounded blind-review reconciliation diagnostics for the frozen benchmark.

This module consumes the frozen 250-target production corpus and existing
corrected test artifacts.  It does not generate targets or acquisitions and it
does not evaluate an additional test-set candidate.  The validation sensitivity
diagnostic fits the predeclared PCA/ridge grid on the training split only; the
direct-cell oracle uses the training-only PCA basis with true held-out target
coefficients.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tomobench.config import load_settings
from tomobench.evaluation.benchmark_v2_final_analysis import (
    BOOTSTRAP_RESAMPLES,
    FEATURE_NAMES,
    PCA_COMPONENT_GRID,
    RIDGE_ALPHA_GRID,
    ProductionCase,
    build_feature_matrices,
    cell_centered_values_from_node_vector,
    fit_observation_pca_model,
    fit_target_pca,
    load_production_cases,
    reconstruct_from_pca,
    _stable_seed,
)
from tomobench.evaluation.robustness_checks import direct_analytic_cell_truth
from tomobench.utils.paths import get_repo_root


PRODUCTION_RELATIVE = Path("outputs/generated/submission_readiness/benchmark_v2_production_250_v1")
CORRECTED_RELATIVE = Path(
    "outputs/generated/submission_readiness/robustness_checks/corrected_classical"
)
OUTPUT_RELATIVE = Path(
    "outputs/generated/submission_readiness/robustness_checks/blind_review_reconciliation"
)
PACKAGED_PRODUCTION_RELATIVE = Path("data/benchmark_v2")
PACKAGED_CORRECTED_RELATIVE = Path("evidence/robustness_checks/corrected_classical")
PACKAGED_OUTPUT_RELATIVE = Path("evidence/robustness_checks/blind_review_reconciliation")
PAPER_RELATIVE = Path("paper/robustness_checks")
SELECTED_COMPONENT_COUNT = 4
SELECTED_RIDGE_ALPHA = 100.0
FAMILY_ORDER = ("block_anomaly", "dyke_intrusion", "faulted", "layered", "salt_dome")
FAMILY_LABELS = {
    "block_anomaly": "Block anomaly",
    "dyke_intrusion": "Dyke intrusion",
    "faulted": "Faulted",
    "layered": "Layered",
    "salt_dome": "Salt dome",
}
RECONCILIATION_METHODS = (
    "realistic_full",
    "travel_time_only",
    "fixed_ray",
    "reference_prior",
)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: _csv_value(row.get(field)) for field in fields} for row in rows)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps(np.asarray(value).tolist(), separators=(",", ":"))
    return value


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _node_shape(case: ProductionCase) -> tuple[int, int, int]:
    return (
        len(case.target_grid.x_coordinates_km),
        len(case.target_grid.y_coordinates_km),
        len(case.target_grid.z_coordinates_km),
    )


def _rmse(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(expected)) ** 2)))


def _mae(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(expected))))


def _bootstrap_mean_ci(values: Sequence[float], seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(array, size=(BOOTSTRAP_RESAMPLES, array.size), replace=True),
        axis=1,
    )
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _summary(values: Sequence[float], seed_label: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    seed = _stable_seed(seed_label)
    lower, upper = _bootstrap_mean_ci(array, seed)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std_population": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
        "bootstrap_ci95_lower": lower,
        "bootstrap_ci95_upper": upper,
    }


def _primary_cell_prediction(
    node_prediction: np.ndarray,
    case: ProductionCase,
    velocity_bounds: tuple[float, float],
) -> np.ndarray:
    """Apply the frozen prediction order: node clipping, then eight-corner average."""

    clipped = np.clip(np.asarray(node_prediction, dtype=float), *velocity_bounds)
    return cell_centered_values_from_node_vector(clipped, _node_shape(case))


def run_direct_cell_pca_oracle(
    cases: Sequence[ProductionCase],
    output_dir: Path,
) -> dict[str, Any]:
    """Compare true held-out PCA projections with direct analytic cell truth."""

    train = [case for case in cases if case.split == "train"]
    test = [case for case in cases if case.split == "test"]
    settings = load_settings()
    velocity_bounds = tuple(
        float(value) for value in settings.velocity_model_generation.velocity_bounds_km_per_s
    )
    pca = fit_target_pca(np.vstack([case.target_vector for case in train]))
    true_targets = np.vstack([case.target_vector for case in test])
    oracle_nodes = reconstruct_from_pca(pca, true_targets, SELECTED_COMPONENT_COUNT)
    rows: list[dict[str, Any]] = []
    truth_transform_rmse: list[float] = []
    oracle_vs_node_average_rmse: list[float] = []
    for index, case in enumerate(test):
        direct_truth = direct_analytic_cell_truth(case)
        predicted_cells = _primary_cell_prediction(oracle_nodes[index], case, velocity_bounds)
        true_node_cells = cell_centered_values_from_node_vector(
            true_targets[index], _node_shape(case)
        )
        truth_transform_rmse.append(_rmse(direct_truth, true_node_cells))
        oracle_vs_node_average_rmse.append(_rmse(true_node_cells, predicted_cells))
        rows.append(
            {
                "target_id": case.target_id,
                "family": case.family,
                "node_rmse_km_per_s": _rmse(true_targets[index], oracle_nodes[index]),
                "node_mae_km_per_s": _mae(true_targets[index], oracle_nodes[index]),
                "direct_cell_rmse_km_per_s": _rmse(direct_truth, predicted_cells),
                "direct_cell_mae_km_per_s": _mae(direct_truth, predicted_cells),
                "direct_truth_vs_true_node_average_rmse_km_per_s": truth_transform_rmse[-1],
                "oracle_cell_vs_true_node_average_rmse_km_per_s": oracle_vs_node_average_rmse[-1],
                "node_clipping_fraction": float(
                    np.mean(
                        (oracle_nodes[index] < velocity_bounds[0])
                        | (oracle_nodes[index] > velocity_bounds[1])
                    )
                ),
            }
        )
    direct_rmse = [row["direct_cell_rmse_km_per_s"] for row in rows]
    direct_mae = [row["direct_cell_mae_km_per_s"] for row in rows]
    node_rmse = [row["node_rmse_km_per_s"] for row in rows]
    summary = {
        "artifact_type": "blind_review_direct_cell_pca_oracle",
        "production_target_count": len(cases),
        "training_target_count": len(train),
        "test_target_count": len(test),
        "pca_component_count": SELECTED_COMPONENT_COUNT,
        "pca_basis_source": "training target vectors only",
        "target_truth": "direct analytic velocity evaluated at target-cell centers",
        "prediction_contract": "clip reconstructed node vector to configured velocity bounds, then average eight corner nodes per cell",
        "node_shape": list(_node_shape(test[0])),
        "cell_shape": [value - 1 for value in _node_shape(test[0])],
        "velocity_bounds_km_per_s": list(velocity_bounds),
        "node_rmse": _summary(node_rmse, "blind-review:direct-cell-oracle:node-rmse"),
        "direct_cell_rmse": _summary(
            direct_rmse, "blind-review:direct-cell-oracle:direct-cell-rmse"
        ),
        "direct_cell_mae": _summary(direct_mae, "blind-review:direct-cell-oracle:direct-cell-mae"),
        "endpoint_decomposition_diagnostics": {
            "mean_direct_truth_vs_true_node_average_rmse_km_per_s": float(
                np.mean(truth_transform_rmse)
            ),
            "mean_oracle_cell_vs_true_node_average_rmse_km_per_s": float(
                np.mean(oracle_vs_node_average_rmse)
            ),
            "interpretation": (
                "The node-space and direct-cell oracle endpoints use the same true PCA coefficients. "
                "Their difference is therefore an endpoint/target-representation effect, not a "
                "coefficient-prediction effect; the two components are not treated as additively separable."
            ),
        },
    }
    _write_csv(rows, output_dir / "direct_cell_pca_oracle_by_target.csv")
    _write_json(summary, output_dir / "direct_cell_pca_oracle_summary.json")
    return {"summary": summary, "rows": rows}


def _candidate_sort_key(
    row: Mapping[str, Any], metric: str
) -> tuple[float, float, int, int, float]:
    return (
        float(row[metric]),
        float(
            row["direct_cell_rmse_mean_km_per_s"]
            if metric == "node_rmse_mean_km_per_s"
            else row["node_rmse_mean_km_per_s"]
        ),
        0 if row["model_name"] == "pca_ridge" else 1,
        int(row["pca_component_count"]),
        float(row["ridge_alpha"] or 0.0),
    )


def run_validation_metric_sensitivity(
    cases: Sequence[ProductionCase],
    output_dir: Path,
) -> dict[str, Any]:
    """Fit the frozen PCA/ridge validation grid under node and direct-cell RMSE."""

    train = [case for case in cases if case.split == "train"]
    validation = [case for case in cases if case.split == "validation"]
    settings = load_settings()
    velocity_bounds = tuple(
        float(value) for value in settings.velocity_model_generation.velocity_bounds_km_per_s
    )
    pca = fit_target_pca(np.vstack([case.target_vector for case in train]))
    features = build_feature_matrices(cases)["realistic_full"]
    train_indices = [index for index, case in enumerate(cases) if case.split == "train"]
    validation_indices = [index for index, case in enumerate(cases) if case.split == "validation"]
    train_targets = np.vstack([case.target_vector for case in train])
    rows: list[dict[str, Any]] = []
    for component_count in PCA_COMPONENT_GRID:
        if component_count > pca.rank:
            continue
        for model_name in ("pca_linear", "pca_ridge"):
            alphas: Sequence[float | None] = (
                (None,) if model_name == "pca_linear" else RIDGE_ALPHA_GRID
            )
            for alpha in alphas:
                model = fit_observation_pca_model(
                    features[train_indices],
                    train_targets,
                    pca,
                    component_count,
                    model_name,
                    alpha,
                    velocity_bounds,
                )
                predictions = model.predict(features[validation_indices])
                node_scores: list[float] = []
                direct_scores: list[float] = []
                for index, case in enumerate(validation):
                    node_scores.append(_rmse(case.target_vector, predictions[index]))
                    direct_scores.append(
                        _rmse(
                            direct_analytic_cell_truth(case),
                            _primary_cell_prediction(predictions[index], case, velocity_bounds),
                        )
                    )
                rows.append(
                    {
                        "feature_variant": "realistic_full",
                        "model_name": model_name,
                        "pca_component_count": component_count,
                        "ridge_alpha": alpha,
                        "validation_target_count": len(validation),
                        "node_rmse_mean_km_per_s": float(np.mean(node_scores)),
                        "direct_cell_rmse_mean_km_per_s": float(np.mean(direct_scores)),
                        "node_rmse_std_population_km_per_s": float(np.std(node_scores)),
                        "direct_cell_rmse_std_population_km_per_s": float(np.std(direct_scores)),
                    }
                )
    node_order = sorted(rows, key=lambda row: _candidate_sort_key(row, "node_rmse_mean_km_per_s"))
    direct_order = sorted(
        rows, key=lambda row: _candidate_sort_key(row, "direct_cell_rmse_mean_km_per_s")
    )
    node_ranks = {id(row): rank for rank, row in enumerate(node_order, start=1)}
    direct_ranks = {id(row): rank for rank, row in enumerate(direct_order, start=1)}
    for row in rows:
        row["node_rmse_rank"] = node_ranks[id(row)]
        row["direct_cell_rmse_rank"] = direct_ranks[id(row)]
        row["is_frozen_selected_configuration"] = bool(
            row["model_name"] == "pca_ridge"
            and int(row["pca_component_count"]) == SELECTED_COMPONENT_COUNT
            and float(row["ridge_alpha"]) == SELECTED_RIDGE_ALPHA
        )
    selected = next(row for row in rows if row["is_frozen_selected_configuration"])
    best_node = node_order[0]
    best_direct = direct_order[0]
    summary = {
        "artifact_type": "blind_review_validation_metric_selection_sensitivity",
        "feature_variant": "realistic_full",
        "training_target_count": len(train),
        "validation_target_count": len(validation),
        "test_evaluated": False,
        "candidate_count": len(rows),
        "candidate_grid": {
            "pca_component_count": list(PCA_COMPONENT_GRID),
            "ridge_alpha": list(RIDGE_ALPHA_GRID),
            "linear_alpha": None,
        },
        "tie_breaking": "metric, other metric, pca_ridge before pca_linear, component count, ridge alpha",
        "frozen_selected_configuration": {
            "model_name": selected["model_name"],
            "pca_component_count": selected["pca_component_count"],
            "ridge_alpha": selected["ridge_alpha"],
            "node_rmse_rank": selected["node_rmse_rank"],
            "direct_cell_rmse_rank": selected["direct_cell_rmse_rank"],
            "node_rmse_mean_km_per_s": selected["node_rmse_mean_km_per_s"],
            "direct_cell_rmse_mean_km_per_s": selected["direct_cell_rmse_mean_km_per_s"],
        },
        "best_by_node_rmse": {
            "model_name": best_node["model_name"],
            "pca_component_count": best_node["pca_component_count"],
            "ridge_alpha": best_node["ridge_alpha"],
            "node_rmse_mean_km_per_s": best_node["node_rmse_mean_km_per_s"],
            "direct_cell_rmse_mean_km_per_s": best_node["direct_cell_rmse_mean_km_per_s"],
        },
        "best_by_direct_cell_rmse": {
            "model_name": best_direct["model_name"],
            "pca_component_count": best_direct["pca_component_count"],
            "ridge_alpha": best_direct["ridge_alpha"],
            "node_rmse_mean_km_per_s": best_direct["node_rmse_mean_km_per_s"],
            "direct_cell_rmse_mean_km_per_s": best_direct["direct_cell_rmse_mean_km_per_s"],
        },
        "selection_changed": bool(best_node is not best_direct),
    }
    _write_csv(rows, output_dir / "validation_metric_selection_sensitivity.csv")
    _write_json(summary, output_dir / "validation_metric_selection_sensitivity_summary.json")
    return {"summary": summary, "rows": rows}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_reference_prior_family_table(corrected_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Write target-level direct-cell RMSE by family for the four main endpoints."""

    source_rows = _read_csv(corrected_dir / "corrected_test_method_metrics.csv")
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in source_rows:
        method = row["method_id"]
        if method not in RECONCILIATION_METHODS:
            continue
        grouped.setdefault((row["family"], method), []).append(
            float(row["direct_cell_center_rmse_km_per_s"])
        )
    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        family_values = {method: grouped[(family, method)] for method in RECONCILIATION_METHODS}
        counts = {len(values) for values in family_values.values()}
        if len(counts) != 1:
            raise ValueError(f"Inconsistent target counts for {family}: {counts}")
        rows.append(
            {
                "family": family,
                "family_label": FAMILY_LABELS[family],
                "target_count": counts.pop(),
                **{
                    f"{method}_mean_direct_cell_rmse_km_per_s": float(np.mean(values))
                    for method, values in family_values.items()
                },
            }
        )
    summary = {
        "artifact_type": "blind_review_reference_prior_family_all_cell_comparison",
        "metric_domain": "direct analytic cell-center all-cell target-level RMSE",
        "source": "corrected_test_method_metrics.csv",
        "methods": list(RECONCILIATION_METHODS),
        "family_target_counts": {row["family"]: row["target_count"] for row in rows},
    }
    _write_csv(rows, output_dir / "reference_prior_family_all_cell_rmse.csv")
    _write_json(summary, output_dir / "reference_prior_family_all_cell_rmse_summary.json")
    return {"summary": summary, "rows": rows}


def run_all_diagnostics(
    production_dir: Path | None = None,
    corrected_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the bounded diagnostics against the frozen corpus."""

    root = get_repo_root()
    if production_dir is None:
        working_production = root / PRODUCTION_RELATIVE
        production = (
            working_production
            if (working_production / "benchmark_v2_manifest.json").is_file()
            else root / PACKAGED_PRODUCTION_RELATIVE
        )
    else:
        production = production_dir
    if corrected_dir is None:
        working_corrected = root / CORRECTED_RELATIVE
        corrected = (
            working_corrected
            if (working_corrected / "corrected_test_method_metrics.csv").is_file()
            else root / PACKAGED_CORRECTED_RELATIVE
        )
    else:
        corrected = corrected_dir
    if output_dir is None:
        output = (
            root / OUTPUT_RELATIVE
            if (root / PRODUCTION_RELATIVE / "benchmark_v2_manifest.json").is_file()
            else root / PACKAGED_OUTPUT_RELATIVE
        )
    else:
        output = output_dir
    cases = load_production_cases(production)
    oracle = run_direct_cell_pca_oracle(cases, output)
    validation = run_validation_metric_sensitivity(cases, output)
    prior = write_reference_prior_family_table(corrected, output)
    manifest = {
        "artifact_type": "blind_review_reconciliation_diagnostics",
        "production_directory": _display_path(production, root),
        "corrected_artifact_directory": _display_path(corrected, root),
        "output_directory": _display_path(output, root),
        "case_count": len(cases),
        "observation_count": int(sum(case.observations.shape[0] for case in cases)),
        "features": list(FEATURE_NAMES),
        "new_targets_generated": False,
        "new_acquisitions_generated": False,
        "new_test_candidate_evaluated": False,
        "direct_cell_oracle": oracle["summary"],
        "validation_metric_sensitivity": validation["summary"],
        "reference_prior_family_table": prior["summary"],
    }
    _write_json(manifest, output / "blind_review_reconciliation_manifest.json")
    return manifest
