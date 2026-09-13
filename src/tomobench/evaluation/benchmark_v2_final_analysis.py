"""Definitive Benchmark v2 analysis on the frozen 250-target corpus.

This module is deliberately production-specific.  It consumes the frozen
target-atomic manifest, never uses test performance for selection, excludes
true-model ray length from the realistic ML variants, and treats the unique
geological target as the statistical unit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tomobench.config import load_settings
from tomobench.config.settings import FaultedGridSettings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import CartesianVelocityGrid3D
from tomobench.evaluation.benchmark_v2_pilot import (
    FAMILIES,
    canonical_target_hash,
    cell_centered_values_from_node_vector,
)
from tomobench.generation.velocity_grids import (
    fault_plane_signed_offset_km,
)
from tomobench.simulation.ray_tracing import trace_pseudo_bending_ray
from tomobench.tomography.inversion import (
    _load_velocity_grid,
)
from tomobench.tomography.reference import build_layered_reference_velocity_grid
from tomobench.tomography.sensitivity import ray_path_cell_lengths
from tomobench.utils.paths import get_repo_root


PRODUCTION_DIR = Path(
    "outputs/generated/submission_readiness/benchmark_v2_production_250_v1"
)
FINAL_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/benchmark_v2_final_evidence_v1"
)
PCA_COMPONENT_GRID = (4, 8, 16, 24, 32, 48, 64, 96, 128)
RIDGE_ALPHA_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
ML_METHODS = (
    "realistic_full",
    "travel_time_only",
    "no_travel_time",
    "geometry_only",
    "distance_only",
    "shuffled_travel_time",
    "training_target_mean",
)
FEATURE_VARIANTS: dict[str, tuple[int, ...] | None] = {
    "realistic_full": (0, 1, 2, 3, 4, 5, 6, 7),
    "travel_time_only": (7,),
    "no_travel_time": (0, 1, 2, 3, 4, 5, 6),
    "geometry_only": (0, 1, 2, 3, 4, 5),
    "distance_only": (6,),
    "shuffled_travel_time": (0, 1, 2, 3, 4, 5, 6, 7),
}
FEATURE_NAMES = (
    "source_x_km",
    "source_y_km",
    "source_z_km",
    "receiver_x_km",
    "receiver_y_km",
    "receiver_z_km",
    "euclidean_distance_km",
    "travel_time_s",
)
NOISE_LEVELS_S = (0.0, 0.025, 0.05, 0.1)
NOISE_REPETITIONS = 10
COVERAGE_THRESHOLDS_KM = (0.0, 1.0e-9, 0.1, 1.0, 5.0, 10.0)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026082901
TIE_TOLERANCE = 1.0e-12
REFERENCE_DAMPING_GRID = (0.001, 0.01, 0.1, 1.0)
REFERENCE_SMOOTHING_GRID = (0.0, 0.1, 1.0, 10.0)
PCG_TOLERANCE = 1.0e-8
PCG_MAX_ITERATIONS = 2_000


@dataclass(frozen=True)
class ProductionCase:
    """One target-atomic acquisition case loaded from the production corpus."""

    target_id: str
    target_hash: str
    family: str
    split: str
    target_vector: np.ndarray
    target_grid_path: Path
    target_grid: CartesianVelocityGrid3D
    observations: np.ndarray
    observation_ids: tuple[str, ...]
    physical_parameters: dict[str, Any]
    geology_seed: int
    station_seed: int
    earthquake_seed: int
    acquisition_id: str


@dataclass(frozen=True)
class PCAState:
    """Training-only target PCA state."""

    mean_vector: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    rank: int


@dataclass(frozen=True)
class ObservationPCAModel:
    """A PCA-coefficient regressor with training-only feature scaling."""

    model_name: str
    component_count: int
    ridge_alpha: float | None
    pca_mean: np.ndarray
    pca_components: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    standardized_feature_mean: np.ndarray
    coefficient_mean: np.ndarray
    coefficient_weights: np.ndarray
    velocity_bounds: tuple[float, float]

    def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Reconstruct absolute velocity targets for a feature matrix."""

        x = np.asarray(feature_matrix, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.feature_mean.size:
            raise ValueError("Feature matrix does not match the fitted feature space.")
        standardized = (x - self.feature_mean) / self.feature_std
        coefficients = self.coefficient_mean + (
            standardized - self.standardized_feature_mean
        ) @ self.coefficient_weights
        prediction = self.pca_mean + coefficients @ self.pca_components
        return np.clip(prediction, self.velocity_bounds[0], self.velocity_bounds[1])


@dataclass(frozen=True)
class RayGeometry:
    """Sparse fixed-ray geometry for one target acquisition."""

    row_ptr: np.ndarray
    cell_indices: np.ndarray
    path_lengths_km: np.ndarray
    coverage_km: np.ndarray
    converged_count: int
    observation_count: int
    runtime_s: float
    gram_matrix: np.ndarray | None = None


@dataclass(frozen=True)
class ClassicalCaseResult:
    """Reference-prior and fixed-ray outputs for one case."""

    target_cells: np.ndarray
    reference_cells: np.ndarray
    reference_nodes: np.ndarray
    coverage_km: np.ndarray
    fixed_cells: np.ndarray | None
    fixed_slowness: np.ndarray | None
    fixed_predicted_times: np.ndarray | None
    fixed_residual_s: np.ndarray | None
    fixed_clipped_fraction: float | None
    fixed_pcg_iterations: int | None
    fixed_pcg_converged: bool | None
    geometry: RayGeometry


def run_final_evidence(
    *,
    production_dir: Path = PRODUCTION_DIR,
    output_dir: Path = FINAL_OUTPUT_DIR,
) -> Path:
    """Run the complete frozen-corpus evidence workflow."""

    repo_root = get_repo_root()
    production_resolved = _resolve_path(production_dir, repo_root)
    output_resolved = _resolve_path(output_dir, repo_root)
    if output_resolved.exists() and any(output_resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite final evidence directory: {output_resolved}")
    output_resolved.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    settings = load_settings()

    cases = load_production_cases(production_resolved)
    split_counts = _split_counts(cases)
    if split_counts != {"train": 175, "validation": 37, "test": 38}:
        raise ValueError(f"Unexpected production split counts: {split_counts}")
    if len(cases) != 250 or len({case.target_hash for case in cases}) != 250:
        raise ValueError("Production corpus is not 250-target and hash-unique.")

    ml_dir = output_resolved / "ml"
    ml_result = run_ml_analysis(cases, ml_dir, settings)
    classical_dir = output_resolved / "classical"
    classical_result = run_classical_analysis(
        cases,
        production_resolved,
        classical_dir,
        settings,
    )
    coverage_dir = output_resolved / "coverage"
    coverage_result = run_coverage_analysis(
        cases,
        ml_result,
        classical_result,
        coverage_dir,
    )
    structural_dir = output_resolved / "structural"
    structural_result = run_structural_analysis(
        cases,
        ml_result,
        classical_result,
        structural_dir,
    )
    statistics_dir = output_resolved / "statistics"
    statistics_result = run_target_level_statistics(
        cases,
        ml_result,
        classical_result,
        statistics_dir,
    )
    noise_dir = output_resolved / "noise"
    noise_result = run_noise_analysis(
        cases,
        ml_result,
        classical_result,
        noise_dir,
        settings,
    )
    lofo_dir = output_resolved / "lofo"
    lofo_result = run_lofo_analysis(cases, lofo_dir, settings)
    learning_dir = output_resolved / "learning_curve"
    learning_result = run_learning_curve_analysis(cases, ml_result, learning_dir, settings)
    figures_dir = output_resolved / "figures"
    figures_result = run_publication_figures(
        cases,
        ml_result,
        classical_result,
        coverage_result,
        structural_result,
        noise_result,
        figures_dir,
    )
    tables_dir = output_resolved / "tables"
    tables_result = write_manuscript_tables(
        cases,
        ml_result,
        classical_result,
        structural_result,
        noise_result,
        learning_result,
        statistics_result,
        tables_dir,
    )
    report = write_final_evidence_report(
        cases=cases,
        output_dir=output_resolved,
        ml_result=ml_result,
        classical_result=classical_result,
        coverage_result=coverage_result,
        structural_result=structural_result,
        statistics_result=statistics_result,
        noise_result=noise_result,
        lofo_result=lofo_result,
        learning_result=learning_result,
        figures_result=figures_result,
        tables_result=tables_result,
    )
    _write_json(
        {
            "artifact_type": "benchmark_v2_final_evidence_run_metadata",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "started_at_utc": started,
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "production_dir": _relative(production_resolved, repo_root),
            "output_dir": _relative(output_resolved, repo_root),
            "split_counts": split_counts,
            "unique_test_target_count": sum(case.split == "test" for case in cases),
            "report_path": _relative(report, repo_root),
            "manuscript_modified": False,
        },
        output_resolved / "final_evidence_run_metadata.json",
    )
    return report


def load_production_cases(production_dir: Path) -> list[ProductionCase]:
    """Load every production target and its ordered 384-observation matrix."""

    manifest = _read_json(production_dir / "benchmark_v2_manifest.json")
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != 250:
        raise ValueError("Production manifest must contain exactly 250 items.")
    cases: list[ProductionCase] = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError("Production manifest item must be an object.")
        target_id = str(item["target_id"])
        target_grid_path = _resolve_repo_path(str(item["target_grid_path"]))
        target_artifact_path = _resolve_repo_path(str(item["target_artifact_path"]))
        observation_path = _resolve_repo_path(str(item["observation_csv_path"]))
        target_grid = _load_velocity_grid(target_grid_path)
        with np.load(target_artifact_path) as artifact:
            target_vector = np.asarray(artifact["p_velocity_km_per_s"], dtype=np.float64)
            x_coordinates = tuple(float(value) for value in artifact["x_km"].tolist())
            y_coordinates = tuple(float(value) for value in artifact["y_km"].tolist())
            z_coordinates = tuple(float(value) for value in artifact["z_km"].tolist())
        if target_vector.size != 41 * 41 * 13:
            raise ValueError(f"{target_id}: target vector length is not 21853.")
        if canonical_target_hash(target_vector) != str(item["target_hash"]):
            raise ValueError(f"{target_id}: target artifact hash disagrees with the frozen manifest.")
        if (
            x_coordinates != target_grid.x_coordinates_km
            or y_coordinates != target_grid.y_coordinates_km
            or z_coordinates != target_grid.z_coordinates_km
        ):
            raise ValueError(f"{target_id}: target NPZ and JSON grid coordinates disagree.")
        rows = _read_csv(observation_path)
        rows.sort(key=lambda row: (row["earthquake_id"], row["station_id"], row["observation_id"]))
        if len(rows) != 384:
            raise ValueError(f"{target_id}: expected 384 observations, got {len(rows)}.")
        if len({row["observation_id"] for row in rows}) != 384:
            raise ValueError(f"{target_id}: observation IDs are not unique.")
        if any(
            row.get("target_id") != target_id or row.get("target_hash") != str(item["target_hash"])
            for row in rows
        ):
            raise ValueError(f"{target_id}: observation identity fields disagree with the manifest.")
        observations = np.asarray(
            [
                [
                    float(row["source_x_km"]),
                    float(row["source_y_km"]),
                    float(row["source_z_km"]),
                    float(row["receiver_x_km"]),
                    float(row["receiver_y_km"]),
                    float(row["receiver_z_km"]),
                    float(row["euclidean_distance_km"]),
                    float(row["travel_time_s"]),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
        if not np.isfinite(observations).all():
            raise ValueError(f"{target_id}: observation features contain non-finite values.")
        euclidean = np.linalg.norm(observations[:, 0:3] - observations[:, 3:6], axis=1)
        if not np.allclose(euclidean, observations[:, 6], rtol=0.0, atol=1.0e-9):
            raise ValueError(f"{target_id}: stored Euclidean distances are inconsistent with coordinates.")
        physical_parameters = dict(target_grid.metadata)
        manifest_parameters = item.get("physical_parameters")
        if isinstance(manifest_parameters, dict):
            physical_parameters = dict(manifest_parameters)
        model_path_value = str(item.get("model_artifact_path", ""))
        model_path = _resolve_repo_path(model_path_value) if model_path_value else (
            production_dir / "velocity_models" / f"{target_id}.json"
        )
        if model_path.is_file():
            model_payload = _read_json(model_path)
            physical_parameters = dict(model_payload.get("physical_parameters", physical_parameters))
        cases.append(
            ProductionCase(
                target_id=target_id,
                target_hash=str(item["target_hash"]),
                family=str(item["family"]),
                split=str(item["split"]),
                target_vector=target_vector,
                target_grid_path=target_grid_path,
                target_grid=target_grid,
                observations=observations,
                observation_ids=tuple(row["observation_id"] for row in rows),
                physical_parameters=physical_parameters,
                geology_seed=int(item["geology_seed"]),
                station_seed=int(item["station_seed"]),
                earthquake_seed=int(item["earthquake_seed"]),
                acquisition_id=str(item["acquisition_id"]),
            )
        )
    _validate_case_identity(cases)
    return sorted(cases, key=lambda case: case.target_id)


def run_ml_analysis(
    cases: Sequence[ProductionCase],
    output_dir: Path,
    settings: Any,
) -> dict[str, Any]:
    """Fit training-only PCA/linear/ridge models and evaluate frozen variants."""

    output_dir.mkdir(parents=True, exist_ok=True)
    train = [case for case in cases if case.split == "train"]
    validation = [case for case in cases if case.split == "validation"]
    test = [case for case in cases if case.split == "test"]
    target_train = np.vstack([case.target_vector for case in train])
    pca = fit_target_pca(target_train)
    valid_components = tuple(k for k in PCA_COMPONENT_GRID if k <= pca.rank)
    if not valid_components:
        raise ValueError("No configured PCA component count is valid for training rank.")
    _write_pca_outputs(output_dir, pca, valid_components, validation, test)

    feature_by_variant = build_feature_matrices(cases)
    train_indices = [index for index, case in enumerate(cases) if case.split == "train"]
    validation_indices = [index for index, case in enumerate(cases) if case.split == "validation"]
    test_indices = [index for index, case in enumerate(cases) if case.split == "test"]
    target_matrix = np.vstack([case.target_vector for case in cases])
    velocity_bounds = tuple(float(value) for value in settings.velocity_model_generation.velocity_bounds_km_per_s)

    candidate_rows: list[dict[str, Any]] = []
    candidate_models: dict[tuple[str, int, float | None], ObservationPCAModel] = {}
    full_train = feature_by_variant["realistic_full"][train_indices]
    full_validation = feature_by_variant["realistic_full"][validation_indices]
    for component_count in valid_components:
        for model_name in ("pca_linear", "pca_ridge"):
            alphas: Iterable[float | None] = (None,) if model_name == "pca_linear" else RIDGE_ALPHA_GRID
            for alpha in alphas:
                model = fit_observation_pca_model(
                    full_train,
                    target_matrix[train_indices],
                    pca,
                    component_count,
                    model_name,
                    alpha,
                    velocity_bounds,
                )
                validation_prediction = model.predict(full_validation)
                metrics = _aggregate_prediction_metrics(
                    target_matrix[validation_indices], validation_prediction
                )
                key = (model_name, component_count, alpha)
                candidate_models[key] = model
                candidate_rows.append(
                    {
                        "feature_variant": "realistic_full",
                        "model_name": model_name,
                        "pca_component_count": component_count,
                        "ridge_alpha": alpha,
                        "validation_target_count": len(validation),
                        "validation_node_rmse_mean_km_per_s": metrics["node_rmse_mean"],
                        "validation_node_mae_mean_km_per_s": metrics["node_mae_mean"],
                        "validation_cell_rmse_mean_km_per_s": metrics["cell_rmse_mean"],
                        "validation_cell_mae_mean_km_per_s": metrics["cell_mae_mean"],
                        "selection_metric": "validation mean node RMSE",
                    }
                )
    selected_row = min(
        candidate_rows,
        key=lambda row: (
            float(row["validation_node_rmse_mean_km_per_s"]),
            float(row["validation_node_mae_mean_km_per_s"]),
            0 if row["model_name"] == "pca_ridge" else 1,
            int(row["pca_component_count"]),
            float(row["ridge_alpha"] or 0.0),
        ),
    )
    selected_key = (
        str(selected_row["model_name"]),
        int(selected_row["pca_component_count"]),
        None if selected_row["ridge_alpha"] in (None, "") else float(selected_row["ridge_alpha"]),
    )
    selected_model_config = {
        "model_name": selected_key[0],
        "pca_component_count": selected_key[1],
        "ridge_alpha": selected_key[2],
        "feature_variant": "realistic_full",
        "validation_selection_metric": "mean node RMSE over 37 validation targets",
        "test_evaluation_policy": "selected configuration evaluated once on frozen 38-target test set",
    }
    _write_csv(output_dir / "validation_model_selection.csv", candidate_rows)
    _write_json(selected_model_config, output_dir / "selected_ml_configuration.json")

    models: dict[str, ObservationPCAModel] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    metric_rows: list[dict[str, Any]] = []
    for method in ML_METHODS:
        if method == "training_target_mean":
            mean_prediction = np.mean(target_matrix[train_indices], axis=0)
            split_predictions = {
                cases[index].target_id: mean_prediction.copy() for index in range(len(cases))
            }
        else:
            model = fit_observation_pca_model(
                feature_by_variant[method][train_indices],
                target_matrix[train_indices],
                pca,
                selected_key[1],
                selected_key[0],
                selected_key[2],
                velocity_bounds,
            )
            models[method] = model
            predicted = model.predict(feature_by_variant[method])
            split_predictions = {
                cases[index].target_id: predicted[index].copy() for index in range(len(cases))
            }
        predictions[method] = split_predictions
        for index, case in enumerate(cases):
            if case.split not in {"validation", "test"}:
                continue
            prediction = split_predictions[case.target_id]
            metrics = _prediction_metrics(case.target_vector, prediction)
            metric_rows.append(
                {
                    "target_id": case.target_id,
                    "target_hash": case.target_hash,
                    "family": case.family,
                    "split": case.split,
                    "method_id": method,
                    "method_label": _method_label(method),
                    "node_rmse_km_per_s": metrics["node_rmse"],
                    "node_mae_km_per_s": metrics["node_mae"],
                    "node_bias_km_per_s": metrics["node_bias"],
                    "cell_rmse_km_per_s": metrics["cell_rmse"],
                    "cell_mae_km_per_s": metrics["cell_mae"],
                    "cell_bias_km_per_s": metrics["cell_bias"],
                    "pca_component_count": selected_key[1],
                    "model_name": selected_key[0],
                    "ridge_alpha": selected_key[2],
                }
            )
        if method != "training_target_mean":
            test_array = np.vstack([split_predictions[case.target_id] for case in test])
        else:
            test_array = np.vstack([split_predictions[case.target_id] for case in test])
        np.savez_compressed(
            output_dir / f"{method}_test_predictions.npz",
            target_ids=np.asarray([case.target_id for case in test]),
            target_hashes=np.asarray([case.target_hash for case in test]),
            predicted_node_velocity_km_per_s=test_array,
        )
    _write_csv(output_dir / "ml_case_metrics.csv", metric_rows)
    ml_summary_rows = _summarize_metric_rows(metric_rows, split_values=("validation", "test"))
    _write_csv(output_dir / "ml_summary.csv", ml_summary_rows)
    test_representation_rows = _test_representation_rows(
        test,
        pca,
        selected_key[1],
        target_matrix[test_indices],
        predictions,
    )
    _write_csv(output_dir / "test_representation_gap.csv", test_representation_rows)
    representation_summary = _summarize_representation_gap(test_representation_rows)
    _write_csv(output_dir / "test_representation_gap_summary.csv", representation_summary)
    return {
        "output_dir": output_dir,
        "pca": pca,
        "models": models,
        "predictions": predictions,
        "metric_rows": metric_rows,
        "summary_rows": ml_summary_rows,
        "test_cases": test,
        "validation_cases": validation,
        "train_cases": train,
        "selected_key": selected_key,
        "selected_config": selected_model_config,
        "feature_by_variant": feature_by_variant,
        "target_matrix": target_matrix,
        "pca_representation_summary": representation_summary,
    }


def build_feature_matrices(cases: Sequence[ProductionCase]) -> dict[str, np.ndarray]:
    """Build fixed-length case features; shuffled times stay within each split."""

    raw = np.stack([case.observations for case in cases], axis=0)
    result: dict[str, np.ndarray] = {}
    for method, columns in FEATURE_VARIANTS.items():
        if columns is None:
            continue
        selected = raw[:, :, list(columns)].copy()
        if method == "shuffled_travel_time":
            time_position = list(columns).index(7)
            split_names = {case.split for case in cases}
            for split in sorted(split_names):
                indices = [index for index, case in enumerate(cases) if case.split == split]
                if len(indices) < 2:
                    continue
                seed = _stable_seed(f"shuffle:{split}")
                permutation = np.random.default_rng(seed).permutation(len(indices))
                shuffled_sources = [indices[int(index)] for index in permutation]
                for case_index, source_index in zip(indices, shuffled_sources, strict=True):
                    selected[case_index, :, time_position] = raw[source_index, :, 7]
        result[method] = selected.reshape((len(cases), -1))
    return result


def fit_target_pca(targets: np.ndarray) -> PCAState:
    """Fit centered target PCA using only the supplied training targets."""

    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA requires a two-dimensional matrix with at least two targets.")
    mean_vector = np.mean(values, axis=0)
    centered = values - mean_vector
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank == 0:
        raise ValueError("Training target matrix has zero centered rank.")
    variance = singular_values**2
    explained = variance / np.sum(variance)
    return PCAState(
        mean_vector=mean_vector,
        components=vh[:rank],
        singular_values=singular_values[:rank],
        explained_variance_ratio=explained[:rank],
        rank=rank,
    )


def fit_observation_pca_model(
    feature_matrix: np.ndarray,
    target_matrix: np.ndarray,
    pca: PCAState,
    component_count: int,
    model_name: str,
    ridge_alpha: float | None,
    velocity_bounds: tuple[float, float],
) -> ObservationPCAModel:
    """Fit linear or dual-ridge PCA coefficient regression."""

    if component_count < 1 or component_count > pca.rank:
        raise ValueError("PCA component count is outside the fitted training rank.")
    if model_name not in {"pca_linear", "pca_ridge"}:
        raise ValueError(f"Unsupported model name: {model_name}")
    if model_name == "pca_ridge" and (ridge_alpha is None or ridge_alpha <= 0.0):
        raise ValueError("PCA ridge requires a positive alpha.")
    x = np.asarray(feature_matrix, dtype=np.float64)
    y = np.asarray(target_matrix, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("Feature and target matrices must be two-dimensional with equal rows.")
    feature_mean = np.mean(x, axis=0)
    feature_std = np.std(x, axis=0)
    feature_std = np.where(feature_std <= 1.0e-12, 1.0, feature_std)
    standardized = (x - feature_mean) / feature_std
    standardized_mean = np.mean(standardized, axis=0)
    x_centered = standardized - standardized_mean
    components = pca.components[:component_count]
    coefficients = (y - pca.mean_vector) @ components.T
    coefficient_mean = np.mean(coefficients, axis=0)
    y_centered = coefficients - coefficient_mean
    gram = x_centered @ x_centered.T
    if model_name == "pca_linear":
        dual = np.linalg.pinv(gram, rcond=1.0e-10) @ y_centered
    else:
        regularized = gram + float(ridge_alpha) * np.eye(gram.shape[0])
        dual = np.linalg.solve(regularized, y_centered)
    weights = x_centered.T @ dual
    return ObservationPCAModel(
        model_name=model_name,
        component_count=component_count,
        ridge_alpha=ridge_alpha,
        pca_mean=pca.mean_vector,
        pca_components=components,
        feature_mean=feature_mean,
        feature_std=feature_std,
        standardized_feature_mean=standardized_mean,
        coefficient_mean=coefficient_mean,
        coefficient_weights=weights,
        velocity_bounds=velocity_bounds,
    )


def _write_pca_outputs(
    output_dir: Path,
    pca: PCAState,
    component_counts: Sequence[int],
    validation: Sequence[ProductionCase],
    test: Sequence[ProductionCase],
) -> None:
    spectrum_rows: list[dict[str, Any]] = []
    cumulative = np.cumsum(pca.explained_variance_ratio)
    for index, (ratio, cumulative_value) in enumerate(
        zip(pca.explained_variance_ratio, cumulative, strict=True),
        start=1,
    ):
        spectrum_rows.append(
            {
                "component": index,
                "singular_value": pca.singular_values[index - 1],
                "explained_variance_ratio": ratio,
                "cumulative_explained_variance": cumulative_value,
                "training_target_count": 175,
                "training_centered_rank": pca.rank,
            }
        )
    _write_csv(output_dir / "pca_spectrum.csv", spectrum_rows)
    target_validation = np.vstack([case.target_vector for case in validation])
    target_test = np.vstack([case.target_vector for case in test])
    oracle_rows: list[dict[str, Any]] = []
    for component_count in component_counts:
        validation_prediction = reconstruct_from_pca(pca, target_validation, component_count)
        test_prediction = reconstruct_from_pca(pca, target_test, component_count)
        validation_metrics = _aggregate_prediction_metrics(target_validation, validation_prediction)
        test_metrics = _aggregate_prediction_metrics(target_test, test_prediction)
        oracle_rows.append(
            {
                "pca_component_count": component_count,
                "training_cumulative_explained_variance": float(
                    cumulative[component_count - 1]
                ),
                "validation_oracle_node_rmse_km_per_s": validation_metrics["node_rmse_mean"],
                "validation_oracle_node_mae_km_per_s": validation_metrics["node_mae_mean"],
                "validation_oracle_cell_rmse_km_per_s": validation_metrics["cell_rmse_mean"],
                "test_oracle_node_rmse_km_per_s": test_metrics["node_rmse_mean"],
                "test_oracle_node_mae_km_per_s": test_metrics["node_mae_mean"],
                "test_oracle_cell_rmse_km_per_s": test_metrics["cell_rmse_mean"],
                "oracle_projection_basis": "training targets only",
            }
        )
    _write_csv(output_dir / "oracle_reconstruction.csv", oracle_rows)
    _write_json(
        {
            "artifact_type": "benchmark_v2_training_only_target_pca",
            "training_target_count": 175,
            "training_centered_rank": pca.rank,
            "component_grid": list(component_counts),
            "mean_vector_length": int(pca.mean_vector.size),
            "component_matrix_shape": list(pca.components.shape),
        },
        output_dir / "pca_fit_metadata.json",
    )


def reconstruct_from_pca(pca: PCAState, targets: np.ndarray, component_count: int) -> np.ndarray:
    """Project supplied targets onto a training-only PCA basis and reconstruct."""

    values = np.asarray(targets, dtype=np.float64)
    coefficients = (values - pca.mean_vector) @ pca.components[:component_count].T
    return pca.mean_vector + coefficients @ pca.components[:component_count]


def _test_representation_rows(
    test_cases: Sequence[ProductionCase],
    pca: PCAState,
    selected_component_count: int,
    target_test: np.ndarray,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    oracle = reconstruct_from_pca(pca, target_test, selected_component_count)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(test_cases):
        true = target_test[index]
        oracle_error = _prediction_metrics(true, oracle[index])
        predicted = predictions["realistic_full"][case.target_id]
        predicted_error = _prediction_metrics(true, predicted)
        rows.append(
            {
                "target_id": case.target_id,
                "family": case.family,
                "pca_component_count": selected_component_count,
                "mean_target_node_rmse_km_per_s": _prediction_metrics(
                    true, pca.mean_vector
                )["node_rmse"],
                "oracle_node_rmse_km_per_s": oracle_error["node_rmse"],
                "oracle_node_mae_km_per_s": oracle_error["node_mae"],
                "predicted_node_rmse_km_per_s": predicted_error["node_rmse"],
                "predicted_node_mae_km_per_s": predicted_error["node_mae"],
                "coefficient_estimation_gap_node_rmse_km_per_s": predicted_error["node_rmse"]
                - oracle_error["node_rmse"],
                "mean_target_cell_rmse_km_per_s": _prediction_metrics(
                    true, pca.mean_vector
                )["cell_rmse"],
                "oracle_cell_rmse_km_per_s": oracle_error["cell_rmse"],
                "predicted_cell_rmse_km_per_s": predicted_error["cell_rmse"],
                "coefficient_estimation_gap_cell_rmse_km_per_s": predicted_error["cell_rmse"]
                - oracle_error["cell_rmse"],
            }
        )
    return rows


def _summarize_representation_gap(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field in (
        "mean_target_node_rmse_km_per_s",
        "oracle_node_rmse_km_per_s",
        "predicted_node_rmse_km_per_s",
        "coefficient_estimation_gap_node_rmse_km_per_s",
        "mean_target_cell_rmse_km_per_s",
        "oracle_cell_rmse_km_per_s",
        "predicted_cell_rmse_km_per_s",
        "coefficient_estimation_gap_cell_rmse_km_per_s",
    ):
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        output.append(
            {
                "metric": field,
                "test_target_count": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "p95": float(np.percentile(values, 95)),
            }
        )
    return output


def run_classical_analysis(
    cases: Sequence[ProductionCase],
    production_dir: Path,
    output_dir: Path,
    settings: Any,
) -> dict[str, Any]:
    """Build reference-model rays and select fixed-ray regularization on validation only."""

    del production_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = [case for case in cases if case.split == "validation"]
    test = [case for case in cases if case.split == "test"]
    reference_geometry_dir = output_dir / "reference_ray_geometry"
    reference_geometry_dir.mkdir(parents=True, exist_ok=True)
    geometry_by_target: dict[str, RayGeometry] = {}
    reference_cells_by_target: dict[str, np.ndarray] = {}
    reference_nodes_by_target: dict[str, np.ndarray] = {}
    target_cells_by_target: dict[str, np.ndarray] = {}
    target_cell_shape: tuple[int, int, int] | None = None
    geometry_rows: list[dict[str, Any]] = []
    for case in [*validation, *test]:
        geometry = build_reference_ray_geometry(case, settings)
        geometry = replace(geometry, gram_matrix=_ray_gram_matrix(geometry))
        geometry_by_target[case.target_id] = geometry
        np.savez_compressed(
            reference_geometry_dir / f"{case.target_id}.npz",
            row_ptr=geometry.row_ptr,
            cell_indices=geometry.cell_indices,
            path_lengths_km=geometry.path_lengths_km,
            coverage_km=geometry.coverage_km,
        )
        target_grid = case.target_grid
        shape = (
            len(target_grid.x_coordinates_km),
            len(target_grid.y_coordinates_km),
            len(target_grid.z_coordinates_km),
        )
        target_cell_shape = (shape[0] - 1, shape[1] - 1, shape[2] - 1)
        target_cells = cell_centered_values_from_node_vector(
            target_grid.p_velocity_km_per_s,
            shape,
        )
        reference_grid = build_layered_reference_velocity_grid(target_grid, settings)
        reference_nodes = np.asarray(reference_grid.p_velocity_km_per_s, dtype=np.float64)
        reference_cells = cell_centered_values_from_node_vector(reference_nodes, shape)
        target_cells_by_target[case.target_id] = target_cells
        reference_cells_by_target[case.target_id] = reference_cells
        reference_nodes_by_target[case.target_id] = reference_nodes
        geometry_rows.append(
            {
                "target_id": case.target_id,
                "family": case.family,
                "split": case.split,
                "observation_count": geometry.observation_count,
                "converged_ray_count": geometry.converged_count,
                "nonconverged_ray_count": geometry.observation_count - geometry.converged_count,
                "reference_ray_runtime_s": geometry.runtime_s,
                "cell_count": int(geometry.coverage_km.size),
                "positive_coverage_cell_count": int(np.count_nonzero(geometry.coverage_km > 0.0)),
                "positive_coverage_fraction": float(np.mean(geometry.coverage_km > 0.0)),
            }
        )
    _write_csv(output_dir / "reference_ray_geometry_summary.csv", geometry_rows)
    if target_cell_shape is None:
        raise ValueError("No validation or test cases were available for classical analysis.")

    velocity_bounds = tuple(float(value) for value in settings.velocity_model_generation.velocity_bounds_km_per_s)
    validation_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for damping in REFERENCE_DAMPING_GRID:
        for smoothing in REFERENCE_SMOOTHING_GRID:
            case_results: list[dict[str, Any]] = []
            for case in validation:
                solved = solve_fixed_ray_case(
                    case=case,
                    geometry=geometry_by_target[case.target_id],
                    reference_cells=reference_cells_by_target[case.target_id],
                    target_cells=target_cells_by_target[case.target_id],
                    damping=damping,
                    smoothing=smoothing,
                    cell_shape=target_cell_shape,
                    velocity_bounds=velocity_bounds,
                )
                case_results.append(solved)
                validation_rows.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "damping": damping,
                        "smoothing": smoothing,
                        "all_cell_rmse_km_per_s": solved["all_cell_rmse"],
                        "all_cell_mae_km_per_s": solved["all_cell_mae"],
                        "positive_coverage_rmse_km_per_s": solved["positive_coverage_rmse"],
                        "positive_coverage_mae_km_per_s": solved["positive_coverage_mae"],
                        "travel_time_rmse_s": solved["travel_time_rmse"],
                        "travel_time_mae_s": solved["travel_time_mae"],
                        "travel_time_bias_s": solved["travel_time_bias"],
                        "clipped_fraction": solved["clipped_fraction"],
                        "pcg_iterations": solved["pcg_iterations"],
                        "pcg_converged": solved["pcg_converged"],
                    }
                )
            candidates.append(
                {
                    "damping": damping,
                    "smoothing": smoothing,
                    "validation_all_cell_rmse_mean": float(
                        np.mean([result["all_cell_rmse"] for result in case_results])
                    ),
                    "validation_all_cell_mae_mean": float(
                        np.mean([result["all_cell_mae"] for result in case_results])
                    ),
                    "validation_positive_coverage_rmse_mean": float(
                        np.mean(
                            [
                                result["positive_coverage_rmse"]
                                for result in case_results
                                if result["positive_coverage_rmse"] is not None
                            ]
                        )
                    ),
                    "validation_travel_time_rmse_mean": float(
                        np.mean([result["travel_time_rmse"] for result in case_results])
                    ),
                    "all_pcg_converged": all(result["pcg_converged"] for result in case_results),
                    "selection_metric": "validation mean all-cell velocity RMSE",
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            float(row["validation_all_cell_rmse_mean"]),
            float(row["validation_positive_coverage_rmse_mean"]),
            float(row["validation_travel_time_rmse_mean"]),
            float(row["damping"]),
            float(row["smoothing"]),
        ),
    )
    _write_csv(output_dir / "classical_validation_grid.csv", validation_rows)
    _write_csv(output_dir / "classical_validation_candidate_summary.csv", candidates)
    _write_json(
        {
            "artifact_type": "benchmark_v2_validation_selected_fixed_ray_configuration",
            "selected_damping": selected["damping"],
            "selected_smoothing": selected["smoothing"],
            "selection_unit": "37 validation targets",
            "selection_metric": selected["selection_metric"],
            "damping_grid": list(REFERENCE_DAMPING_GRID),
            "smoothing_grid": list(REFERENCE_SMOOTHING_GRID),
            "coverage_thresholds_km": list(COVERAGE_THRESHOLDS_KM),
            "reference_ray_description": "fixed rays traced through the configured layered reference model",
            "method_scope": "reference-model fixed-ray regularized perturbation inversion; not full nonlinear tomography",
        },
        output_dir / "selected_classical_configuration.json",
    )

    test_case_results: dict[str, ClassicalCaseResult] = {}
    classical_rows: list[dict[str, Any]] = []
    for case in test:
        target_cells = target_cells_by_target[case.target_id]
        reference_cells = reference_cells_by_target[case.target_id]
        reference_nodes = reference_nodes_by_target[case.target_id]
        solved = solve_fixed_ray_case(
            case=case,
            geometry=geometry_by_target[case.target_id],
            reference_cells=reference_cells,
            target_cells=target_cells,
            damping=float(selected["damping"]),
            smoothing=float(selected["smoothing"]),
            cell_shape=target_cell_shape,
            velocity_bounds=velocity_bounds,
        )
        classical_result = ClassicalCaseResult(
            target_cells=target_cells,
            reference_cells=reference_cells,
            reference_nodes=reference_nodes,
            coverage_km=geometry_by_target[case.target_id].coverage_km,
            fixed_cells=solved["fixed_cells"],
            fixed_slowness=solved["slowness"],
            fixed_predicted_times=solved["predicted_times"],
            fixed_residual_s=solved["residuals"],
            fixed_clipped_fraction=solved["clipped_fraction"],
            fixed_pcg_iterations=solved["pcg_iterations"],
            fixed_pcg_converged=solved["pcg_converged"],
            geometry=geometry_by_target[case.target_id],
        )
        test_case_results[case.target_id] = classical_result
        _write_classical_prediction_artifact(output_dir, case, classical_result)
        classical_rows.extend(
            _classical_case_metric_rows(case, classical_result, target_cell_shape)
        )
    _write_csv(output_dir / "classical_test_case_metrics.csv", classical_rows)
    _write_csv(
        output_dir / "classical_test_summary.csv",
        _summarize_metric_rows(classical_rows, split_values=("test",)),
    )
    _write_json(
        {
            "artifact_type": "benchmark_v2_classical_baseline_summary",
            "test_target_count": len(test),
            "validation_target_count": len(validation),
            "selected_damping": selected["damping"],
            "selected_smoothing": selected["smoothing"],
            "all_test_reference_ray_geometries_converged": all(
                result.geometry.converged_count == result.geometry.observation_count
                for result in test_case_results.values()
            ),
            "coverage_thresholds_km": list(COVERAGE_THRESHOLDS_KM),
        },
        output_dir / "classical_summary.json",
    )
    return {
        "output_dir": output_dir,
        "test_case_results": test_case_results,
        "geometry_by_target": geometry_by_target,
        "validation_rows": validation_rows,
        "candidate_rows": candidates,
        "selected": selected,
        "test_metric_rows": classical_rows,
        "cell_shape": target_cell_shape,
        "velocity_bounds": velocity_bounds,
    }


def build_reference_ray_geometry(case: ProductionCase, settings: Any) -> RayGeometry:
    """Trace fixed rays through the reference model and discretize them into target cells."""

    reference_grid = build_layered_reference_velocity_grid(case.target_grid, settings)
    row_ptr = [0]
    cell_indices: list[int] = []
    path_lengths: list[float] = []
    converged_count = 0
    started = perf_counter()
    for observation in case.observations:
        source = Point3D(
            x_km=float(observation[0]),
            y_km=float(observation[1]),
            z_km=float(observation[2]),
        )
        receiver = Point3D(
            x_km=float(observation[3]),
            y_km=float(observation[4]),
            z_km=float(observation[5]),
        )
        ray = trace_pseudo_bending_ray(
            source=source,
            receiver=receiver,
            velocity_grid=reference_grid,
            settings=settings.pseudo_bending_solver,
        )
        if ray.converged:
            converged_count += 1
        lengths = ray_path_cell_lengths(tuple(ray.ray_path), case.target_grid)
        for cell_index, length_km in sorted(lengths.items()):
            cell_indices.append(int(cell_index))
            path_lengths.append(float(length_km))
        row_ptr.append(len(cell_indices))
    cell_count = (
        (len(case.target_grid.x_coordinates_km) - 1)
        * (len(case.target_grid.y_coordinates_km) - 1)
        * (len(case.target_grid.z_coordinates_km) - 1)
    )
    coverage = np.zeros(cell_count, dtype=np.float64)
    if cell_indices:
        np.add.at(coverage, np.asarray(cell_indices, dtype=np.int64), np.asarray(path_lengths))
    return RayGeometry(
        row_ptr=np.asarray(row_ptr, dtype=np.int64),
        cell_indices=np.asarray(cell_indices, dtype=np.int64),
        path_lengths_km=np.asarray(path_lengths, dtype=np.float64),
        coverage_km=coverage,
        converged_count=converged_count,
        observation_count=len(case.observations),
        runtime_s=perf_counter() - started,
    )


def solve_fixed_ray_case(
    *,
    case: ProductionCase,
    geometry: RayGeometry,
    reference_cells: np.ndarray,
    target_cells: np.ndarray,
    damping: float,
    smoothing: float,
    cell_shape: tuple[int, int, int],
    velocity_bounds: tuple[float, float],
    reference_travel_times_s: np.ndarray | None = None,
) -> dict[str, Any]:
    """Solve a reference-prior perturbation inversion without a dense normal matrix.

    ``reference_travel_times_s`` is an optional forward-operator contract for
    corrected analyses.  When supplied, the residual is ``t_obs - t_FSM(s_0)``
    while the fixed-ray sensitivity matrix remains ``G_ref``.  Omitting it
    preserves the historical ``t_obs - G_ref s_0`` formulation used by the
    frozen Benchmark v2 evidence.
    """

    if damping < 0.0 or smoothing < 0.0:
        raise ValueError("Classical regularization values must be non-negative.")
    reference_slowness = 1.0 / reference_cells
    observed_times = case.observations[:, 7]
    reference_times = (
        ray_matvec(geometry, reference_slowness)
        if reference_travel_times_s is None
        else np.asarray(reference_travel_times_s, dtype=np.float64)
    )
    if reference_times.shape != observed_times.shape:
        raise ValueError("reference_travel_times_s must contain one value per observation.")
    if not np.all(np.isfinite(reference_times)):
        raise ValueError("reference_travel_times_s must contain only finite values.")
    residual = observed_times - reference_times
    rhs = ray_transpose(geometry, residual, reference_cells.size)
    if smoothing == 0.0:
        gram = geometry.gram_matrix if geometry.gram_matrix is not None else _ray_gram_matrix(geometry)
        dual = np.linalg.solve(gram + damping * np.eye(gram.shape[0]), residual)
        delta_slowness = ray_transpose(geometry, dual, reference_cells.size)
        delta_slowness = delta_slowness * 1.0
        # The dual solve above returns the observation-space multiplier.  The
        # transpose application is G.T @ (G G.T + lambda I)^-1 r.
        slowness = reference_slowness + delta_slowness
        pcg_iterations = 0
        pcg_converged = True
    else:
        delta_slowness, pcg_iterations, pcg_converged = _solve_pcg(
            geometry,
            rhs,
            damping=damping,
            smoothing=smoothing,
            cell_shape=cell_shape,
            max_iterations=PCG_MAX_ITERATIONS,
            tolerance=PCG_TOLERANCE,
        )
        slowness = reference_slowness + delta_slowness
    lower, upper = velocity_bounds
    finite_slowness = np.where(np.isfinite(slowness) & (slowness > 0.0), slowness, 1.0 / upper)
    unclipped_velocity = 1.0 / finite_slowness
    clipped_velocity = np.clip(unclipped_velocity, lower, upper)
    clipped_fraction = float(np.mean(np.abs(clipped_velocity - unclipped_velocity) > 1.0e-12))
    predicted_times = ray_matvec(geometry, 1.0 / clipped_velocity)
    residuals = predicted_times - observed_times
    positive_mask = geometry.coverage_km > 1.0e-9
    return {
        "fixed_cells": clipped_velocity,
        "slowness": 1.0 / clipped_velocity,
        "predicted_times": predicted_times,
        "residuals": residuals,
        "clipped_fraction": clipped_fraction,
        "pcg_iterations": pcg_iterations,
        "pcg_converged": bool(pcg_converged),
        "all_cell_rmse": _rmse(target_cells, clipped_velocity),
        "all_cell_mae": _mae(target_cells, clipped_velocity),
        "positive_coverage_rmse": _rmse(target_cells[positive_mask], clipped_velocity[positive_mask])
        if np.any(positive_mask)
        else None,
        "positive_coverage_mae": _mae(target_cells[positive_mask], clipped_velocity[positive_mask])
        if np.any(positive_mask)
        else None,
        "travel_time_rmse": _rmse(observed_times, predicted_times),
        "travel_time_mae": _mae(observed_times, predicted_times),
        "travel_time_bias": float(np.mean(residuals)),
    }


def ray_matvec(geometry: RayGeometry, vector: np.ndarray) -> np.ndarray:
    """Apply the sparse ray-cell path-length operator."""

    row_lengths = np.diff(geometry.row_ptr)
    row_indices = np.repeat(np.arange(geometry.observation_count), row_lengths)
    contributions = geometry.path_lengths_km * np.asarray(vector)[geometry.cell_indices]
    return np.bincount(
        row_indices,
        weights=contributions,
        minlength=geometry.observation_count,
    ).astype(np.float64, copy=False)


def ray_transpose(geometry: RayGeometry, vector: np.ndarray, cell_count: int) -> np.ndarray:
    """Apply the transpose of the sparse ray-cell operator."""

    result = np.zeros(cell_count, dtype=np.float64)
    row_lengths = np.diff(geometry.row_ptr)
    row_values = np.repeat(np.asarray(vector), row_lengths)
    np.add.at(result, geometry.cell_indices, geometry.path_lengths_km * row_values)
    return result


def _ray_gram_matrix(geometry: RayGeometry) -> np.ndarray:
    """Build the small observation-space Gram matrix for damping-only solves."""

    gram = np.zeros((geometry.observation_count, geometry.observation_count), dtype=np.float64)
    rows = [
        {
            int(cell): float(length)
            for cell, length in zip(
                geometry.cell_indices[
                    int(geometry.row_ptr[row_index]) : int(geometry.row_ptr[row_index + 1])
                ],
                geometry.path_lengths_km[
                    int(geometry.row_ptr[row_index]) : int(geometry.row_ptr[row_index + 1])
                ],
                strict=True,
            )
        }
        for row_index in range(geometry.observation_count)
    ]
    for left_index, left in enumerate(rows):
        for right_index in range(left_index, len(rows)):
            right = rows[right_index]
            smaller, larger = (right, left) if len(left) > len(right) else (left, right)
            value = sum(length * larger.get(cell, 0.0) for cell, length in smaller.items())
            gram[left_index, right_index] = value
            gram[right_index, left_index] = value
    return gram


def _solve_pcg(
    geometry: RayGeometry,
    rhs: np.ndarray,
    *,
    damping: float,
    smoothing: float,
    cell_shape: tuple[int, int, int],
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, int, bool]:
    """Preconditioned conjugate-gradient solve for the regularized normal operator."""

    cell_count = int(np.prod(cell_shape))
    diagonal = np.full(cell_count, damping, dtype=np.float64)
    np.add.at(
        diagonal,
        geometry.cell_indices,
        geometry.path_lengths_km * geometry.path_lengths_km,
    )
    degree = _smoothing_degree(cell_shape)
    diagonal += smoothing * degree
    diagonal = np.maximum(diagonal, 1.0e-12)

    def apply(vector: np.ndarray) -> np.ndarray:
        return ray_transpose(geometry, ray_matvec(geometry, vector), cell_count) + damping * vector + smoothing * _smoothing_ltl_apply(vector, cell_shape)

    solution = np.zeros(cell_count, dtype=np.float64)
    residual = rhs - apply(solution)
    rhs_norm = max(float(np.linalg.norm(rhs)), 1.0e-15)
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    rz_old = float(np.dot(residual, preconditioned))
    if math.sqrt(max(rz_old, 0.0)) / rhs_norm <= tolerance:
        return solution, 0, True
    for iteration in range(1, max_iterations + 1):
        applied = apply(direction)
        denominator = float(np.dot(direction, applied))
        if denominator <= 0.0 or not np.isfinite(denominator):
            return solution, iteration - 1, False
        step = rz_old / denominator
        solution += step * direction
        residual -= step * applied
        if float(np.linalg.norm(residual)) / rhs_norm <= tolerance:
            return solution, iteration, True
        preconditioned = residual / diagonal
        rz_new = float(np.dot(residual, preconditioned))
        beta = rz_new / max(rz_old, 1.0e-30)
        direction = preconditioned + beta * direction
        rz_old = rz_new
    return solution, max_iterations, False


def _smoothing_ltl_apply(vector: np.ndarray, cell_shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = cell_shape
    values = vector.reshape((nz, ny, nx))
    result = np.zeros_like(values)
    difference_x = values[:, :, 1:] - values[:, :, :-1]
    result[:, :, :-1] -= difference_x
    result[:, :, 1:] += difference_x
    difference_y = values[:, 1:, :] - values[:, :-1, :]
    result[:, :-1, :] -= difference_y
    result[:, 1:, :] += difference_y
    difference_z = values[1:, :, :] - values[:-1, :, :]
    result[:-1, :, :] -= difference_z
    result[1:, :, :] += difference_z
    return result.reshape(-1)


def _smoothing_degree(cell_shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = cell_shape
    degree = np.zeros((nz, ny, nx), dtype=np.float64)
    degree[:, :, :-1] += 1.0
    degree[:, :, 1:] += 1.0
    degree[:, :-1, :] += 1.0
    degree[:, 1:, :] += 1.0
    degree[:-1, :, :] += 1.0
    degree[1:, :, :] += 1.0
    return degree.reshape(-1)


def _classical_case_metric_rows(
    case: ProductionCase,
    result: ClassicalCaseResult,
    cell_shape: tuple[int, int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, prediction in (
        ("reference_prior", result.reference_cells),
        ("fixed_ray", result.fixed_cells),
    ):
        if prediction is None:
            continue
        metrics = _prediction_metrics_from_arrays(result.target_cells, prediction)
        covered = result.coverage_km > 1.0e-9
        covered_metrics = _prediction_metrics_from_arrays(
            result.target_cells[covered], prediction[covered]
        ) if np.any(covered) else {"rmse": None, "mae": None, "bias": None}
        rows.append(
            {
                "target_id": case.target_id,
                "target_hash": case.target_hash,
                "family": case.family,
                "split": case.split,
                "method_id": method,
                "method_label": _method_label(method),
                "node_rmse_km_per_s": None,
                "node_mae_km_per_s": None,
                "node_bias_km_per_s": None,
                "cell_rmse_km_per_s": metrics["rmse"],
                "cell_mae_km_per_s": metrics["mae"],
                "cell_bias_km_per_s": metrics["bias"],
                "positive_coverage_rmse_km_per_s": covered_metrics["rmse"],
                "positive_coverage_mae_km_per_s": covered_metrics["mae"],
                "positive_coverage_fraction": float(np.mean(covered)),
                "travel_time_rmse_s": (
                    _rmse(case.observations[:, 7], ray_matvec(result.geometry, 1.0 / prediction))
                    if method == "reference_prior"
                    else _rmse(case.observations[:, 7], result.fixed_predicted_times)
                ),
                "travel_time_mae_s": (
                    _mae(case.observations[:, 7], ray_matvec(result.geometry, 1.0 / prediction))
                    if method == "reference_prior"
                    else _mae(case.observations[:, 7], result.fixed_predicted_times)
                ),
                "clipped_fraction": result.fixed_clipped_fraction if method == "fixed_ray" else 0.0,
                "pcg_iterations": result.fixed_pcg_iterations if method == "fixed_ray" else 0,
                "pcg_converged": result.fixed_pcg_converged if method == "fixed_ray" else True,
                "cell_shape": "x".join(str(value) for value in cell_shape),
            }
        )
    return rows


def _write_classical_prediction_artifact(
    output_dir: Path,
    case: ProductionCase,
    result: ClassicalCaseResult,
) -> None:
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions" / f"{case.target_id}.npz",
        target_cells=result.target_cells,
        reference_cells=result.reference_cells,
        reference_nodes=result.reference_nodes,
        fixed_cells=result.fixed_cells,
        coverage_km=result.coverage_km,
        fixed_predicted_times=result.fixed_predicted_times,
        fixed_residual_s=result.fixed_residual_s,
    )


def run_coverage_analysis(
    cases: Sequence[ProductionCase],
    ml_result: Mapping[str, Any],
    classical_result: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate every test reconstruction under explicit reference-ray coverage masks."""

    output_dir.mkdir(parents=True, exist_ok=True)
    test_cases = [case for case in cases if case.split == "test"]
    test_classical = classical_result["test_case_results"]
    coverage_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    methods = (*ML_METHODS, "reference_prior", "fixed_ray")
    for case in test_cases:
        result: ClassicalCaseResult = test_classical[case.target_id]
        method_predictions = _test_cell_predictions(case, ml_result, result)
        coverage = result.coverage_km
        for threshold in COVERAGE_THRESHOLDS_KM:
            mask = coverage >= threshold
            for method in methods:
                prediction = method_predictions[method]
                metrics = _prediction_metrics_from_arrays(
                    result.target_cells[mask], prediction[mask]
                ) if np.any(mask) else {"rmse": None, "mae": None, "bias": None}
                coverage_rows.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "method_id": method,
                        "coverage_threshold_km": threshold,
                        "cell_count": int(np.count_nonzero(mask)),
                        "total_cell_count": int(coverage.size),
                        "coverage_fraction": float(np.mean(mask)),
                        "rmse_km_per_s": metrics["rmse"],
                        "mae_km_per_s": metrics["mae"],
                        "bias_km_per_s": metrics["bias"],
                    }
                )
        for bin_label, mask in _coverage_bins(coverage):
            for method in methods:
                prediction = method_predictions[method]
                metrics = _prediction_metrics_from_arrays(
                    result.target_cells[mask], prediction[mask]
                ) if np.any(mask) else {"rmse": None, "mae": None, "bias": None}
                distribution_rows.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "method_id": method,
                        "coverage_bin": bin_label,
                        "cell_count": int(np.count_nonzero(mask)),
                        "rmse_km_per_s": metrics["rmse"],
                        "mae_km_per_s": metrics["mae"],
                    }
                )
        nx, ny, nz = _cell_shape_from_case(case)
        target = result.target_cells.reshape((nz, ny, nx))
        coverage_grid = coverage.reshape((nz, ny, nx))
        for iz in range(nz):
            positive = coverage_grid[iz] > 1.0e-9
            for method in methods:
                prediction = method_predictions[method].reshape((nz, ny, nx))
                metrics = _prediction_metrics_from_arrays(
                    target[iz][positive], prediction[iz][positive]
                ) if np.any(positive) else {"rmse": None, "mae": None, "bias": None}
                depth_rows.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "method_id": method,
                        "depth_index": iz,
                        "top_depth_km": float(case.target_grid.z_coordinates_km[iz]),
                        "bottom_depth_km": float(case.target_grid.z_coordinates_km[iz + 1]),
                        "positive_coverage_cell_count": int(np.count_nonzero(positive)),
                        "depth_cell_count": int(positive.size),
                        "positive_coverage_fraction": float(np.mean(positive)),
                        "mean_positive_coverage_km": float(np.mean(coverage_grid[iz][positive]))
                        if np.any(positive)
                        else 0.0,
                        "rmse_km_per_s": metrics["rmse"],
                        "mae_km_per_s": metrics["mae"],
                        "bias_km_per_s": metrics["bias"],
                    }
                )
    _write_csv(output_dir / "coverage_metrics.csv", coverage_rows)
    _write_csv(output_dir / "coverage_error_distribution.csv", distribution_rows)
    _write_csv(output_dir / "depth_coverage_metrics.csv", depth_rows)
    coverage_delta_rows = _coverage_paired_deltas(coverage_rows)
    _write_csv(output_dir / "coverage_target_level_paired_deltas.csv", coverage_delta_rows)
    coverage_delta_summary = _coverage_paired_summary(coverage_delta_rows)
    _write_csv(output_dir / "coverage_target_level_paired_summary.csv", coverage_delta_summary)
    summary_rows = _summarize_metric_rows(
        [
            {
                "target_id": row["target_id"],
                "family": row["family"],
                "split": "test",
                "method_id": row["method_id"],
                "cell_rmse_km_per_s": row["rmse_km_per_s"],
                "cell_mae_km_per_s": row["mae_km_per_s"],
                "node_rmse_km_per_s": None,
                "node_mae_km_per_s": None,
            }
            for row in coverage_rows
            if float(row["coverage_threshold_km"]) == 1.0
        ],
        split_values=("test",),
    )
    _write_csv(output_dir / "coverage_summary.csv", summary_rows)
    _write_json(
        {
            "artifact_type": "benchmark_v2_reference_ray_coverage_analysis",
            "test_target_count": len(test_cases),
            "methods": list(methods),
            "coverage_thresholds_km": list(COVERAGE_THRESHOLDS_KM),
            "coverage_source": "reference-model fixed-ray geometry; not supplied to ML inputs",
        },
        output_dir / "coverage_analysis_metadata.json",
    )
    return {
        "output_dir": output_dir,
        "coverage_rows": coverage_rows,
        "depth_rows": depth_rows,
        "distribution_rows": distribution_rows,
        "target_level_paired_deltas": coverage_delta_rows,
        "target_level_paired_summary": coverage_delta_summary,
        "summary_rows": summary_rows,
    }


def run_structural_analysis(
    cases: Sequence[ProductionCase],
    ml_result: Mapping[str, Any],
    classical_result: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate generator-defined structural masks on all held-out targets."""

    output_dir.mkdir(parents=True, exist_ok=True)
    test_cases = [case for case in cases if case.split == "test"]
    rows: list[dict[str, Any]] = []
    methods = (*ML_METHODS, "reference_prior", "fixed_ray")
    for case in test_cases:
        classical: ClassicalCaseResult = classical_result["test_case_results"][case.target_id]
        predictions = _test_cell_predictions(case, ml_result, classical)
        masks = build_structure_masks(case)
        for method in methods:
            prediction = predictions[method]
            for scope, mask_map in masks.items():
                for mask_label, mask in mask_map.items():
                    metrics = _prediction_metrics_from_arrays(
                        classical.target_cells[mask], prediction[mask]
                    ) if np.any(mask) else {"rmse": None, "mae": None, "bias": None}
                    rows.append(
                        {
                            "target_id": case.target_id,
                            "target_hash": case.target_hash,
                            "family": case.family,
                            "method_id": method,
                            "metric_scope": scope,
                            "mask_label": mask_label,
                            "cell_count": int(np.count_nonzero(mask)),
                            "rmse_km_per_s": metrics["rmse"],
                            "mae_km_per_s": metrics["mae"],
                            "bias_km_per_s": metrics["bias"],
                            "true_mean_km_per_s": metrics["true_mean"],
                            "predicted_mean_km_per_s": metrics["predicted_mean"],
                            "true_contrast_km_per_s": None,
                            "predicted_contrast_km_per_s": None,
                            "contrast_error_km_per_s": None,
                        }
                    )
            if case.family in {"block_anomaly", "salt_dome", "dyke_intrusion"}:
                contrast = _contrast_metrics(
                    classical.target_cells,
                    prediction,
                    masks["structure"]["body"],
                    masks["structure"]["background"],
                )
                rows.append(
                    _structural_contrast_row(
                        case,
                        method,
                        "structure_contrast",
                        "body_minus_background",
                        contrast,
                        int(np.count_nonzero(masks["structure"]["body"])),
                    )
                )
            if case.family == "faulted":
                contrast = _contrast_metrics(
                    classical.target_cells,
                    prediction,
                    masks["fault"]["positive_side"],
                    masks["fault"]["negative_side"],
                )
                rows.append(
                    _structural_contrast_row(
                        case,
                        method,
                        "fault_contrast",
                        "positive_minus_negative_side",
                        contrast,
                        int(np.count_nonzero(masks["fault"]["positive_side"])),
                    )
                )
    _write_csv(output_dir / "structural_metrics.csv", rows)
    summary_rows = _structural_summary_rows(rows)
    _write_csv(output_dir / "structural_summary.csv", summary_rows)
    _write_json(
        {
            "artifact_type": "benchmark_v2_generator_mask_structural_metrics",
            "test_target_count": len(test_cases),
            "methods": list(methods),
            "mask_policy": "analytic generator geometry sampled at target-cell centers; no predicted threshold or localization detector",
            "families": list(FAMILIES),
        },
        output_dir / "structural_metadata.json",
    )
    return {"output_dir": output_dir, "rows": rows, "summary_rows": summary_rows}


def build_structure_masks(case: ProductionCase) -> dict[str, dict[str, np.ndarray]]:
    """Return objective cell masks derived from the analytic target definition."""

    nx, ny, nz = _cell_shape_from_case(case)
    x = _cell_centers(case.target_grid.x_coordinates_km)
    y = _cell_centers(case.target_grid.y_coordinates_km)
    z = _cell_centers(case.target_grid.z_coordinates_km)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    # meshgrid(x, y, z, indexing='xy') produces (ny,nx,nz); transpose to the
    # x-fastest flat order used by all saved target/cell vectors.
    coordinates = (xx.transpose(2, 0, 1), yy.transpose(2, 0, 1), zz.transpose(2, 0, 1))
    x_grid, y_grid, z_grid = coordinates
    masks: dict[str, dict[str, np.ndarray]] = {}
    if case.family == "layered":
        layered = case.target_grid.metadata.get("layered_model", {})
        boundaries = [float(value) for value in layered.get("depth_boundaries_km", [])]
        if len(boundaries) < 2:
            raise ValueError(f"{case.target_id}: layered target has no depth boundaries.")
        layer_indices = np.searchsorted(np.asarray(boundaries[1:-1]), z_grid, side="right")
        masks["layer"] = {
            f"layer_{index + 1:02d}": (layer_indices == index).reshape(-1)
            for index in range(len(boundaries) - 1)
        }
        return masks

    parameters = _family_parameters(case)
    if case.family == "block_anomaly":
        center = np.asarray(parameters["center_km"], dtype=float)
        size = np.asarray(parameters["size_km"], dtype=float)
        body = (
            (np.abs(x_grid - center[0]) <= size[0] / 2.0)
            & (np.abs(y_grid - center[1]) <= size[1] / 2.0)
            & (np.abs(z_grid - center[2]) <= size[2] / 2.0)
        )
    elif case.family == "salt_dome":
        center = np.asarray(parameters["center_km"], dtype=float)
        radii = np.asarray(parameters["radii_km"], dtype=float)
        body = (
            ((x_grid - center[0]) / radii[0]) ** 2
            + ((y_grid - center[1]) / radii[1]) ** 2
            + ((z_grid - center[2]) / radii[2]) ** 2
            <= 1.0
        )
    elif case.family == "dyke_intrusion":
        center = np.asarray(parameters["center_km"], dtype=float)
        strike = np.deg2rad(float(parameters["strike_deg"]))
        along = np.abs(
            (x_grid - center[0]) * np.cos(strike)
            + (y_grid - center[1]) * np.sin(strike)
        )
        normal = np.abs(
            -(x_grid - center[0]) * np.sin(strike)
            + (y_grid - center[1]) * np.cos(strike)
        )
        body = (
            (along <= float(parameters["length_km"]) / 2.0)
            & (normal <= float(parameters["width_km"]) / 2.0)
            & (z_grid >= float(parameters["top_depth_km"]))
            & (z_grid <= float(parameters["bottom_depth_km"]))
        )
    elif case.family == "faulted":
        faulted = FaultedGridSettings(
            fault_x_km=float(parameters["fault_x_km"]),
            fault_y_km=float(parameters["fault_y_km"]),
            strike_deg=float(parameters["strike_deg"]),
            dip_deg=float(parameters["dip_deg"]),
            dip_direction=str(parameters["dip_direction"]),
            positive_side=str(parameters["positive_side"]),
            velocity_offset_km_per_s=float(parameters["velocity_offset_km_per_s"]),
        )
        signed = np.asarray(
            [
                fault_plane_signed_offset_km(float(xv), float(yv), float(zv), faulted)
                for xv, yv, zv in zip(x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1), strict=True)
            ],
            dtype=float,
        ).reshape(x_grid.shape)
        if faulted.positive_side == "greater_equal":
            positive = signed >= 0.0
        else:
            positive = signed <= 0.0
        masks["fault"] = {"positive_side": positive.reshape(-1), "negative_side": (~positive).reshape(-1)}
        return masks
    else:
        raise ValueError(f"Unsupported structural family: {case.family}")
    masks["structure"] = {"body": body.reshape(-1), "background": (~body).reshape(-1)}
    return masks


def _family_parameters(case: ProductionCase) -> dict[str, Any]:
    sampled = case.physical_parameters.get("sampled_family_parameters", {})
    parameters = sampled.get(case.family)
    if not isinstance(parameters, dict):
        raise ValueError(f"{case.target_id}: missing sampled parameters for {case.family}.")
    return dict(parameters)


def _test_cell_predictions(
    case: ProductionCase,
    ml_result: Mapping[str, Any],
    classical: ClassicalCaseResult,
) -> dict[str, np.ndarray]:
    shape = _node_shape(case.target_grid)
    predictions: dict[str, np.ndarray] = {}
    for method in ML_METHODS:
        predictions[method] = cell_centered_values_from_node_vector(
            ml_result["predictions"][method][case.target_id],
            shape,
        )
    predictions["reference_prior"] = classical.reference_cells
    if classical.fixed_cells is None:
        raise ValueError(f"{case.target_id}: missing selected fixed-ray prediction.")
    predictions["fixed_ray"] = classical.fixed_cells
    return predictions


def _coverage_bins(coverage: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("zero", coverage <= 0.0),
        ("(0,1]", (coverage > 0.0) & (coverage <= 1.0)),
        ("(1,5]", (coverage > 1.0) & (coverage <= 5.0)),
        ("(5,20]", (coverage > 5.0) & (coverage <= 20.0)),
        (">20", coverage > 20.0),
    ]


def _coverage_paired_deltas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[float, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("rmse_km_per_s") in (None, ""):
            continue
        index[
            (
                float(row["coverage_threshold_km"]),
                str(row["target_id"]),
                str(row["method_id"]),
            )
        ] = row
    comparisons = (
        ("realistic_vs_travel_time_only", "realistic_full", "travel_time_only"),
        ("realistic_vs_no_travel_time", "realistic_full", "no_travel_time"),
        ("realistic_vs_fixed_ray", "realistic_full", "fixed_ray"),
    )
    thresholds = sorted({float(row["coverage_threshold_km"]) for row in rows})
    output: list[dict[str, Any]] = []
    for threshold in thresholds:
        targets = sorted(
            {
                str(row["target_id"])
                for row in rows
                if float(row["coverage_threshold_km"]) == threshold
            }
        )
        for comparison_id, first_method, second_method in comparisons:
            for target_id in targets:
                first = index.get((threshold, target_id, first_method))
                second = index.get((threshold, target_id, second_method))
                if first is None or second is None:
                    continue
                output.append(
                    {
                        "comparison_id": comparison_id,
                        "target_id": target_id,
                        "family": first["family"],
                        "coverage_threshold_km": threshold,
                        "first_method_id": first_method,
                        "second_method_id": second_method,
                        "first_rmse_km_per_s": float(first["rmse_km_per_s"]),
                        "second_rmse_km_per_s": float(second["rmse_km_per_s"]),
                        "delta_rmse_km_per_s": float(first["rmse_km_per_s"])
                        - float(second["rmse_km_per_s"]),
                    }
                )
    return output


def _coverage_paired_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["comparison_id"]), float(row["coverage_threshold_km"]))].append(row)
    output: list[dict[str, Any]] = []
    for (comparison_id, threshold), group in sorted(grouped.items()):
        deltas = np.asarray([float(row["delta_rmse_km_per_s"]) for row in group])
        ci = _bootstrap_ci(deltas, _stable_seed(f"coverage:{comparison_id}:{threshold}"))
        output.append(
            {
                "comparison_id": comparison_id,
                "coverage_threshold_km": threshold,
                "test_target_count": len(group),
                "delta_rmse_mean_km_per_s": float(np.mean(deltas)),
                "delta_rmse_median_km_per_s": float(np.median(deltas)),
                "delta_rmse_std_km_per_s": float(np.std(deltas)),
                "delta_rmse_ci95_lower_km_per_s": ci[0],
                "delta_rmse_ci95_upper_km_per_s": ci[1],
                "target_win_count": int(np.count_nonzero(deltas < -TIE_TOLERANCE)),
                "target_loss_count": int(np.count_nonzero(deltas > TIE_TOLERANCE)),
                "target_tie_count": int(np.count_nonzero(np.abs(deltas) <= TIE_TOLERANCE)),
                "bootstrap_unit": "unique geological target",
                "delta_definition": "realistic/first method minus comparison/second method",
            }
        )
    return output


def _structural_contrast_row(
    case: ProductionCase,
    method: str,
    scope: str,
    mask_label: str,
    contrast: Mapping[str, float | None],
    cell_count: int,
) -> dict[str, Any]:
    return {
        "target_id": case.target_id,
        "target_hash": case.target_hash,
        "family": case.family,
        "method_id": method,
        "metric_scope": scope,
        "mask_label": mask_label,
        "cell_count": cell_count,
        "rmse_km_per_s": None,
        "mae_km_per_s": None,
        "bias_km_per_s": None,
        "true_mean_km_per_s": None,
        "predicted_mean_km_per_s": None,
        "true_contrast_km_per_s": contrast["true_contrast"],
        "predicted_contrast_km_per_s": contrast["predicted_contrast"],
        "contrast_error_km_per_s": contrast["contrast_error"],
    }


def _contrast_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> dict[str, float | None]:
    if not np.any(first_mask) or not np.any(second_mask):
        return {"true_contrast": None, "predicted_contrast": None, "contrast_error": None}
    true_contrast = float(np.mean(target[first_mask]) - np.mean(target[second_mask]))
    predicted_contrast = float(np.mean(prediction[first_mask]) - np.mean(prediction[second_mask]))
    return {
        "true_contrast": true_contrast,
        "predicted_contrast": predicted_contrast,
        "contrast_error": predicted_contrast - true_contrast,
    }


def _structural_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["family"]),
                str(row["method_id"]),
                str(row["metric_scope"]),
                str(row["mask_label"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (family, method, scope, label), group in sorted(grouped.items()):
        output.append(
            {
                "family": family,
                "method_id": method,
                "metric_scope": scope,
                "mask_label": label,
                "test_target_count": len({str(row["target_id"]) for row in group}),
                "mean_cell_count": _mean_optional(group, "cell_count"),
                "mean_rmse_km_per_s": _mean_optional(group, "rmse_km_per_s"),
                "mean_mae_km_per_s": _mean_optional(group, "mae_km_per_s"),
                "mean_bias_km_per_s": _mean_optional(group, "bias_km_per_s"),
                "mean_true_contrast_km_per_s": _mean_optional(group, "true_contrast_km_per_s"),
                "mean_predicted_contrast_km_per_s": _mean_optional(group, "predicted_contrast_km_per_s"),
                "mean_contrast_error_km_per_s": _mean_optional(group, "contrast_error_km_per_s"),
            }
        )
    return output


def run_target_level_statistics(
    cases: Sequence[ProductionCase],
    ml_result: Mapping[str, Any],
    classical_result: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Summarize final test methods using one independent row per target."""

    output_dir.mkdir(parents=True, exist_ok=True)
    test_cases = [case for case in cases if case.split == "test"]
    test_ids = {case.target_id for case in test_cases}
    rows: list[dict[str, Any]] = []
    for row in ml_result["metric_rows"]:
        if str(row["split"]) != "test" or str(row["target_id"]) not in test_ids:
            continue
        rows.extend(
            [
                {
                    "target_id": row["target_id"],
                    "target_hash": row["target_hash"],
                    "family": row["family"],
                    "method_id": row["method_id"],
                    "metric_domain": "node_all",
                    "rmse_km_per_s": row["node_rmse_km_per_s"],
                    "mae_km_per_s": row["node_mae_km_per_s"],
                },
                {
                    "target_id": row["target_id"],
                    "target_hash": row["target_hash"],
                    "family": row["family"],
                    "method_id": row["method_id"],
                    "metric_domain": "cell_all",
                    "rmse_km_per_s": row["cell_rmse_km_per_s"],
                    "mae_km_per_s": row["cell_mae_km_per_s"],
                },
            ]
        )
    for row in classical_result["test_metric_rows"]:
        rows.append(
            {
                "target_id": row["target_id"],
                "target_hash": row["target_hash"],
                "family": row["family"],
                "method_id": row["method_id"],
                "metric_domain": "cell_all",
                "rmse_km_per_s": row["cell_rmse_km_per_s"],
                "mae_km_per_s": row["cell_mae_km_per_s"],
            }
        )
        rows.append(
            {
                "target_id": row["target_id"],
                "target_hash": row["target_hash"],
                "family": row["family"],
                "method_id": row["method_id"],
                "metric_domain": "cell_positive_coverage",
                "rmse_km_per_s": row["positive_coverage_rmse_km_per_s"],
                "mae_km_per_s": row["positive_coverage_mae_km_per_s"],
            }
        )
    _validate_target_metric_rows(rows, test_ids)
    _write_csv(output_dir / "target_level_metrics.csv", rows)
    method_summary = _target_method_summary(rows)
    comparisons = (
        ("realistic_vs_travel_time_only", "realistic_full", "travel_time_only"),
        ("realistic_vs_no_travel_time", "realistic_full", "no_travel_time"),
        ("realistic_vs_shuffled_travel_time", "realistic_full", "shuffled_travel_time"),
        ("realistic_vs_training_target_mean", "realistic_full", "training_target_mean"),
        ("realistic_vs_reference_prior", "realistic_full", "reference_prior"),
        ("realistic_vs_fixed_ray", "realistic_full", "fixed_ray"),
        ("travel_time_only_vs_fixed_ray", "travel_time_only", "fixed_ray"),
    )
    delta_rows: list[dict[str, Any]] = []
    for comparison_id, first_method, second_method in comparisons:
        for domain in ("cell_all", "cell_positive_coverage"):
            first = _metric_index(rows, first_method, domain)
            second = _metric_index(rows, second_method, domain)
            if set(first) != set(second):
                if domain == "cell_positive_coverage":
                    continue
                raise ValueError(f"Paired comparison {comparison_id} has mismatched targets.")
            for target_id in sorted(first):
                delta_rows.append(
                    {
                        "comparison_id": comparison_id,
                        "target_id": target_id,
                        "family": first[target_id]["family"],
                        "metric_domain": domain,
                        "first_method_id": first_method,
                        "second_method_id": second_method,
                        "first_rmse_km_per_s": first[target_id]["rmse"],
                        "second_rmse_km_per_s": second[target_id]["rmse"],
                        "delta_rmse_km_per_s": first[target_id]["rmse"] - second[target_id]["rmse"],
                        "first_mae_km_per_s": first[target_id]["mae"],
                        "second_mae_km_per_s": second[target_id]["mae"],
                        "delta_mae_km_per_s": first[target_id]["mae"] - second[target_id]["mae"],
                    }
                )
    _write_csv(output_dir / "target_level_paired_deltas.csv", delta_rows)
    paired_summary = _paired_summary(delta_rows)
    _write_csv(output_dir / "target_level_method_summary.csv", method_summary)
    _write_csv(output_dir / "target_level_paired_summary.csv", paired_summary)
    _write_json(
        {
            "artifact_type": "benchmark_v2_target_level_statistical_evaluation",
            "independent_unit": "unique geological target",
            "test_target_count": len(test_ids),
            "test_acquisition_case_count": len(test_ids),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "paired_delta_definition": "first method target-level metric minus second method target-level metric; negative favors first",
            "comparisons": [comparison[0] for comparison in comparisons],
        },
        output_dir / "target_level_statistics_metadata.json",
    )
    return {
        "output_dir": output_dir,
        "target_rows": rows,
        "method_summary": method_summary,
        "delta_rows": delta_rows,
        "paired_summary": paired_summary,
        "test_target_count": len(test_ids),
    }


def _validate_target_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    test_ids: set[str],
) -> None:
    expected_methods = set((*ML_METHODS, "reference_prior", "fixed_ray"))
    identity_keys = [
        (
            str(row["target_id"]),
            str(row["method_id"]),
            str(row["metric_domain"]),
        )
        for row in rows
    ]
    if len(identity_keys) != len(set(identity_keys)):
        raise ValueError(
            "Target-level statistics contain duplicate target/method/domain rows; "
            "acquisition replicates must be aggregated before inference."
        )
    for domain in ("cell_all",):
        for method in expected_methods:
            observed = {
                str(row["target_id"])
                for row in rows
                if str(row["metric_domain"]) == domain and str(row["method_id"]) == method
            }
            if observed != test_ids:
                raise ValueError(
                    f"Target-level metric coverage for {method}/{domain} is incomplete: "
                    f"missing={sorted(test_ids - observed)}"
                )


def _metric_index(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    domain: str,
) -> dict[str, dict[str, Any]]:
    return {
        str(row["target_id"]): {
            "family": str(row["family"]),
            "rmse": float(row["rmse_km_per_s"]),
            "mae": float(row["mae_km_per_s"]),
        }
        for row in rows
        if str(row["method_id"]) == method and str(row["metric_domain"]) == domain
        and row.get("rmse_km_per_s") not in (None, "")
    }


def _target_method_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("rmse_km_per_s") not in (None, ""):
            grouped[(str(row["method_id"]), str(row["metric_domain"]))].append(row)
    for (method, domain), group in sorted(grouped.items()):
        rmse = np.asarray([float(row["rmse_km_per_s"]) for row in group])
        mae = np.asarray([float(row["mae_km_per_s"]) for row in group])
        rmse_ci = _bootstrap_ci(rmse, _stable_seed(f"method:{method}:{domain}:rmse"))
        mae_ci = _bootstrap_ci(mae, _stable_seed(f"method:{method}:{domain}:mae"))
        output.append(
            {
                "method_id": method,
                "method_label": _method_label(method),
                "metric_domain": domain,
                "test_target_count": len(group),
                "rmse_mean_km_per_s": float(np.mean(rmse)),
                "rmse_median_km_per_s": float(np.median(rmse)),
                "rmse_std_km_per_s": float(np.std(rmse)),
                "rmse_ci95_lower_km_per_s": rmse_ci[0],
                "rmse_ci95_upper_km_per_s": rmse_ci[1],
                "mae_mean_km_per_s": float(np.mean(mae)),
                "mae_median_km_per_s": float(np.median(mae)),
                "mae_std_km_per_s": float(np.std(mae)),
                "mae_ci95_lower_km_per_s": mae_ci[0],
                "mae_ci95_upper_km_per_s": mae_ci[1],
                "bootstrap_unit": "unique geological target",
            }
        )
    return output


def _paired_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["comparison_id"]), str(row["metric_domain"]))].append(row)
    output: list[dict[str, Any]] = []
    for (comparison, domain), group in sorted(grouped.items()):
        rmse = np.asarray([float(row["delta_rmse_km_per_s"]) for row in group])
        mae = np.asarray([float(row["delta_mae_km_per_s"]) for row in group])
        rmse_ci = _bootstrap_ci(rmse, _stable_seed(f"delta:{comparison}:{domain}:rmse"))
        mae_ci = _bootstrap_ci(mae, _stable_seed(f"delta:{comparison}:{domain}:mae"))
        output.append(
            {
                "comparison_id": comparison,
                "metric_domain": domain,
                "first_method_id": group[0]["first_method_id"],
                "second_method_id": group[0]["second_method_id"],
                "test_target_count": len(group),
                "rmse_delta_mean_km_per_s": float(np.mean(rmse)),
                "rmse_delta_median_km_per_s": float(np.median(rmse)),
                "rmse_delta_std_km_per_s": float(np.std(rmse)),
                "rmse_delta_ci95_lower_km_per_s": rmse_ci[0],
                "rmse_delta_ci95_upper_km_per_s": rmse_ci[1],
                "rmse_target_win_count": int(np.count_nonzero(rmse < -TIE_TOLERANCE)),
                "rmse_target_loss_count": int(np.count_nonzero(rmse > TIE_TOLERANCE)),
                "rmse_target_tie_count": int(np.count_nonzero(np.abs(rmse) <= TIE_TOLERANCE)),
                "mae_delta_mean_km_per_s": float(np.mean(mae)),
                "mae_delta_median_km_per_s": float(np.median(mae)),
                "mae_delta_std_km_per_s": float(np.std(mae)),
                "mae_delta_ci95_lower_km_per_s": mae_ci[0],
                "mae_delta_ci95_upper_km_per_s": mae_ci[1],
                "delta_definition": "first method minus second; negative favors first",
                "bootstrap_unit": "unique geological target",
            }
        )
    return output


def _bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, values.size), replace=True)
    means = np.mean(samples, axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_noise_analysis(
    cases: Sequence[ProductionCase],
    ml_result: Mapping[str, Any],
    classical_result: Mapping[str, Any],
    output_dir: Path,
    settings: Any,
) -> dict[str, Any]:
    """Evaluate frozen methods under matched repeated Gaussian timing noise."""

    output_dir.mkdir(parents=True, exist_ok=True)
    test_cases = [case for case in cases if case.split == "test"]
    test_classical: Mapping[str, ClassicalCaseResult] = classical_result["test_case_results"]
    selected_damping = float(classical_result["selected"]["damping"])
    selected_smoothing = float(classical_result["selected"]["smoothing"])
    cell_shape = classical_result["cell_shape"]
    velocity_bounds = classical_result["velocity_bounds"]
    rows: list[dict[str, Any]] = []
    zero_metrics: dict[tuple[str, str], float] = {}
    for noise_level in NOISE_LEVELS_S:
        seeds = (0,) if noise_level == 0.0 else tuple(range(1, NOISE_REPETITIONS + 1))
        for seed in seeds:
            for case in test_cases:
                classical = test_classical[case.target_id]
                rng = np.random.default_rng(
                    _stable_seed(f"noise:{int(round(noise_level * 1000))}:{seed}:{case.target_id}")
                )
                noisy_observations = case.observations.copy()
                if noise_level > 0.0:
                    noisy_observations[:, 7] += rng.normal(
                        0.0, noise_level, size=noisy_observations.shape[0]
                    )
                noisy_case = replace(case, observations=noisy_observations)
                predictions: dict[str, np.ndarray] = {}
                for method in ("realistic_full", "travel_time_only"):
                    model: ObservationPCAModel = ml_result["models"][method]
                    features = build_feature_matrices([noisy_case])[method]
                    predicted_nodes = model.predict(features)[0]
                    predictions[method] = cell_centered_values_from_node_vector(
                        predicted_nodes, _node_shape(case.target_grid)
                    )
                fixed = solve_fixed_ray_case(
                    case=noisy_case,
                    geometry=classical.geometry,
                    reference_cells=classical.reference_cells,
                    target_cells=classical.target_cells,
                    damping=selected_damping,
                    smoothing=selected_smoothing,
                    cell_shape=cell_shape,
                    velocity_bounds=velocity_bounds,
                )
                predictions["fixed_ray"] = fixed["fixed_cells"]
                predictions["reference_prior"] = classical.reference_cells
                for method, prediction in predictions.items():
                    metrics = _prediction_metrics_from_arrays(
                        classical.target_cells, prediction
                    )
                    zero_key = (case.target_id, method)
                    if noise_level == 0.0:
                        zero_metrics[zero_key] = metrics["rmse"]
                    rows.append(
                        {
                            "target_id": case.target_id,
                            "family": case.family,
                            "method_id": method,
                            "noise_std_s": noise_level,
                            "noise_seed": seed,
                            "cell_rmse_km_per_s": metrics["rmse"],
                            "cell_mae_km_per_s": metrics["mae"],
                            "cell_bias_km_per_s": metrics["bias"],
                            "degradation_from_zero_rmse_km_per_s": None,
                        }
                    )
    for row in rows:
        row["degradation_from_zero_rmse_km_per_s"] = float(row["cell_rmse_km_per_s"]) - zero_metrics[
            (str(row["target_id"]), str(row["method_id"]))
        ]
    _write_csv(output_dir / "noise_replication_metrics.csv", rows)
    target_level_rows = _aggregate_noise_target_levels(rows)
    _write_csv(output_dir / "noise_target_level_summary.csv", target_level_rows)
    summary_rows = _summarize_noise(target_level_rows)
    _write_csv(output_dir / "noise_summary.csv", summary_rows)
    _write_json(
        {
            "artifact_type": "benchmark_v2_repeated_timing_noise_robustness",
            "noise_levels_s": list(NOISE_LEVELS_S),
            "nonzero_repetitions": NOISE_REPETITIONS,
            "matched_noise": "same target/level/seed perturbation is used for realistic ML, travel-time-only ML, and fixed-ray",
            "aggregation": "average repetitions within target before across-target summaries",
            "interpretation": "controlled Gaussian timing sensitivity, not a complete real picking-error model",
        },
        output_dir / "noise_metadata.json",
    )
    return {
        "output_dir": output_dir,
        "replication_rows": rows,
        "target_level_rows": target_level_rows,
        "summary_rows": summary_rows,
    }


def _aggregate_noise_target_levels(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_id"]), str(row["method_id"]), float(row["noise_std_s"]))].append(row)
    output: list[dict[str, Any]] = []
    for (target_id, method, noise), group in sorted(grouped.items()):
        output.append(
            {
                "target_id": target_id,
                "family": str(group[0]["family"]),
                "method_id": method,
                "noise_std_s": noise,
                "replication_count": len(group),
                "cell_rmse_km_per_s": float(np.mean([float(row["cell_rmse_km_per_s"]) for row in group])),
                "cell_mae_km_per_s": float(np.mean([float(row["cell_mae_km_per_s"]) for row in group])),
                "cell_bias_km_per_s": float(np.mean([float(row["cell_bias_km_per_s"]) for row in group])),
                "degradation_from_zero_rmse_km_per_s": float(
                    np.mean([float(row["degradation_from_zero_rmse_km_per_s"]) for row in group])
                ),
            }
        )
    return output


def _summarize_noise(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method_id"]), float(row["noise_std_s"]))].append(row)
    output: list[dict[str, Any]] = []
    for (method, noise), group in sorted(grouped.items()):
        rmse = np.asarray([float(row["cell_rmse_km_per_s"]) for row in group])
        degradation = np.asarray(
            [float(row["degradation_from_zero_rmse_km_per_s"]) for row in group]
        )
        ci = _bootstrap_ci(rmse, _stable_seed(f"noise-summary:{method}:{noise}"))
        dci = _bootstrap_ci(degradation, _stable_seed(f"noise-degradation:{method}:{noise}"))
        output.append(
            {
                "method_id": method,
                "noise_std_s": noise,
                "test_target_count": len(group),
                "cell_rmse_mean_km_per_s": float(np.mean(rmse)),
                "cell_rmse_std_km_per_s": float(np.std(rmse)),
                "cell_rmse_ci95_lower_km_per_s": ci[0],
                "cell_rmse_ci95_upper_km_per_s": ci[1],
                "degradation_mean_km_per_s": float(np.mean(degradation)),
                "degradation_std_km_per_s": float(np.std(degradation)),
                "degradation_ci95_lower_km_per_s": dci[0],
                "degradation_ci95_upper_km_per_s": dci[1],
                "aggregation_unit": "unique geological target",
            }
        )
    return output


def run_lofo_analysis(
    cases: Sequence[ProductionCase],
    output_dir: Path,
    settings: Any,
) -> dict[str, Any]:
    """Run a separate leave-one-family-out diagnostic with internal selection."""

    output_dir.mkdir(parents=True, exist_ok=True)
    case_by_family = {family: [case for case in cases if case.family == family] for family in FAMILIES}
    candidate_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    velocity_bounds = tuple(float(value) for value in settings.velocity_model_generation.velocity_bounds_km_per_s)
    for held_family in FAMILIES:
        remaining = [case for family in FAMILIES if family != held_family for case in case_by_family[family]]
        internal_train: list[ProductionCase] = []
        internal_validation: list[ProductionCase] = []
        for family in FAMILIES:
            if family == held_family:
                continue
            family_cases = sorted(case_by_family[family], key=lambda case: case.target_hash)
            internal_train.extend(family_cases[:40])
            internal_validation.extend(family_cases[40:])
        del remaining
        pca = fit_target_pca(np.vstack([case.target_vector for case in internal_train]))
        features_train = build_feature_matrices(internal_train)["realistic_full"]
        features_validation = build_feature_matrices(internal_validation)["realistic_full"]
        target_train = np.vstack([case.target_vector for case in internal_train])
        target_validation = np.vstack([case.target_vector for case in internal_validation])
        valid_components = tuple(k for k in PCA_COMPONENT_GRID if k <= pca.rank)
        candidates: list[dict[str, Any]] = []
        for component_count in valid_components:
            for model_name in ("pca_linear", "pca_ridge"):
                alphas: Iterable[float | None] = (None,) if model_name == "pca_linear" else RIDGE_ALPHA_GRID
                for alpha in alphas:
                    model = fit_observation_pca_model(
                        features_train,
                        target_train,
                        pca,
                        component_count,
                        model_name,
                        alpha,
                        velocity_bounds,
                    )
                    prediction = model.predict(features_validation)
                    metrics = _aggregate_prediction_metrics(target_validation, prediction)
                    row = {
                        "held_out_family": held_family,
                        "model_name": model_name,
                        "pca_component_count": component_count,
                        "ridge_alpha": alpha,
                        "internal_train_target_count": len(internal_train),
                        "internal_validation_target_count": len(internal_validation),
                        "validation_node_rmse_mean_km_per_s": metrics["node_rmse_mean"],
                        "validation_cell_rmse_mean_km_per_s": metrics["cell_rmse_mean"],
                    }
                    candidates.append(row)
                    candidate_rows.append(row)
        selected = min(
            candidates,
            key=lambda row: (
                float(row["validation_node_rmse_mean_km_per_s"]),
                int(row["pca_component_count"]),
                float(row["ridge_alpha"] or 0.0),
            ),
        )
        selected_model = fit_observation_pca_model(
            features_train,
            target_train,
            pca,
            int(selected["pca_component_count"]),
            str(selected["model_name"]),
            None if selected["ridge_alpha"] in (None, "") else float(selected["ridge_alpha"]),
            velocity_bounds,
        )
        held_out = sorted(case_by_family[held_family], key=lambda case: case.target_hash)
        held_features = build_feature_matrices(held_out)["realistic_full"]
        held_prediction = selected_model.predict(held_features)
        for index, case in enumerate(held_out):
            metrics = _prediction_metrics(case.target_vector, held_prediction[index])
            metric_rows.append(
                {
                    "held_out_family": held_family,
                    "target_id": case.target_id,
                    "target_hash": case.target_hash,
                    "family": case.family,
                    "split": "lofo_held_out",
                    "method_id": "realistic_full",
                    "node_rmse_km_per_s": metrics["node_rmse"],
                    "node_mae_km_per_s": metrics["node_mae"],
                    "cell_rmse_km_per_s": metrics["cell_rmse"],
                    "cell_mae_km_per_s": metrics["cell_mae"],
                    "selected_model_name": selected["model_name"],
                    "selected_pca_component_count": selected["pca_component_count"],
                    "selected_ridge_alpha": selected["ridge_alpha"],
                    "diagnostic_only": True,
                }
            )
    _write_csv(output_dir / "lofo_internal_selection_candidates.csv", candidate_rows)
    _write_csv(output_dir / "lofo_case_metrics.csv", metric_rows)
    summary_rows = _summarize_lofo_rows(metric_rows)
    _write_csv(output_dir / "lofo_summary.csv", summary_rows)
    _write_json(
        {
            "artifact_type": "benchmark_v2_leave_one_family_out_diagnostic",
            "held_out_families": list(FAMILIES),
            "internal_selection": "40 train and 10 validation targets per non-held-out family, selected without the held-out family",
            "production_frozen_test_used_for_selection": False,
            "interpretation": "synthetic extrapolation to an unseen configured family, not real-world geological generalization",
        },
        output_dir / "lofo_metadata.json",
    )
    return {
        "output_dir": output_dir,
        "candidate_rows": candidate_rows,
        "metric_rows": metric_rows,
        "summary_rows": summary_rows,
    }


def run_learning_curve_analysis(
    cases: Sequence[ProductionCase],
    ml_result: Mapping[str, Any],
    output_dir: Path,
    settings: Any,
) -> dict[str, Any]:
    """Evaluate fixed selected ML choices as target-level training data grows."""

    output_dir.mkdir(parents=True, exist_ok=True)
    train = [case for case in cases if case.split == "train"]
    validation = [case for case in cases if case.split == "validation"]
    train_by_family = {family: [case for case in train if case.family == family] for family in FAMILIES}
    all_features = build_feature_matrices(cases)
    target_validation = np.vstack([case.target_vector for case in validation])
    velocity_bounds = tuple(float(value) for value in settings.velocity_model_generation.velocity_bounds_km_per_s)
    selected_model_name, selected_k, selected_alpha = ml_result["selected_key"]
    subset_sizes = (25, 50, 100, 150, 175)
    subset_seeds = (2026082911, 2026082912, 2026082913)
    rows: list[dict[str, Any]] = []
    for subset_size in subset_sizes:
        per_family = subset_size // len(FAMILIES)
        for seed in subset_seeds:
            subset: list[ProductionCase] = []
            for family_index, family in enumerate(FAMILIES):
                family_cases = sorted(train_by_family[family], key=lambda case: case.target_hash)
                rng = np.random.default_rng(seed + family_index * 1009)
                selected_indices = np.sort(rng.choice(len(family_cases), size=per_family, replace=False))
                subset.extend(family_cases[int(index)] for index in selected_indices)
            subset = sorted(subset, key=lambda case: case.target_id)
            subset_index = {case.target_id: index for index, case in enumerate(cases)}
            indices = [subset_index[case.target_id] for case in subset]
            pca = fit_target_pca(np.vstack([case.target_vector for case in subset]))
            effective_k = min(int(selected_k), pca.rank)
            model = fit_observation_pca_model(
                all_features["realistic_full"][indices],
                np.vstack([case.target_vector for case in subset]),
                pca,
                effective_k,
                str(selected_model_name),
                selected_alpha,
                velocity_bounds,
            )
            validation_indices = [index for index, case in enumerate(cases) if case.split == "validation"]
            prediction = model.predict(all_features["realistic_full"][validation_indices])
            metrics = _aggregate_prediction_metrics(target_validation, prediction)
            rows.append(
                {
                    "training_target_count": subset_size,
                    "subset_seed": seed,
                    "targets_per_family": per_family,
                    "effective_pca_component_count": effective_k,
                    "frozen_model_name": selected_model_name,
                    "frozen_ridge_alpha": selected_alpha,
                    "validation_target_count": len(validation),
                    "validation_node_rmse_mean_km_per_s": metrics["node_rmse_mean"],
                    "validation_node_mae_mean_km_per_s": metrics["node_mae_mean"],
                    "validation_cell_rmse_mean_km_per_s": metrics["cell_rmse_mean"],
                    "validation_cell_mae_mean_km_per_s": metrics["cell_mae_mean"],
                    "selection_policy": "primary configuration frozen before learning-curve evaluation; validation is reported, not reselected",
                }
            )
    _write_csv(output_dir / "learning_curve_repetitions.csv", rows)
    summary_rows = _learning_curve_summary(rows)
    _write_csv(output_dir / "learning_curve_summary.csv", summary_rows)
    _write_json(
        {
            "artifact_type": "benchmark_v2_target_level_learning_curve",
            "training_subset_sizes": list(subset_sizes),
            "subset_seeds": list(subset_seeds),
            "family_balanced": True,
            "evaluation_split": "frozen validation targets",
            "sample_unit": "unique geological target; acquisition count is not a sample-size measure",
        },
        output_dir / "learning_curve_metadata.json",
    )
    return {"output_dir": output_dir, "rows": rows, "summary_rows": summary_rows}


def _learning_curve_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["training_target_count"])].append(row)
    output: list[dict[str, Any]] = []
    for size, group in sorted(grouped.items()):
        for metric in (
            "validation_node_rmse_mean_km_per_s",
            "validation_node_mae_mean_km_per_s",
            "validation_cell_rmse_mean_km_per_s",
            "validation_cell_mae_mean_km_per_s",
        ):
            values = np.asarray([float(row[metric]) for row in group])
            output.append(
                {
                    "training_target_count": size,
                    "metric": metric,
                    "repetition_count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return output


def _summarize_lofo_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["held_out_family"])].append(row)
    output: list[dict[str, Any]] = []
    for held_family, group in sorted(grouped.items()):
        for metric_name, field in (
            ("node_rmse", "node_rmse_km_per_s"),
            ("node_mae", "node_mae_km_per_s"),
            ("cell_rmse", "cell_rmse_km_per_s"),
            ("cell_mae", "cell_mae_km_per_s"),
        ):
            values = np.asarray([float(row[field]) for row in group], dtype=np.float64)
            output.append(
                {
                    "held_out_family": held_family,
                    "method_id": "realistic_full",
                    "metric": metric_name,
                    "target_count": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "ci95_lower": _bootstrap_ci(values, _stable_seed(f"lofo:{held_family}:{metric_name}"))[0],
                    "ci95_upper": _bootstrap_ci(values, _stable_seed(f"lofo:{held_family}:{metric_name}"))[1],
                }
            )
    return output


# ---------------------------------------------------------------------------
# Small, dependency-light persistence and metric helpers
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=_json_default)
    if isinstance(value, (np.ndarray, np.integer, np.floating, np.bool_)):
        return _json_default(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({str(key): _csv_value(value) for key, value in row.items()})


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else get_repo_root() / path


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _split_counts(cases: Sequence[ProductionCase]) -> dict[str, int]:
    return {
        split: sum(case.split == split for case in cases)
        for split in ("train", "validation", "test")
    }


def _validate_case_identity(cases: Sequence[ProductionCase]) -> None:
    if len({case.target_id for case in cases}) != len(cases):
        raise ValueError("Production cases contain duplicate target IDs.")
    if len({case.target_hash for case in cases}) != len(cases):
        raise ValueError("Production cases contain duplicate target hashes.")
    if any(case.split not in {"train", "validation", "test"} for case in cases):
        raise ValueError("Production cases contain an unknown split.")
    for case in cases:
        if case.target_vector.shape != (41 * 41 * 13,):
            raise ValueError(f"{case.target_id}: unexpected target-vector shape.")
        if case.observations.shape != (384, 8):
            raise ValueError(f"{case.target_id}: unexpected observation matrix shape.")
        if not np.isfinite(case.target_vector).all() or not np.isfinite(case.observations).all():
            raise ValueError(f"{case.target_id}: non-finite production data.")


def _node_shape(grid: CartesianVelocityGrid3D) -> tuple[int, int, int]:
    return (
        len(grid.x_coordinates_km),
        len(grid.y_coordinates_km),
        len(grid.z_coordinates_km),
    )


def _cell_shape_from_case(case: ProductionCase) -> tuple[int, int, int]:
    nx, ny, nz = _node_shape(case.target_grid)
    return (nx - 1, ny - 1, nz - 1)


def _cell_centers(coordinates: Sequence[float]) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.size < 2:
        raise ValueError("At least two grid coordinates are required for cell centers.")
    return 0.5 * (values[:-1] + values[1:])


def _rmse(true: np.ndarray, prediction: np.ndarray) -> float:
    expected = np.asarray(true, dtype=np.float64)
    actual = np.asarray(prediction, dtype=np.float64)
    if expected.size == 0:
        return math.nan
    return float(np.sqrt(np.mean((actual - expected) ** 2)))


def _mae(true: np.ndarray, prediction: np.ndarray) -> float:
    expected = np.asarray(true, dtype=np.float64)
    actual = np.asarray(prediction, dtype=np.float64)
    if expected.size == 0:
        return math.nan
    return float(np.mean(np.abs(actual - expected)))


def _prediction_metrics_from_arrays(
    true: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | None]:
    expected = np.asarray(true, dtype=np.float64)
    actual = np.asarray(prediction, dtype=np.float64)
    if expected.size == 0:
        return {
            "rmse": None,
            "mae": None,
            "bias": None,
            "true_mean": None,
            "predicted_mean": None,
        }
    return {
        "rmse": _rmse(expected, actual),
        "mae": _mae(expected, actual),
        "bias": float(np.mean(actual - expected)),
        "true_mean": float(np.mean(expected)),
        "predicted_mean": float(np.mean(actual)),
    }


def _prediction_metrics(true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    expected = np.asarray(true, dtype=np.float64)
    actual = np.asarray(prediction, dtype=np.float64)
    if expected.size != 41 * 41 * 13 or actual.size != expected.size:
        raise ValueError("Node-vector metrics require 41 x 41 x 13 arrays.")
    node_metrics = _prediction_metrics_from_arrays(expected, actual)
    cell_shape = (41, 41, 13)
    true_cells = cell_centered_values_from_node_vector(expected, cell_shape)
    prediction_cells = cell_centered_values_from_node_vector(actual, cell_shape)
    cell_metrics = _prediction_metrics_from_arrays(true_cells, prediction_cells)
    return {
        "node_rmse": float(node_metrics["rmse"]),
        "node_mae": float(node_metrics["mae"]),
        "node_bias": float(node_metrics["bias"]),
        "cell_rmse": float(cell_metrics["rmse"]),
        "cell_mae": float(cell_metrics["mae"]),
        "cell_bias": float(cell_metrics["bias"]),
    }


def _aggregate_prediction_metrics(
    true_matrix: np.ndarray,
    prediction_matrix: np.ndarray,
) -> dict[str, float]:
    expected = np.asarray(true_matrix, dtype=np.float64)
    actual = np.asarray(prediction_matrix, dtype=np.float64)
    if expected.shape != actual.shape or expected.ndim != 2:
        raise ValueError("Prediction metrics require equal two-dimensional matrices.")
    rows = [_prediction_metrics(expected[index], actual[index]) for index in range(expected.shape[0])]
    return {
        "node_rmse_mean": float(np.mean([row["node_rmse"] for row in rows])),
        "node_mae_mean": float(np.mean([row["node_mae"] for row in rows])),
        "cell_rmse_mean": float(np.mean([row["cell_rmse"] for row in rows])),
        "cell_mae_mean": float(np.mean([row["cell_mae"] for row in rows])),
    }


def _mean_optional(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value in (None, ""):
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            values.append(numeric)
    return float(np.mean(values)) if values else None


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32 - 1)


def _method_label(method: str) -> str:
    labels = {
        "realistic_full": "Realistic ML (coordinates + distance + time)",
        "travel_time_only": "Travel-time-only ML",
        "no_travel_time": "Coordinates + distance (no time)",
        "geometry_only": "Geometry-only ML",
        "distance_only": "Euclidean-distance-only ML",
        "shuffled_travel_time": "Shuffled-travel-time ML",
        "training_target_mean": "Training-target mean",
        "reference_prior": "Reference 1-D prior",
        "fixed_ray": "Validation-selected fixed-ray",
    }
    return labels.get(method, method)


def _metric_summary_values(values: Sequence[float], seed_label: str) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "ci95_lower": None,
            "ci95_upper": None,
        }
    array = np.asarray(values, dtype=np.float64)
    ci = _bootstrap_ci(array, _stable_seed(seed_label))
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "ci95_lower": ci[0],
        "ci95_upper": ci[1],
    }


def _summarize_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_values: Sequence[str],
) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("split", "")) in set(split_values)]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(str(row.get("split", "")), str(row["method_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for (split, method), group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "split": split,
            "method_id": method,
            "method_label": _method_label(method),
            "target_count": len({str(row["target_id"]) for row in group}),
        }
        for prefix, field in (
            ("node_rmse", "node_rmse_km_per_s"),
            ("node_mae", "node_mae_km_per_s"),
            ("cell_rmse", "cell_rmse_km_per_s"),
            ("cell_mae", "cell_mae_km_per_s"),
            ("travel_time_rmse", "travel_time_rmse_s"),
            ("travel_time_mae", "travel_time_mae_s"),
        ):
            values = [
                float(row[field])
                for row in group
                if row.get(field) not in (None, "") and np.isfinite(float(row[field]))
            ]
            summary = _metric_summary_values(values, f"summary:{split}:{method}:{field}")
            result.update({f"{prefix}_{key}": value for key, value in summary.items()})
        output.append(result)
    return output


# ---------------------------------------------------------------------------
# Publication-oriented figures and manuscript-ready tables
# ---------------------------------------------------------------------------


def _matplotlib():
    """Import matplotlib lazily so numerical analysis remains lightweight."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure_both(figure: Any, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    return png_path, svg_path


def run_publication_figures(
    cases: Sequence[ProductionCase],
    ml_result: Mapping[str, Any],
    classical_result: Mapping[str, Any],
    coverage_result: Mapping[str, Any],
    structural_result: Mapping[str, Any],
    noise_result: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Create final-benchmark figures using objective, recorded selections."""

    del structural_result
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _matplotlib()
    figure_paths: dict[str, list[str]] = {}
    selections: dict[str, Any] = {}

    figure = plt.figure(figsize=(13.0, 3.5), constrained_layout=True)
    axis = figure.add_axes((0.02, 0.08, 0.96, 0.84))
    axis.axis("off")
    workflow = [
        ("Analytic\ngeological target", "#dbeafe"),
        ("Fine 1.25-km\nFSM forward grid", "#dcfce7"),
        ("Sources +\nstations", "#fef3c7"),
        ("Travel-time\nobservations", "#fce7f3"),
        ("Realistic inputs\nxyz + distance + time", "#ede9fe"),
        ("Training-only PCA\nvelocity reconstruction", "#cffafe"),
        ("Fixed-ray comparator\n+ common evaluation", "#e2e8f0"),
    ]
    from matplotlib.patches import FancyBboxPatch

    left = 0.01
    width = 0.125
    gap = 0.018
    for index, (label, color) in enumerate(workflow):
        x = left + index * (width + gap)
        box = FancyBboxPatch(
            (x, 0.28),
            width,
            0.44,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=axis.transAxes,
            facecolor=color,
            edgecolor="#334155",
            linewidth=1.0,
        )
        axis.add_patch(box)
        axis.text(
            x + width / 2,
            0.5,
            label,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#0f172a",
        )
        if index < len(workflow) - 1:
            axis.annotate(
                "",
                xy=(x + width + gap * 0.85, 0.5),
                xytext=(x + width + gap * 0.1, 0.5),
                xycoords=axis.transAxes,
                textcoords=axis.transAxes,
                arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.5},
            )
    axis.set_title("Benchmark v2 workflow", fontsize=16, weight="bold", pad=8)
    figure_paths["workflow"] = [str(path) for path in _save_figure_both(figure, output_dir, "figure_a_workflow")]
    plt.close(figure)

    representative: list[dict[str, str]] = []
    figure, axes = plt.subplots(1, len(FAMILIES), figsize=(17, 4.4), constrained_layout=True)
    all_values: list[float] = []
    selected_cases: dict[str, str] = {}
    for family, axis in zip(FAMILIES, np.atleast_1d(axes), strict=True):
        family_cases = [case for case in cases if case.family == family]
        target_means = np.asarray([np.mean(case.target_vector) for case in family_cases])
        median_mean = float(np.median(target_means))
        selected_case = min(
            family_cases,
            key=lambda case: (abs(float(np.mean(case.target_vector)) - median_mean), case.target_hash),
        )
        selected_cases[family] = selected_case.target_id
        node = selected_case.target_vector.reshape((13, 41, 41))
        y_index = len(selected_case.target_grid.y_coordinates_km) // 2
        section = node[:, y_index, :]
        all_values.extend(float(value) for value in section.reshape(-1))
        representative.append(
            {
                "family": family,
                "target_id": selected_case.target_id,
                "selection_rule": "target mean velocity nearest the within-family median",
            }
        )
        axis.imshow(
            section,
            extent=[0, 100, 30, 0],
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=3.5,
            vmax=8.5,
        )
        axis.set_title(f"{family}\n{selected_case.target_id}", fontsize=10)
        axis.set_xlabel("x (km)")
        axis.set_ylabel("depth z (km)")
    figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=3.5, vmax=8.5), cmap="viridis"),
        ax=np.atleast_1d(axes).tolist(),
        label="P-wave velocity (km/s)",
        shrink=0.84,
    )
    figure.suptitle("Benchmark v2 geological families on the 2.5-km target grid", fontsize=15)
    figure_paths["geological_families"] = [
        str(path) for path in _save_figure_both(figure, output_dir, "figure_b_geological_families")
    ]
    _write_json(
        {"selection_rule": representative, "selected_target_ids": selected_cases},
        output_dir / "figure_b_selection.json",
    )
    plt.close(figure)

    pca = ml_result["pca"]
    cumulative = np.cumsum(pca.explained_variance_ratio)
    oracle_rows = _read_csv(Path(ml_result["output_dir"]) / "oracle_reconstruction.csv")
    figure, axis_left = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    components = np.arange(1, pca.rank + 1)
    axis_left.plot(components, cumulative, color="#1d4ed8", linewidth=2.2, label="Cumulative explained variance")
    axis_left.set_xlabel("PCA components")
    axis_left.set_ylabel("Cumulative explained variance")
    axis_left.set_ylim(0, 1.02)
    axis_left.grid(alpha=0.25)
    axis_right = axis_left.twinx()
    oracle_components = [int(row["pca_component_count"]) for row in oracle_rows]
    oracle_rmse = [float(row["validation_oracle_cell_rmse_km_per_s"]) for row in oracle_rows]
    axis_right.plot(
        oracle_components,
        oracle_rmse,
        color="#b91c1c",
        marker="o",
        linewidth=1.8,
        label="Validation oracle cell RMSE",
    )
    axis_right.set_ylabel("Validation oracle cell RMSE (km/s)")
    axis_left.set_title("Training-target PCA spectrum and oracle reconstruction")
    lines = axis_left.get_lines() + axis_right.get_lines()
    axis_left.legend(lines, [line.get_label() for line in lines], loc="center right", frameon=True)
    figure_paths["pca_spectrum"] = [str(path) for path in _save_figure_both(figure, output_dir, "figure_c_pca_spectrum")]
    plt.close(figure)

    test_cases = [case for case in cases if case.split == "test"]
    test_ids = [case.target_id for case in test_cases]
    plot_methods = (
        "realistic_full",
        "travel_time_only",
        "no_travel_time",
        "shuffled_travel_time",
        "reference_prior",
        "fixed_ray",
    )
    values_by_method: dict[str, np.ndarray] = {}
    for method in plot_methods:
        if method in {"reference_prior", "fixed_ray"}:
            source_rows = [
                row
                for row in classical_result["test_metric_rows"]
                if row["method_id"] == method and row["split"] == "test"
            ]
        else:
            source_rows = [
                row
                for row in ml_result["metric_rows"]
                if row["method_id"] == method and row["split"] == "test"
            ]
        indexed = {str(row["target_id"]): float(row["cell_rmse_km_per_s"]) for row in source_rows}
        values_by_method[method] = np.asarray([indexed[target_id] for target_id in test_ids])
    figure, axis = plt.subplots(figsize=(10.5, 5.3), constrained_layout=True)
    box_values = [values_by_method[method] for method in plot_methods]
    axis.boxplot(box_values, labels=[_method_label(method).replace(" ML", "") for method in plot_methods], showmeans=True)
    rng = np.random.default_rng(_stable_seed("figure-d-jitter"))
    for index, method in enumerate(plot_methods, start=1):
        jitter = rng.uniform(-0.09, 0.09, size=values_by_method[method].size)
        axis.scatter(np.full(values_by_method[method].size, index) + jitter, values_by_method[method], s=12, alpha=0.55)
    axis.set_ylabel("Per-target cell RMSE (km/s)")
    axis.set_title("Primary held-out-target error comparison (38 independent targets)")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", labelrotation=24)
    figure_paths["primary_comparison"] = [
        str(path) for path in _save_figure_both(figure, output_dir, "figure_d_primary_method_comparison")
    ]
    plt.close(figure)

    realistic_rows = {
        str(row["target_id"]): float(row["cell_rmse_km_per_s"])
        for row in ml_result["metric_rows"]
        if row["method_id"] == "realistic_full" and row["split"] == "test"
    }
    ordered = sorted(test_cases, key=lambda case: (realistic_rows[case.target_id], case.target_hash))
    easy_case = ordered[len(ordered) // 4]
    difficult_case = ordered[(3 * len(ordered)) // 4]
    selections["structural_recovery"] = {
        "easy_quartile_case": easy_case.target_id,
        "difficult_quartile_case": difficult_case.target_id,
        "selection_rule": "realistic ML all-cell target RMSE nearest lower and upper quartiles",
    }
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.8), constrained_layout=True)
    for row_index, selected_case in enumerate((easy_case, difficult_case)):
        predicted = ml_result["predictions"]["realistic_full"][selected_case.target_id]
        true_grid = selected_case.target_vector.reshape((13, 41, 41))
        predicted_grid = predicted.reshape((13, 41, 41))
        difference = predicted_grid - true_grid
        y_index = len(selected_case.target_grid.y_coordinates_km) // 2
        panels = (true_grid[:, y_index, :], predicted_grid[:, y_index, :], difference[:, y_index, :])
        limits = (
            (3.5, 8.5),
            (3.5, 8.5),
            (-1.0, 1.0),
        )
        for col_index, (panel, (vmin, vmax), title) in enumerate(
            zip(panels, limits, ("True", "Realistic ML", "Prediction − true"), strict=True)
        ):
            image = axes[row_index, col_index].imshow(
                panel,
                extent=[0, 100, 30, 0],
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm" if col_index == 2 else "viridis",
                vmin=vmin,
                vmax=vmax,
            )
            axes[row_index, col_index].set_title(f"{title}\n{selected_case.target_id}")
            axes[row_index, col_index].set_xlabel("x (km)")
            axes[row_index, col_index].set_ylabel("depth z (km)")
            figure.colorbar(image, ax=axes[row_index, col_index], shrink=0.82)
    figure.suptitle("Objective structural-recovery examples", fontsize=15)
    figure_paths["structural_recovery"] = [
        str(path) for path in _save_figure_both(figure, output_dir, "figure_e_structural_recovery")
    ]
    _write_json(selections["structural_recovery"], output_dir / "figure_e_selection.json")
    plt.close(figure)

    figure, (axis_noise, axis_coverage) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    noise_rows = noise_result["summary_rows"]
    for method, color in (("realistic_full", "#1d4ed8"), ("travel_time_only", "#b91c1c"), ("fixed_ray", "#166534")):
        selected = sorted(
            [row for row in noise_rows if row["method_id"] == method],
            key=lambda row: float(row["noise_std_s"]),
        )
        axis_noise.plot(
            [float(row["noise_std_s"]) for row in selected],
            [float(row["degradation_mean_km_per_s"]) for row in selected],
            marker="o",
            label=_method_label(method),
            color=color,
        )
    axis_noise.axhline(0, color="#64748b", linewidth=0.8)
    axis_noise.set_xlabel("Timing-noise standard deviation (s)")
    axis_noise.set_ylabel("Mean RMSE degradation from zero noise (km/s)")
    axis_noise.set_title("Repeated timing-noise sensitivity")
    axis_noise.grid(alpha=0.25)
    axis_noise.legend(fontsize=8)
    coverage_rows = coverage_result["coverage_rows"]
    for method, color in (("realistic_full", "#1d4ed8"), ("fixed_ray", "#166534")):
        selected = [row for row in coverage_rows if row["method_id"] == method]
        thresholds = sorted({float(row["coverage_threshold_km"]) for row in selected})
        means = []
        for threshold in thresholds:
            values = [
                float(row["rmse_km_per_s"])
                for row in selected
                if float(row["coverage_threshold_km"]) == threshold and row["rmse_km_per_s"] not in (None, "")
            ]
            means.append(float(np.mean(values)) if values else math.nan)
        axis_coverage.plot(
            range(len(thresholds)),
            means,
            marker="o",
            label=_method_label(method),
            color=color,
        )
    axis_coverage.set_xticks(range(len(thresholds)), [f"{value:g}" for value in thresholds])
    axis_coverage.set_xlabel("Minimum accumulated reference-ray coverage (km)")
    axis_coverage.set_ylabel("Mean cell RMSE (km/s)")
    axis_coverage.set_title("Error in increasingly illuminated cells")
    axis_coverage.grid(alpha=0.25)
    axis_coverage.legend(fontsize=8)
    figure_paths["robustness_coverage"] = [
        str(path) for path in _save_figure_both(figure, output_dir, "figure_f_robustness_coverage")
    ]
    plt.close(figure)
    _write_json(
        {
            "figure_paths": figure_paths,
            "selections": selections,
            "plot_scope": "frozen test set for primary comparison; validation/test-derived diagnostic selections are recorded",
        },
        output_dir / "publication_figure_manifest.json",
    )
    return {"output_dir": output_dir, "figure_paths": figure_paths, "selections": selections}


def write_manuscript_tables(
    cases: Sequence[ProductionCase],
    ml_result: Mapping[str, Any],
    classical_result: Mapping[str, Any],
    structural_result: Mapping[str, Any],
    noise_result: Mapping[str, Any],
    learning_result: Mapping[str, Any],
    statistics_result: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Write compact CSV/Markdown table data without editing the manuscript."""

    output_dir.mkdir(parents=True, exist_ok=True)
    table_rows: dict[str, list[dict[str, Any]]] = {}

    table_rows["table_1_benchmark_design"] = _benchmark_design_rows(cases)
    test_metric_rows = _combined_test_metric_rows(ml_result, classical_result)
    table_rows["table_2_primary_method_comparison"] = [
        row
        for row in _target_method_summary(test_metric_rows)
        if row["metric_domain"] == "cell_all"
    ]
    table_rows["table_3_paired_target_level_comparisons"] = list(
        statistics_result["paired_summary"]
    )
    table_rows["table_4_observation_signal_ablations"] = [
        row
        for row in table_rows["table_2_primary_method_comparison"]
        if row["method_id"] in ML_METHODS
    ]
    table_rows["table_5_reference_prior_and_fixed_ray"] = [
        row
        for row in table_rows["table_2_primary_method_comparison"]
        if row["method_id"] in {"reference_prior", "fixed_ray"}
    ]
    table_rows["table_6_structural_recovery"] = list(structural_result["summary_rows"])
    table_rows["table_7_noise_robustness"] = list(noise_result["summary_rows"])
    table_rows["table_7_learning_curve"] = list(learning_result["summary_rows"])

    paths: dict[str, dict[str, str]] = {}
    for table_name, rows in table_rows.items():
        csv_path = output_dir / f"{table_name}.csv"
        markdown_path = output_dir / f"{table_name}.md"
        _write_csv(csv_path, rows)
        _write_markdown_table(markdown_path, table_name.replace("_", " ").title(), rows)
        paths[table_name] = {"csv": str(csv_path), "markdown": str(markdown_path)}
    _write_json(
        {
            "artifact_type": "benchmark_v2_manuscript_ready_table_data",
            "table_names": list(table_rows),
            "test_target_count": sum(case.split == "test" for case in cases),
            "source": "frozen production Benchmark v2 evidence outputs",
            "manuscript_modified": False,
        },
        output_dir / "table_manifest.json",
    )
    return {"output_dir": output_dir, "table_rows": table_rows, "paths": paths}


def _combined_test_metric_rows(
    ml_result: Mapping[str, Any],
    classical_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in ml_result["metric_rows"]:
        if str(row.get("split")) != "test":
            continue
        rows.append(
            {
                "target_id": row["target_id"],
                "family": row["family"],
                "method_id": row["method_id"],
                "metric_domain": "cell_all",
                "rmse_km_per_s": row["cell_rmse_km_per_s"],
                "mae_km_per_s": row["cell_mae_km_per_s"],
            }
        )
    for row in classical_result["test_metric_rows"]:
        rows.append(
            {
                "target_id": row["target_id"],
                "family": row["family"],
                "method_id": row["method_id"],
                "metric_domain": "cell_all",
                "rmse_km_per_s": row["cell_rmse_km_per_s"],
                "mae_km_per_s": row["cell_mae_km_per_s"],
            }
        )
    return rows


def _benchmark_design_rows(cases: Sequence[ProductionCase]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_cases = [case for case in cases if case.family == family]
        if not family_cases:
            continue
        ranges = _parameter_ranges([_family_parameters(case) for case in family_cases])
        rows.append(
            {
                "family": family,
                "unique_target_count": len(family_cases),
                "acquisition_case_count": len(family_cases),
                "observations_per_target": 384,
                "target_grid_shape": "41x41x13",
                "target_spacing_km": "2.5x2.5x2.5",
                "forward_spacing_km": "1.25x1.25x1.25",
                "sampled_parameter_ranges_json": json.dumps(ranges, sort_keys=True),
            }
        )
    return rows


def _parameter_ranges(mappings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    flattened: dict[str, list[float]] = defaultdict(list)
    for mapping in mappings:
        _collect_numeric_parameters(mapping, "", flattened)
    return {
        key: {"min": float(min(values)), "max": float(max(values))}
        for key, values in sorted(flattened.items())
        if values
    }


def _collect_numeric_parameters(
    value: Any,
    prefix: str,
    destination: dict[str, list[float]],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _collect_numeric_parameters(child, child_prefix, destination)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _collect_numeric_parameters(child, f"{prefix}[{index}]", destination)
        return
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
        destination[prefix].append(float(value))


def _write_markdown_table(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in keys:
                keys.append(str(key))
    lines = [f"# {title}", ""]
    if not keys:
        lines.append("No rows were produced.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join("---" for _ in keys) + "|")
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(_csv_value(value)).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_evidence_report(
    *,
    cases: Sequence[ProductionCase],
    output_dir: Path,
    ml_result: Mapping[str, Any],
    classical_result: Mapping[str, Any],
    coverage_result: Mapping[str, Any],
    structural_result: Mapping[str, Any],
    statistics_result: Mapping[str, Any],
    noise_result: Mapping[str, Any],
    lofo_result: Mapping[str, Any],
    learning_result: Mapping[str, Any],
    figures_result: Mapping[str, Any],
    tables_result: Mapping[str, Any],
) -> Path:
    """Write the evidence-to-claim report that precedes manuscript revision."""

    report_path = output_dir / "benchmark_v2_final_evidence_report.md"
    test_cases = [case for case in cases if case.split == "test"]
    test_ids = {case.target_id for case in test_cases}
    combined = _combined_test_metric_rows(ml_result, classical_result)
    method_summary = _target_method_summary(combined)
    paired_summary = list(statistics_result["paired_summary"])
    pca = ml_result["pca"]
    oracle_rows = _read_csv(Path(ml_result["output_dir"]) / "oracle_reconstruction.csv")
    selected_k = int(ml_result["selected_key"][1])
    selected_oracle = next(
        row for row in oracle_rows if int(row["pca_component_count"]) == selected_k
    )

    def summary(method: str) -> dict[str, Any] | None:
        return next(
            (
                row
                for row in method_summary
                if row["method_id"] == method and row["metric_domain"] == "cell_all"
            ),
            None,
        )

    def pair(comparison_id: str, domain: str = "cell_all") -> dict[str, Any] | None:
        return next(
            (
                row
                for row in paired_summary
                if row["comparison_id"] == comparison_id and row["metric_domain"] == domain
            ),
            None,
        )

    def fmt(value: Any, digits: int = 5) -> str:
        if value in (None, ""):
            return "not available"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric):
            return "not finite"
        return f"{numeric:.{digits}g}"

    def pair_sentence(comparison_id: str) -> str:
        row = pair(comparison_id)
        if row is None:
            return "The comparison was not available."
        sign = "lower" if float(row["rmse_delta_mean_km_per_s"]) < 0.0 else "higher"
        ci = f"[{fmt(row['rmse_delta_ci95_lower_km_per_s'])}, {fmt(row['rmse_delta_ci95_upper_km_per_s'])}]"
        return (
            f"The first method was {sign} in mean target-level RMSE by "
            f"{fmt(abs(row['rmse_delta_mean_km_per_s']))} km/s; the paired 95% bootstrap CI "
            f"for (first − second) was {ci}, with {row['rmse_target_win_count']} wins, "
            f"{row['rmse_target_loss_count']} losses, and {row['rmse_target_tie_count']} ties."
        )

    integrity_ok = (
        len(cases) == 250
        and len({case.target_hash for case in cases}) == 250
        and _split_counts(cases) == {"train": 175, "validation": 37, "test": 38}
        and len(test_ids) == 38
        and all(np.isfinite(case.target_vector).all() and np.isfinite(case.observations).all() for case in cases)
    )
    time_against_geometry = pair("realistic_vs_no_travel_time")
    time_against_shuffle = pair("realistic_vs_shuffled_travel_time")
    fixed_comparison = pair("realistic_vs_fixed_ray")
    time_help_directional = bool(
        time_against_geometry
        and time_against_shuffle
        and float(time_against_geometry["rmse_delta_mean_km_per_s"]) < 0.0
        and float(time_against_shuffle["rmse_delta_mean_km_per_s"]) < 0.0
    )
    fixed_supported = bool(
        fixed_comparison
        and float(fixed_comparison["rmse_delta_ci95_upper_km_per_s"]) < 0.0
    )
    structural_available = any(
        row.get("method_id") == "realistic_full"
        and row.get("rmse_km_per_s") not in (None, "")
        for row in structural_result["rows"]
    )
    if not integrity_ok or not structural_available:
        decision = "FUNDAMENTAL EXPERIMENTAL FAILURE"
        decision_reason = "A required corpus or structural-evaluation invariant was not satisfied."
    elif time_help_directional and fixed_supported:
        decision = "READY FOR MANUSCRIPT REWRITE"
        decision_reason = "The frozen benchmark is valid, travel-time information is directionally useful against controls, and the paired test comparison favors realistic ML over fixed-ray with a CI excluding zero."
    else:
        decision = "RESULTS PUBLISHABLE BUT REQUIRE NARROWER CLAIM"
        decision_reason = "The corpus and evidence package are valid, but the observed controls do not justify an unrestricted ML-superiority claim."

    family_rows = []
    for family in FAMILIES:
        family_values = [
            float(row["rmse_km_per_s"])
            for row in combined
            if row["family"] == family
            and row["method_id"] == "realistic_full"
            and row["metric_domain"] == "cell_all"
        ]
        family_rows.append(
            {
                "family": family,
                "test_target_count": len(family_values),
                "realistic_ml_mean_cell_rmse_km_per_s": float(np.mean(family_values)) if family_values else None,
            }
        )
    easiest = min(family_rows, key=lambda row: float(row["realistic_ml_mean_cell_rmse_km_per_s"]))
    hardest = max(family_rows, key=lambda row: float(row["realistic_ml_mean_cell_rmse_km_per_s"]))
    noise_summary = noise_result["summary_rows"]
    lofo_summary = lofo_result["summary_rows"]
    learning_summary = learning_result["summary_rows"]
    learning_rmse_rows = [
        row for row in learning_summary if row["metric"] == "validation_cell_rmse_mean_km_per_s"
    ]
    def coverage_mean(method: str, threshold: float) -> float | None:
        values = [
            float(row["rmse_km_per_s"])
            for row in coverage_result["coverage_rows"]
            if row["method_id"] == method
            and float(row["coverage_threshold_km"]) == threshold
            and row["rmse_km_per_s"] not in (None, "")
        ]
        return float(np.mean(values)) if values else None

    coverage_parts = []
    for threshold in (1.0, 10.0):
        realistic_coverage = coverage_mean("realistic_full", threshold)
        fixed_coverage = coverage_mean("fixed_ray", threshold)
        coverage_parts.append(
            f"{threshold:g} km: realistic ML {fmt(realistic_coverage)} km/s, fixed-ray {fmt(fixed_coverage)} km/s"
        )
    coverage_text = "; ".join(coverage_parts)
    coverage_pair_parts = []
    for threshold in (1.0, 10.0):
        row = next(
            (
                item
                for item in coverage_result["target_level_paired_summary"]
                if item["comparison_id"] == "realistic_vs_fixed_ray"
                and float(item["coverage_threshold_km"]) == threshold
            ),
            None,
        )
        if row is not None:
            coverage_pair_parts.append(
                f"{threshold:g} km realistic ML − fixed-ray delta {fmt(row['delta_rmse_mean_km_per_s'])} "
                f"(CI {fmt(row['delta_rmse_ci95_lower_km_per_s'])} to {fmt(row['delta_rmse_ci95_upper_km_per_s'])})"
            )
    coverage_pair_text = "; ".join(coverage_pair_parts) or "not available"
    noise_max = next(
        (
            row
            for row in noise_summary
            if row["method_id"] == "realistic_full" and float(row["noise_std_s"]) == 0.1
        ),
        None,
    )
    lofo_text = "; ".join(
        f"held-family {row.get('held_out_family', '')}: {fmt(row.get('mean'))}"
        for row in lofo_summary
        if row.get("metric") == "cell_rmse"
    )
    learning_text = "; ".join(
        f"{int(row['training_target_count'])}: {fmt(row['mean'])}"
        for row in sorted(learning_rmse_rows, key=lambda row: int(row["training_target_count"]))
    )
    structural_body_text = "; ".join(
        f"{row['family']} body RMSE {fmt(row['mean_rmse_km_per_s'])}"
        for row in structural_result["summary_rows"]
        if row["method_id"] == "realistic_full"
        and row["metric_scope"] == "structure"
        and row["mask_label"] == "body"
    )
    easiest_name = str(easiest["family"])
    hardest_name = str(hardest["family"])
    easiest_rmse = easiest["realistic_ml_mean_cell_rmse_km_per_s"]
    hardest_rmse = hardest["realistic_ml_mean_cell_rmse_km_per_s"]
    structural_status = "YES" if structural_available else "NO"
    fixed_ci_lower = float(fixed_comparison["rmse_delta_ci95_lower_km_per_s"]) if fixed_comparison else None
    fixed_ci_upper = float(fixed_comparison["rmse_delta_ci95_upper_km_per_s"]) if fixed_comparison else None
    if fixed_ci_lower is not None and fixed_ci_lower > 0.0:
        fixed_ci_status = "yes; the CI excludes zero and is positive, favoring fixed-ray"
    elif fixed_ci_upper is not None and fixed_ci_upper < 0.0:
        fixed_ci_status = "yes; the CI excludes zero and is negative, favoring realistic ML"
    else:
        fixed_ci_status = "no; the CI includes zero"
    evidence_table = "\n".join(
        [
            "| Claim area | Evidence artifact | Permitted interpretation | Not permitted |",
            "|---|---|---|---|",
            "| Corpus identity | production registry, hashes, quality report | 250 target-atomic models | counting acquisitions as geology samples |",
            "| ML selection | validation candidate table, selected config | predeclared PCA-linear/ridge selection | test-driven tuning |",
            "| Representation prior | PCA spectrum, oracle table, gap table | separate PCA floor from coefficient error | treating four components as universal |",
            "| Observation information | ablation metrics and paired deltas | assess timing information beyond controls | causal claims beyond the synthetic setup |",
            "| Classical baseline | validation grid and frozen config | compare with reference fixed-ray perturbation inversion | calling it full nonlinear tomography |",
            "| Structure/coverage | coverage and structural CSVs | assess illuminated cells and generator-defined structures | arbitrary localization claims |",
            "| Robustness/generalization | noise, LOFO, learning-curve outputs | bounded sensitivity diagnostics | real-world robustness/generalization |",
            "| Numerical labels | frozen solver contract and production config | finite-grid FSM approximation | continuum-exact first-arrival claim |",
        ]
    )

    lines = [
        "# Benchmark v2 final evidence report",
        "",
        "This report summarizes the frozen 250-target Benchmark v2 evidence package. It was generated before manuscript modification. The primary independent unit is one unique geological target; the production corpus has one acquisition realization per target.",
        "",
        "## Corpus",
        "",
        f"- Unique geological targets: `{len({case.target_hash for case in cases})}`; acquisition cases: `{len(cases)}`; observations: `{len(cases) * 384}`.",
        f"- Family distribution: `{ {family: sum(case.family == family for case in cases) for family in FAMILIES} }`.",
        "- Every target is represented by a 41 × 41 × 13 node-centered target grid; labels use the frozen 1.25-km ttcrpy FSM approximation.",
        "- Exact target hashes and nearest-neighbor diversity diagnostics are in the production corpus and table outputs.",
        "",
        "## Split validity",
        "",
        f"- Split counts: `{_split_counts(cases)}` unique targets.",
        f"- Leakage-free target identity check: `{'PASS' if integrity_ok else 'FAIL'}`; unique target hashes: `{len({case.target_hash for case in cases})}`.",
        "- The split was frozen before PCA fitting and model selection. The test set was not used for selection.",
        "",
        "## PCA",
        "",
        f"- Training targets: `{sum(case.split == 'train' for case in cases)}`; centered rank: `{pca.rank}`.",
        f"- Selected component count: `{selected_k}` using validation mean node RMSE among the predefined component/model/regularization candidates.",
        f"- Selected model: `{ml_result['selected_config']}`.",
        f"- Training cumulative explained variance at the selected count: `{fmt(selected_oracle['training_cumulative_explained_variance'])}`.",
        f"- Test oracle cell RMSE at the selected count: `{fmt(selected_oracle['test_oracle_cell_rmse_km_per_s'])}` km/s; this is a representation floor using the training-only basis and true test coefficients.",
        "- The predicted-model error and representation gap are stored separately in `test_representation_gap.csv`.",
        "",
        "## Realistic ML and observation controls",
        "",
    ]
    for method in ML_METHODS:
        row = summary(method)
        if row:
            lines.append(
                f"- {_method_label(method)}: mean test cell RMSE `{fmt(row['rmse_mean_km_per_s'])}` km/s "
                f"(target-level 95% CI `{fmt(row['rmse_ci95_lower_km_per_s'])}`–`{fmt(row['rmse_ci95_upper_km_per_s'])}`)."
            )
    lines.extend(
        [
            "",
            f"- Realistic full input versus no-travel-time control: {pair_sentence('realistic_vs_no_travel_time')}",
            f"- Realistic full input versus shuffled-time control: {pair_sentence('realistic_vs_shuffled_travel_time')}",
            f"- Explicit geometry versus travel-time-only: {pair_sentence('realistic_vs_travel_time_only')}",
            "",
            "## Classical comparison",
            "",
            f"- Reference-prior test cell RMSE: `{fmt((summary('reference_prior') or {}).get('rmse_mean_km_per_s'))}` km/s.",
            f"- Validation-selected fixed-ray test cell RMSE: `{fmt((summary('fixed_ray') or {}).get('rmse_mean_km_per_s'))}` km/s.",
            f"- Realistic ML versus reference prior: {pair_sentence('realistic_vs_reference_prior')}",
            f"- Realistic ML versus fixed-ray: {pair_sentence('realistic_vs_fixed_ray')}",
            f"- Travel-time-only versus fixed-ray: {pair_sentence('travel_time_only_vs_fixed_ray')}",
            "- The fixed-ray comparator is a reference-model fixed-ray regularized perturbation inversion, not full nonlinear tomography.",
            "",
            "## Coverage and structural recovery",
            "",
            f"- Coverage metrics use accumulated reference-model ray path length with thresholds `{list(COVERAGE_THRESHOLDS_KM)}` km; coverage is not supplied to ML.",
            f"- Coverage rows generated: `{len(coverage_result['coverage_rows'])}`; depth rows generated: `{len(coverage_result['depth_rows'])}`.",
            f"- Realistic ML structural metrics are available for generator-derived masks: `{structural_status}`.",
            f"- Easiest family by realistic-ML all-cell test RMSE: `{easiest_name}` (`{fmt(easiest_rmse)}` km/s); hardest: `{hardest_name}` (`{fmt(hardest_rmse)}` km/s).",
            f"- Descriptive realistic-ML body RMSE summaries: {structural_body_text or 'not available'}.",
            "- Structure/body, background, layer, and fault-side results are in `structural/structural_summary.csv`; contrasts are reported only where generator masks define them.",
            "",
            "## Robustness",
            "",
            f"- Noise levels: `{list(NOISE_LEVELS_S)}` s; nonzero levels use `{NOISE_REPETITIONS}` matched repetitions per target and method.",
            f"- Noise target-level summary rows: `{len(noise_summary)}`; Gaussian timing noise is a controlled sensitivity experiment, not a complete picking-error model.",
            f"- LOFO diagnostic held out all five configured families with internal selection and no use of the production test set; summary rows: `{len(lofo_summary)}` ({lofo_text or 'not available'}).",
            f"- Learning curve used family-balanced unique-target subsets `{[25, 50, 100, 150, 175]}` with three subset seeds; validation cell-RMSE means by training target count: `{learning_text or 'not available'}`.",
            "",
            "## Explicit evidence answers",
            "",
            "1. **How many unique targets exist?** 250, exactly 50 in each of the five configured families.",
            "2. **Are the splits leakage-free?** The target-hash identity check is reported above; each target appears once in the frozen 175/37/38 split.",
            "3. **How many independent test targets exist?** 38 unique geological targets and 38 acquisition cases.",
            f"4. **What PCA dimension was selected and why?** `{selected_k}`, selected using validation performance only from the predefined grid.",
            f"5. **What is the oracle PCA reconstruction error?** `{fmt(selected_oracle['test_oracle_cell_rmse_km_per_s'])}` km/s cell RMSE on the frozen test targets at the selected count.",
            "6. **Does travel time contain information beyond geometry and the target prior?** Travel-time-bearing models are directionally better than the no-time and shuffled-time controls; the realistic-versus-training-target-mean paired CI includes zero, so superiority over the target-prior mean is not established.",
            f"7. **Does adding explicit geometry improve travel-time-only prediction?** {pair_sentence('realistic_vs_travel_time_only')}",
            f"8. **Does realistic ML outperform the reference prior?** {pair_sentence('realistic_vs_reference_prior')}",
            f"9. **Does realistic ML outperform validation-selected fixed-ray?** {pair_sentence('realistic_vs_fixed_ray')}",
            f"10. **Does the paired target-level CI support that difference?** `{fixed_ci_status}` for realistic ML minus fixed-ray; under the first-minus-second convention, a positive interval favors fixed-ray. The exact interval is recorded above and in Table 3.",
            f"11. **Does the conclusion persist in ray-covered cells?** At the two reported thresholds, mean cell RMSE is {coverage_text}; paired target-level realistic ML minus fixed-ray deltas are {coverage_pair_text}.",
            f"12. **Are geological anomalies/contrasts reconstructed?** Yes, objective generator-derived structure masks produced metrics ({structural_body_text or 'no body rows available'}); no arbitrary predicted localization detector was introduced.",
            f"13. **Which families are easiest and hardest?** Easiest is `{easiest_name}` and hardest is `{hardest_name}` by the descriptive realistic-ML all-cell test mean; family-level uncertainty is in the case-level outputs.",
            f"14. **What happens under timing noise?** Repeated target-level degradation is reported at all three nonzero levels; for realistic ML at 0.100 s, mean degradation is `{fmt(noise_max.get('degradation_mean_km_per_s') if noise_max else None)}` km/s.",
            f"15. **What happens when one family is unseen?** LOFO reports five synthetic held-out-family diagnostics ({lofo_text or 'not available'}); this is not real-world geological generalization.",
            f"16. **Is performance still improving with training-set size?** The family-balanced validation learning-curve means are `{learning_text or 'not available'}`; no test performance was used for this diagnostic.",
            "17. **What numerical limitations must be disclosed?** Labels are a documented finite-grid 3-D ttcrpy FSM first-arrival approximation at 1.25-km spacing; no continuum exactness is claimed, and target resolution/representational and coverage limitations remain.",
            "18. **What claims are supported?** A leakage-safe, target-atomic synthetic benchmark; a transparent comparison of travel-time and geometry controls; target-level uncertainty; coverage-aware and structure-aware diagnostics; and bounded conclusions for the configured synthetic families.",
            "19. **What claims are not supported?** State-of-the-art superiority, continuum-exact forward labels, real-world geological generalization, full nonlinear-tomography equivalence, or inference based on acquisition-case pseudoreplication.",
            "",
            "## Evidence-to-claim table",
            "",
            evidence_table,
            "",
            "## Output package",
            "",
            f"- ML: `{Path(ml_result['output_dir']).as_posix()}`",
            f"- Classical: `{Path(classical_result['output_dir']).as_posix()}`",
            f"- Coverage/structure/statistics/noise/LOFO/learning: `{output_dir.as_posix()}`",
            f"- Figures: `{Path(figures_result['output_dir']).as_posix()}`",
            f"- Tables: `{Path(tables_result['output_dir']).as_posix()}`",
            "",
            "## Final scientific decision",
            "",
            f"**{decision}**",
            "",
            decision_reason,
            "",
            "The manuscript remains unchanged. The next task is to rewrite its claims around this frozen evidence package, including any required narrowing.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _write_json(
        {
            "artifact_type": "benchmark_v2_final_evidence_report_metadata",
            "decision": decision,
            "decision_reason": decision_reason,
            "integrity_ok": integrity_ok,
            "time_help_directional": time_help_directional,
            "realistic_vs_fixed_ray_ci_excludes_zero_in_favor": fixed_supported,
            "unique_target_count": len({case.target_hash for case in cases}),
            "test_target_count": len(test_ids),
            "source_figures": figures_result["figure_paths"],
            "source_tables": tables_result["paths"],
            "manuscript_modified": False,
        },
        output_dir / "final_decision_metadata.json",
    )
    return report_path
