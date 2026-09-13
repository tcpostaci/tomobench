"""Reference-model fixed-ray classical tomography diagnostics."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tomobench.config import load_settings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import CartesianVelocityGrid3D, TravelTimeRayPath
from tomobench.generation.velocity_grids import save_velocity_grid
from tomobench.simulation.ray_tracing import trace_pseudo_bending_ray
from tomobench.simulation.travel_times import save_travel_time_ray_paths_jsonl
from tomobench.tomography.corpus import (
    DEFAULT_CORPUS_MANIFEST,
    DEFAULT_KNOWN_RAY_CORPUS_OUTPUT_DIR,
    DEFAULT_ML_SUITE_SUMMARY,
    _aggregate,
    _case_observation_path,
    _case_summary_row,
    _case_velocity_grid_path,
    _extract_pca_linear_reference,
    _family_wise_metrics,
    _float_key,
    _optional_float,
    _read_json,
    _resolve_path,
    _write_case_summary_csv,
    _write_json,
    _write_summary_csv,
)
from tomobench.tomography.inversion import (
    _assemble_dense_g_matrix,
    _cell_count_from_metadata,
    _cell_shape_from_metadata,
    _covered_velocity_metrics,
    _load_observation_travel_times,
    _metric_sort_value,
    _ordered_observation_ids,
    _slowness_to_velocity_with_clipping,
    _travel_time_metrics,
    _velocity_bounds_from_grid,
    _velocity_metrics,
    _write_coverage_csv,
    _write_predicted_travel_times,
    cell_centered_velocity_target,
    cell_coverage_diagnostics,
    smoothing_normal_matrix,
    summarize_cell_coverage,
)
from tomobench.tomography.sensitivity import (
    load_sensitivity_records_jsonl,
    save_ray_path_sensitivity_sidecar,
)
from tomobench.utils.paths import get_repo_root

DEFAULT_REFERENCE_RAY_CORPUS_OUTPUT_DIR = Path(
    "outputs/generated/classical_tomography/reference_ray_corpus_v1"
)


@dataclass(frozen=True)
class ReferenceRayCorpusInversionOutputs:
    """Artifacts written by the corpus-scale reference-ray diagnostic."""

    output_dir: Path
    case_summary_csv: Path
    summary_csv: Path
    summary_json: Path
    comparison_json: Path
    comparison_csv: Path


@dataclass(frozen=True)
class ReferenceRayRegularizationSensitivityOutputs:
    """Artifacts written by the Phase 7B reference-ray regularization sensitivity audit."""

    output_dir: Path
    summary_json: Path
    summary_csv: Path
    case_metrics_csv: Path
    family_wise_csv: Path
    tradeoff_csv: Path
    experiment_note: Path


def run_reference_ray_corpus_inversion(
    manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    output_dir: Path = DEFAULT_REFERENCE_RAY_CORPUS_OUTPUT_DIR,
    known_ray_summary_path: Path | None = DEFAULT_KNOWN_RAY_CORPUS_OUTPUT_DIR
    / "known_ray_corpus_summary.json",
    ml_suite_summary_path: Path | None = DEFAULT_ML_SUITE_SUMMARY,
    damping_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    smoothing_values: Sequence[float] = (0.0, 0.01, 0.1, 1.0),
    minimum_coverage_km: float = 1.0e-9,
) -> ReferenceRayCorpusInversionOutputs:
    """Run reference-model fixed-ray perturbation inversion for each corpus case."""
    if not damping_values:
        raise ValueError("damping_values must contain at least one value.")
    if not smoothing_values:
        raise ValueError("smoothing_values must contain at least one value.")

    repo_root = get_repo_root()
    settings = load_settings()
    resolved_manifest_path = _resolve_path(manifest_path, repo_root, None)
    manifest = _read_json(resolved_manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Corpus manifest must contain at least one item.")

    output_dir = _resolve_path(output_dir, repo_root, None)
    cases_dir = output_dir / "cases"
    case_rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError("Corpus manifest item must be a JSON object.")
        case_id = str(item["case_id"])
        case_output_dir = cases_dir / case_id
        case_summary_path = run_reference_ray_case_inversion(
            observations_path=_case_observation_path(item, repo_root, resolved_manifest_path),
            true_velocity_grid_path=_case_velocity_grid_path(
                item,
                repo_root,
                resolved_manifest_path,
            ),
            output_dir=case_output_dir,
            settings=settings,
            manifest_item=item,
            damping_values=damping_values,
            smoothing_values=smoothing_values,
            minimum_coverage_km=minimum_coverage_km,
        )
        case_rows.append(_case_summary_row(item, _read_json(case_summary_path), case_summary_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    case_summary_csv = output_dir / "reference_ray_corpus_case_summary.csv"
    summary_csv = output_dir / "reference_ray_corpus_summary.csv"
    summary_json = output_dir / "reference_ray_corpus_summary.json"
    comparison_json = output_dir / "classical_ml_reference_ray_comparison.json"
    comparison_csv = output_dir / "classical_ml_reference_ray_comparison.csv"
    _write_case_summary_csv(case_summary_csv, case_rows)
    summary = _reference_corpus_summary_payload(
        manifest=manifest,
        manifest_path=resolved_manifest_path,
        output_dir=output_dir,
        case_rows=case_rows,
        damping_values=damping_values,
        smoothing_values=smoothing_values,
        minimum_coverage_km=minimum_coverage_km,
    )
    _write_json(summary_json, summary)
    _write_summary_csv(summary_csv, summary)
    comparison = _classical_ml_comparison_payload(
        reference_summary=summary,
        known_ray_summary_path=known_ray_summary_path,
        ml_suite_summary_path=ml_suite_summary_path,
        repo_root=repo_root,
    )
    _write_json(comparison_json, comparison)
    _write_classical_ml_comparison_csv(comparison_csv, comparison)
    return ReferenceRayCorpusInversionOutputs(
        output_dir=output_dir,
        case_summary_csv=case_summary_csv,
        summary_csv=summary_csv,
        summary_json=summary_json,
        comparison_json=comparison_json,
        comparison_csv=comparison_csv,
    )


def run_reference_ray_regularization_sensitivity(
    manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    reference_case_summary_path: Path = DEFAULT_REFERENCE_RAY_CORPUS_OUTPUT_DIR
    / "reference_ray_corpus_case_summary.csv",
    output_dir: Path = Path(
        "outputs/generated/classical_tomography/reference_ray_regularization_sensitivity_v1"
    ),
    damping_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0),
    smoothing_values: Sequence[float] = (0.0, 0.1, 1.0, 10.0),
    minimum_coverage_km: float = 1.0e-9,
    case_limit: int | None = None,
) -> ReferenceRayRegularizationSensitivityOutputs:
    """Audit reference-ray regularization sensitivity using existing Phase 6 sidecars."""
    parameter_grid = reference_regularization_parameter_grid(damping_values, smoothing_values)
    if minimum_coverage_km < 0.0:
        raise ValueError("minimum_coverage_km must be non-negative.")
    if case_limit is not None and case_limit <= 0:
        raise ValueError("case_limit must be positive when provided.")

    repo_root = get_repo_root()
    manifest_path = _resolve_path(manifest_path, repo_root, None)
    reference_case_summary_path = _resolve_path(reference_case_summary_path, repo_root, None)
    output_dir = _resolve_path(output_dir, repo_root, None)
    manifest = _read_json(manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Corpus manifest must contain at least one item.")
    selected_items = _balanced_case_subset([_dict_item(item) for item in items], case_limit)
    reference_rows_by_case = {
        str(row["case_id"]): row for row in _read_csv_rows(reference_case_summary_path)
    }
    case_metric_rows: list[dict[str, Any]] = []
    for item in selected_items:
        case_id = str(item["case_id"])
        if case_id not in reference_rows_by_case:
            raise ValueError(f"Missing Phase 6 reference case summary row for {case_id}.")
        phase6_summary_path = _resolve_path(
            Path(str(reference_rows_by_case[case_id]["summary_json"])),
            repo_root,
            manifest_path,
        )
        case_metric_rows.extend(
            _reference_sensitivity_rows_for_case(
                item=item,
                phase6_summary_path=phase6_summary_path,
                damping_smoothing_grid=parameter_grid,
                minimum_coverage_km=minimum_coverage_km,
            )
        )

    summary_rows = aggregate_reference_regularization_sensitivity(case_metric_rows)
    family_rows = _family_wise_sensitivity_rows(case_metric_rows)
    audit = select_reference_regularization_audit(summary_rows)
    sample_scope = {
        "case_count": len(selected_items),
        "full_corpus_case_count": len(items),
        "is_full_corpus": len(selected_items) == len(items),
        "case_limit": case_limit,
        "families": sorted({str(item.get("family") or item.get("scenario")) for item in selected_items}),
    }
    summary_payload = {
        "artifact_type": "reference_ray_regularization_sensitivity_summary",
        "experiment_id": "reference_ray_regularization_sensitivity_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest_path": manifest_path.as_posix(),
        "source_reference_case_summary_csv": reference_case_summary_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "input_corpus_id": manifest.get("corpus_id") or manifest.get("batch_id"),
        "diagnostic_type": "reference_model_fixed_ray_regularization_and_clipping_sensitivity",
        "damping_values": sorted({damping for damping, _smoothing in parameter_grid}),
        "smoothing_values": sorted({smoothing for _damping, smoothing in parameter_grid}),
        "parameter_grid": [
            {"damping": damping, "smoothing": smoothing}
            for damping, smoothing in parameter_grid
        ],
        "minimum_coverage_threshold_km": minimum_coverage_km,
        "sample_scope": sample_scope,
        "summary_rows": summary_rows,
        "selection_audit": audit,
        "assumptions": [
            "This Phase 7B audit reuses Phase 6 reference-ray geometry and sensitivity sidecars; it does not retrace rays or alter Phase 6 outputs.",
            "Each lambda/alpha pair is evaluated against the same cell-centered target and same coverage mask per case.",
            "Velocity clipping is counted for every candidate and split into covered and uncovered cells.",
            "This remains synthetic-only fixed-ray reference-model inversion, not full nonlinear iterative tomography.",
        ],
        "limitations": [
            "The sweep covers a bounded regularization grid and should not be described as globally optimized classical tomography.",
            "If sample_scope.is_full_corpus is false, the output is a balanced representative subset diagnostic only.",
            "Phase 6 results are preserved; this sensitivity audit does not replace them unless a separate bug fix is identified.",
        ],
    }

    summary_json = output_dir / "regularization_sensitivity_summary.json"
    summary_csv = output_dir / "regularization_sensitivity_summary.csv"
    case_metrics_csv = output_dir / "regularization_sensitivity_case_metrics.csv"
    family_wise_csv = output_dir / "family_wise_sensitivity.csv"
    tradeoff_csv = output_dir / "rmse_clipping_tradeoff.csv"
    experiment_note = (
        repo_root
        / "wiki/experiments/reference_ray_regularization_sensitivity_v1.md"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary_json, summary_payload)
    _write_csv(summary_csv, summary_rows)
    _write_csv(case_metrics_csv, case_metric_rows)
    _write_csv(family_wise_csv, family_rows)
    _write_csv(tradeoff_csv, _tradeoff_rows(summary_rows))
    _write_reference_regularization_sensitivity_note(
        experiment_note,
        payload=summary_payload,
        summary_json=summary_json,
        summary_csv=summary_csv,
        case_metrics_csv=case_metrics_csv,
        family_wise_csv=family_wise_csv,
        tradeoff_csv=tradeoff_csv,
        repo_root=repo_root,
    )
    return ReferenceRayRegularizationSensitivityOutputs(
        output_dir=output_dir,
        summary_json=summary_json,
        summary_csv=summary_csv,
        case_metrics_csv=case_metrics_csv,
        family_wise_csv=family_wise_csv,
        tradeoff_csv=tradeoff_csv,
        experiment_note=experiment_note,
    )


def run_reference_ray_case_inversion(
    observations_path: Path,
    true_velocity_grid_path: Path,
    output_dir: Path,
    settings: Any | None = None,
    manifest_item: dict[str, Any] | None = None,
    damping_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    smoothing_values: Sequence[float] = (0.0, 0.01, 0.1, 1.0),
    minimum_coverage_km: float = 1.0e-9,
) -> Path:
    """Run one reference-ray prior/perturbation inversion and return its summary path."""
    loaded_settings = settings or load_settings()
    true_grid = _load_velocity_grid(true_velocity_grid_path)
    reference_grid = build_layered_reference_velocity_grid(true_grid, loaded_settings)
    observations = _read_observation_rows(observations_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_grid_path = save_velocity_grid(
        reference_grid,
        output_dir / "reference_layered_velocity_grid.json",
    )
    ray_paths = _trace_reference_ray_paths(observations, reference_grid, loaded_settings, manifest_item)
    ray_paths_path = save_travel_time_ray_paths_jsonl(
        ray_paths,
        output_dir / "reference_ray_paths.jsonl",
        loaded_settings,
    )
    sensitivity_path, sensitivity_metadata_path, validation = save_ray_path_sensitivity_sidecar(
        ray_paths,
        reference_grid,
        output_dir / "reference_ray_cell_sensitivity.jsonl",
        output_dir / "reference_ray_cell_sensitivity.metadata.json",
        loaded_settings,
        source_ray_path_sidecar=ray_paths_path,
    )
    if not validation.passed:
        raise ValueError(
            "Reference-ray sensitivity path lengths failed validation: "
            + "; ".join(validation.errors)
        )

    summary_path = output_dir / "reference_ray_perturbation_summary.json"
    _run_reference_ray_solve(
        sensitivity_path=sensitivity_path,
        sensitivity_metadata_path=sensitivity_metadata_path,
        observations_path=observations_path,
        true_velocity_grid_path=true_velocity_grid_path,
        reference_velocity_grid_path=reference_grid_path,
        output_dir=output_dir,
        summary_path=summary_path,
        damping_values=damping_values,
        smoothing_values=smoothing_values,
        minimum_coverage_km=minimum_coverage_km,
    )
    return summary_path


def build_layered_reference_velocity_grid(
    true_grid: CartesianVelocityGrid3D,
    settings: Any | None = None,
) -> CartesianVelocityGrid3D:
    """Create a configured layered background on the same node coordinates."""
    loaded_settings = settings or load_settings()
    nx = len(true_grid.x_coordinates_km)
    ny = len(true_grid.y_coordinates_km)
    nz = len(true_grid.z_coordinates_km)
    if len(true_grid.p_velocity_km_per_s) != nx * ny * nz:
        raise ValueError("Velocity grid shape does not match flattened velocity count.")
    reference_values: list[float] = []
    for z_km in true_grid.z_coordinates_km:
        reference_values.extend(
            [_configured_layered_velocity_at_depth(loaded_settings, z_km)] * nx * ny
        )
    return CartesianVelocityGrid3D(
        grid_id=f"{true_grid.grid_id}_configured_layered_reference",
        source_velocity_model_id="configured_layered_reference_model",
        x_coordinates_km=true_grid.x_coordinates_km,
        y_coordinates_km=true_grid.y_coordinates_km,
        z_coordinates_km=true_grid.z_coordinates_km,
        p_velocity_km_per_s=tuple(reference_values),
        metadata={
            "artifact_type": "layered_reference_cartesian_p_velocity_grid_3d",
            "coordinate_system": true_grid.metadata.get("coordinate_system", "cartesian_km"),
            "reference_model_role": "configured_layered_background",
            "reference_source": "config/benchmark_config.yaml machine_learning-independent layered velocity settings",
            "source_grid_id": true_grid.grid_id,
            "grid_shape": {"nx": nx, "ny": ny, "nz": nz},
            "layered_model": {
                "depth_boundaries_km": list(
                    loaded_settings.velocity_model_generation.layered.depth_boundaries_km
                ),
                "velocities_km_per_s": list(
                    loaded_settings.velocity_model_generation.layered.velocities_km_per_s
                ),
            },
            "velocity_representation": "node_centered",
            "construction": "configured 1D layered velocity sampled onto target grid coordinates by depth",
            "assumptions": [
                "Reference velocities come from centralized configuration, not from case-specific anomaly/fault/salt/dyke target values.",
                "The true target grid contributes only Cartesian node coordinates and final evaluation targets.",
            ],
        },
    )


def solve_reference_ray_perturbation_slowness(
    g_matrix: np.ndarray,
    observed_travel_times: np.ndarray,
    reference_slowness: np.ndarray,
    damping: float,
    smoothing: float,
    smoothing_ltl: np.ndarray,
) -> np.ndarray:
    """Solve ``t_obs - G s0 = G delta_s`` and return ``s0 + delta_s``."""
    if reference_slowness.ndim != 1 or reference_slowness.shape[0] != g_matrix.shape[1]:
        raise ValueError("reference_slowness length must match G columns.")
    if observed_travel_times.ndim != 1 or observed_travel_times.shape[0] != g_matrix.shape[0]:
        raise ValueError("observed travel-time length must match G rows.")
    if damping < 0.0 or smoothing < 0.0:
        raise ValueError("damping and smoothing must be non-negative.")
    if smoothing_ltl.shape != (g_matrix.shape[1], g_matrix.shape[1]):
        raise ValueError("smoothing_ltl shape must match G columns.")
    residual_travel_times = observed_travel_times - (g_matrix @ reference_slowness)
    normal_matrix = g_matrix.T @ g_matrix
    if damping > 0.0:
        normal_matrix = normal_matrix + damping * np.eye(g_matrix.shape[1])
    if smoothing > 0.0:
        normal_matrix = normal_matrix + smoothing * smoothing_ltl
    delta_slowness = np.linalg.solve(normal_matrix, g_matrix.T @ residual_travel_times)
    return reference_slowness + delta_slowness


def reference_regularization_parameter_grid(
    damping_values: Sequence[float],
    smoothing_values: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    """Return a deterministic lambda/alpha grid for reference-ray sensitivity."""
    if not damping_values:
        raise ValueError("damping_values must contain at least one value.")
    if not smoothing_values:
        raise ValueError("smoothing_values must contain at least one value.")
    damping = tuple(float(value) for value in damping_values)
    smoothing = tuple(float(value) for value in smoothing_values)
    if any(value < 0.0 for value in damping):
        raise ValueError("damping_values must be non-negative.")
    if any(value < 0.0 for value in smoothing):
        raise ValueError("smoothing_values must be non-negative.")
    return tuple((lambda_value, alpha_value) for lambda_value in damping for alpha_value in smoothing)


def aggregate_reference_regularization_sensitivity(
    case_metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate per-case sensitivity rows by lambda/alpha pair."""
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in case_metric_rows:
        key = (float(row["damping"]), float(row["smoothing"]))
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for damping, smoothing in sorted(grouped):
        rows_for_pair = grouped[(damping, smoothing)]
        rows.append(
            {
                "damping": damping,
                "smoothing": smoothing,
                "case_count": len(rows_for_pair),
                "travel_time_rmse_s_mean": _mean(rows_for_pair, "travel_time_rmse_s"),
                "travel_time_mae_s_mean": _mean(rows_for_pair, "travel_time_mae_s"),
                "all_cell_velocity_rmse_km_per_s_mean": _mean(
                    rows_for_pair,
                    "all_cell_velocity_rmse_km_per_s",
                ),
                "all_cell_velocity_mae_km_per_s_mean": _mean(
                    rows_for_pair,
                    "all_cell_velocity_mae_km_per_s",
                ),
                "covered_cell_velocity_rmse_km_per_s_mean": _mean(
                    rows_for_pair,
                    "covered_cell_velocity_rmse_km_per_s",
                ),
                "covered_cell_velocity_mae_km_per_s_mean": _mean(
                    rows_for_pair,
                    "covered_cell_velocity_mae_km_per_s",
                ),
                "clipped_velocity_cell_count_mean": _mean(
                    rows_for_pair,
                    "clipped_velocity_cell_count",
                ),
                "clipped_velocity_cell_percentage_mean": _mean(
                    rows_for_pair,
                    "clipped_velocity_cell_percentage",
                ),
                "covered_clipped_cell_count_mean": _mean(
                    rows_for_pair,
                    "covered_clipped_cell_count",
                ),
                "uncovered_clipped_cell_count_mean": _mean(
                    rows_for_pair,
                    "uncovered_clipped_cell_count",
                ),
                "nonphysical_slowness_count_mean": _mean(
                    rows_for_pair,
                    "nonphysical_slowness_count",
                ),
            }
        )
    return rows


def select_reference_regularization_audit(
    summary_rows: list[dict[str, Any]],
    *,
    phase6_damping: float = 0.01,
    phase6_smoothing: float = 1.0,
    minor_rmse_cost_fraction: float = 0.05,
) -> dict[str, Any]:
    """Select diagnostic regularization and clipping tradeoff notes."""
    if not summary_rows:
        raise ValueError("summary_rows must not be empty.")
    best_rmse = min(
        summary_rows,
        key=lambda row: (
            float(row["covered_cell_velocity_rmse_km_per_s_mean"]),
            float(row["all_cell_velocity_rmse_km_per_s_mean"]),
            float(row["clipped_velocity_cell_percentage_mean"]),
        ),
    )
    lowest_clipping = min(
        summary_rows,
        key=lambda row: (
            float(row["clipped_velocity_cell_percentage_mean"]),
            float(row["covered_cell_velocity_rmse_km_per_s_mean"]),
        ),
    )
    phase6 = next(
        (
            row
            for row in summary_rows
            if float(row["damping"]) == float(phase6_damping)
            and float(row["smoothing"]) == float(phase6_smoothing)
        ),
        None,
    )
    if phase6 is None:
        raise ValueError("Phase 6 lambda/alpha pair is not present in summary_rows.")
    phase6_rmse = float(phase6["covered_cell_velocity_rmse_km_per_s_mean"])
    phase6_clipping = float(phase6["clipped_velocity_cell_percentage_mean"])
    minor_cost_limit = phase6_rmse * (1.0 + minor_rmse_cost_fraction)
    lower_clipping_candidates = [
        row
        for row in summary_rows
        if float(row["clipped_velocity_cell_percentage_mean"]) < phase6_clipping
        and float(row["covered_cell_velocity_rmse_km_per_s_mean"]) <= minor_cost_limit
    ]
    lower_clipping_minor_cost = (
        min(
            lower_clipping_candidates,
            key=lambda row: (
                float(row["clipped_velocity_cell_percentage_mean"]),
                float(row["covered_cell_velocity_rmse_km_per_s_mean"]),
            ),
        )
        if lower_clipping_candidates
        else None
    )
    best_rmse_pair_is_phase6 = (
        float(best_rmse["damping"]) == float(phase6_damping)
        and float(best_rmse["smoothing"]) == float(phase6_smoothing)
    )
    phase6_within_minor_cost_of_best = phase6_rmse <= float(
        best_rmse["covered_cell_velocity_rmse_km_per_s_mean"]
    ) * (1.0 + minor_rmse_cost_fraction)
    if lower_clipping_minor_cost is not None:
        statement = (
            "Another setting reduces clipping with only minor covered-cell RMSE cost; "
            "Phase 6 should be reported alongside this tradeoff rather than silently replaced."
        )
    elif best_rmse_pair_is_phase6:
        statement = "The Phase 6 lambda=0.01, alpha=1.0 setting remains the lowest covered-cell RMSE choice in this sweep."
    elif phase6_within_minor_cost_of_best:
        statement = (
            "The Phase 6 lambda=0.01, alpha=1.0 setting remains reasonable within the bounded sweep, "
            "with no substantially lower-clipping alternative inside the configured minor RMSE-cost band."
        )
    else:
        statement = (
            "The sweep identifies a lower-RMSE setting than Phase 6; this should be treated as a sensitivity finding, "
            "not an automatic replacement of the preserved Phase 6 result."
        )
    return {
        "phase6_reference": _audit_compact_row(phase6),
        "best_covered_cell_rmse": _audit_compact_row(best_rmse),
        "lowest_clipping": _audit_compact_row(lowest_clipping),
        "lower_clipping_with_minor_rmse_cost": _audit_compact_row(lower_clipping_minor_cost)
        if lower_clipping_minor_cost is not None
        else None,
        "minor_rmse_cost_fraction": minor_rmse_cost_fraction,
        "phase6_pair_is_best_covered_rmse": best_rmse_pair_is_phase6,
        "phase6_within_minor_cost_of_best": phase6_within_minor_cost_of_best,
        "statement": statement,
    }


def _configured_layered_velocity_at_depth(settings: Any, depth_km: float) -> float:
    layered = settings.velocity_model_generation.layered
    for index, velocity in enumerate(layered.velocities_km_per_s):
        top_depth = layered.depth_boundaries_km[index]
        bottom_depth = layered.depth_boundaries_km[index + 1]
        if top_depth <= depth_km < bottom_depth:
            return float(velocity)
    if depth_km == layered.depth_boundaries_km[-1]:
        return float(layered.velocities_km_per_s[-1])
    raise ValueError(f"Depth {depth_km} km is outside the configured layered model.")


def _run_reference_ray_solve(
    sensitivity_path: Path,
    sensitivity_metadata_path: Path,
    observations_path: Path,
    true_velocity_grid_path: Path,
    reference_velocity_grid_path: Path,
    output_dir: Path,
    summary_path: Path,
    damping_values: Sequence[float],
    smoothing_values: Sequence[float],
    minimum_coverage_km: float,
) -> None:
    sensitivity_metadata = _read_json(sensitivity_metadata_path)
    sensitivity_records = load_sensitivity_records_jsonl(sensitivity_path)
    observations = _load_observation_travel_times(observations_path)
    true_grid = _load_velocity_grid(true_velocity_grid_path)
    reference_grid = _load_velocity_grid(reference_velocity_grid_path)
    observation_ids = _ordered_observation_ids(sensitivity_records, observations)
    cell_count = _cell_count_from_metadata(sensitivity_metadata)
    cell_shape = _cell_shape_from_metadata(sensitivity_metadata, true_grid)
    g_matrix = _assemble_dense_g_matrix(sensitivity_records, observation_ids, cell_count)
    coverage = cell_coverage_diagnostics(sensitivity_records, cell_shape)
    coverage_summary = summarize_cell_coverage(coverage, minimum_coverage_km)
    covered_mask = np.array(
        [item["total_path_length_coverage_km"] >= minimum_coverage_km for item in coverage],
        dtype=bool,
    )
    observed_times = np.array([observations[observation_id] for observation_id in observation_ids])
    true_target_velocity = np.array(cell_centered_velocity_target(true_grid))
    reference_target_velocity = np.array(cell_centered_velocity_target(reference_grid))
    reference_slowness = 1.0 / reference_target_velocity
    smoothing_ltl = smoothing_normal_matrix(cell_shape)
    bounds = _velocity_bounds_from_grid(true_grid)
    candidates = [
        _solve_reference_candidate(
            g_matrix=g_matrix,
            observed_travel_times=observed_times,
            target_velocity=true_target_velocity,
            reference_slowness=reference_slowness,
            covered_mask=covered_mask,
            damping=float(damping),
            smoothing=float(smoothing),
            smoothing_ltl=smoothing_ltl,
            velocity_bounds=bounds,
        )
        for damping in damping_values
        for smoothing in smoothing_values
    ]
    selected = min(
        candidates,
        key=lambda result: (
            _metric_sort_value(
                result["covered_cell_reconstruction_metrics"]["velocity_rmse_km_per_s"]
            ),
            result["residual_metrics"]["travel_time_rmse_s"],
            result["all_cell_reconstruction_metrics"]["velocity_rmse_km_per_s"],
        ),
    )
    _write_coverage_csv(output_dir / "reference_ray_cell_coverage.csv", coverage, minimum_coverage_km)
    _write_reference_cell_predictions(
        output_dir / "reference_ray_cell_velocity_predictions.csv",
        selected=selected,
        target_velocity=true_target_velocity,
        reference_velocity=reference_target_velocity,
        coverage=coverage,
        minimum_coverage_km=minimum_coverage_km,
    )
    _write_predicted_travel_times(
        output_dir / "reference_ray_predicted_travel_times.csv",
        observation_ids,
        observed_times,
        selected,
    )
    _write_reference_summary(
        summary_path,
        sensitivity_path=sensitivity_path,
        sensitivity_metadata_path=sensitivity_metadata_path,
        observations_path=observations_path,
        true_velocity_grid_path=true_velocity_grid_path,
        reference_velocity_grid_path=reference_velocity_grid_path,
        sensitivity_metadata=sensitivity_metadata,
        observation_count=len(observation_ids),
        cell_count=cell_count,
        damping_values=damping_values,
        smoothing_values=smoothing_values,
        minimum_coverage_km=minimum_coverage_km,
        coverage_summary=coverage_summary,
        reference_velocity_metrics=_velocity_metrics(true_target_velocity, reference_target_velocity),
        candidate_results=candidates,
        selected=selected,
        bounds=bounds,
    )


def _solve_reference_candidate(
    g_matrix: np.ndarray,
    observed_travel_times: np.ndarray,
    target_velocity: np.ndarray,
    reference_slowness: np.ndarray,
    covered_mask: np.ndarray,
    damping: float,
    smoothing: float,
    smoothing_ltl: np.ndarray,
    velocity_bounds: tuple[float, float],
) -> dict[str, Any]:
    slowness = solve_reference_ray_perturbation_slowness(
        g_matrix,
        observed_travel_times,
        reference_slowness,
        damping,
        smoothing,
        smoothing_ltl,
    )
    predicted_times = g_matrix @ slowness
    velocity, nonphysical_count, clipped_count = _slowness_to_velocity_with_clipping(
        slowness,
        velocity_bounds,
    )
    clipping = _reference_clipping_counts(slowness, velocity_bounds, covered_mask)
    residuals = predicted_times - observed_travel_times
    return {
        "damping": damping,
        "smoothing": smoothing,
        "slowness_s_per_km": slowness.tolist(),
        "delta_slowness_s_per_km": (slowness - reference_slowness).tolist(),
        "velocity_km_per_s": velocity.tolist(),
        "predicted_travel_times_s": predicted_times.tolist(),
        "travel_time_residuals_s": residuals.tolist(),
        "nonphysical_slowness_count": nonphysical_count,
        "clipped_velocity_cell_count": clipped_count,
        "covered_clipped_cell_count": clipping["covered_clipped_cell_count"],
        "uncovered_clipped_cell_count": clipping["uncovered_clipped_cell_count"],
        "clipped_velocity_cell_percentage": clipping["clipped_velocity_cell_percentage"],
        "residual_metrics": _travel_time_metrics(observed_travel_times, predicted_times, residuals),
        "all_cell_reconstruction_metrics": _velocity_metrics(target_velocity, velocity),
        "covered_cell_reconstruction_metrics": _covered_velocity_metrics(
            target_velocity,
            velocity,
            covered_mask,
        ),
    }


def _reference_sensitivity_rows_for_case(
    *,
    item: dict[str, Any],
    phase6_summary_path: Path,
    damping_smoothing_grid: tuple[tuple[float, float], ...],
    minimum_coverage_km: float,
) -> list[dict[str, Any]]:
    phase6_summary = _read_json(phase6_summary_path)
    source_files = phase6_summary["source_files"]
    sensitivity_metadata_path = Path(str(source_files["reference_sensitivity_metadata"]))
    sensitivity_path = Path(str(source_files["reference_sensitivity_sidecar"]))
    observations_path = Path(str(source_files["observed_travel_times"]))
    true_velocity_grid_path = Path(str(source_files["true_target_velocity_grid"]))
    reference_velocity_grid_path = Path(str(source_files["reference_velocity_grid"]))
    sensitivity_metadata = _read_json(sensitivity_metadata_path)
    sensitivity_records = load_sensitivity_records_jsonl(sensitivity_path)
    observations = _load_observation_travel_times(observations_path)
    true_grid = _load_velocity_grid(true_velocity_grid_path)
    reference_grid = _load_velocity_grid(reference_velocity_grid_path)
    observation_ids = _ordered_observation_ids(sensitivity_records, observations)
    cell_count = _cell_count_from_metadata(sensitivity_metadata)
    cell_shape = _cell_shape_from_metadata(sensitivity_metadata, true_grid)
    g_matrix = _assemble_dense_g_matrix(sensitivity_records, observation_ids, cell_count)
    coverage = cell_coverage_diagnostics(sensitivity_records, cell_shape)
    covered_mask = np.array(
        [item["total_path_length_coverage_km"] >= minimum_coverage_km for item in coverage],
        dtype=bool,
    )
    observed_times = np.array([observations[observation_id] for observation_id in observation_ids])
    true_target_velocity = np.array(cell_centered_velocity_target(true_grid))
    reference_target_velocity = np.array(cell_centered_velocity_target(reference_grid))
    reference_slowness = 1.0 / reference_target_velocity
    smoothing_ltl = smoothing_normal_matrix(cell_shape)
    bounds = _velocity_bounds_from_grid(true_grid)
    coverage_summary = summarize_cell_coverage(coverage, minimum_coverage_km)
    rows: list[dict[str, Any]] = []
    for damping, smoothing in damping_smoothing_grid:
        result = _solve_reference_candidate(
            g_matrix=g_matrix,
            observed_travel_times=observed_times,
            target_velocity=true_target_velocity,
            reference_slowness=reference_slowness,
            covered_mask=covered_mask,
            damping=damping,
            smoothing=smoothing,
            smoothing_ltl=smoothing_ltl,
            velocity_bounds=bounds,
        )
        residual = result["residual_metrics"]
        all_cell = result["all_cell_reconstruction_metrics"]
        covered_cell = result["covered_cell_reconstruction_metrics"]
        rows.append(
            {
                "case_id": str(item["case_id"]),
                "family": str(item.get("family") or item.get("scenario") or "unknown"),
                "damping": damping,
                "smoothing": smoothing,
                "travel_time_rmse_s": residual["travel_time_rmse_s"],
                "travel_time_mae_s": residual["travel_time_mae_s"],
                "travel_time_bias_s": residual["travel_time_bias_s"],
                "all_cell_velocity_rmse_km_per_s": all_cell["velocity_rmse_km_per_s"],
                "all_cell_velocity_mae_km_per_s": all_cell["velocity_mae_km_per_s"],
                "covered_cell_velocity_rmse_km_per_s": covered_cell[
                    "velocity_rmse_km_per_s"
                ],
                "covered_cell_velocity_mae_km_per_s": covered_cell["velocity_mae_km_per_s"],
                "covered_cell_count": coverage_summary["covered_cell_count"],
                "uncovered_cell_count": coverage_summary["uncovered_cell_count"],
                "coverage_fraction": coverage_summary["coverage_fraction"],
                "cell_count": cell_count,
                "clipped_velocity_cell_count": result["clipped_velocity_cell_count"],
                "clipped_velocity_cell_percentage": result[
                    "clipped_velocity_cell_percentage"
                ],
                "covered_clipped_cell_count": result["covered_clipped_cell_count"],
                "uncovered_clipped_cell_count": result["uncovered_clipped_cell_count"],
                "nonphysical_slowness_count": result["nonphysical_slowness_count"],
                "phase6_summary_json": phase6_summary_path.as_posix(),
            }
        )
    return rows


def _reference_clipping_counts(
    slowness: np.ndarray,
    velocity_bounds: tuple[float, float],
    covered_mask: np.ndarray,
) -> dict[str, float | int]:
    lower_velocity, upper_velocity = velocity_bounds
    min_slowness = 1.0 / upper_velocity
    max_slowness = 1.0 / lower_velocity
    clipped_mask = ~np.isclose(slowness, np.clip(slowness, min_slowness, max_slowness))
    clipped_count = int(np.count_nonzero(clipped_mask))
    covered_clipped = int(np.count_nonzero(clipped_mask & covered_mask))
    uncovered_clipped = int(np.count_nonzero(clipped_mask & ~covered_mask))
    return {
        "clipped_velocity_cell_count": clipped_count,
        "clipped_velocity_cell_percentage": clipped_count / len(slowness)
        if len(slowness)
        else 0.0,
        "covered_clipped_cell_count": covered_clipped,
        "uncovered_clipped_cell_count": uncovered_clipped,
    }


def _family_wise_sensitivity_rows(case_metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, str], list[dict[str, Any]]] = {}
    for row in case_metric_rows:
        key = (float(row["damping"]), float(row["smoothing"]), str(row["family"]))
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for damping, smoothing, family in sorted(grouped):
        family_rows = grouped[(damping, smoothing, family)]
        rows.append(
            {
                "damping": damping,
                "smoothing": smoothing,
                "family": family,
                "case_count": len(family_rows),
                "travel_time_rmse_s_mean": _mean(family_rows, "travel_time_rmse_s"),
                "travel_time_mae_s_mean": _mean(family_rows, "travel_time_mae_s"),
                "all_cell_velocity_rmse_km_per_s_mean": _mean(
                    family_rows,
                    "all_cell_velocity_rmse_km_per_s",
                ),
                "all_cell_velocity_mae_km_per_s_mean": _mean(
                    family_rows,
                    "all_cell_velocity_mae_km_per_s",
                ),
                "covered_cell_velocity_rmse_km_per_s_mean": _mean(
                    family_rows,
                    "covered_cell_velocity_rmse_km_per_s",
                ),
                "covered_cell_velocity_mae_km_per_s_mean": _mean(
                    family_rows,
                    "covered_cell_velocity_mae_km_per_s",
                ),
                "clipped_velocity_cell_count_mean": _mean(
                    family_rows,
                    "clipped_velocity_cell_count",
                ),
                "clipped_velocity_cell_percentage_mean": _mean(
                    family_rows,
                    "clipped_velocity_cell_percentage",
                ),
                "covered_clipped_cell_count_mean": _mean(
                    family_rows,
                    "covered_clipped_cell_count",
                ),
                "uncovered_clipped_cell_count_mean": _mean(
                    family_rows,
                    "uncovered_clipped_cell_count",
                ),
            }
        )
    return rows


def _tradeoff_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "damping": row["damping"],
            "smoothing": row["smoothing"],
            "covered_cell_velocity_rmse_km_per_s_mean": row[
                "covered_cell_velocity_rmse_km_per_s_mean"
            ],
            "all_cell_velocity_rmse_km_per_s_mean": row[
                "all_cell_velocity_rmse_km_per_s_mean"
            ],
            "travel_time_rmse_s_mean": row["travel_time_rmse_s_mean"],
            "clipped_velocity_cell_count_mean": row["clipped_velocity_cell_count_mean"],
            "clipped_velocity_cell_percentage_mean": row[
                "clipped_velocity_cell_percentage_mean"
            ],
        }
        for row in summary_rows
    ]


def _trace_reference_ray_paths(
    observations: tuple[dict[str, Any], ...],
    reference_grid: CartesianVelocityGrid3D,
    settings: Any,
    manifest_item: dict[str, Any] | None,
) -> tuple[TravelTimeRayPath, ...]:
    ray_paths: list[TravelTimeRayPath] = []
    case_id = str(manifest_item.get("case_id")) if manifest_item else None
    for row in observations:
        source = Point3D(
            x_km=float(row["source_x_km"]),
            y_km=float(row["source_y_km"]),
            z_km=float(row["source_z_km"]),
        )
        receiver = Point3D(
            x_km=float(row["receiver_x_km"]),
            y_km=float(row["receiver_y_km"]),
            z_km=float(row["receiver_z_km"]),
        )
        ray = trace_pseudo_bending_ray(
            source=source,
            receiver=receiver,
            velocity_grid=reference_grid,
            settings=settings.pseudo_bending_solver,
        )
        ray_paths.append(
            TravelTimeRayPath(
                observation_id=str(row["observation_id"]),
                earthquake_id=str(row.get("earthquake_id") or ""),
                station_id=str(row.get("station_id") or ""),
                source=source,
                receiver=receiver,
                ray_path=ray.ray_path,
                travel_time_s=ray.travel_time_s,
                path_length_km=ray.path_length_km,
                phase=str(row.get("phase") or settings.simulation.phase),
                simulation_method="reference_model_pseudo_bending_3d",
                simulator_solver_type="cartesian_grid_pseudo_bending_fixed_reference_model",
                station_configuration_id=str(
                    manifest_item.get("station_configuration_path") if manifest_item else ""
                ),
                earthquake_configuration_id=str(
                    manifest_item.get("earthquake_configuration_path") if manifest_item else ""
                ),
                velocity_model_id=reference_grid.source_velocity_model_id,
                simulation_id=f"reference_ray_{case_id or 'case'}",
                case_id=case_id,
                converged=ray.converged,
                iteration_count=ray.iteration_count,
            )
        )
    return tuple(ray_paths)


def _write_reference_cell_predictions(
    path: Path,
    selected: dict[str, Any],
    target_velocity: np.ndarray,
    reference_velocity: np.ndarray,
    coverage: tuple[dict[str, Any], ...],
    minimum_coverage_km: float,
) -> None:
    fieldnames = (
        "cell_index",
        "target_velocity_km_per_s",
        "reference_velocity_km_per_s",
        "predicted_velocity_km_per_s",
        "predicted_slowness_s_per_km",
        "delta_slowness_s_per_km",
        "velocity_error_km_per_s",
        "total_path_length_coverage_km",
        "observation_touch_count",
        "covered_by_threshold",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell_index, target in enumerate(target_velocity.tolist()):
            coverage_item = coverage[cell_index]
            coverage_km = float(coverage_item["total_path_length_coverage_km"])
            predicted = float(selected["velocity_km_per_s"][cell_index])
            writer.writerow(
                {
                    "cell_index": cell_index,
                    "target_velocity_km_per_s": target,
                    "reference_velocity_km_per_s": float(reference_velocity[cell_index]),
                    "predicted_velocity_km_per_s": predicted,
                    "predicted_slowness_s_per_km": selected["slowness_s_per_km"][cell_index],
                    "delta_slowness_s_per_km": selected["delta_slowness_s_per_km"][cell_index],
                    "velocity_error_km_per_s": predicted - target,
                    "total_path_length_coverage_km": coverage_km,
                    "observation_touch_count": coverage_item["observation_touch_count"],
                    "covered_by_threshold": coverage_km >= minimum_coverage_km,
                }
            )


def _write_reference_summary(
    path: Path,
    sensitivity_path: Path,
    sensitivity_metadata_path: Path,
    observations_path: Path,
    true_velocity_grid_path: Path,
    reference_velocity_grid_path: Path,
    sensitivity_metadata: dict[str, Any],
    observation_count: int,
    cell_count: int,
    damping_values: Sequence[float],
    smoothing_values: Sequence[float],
    minimum_coverage_km: float,
    coverage_summary: dict[str, Any],
    reference_velocity_metrics: dict[str, float],
    candidate_results: list[dict[str, Any]],
    selected: dict[str, Any],
    bounds: tuple[float, float],
) -> None:
    compact_candidates = [
        {
            "damping": result["damping"],
            "smoothing": result["smoothing"],
            "residual_metrics": result["residual_metrics"],
            "all_cell_reconstruction_metrics": result["all_cell_reconstruction_metrics"],
            "covered_cell_reconstruction_metrics": result[
                "covered_cell_reconstruction_metrics"
            ],
            "nonphysical_slowness_count": result["nonphysical_slowness_count"],
            "clipped_velocity_cell_count": result["clipped_velocity_cell_count"],
            "clipped_velocity_cell_percentage": result[
                "clipped_velocity_cell_percentage"
            ],
            "covered_clipped_cell_count": result["covered_clipped_cell_count"],
            "uncovered_clipped_cell_count": result["uncovered_clipped_cell_count"],
        }
        for result in candidate_results
    ]
    _write_json(
        path,
        {
            "artifact_type": "reference_model_fixed_ray_perturbation_inversion_diagnostic",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "source_files": {
                "reference_sensitivity_sidecar": sensitivity_path.as_posix(),
                "reference_sensitivity_metadata": sensitivity_metadata_path.as_posix(),
                "observed_travel_times": observations_path.as_posix(),
                "true_target_velocity_grid": true_velocity_grid_path.as_posix(),
                "reference_velocity_grid": reference_velocity_grid_path.as_posix(),
            },
            "simulation_method": sensitivity_metadata.get("simulation_method"),
            "diagnostic_type": "reference_model_fixed_ray_damped_smoothed_perturbation",
            "normal_equation": "(G_ref.T @ G_ref + lambda * I + alpha * L.T @ L) @ delta_s = G_ref.T @ (t_observed - G_ref @ s0)",
            "observation_count": observation_count,
            "cell_count": cell_count,
            "damping_values": [float(value) for value in damping_values],
            "smoothing_values": [float(value) for value in smoothing_values],
            "selected_damping": selected["damping"],
            "selected_smoothing": selected["smoothing"],
            "minimum_coverage_threshold_km": minimum_coverage_km,
            "coverage_summary": coverage_summary,
            "velocity_bounds_km_per_s": list(bounds),
            "reference_model_all_cell_metrics": reference_velocity_metrics,
            "residual_metrics": selected["residual_metrics"],
            "all_cell_reconstruction_metrics": selected["all_cell_reconstruction_metrics"],
            "covered_cell_reconstruction_metrics": selected[
                "covered_cell_reconstruction_metrics"
            ],
            "nonphysical_slowness_count": selected["nonphysical_slowness_count"],
            "clipped_velocity_cell_count": selected["clipped_velocity_cell_count"],
            "candidate_summaries": compact_candidates,
            "target_comparison_contract": {
                "classical_target": "cell_centered_velocity_from_node_grid_arithmetic_cell_means",
                "reference_model": "configured layered node grid sampled onto the target grid coordinates",
                "sensitivity_representation": "reference fixed-ray cell path lengths between adjacent grid nodes",
            },
            "assumptions": [
                "Observed travel times come from the true pseudo-bending corpus.",
                "Reference rays are traced through the configured layered background, not the case-specific anomaly/fault/salt/dyke target values.",
                "The inversion solves for slowness perturbations around the reference model.",
                "This is a fairer classical baseline than known-ray inversion, but it is not a full nonlinear iterative tomography workflow.",
                "No real-data validation or fair ML superiority claim is made by this artifact.",
            ],
        },
    )


def _reference_corpus_summary_payload(
    manifest: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
    case_rows: list[dict[str, Any]],
    damping_values: Sequence[float],
    smoothing_values: Sequence[float],
    minimum_coverage_km: float,
) -> dict[str, Any]:
    families = sorted({str(row["family"]) for row in case_rows})
    return {
        "artifact_type": "reference_ray_coverage_smoothing_corpus_summary",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest_path": manifest_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "input_corpus_id": manifest.get("corpus_id") or manifest.get("batch_id"),
        "diagnostic_type": "reference_model_fixed_ray_damped_smoothed_perturbation_corpus",
        "sample_count": len(case_rows),
        "family_list": families,
        "damping_values": [float(value) for value in damping_values],
        "smoothing_values": [float(value) for value in smoothing_values],
        "minimum_coverage_threshold_km": float(minimum_coverage_km),
        "selected_damping_distribution": _value_distribution(case_rows, "selected_damping"),
        "selected_smoothing_distribution": _value_distribution(case_rows, "selected_smoothing"),
        "selected_lambda_alpha_distribution": _parameter_pair_distribution(case_rows),
        "travel_time_rmse_s": _aggregate(case_rows, "travel_time_rmse_s"),
        "travel_time_mae_s": _aggregate(case_rows, "travel_time_mae_s"),
        "all_cell_velocity_rmse_km_per_s": _aggregate(
            case_rows,
            "all_cell_velocity_rmse_km_per_s",
        ),
        "all_cell_velocity_mae_km_per_s": _aggregate(
            case_rows,
            "all_cell_velocity_mae_km_per_s",
        ),
        "covered_cell_velocity_rmse_km_per_s": _aggregate(
            case_rows,
            "covered_cell_velocity_rmse_km_per_s",
        ),
        "covered_cell_velocity_mae_km_per_s": _aggregate(
            case_rows,
            "covered_cell_velocity_mae_km_per_s",
        ),
        "coverage_fraction": _aggregate(case_rows, "coverage_fraction"),
        "clipped_velocity_cell_count": _aggregate(case_rows, "clipped_velocity_cell_count"),
        "nonphysical_slowness_count": _aggregate(case_rows, "nonphysical_slowness_count"),
        "family_wise_metrics": _family_wise_metrics(case_rows),
        "case_summary_csv": (output_dir / "reference_ray_corpus_case_summary.csv").as_posix(),
        "summary_csv": (output_dir / "reference_ray_corpus_summary.csv").as_posix(),
        "target_comparison_contract": {
            "classical_target": "cell_centered_velocity_from_node_grid_arithmetic_cell_means",
            "ml_velocity_metric_domain": "node-centered unless explicitly converted",
        },
        "assumptions": [
            "Reference rays are traced through a configured layered background sampled onto each case grid.",
            "The inversion solves slowness perturbations around the reference model using fixed reference geometry.",
            "This is synthetic-only and not full nonlinear iterative tomography.",
            "No claim that ML improves over classical tomography should be made until these metrics are reviewed.",
        ],
    }


def _classical_ml_comparison_payload(
    reference_summary: dict[str, Any],
    known_ray_summary_path: Path | None,
    ml_suite_summary_path: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    known_summary = None
    if known_ray_summary_path is not None:
        resolved_known = _resolve_path(known_ray_summary_path, repo_root, None)
        if resolved_known.is_file():
            known_summary = _read_json(resolved_known)
    ml_reference = None
    if ml_suite_summary_path is not None:
        resolved_ml = _resolve_path(ml_suite_summary_path, repo_root, None)
        if resolved_ml.is_file():
            ml_reference = _extract_pca_linear_reference(_read_json(resolved_ml), resolved_ml)
    return {
        "artifact_type": "classical_reference_ray_known_ray_ml_comparison_caveated",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "known_ray_reference": _compact_classical_summary(known_summary),
        "reference_ray_reference": _compact_classical_summary(reference_summary),
        "ml_reference": ml_reference,
        "comparison_contract": {
            "known_ray_velocity_metric_domain": "cell-centered derived Cartesian cells",
            "reference_ray_velocity_metric_domain": "cell-centered derived Cartesian cells",
            "ml_velocity_metric_domain": "node-centered full Cartesian velocity grid",
            "known_vs_reference_shared_classical_contract": True,
            "classical_vs_ml_shared_metric_contract": False,
            "interpretation": (
                "Known-ray and reference-ray classical metrics share the same "
                "cell-centered contract. ML metrics are included as context under "
                "their existing node-centered target contract."
            ),
        },
        "scientific_caution": [
            "The known-ray result is optimistic because it uses true-model ray geometry.",
            "The reference-ray result is fairer because geometry is traced through a background model, but it is not nonlinear iterative tomography.",
            "The selected ML model reference is not under the same velocity metric contract unless converted.",
            "Do not claim ML superiority from this artifact without further review.",
        ],
    }


def _write_classical_ml_comparison_csv(path: Path, comparison: dict[str, Any]) -> None:
    known = comparison.get("known_ray_reference") or {}
    reference = comparison["reference_ray_reference"]
    ml = comparison.get("ml_reference") or {}
    rows = []
    for metric_key, ml_key in (
        ("all_cell_velocity_rmse_km_per_s", "fixed_test_rmse_node_velocity_km_per_s"),
        ("covered_cell_velocity_rmse_km_per_s", "fixed_test_rmse_node_velocity_km_per_s"),
        ("all_cell_velocity_mae_km_per_s", "fixed_test_mae_node_velocity_km_per_s"),
        ("covered_cell_velocity_mae_km_per_s", "fixed_test_mae_node_velocity_km_per_s"),
    ):
        rows.append(
            {
                "metric": metric_key,
                "known_ray_mean": _mean_metric(known, metric_key),
                "reference_ray_mean": _mean_metric(reference, metric_key),
                "ml_fixed_test_node_centered": ml.get(ml_key),
                "classical_vs_ml_shared_metric_contract": False,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _compact_classical_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "sample_count": summary.get("sample_count"),
        "diagnostic_type": summary.get("diagnostic_type"),
        "travel_time_rmse_s": summary.get("travel_time_rmse_s"),
        "travel_time_mae_s": summary.get("travel_time_mae_s"),
        "all_cell_velocity_rmse_km_per_s": summary.get(
            "all_cell_velocity_rmse_km_per_s"
        ),
        "all_cell_velocity_mae_km_per_s": summary.get("all_cell_velocity_mae_km_per_s"),
        "covered_cell_velocity_rmse_km_per_s": summary.get(
            "covered_cell_velocity_rmse_km_per_s"
        ),
        "covered_cell_velocity_mae_km_per_s": summary.get(
            "covered_cell_velocity_mae_km_per_s"
        ),
        "coverage_fraction": summary.get("coverage_fraction"),
        "clipped_velocity_cell_count": summary.get("clipped_velocity_cell_count"),
    }


def _write_reference_regularization_sensitivity_note(
    path: Path,
    *,
    payload: dict[str, Any],
    summary_json: Path,
    summary_csv: Path,
    case_metrics_csv: Path,
    family_wise_csv: Path,
    tradeoff_csv: Path,
    repo_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audit = payload["selection_audit"]
    phase6 = audit["phase6_reference"]
    best = audit["best_covered_cell_rmse"]
    lowest_clipping = audit["lowest_clipping"]
    scope = payload["sample_scope"]
    lines = [
        "# Experiment: Reference-Ray Regularization Sensitivity v1",
        "",
        "Experiment ID: `reference_ray_regularization_sensitivity_v1`",
        "",
        "## Objective",
        "",
        (
            "Audit whether the Phase 6 reference-ray fixed-ray baseline was materially affected "
            "by one arbitrary damping and smoothing choice, with explicit clipping diagnostics."
        ),
        "",
        "## Inputs Used",
        "",
        f"- Input corpus ID: `{payload['input_corpus_id']}`",
        f"- Source manifest: `{payload['source_manifest_path']}`",
        f"- Source Phase 6 case summary: `{payload['source_reference_case_summary_csv']}`",
        f"- Case count: `{scope['case_count']}` of `{scope['full_corpus_case_count']}`",
        f"- Full corpus: `{str(scope['is_full_corpus']).lower()}`",
        "",
        "## Selection Audit",
        "",
        audit["statement"],
        "",
        f"- Phase 6 pair: `lambda={phase6['damping']}`, `alpha={phase6['smoothing']}`; "
        f"covered-cell RMSE `{phase6['covered_cell_velocity_rmse_km_per_s_mean']:.6f}` km/s; "
        f"clipping `{phase6['clipped_velocity_cell_percentage_mean']:.6f}`",
        f"- Best covered-cell RMSE pair: `lambda={best['damping']}`, `alpha={best['smoothing']}`; "
        f"covered-cell RMSE `{best['covered_cell_velocity_rmse_km_per_s_mean']:.6f}` km/s; "
        f"clipping `{best['clipped_velocity_cell_percentage_mean']:.6f}`",
        f"- Lowest clipping pair: `lambda={lowest_clipping['damping']}`, "
        f"`alpha={lowest_clipping['smoothing']}`; covered-cell RMSE "
        f"`{lowest_clipping['covered_cell_velocity_rmse_km_per_s_mean']:.6f}` km/s; "
        f"clipping `{lowest_clipping['clipped_velocity_cell_percentage_mean']:.6f}`",
        "",
        "## Outputs Created",
        "",
        f"- Summary JSON: `{_relative_to_repo(summary_json, repo_root)}`",
        f"- Summary CSV: `{_relative_to_repo(summary_csv, repo_root)}`",
        f"- Case metrics CSV: `{_relative_to_repo(case_metrics_csv, repo_root)}`",
        f"- Family-wise CSV: `{_relative_to_repo(family_wise_csv, repo_root)}`",
        f"- Tradeoff CSV: `{_relative_to_repo(tradeoff_csv, repo_root)}`",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {limitation}" for limitation in payload["limitations"])
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            (
                "Use this as a sensitivity audit when discussing Phase 6; do not replace the "
                "preserved Phase 6 baseline unless a separate bug is found."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _balanced_case_subset(
    items: list[dict[str, Any]],
    case_limit: int | None,
) -> list[dict[str, Any]]:
    if case_limit is None or case_limit >= len(items):
        return items
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get("family") or item.get("scenario") or "unknown"), []).append(
            item
        )
    selected: list[dict[str, Any]] = []
    family_names = sorted(grouped)
    cursor = 0
    while len(selected) < case_limit:
        family = family_names[cursor % len(family_names)]
        family_items = grouped[family]
        item_index = cursor // len(family_names)
        if item_index < len(family_items):
            selected.append(family_items[item_index])
        cursor += 1
        if cursor > len(items) * len(family_names):
            break
    return selected[:case_limit]


def _audit_compact_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "damping": float(row["damping"]),
        "smoothing": float(row["smoothing"]),
        "travel_time_rmse_s_mean": float(row["travel_time_rmse_s_mean"]),
        "all_cell_velocity_rmse_km_per_s_mean": float(
            row["all_cell_velocity_rmse_km_per_s_mean"]
        ),
        "covered_cell_velocity_rmse_km_per_s_mean": float(
            row["covered_cell_velocity_rmse_km_per_s_mean"]
        ),
        "clipped_velocity_cell_count_mean": float(row["clipped_velocity_cell_count_mean"]),
        "clipped_velocity_cell_percentage_mean": float(
            row["clipped_velocity_cell_percentage_mean"]
        ),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and not np.isnan(float(row[key]))
    ]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _dict_item(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Corpus manifest item must be a JSON object.")
    return dict(value)


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_observation_rows(path: Path) -> tuple[dict[str, Any], ...]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _load_velocity_grid(path: Path) -> CartesianVelocityGrid3D:
    payload = _read_json(path)
    return CartesianVelocityGrid3D(
        grid_id=str(payload["grid_id"]),
        source_velocity_model_id=str(payload["source_velocity_model_id"]),
        x_coordinates_km=tuple(float(value) for value in payload["x_coordinates_km"]),
        y_coordinates_km=tuple(float(value) for value in payload["y_coordinates_km"]),
        z_coordinates_km=tuple(float(value) for value in payload["z_coordinates_km"]),
        p_velocity_km_per_s=tuple(float(value) for value in payload["p_velocity_km_per_s"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _value_distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(_float_key(row[key]) for row in rows).items()))


def _parameter_pair_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(
        f"lambda={_float_key(row['selected_damping'])},alpha={_float_key(row['selected_smoothing'])}"
        for row in rows
    )
    return dict(sorted(counter.items()))


def _mean_metric(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, dict):
        return _optional_float(value.get("mean"))
    return None
