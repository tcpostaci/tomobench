"""Known-ray damped least-squares travel-time inversion diagnostics."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tomobench.domain.schemas import CartesianVelocityGrid3D
from tomobench.evaluation.metrics import mean_absolute_error, root_mean_squared_error
from tomobench.tomography.sensitivity import load_sensitivity_records_jsonl


@dataclass(frozen=True)
class KnownRayInversionOutputs:
    """Artifacts written by one known-ray damped least-squares diagnostic."""

    output_dir: Path
    summary_json: Path
    cell_velocity_csv: Path
    predicted_travel_times_csv: Path


@dataclass(frozen=True)
class CoverageAwareKnownRayInversionOutputs:
    """Artifacts written by one coverage-aware known-ray inversion diagnostic."""

    output_dir: Path
    summary_json: Path
    cell_velocity_csv: Path
    predicted_travel_times_csv: Path
    coverage_csv: Path


@dataclass(frozen=True)
class KnownRayPerturbationInversionOutputs:
    """Artifacts written by the corrected known-ray perturbation diagnostic."""

    output_dir: Path
    summary_json: Path
    cell_velocity_csv: Path
    predicted_travel_times_csv: Path
    coverage_csv: Path


def run_known_ray_inversion_diagnostic(
    sensitivity_path: Path,
    sensitivity_metadata_path: Path,
    observations_path: Path,
    velocity_grid_path: Path,
    output_dir: Path,
    damping_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
) -> KnownRayInversionOutputs:
    """Run a known-ray damped least-squares inversion for one generated case.

    This diagnostic uses the persisted pseudo-bending ray geometry from the true
    synthetic model. It is therefore an optimistic baseline and should not be
    described as the final fair classical tomography comparison.
    """
    sensitivity_metadata = _read_json(sensitivity_metadata_path)
    sensitivity_records = load_sensitivity_records_jsonl(sensitivity_path)
    observations = _load_observation_travel_times(observations_path)
    grid = _load_velocity_grid(velocity_grid_path)

    if not damping_values:
        raise ValueError("damping_values must contain at least one value.")
    observation_ids = _ordered_observation_ids(sensitivity_records, observations)
    cell_count = _cell_count_from_metadata(sensitivity_metadata)
    g_matrix = _assemble_dense_g_matrix(sensitivity_records, observation_ids, cell_count)
    travel_times = np.array([observations[observation_id] for observation_id in observation_ids])
    target_velocity = np.array(cell_centered_velocity_target(grid))
    if len(target_velocity) != cell_count:
        raise ValueError("Cell-centered target length does not match sensitivity cell_count.")
    bounds = _velocity_bounds_from_grid(grid)

    candidate_results = [
        _solve_candidate(
            g_matrix=g_matrix,
            travel_times=travel_times,
            target_velocity=target_velocity,
            damping=float(damping),
            velocity_bounds=bounds,
        )
        for damping in damping_values
    ]
    selected = min(
        candidate_results,
        key=lambda result: (
            result["residual_metrics"]["travel_time_rmse_s"],
            result["reconstruction_metrics"]["velocity_rmse_km_per_s"],
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "known_ray_damped_least_squares_summary.json"
    cell_velocity_path = output_dir / "known_ray_cell_velocity_predictions.csv"
    predicted_times_path = output_dir / "known_ray_predicted_travel_times.csv"
    _write_cell_velocity_predictions(cell_velocity_path, selected, target_velocity)
    _write_predicted_travel_times(predicted_times_path, observation_ids, travel_times, selected)
    _write_summary(
        summary_path,
        sensitivity_path=sensitivity_path,
        sensitivity_metadata_path=sensitivity_metadata_path,
        observations_path=observations_path,
        velocity_grid_path=velocity_grid_path,
        sensitivity_metadata=sensitivity_metadata,
        observation_count=len(observation_ids),
        cell_count=cell_count,
        damping_values=damping_values,
        candidate_results=candidate_results,
        selected=selected,
        bounds=bounds,
    )
    return KnownRayInversionOutputs(
        output_dir=output_dir,
        summary_json=summary_path,
        cell_velocity_csv=cell_velocity_path,
        predicted_travel_times_csv=predicted_times_path,
    )


def run_coverage_aware_known_ray_inversion_diagnostic(
    sensitivity_path: Path,
    sensitivity_metadata_path: Path,
    observations_path: Path,
    velocity_grid_path: Path,
    output_dir: Path,
    damping_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    smoothing_values: Sequence[float] = (0.0, 0.01, 0.1, 1.0),
    minimum_coverage_km: float = 1.0e-9,
) -> CoverageAwareKnownRayInversionOutputs:
    """Run a coverage-aware known-ray inversion with damping and smoothing sweeps."""
    sensitivity_metadata = _read_json(sensitivity_metadata_path)
    sensitivity_records = load_sensitivity_records_jsonl(sensitivity_path)
    observations = _load_observation_travel_times(observations_path)
    grid = _load_velocity_grid(velocity_grid_path)

    if not damping_values:
        raise ValueError("damping_values must contain at least one value.")
    if not smoothing_values:
        raise ValueError("smoothing_values must contain at least one value.")
    if minimum_coverage_km < 0.0:
        raise ValueError("minimum_coverage_km must be non-negative.")

    observation_ids = _ordered_observation_ids(sensitivity_records, observations)
    cell_count = _cell_count_from_metadata(sensitivity_metadata)
    cell_shape = _cell_shape_from_metadata(sensitivity_metadata, grid)
    g_matrix = _assemble_dense_g_matrix(sensitivity_records, observation_ids, cell_count)
    coverage = cell_coverage_diagnostics(sensitivity_records, cell_shape)
    coverage_summary = summarize_cell_coverage(coverage, minimum_coverage_km)
    covered_mask = np.array(
        [item["total_path_length_coverage_km"] >= minimum_coverage_km for item in coverage],
        dtype=bool,
    )
    travel_times = np.array([observations[observation_id] for observation_id in observation_ids])
    target_velocity = np.array(cell_centered_velocity_target(grid))
    if len(target_velocity) != cell_count:
        raise ValueError("Cell-centered target length does not match sensitivity cell_count.")
    bounds = _velocity_bounds_from_grid(grid)
    smoothing_ltl = smoothing_normal_matrix(cell_shape)

    candidate_results = [
        _solve_smoothed_candidate(
            g_matrix=g_matrix,
            travel_times=travel_times,
            target_velocity=target_velocity,
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
        candidate_results,
        key=lambda result: (
            _metric_sort_value(
                result["covered_cell_reconstruction_metrics"]["velocity_rmse_km_per_s"]
            ),
            result["residual_metrics"]["travel_time_rmse_s"],
            result["all_cell_reconstruction_metrics"]["velocity_rmse_km_per_s"],
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "known_ray_coverage_smoothing_summary.json"
    cell_velocity_path = output_dir / "known_ray_coverage_smoothing_cell_velocity_predictions.csv"
    predicted_times_path = output_dir / "known_ray_coverage_smoothing_predicted_travel_times.csv"
    coverage_path = output_dir / "known_ray_cell_coverage.csv"
    _write_coverage_csv(coverage_path, coverage, minimum_coverage_km)
    _write_cell_velocity_predictions(
        cell_velocity_path,
        selected,
        target_velocity,
        coverage=coverage,
        minimum_coverage_km=minimum_coverage_km,
    )
    _write_predicted_travel_times(predicted_times_path, observation_ids, travel_times, selected)
    _write_coverage_aware_summary(
        summary_path,
        sensitivity_path=sensitivity_path,
        sensitivity_metadata_path=sensitivity_metadata_path,
        observations_path=observations_path,
        velocity_grid_path=velocity_grid_path,
        sensitivity_metadata=sensitivity_metadata,
        observation_count=len(observation_ids),
        cell_count=cell_count,
        damping_values=damping_values,
        smoothing_values=smoothing_values,
        minimum_coverage_km=minimum_coverage_km,
        coverage_summary=coverage_summary,
        candidate_results=candidate_results,
        selected=selected,
        bounds=bounds,
    )
    return CoverageAwareKnownRayInversionOutputs(
        output_dir=output_dir,
        summary_json=summary_path,
        cell_velocity_csv=cell_velocity_path,
        predicted_travel_times_csv=predicted_times_path,
        coverage_csv=coverage_path,
    )


def run_known_ray_perturbation_inversion_diagnostic(
    sensitivity_path: Path,
    sensitivity_metadata_path: Path,
    observations_path: Path,
    velocity_grid_path: Path,
    reference_velocity_grid_path: Path,
    output_dir: Path,
    damping: float = 0.01,
    smoothing: float = 1.0,
    minimum_coverage_km: float = 1.0e-9,
) -> KnownRayPerturbationInversionOutputs:
    """Run a true-geometry inversion for slowness perturbations around a prior.

    This isolates the formulation issue in the original known-ray diagnostic:
    the solve estimates ``delta_s`` around the same layered reference prior used
    by the reference-ray baseline, while ``G`` still comes from true-model rays.
    It is a geometry/formulation diagnostic, not a fair test-set baseline.
    """
    if damping < 0.0 or smoothing < 0.0:
        raise ValueError("damping and smoothing must be non-negative.")
    if minimum_coverage_km < 0.0:
        raise ValueError("minimum_coverage_km must be non-negative.")

    sensitivity_metadata = _read_json(sensitivity_metadata_path)
    sensitivity_records = load_sensitivity_records_jsonl(sensitivity_path)
    observations = _load_observation_travel_times(observations_path)
    target_grid = _load_velocity_grid(velocity_grid_path)
    reference_grid = _load_velocity_grid(reference_velocity_grid_path)
    observation_ids = _ordered_observation_ids(sensitivity_records, observations)
    cell_count = _cell_count_from_metadata(sensitivity_metadata)
    cell_shape = _cell_shape_from_metadata(sensitivity_metadata, target_grid)
    g_matrix = _assemble_dense_g_matrix(sensitivity_records, observation_ids, cell_count)
    coverage = cell_coverage_diagnostics(sensitivity_records, cell_shape)
    coverage_summary = summarize_cell_coverage(coverage, minimum_coverage_km)
    covered_mask = np.array(
        [item["total_path_length_coverage_km"] >= minimum_coverage_km for item in coverage],
        dtype=bool,
    )
    observed_times = np.array([observations[observation_id] for observation_id in observation_ids])
    target_velocity = np.array(cell_centered_velocity_target(target_grid))
    reference_velocity = np.array(cell_centered_velocity_target(reference_grid))
    if len(target_velocity) != cell_count or len(reference_velocity) != cell_count:
        raise ValueError("Cell-centered target/reference lengths must match sensitivity cell_count.")
    reference_slowness = 1.0 / reference_velocity
    smoothing_ltl = smoothing_normal_matrix(cell_shape)
    bounds = _velocity_bounds_from_grid(target_grid)
    selected = _solve_known_ray_perturbation_candidate(
        g_matrix=g_matrix,
        observed_travel_times=observed_times,
        target_velocity=target_velocity,
        reference_velocity=reference_velocity,
        reference_slowness=reference_slowness,
        covered_mask=covered_mask,
        damping=float(damping),
        smoothing=float(smoothing),
        smoothing_ltl=smoothing_ltl,
        velocity_bounds=bounds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "known_ray_perturbation_summary.json"
    cell_velocity_path = output_dir / "known_ray_perturbation_cell_velocity_predictions.csv"
    predicted_times_path = output_dir / "known_ray_perturbation_predicted_travel_times.csv"
    coverage_path = output_dir / "known_ray_perturbation_cell_coverage.csv"
    _write_coverage_csv(coverage_path, coverage, minimum_coverage_km)
    _write_cell_velocity_predictions(
        cell_velocity_path,
        selected,
        target_velocity,
        coverage=coverage,
        minimum_coverage_km=minimum_coverage_km,
    )
    _write_predicted_travel_times(predicted_times_path, observation_ids, observed_times, selected)
    _write_known_ray_perturbation_summary(
        summary_path,
        sensitivity_path=sensitivity_path,
        sensitivity_metadata_path=sensitivity_metadata_path,
        observations_path=observations_path,
        velocity_grid_path=velocity_grid_path,
        reference_velocity_grid_path=reference_velocity_grid_path,
        sensitivity_metadata=sensitivity_metadata,
        observation_count=len(observation_ids),
        cell_count=cell_count,
        damping=float(damping),
        smoothing=float(smoothing),
        minimum_coverage_km=minimum_coverage_km,
        coverage_summary=coverage_summary,
        target_velocity=target_velocity,
        reference_velocity=reference_velocity,
        selected=selected,
        bounds=bounds,
    )
    return KnownRayPerturbationInversionOutputs(
        output_dir=output_dir,
        summary_json=summary_path,
        cell_velocity_csv=cell_velocity_path,
        predicted_travel_times_csv=predicted_times_path,
        coverage_csv=coverage_path,
    )


def solve_damped_least_squares_slowness(
    g_matrix: np.ndarray,
    travel_times: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Solve ``(G.T @ G + damping * I) @ s = G.T @ t`` for cell slowness."""
    if damping < 0.0:
        raise ValueError("damping must be non-negative.")
    if g_matrix.ndim != 2:
        raise ValueError("g_matrix must be two-dimensional.")
    if travel_times.ndim != 1 or travel_times.shape[0] != g_matrix.shape[0]:
        raise ValueError("travel_times length must match G rows.")
    normal_matrix = g_matrix.T @ g_matrix
    if damping > 0.0:
        normal_matrix = normal_matrix + damping * np.eye(g_matrix.shape[1])
    right_hand_side = g_matrix.T @ travel_times
    return np.linalg.solve(normal_matrix, right_hand_side)


def solve_smoothed_damped_least_squares_slowness(
    g_matrix: np.ndarray,
    travel_times: np.ndarray,
    damping: float,
    smoothing: float,
    smoothing_ltl: np.ndarray,
) -> np.ndarray:
    """Solve ``(G.T @ G + lambda * I + alpha * L.T @ L) @ s = G.T @ t``."""
    if damping < 0.0:
        raise ValueError("damping must be non-negative.")
    if smoothing < 0.0:
        raise ValueError("smoothing must be non-negative.")
    if g_matrix.ndim != 2:
        raise ValueError("g_matrix must be two-dimensional.")
    if travel_times.ndim != 1 or travel_times.shape[0] != g_matrix.shape[0]:
        raise ValueError("travel_times length must match G rows.")
    if smoothing_ltl.shape != (g_matrix.shape[1], g_matrix.shape[1]):
        raise ValueError("smoothing_ltl shape must match G columns.")
    normal_matrix = g_matrix.T @ g_matrix
    if damping > 0.0:
        normal_matrix = normal_matrix + damping * np.eye(g_matrix.shape[1])
    if smoothing > 0.0:
        normal_matrix = normal_matrix + smoothing * smoothing_ltl
    return np.linalg.solve(normal_matrix, g_matrix.T @ travel_times)


def solve_prior_perturbation_slowness(
    g_matrix: np.ndarray,
    observed_travel_times: np.ndarray,
    reference_slowness: np.ndarray,
    damping: float,
    smoothing: float,
    smoothing_ltl: np.ndarray,
) -> np.ndarray:
    """Solve a damped/smoothed slowness perturbation around a reference prior."""
    if g_matrix.ndim != 2:
        raise ValueError("g_matrix must be two-dimensional.")
    if observed_travel_times.ndim != 1 or observed_travel_times.shape[0] != g_matrix.shape[0]:
        raise ValueError("observed_travel_times length must match G rows.")
    if reference_slowness.ndim != 1 or reference_slowness.shape[0] != g_matrix.shape[1]:
        raise ValueError("reference_slowness length must match G columns.")
    if damping < 0.0 or smoothing < 0.0:
        raise ValueError("damping and smoothing must be non-negative.")
    if smoothing_ltl.shape != (g_matrix.shape[1], g_matrix.shape[1]):
        raise ValueError("smoothing_ltl shape must match G columns.")
    normal_matrix = g_matrix.T @ g_matrix
    if damping > 0.0:
        normal_matrix = normal_matrix + damping * np.eye(g_matrix.shape[1])
    if smoothing > 0.0:
        normal_matrix = normal_matrix + smoothing * smoothing_ltl
    residual_travel_times = observed_travel_times - (g_matrix @ reference_slowness)
    delta_slowness = np.linalg.solve(normal_matrix, g_matrix.T @ residual_travel_times)
    return reference_slowness + delta_slowness


def smoothing_difference_matrix(cell_shape: tuple[int, int, int]) -> np.ndarray:
    """Build first-difference rows between neighboring Cartesian cells."""
    nx_cells, ny_cells, nz_cells = cell_shape
    _validate_cell_shape(cell_shape)
    rows: list[np.ndarray] = []
    cell_count = nx_cells * ny_cells * nz_cells
    for iz in range(nz_cells):
        for iy in range(ny_cells):
            for ix in range(nx_cells):
                current = _cell_index(ix, iy, iz, nx_cells, ny_cells)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    neighbor_ix = ix + dx
                    neighbor_iy = iy + dy
                    neighbor_iz = iz + dz
                    if (
                        neighbor_ix >= nx_cells
                        or neighbor_iy >= ny_cells
                        or neighbor_iz >= nz_cells
                    ):
                        continue
                    row = np.zeros(cell_count, dtype=float)
                    row[current] = 1.0
                    row[_cell_index(neighbor_ix, neighbor_iy, neighbor_iz, nx_cells, ny_cells)] = (
                        -1.0
                    )
                    rows.append(row)
    if not rows:
        return np.zeros((0, cell_count), dtype=float)
    return np.vstack(rows)


def smoothing_normal_matrix(cell_shape: tuple[int, int, int]) -> np.ndarray:
    """Return ``L.T @ L`` for Cartesian nearest-neighbor smoothing."""
    difference_matrix = smoothing_difference_matrix(cell_shape)
    return difference_matrix.T @ difference_matrix


def cell_centered_velocity_target(grid: CartesianVelocityGrid3D) -> tuple[float, ...]:
    """Convert the node-centered velocity grid to cell-centered target velocities."""
    nx = len(grid.x_coordinates_km)
    ny = len(grid.y_coordinates_km)
    nz = len(grid.z_coordinates_km)
    expected_node_count = nx * ny * nz
    if len(grid.p_velocity_km_per_s) != expected_node_count:
        raise ValueError("Velocity grid shape does not match flattened velocity count.")
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("Velocity grid must contain at least two nodes on each axis.")

    target: list[float] = []
    for iz in range(nz - 1):
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                target.append(
                    sum(
                        grid.p_velocity_km_per_s[_node_index(ix + dx, iy + dy, iz + dz, nx, ny)]
                        for dz in (0, 1)
                        for dy in (0, 1)
                        for dx in (0, 1)
                    )
                    / 8.0
                )
    return tuple(target)


def slowness_to_velocity(
    slowness: np.ndarray,
    velocity_bounds: tuple[float, float],
) -> tuple[np.ndarray, int]:
    """Convert slowness to velocity, clipping nonphysical values to configured bounds."""
    lower_velocity, upper_velocity = velocity_bounds
    min_slowness = 1.0 / upper_velocity
    max_slowness = 1.0 / lower_velocity
    clipped_slowness = np.clip(slowness, min_slowness, max_slowness)
    nonphysical_count = int(np.count_nonzero(slowness <= 0.0))
    return 1.0 / clipped_slowness, nonphysical_count


def cell_coverage_diagnostics(
    sensitivity_records: tuple[dict[str, Any], ...],
    cell_shape: tuple[int, int, int],
) -> tuple[dict[str, Any], ...]:
    """Return per-cell path-length and observation-touch coverage diagnostics."""
    _validate_cell_shape(cell_shape)
    nx_cells, ny_cells, nz_cells = cell_shape
    cell_count = nx_cells * ny_cells * nz_cells
    path_length_by_cell = [0.0] * cell_count
    observation_ids_by_cell: list[set[str]] = [set() for _ in range(cell_count)]
    for record in sensitivity_records:
        cell_index = int(record["cell_index"])
        if not 0 <= cell_index < cell_count:
            raise ValueError(f"Sensitivity cell_index is outside cell shape: {cell_index}")
        path_length_by_cell[cell_index] += float(record["path_length_km"])
        observation_ids_by_cell[cell_index].add(str(record["observation_id"]))

    diagnostics: list[dict[str, Any]] = []
    for cell_index in range(cell_count):
        ix, iy, iz = _cell_coordinates(cell_index, nx_cells, ny_cells)
        diagnostics.append(
            {
                "cell_index": cell_index,
                "ix": ix,
                "iy": iy,
                "iz": iz,
                "total_path_length_coverage_km": path_length_by_cell[cell_index],
                "observation_touch_count": len(observation_ids_by_cell[cell_index]),
            }
        )
    return tuple(diagnostics)


def summarize_cell_coverage(
    coverage: tuple[dict[str, Any], ...],
    minimum_coverage_km: float,
) -> dict[str, Any]:
    """Summarize per-cell ray coverage for interpretation and metric masking."""
    positive_coverages = [
        float(item["total_path_length_coverage_km"])
        for item in coverage
        if float(item["total_path_length_coverage_km"]) > 0.0
    ]
    covered_count = sum(
        float(item["total_path_length_coverage_km"]) >= minimum_coverage_km for item in coverage
    )
    cell_count = len(coverage)
    return {
        "minimum_coverage_threshold_km": minimum_coverage_km,
        "covered_cell_count": covered_count,
        "uncovered_cell_count": cell_count - covered_count,
        "coverage_fraction": covered_count / cell_count if cell_count else 0.0,
        "min_positive_coverage_km": min(positive_coverages) if positive_coverages else 0.0,
        "mean_positive_coverage_km": (
            sum(positive_coverages) / len(positive_coverages) if positive_coverages else 0.0
        ),
        "max_positive_coverage_km": max(positive_coverages) if positive_coverages else 0.0,
    }


def _solve_candidate(
    g_matrix: np.ndarray,
    travel_times: np.ndarray,
    target_velocity: np.ndarray,
    damping: float,
    velocity_bounds: tuple[float, float],
) -> dict[str, Any]:
    slowness = solve_damped_least_squares_slowness(g_matrix, travel_times, damping)
    predicted_travel_times = g_matrix @ slowness
    velocity, nonphysical_count = slowness_to_velocity(slowness, velocity_bounds)
    residuals = predicted_travel_times - travel_times
    return {
        "damping": damping,
        "slowness_s_per_km": slowness.tolist(),
        "velocity_km_per_s": velocity.tolist(),
        "predicted_travel_times_s": predicted_travel_times.tolist(),
        "travel_time_residuals_s": residuals.tolist(),
        "nonphysical_slowness_count": nonphysical_count,
        "residual_metrics": {
            "travel_time_mae_s": mean_absolute_error(
                travel_times.tolist(),
                predicted_travel_times.tolist(),
            ),
            "travel_time_rmse_s": root_mean_squared_error(
                travel_times.tolist(),
                predicted_travel_times.tolist(),
            ),
            "travel_time_bias_s": float(np.mean(residuals)),
        },
        "reconstruction_metrics": {
            "velocity_mae_km_per_s": mean_absolute_error(
                target_velocity.tolist(),
                velocity.tolist(),
            ),
            "velocity_rmse_km_per_s": root_mean_squared_error(
                target_velocity.tolist(),
                velocity.tolist(),
            ),
        },
    }


def _solve_smoothed_candidate(
    g_matrix: np.ndarray,
    travel_times: np.ndarray,
    target_velocity: np.ndarray,
    covered_mask: np.ndarray,
    damping: float,
    smoothing: float,
    smoothing_ltl: np.ndarray,
    velocity_bounds: tuple[float, float],
) -> dict[str, Any]:
    slowness = solve_smoothed_damped_least_squares_slowness(
        g_matrix,
        travel_times,
        damping,
        smoothing,
        smoothing_ltl,
    )
    predicted_travel_times = g_matrix @ slowness
    velocity, nonphysical_count, clipped_count = _slowness_to_velocity_with_clipping(
        slowness,
        velocity_bounds,
    )
    residuals = predicted_travel_times - travel_times
    return {
        "damping": damping,
        "smoothing": smoothing,
        "slowness_s_per_km": slowness.tolist(),
        "velocity_km_per_s": velocity.tolist(),
        "predicted_travel_times_s": predicted_travel_times.tolist(),
        "travel_time_residuals_s": residuals.tolist(),
        "nonphysical_slowness_count": nonphysical_count,
        "clipped_velocity_cell_count": clipped_count,
        "residual_metrics": _travel_time_metrics(travel_times, predicted_travel_times, residuals),
        "all_cell_reconstruction_metrics": _velocity_metrics(target_velocity, velocity),
        "covered_cell_reconstruction_metrics": _covered_velocity_metrics(
            target_velocity,
            velocity,
            covered_mask,
        ),
    }


def _solve_known_ray_perturbation_candidate(
    g_matrix: np.ndarray,
    observed_travel_times: np.ndarray,
    target_velocity: np.ndarray,
    reference_velocity: np.ndarray,
    reference_slowness: np.ndarray,
    covered_mask: np.ndarray,
    damping: float,
    smoothing: float,
    smoothing_ltl: np.ndarray,
    velocity_bounds: tuple[float, float],
) -> dict[str, Any]:
    slowness = solve_prior_perturbation_slowness(
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
        "residual_metrics": _travel_time_metrics(
            observed_travel_times,
            predicted_times,
            residuals,
        ),
        "reference_prior_reconstruction_metrics": _velocity_metrics(
            target_velocity,
            reference_velocity,
        ),
        "all_cell_reconstruction_metrics": _velocity_metrics(target_velocity, velocity),
        "covered_cell_reconstruction_metrics": _covered_velocity_metrics(
            target_velocity,
            velocity,
            covered_mask,
        ),
        "reference_prior_covered_cell_reconstruction_metrics": _covered_velocity_metrics(
            target_velocity,
            reference_velocity,
            covered_mask,
        ),
    }


def _travel_time_metrics(
    travel_times: np.ndarray,
    predicted_travel_times: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, float]:
    return {
        "travel_time_mae_s": mean_absolute_error(
            travel_times.tolist(),
            predicted_travel_times.tolist(),
        ),
        "travel_time_rmse_s": root_mean_squared_error(
            travel_times.tolist(),
            predicted_travel_times.tolist(),
        ),
        "travel_time_bias_s": float(np.mean(residuals)),
    }


def _velocity_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "velocity_mae_km_per_s": mean_absolute_error(actual.tolist(), predicted.tolist()),
        "velocity_rmse_km_per_s": root_mean_squared_error(actual.tolist(), predicted.tolist()),
    }


def _covered_velocity_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    covered_mask: np.ndarray,
) -> dict[str, float | int | None]:
    covered_count = int(np.count_nonzero(covered_mask))
    if covered_count == 0:
        return {
            "covered_cell_count": 0,
            "velocity_mae_km_per_s": None,
            "velocity_rmse_km_per_s": None,
        }
    actual_covered = actual[covered_mask]
    predicted_covered = predicted[covered_mask]
    return {
        "covered_cell_count": covered_count,
        **_velocity_metrics(actual_covered, predicted_covered),
    }


def _slowness_to_velocity_with_clipping(
    slowness: np.ndarray,
    velocity_bounds: tuple[float, float],
) -> tuple[np.ndarray, int, int]:
    lower_velocity, upper_velocity = velocity_bounds
    min_slowness = 1.0 / upper_velocity
    max_slowness = 1.0 / lower_velocity
    clipped_slowness = np.clip(slowness, min_slowness, max_slowness)
    nonphysical_count = int(np.count_nonzero(slowness <= 0.0))
    clipped_count = int(np.count_nonzero(~np.isclose(slowness, clipped_slowness)))
    return 1.0 / clipped_slowness, nonphysical_count, clipped_count


def _assemble_dense_g_matrix(
    sensitivity_records: tuple[dict[str, Any], ...],
    observation_ids: tuple[str, ...],
    cell_count: int,
) -> np.ndarray:
    row_by_observation_id = {observation_id: row for row, observation_id in enumerate(observation_ids)}
    g_matrix = np.zeros((len(observation_ids), cell_count), dtype=float)
    for record in sensitivity_records:
        observation_id = str(record["observation_id"])
        if observation_id not in row_by_observation_id:
            raise ValueError(f"Sensitivity row has no matching observation: {observation_id}")
        cell_index = int(record["cell_index"])
        if not 0 <= cell_index < cell_count:
            raise ValueError(f"Sensitivity cell_index is outside metadata cell_count: {cell_index}")
        g_matrix[row_by_observation_id[observation_id], cell_index] += float(
            record["path_length_km"]
        )
    return g_matrix


def _ordered_observation_ids(
    sensitivity_records: tuple[dict[str, Any], ...],
    observations: dict[str, float],
) -> tuple[str, ...]:
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for record in sensitivity_records:
        observation_id = str(record["observation_id"])
        if observation_id in seen:
            continue
        if observation_id not in observations:
            raise ValueError(f"Missing observed travel time for {observation_id}.")
        ordered_ids.append(observation_id)
        seen.add(observation_id)
    extra_observations = set(observations) - seen
    if extra_observations:
        raise ValueError(
            "Observation travel-time file contains IDs without sensitivity rows: "
            + ", ".join(sorted(extra_observations)[:5])
        )
    return tuple(ordered_ids)


def _load_observation_travel_times(path: Path) -> dict[str, float]:
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            return {
                str(row["observation_id"]): float(row["travel_time_s"])
                for row in csv.DictReader(handle)
            }
    payload = _read_json(path)
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("Observation JSON must contain an observations list.")
    return {
        str(row["observation_id"]): float(row["travel_time_s"])
        for row in observations
        if isinstance(row, dict)
    }


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


def _cell_count_from_metadata(metadata: dict[str, Any]) -> int:
    if "cell_count" in metadata:
        return int(metadata["cell_count"])
    shape = metadata.get("grid_shape_cells")
    if not isinstance(shape, dict):
        raise ValueError("Sensitivity metadata must contain cell_count or grid_shape_cells.")
    return int(shape["nx"]) * int(shape["ny"]) * int(shape["nz"])


def _velocity_bounds_from_grid(grid: CartesianVelocityGrid3D) -> tuple[float, float]:
    values = tuple(float(value) for value in grid.p_velocity_km_per_s)
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1.0e-12):
        margin = max(lower * 0.25, 1.0)
        return max(1.0e-6, lower - margin), upper + margin
    return lower, upper


def _write_cell_velocity_predictions(
    path: Path,
    selected: dict[str, Any],
    target_velocity: np.ndarray,
    coverage: tuple[dict[str, Any], ...] | None = None,
    minimum_coverage_km: float | None = None,
) -> None:
    fieldnames = [
        "cell_index",
        "target_velocity_km_per_s",
        "predicted_velocity_km_per_s",
        "predicted_slowness_s_per_km",
        "velocity_error_km_per_s",
    ]
    if coverage is not None:
        fieldnames.extend(
            [
                "total_path_length_coverage_km",
                "observation_touch_count",
                "covered_by_threshold",
            ]
        )
    predicted_velocity = selected["velocity_km_per_s"]
    predicted_slowness = selected["slowness_s_per_km"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell_index, target in enumerate(target_velocity.tolist()):
            predicted = float(predicted_velocity[cell_index])
            row = {
                "cell_index": cell_index,
                "target_velocity_km_per_s": target,
                "predicted_velocity_km_per_s": predicted,
                "predicted_slowness_s_per_km": predicted_slowness[cell_index],
                "velocity_error_km_per_s": predicted - target,
            }
            if coverage is not None:
                coverage_item = coverage[cell_index]
                coverage_km = float(coverage_item["total_path_length_coverage_km"])
                row.update(
                    {
                        "total_path_length_coverage_km": coverage_km,
                        "observation_touch_count": coverage_item["observation_touch_count"],
                        "covered_by_threshold": (
                            coverage_km >= float(minimum_coverage_km or 0.0)
                        ),
                    }
                )
            writer.writerow(row)


def _write_predicted_travel_times(
    path: Path,
    observation_ids: tuple[str, ...],
    travel_times: np.ndarray,
    selected: dict[str, Any],
) -> None:
    fieldnames = (
        "observation_id",
        "observed_travel_time_s",
        "predicted_travel_time_s",
        "residual_s",
    )
    predictions = selected["predicted_travel_times_s"]
    residuals = selected["travel_time_residuals_s"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, observation_id in enumerate(observation_ids):
            writer.writerow(
                {
                    "observation_id": observation_id,
                    "observed_travel_time_s": travel_times[index],
                    "predicted_travel_time_s": predictions[index],
                    "residual_s": residuals[index],
                }
            )


def _write_summary(
    path: Path,
    sensitivity_path: Path,
    sensitivity_metadata_path: Path,
    observations_path: Path,
    velocity_grid_path: Path,
    sensitivity_metadata: dict[str, Any],
    observation_count: int,
    cell_count: int,
    damping_values: Sequence[float],
    candidate_results: list[dict[str, Any]],
    selected: dict[str, Any],
    bounds: tuple[float, float],
) -> None:
    compact_candidates = [
        {
            "damping": result["damping"],
            "residual_metrics": result["residual_metrics"],
            "reconstruction_metrics": result["reconstruction_metrics"],
            "nonphysical_slowness_count": result["nonphysical_slowness_count"],
        }
        for result in candidate_results
    ]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "artifact_type": "known_ray_damped_least_squares_inversion_diagnostic",
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "source_files": {
                    "sensitivity_sidecar": sensitivity_path.as_posix(),
                    "sensitivity_metadata": sensitivity_metadata_path.as_posix(),
                    "observations": observations_path.as_posix(),
                    "target_velocity_grid": velocity_grid_path.as_posix(),
                },
                "simulation_method": sensitivity_metadata.get("simulation_method"),
                "diagnostic_type": "known_ray_true_geometry_damped_least_squares",
                "normal_equation": "(G.T @ G + lambda * I) @ s = G.T @ t",
                "observation_count": observation_count,
                "cell_count": cell_count,
                "damping_values": [float(value) for value in damping_values],
                "selected_damping": selected["damping"],
                "velocity_bounds_km_per_s": list(bounds),
                "residual_metrics": selected["residual_metrics"],
                "reconstruction_metrics": selected["reconstruction_metrics"],
                "candidate_summaries": compact_candidates,
                "target_comparison_contract": {
                    "sensitivity_representation": "cell_centered_path_lengths_between_adjacent_grid_nodes",
                    "source_velocity_grid_representation": "node_centered_absolute_velocity",
                    "target_conversion": "cell velocity is the arithmetic mean of the eight surrounding node velocities",
                    "metric_domain": "derived_cell_grid",
                },
                "assumptions": [
                    "This is a known-ray diagnostic because G is built from pseudo-bending rays traced through the true synthetic velocity model.",
                    "It is optimistic and should not be described as the final fair classical tomography comparison.",
                    "The unknown model vector is cell slowness in seconds per kilometer.",
                    "Predicted travel time is G multiplied by cell slowness.",
                    "Nonphysical or out-of-range slowness values are clipped to the velocity range inferred from the target grid before velocity metrics are computed.",
                    "No real-data validation or comparison against ML methods is claimed by this artifact.",
                ],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _write_coverage_csv(
    path: Path,
    coverage: tuple[dict[str, Any], ...],
    minimum_coverage_km: float,
) -> None:
    fieldnames = (
        "cell_index",
        "ix",
        "iy",
        "iz",
        "total_path_length_coverage_km",
        "observation_touch_count",
        "covered_by_threshold",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in coverage:
            coverage_km = float(item["total_path_length_coverage_km"])
            writer.writerow(
                {
                    **item,
                    "covered_by_threshold": coverage_km >= minimum_coverage_km,
                }
            )


def _write_coverage_aware_summary(
    path: Path,
    sensitivity_path: Path,
    sensitivity_metadata_path: Path,
    observations_path: Path,
    velocity_grid_path: Path,
    sensitivity_metadata: dict[str, Any],
    observation_count: int,
    cell_count: int,
    damping_values: Sequence[float],
    smoothing_values: Sequence[float],
    minimum_coverage_km: float,
    coverage_summary: dict[str, Any],
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
        }
        for result in candidate_results
    ]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "artifact_type": "coverage_aware_smoothed_known_ray_inversion_diagnostic",
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "source_files": {
                    "sensitivity_sidecar": sensitivity_path.as_posix(),
                    "sensitivity_metadata": sensitivity_metadata_path.as_posix(),
                    "observations": observations_path.as_posix(),
                    "target_velocity_grid": velocity_grid_path.as_posix(),
                },
                "simulation_method": sensitivity_metadata.get("simulation_method"),
                "diagnostic_type": "known_ray_true_geometry_damped_smoothed_least_squares",
                "normal_equation": "(G.T @ G + lambda * I + alpha * L.T @ L) @ s = G.T @ t",
                "observation_count": observation_count,
                "cell_count": cell_count,
                "damping_values": [float(value) for value in damping_values],
                "smoothing_values": [float(value) for value in smoothing_values],
                "selected_damping": selected["damping"],
                "selected_smoothing": selected["smoothing"],
                "minimum_coverage_threshold_km": minimum_coverage_km,
                "coverage_summary": coverage_summary,
                "velocity_bounds_km_per_s": list(bounds),
                "residual_metrics": selected["residual_metrics"],
                "all_cell_reconstruction_metrics": selected[
                    "all_cell_reconstruction_metrics"
                ],
                "covered_cell_reconstruction_metrics": selected[
                    "covered_cell_reconstruction_metrics"
                ],
                "nonphysical_slowness_count": selected["nonphysical_slowness_count"],
                "clipped_velocity_cell_count": selected["clipped_velocity_cell_count"],
                "candidate_summaries": compact_candidates,
                "target_comparison_contract": {
                    "sensitivity_representation": "cell_centered_path_lengths_between_adjacent_grid_nodes",
                    "source_velocity_grid_representation": "node_centered_absolute_velocity",
                    "target_conversion": "cell velocity is the arithmetic mean of the eight surrounding node velocities",
                    "all_cell_metric_domain": "all derived Cartesian cells",
                    "covered_cell_metric_domain": "cells with total path-length coverage greater than or equal to the configured threshold",
                },
                "assumptions": [
                    "This remains a known-ray diagnostic because G is built from pseudo-bending rays traced through the true synthetic velocity model.",
                    "The smoothing term penalizes first differences between neighboring Cartesian cell slownesses.",
                    "Covered-cell metrics exclude cells below the configured coverage threshold to avoid interpreting unconstrained cells as reconstruction evidence.",
                    "All-cell metrics are still reported to show the effect of weakly or completely unconstrained cells.",
                    "Velocity clipping is explicit and counted; clipped cells should be treated as evidence of poor constraint or regularization limits.",
                    "No fair classical-versus-ML result is claimed until a reference-model fixed-ray inversion is implemented and evaluated.",
                ],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _write_known_ray_perturbation_summary(
    path: Path,
    sensitivity_path: Path,
    sensitivity_metadata_path: Path,
    observations_path: Path,
    velocity_grid_path: Path,
    reference_velocity_grid_path: Path,
    sensitivity_metadata: dict[str, Any],
    observation_count: int,
    cell_count: int,
    damping: float,
    smoothing: float,
    minimum_coverage_km: float,
    coverage_summary: dict[str, Any],
    target_velocity: np.ndarray,
    reference_velocity: np.ndarray,
    selected: dict[str, Any],
    bounds: tuple[float, float],
) -> None:
    prior_metrics = selected["reference_prior_reconstruction_metrics"]
    optimized_metrics = selected["all_cell_reconstruction_metrics"]
    prior_covered_metrics = selected["reference_prior_covered_cell_reconstruction_metrics"]
    optimized_covered_metrics = selected["covered_cell_reconstruction_metrics"]
    payload = {
        "artifact_type": "known_ray_true_geometry_perturbation_inversion_diagnostic",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_files": {
            "sensitivity_sidecar": sensitivity_path.as_posix(),
            "sensitivity_metadata": sensitivity_metadata_path.as_posix(),
            "observations": observations_path.as_posix(),
            "target_velocity_grid": velocity_grid_path.as_posix(),
            "reference_velocity_grid": reference_velocity_grid_path.as_posix(),
        },
        "simulation_method": sensitivity_metadata.get("simulation_method"),
        "diagnostic_type": "known_ray_true_geometry_damped_smoothed_perturbation",
        "normal_equation": (
            "(G.T @ G + lambda * I + alpha * L.T @ L) @ delta_s "
            "= G.T @ (t_obs - G @ s_reference); s = s_reference + delta_s"
        ),
        "parameter_selection_scope": "fixed_parameters_for_formulation_diagnosis",
        "observation_count": observation_count,
        "cell_count": cell_count,
        "damping": damping,
        "smoothing": smoothing,
        "minimum_coverage_threshold_km": minimum_coverage_km,
        "coverage_summary": coverage_summary,
        "velocity_bounds_km_per_s": list(bounds),
        "reference_prior_metrics": prior_metrics,
        "reference_prior_covered_cell_metrics": prior_covered_metrics,
        "residual_metrics": selected["residual_metrics"],
        "all_cell_reconstruction_metrics": optimized_metrics,
        "covered_cell_reconstruction_metrics": optimized_covered_metrics,
        "clipped_velocity_cell_count": selected["clipped_velocity_cell_count"],
        "nonphysical_slowness_count": selected["nonphysical_slowness_count"],
        "paired_difference_metrics": {
            "all_cell_rmse_optimized_minus_prior_km_per_s": (
                optimized_metrics["velocity_rmse_km_per_s"]
                - prior_metrics["velocity_rmse_km_per_s"]
            ),
            "covered_cell_rmse_optimized_minus_prior_km_per_s": (
                optimized_covered_metrics["velocity_rmse_km_per_s"]
                - prior_covered_metrics["velocity_rmse_km_per_s"]
                if optimized_covered_metrics["velocity_rmse_km_per_s"] is not None
                and prior_covered_metrics["velocity_rmse_km_per_s"] is not None
                else None
            ),
        },
        "target_comparison_contract": {
            "target": "cell_centered_velocity_from_node_centered_target_grid",
            "reference_prior": "cell_centered_velocity_from_the_same_reference_grid_used_by_reference_ray_baseline",
            "geometry": "true-model pseudo-bending ray sensitivity matrix",
            "metric_domains": [
                "all derived Cartesian cells",
                "cells above the configured total path-length coverage threshold",
            ],
        },
        "diagnostic_interpretation": [
            "The earlier supplied-ray diagnostic solved for absolute slowness and regularized that vector toward zero; this corrected diagnostic regularizes a perturbation around the reference prior.",
            "The true-model ray geometry remains privileged, so this result is an optimistic formulation/geometry diagnostic rather than a fair held-out classical baseline.",
            "The fixed parameters are used to isolate the formulation change and are not selected from the test reconstruction error.",
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _metric_sort_value(value: float | int | None) -> float:
    return math.inf if value is None else float(value)


def _cell_shape_from_metadata(
    metadata: dict[str, Any],
    grid: CartesianVelocityGrid3D,
) -> tuple[int, int, int]:
    shape = metadata.get("grid_shape_cells")
    if isinstance(shape, dict):
        return int(shape["nx"]), int(shape["ny"]), int(shape["nz"])
    return (
        len(grid.x_coordinates_km) - 1,
        len(grid.y_coordinates_km) - 1,
        len(grid.z_coordinates_km) - 1,
    )


def _validate_cell_shape(cell_shape: tuple[int, int, int]) -> None:
    if len(cell_shape) != 3 or any(value <= 0 for value in cell_shape):
        raise ValueError("cell_shape must contain three positive dimensions.")


def _cell_index(ix: int, iy: int, iz: int, nx_cells: int, ny_cells: int) -> int:
    return ix + nx_cells * (iy + ny_cells * iz)


def _cell_coordinates(
    cell_index: int,
    nx_cells: int,
    ny_cells: int,
) -> tuple[int, int, int]:
    iz, remainder = divmod(cell_index, nx_cells * ny_cells)
    iy, ix = divmod(remainder, nx_cells)
    return ix, iy, iz


def _node_index(ix: int, iy: int, iz: int, nx: int, ny: int) -> int:
    return ix + nx * (iy + ny * iz)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload
