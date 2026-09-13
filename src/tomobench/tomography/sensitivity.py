"""Sparse ray-to-cell sensitivity construction for classical travel-time inversion."""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tomobench.config.settings import BenchmarkSettings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import CartesianVelocityGrid3D, TravelTimeRayPath

PATH_LENGTH_ABS_TOLERANCE_KM = 1.0e-6


@dataclass(frozen=True)
class RayPathSensitivityValidationResult:
    """Validation result for ray-path length conservation in sparse sensitivity rows."""

    passed: bool
    checked_observation_count: int
    max_abs_difference_km: float
    errors: tuple[str, ...]


def ray_path_cell_lengths(
    ray_path: tuple[Point3D, ...],
    grid: CartesianVelocityGrid3D,
) -> dict[int, float]:
    """Accumulate polyline path length through Cartesian cells.

    The velocity grid stores values at nodes. This Phase 6 sensitivity matrix
    treats inversion cells as the intervals between adjacent grid nodes, giving
    a cell shape of ``(nx - 1, ny - 1, nz - 1)``. Cell indexes use x-fastest
    ordering: ``cell_index = ix + (nx - 1) * (iy + (ny - 1) * iz)``.

    When a ray segment lies exactly on a grid plane, midpoint assignment uses a
    lower-inclusive, upper-clamped convention: internal boundary coordinates are
    assigned to the lower-index cell and outer boundaries to the adjacent cell.
    """
    _validate_grid_cells(grid)
    if len(ray_path) < 2:
        raise ValueError("ray_path must contain at least two points.")

    lengths_by_cell: dict[int, float] = {}
    for start, end in zip(ray_path, ray_path[1:]):
        for cell_index, length_km in _segment_cell_lengths(start, end, grid).items():
            lengths_by_cell[cell_index] = lengths_by_cell.get(cell_index, 0.0) + length_km
    return {index: length for index, length in sorted(lengths_by_cell.items()) if length > 0.0}


def build_ray_path_sensitivity_records(
    ray_paths: tuple[TravelTimeRayPath, ...],
    grid: CartesianVelocityGrid3D,
) -> tuple[dict[str, Any], ...]:
    """Build sparse ``G`` records from in-memory pseudo-bending ray paths."""
    records: list[dict[str, Any]] = []
    nx_cells, ny_cells, _ = _cell_shape(grid)
    for ray_path in ray_paths:
        for cell_index, path_length_km in ray_path_cell_lengths(ray_path.ray_path, grid).items():
            ix, iy, iz = _cell_coordinates(cell_index, nx_cells, ny_cells)
            records.append(
                {
                    "observation_id": ray_path.observation_id,
                    "case_id": ray_path.case_id,
                    "cell_index": cell_index,
                    "ix": ix,
                    "iy": iy,
                    "iz": iz,
                    "path_length_km": path_length_km,
                    "simulation_id": ray_path.simulation_id,
                    "simulation_method": ray_path.simulation_method,
                    "velocity_model_id": ray_path.velocity_model_id,
                }
            )
    return tuple(records)


def save_ray_path_sensitivity_sidecar(
    ray_paths: tuple[TravelTimeRayPath, ...],
    grid: CartesianVelocityGrid3D,
    sensitivity_path: Path,
    metadata_path: Path,
    settings: BenchmarkSettings,
    source_ray_path_sidecar: Path | None = None,
) -> tuple[Path, Path, RayPathSensitivityValidationResult]:
    """Save sparse ray-cell sensitivity rows and metadata sidecars."""
    records = build_ray_path_sensitivity_records(ray_paths, grid)
    validation = validate_sensitivity_path_lengths(records, ray_paths)
    sensitivity_path.parent.mkdir(parents=True, exist_ok=True)
    with sensitivity_path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(
                {
                    "artifact_type": "cartesian_ray_cell_sensitivity_record",
                    "schema_version": settings.dataset_export.schema_version,
                    **record,
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _sensitivity_metadata_payload(
                records=records,
                ray_paths=ray_paths,
                grid=grid,
                settings=settings,
                sensitivity_path=sensitivity_path,
                source_ray_path_sidecar=source_ray_path_sidecar,
                validation=validation,
            ),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return sensitivity_path, metadata_path, validation


def validate_sensitivity_path_lengths(
    records: tuple[dict[str, Any], ...],
    ray_paths: tuple[TravelTimeRayPath, ...],
    abs_tolerance_km: float = PATH_LENGTH_ABS_TOLERANCE_KM,
) -> RayPathSensitivityValidationResult:
    """Validate that sparse sensitivity row sums conserve persisted ray lengths."""
    lengths_by_observation: dict[str, float] = {}
    for record in records:
        observation_id = str(record["observation_id"])
        lengths_by_observation[observation_id] = lengths_by_observation.get(
            observation_id,
            0.0,
        ) + float(record["path_length_km"])

    errors: list[str] = []
    max_abs_difference = 0.0
    for ray_path in ray_paths:
        observed = lengths_by_observation.get(ray_path.observation_id)
        if observed is None:
            errors.append(f"{ray_path.observation_id}: missing sensitivity rows.")
            continue
        difference = abs(observed - ray_path.path_length_km)
        max_abs_difference = max(max_abs_difference, difference)
        if difference > abs_tolerance_km:
            errors.append(
                f"{ray_path.observation_id}: sensitivity length sum differs from ray path by "
                f"{difference:.6g} km."
            )

    extra_ids = set(lengths_by_observation) - {ray_path.observation_id for ray_path in ray_paths}
    for observation_id in sorted(extra_ids):
        errors.append(f"{observation_id}: sensitivity rows have no matching ray path.")

    return RayPathSensitivityValidationResult(
        passed=not errors,
        checked_observation_count=len(ray_paths),
        max_abs_difference_km=max_abs_difference,
        errors=tuple(errors),
    )


def load_travel_time_ray_paths_jsonl(path: Path) -> tuple[TravelTimeRayPath, ...]:
    """Load persisted pseudo-bending ray-path sidecar records."""
    ray_paths: list[TravelTimeRayPath] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"Expected JSON object line in {path}.")
            ray_paths.append(_ray_path_from_record(record))
    return tuple(ray_paths)


def load_sensitivity_records_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """Load sparse sensitivity JSONL records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"Expected JSON object line in {path}.")
            records.append(dict(record))
    return tuple(records)


def validate_sensitivity_sidecar_files(
    sensitivity_path: Path,
    ray_path_sidecar_path: Path,
    abs_tolerance_km: float = PATH_LENGTH_ABS_TOLERANCE_KM,
) -> RayPathSensitivityValidationResult:
    """Validate persisted sparse sensitivity rows against persisted ray paths."""
    return validate_sensitivity_path_lengths(
        load_sensitivity_records_jsonl(sensitivity_path),
        load_travel_time_ray_paths_jsonl(ray_path_sidecar_path),
        abs_tolerance_km=abs_tolerance_km,
    )


def _segment_cell_lengths(
    start: Point3D,
    end: Point3D,
    grid: CartesianVelocityGrid3D,
) -> dict[int, float]:
    dx = end.x_km - start.x_km
    dy = end.y_km - start.y_km
    dz = end.z_km - start.z_km
    segment_length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if segment_length == 0.0:
        return {}

    breakpoints = _segment_grid_breakpoints(start, end, grid)
    lengths_by_cell: dict[int, float] = {}
    nx_cells, ny_cells, _ = _cell_shape(grid)
    for left, right in zip(breakpoints, breakpoints[1:]):
        if right <= left:
            continue
        midpoint = _interpolate_point(start, end, (left + right) / 2.0)
        ix = _cell_axis_index(grid.x_coordinates_km, midpoint.x_km)
        iy = _cell_axis_index(grid.y_coordinates_km, midpoint.y_km)
        iz = _cell_axis_index(grid.z_coordinates_km, midpoint.z_km)
        cell_index = ix + nx_cells * (iy + ny_cells * iz)
        lengths_by_cell[cell_index] = (
            lengths_by_cell.get(cell_index, 0.0) + segment_length * (right - left)
        )
    return lengths_by_cell


def _segment_grid_breakpoints(
    start: Point3D,
    end: Point3D,
    grid: CartesianVelocityGrid3D,
) -> tuple[float, ...]:
    breakpoints = [0.0, 1.0]
    for start_value, end_value, coordinates in (
        (start.x_km, end.x_km, grid.x_coordinates_km),
        (start.y_km, end.y_km, grid.y_coordinates_km),
        (start.z_km, end.z_km, grid.z_coordinates_km),
    ):
        delta = end_value - start_value
        if abs(delta) < 1.0e-15:
            continue
        low = min(start_value, end_value)
        high = max(start_value, end_value)
        for coordinate in coordinates:
            if low < coordinate < high:
                breakpoints.append((coordinate - start_value) / delta)
    return _unique_sorted_unit_values(breakpoints)


def _unique_sorted_unit_values(values: list[float]) -> tuple[float, ...]:
    sorted_values = sorted(values)
    unique: list[float] = []
    for value in sorted_values:
        clamped = min(max(value, 0.0), 1.0)
        if not unique or abs(clamped - unique[-1]) > 1.0e-12:
            unique.append(clamped)
    return tuple(unique)


def _cell_axis_index(coordinates: tuple[float, ...], value: float) -> int:
    if value < coordinates[0] - 1.0e-9 or value > coordinates[-1] + 1.0e-9:
        raise ValueError(f"Ray point coordinate {value} lies outside grid bounds.")
    if value <= coordinates[0]:
        return 0
    if value >= coordinates[-1]:
        return len(coordinates) - 2
    return max(0, min(bisect_right(coordinates, value) - 1, len(coordinates) - 2))


def _cell_coordinates(
    cell_index: int,
    nx_cells: int,
    ny_cells: int,
) -> tuple[int, int, int]:
    iz, remainder = divmod(cell_index, nx_cells * ny_cells)
    iy, ix = divmod(remainder, nx_cells)
    return ix, iy, iz


def _cell_shape(grid: CartesianVelocityGrid3D) -> tuple[int, int, int]:
    return (
        len(grid.x_coordinates_km) - 1,
        len(grid.y_coordinates_km) - 1,
        len(grid.z_coordinates_km) - 1,
    )


def _validate_grid_cells(grid: CartesianVelocityGrid3D) -> None:
    for label, coordinates in (
        ("x", grid.x_coordinates_km),
        ("y", grid.y_coordinates_km),
        ("z", grid.z_coordinates_km),
    ):
        if len(coordinates) < 2:
            raise ValueError(f"Grid {label} axis must contain at least two nodes.")
        if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
            raise ValueError(f"Grid {label} coordinates must be strictly increasing.")


def _sensitivity_metadata_payload(
    records: tuple[dict[str, Any], ...],
    ray_paths: tuple[TravelTimeRayPath, ...],
    grid: CartesianVelocityGrid3D,
    settings: BenchmarkSettings,
    sensitivity_path: Path,
    source_ray_path_sidecar: Path | None,
    validation: RayPathSensitivityValidationResult,
) -> dict[str, Any]:
    nx_cells, ny_cells, nz_cells = _cell_shape(grid)
    return {
        "artifact_type": "cartesian_ray_cell_sensitivity_matrix_sidecar_metadata",
        "schema_version": settings.dataset_export.schema_version,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "sensitivity_sidecar_path": sensitivity_path.as_posix(),
        "source_ray_path_sidecar": (
            source_ray_path_sidecar.as_posix() if source_ray_path_sidecar is not None else None
        ),
        "simulation_method": settings.simulation.method,
        "velocity_grid_id": grid.grid_id,
        "grid_shape_nodes": {
            "nx": len(grid.x_coordinates_km),
            "ny": len(grid.y_coordinates_km),
            "nz": len(grid.z_coordinates_km),
        },
        "grid_shape_cells": {
            "nx": nx_cells,
            "ny": ny_cells,
            "nz": nz_cells,
        },
        "cell_count": nx_cells * ny_cells * nz_cells,
        "coordinate_bounds_km": {
            "x": [grid.x_coordinates_km[0], grid.x_coordinates_km[-1]],
            "y": [grid.y_coordinates_km[0], grid.y_coordinates_km[-1]],
            "z": [grid.z_coordinates_km[0], grid.z_coordinates_km[-1]],
        },
        "cell_index_order": "x_fastest_index_ix_plus_nx_cells_times_iy_plus_ny_cells_times_iz",
        "record_count": len(records),
        "observation_count": len(ray_paths),
        "validation": {
            "path_length_conservation_passed": validation.passed,
            "checked_observation_count": validation.checked_observation_count,
            "max_abs_difference_km": validation.max_abs_difference_km,
            "errors": list(validation.errors),
        },
        "assumptions": [
            "Sensitivity cells are intervals between adjacent node-centered velocity-grid coordinates.",
            "Each sparse row stores geometric path length through one cell for one observation.",
            "Cell indexes use x-fastest order over the derived cell grid, not the node-centered velocity vector.",
            "Ray segments are split at Cartesian grid planes and subsegment length is assigned by midpoint cell membership.",
            "This artifact constructs the G-matrix geometry needed for later regularized travel-time inversion; it is not an inversion result.",
        ],
    }


def _interpolate_point(start: Point3D, end: Point3D, fraction: float) -> Point3D:
    return Point3D(
        x_km=start.x_km + (end.x_km - start.x_km) * fraction,
        y_km=start.y_km + (end.y_km - start.y_km) * fraction,
        z_km=start.z_km + (end.z_km - start.z_km) * fraction,
    )


def _ray_path_from_record(record: dict[str, Any]) -> TravelTimeRayPath:
    return TravelTimeRayPath(
        observation_id=str(record["observation_id"]),
        earthquake_id=str(record["earthquake_id"]),
        station_id=str(record["station_id"]),
        source=_point_from_record(record["source"]),
        receiver=_point_from_record(record["receiver"]),
        ray_path=tuple(_point_from_record(point) for point in record["ray_path"]),
        travel_time_s=float(record["travel_time_s"]),
        path_length_km=float(record["path_length_km"]),
        phase=str(record["phase"]),
        simulation_method=str(record["simulation_method"]),
        simulator_solver_type=str(record["simulator_solver_type"]),
        station_configuration_id=str(record["station_configuration_id"]),
        earthquake_configuration_id=str(record["earthquake_configuration_id"]),
        velocity_model_id=str(record["velocity_model_id"]),
        simulation_id=str(record["simulation_id"]),
        case_id=str(record["case_id"]) if record.get("case_id") is not None else None,
        converged=bool(record["converged"]),
        iteration_count=int(record["iteration_count"]),
    )


def _point_from_record(record: object) -> Point3D:
    if not isinstance(record, dict):
        raise TypeError("Expected point records to be JSON objects.")
    return Point3D(
        x_km=float(record["x_km"]),
        y_km=float(record["y_km"]),
        z_km=float(record["z_km"]),
    )
