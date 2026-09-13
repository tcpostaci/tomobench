"""Frozen target-grid representation helpers for later inversion datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tomobench.config.settings import BenchmarkSettings
from tomobench.domain.schemas import CartesianVelocityGrid3D


def build_velocity_grid_target_spec(
    grid: CartesianVelocityGrid3D,
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    """Build the frozen ML target specification from the current grid artifact."""
    target_grid = settings.machine_learning.target_grid
    nx = len(grid.x_coordinates_km)
    ny = len(grid.y_coordinates_km)
    nz = len(grid.z_coordinates_km)
    return {
        "target_representation": settings.machine_learning.target_representation,
        "representation": target_grid.representation,
        "value_field": target_grid.value_field,
        "value_semantics": target_grid.value_semantics,
        "value_location": target_grid.value_location,
        "flattening_order": target_grid.flattening_order,
        "grid_id": grid.grid_id,
        "source_velocity_model_id": grid.source_velocity_model_id,
        "grid_shape": {"nx": nx, "ny": ny, "nz": nz},
        "target_vector_length": nx * ny * nz,
        "coordinate_system": grid.metadata["coordinate_system"],
        "grid_spacing_km": grid.metadata["grid_spacing_km"],
        "allowed_scenarios": list(target_grid.scenario_batch),
        "current_scenario": grid.metadata["grid_scenario"],
        "notes": [
            "This specification freezes the target-grid container rather than the full future set of geological scenarios.",
            "Target values are currently absolute node-centered P-wave velocities.",
            "The current first ML target batch is limited to layered, block-anomaly, and faulted scenarios.",
            "Later geometries may be added only if they preserve this same target contract.",
            "Expanded supervised-pairing cases may therefore use newer scenario families without changing the container fields above.",
        ],
    }


def save_velocity_grid_target_spec(spec: dict[str, Any], path: Path) -> Path:
    """Persist the frozen target-grid specification as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
