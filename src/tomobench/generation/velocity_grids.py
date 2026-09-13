"""3D Cartesian velocity-grid generation for eikonal prototype workflows."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from tomobench.config.settings import (
    BlockAnomalySettings,
    DykeIntrusionSettings,
    FaultedGridSettings,
    SaltDomeSettings,
    BenchmarkSettings,
)
from tomobench.domain.schemas import (
    CartesianCellVelocityField3D,
    CartesianVelocityGrid3D,
    LayeredVelocityModel,
    dataclass_to_dict,
)
from tomobench.generation.velocity_models import velocity_at_depth

SUPPORTED_CARTESIAN_GRID_SCENARIOS = (
    "layered",
    "block_anomaly",
    "faulted",
    "salt_dome",
    "dyke_intrusion",
)


def build_cartesian_velocity_grid(
    settings: BenchmarkSettings,
    velocity_model: LayeredVelocityModel,
    scenario: str | None = None,
    *,
    spacing_by_axis_km: tuple[float, float, float] | None = None,
    max_grid_nodes: int | None = None,
    grid_role: str = "target_grid",
) -> CartesianVelocityGrid3D:
    """Build a node-centered 3D Cartesian grid for the configured scenario.

    ``spacing_by_axis_km`` makes the sampling grid explicit.  In particular, a
    forward-model grid can be sampled directly from the analytic scenario at a
    finer spacing than the ML target grid; it is not obtained by interpolating a
    previously built target grid.
    """
    selected_scenario = _normalize_grid_scenario(
        scenario or settings.velocity_model_generation.default_scenario
    )
    if spacing_by_axis_km is None:
        x_spacing_km = settings.eikonal_solver.x_grid_spacing_km
        y_spacing_km = settings.eikonal_solver.y_grid_spacing_km
        z_spacing_km = settings.eikonal_solver.z_grid_spacing_km
    else:
        if len(spacing_by_axis_km) != 3 or any(spacing <= 0.0 for spacing in spacing_by_axis_km):
            raise ValueError("spacing_by_axis_km must contain three positive values.")
        x_spacing_km, y_spacing_km, z_spacing_km = spacing_by_axis_km
    x_coordinates = _axis_coordinates(settings.study_area.x_range_km, x_spacing_km)
    y_coordinates = _axis_coordinates(settings.study_area.y_range_km, y_spacing_km)
    z_coordinates = _axis_coordinates(settings.study_area.z_range_km, z_spacing_km)
    node_count = len(x_coordinates) * len(y_coordinates) * len(z_coordinates)
    node_limit = (
        settings.eikonal_solver.max_grid_nodes
        if max_grid_nodes is None
        else max_grid_nodes
    )
    if node_limit <= 0:
        raise ValueError("max_grid_nodes must be positive.")
    if node_count > node_limit:
        raise ValueError(
            f"Eikonal prototype grid would contain {node_count} nodes, exceeding "
            f"configured max_grid_nodes={node_limit}."
        )

    velocities, scenario_metadata = _sample_scenario_values(
        settings,
        velocity_model,
        selected_scenario,
        x_coordinates,
        y_coordinates,
        z_coordinates,
    )

    metadata = {
        "artifact_type": "node_centered_cartesian_p_velocity_grid_3d",
        "generator_type": "analytic_scenario_sampling_at_requested_grid_nodes",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "coordinate_system": settings.study_area.coordinate_system,
        "grid_role": grid_role,
        "sampling_source": "analytic_layered_background_and_scenario_indicator_evaluated_directly",
        "grid_spacing_km": {
            "x": x_spacing_km,
            "y": y_spacing_km,
            "z": z_spacing_km,
        },
        "legacy_x_grid_spacing_km": x_spacing_km,
        "grid_scenario": selected_scenario,
        "node_count": node_count,
        "shape": {
            "nx": len(x_coordinates),
            "ny": len(y_coordinates),
            "nz": len(z_coordinates),
        },
        "storage_order": "x_fastest_index_ix_plus_nx_times_iy_plus_ny_times_iz",
        "velocity_representation": "node_centered",
        "source_model_type": "layered_1d_p_velocity_model",
        "layered_model": {
            "depth_boundaries_km": [layer.top_depth_km for layer in velocity_model.layers]
            + [velocity_model.layers[-1].bottom_depth_km],
            "velocities_km_per_s": [layer.p_velocity_km_per_s for layer in velocity_model.layers],
        },
        "assumptions": [
            "Velocity values are stored at grid nodes rather than cell centers.",
            "The base grid is sampled from the 1D layered P-wave model by depth only.",
            "Laterally heterogeneous scenarios are evaluated directly at the requested nodes.",
            "This grid may be a forward-model grid or an ML target grid; the roles are recorded explicitly.",
        ],
    }
    metadata.update(scenario_metadata)
    return CartesianVelocityGrid3D(
        grid_id=f"{velocity_model.model_id}_{selected_scenario}_cartesian_3d_grid",
        source_velocity_model_id=velocity_model.model_id,
        x_coordinates_km=x_coordinates,
        y_coordinates_km=y_coordinates,
        z_coordinates_km=z_coordinates,
        p_velocity_km_per_s=tuple(velocities),
        metadata=metadata,
    )


def build_cartesian_forward_grid(
    settings: BenchmarkSettings,
    velocity_model: LayeredVelocityModel,
    scenario: str | None = None,
    *,
    spacing_by_axis_km: tuple[float, float, float],
    max_grid_nodes: int = 5_000_000,
) -> CartesianVelocityGrid3D:
    """Sample the analytic scenario directly onto an independent forward grid."""
    return build_cartesian_velocity_grid(
        settings,
        velocity_model,
        scenario=scenario,
        spacing_by_axis_km=spacing_by_axis_km,
        max_grid_nodes=max_grid_nodes,
        grid_role="forward_solver_grid",
    )


def build_cartesian_cell_velocity_field(
    settings: BenchmarkSettings,
    velocity_model: LayeredVelocityModel,
    scenario: str | None = None,
    *,
    spacing_by_axis_km: tuple[float, float, float],
    max_grid_nodes: int = 5_000_000,
) -> CartesianCellVelocityField3D:
    """Sample the analytic scenario directly at cell centers.

    The returned axes are cell boundaries and the velocity values have one
    fewer sample along each axis.  Sharp structures are assigned by evaluating
    their analytic indicator at the cell center; no node-to-cell interpolation
    is performed.
    """
    selected_scenario = _normalize_grid_scenario(
        scenario or settings.velocity_model_generation.default_scenario
    )
    if len(spacing_by_axis_km) != 3 or any(spacing <= 0.0 for spacing in spacing_by_axis_km):
        raise ValueError("spacing_by_axis_km must contain three positive values.")
    x_spacing_km, y_spacing_km, z_spacing_km = spacing_by_axis_km
    x_coordinates = _axis_coordinates(settings.study_area.x_range_km, x_spacing_km)
    y_coordinates = _axis_coordinates(settings.study_area.y_range_km, y_spacing_km)
    z_coordinates = _axis_coordinates(settings.study_area.z_range_km, z_spacing_km)
    cell_count = (len(x_coordinates) - 1) * (len(y_coordinates) - 1) * (len(z_coordinates) - 1)
    if max_grid_nodes <= 0:
        raise ValueError("max_grid_nodes must be positive.")
    if cell_count > max_grid_nodes:
        raise ValueError(
            f"Cell-centered forward field would contain {cell_count} cells, exceeding "
            f"configured max_grid_nodes={max_grid_nodes}."
        )
    x_centers = _cell_centers(x_coordinates)
    y_centers = _cell_centers(y_coordinates)
    z_centers = _cell_centers(z_coordinates)
    velocities, scenario_metadata = _sample_scenario_values(
        settings,
        velocity_model,
        selected_scenario,
        x_centers,
        y_centers,
        z_centers,
    )
    metadata: dict[str, object] = {
        "artifact_type": "cell_centered_cartesian_p_velocity_field_3d",
        "generator_type": "analytic_scenario_sampling_at_cell_centers",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "coordinate_system": settings.study_area.coordinate_system,
        "grid_role": "forward_solver_grid",
        "sampling_source": "analytic_layered_background_and_scenario_indicator_evaluated_directly",
        "grid_spacing_km": {
            "x": x_spacing_km,
            "y": y_spacing_km,
            "z": z_spacing_km,
        },
        "grid_scenario": selected_scenario,
        "cell_count": cell_count,
        "shape": {
            "nx": len(x_coordinates) - 1,
            "ny": len(y_coordinates) - 1,
            "nz": len(z_coordinates) - 1,
        },
        "boundary_shape": {
            "nx": len(x_coordinates),
            "ny": len(y_coordinates),
            "nz": len(z_coordinates),
        },
        "storage_order": "x_fastest_index_ix_plus_nx_times_iy_plus_ny_times_iz",
        "velocity_representation": "cell_centered_velocity",
        "slowness_representation": "cell_centered_slowness_when_passed_to_ttcrpy",
        "source_model_type": "layered_1d_p_velocity_model",
        "layered_model": {
            "depth_boundaries_km": [layer.top_depth_km for layer in velocity_model.layers]
            + [velocity_model.layers[-1].bottom_depth_km],
            "velocities_km_per_s": [layer.p_velocity_km_per_s for layer in velocity_model.layers],
        },
        "assumptions": [
            "Velocity values are sampled independently at cell centers rather than interpolated from nodes.",
            "The cell field uses the boundary axes required by ttcrpy cell_slowness=True.",
            "Sharp scenario interfaces are assigned according to the analytic indicator at each cell center.",
        ],
    }
    metadata.update(scenario_metadata)
    return CartesianCellVelocityField3D(
        field_id=f"{velocity_model.model_id}_{selected_scenario}_cell_velocity_field_3d",
        source_velocity_model_id=velocity_model.model_id,
        x_coordinates_km=x_coordinates,
        y_coordinates_km=y_coordinates,
        z_coordinates_km=z_coordinates,
        p_velocity_km_per_s=tuple(velocities),
        metadata=metadata,
    )


def build_cartesian_velocity_grid_from_layered_model(
    settings: BenchmarkSettings,
    velocity_model: LayeredVelocityModel,
) -> CartesianVelocityGrid3D:
    """Backward-compatible helper for the currently configured grid scenario."""
    return build_cartesian_velocity_grid(settings, velocity_model)


def save_velocity_grid(grid: CartesianVelocityGrid3D, path: Path) -> Path:
    """Save a 3D Cartesian velocity grid artifact as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclass_to_dict(grid)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _base_layered_grid_velocities(
    velocity_model: LayeredVelocityModel,
    x_coordinates: tuple[float, ...],
    y_coordinates: tuple[float, ...],
    z_coordinates: tuple[float, ...],
) -> list[float]:
    velocities: list[float] = []
    for z_km in z_coordinates:
        velocity = velocity_at_depth(velocity_model, z_km)
        for _ in y_coordinates:
            for _ in x_coordinates:
                velocities.append(velocity)
    return velocities


def _sample_scenario_values(
    settings: BenchmarkSettings,
    velocity_model: LayeredVelocityModel,
    selected_scenario: str,
    x_coordinates: tuple[float, ...],
    y_coordinates: tuple[float, ...],
    z_coordinates: tuple[float, ...],
) -> tuple[list[float], dict[str, object]]:
    velocities = _base_layered_grid_velocities(
        velocity_model, x_coordinates, y_coordinates, z_coordinates
    )
    scenario_metadata: dict[str, object] = {}
    if selected_scenario == "block_anomaly":
        velocities, scenario_metadata = _apply_block_anomaly(
            velocities=velocities,
            x_coordinates=x_coordinates,
            y_coordinates=y_coordinates,
            z_coordinates=z_coordinates,
            settings=settings,
            anomaly=settings.velocity_model_generation.block_anomaly,
        )
    if selected_scenario == "faulted":
        velocities, scenario_metadata = _apply_faulted_offset(
            velocities=velocities,
            x_coordinates=x_coordinates,
            y_coordinates=y_coordinates,
            z_coordinates=z_coordinates,
            settings=settings,
            faulted=settings.velocity_model_generation.faulted,
        )
    if selected_scenario == "salt_dome":
        velocities, scenario_metadata = _apply_salt_dome(
            velocities=velocities,
            x_coordinates=x_coordinates,
            y_coordinates=y_coordinates,
            z_coordinates=z_coordinates,
            settings=settings,
            salt_dome=settings.velocity_model_generation.salt_dome,
        )
    if selected_scenario == "dyke_intrusion":
        velocities, scenario_metadata = _apply_dyke_intrusion(
            velocities=velocities,
            x_coordinates=x_coordinates,
            y_coordinates=y_coordinates,
            z_coordinates=z_coordinates,
            settings=settings,
            dyke=settings.velocity_model_generation.dyke_intrusion,
        )
    return velocities, scenario_metadata


def _cell_centers(coordinates: tuple[float, ...]) -> tuple[float, ...]:
    return tuple((left + right) / 2.0 for left, right in zip(coordinates, coordinates[1:]))


def _apply_block_anomaly(
    velocities: list[float],
    x_coordinates: tuple[float, ...],
    y_coordinates: tuple[float, ...],
    z_coordinates: tuple[float, ...],
    settings: BenchmarkSettings,
    anomaly: BlockAnomalySettings,
) -> tuple[list[float], dict[str, object]]:
    x_half, y_half, z_half = (size_component / 2.0 for size_component in anomaly.size_km)
    x_min = anomaly.center_km[0] - x_half
    x_max = anomaly.center_km[0] + x_half
    y_min = anomaly.center_km[1] - y_half
    y_max = anomaly.center_km[1] + y_half
    z_min = anomaly.center_km[2] - z_half
    z_max = anomaly.center_km[2] + z_half
    nx = len(x_coordinates)
    ny = len(y_coordinates)

    affected_nodes = 0
    for iz, z_km in enumerate(z_coordinates):
        if not (z_min <= z_km <= z_max):
            continue
        for iy, y_km in enumerate(y_coordinates):
            if not (y_min <= y_km <= y_max):
                continue
            for ix, x_km in enumerate(x_coordinates):
                if not (x_min <= x_km <= x_max):
                    continue
                index = ix + nx * (iy + ny * iz)
                updated_velocity = velocities[index] + anomaly.velocity_delta_km_per_s
                lower, upper = settings.velocity_model_generation.velocity_bounds_km_per_s
                if not lower <= updated_velocity <= upper:
                    raise ValueError(
                        "Block-anomaly velocity would fall outside configured velocity bounds."
                    )
                velocities[index] = updated_velocity
                affected_nodes += 1

    return velocities, {
        "grid_scenario": "block_anomaly",
        "block_anomaly": {
            "center_km": list(anomaly.center_km),
            "size_km": list(anomaly.size_km),
            "velocity_delta_km_per_s": anomaly.velocity_delta_km_per_s,
            "bounds_km": {
                "x": [x_min, x_max],
                "y": [y_min, y_max],
                "z": [z_min, z_max],
            },
            "affected_node_count": affected_nodes,
        },
    }


def _apply_faulted_offset(
    velocities: list[float],
    x_coordinates: tuple[float, ...],
    y_coordinates: tuple[float, ...],
    z_coordinates: tuple[float, ...],
    settings: BenchmarkSettings,
    faulted: FaultedGridSettings,
) -> tuple[list[float], dict[str, object]]:
    nx = len(x_coordinates)
    ny = len(y_coordinates)
    lower, upper = settings.velocity_model_generation.velocity_bounds_km_per_s
    affected_nodes = 0

    for iz, z_km in enumerate(z_coordinates):
        for iy, y_km in enumerate(y_coordinates):
            for ix, x_km in enumerate(x_coordinates):
                if not _is_fault_positive_side(x_km, y_km, z_km, faulted):
                    continue
                index = ix + nx * (iy + ny * iz)
                updated_velocity = velocities[index] + faulted.velocity_offset_km_per_s
                if not lower <= updated_velocity <= upper:
                    raise ValueError(
                        "Faulted-grid velocity would fall outside configured velocity bounds."
                    )
                velocities[index] = updated_velocity
                affected_nodes += 1

    return velocities, {
        "grid_scenario": "faulted",
        "faulted": {
            "fault_x_km": faulted.fault_x_km,
            "fault_y_km": faulted.fault_y_km,
            "strike_deg": faulted.strike_deg,
            "dip_deg": faulted.dip_deg,
            "dip_direction": faulted.dip_direction,
            "positive_side": faulted.positive_side,
            "velocity_offset_km_per_s": faulted.velocity_offset_km_per_s,
            "affected_node_count": affected_nodes,
        },
    }


def _apply_salt_dome(
    velocities: list[float],
    x_coordinates: tuple[float, ...],
    y_coordinates: tuple[float, ...],
    z_coordinates: tuple[float, ...],
    settings: BenchmarkSettings,
    salt_dome: SaltDomeSettings,
) -> tuple[list[float], dict[str, object]]:
    nx = len(x_coordinates)
    ny = len(y_coordinates)
    lower, upper = settings.velocity_model_generation.velocity_bounds_km_per_s
    affected_nodes = 0
    cx, cy, cz = salt_dome.center_km
    rx, ry, rz = salt_dome.radii_km

    for iz, z_km in enumerate(z_coordinates):
        normalized_z = ((z_km - cz) / rz) ** 2
        if normalized_z > 1.0:
            continue
        for iy, y_km in enumerate(y_coordinates):
            normalized_y = ((y_km - cy) / ry) ** 2
            if normalized_y + normalized_z > 1.0:
                continue
            for ix, x_km in enumerate(x_coordinates):
                normalized_x = ((x_km - cx) / rx) ** 2
                if normalized_x + normalized_y + normalized_z > 1.0:
                    continue
                index = ix + nx * (iy + ny * iz)
                updated_velocity = salt_dome.body_velocity_km_per_s
                if not lower <= updated_velocity <= upper:
                    raise ValueError(
                        "Salt-dome velocity would fall outside configured velocity bounds."
                    )
                velocities[index] = updated_velocity
                affected_nodes += 1

    return velocities, {
        "grid_scenario": "salt_dome",
        "salt_dome": {
            "center_km": list(salt_dome.center_km),
            "radii_km": list(salt_dome.radii_km),
            "body_velocity_km_per_s": salt_dome.body_velocity_km_per_s,
            "affected_node_count": affected_nodes,
        },
    }


def _apply_dyke_intrusion(
    velocities: list[float],
    x_coordinates: tuple[float, ...],
    y_coordinates: tuple[float, ...],
    z_coordinates: tuple[float, ...],
    settings: BenchmarkSettings,
    dyke: DykeIntrusionSettings,
) -> tuple[list[float], dict[str, object]]:
    nx = len(x_coordinates)
    ny = len(y_coordinates)
    lower, upper = settings.velocity_model_generation.velocity_bounds_km_per_s
    affected_nodes = 0
    strike_rad = math.radians(dyke.strike_deg)
    along_x = math.cos(strike_rad)
    along_y = math.sin(strike_rad)
    normal_x = -along_y
    normal_y = along_x

    for iz, z_km in enumerate(z_coordinates):
        if not dyke.top_depth_km <= z_km <= dyke.bottom_depth_km:
            continue
        for iy, y_km in enumerate(y_coordinates):
            for ix, x_km in enumerate(x_coordinates):
                dx = x_km - dyke.center_km[0]
                dy = y_km - dyke.center_km[1]
                along_distance = abs(dx * along_x + dy * along_y)
                normal_distance = abs(dx * normal_x + dy * normal_y)
                if along_distance > (dyke.length_km / 2.0):
                    continue
                if normal_distance > (dyke.width_km / 2.0):
                    continue
                index = ix + nx * (iy + ny * iz)
                updated_velocity = dyke.body_velocity_km_per_s
                if not lower <= updated_velocity <= upper:
                    raise ValueError(
                        "Dyke-intrusion velocity would fall outside configured velocity bounds."
                    )
                velocities[index] = updated_velocity
                affected_nodes += 1

    return velocities, {
        "grid_scenario": "dyke_intrusion",
        "dyke_intrusion": {
            "center_km": list(dyke.center_km),
            "strike_deg": dyke.strike_deg,
            "length_km": dyke.length_km,
            "width_km": dyke.width_km,
            "top_depth_km": dyke.top_depth_km,
            "bottom_depth_km": dyke.bottom_depth_km,
            "body_velocity_km_per_s": dyke.body_velocity_km_per_s,
            "affected_node_count": affected_nodes,
        },
    }


def _normalize_grid_scenario(scenario: str) -> str:
    normalized = scenario.strip().lower().replace("-", "_")
    if normalized not in SUPPORTED_CARTESIAN_GRID_SCENARIOS:
        raise ValueError(
            "Unsupported Cartesian velocity-grid scenario. "
            f"Supported scenarios: {', '.join(SUPPORTED_CARTESIAN_GRID_SCENARIOS)}."
        )
    return normalized


def fault_plane_horizontal_normal(faulted: FaultedGridSettings) -> tuple[float, float]:
    """Return the horizontal unit normal consistent with the configured strike."""
    strike_rad = math.radians(faulted.strike_deg)
    return math.sin(strike_rad), -math.cos(strike_rad)


def fault_dip_direction_sign(faulted: FaultedGridSettings) -> float:
    """Return the sign for down-dip movement along the horizontal normal."""
    if faulted.dip_direction == "positive_normal":
        return 1.0
    return -1.0


def fault_plane_signed_offset_km(
    x_km: float,
    y_km: float,
    z_km: float,
    faulted: FaultedGridSettings,
) -> float:
    """Signed offset from the dipping fault plane in kilometers."""
    normal_x, normal_y = fault_plane_horizontal_normal(faulted)
    horizontal_offset_km = 0.0
    if not math.isclose(faulted.dip_deg, 90.0, rel_tol=0.0, abs_tol=1.0e-9):
        horizontal_offset_km = z_km / math.tan(math.radians(faulted.dip_deg))
    return (
        (x_km - faulted.fault_x_km) * normal_x
        + (y_km - faulted.fault_y_km) * normal_y
        - fault_dip_direction_sign(faulted) * horizontal_offset_km
    )


def _is_fault_positive_side(
    x_km: float,
    y_km: float,
    z_km: float,
    faulted: FaultedGridSettings,
) -> bool:
    signed_offset = fault_plane_signed_offset_km(x_km, y_km, z_km, faulted)
    if faulted.positive_side == "greater_equal":
        return signed_offset >= 0.0
    return signed_offset <= 0.0


def _axis_coordinates(bounds_km: tuple[float, float], spacing_km: float) -> tuple[float, ...]:
    low, high = bounds_km
    count = int(math.floor((high - low) / spacing_km)) + 1
    coordinates = [round(low + index * spacing_km, 10) for index in range(count)]
    if not math.isclose(coordinates[-1], high, rel_tol=0.0, abs_tol=1.0e-9):
        coordinates.append(high)
    return tuple(coordinates)
