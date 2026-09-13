"""Interface-aware ray tracing helpers for synthetic P-wave travel times."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from tomobench.config.settings import PseudoBendingSolverSettings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import CartesianVelocityGrid3D

MappingLike = dict[str, Any]


@dataclass(frozen=True)
class PseudoBendingRayResult:
    """Result of one interface-aware pseudo-bending trace."""

    travel_time_s: float
    path_length_km: float
    ray_path: tuple[Point3D, ...]
    iteration_count: int
    converged: bool
    termination_reason: str = "unknown"


@dataclass(frozen=True)
class _InterfaceConstraint:
    kind: str
    metadata: MappingLike


@dataclass(frozen=True)
class _Waypoint:
    point: Point3D
    constraint: _InterfaceConstraint | None


class CartesianVelocityGridInterpolator:
    """Piecewise-constant velocity lookup for interface-aware ray tracing."""

    def __init__(self, grid: CartesianVelocityGrid3D) -> None:
        self.grid = grid
        self.x_coordinates = grid.x_coordinates_km
        self.y_coordinates = grid.y_coordinates_km
        self.z_coordinates = grid.z_coordinates_km
        self.nx = len(self.x_coordinates)
        self.ny = len(self.y_coordinates)
        self.nz = len(self.z_coordinates)
        expected_node_count = self.nx * self.ny * self.nz
        if len(grid.p_velocity_km_per_s) != expected_node_count:
            raise ValueError("Velocity grid shape does not match flattened velocity count.")
        self.layers = _layer_specs_from_grid(grid)

    def velocity_at(self, point: Point3D) -> float:
        """Return piecewise-constant velocity at a clamped model point."""
        clamped = self.clamp_point(point)
        return _velocity_for_point(clamped, self.grid, self.layers)

    def clamp_point(self, point: Point3D) -> Point3D:
        """Clamp a point to the velocity-grid domain."""
        return Point3D(
            x_km=_clamp(point.x_km, self.x_coordinates[0], self.x_coordinates[-1]),
            y_km=_clamp(point.y_km, self.y_coordinates[0], self.y_coordinates[-1]),
            z_km=_clamp(point.z_km, self.z_coordinates[0], self.z_coordinates[-1]),
        )


def trace_pseudo_bending_ray(
    source: Point3D,
    receiver: Point3D,
    velocity_grid: CartesianVelocityGrid3D,
    settings: PseudoBendingSolverSettings,
) -> PseudoBendingRayResult:
    """Trace an interface-aware, piecewise-straight pseudo-bending ray.

    The initial path is a Snell-law ray-parameter solution through the horizontal
    layered background. Additional geological-family interface crossing points
    are inserted and then constrained to slide along their surfaces while total
    piecewise-constant travel time is minimized.
    """
    interpolator = CartesianVelocityGridInterpolator(velocity_grid)
    source = interpolator.clamp_point(source)
    receiver = interpolator.clamp_point(receiver)
    waypoints = _initial_waypoints(source, receiver, velocity_grid, interpolator)
    return _trace_from_waypoints(waypoints, interpolator, velocity_grid, settings)


def trace_pseudo_bending_ray_with_initial_offset(
    source: Point3D,
    receiver: Point3D,
    velocity_grid: CartesianVelocityGrid3D,
    settings: PseudoBendingSolverSettings,
    offset_km: tuple[float, float, float],
) -> PseudoBendingRayResult:
    """Trace from an explicitly offset interface-waypoint initialization.

    This diagnostic entry point keeps the configured solver unchanged while making
    multi-start experiments reproducible. Interior interface waypoints are shifted
    by the supplied vector and projected back to their constraints before local
    optimization begins.
    """

    interpolator = CartesianVelocityGridInterpolator(velocity_grid)
    source = interpolator.clamp_point(source)
    receiver = interpolator.clamp_point(receiver)
    waypoints = _initial_waypoints(source, receiver, velocity_grid, interpolator)
    offset_waypoints: list[_Waypoint] = [waypoints[0]]
    for waypoint in waypoints[1:-1]:
        if waypoint.constraint is None:
            offset_waypoints.append(waypoint)
            continue
        moved = Point3D(
            waypoint.point.x_km + float(offset_km[0]),
            waypoint.point.y_km + float(offset_km[1]),
            waypoint.point.z_km + float(offset_km[2]),
        )
        projected = _project_to_constraint(interpolator.clamp_point(moved), waypoint.constraint, velocity_grid)
        offset_waypoints.append(_Waypoint(projected, waypoint.constraint))
    offset_waypoints.append(waypoints[-1])
    return _trace_from_waypoints(
        _dedupe_consecutive_waypoints(tuple(offset_waypoints)),
        interpolator,
        velocity_grid,
        settings,
    )


def trace_pseudo_bending_ray_with_restarts(
    source: Point3D,
    receiver: Point3D,
    velocity_grid: CartesianVelocityGrid3D,
    settings: PseudoBendingSolverSettings,
    offsets_km: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (2.5, 0.0, 0.0),
        (-2.5, 0.0, 0.0),
        (0.0, 2.5, 0.0),
        (0.0, -2.5, 0.0),
    ),
) -> PseudoBendingRayResult:
    """Run deterministic multi-start traces and select the best valid candidate."""

    candidates = tuple(
        trace_pseudo_bending_ray_with_initial_offset(
            source=source,
            receiver=receiver,
            velocity_grid=velocity_grid,
            settings=settings,
            offset_km=offset,
        )
        for offset in offsets_km
    )
    converged_candidates = tuple(candidate for candidate in candidates if candidate.converged)
    pool = converged_candidates or candidates
    selected = min(pool, key=lambda candidate: candidate.travel_time_s)
    selected_index = candidates.index(selected)
    return PseudoBendingRayResult(
        travel_time_s=selected.travel_time_s,
        path_length_km=selected.path_length_km,
        ray_path=selected.ray_path,
        iteration_count=selected.iteration_count,
        converged=selected.converged,
        termination_reason=(
            f"restart_{selected_index}:{selected.termination_reason}"
            if len(candidates) > 1
            else selected.termination_reason
        ),
    )


def _initial_waypoints(
    source: Point3D,
    receiver: Point3D,
    velocity_grid: CartesianVelocityGrid3D,
    interpolator: CartesianVelocityGridInterpolator,
) -> tuple[_Waypoint, ...]:
    waypoints = _layered_snell_waypoints(source, receiver, interpolator.layers)
    return _insert_scenario_interface_crossings(waypoints, velocity_grid)


def _trace_from_waypoints(
    waypoints: tuple[_Waypoint, ...],
    interpolator: CartesianVelocityGridInterpolator,
    velocity_grid: CartesianVelocityGrid3D,
    settings: PseudoBendingSolverSettings,
) -> PseudoBendingRayResult:
    waypoints, iterations, converged, termination_reason = _optimize_interface_waypoints(
        waypoints,
        interpolator,
        velocity_grid,
        settings,
    )
    path = tuple(waypoint.point for waypoint in waypoints)
    return PseudoBendingRayResult(
        travel_time_s=polyline_travel_time_s(path, interpolator),
        path_length_km=polyline_length_km(path),
        ray_path=path,
        iteration_count=iterations,
        converged=converged,
        termination_reason=termination_reason,
    )


def polyline_travel_time_s(
    path: tuple[Point3D, ...],
    interpolator: CartesianVelocityGridInterpolator,
) -> float:
    """Return travel time through piecewise constant regions along a polyline."""
    return sum(
        _segment_travel_time_s(start, end, interpolator) for start, end in zip(path, path[1:])
    )


def polyline_length_km(path: tuple[Point3D, ...]) -> float:
    """Return total length of a piecewise-linear ray."""
    return sum(_distance_km(start, end) for start, end in zip(path, path[1:]))


def planned_ray_tracing_note() -> str:
    """Return the current scope note for the ray-tracing module."""
    return (
        "The pseudo_bending_3d backend is interface-aware: it uses a Snell-law "
        "layered backbone and constrained crossing-point optimization on explicit "
        "synthetic geological interfaces."
    )


def _optimize_interface_waypoints(
    waypoints: tuple[_Waypoint, ...],
    interpolator: CartesianVelocityGridInterpolator,
    grid: CartesianVelocityGrid3D,
    settings: PseudoBendingSolverSettings,
) -> tuple[tuple[_Waypoint, ...], int, bool, str]:
    if len(waypoints) <= 2:
        return waypoints, 0, True, "no_interface_waypoints"

    current = list(waypoints)
    initial_step = max(settings.perturbation_step_km, _minimum_grid_spacing_km(grid))
    min_step = settings.min_perturbation_step_km
    iterations = 0
    converged = False
    termination_reason = "max_iterations_reached"

    for iterations in range(1, settings.max_iterations_per_level + 1):
        old_time = _waypoint_travel_time_s(current, interpolator)
        improved = False
        step = float(initial_step)
        while step >= min_step:
            pass_improved = False
            for index in range(1, len(current) - 1):
                constraint = current[index].constraint
                if constraint is None:
                    continue
                replacement = _best_local_relocation(
                    previous=current[index - 1].point,
                    waypoint=current[index],
                    next_point=current[index + 1].point,
                    interpolator=interpolator,
                    grid=grid,
                    step_km=step,
                )
                if replacement.point != current[index].point:
                    current[index] = replacement
                    pass_improved = True
            improved = improved or pass_improved
            if pass_improved:
                break
            step *= 0.5

        new_time = _waypoint_travel_time_s(current, interpolator)
        improvement = old_time - new_time
        if not improved:
            converged = True
            termination_reason = "no_improvement"
            break
        if improvement <= settings.convergence_tolerance_s:
            converged = True
            termination_reason = "convergence_tolerance"
            break

    return tuple(current), iterations, converged, termination_reason


def _best_local_relocation(
    previous: Point3D,
    waypoint: _Waypoint,
    next_point: Point3D,
    interpolator: CartesianVelocityGridInterpolator,
    grid: CartesianVelocityGrid3D,
    step_km: float,
) -> _Waypoint:
    assert waypoint.constraint is not None
    best_point = waypoint.point
    best_time = _local_travel_time_s(previous, waypoint.point, next_point, interpolator)
    for direction in _constraint_tangent_directions(waypoint.point, waypoint.constraint):
        for sign in (-1.0, 1.0):
            moved = Point3D(
                x_km=waypoint.point.x_km + sign * step_km * direction[0],
                y_km=waypoint.point.y_km + sign * step_km * direction[1],
                z_km=waypoint.point.z_km + sign * step_km * direction[2],
            )
            candidate = _project_to_constraint(
                interpolator.clamp_point(moved),
                waypoint.constraint,
                grid,
            )
            candidate_time = _local_travel_time_s(previous, candidate, next_point, interpolator)
            if candidate_time + 1.0e-12 < best_time:
                best_point = candidate
                best_time = candidate_time
    return _Waypoint(best_point, waypoint.constraint)


def _waypoint_travel_time_s(
    waypoints: list[_Waypoint],
    interpolator: CartesianVelocityGridInterpolator,
) -> float:
    return polyline_travel_time_s(tuple(waypoint.point for waypoint in waypoints), interpolator)


def _local_travel_time_s(
    previous: Point3D,
    current: Point3D,
    next_point: Point3D,
    interpolator: CartesianVelocityGridInterpolator,
) -> float:
    return _segment_travel_time_s(previous, current, interpolator) + _segment_travel_time_s(
        current,
        next_point,
        interpolator,
    )


def _layered_snell_waypoints(
    source: Point3D,
    receiver: Point3D,
    layers: tuple[tuple[float, float, float], ...],
) -> tuple[_Waypoint, ...]:
    horizontal_offset = math.hypot(source.x_km - receiver.x_km, source.y_km - receiver.y_km)
    vertical_span = abs(source.z_km - receiver.z_km)
    if horizontal_offset == 0.0 or vertical_span == 0.0:
        return (_Waypoint(source, None), _Waypoint(receiver, None))

    ux = (receiver.x_km - source.x_km) / horizontal_offset
    uy = (receiver.y_km - source.y_km) / horizontal_offset
    z_values = _crossed_depth_values(source.z_km, receiver.z_km, layers)
    segments = []
    for z0, z1 in zip(z_values, z_values[1:]):
        thickness = abs(z1 - z0)
        velocity = _base_layer_velocity_at((z0 + z1) / 2.0, layers)
        segments.append((thickness, velocity))

    p = _solve_layered_ray_parameter(horizontal_offset, tuple(segments))
    waypoints = [_Waypoint(source, None)]
    current_x = source.x_km
    current_y = source.y_km
    for z_next, (thickness, velocity) in zip(z_values[1:], segments):
        segment_horizontal = (
            thickness * p * velocity / math.sqrt(max(1.0 - (p * velocity) ** 2, 1.0e-15))
        )
        current_x += ux * segment_horizontal
        current_y += uy * segment_horizontal
        point = Point3D(current_x, current_y, z_next)
        constraint = (
            None
            if z_next == receiver.z_km
            else _InterfaceConstraint(
                "horizontal",
                {"z_km": z_next},
            )
        )
        waypoints.append(_Waypoint(point, constraint))
    waypoints[-1] = _Waypoint(receiver, None)
    return tuple(waypoints)


def _solve_layered_ray_parameter(
    horizontal_offset: float,
    segments: tuple[tuple[float, float], ...],
) -> float:
    max_velocity = max(velocity for _, velocity in segments)
    p_low = 0.0
    p_high = (1.0 / max_velocity) * (1.0 - 1.0e-12)
    for _ in range(120):
        p_mid = (p_low + p_high) / 2.0
        modeled_offset = sum(
            thickness * p_mid * velocity / math.sqrt(max(1.0 - (p_mid * velocity) ** 2, 1.0e-15))
            for thickness, velocity in segments
        )
        if modeled_offset < horizontal_offset:
            p_low = p_mid
        else:
            p_high = p_mid
    return (p_low + p_high) / 2.0


def _crossed_depth_values(
    source_z_km: float,
    receiver_z_km: float,
    layers: tuple[tuple[float, float, float], ...],
) -> tuple[float, ...]:
    low = min(source_z_km, receiver_z_km)
    high = max(source_z_km, receiver_z_km)
    boundaries = sorted(
        {
            boundary
            for top, bottom, _ in layers
            for boundary in (top, bottom)
            if low < boundary < high
        }
    )
    if receiver_z_km < source_z_km:
        boundaries.reverse()
    return tuple([source_z_km, *boundaries, receiver_z_km])


def _insert_scenario_interface_crossings(
    waypoints: tuple[_Waypoint, ...],
    grid: CartesianVelocityGrid3D,
) -> tuple[_Waypoint, ...]:
    inserted: list[_Waypoint] = [waypoints[0]]
    for start, end in zip(waypoints, waypoints[1:]):
        crossings = sorted(
            _segment_interface_crossings(start.point, end.point, grid),
            key=lambda crossing: crossing[0],
        )
        for parameter, constraint in crossings:
            if 1.0e-7 < parameter < 1.0 - 1.0e-7:
                crossing_point = _project_to_constraint(
                    _interpolate_point(start.point, end.point, parameter),
                    constraint,
                    grid,
                )
                inserted.append(_Waypoint(crossing_point, constraint))
        inserted.append(end)
    return _dedupe_consecutive_waypoints(tuple(inserted))


def _segment_interface_crossings(
    start: Point3D,
    end: Point3D,
    grid: CartesianVelocityGrid3D,
) -> tuple[tuple[float, _InterfaceConstraint], ...]:
    scenario = str(grid.metadata.get("grid_scenario", "layered"))
    if scenario == "block_anomaly":
        return _line_box_intersection_crossings(
            start,
            end,
            grid.metadata["block_anomaly"]["bounds_km"],  # type: ignore[index]
            "box_face",
        )
    if scenario == "salt_dome":
        metadata = grid.metadata["salt_dome"]  # type: ignore[assignment]
        return tuple(
            (parameter, _InterfaceConstraint("ellipsoid", metadata))
            for parameter in _line_ellipsoid_intersection_parameters(start, end, metadata)
        )
    if scenario == "dyke_intrusion":
        return _line_dyke_intersection_crossings(
            start,
            end,
            grid.metadata["dyke_intrusion"],  # type: ignore[arg-type]
        )
    if scenario == "faulted":
        metadata = grid.metadata["faulted"]  # type: ignore[assignment]
        parameter = _line_fault_plane_parameter(start, end, metadata)
        return (
            ()
            if parameter is None
            else ((parameter, _InterfaceConstraint("fault_plane", metadata)),)
        )
    return ()


def _line_box_intersection_crossings(
    start: Point3D,
    end: Point3D,
    bounds: MappingLike,
    kind: str,
) -> tuple[tuple[float, _InterfaceConstraint], ...]:
    axis_bounds = {
        "x": tuple(float(value) for value in bounds["x"]),
        "y": tuple(float(value) for value in bounds["y"]),
        "z": tuple(float(value) for value in bounds["z"]),
    }
    parameters = _line_convex_slab_parameters(
        start,
        end,
        coordinate_getters={
            "x": lambda point: point.x_km,
            "y": lambda point: point.y_km,
            "z": lambda point: point.z_km,
        },
        axis_bounds=axis_bounds,
    )
    crossings: list[tuple[float, _InterfaceConstraint]] = []
    for parameter in parameters:
        point = _interpolate_point(start, end, parameter)
        face_axis, face_value = _nearest_box_face(point, axis_bounds)
        crossings.append(
            (
                parameter,
                _InterfaceConstraint(
                    kind,
                    {"bounds": axis_bounds, "face_axis": face_axis, "face_value": face_value},
                ),
            )
        )
    return tuple(crossings)


def _line_dyke_intersection_crossings(
    start: Point3D,
    end: Point3D,
    metadata: MappingLike,
) -> tuple[tuple[float, _InterfaceConstraint], ...]:
    local = _dyke_local_frame(metadata)
    axis_bounds = {
        "along": (-float(metadata["length_km"]) / 2.0, float(metadata["length_km"]) / 2.0),
        "normal": (-float(metadata["width_km"]) / 2.0, float(metadata["width_km"]) / 2.0),
        "z": (float(metadata["top_depth_km"]), float(metadata["bottom_depth_km"])),
    }
    parameters = _line_convex_slab_parameters(
        start,
        end,
        coordinate_getters={
            "along": lambda point: _dyke_coordinates(point, local)[0],
            "normal": lambda point: _dyke_coordinates(point, local)[1],
            "z": lambda point: point.z_km,
        },
        axis_bounds=axis_bounds,
    )
    crossings: list[tuple[float, _InterfaceConstraint]] = []
    for parameter in parameters:
        point = _interpolate_point(start, end, parameter)
        along, normal, z_km = _dyke_coordinates(point, local)
        face_axis, face_value = _nearest_box_face_values(
            {"along": along, "normal": normal, "z": z_km},
            axis_bounds,
        )
        crossings.append(
            (
                parameter,
                _InterfaceConstraint(
                    "dyke_face",
                    {
                        "dyke": metadata,
                        "face_axis": face_axis,
                        "face_value": face_value,
                    },
                ),
            )
        )
    return tuple(crossings)


def _line_convex_slab_parameters(
    start: Point3D,
    end: Point3D,
    coordinate_getters: dict[str, Any],
    axis_bounds: dict[str, tuple[float, float]],
) -> tuple[float, ...]:
    t_min = 0.0
    t_max = 1.0
    for axis, getter in coordinate_getters.items():
        low, high = axis_bounds[axis]
        start_value = getter(start)
        end_value = getter(end)
        delta = end_value - start_value
        if abs(delta) < 1.0e-12:
            if not low <= start_value <= high:
                return ()
            continue
        first = (low - start_value) / delta
        second = (high - start_value) / delta
        near = min(first, second)
        far = max(first, second)
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max:
            return ()
    return tuple(t for t in (t_min, t_max) if 0.0 < t < 1.0)


def _nearest_box_face(
    point: Point3D,
    axis_bounds: dict[str, tuple[float, float]],
) -> tuple[str, float]:
    return _nearest_box_face_values(
        {"x": point.x_km, "y": point.y_km, "z": point.z_km},
        axis_bounds,
    )


def _nearest_box_face_values(
    values: dict[str, float],
    axis_bounds: dict[str, tuple[float, float]],
) -> tuple[str, float]:
    candidates = []
    for axis, value in values.items():
        low, high = axis_bounds[axis]
        candidates.append((abs(value - low), axis, low))
        candidates.append((abs(value - high), axis, high))
    _, axis, face_value = min(candidates, key=lambda item: item[0])
    return axis, face_value


def _line_ellipsoid_intersection_parameters(
    start: Point3D,
    end: Point3D,
    metadata: MappingLike,
) -> tuple[float, ...]:
    cx, cy, cz = (float(value) for value in metadata["center_km"])
    rx, ry, rz = (float(value) for value in metadata["radii_km"])
    dx = end.x_km - start.x_km
    dy = end.y_km - start.y_km
    dz = end.z_km - start.z_km
    sx = start.x_km - cx
    sy = start.y_km - cy
    sz = start.z_km - cz
    a = (dx / rx) ** 2 + (dy / ry) ** 2 + (dz / rz) ** 2
    b = 2.0 * (sx * dx / (rx * rx) + sy * dy / (ry * ry) + sz * dz / (rz * rz))
    c = (sx / rx) ** 2 + (sy / ry) ** 2 + (sz / rz) ** 2 - 1.0
    return _quadratic_unit_interval_roots(a, b, c)


def _line_fault_plane_parameter(
    start: Point3D,
    end: Point3D,
    metadata: MappingLike,
) -> float | None:
    start_value = _fault_signed_offset(start, metadata)
    end_value = _fault_signed_offset(end, metadata)
    denominator = start_value - end_value
    if abs(denominator) < 1.0e-12:
        return None
    parameter = start_value / denominator
    if 0.0 < parameter < 1.0:
        return parameter
    return None


def _quadratic_unit_interval_roots(a: float, b: float, c: float) -> tuple[float, ...]:
    if abs(a) < 1.0e-12:
        if abs(b) < 1.0e-12:
            return ()
        root = -c / b
        return (root,) if 0.0 < root < 1.0 else ()
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return ()
    sqrt_discriminant = math.sqrt(discriminant)
    roots = (
        (-b - sqrt_discriminant) / (2.0 * a),
        (-b + sqrt_discriminant) / (2.0 * a),
    )
    return tuple(root for root in roots if 0.0 < root < 1.0)


def _constraint_tangent_directions(
    point: Point3D,
    constraint: _InterfaceConstraint,
) -> tuple[tuple[float, float, float], ...]:
    if constraint.kind == "horizontal":
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    if constraint.kind in ("fault_plane",):
        normal = _fault_normal(constraint.metadata)
        return _orthonormal_tangent_basis(normal)
    if constraint.kind == "ellipsoid":
        normal = _ellipsoid_normal(point, constraint.metadata)
        return _orthonormal_tangent_basis(normal)
    if constraint.kind == "box_face":
        axis = str(constraint.metadata["face_axis"])
        if axis == "x":
            return ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        if axis == "y":
            return ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    if constraint.kind == "dyke_face":
        local = _dyke_local_frame(constraint.metadata["dyke"])
        axis = str(constraint.metadata["face_axis"])
        if axis == "along":
            return (local["normal_vector"], (0.0, 0.0, 1.0))
        if axis == "normal":
            return (local["along_vector"], (0.0, 0.0, 1.0))
        return (local["along_vector"], local["normal_vector"])
    return ()


def _project_to_constraint(
    point: Point3D,
    constraint: _InterfaceConstraint,
    grid: CartesianVelocityGrid3D,
) -> Point3D:
    if constraint.kind == "horizontal":
        return _clamp_to_grid(
            Point3D(point.x_km, point.y_km, float(constraint.metadata["z_km"])),
            grid,
        )
    if constraint.kind == "fault_plane":
        return _clamp_to_grid(_project_to_fault_plane(point, constraint.metadata), grid)
    if constraint.kind == "ellipsoid":
        return _clamp_to_grid(_project_to_ellipsoid(point, constraint.metadata), grid)
    if constraint.kind == "box_face":
        return _clamp_to_grid(_project_to_box_face(point, constraint.metadata), grid)
    if constraint.kind == "dyke_face":
        return _clamp_to_grid(_project_to_dyke_face(point, constraint.metadata), grid)
    return _clamp_to_grid(point, grid)


def _project_to_box_face(point: Point3D, metadata: MappingLike) -> Point3D:
    bounds = metadata["bounds"]
    axis = str(metadata["face_axis"])
    values = {
        "x": _clamp(point.x_km, bounds["x"][0], bounds["x"][1]),
        "y": _clamp(point.y_km, bounds["y"][0], bounds["y"][1]),
        "z": _clamp(point.z_km, bounds["z"][0], bounds["z"][1]),
    }
    values[axis] = float(metadata["face_value"])
    return Point3D(values["x"], values["y"], values["z"])


def _project_to_dyke_face(point: Point3D, metadata: MappingLike) -> Point3D:
    dyke = metadata["dyke"]
    local = _dyke_local_frame(dyke)
    along, normal, z_km = _dyke_coordinates(point, local)
    bounds = {
        "along": (-float(dyke["length_km"]) / 2.0, float(dyke["length_km"]) / 2.0),
        "normal": (-float(dyke["width_km"]) / 2.0, float(dyke["width_km"]) / 2.0),
        "z": (float(dyke["top_depth_km"]), float(dyke["bottom_depth_km"])),
    }
    values = {
        "along": _clamp(along, bounds["along"][0], bounds["along"][1]),
        "normal": _clamp(normal, bounds["normal"][0], bounds["normal"][1]),
        "z": _clamp(z_km, bounds["z"][0], bounds["z"][1]),
    }
    values[str(metadata["face_axis"])] = float(metadata["face_value"])
    return _point_from_dyke_coordinates(values["along"], values["normal"], values["z"], local)


def _project_to_ellipsoid(point: Point3D, metadata: MappingLike) -> Point3D:
    cx, cy, cz = (float(value) for value in metadata["center_km"])
    rx, ry, rz = (float(value) for value in metadata["radii_km"])
    dx = point.x_km - cx
    dy = point.y_km - cy
    dz = point.z_km - cz
    scale = math.sqrt((dx / rx) ** 2 + (dy / ry) ** 2 + (dz / rz) ** 2)
    if scale == 0.0:
        return Point3D(cx + rx, cy, cz)
    return Point3D(cx + dx / scale, cy + dy / scale, cz + dz / scale)


def _project_to_fault_plane(point: Point3D, metadata: MappingLike) -> Point3D:
    normal = _fault_normal(metadata)
    signed = _fault_signed_offset(point, metadata)
    return Point3D(
        point.x_km - signed * normal[0],
        point.y_km - signed * normal[1],
        point.z_km - signed * normal[2],
    )


def _fault_normal(metadata: MappingLike) -> tuple[float, float, float]:
    strike = math.radians(float(metadata["strike_deg"]))
    normal_x = math.sin(strike)
    normal_y = -math.cos(strike)
    dip_deg = float(metadata["dip_deg"])
    normal_z = 0.0
    if not math.isclose(dip_deg, 90.0, rel_tol=0.0, abs_tol=1.0e-9):
        dip_direction_sign = 1.0 if metadata["dip_direction"] == "positive_normal" else -1.0
        normal_z = -dip_direction_sign / math.tan(math.radians(dip_deg))
    return _normalize((normal_x, normal_y, normal_z))


def _ellipsoid_normal(point: Point3D, metadata: MappingLike) -> tuple[float, float, float]:
    cx, cy, cz = (float(value) for value in metadata["center_km"])
    rx, ry, rz = (float(value) for value in metadata["radii_km"])
    return _normalize(
        (
            (point.x_km - cx) / (rx * rx),
            (point.y_km - cy) / (ry * ry),
            (point.z_km - cz) / (rz * rz),
        )
    )


def _orthonormal_tangent_basis(
    normal: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    reference = (0.0, 0.0, 1.0)
    if abs(_dot(normal, reference)) > 0.9:
        reference = (1.0, 0.0, 0.0)
    first = _normalize(_cross(normal, reference))
    second = _normalize(_cross(normal, first))
    return first, second


def _segment_travel_time_s(
    start: Point3D,
    end: Point3D,
    interpolator: CartesianVelocityGridInterpolator,
) -> float:
    length_km = _distance_km(start, end)
    if length_km == 0.0:
        return 0.0
    midpoint = _interpolate_point(start, end, 0.5)
    return length_km / interpolator.velocity_at(midpoint)


def _velocity_for_point(
    point: Point3D,
    grid: CartesianVelocityGrid3D,
    layers: tuple[tuple[float, float, float], ...],
) -> float:
    velocity = _base_layer_velocity_at(point.z_km, layers)
    scenario = str(grid.metadata.get("grid_scenario", "layered"))
    if scenario == "block_anomaly" and _inside_block(point, grid.metadata["block_anomaly"]):  # type: ignore[arg-type]
        velocity += float(grid.metadata["block_anomaly"]["velocity_delta_km_per_s"])  # type: ignore[index]
    if scenario == "salt_dome" and _inside_ellipsoid(point, grid.metadata["salt_dome"]):  # type: ignore[arg-type]
        velocity = float(grid.metadata["salt_dome"]["body_velocity_km_per_s"])  # type: ignore[index]
    if scenario == "dyke_intrusion" and _inside_dyke(point, grid.metadata["dyke_intrusion"]):  # type: ignore[arg-type]
        velocity = float(grid.metadata["dyke_intrusion"]["body_velocity_km_per_s"])  # type: ignore[index]
    if scenario == "faulted" and _inside_faulted_side(point, grid.metadata["faulted"]):  # type: ignore[arg-type]
        velocity += float(grid.metadata["faulted"]["velocity_offset_km_per_s"])  # type: ignore[index]
    return velocity


def _base_layer_velocity_at(
    depth_km: float,
    layers: tuple[tuple[float, float, float], ...],
) -> float:
    for top, bottom, velocity in layers:
        if top <= depth_km < bottom:
            return velocity
    deepest = layers[-1]
    if math.isclose(depth_km, deepest[1], rel_tol=0.0, abs_tol=1.0e-9):
        return deepest[2]
    raise ValueError(f"Depth {depth_km} km is outside the layered velocity model.")


def _layer_specs_from_grid(grid: CartesianVelocityGrid3D) -> tuple[tuple[float, float, float], ...]:
    layered_model = grid.metadata.get("layered_model")
    if isinstance(layered_model, dict):
        boundaries = tuple(float(value) for value in layered_model["depth_boundaries_km"])
        velocities = tuple(float(value) for value in layered_model["velocities_km_per_s"])
        return tuple(
            (boundaries[index], boundaries[index + 1], velocities[index])
            for index in range(len(velocities))
        )
    return (
        (
            grid.z_coordinates_km[0],
            grid.z_coordinates_km[-1],
            sum(float(value) for value in grid.p_velocity_km_per_s) / len(grid.p_velocity_km_per_s),
        ),
    )


def _inside_block(point: Point3D, metadata: MappingLike) -> bool:
    bounds = metadata["bounds_km"]
    return (
        float(bounds["x"][0]) <= point.x_km <= float(bounds["x"][1])
        and float(bounds["y"][0]) <= point.y_km <= float(bounds["y"][1])
        and float(bounds["z"][0]) <= point.z_km <= float(bounds["z"][1])
    )


def _inside_ellipsoid(point: Point3D, metadata: MappingLike) -> bool:
    cx, cy, cz = (float(value) for value in metadata["center_km"])
    rx, ry, rz = (float(value) for value in metadata["radii_km"])
    return ((point.x_km - cx) / rx) ** 2 + ((point.y_km - cy) / ry) ** 2 + (
        (point.z_km - cz) / rz
    ) ** 2 <= 1.0


def _inside_dyke(point: Point3D, metadata: MappingLike) -> bool:
    local = _dyke_local_frame(metadata)
    along, normal, z_km = _dyke_coordinates(point, local)
    return (
        abs(along) <= float(metadata["length_km"]) / 2.0
        and abs(normal) <= float(metadata["width_km"]) / 2.0
        and float(metadata["top_depth_km"]) <= z_km <= float(metadata["bottom_depth_km"])
    )


def _inside_faulted_side(point: Point3D, metadata: MappingLike) -> bool:
    signed_offset = _fault_signed_offset(point, metadata)
    if metadata["positive_side"] == "greater_equal":
        return signed_offset >= 0.0
    return signed_offset <= 0.0


def _fault_signed_offset(point: Point3D, metadata: MappingLike) -> float:
    strike = math.radians(float(metadata["strike_deg"]))
    normal_x = math.sin(strike)
    normal_y = -math.cos(strike)
    horizontal_offset_km = 0.0
    dip_deg = float(metadata["dip_deg"])
    if not math.isclose(dip_deg, 90.0, rel_tol=0.0, abs_tol=1.0e-9):
        horizontal_offset_km = point.z_km / math.tan(math.radians(dip_deg))
    dip_direction_sign = 1.0 if metadata["dip_direction"] == "positive_normal" else -1.0
    return (
        (point.x_km - float(metadata["fault_x_km"])) * normal_x
        + (point.y_km - float(metadata["fault_y_km"])) * normal_y
        - dip_direction_sign * horizontal_offset_km
    )


def _dyke_local_frame(metadata: MappingLike) -> MappingLike:
    center_x, center_y, _ = (float(value) for value in metadata["center_km"])
    strike = math.radians(float(metadata["strike_deg"]))
    along_vector = (math.cos(strike), math.sin(strike), 0.0)
    normal_vector = (-math.sin(strike), math.cos(strike), 0.0)
    return {
        "center_x": center_x,
        "center_y": center_y,
        "along_vector": along_vector,
        "normal_vector": normal_vector,
    }


def _dyke_coordinates(point: Point3D, local: MappingLike) -> tuple[float, float, float]:
    dx = point.x_km - float(local["center_x"])
    dy = point.y_km - float(local["center_y"])
    along = dx * local["along_vector"][0] + dy * local["along_vector"][1]
    normal = dx * local["normal_vector"][0] + dy * local["normal_vector"][1]
    return along, normal, point.z_km


def _point_from_dyke_coordinates(
    along: float,
    normal: float,
    z_km: float,
    local: MappingLike,
) -> Point3D:
    return Point3D(
        x_km=(
            float(local["center_x"])
            + along * local["along_vector"][0]
            + normal * local["normal_vector"][0]
        ),
        y_km=(
            float(local["center_y"])
            + along * local["along_vector"][1]
            + normal * local["normal_vector"][1]
        ),
        z_km=z_km,
    )


def _dedupe_consecutive_waypoints(path: tuple[_Waypoint, ...]) -> tuple[_Waypoint, ...]:
    deduped = [path[0]]
    for waypoint in path[1:]:
        if _distance_km(waypoint.point, deduped[-1].point) > 1.0e-7:
            deduped.append(waypoint)
    return tuple(deduped)


def _clamp_to_grid(point: Point3D, grid: CartesianVelocityGrid3D) -> Point3D:
    return Point3D(
        x_km=_clamp(point.x_km, grid.x_coordinates_km[0], grid.x_coordinates_km[-1]),
        y_km=_clamp(point.y_km, grid.y_coordinates_km[0], grid.y_coordinates_km[-1]),
        z_km=_clamp(point.z_km, grid.z_coordinates_km[0], grid.z_coordinates_km[-1]),
    )


def _interpolate_point(start: Point3D, end: Point3D, fraction: float) -> Point3D:
    return Point3D(
        x_km=start.x_km + (end.x_km - start.x_km) * fraction,
        y_km=start.y_km + (end.y_km - start.y_km) * fraction,
        z_km=start.z_km + (end.z_km - start.z_km) * fraction,
    )


def _distance_km(first: Point3D, second: Point3D) -> float:
    return math.sqrt(
        (first.x_km - second.x_km) ** 2
        + (first.y_km - second.y_km) ** 2
        + (first.z_km - second.z_km) ** 2
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length == 0.0:
        return (1.0, 0.0, 0.0)
    return tuple(component / length for component in vector)


def _dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _minimum_grid_spacing_km(grid: CartesianVelocityGrid3D) -> float:
    spacing = grid.metadata.get("grid_spacing_km", 1.0)
    if isinstance(spacing, dict):
        return min(float(value) for value in spacing.values())
    return float(spacing)
