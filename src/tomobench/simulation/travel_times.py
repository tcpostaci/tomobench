"""P-wave first-arrival travel-time simulation interfaces and first implementation."""

from __future__ import annotations

import csv
import heapq
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from tomobench.config.settings import BenchmarkSettings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import (
    CartesianVelocityGrid3D,
    EarthquakeConfiguration,
    LayeredVelocityModel,
    StationConfiguration,
    TravelTimeObservation,
    TravelTimeRayPath,
)
from tomobench.generation.velocity_grids import build_cartesian_velocity_grid
from tomobench.simulation.ray_tracing import trace_pseudo_bending_ray

STRAIGHT_LINE_LAYERED_METHOD = "straight_line_layered_placeholder"
LAYERED_RAY_TRACING_METHOD = "layered_ray_tracing"
EIKONAL_3D_FIRST_ARRIVAL_METHOD = "eikonal_3d_first_arrival_prototype"
EIKONAL_3D_SOLVER_TYPE = "graph_based_cartesian_eikonal_approximation"
PSEUDO_BENDING_3D_METHOD = "pseudo_bending_3d"
PSEUDO_BENDING_SOLVER_TYPE = "cartesian_grid_pseudo_bending"
SUPPORTED_SIMULATION_METHODS = (
    STRAIGHT_LINE_LAYERED_METHOD,
    LAYERED_RAY_TRACING_METHOD,
    EIKONAL_3D_FIRST_ARRIVAL_METHOD,
    PSEUDO_BENDING_3D_METHOD,
)


class TravelTimeSimulator(ABC):
    """Interface for interchangeable travel-time simulation backends."""

    method_name: str

    @abstractmethod
    def simulate(
        self,
        stations: StationConfiguration,
        earthquakes: EarthquakeConfiguration,
        velocity_model: LayeredVelocityModel,
    ) -> tuple[TravelTimeObservation, ...]:
        """Return source-receiver travel-time observations."""


class StraightLineLayeredTravelTimeSimulator(TravelTimeSimulator):
    """Straight-line placeholder simulator for a 1D layered velocity model.

    This implementation intentionally does not perform physical ray tracing. It
    estimates a travel time by intersecting the straight source-receiver segment
    with horizontal layers and summing layer-wise distance divided by velocity.
    """

    method_name = STRAIGHT_LINE_LAYERED_METHOD

    def __init__(self, settings: BenchmarkSettings) -> None:
        self.settings = settings

    def simulate(
        self,
        stations: StationConfiguration,
        earthquakes: EarthquakeConfiguration,
        velocity_model: LayeredVelocityModel,
    ) -> tuple[TravelTimeObservation, ...]:
        """Return one travel-time observation for every earthquake-station pair."""
        observations: list[TravelTimeObservation] = []
        for earthquake in earthquakes.earthquakes:
            for station in stations.stations:
                path_length = euclidean_distance_km(earthquake.hypocenter, station.location)
                travel_time = layered_straight_line_travel_time_s(
                    source=earthquake.hypocenter,
                    receiver=station.location,
                    velocity_model=velocity_model,
                )
                observations.append(
                    TravelTimeObservation(
                        observation_id=f"{earthquake.earthquake_id}_{station.station_id}",
                        earthquake_id=earthquake.earthquake_id,
                        station_id=station.station_id,
                        source=earthquake.hypocenter,
                        receiver=station.location,
                        travel_time_s=travel_time,
                        path_length_km=path_length,
                        phase=self.settings.simulation.phase,
                        simulation_method=self.method_name,
                        station_configuration_id=stations.configuration_id,
                        earthquake_configuration_id=earthquakes.configuration_id,
                        velocity_model_id=velocity_model.model_id,
                        simulation_id=self.settings.vertical_slice.simulation_id,
                    )
                )
        return tuple(observations)


class LayeredRayTracingTravelTimeSimulator(TravelTimeSimulator):
    """Direct-ray simulator for a 1D horizontally layered velocity model.

    The method searches for a horizontal slowness, or ray parameter, that
    reproduces the source-receiver horizontal offset through the crossed
    layers. It models Snell-law bending in a 1D layered medium, but it does
    not model head waves, turning waves, lateral heterogeneity, or full 3D
    grid-based ray tracing.
    """

    method_name = LAYERED_RAY_TRACING_METHOD

    def __init__(self, settings: BenchmarkSettings) -> None:
        self.settings = settings

    def simulate(
        self,
        stations: StationConfiguration,
        earthquakes: EarthquakeConfiguration,
        velocity_model: LayeredVelocityModel,
    ) -> tuple[TravelTimeObservation, ...]:
        """Return one layered direct-ray observation for every earthquake-station pair."""
        observations: list[TravelTimeObservation] = []
        for earthquake in earthquakes.earthquakes:
            for station in stations.stations:
                path_length = euclidean_distance_km(earthquake.hypocenter, station.location)
                travel_time = layered_ray_parameter_travel_time_s(
                    source=earthquake.hypocenter,
                    receiver=station.location,
                    velocity_model=velocity_model,
                    numerical_tolerance=self.settings.simulation.numerical_tolerance,
                )
                observations.append(
                    TravelTimeObservation(
                        observation_id=f"{earthquake.earthquake_id}_{station.station_id}",
                        earthquake_id=earthquake.earthquake_id,
                        station_id=station.station_id,
                        source=earthquake.hypocenter,
                        receiver=station.location,
                        travel_time_s=travel_time,
                        path_length_km=path_length,
                        phase=self.settings.simulation.phase,
                        simulation_method=self.method_name,
                        station_configuration_id=stations.configuration_id,
                        earthquake_configuration_id=earthquakes.configuration_id,
                        velocity_model_id=velocity_model.model_id,
                        simulation_id=self.settings.vertical_slice.simulation_id,
                    )
                )
        return tuple(observations)


class Eikonal3DFirstArrivalTravelTimeSimulator(TravelTimeSimulator):
    """Prototype graph-based 3D first-arrival eikonal approximation.

    This backend samples or accepts a node-centered Cartesian velocity grid and
    computes shortest travel-time paths through grid-neighbor edges. It is a
    bounded prototype for first-arrival P-wave synthetic data generation, not a
    high-order fast-marching solver and not full 3D ray tracing.
    """

    method_name = EIKONAL_3D_FIRST_ARRIVAL_METHOD
    solver_type = EIKONAL_3D_SOLVER_TYPE

    def __init__(
        self,
        settings: BenchmarkSettings,
        velocity_grid: CartesianVelocityGrid3D | None = None,
    ) -> None:
        self.settings = settings
        self.velocity_grid = velocity_grid

    def simulate(
        self,
        stations: StationConfiguration,
        earthquakes: EarthquakeConfiguration,
        velocity_model: LayeredVelocityModel,
    ) -> tuple[TravelTimeObservation, ...]:
        """Return one first-arrival prototype observation for every source-receiver pair."""
        grid = self.velocity_grid or build_cartesian_velocity_grid(self.settings, velocity_model)
        graph = _CartesianGridGraph(grid, self.settings.eikonal_solver.connectivity)

        observations: list[TravelTimeObservation] = []
        station_nodes = {
            station.station_id: graph.nearest_node_index(station.location)
            for station in stations.stations
        }
        for earthquake in earthquakes.earthquakes:
            source_node = graph.nearest_node_index(earthquake.hypocenter)
            travel_times_by_node = graph.shortest_travel_times_from(source_node)
            for station in stations.stations:
                receiver_node = station_nodes[station.station_id]
                travel_time = travel_times_by_node[receiver_node]
                path_length = euclidean_distance_km(earthquake.hypocenter, station.location)
                observations.append(
                    TravelTimeObservation(
                        observation_id=f"{earthquake.earthquake_id}_{station.station_id}",
                        earthquake_id=earthquake.earthquake_id,
                        station_id=station.station_id,
                        source=earthquake.hypocenter,
                        receiver=station.location,
                        travel_time_s=travel_time,
                        path_length_km=path_length,
                        phase=self.settings.simulation.phase,
                        simulation_method=self.method_name,
                        station_configuration_id=stations.configuration_id,
                        earthquake_configuration_id=earthquakes.configuration_id,
                        velocity_model_id=velocity_model.model_id,
                        simulation_id=self.settings.vertical_slice.simulation_id,
                    )
                )
        return tuple(observations)


class PseudoBending3DTravelTimeSimulator(TravelTimeSimulator):
    """Cartesian-grid 3D pseudo-bending ray tracer for synthetic P-wave arrivals."""

    method_name = PSEUDO_BENDING_3D_METHOD
    solver_type = PSEUDO_BENDING_SOLVER_TYPE

    def __init__(
        self,
        settings: BenchmarkSettings,
        velocity_grid: CartesianVelocityGrid3D | None = None,
    ) -> None:
        self.settings = settings
        self.velocity_grid = velocity_grid

    def simulate(
        self,
        stations: StationConfiguration,
        earthquakes: EarthquakeConfiguration,
        velocity_model: LayeredVelocityModel,
    ) -> tuple[TravelTimeObservation, ...]:
        """Return one pseudo-bending observation for every source-receiver pair."""
        observations, _ = self.simulate_with_ray_paths(stations, earthquakes, velocity_model)
        return observations

    def simulate_with_ray_paths(
        self,
        stations: StationConfiguration,
        earthquakes: EarthquakeConfiguration,
        velocity_model: LayeredVelocityModel,
    ) -> tuple[tuple[TravelTimeObservation, ...], tuple[TravelTimeRayPath, ...]]:
        """Return pseudo-bending observations with traceable ray-path sidecars."""
        grid = self.velocity_grid or build_cartesian_velocity_grid(self.settings, velocity_model)

        observations: list[TravelTimeObservation] = []
        ray_paths: list[TravelTimeRayPath] = []
        for earthquake in earthquakes.earthquakes:
            for station in stations.stations:
                observation_id = f"{earthquake.earthquake_id}_{station.station_id}"
                ray = trace_pseudo_bending_ray(
                    source=earthquake.hypocenter,
                    receiver=station.location,
                    velocity_grid=grid,
                    settings=self.settings.pseudo_bending_solver,
                )
                observations.append(
                    TravelTimeObservation(
                        observation_id=observation_id,
                        earthquake_id=earthquake.earthquake_id,
                        station_id=station.station_id,
                        source=earthquake.hypocenter,
                        receiver=station.location,
                        travel_time_s=ray.travel_time_s,
                        path_length_km=ray.path_length_km,
                        phase=self.settings.simulation.phase,
                        simulation_method=self.method_name,
                        station_configuration_id=stations.configuration_id,
                        earthquake_configuration_id=earthquakes.configuration_id,
                        velocity_model_id=velocity_model.model_id,
                        simulation_id=self.settings.vertical_slice.simulation_id,
                    )
                )
                ray_paths.append(
                    TravelTimeRayPath(
                        observation_id=observation_id,
                        earthquake_id=earthquake.earthquake_id,
                        station_id=station.station_id,
                        source=earthquake.hypocenter,
                        receiver=station.location,
                        ray_path=ray.ray_path,
                        travel_time_s=ray.travel_time_s,
                        path_length_km=ray.path_length_km,
                        phase=self.settings.simulation.phase,
                        simulation_method=self.method_name,
                        simulator_solver_type=self.solver_type,
                        station_configuration_id=stations.configuration_id,
                        earthquake_configuration_id=earthquakes.configuration_id,
                        velocity_model_id=velocity_model.model_id,
                        simulation_id=self.settings.vertical_slice.simulation_id,
                        case_id=self.settings.vertical_slice.case_id,
                        converged=ray.converged,
                        iteration_count=ray.iteration_count,
                        termination_reason=ray.termination_reason,
                    )
                )
        return tuple(observations), tuple(ray_paths)


def build_travel_time_simulator(settings: BenchmarkSettings) -> TravelTimeSimulator:
    """Create the configured travel-time simulator."""
    if settings.simulation.method == STRAIGHT_LINE_LAYERED_METHOD:
        return StraightLineLayeredTravelTimeSimulator(settings)
    if settings.simulation.method == LAYERED_RAY_TRACING_METHOD:
        return LayeredRayTracingTravelTimeSimulator(settings)
    if settings.simulation.method == EIKONAL_3D_FIRST_ARRIVAL_METHOD:
        return Eikonal3DFirstArrivalTravelTimeSimulator(settings)
    if settings.simulation.method == PSEUDO_BENDING_3D_METHOD:
        return PseudoBending3DTravelTimeSimulator(settings)
    raise ValueError(
        f"Unsupported simulation method '{settings.simulation.method}'. "
        f"Supported methods: {', '.join(SUPPORTED_SIMULATION_METHODS)}."
    )


def euclidean_distance_km(first: Point3D, second: Point3D) -> float:
    """Return straight-line distance between two points in kilometers."""
    return math.sqrt(
        (first.x_km - second.x_km) ** 2
        + (first.y_km - second.y_km) ** 2
        + (first.z_km - second.z_km) ** 2
    )


def layered_straight_line_travel_time_s(
    source: Point3D,
    receiver: Point3D,
    velocity_model: LayeredVelocityModel,
) -> float:
    """Compute straight-segment travel time through horizontal velocity layers."""
    path_length = euclidean_distance_km(source, receiver)
    vertical_span = abs(source.z_km - receiver.z_km)
    if path_length == 0:
        return 0.0
    if vertical_span == 0:
        return path_length / _velocity_at_depth(velocity_model, source.z_km)

    shallow_depth = min(source.z_km, receiver.z_km)
    deep_depth = max(source.z_km, receiver.z_km)
    travel_time_s = 0.0
    for layer in velocity_model.layers:
        overlap_top = max(shallow_depth, layer.top_depth_km)
        overlap_bottom = min(deep_depth, layer.bottom_depth_km)
        overlap_km = max(0.0, overlap_bottom - overlap_top)
        if overlap_km == 0.0:
            continue
        segment_fraction = overlap_km / vertical_span
        travel_time_s += (path_length * segment_fraction) / layer.p_velocity_km_per_s
    return travel_time_s


def layered_ray_parameter_travel_time_s(
    source: Point3D,
    receiver: Point3D,
    velocity_model: LayeredVelocityModel,
    numerical_tolerance: float = 1.0e-6,
) -> float:
    """Compute a direct P-wave travel time for a horizontally layered medium.

    The ray parameter ``p`` is solved so that the sum of layer-wise horizontal
    offsets equals the source-receiver horizontal offset:

    ``dx_i = h_i * p * v_i / sqrt(1 - (p * v_i)^2)``

    The corresponding segment time is:

    ``dt_i = h_i / (v_i * sqrt(1 - (p * v_i)^2))``

    where ``h_i`` is the vertical thickness crossed in layer ``i`` and ``v_i``
    is that layer's P-wave velocity. This is a direct-ray approximation; it
    intentionally excludes head-wave and turning-ray branches.
    """
    horizontal_offset = math.sqrt(
        (source.x_km - receiver.x_km) ** 2 + (source.y_km - receiver.y_km) ** 2
    )
    vertical_span = abs(source.z_km - receiver.z_km)
    if horizontal_offset == 0.0 and vertical_span == 0.0:
        return 0.0
    if vertical_span == 0.0:
        return horizontal_offset / _velocity_at_depth(velocity_model, source.z_km)
    if horizontal_offset == 0.0:
        return sum(
            thickness_km / velocity_km_per_s
            for thickness_km, velocity_km_per_s in _traversed_layer_segments(
                source, receiver, velocity_model
            )
        )

    segments = _traversed_layer_segments(source, receiver, velocity_model)
    max_velocity = max(velocity for _, velocity in segments)
    p_low = 0.0
    p_high = (1.0 / max_velocity) * (1.0 - 1.0e-12)
    tolerance = max(float(numerical_tolerance), 1.0e-12)

    for _ in range(100):
        p_mid = (p_low + p_high) / 2.0
        modeled_offset = _horizontal_offset_for_ray_parameter(segments, p_mid)
        if abs(modeled_offset - horizontal_offset) <= tolerance:
            p_low = p_mid
            p_high = p_mid
            break
        if modeled_offset < horizontal_offset:
            p_low = p_mid
        else:
            p_high = p_mid

    ray_parameter = (p_low + p_high) / 2.0
    travel_time_s = 0.0
    for thickness_km, velocity_km_per_s in segments:
        pv = ray_parameter * velocity_km_per_s
        travel_time_s += thickness_km / (velocity_km_per_s * math.sqrt(max(1.0 - pv * pv, 1.0e-15)))
    return travel_time_s


def layered_first_arrival_travel_time_s(
    source: Point3D,
    receiver: Point3D,
    velocity_model: LayeredVelocityModel,
    numerical_tolerance: float = 1.0e-6,
) -> float:
    """Estimate a layered-medium first arrival from direct and head-wave branches.

    The existing :func:`layered_ray_parameter_travel_time_s` function is a
    direct-ray calculation.  For horizontally layered models with increasing
    velocity, a long-offset first arrival can instead travel critically along
    an interface below both endpoints.  This helper evaluates the direct branch
    and every physically admissible critical-interface candidate, then returns
    the smallest time.  It is an analytical validation reference for the
    monotonic layered cases used here; it is not a general turning-ray solver.
    """
    direct_time = layered_ray_parameter_travel_time_s(
        source,
        receiver,
        velocity_model,
        numerical_tolerance=numerical_tolerance,
    )
    horizontal_offset = math.sqrt(
        (source.x_km - receiver.x_km) ** 2 + (source.y_km - receiver.y_km) ** 2
    )
    deepest_endpoint = max(source.z_km, receiver.z_km)
    candidates = [direct_time]
    for layer_index, layer in enumerate(velocity_model.layers[:-1]):
        interface_depth = layer.bottom_depth_km
        if interface_depth <= deepest_endpoint + numerical_tolerance:
            continue
        head_velocity = velocity_model.layers[layer_index + 1].p_velocity_km_per_s
        head_time = 0.0
        critical_offset = 0.0
        admissible = True
        for endpoint in (source, receiver):
            segments = _layer_segments_between_depths(
                endpoint.z_km,
                interface_depth,
                velocity_model,
            )
            for thickness_km, velocity_km_per_s in segments:
                ratio = velocity_km_per_s / head_velocity
                if ratio >= 1.0 - 1.0e-12:
                    admissible = False
                    break
                cosine = math.sqrt(1.0 - ratio * ratio)
                critical_offset += thickness_km * ratio / cosine
                head_time += thickness_km / (velocity_km_per_s * cosine)
            if not admissible:
                break
        if admissible and horizontal_offset + numerical_tolerance >= critical_offset:
            candidates.append(
                head_time + (horizontal_offset - critical_offset) / head_velocity
            )
    return min(candidates)


def save_travel_time_observations(
    observations: tuple[TravelTimeObservation, ...],
    path: Path,
    settings: BenchmarkSettings,
) -> Path:
    """Save simulation outputs and metadata as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "simulation_id": settings.vertical_slice.simulation_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": settings.simulation.phase,
        "method": settings.simulation.method,
        "simulator_methods": sorted(
            {observation.simulation_method for observation in observations}
        ),
        "solver_type": (
            EIKONAL_3D_SOLVER_TYPE
            if settings.simulation.method == EIKONAL_3D_FIRST_ARRIVAL_METHOD
            else PSEUDO_BENDING_SOLVER_TYPE
            if settings.simulation.method == PSEUDO_BENDING_3D_METHOD
            else None
        ),
        "eikonal_solver": (
            {
                "grid_spacing_km": settings.eikonal_solver.grid_spacing_by_axis_km,
                "connectivity": settings.eikonal_solver.connectivity,
                "interpolation": settings.eikonal_solver.interpolation,
                "max_grid_nodes": settings.eikonal_solver.max_grid_nodes,
                "grid_scenario": settings.velocity_model_generation.default_scenario,
            }
            if settings.simulation.method == EIKONAL_3D_FIRST_ARRIVAL_METHOD
            else None
        ),
        "pseudo_bending_solver": (
            {
                "grid_spacing_km": settings.eikonal_solver.grid_spacing_by_axis_km,
                "initial_ray_point_count": (settings.pseudo_bending_solver.initial_ray_point_count),
                "max_ray_point_count": settings.pseudo_bending_solver.max_ray_point_count,
                "max_iterations_per_level": (
                    settings.pseudo_bending_solver.max_iterations_per_level
                ),
                "convergence_tolerance_s": (settings.pseudo_bending_solver.convergence_tolerance_s),
                "perturbation_step_km": settings.pseudo_bending_solver.perturbation_step_km,
                "finite_difference_step_km": (
                    settings.pseudo_bending_solver.finite_difference_step_km
                ),
                "velocity_interpolation": (settings.pseudo_bending_solver.velocity_interpolation),
                "boundary_handling": settings.pseudo_bending_solver.boundary_handling,
                "grid_scenario": settings.velocity_model_generation.default_scenario,
            }
            if settings.simulation.method == PSEUDO_BENDING_3D_METHOD
            else None
        ),
        "assumptions": list(settings.simulation.assumptions),
        "observation_count": len(observations),
        "observations": [asdict(observation) for observation in observations],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def save_travel_time_csv(
    observations: tuple[TravelTimeObservation, ...],
    path: Path,
) -> Path:
    """Save compact simulation outputs as CSV for quick inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "observation_id",
        "earthquake_id",
        "station_id",
        "travel_time_s",
        "path_length_km",
        "phase",
        "simulation_method",
        "velocity_model_id",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for observation in observations:
            writer.writerow({field: getattr(observation, field) for field in fieldnames})
    return path


def save_travel_time_ray_paths_jsonl(
    ray_paths: tuple[TravelTimeRayPath, ...],
    path: Path,
    settings: BenchmarkSettings,
) -> Path:
    """Save pseudo-bending ray paths as one traceable JSON object per observation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ray_path in ray_paths:
            record = {
                "artifact_type": "pseudo_bending_ray_path_sidecar",
                "schema_version": settings.dataset_export.schema_version,
                **asdict(ray_path),
            }
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
    return path


def _velocity_at_depth(model: LayeredVelocityModel, depth_km: float) -> float:
    for layer in model.layers:
        if layer.top_depth_km <= depth_km < layer.bottom_depth_km:
            return layer.p_velocity_km_per_s
    deepest = model.layers[-1]
    if depth_km == deepest.bottom_depth_km:
        return deepest.p_velocity_km_per_s
    raise ValueError(f"Depth {depth_km} km is outside the layered velocity model.")


def _traversed_layer_segments(
    source: Point3D,
    receiver: Point3D,
    velocity_model: LayeredVelocityModel,
) -> tuple[tuple[float, float], ...]:
    return _layer_segments_between_depths(
        min(source.z_km, receiver.z_km),
        max(source.z_km, receiver.z_km),
        velocity_model,
    )


def _layer_segments_between_depths(
    shallow_depth: float,
    deep_depth: float,
    velocity_model: LayeredVelocityModel,
) -> tuple[tuple[float, float], ...]:
    segments: list[tuple[float, float]] = []
    for layer in velocity_model.layers:
        overlap_top = max(shallow_depth, layer.top_depth_km)
        overlap_bottom = min(deep_depth, layer.bottom_depth_km)
        thickness_km = max(0.0, overlap_bottom - overlap_top)
        if thickness_km > 0.0:
            segments.append((thickness_km, layer.p_velocity_km_per_s))
    if not segments:
        raise ValueError(
            "Source-receiver vertical interval does not intersect the layered velocity model."
        )
    return tuple(segments)


def _horizontal_offset_for_ray_parameter(
    segments: tuple[tuple[float, float], ...],
    ray_parameter: float,
) -> float:
    offset_km = 0.0
    for thickness_km, velocity_km_per_s in segments:
        pv = ray_parameter * velocity_km_per_s
        offset_km += thickness_km * pv / math.sqrt(max(1.0 - pv * pv, 1.0e-15))
    return offset_km


class _CartesianGridGraph:
    """Small Dijkstra graph wrapper around a node-centered Cartesian grid."""

    def __init__(self, grid: CartesianVelocityGrid3D, connectivity: int) -> None:
        self.grid = grid
        self.connectivity = connectivity
        self.x_coordinates = grid.x_coordinates_km
        self.y_coordinates = grid.y_coordinates_km
        self.z_coordinates = grid.z_coordinates_km
        self.nx = len(self.x_coordinates)
        self.ny = len(self.y_coordinates)
        self.nz = len(self.z_coordinates)
        self.node_count = self.nx * self.ny * self.nz
        if len(grid.p_velocity_km_per_s) != self.node_count:
            raise ValueError("Velocity grid shape does not match flattened velocity count.")
        self.neighbor_offsets = _neighbor_offsets(connectivity)

    def nearest_node_index(self, point: Point3D) -> int:
        """Return the nearest grid-node index for a possibly off-grid point."""
        ix = _nearest_axis_index(self.x_coordinates, point.x_km)
        iy = _nearest_axis_index(self.y_coordinates, point.y_km)
        iz = _nearest_axis_index(self.z_coordinates, point.z_km)
        return self._flat_index(ix, iy, iz)

    def shortest_travel_times_from(self, source_index: int) -> list[float]:
        """Compute graph shortest travel times from one source node."""
        distances, _ = self.shortest_travel_times_and_predecessors_from(source_index)
        return distances

    def shortest_travel_times_and_predecessors_from(
        self,
        source_index: int,
    ) -> tuple[list[float], list[int | None]]:
        """Compute graph shortest travel times and predecessor links from one source node."""
        distances = [math.inf] * self.node_count
        predecessors: list[int | None] = [None] * self.node_count
        distances[source_index] = 0.0
        queue: list[tuple[float, int]] = [(0.0, source_index)]

        while queue:
            current_time, node_index = heapq.heappop(queue)
            if current_time > distances[node_index]:
                continue
            ix, iy, iz = self._unflat_index(node_index)
            source_velocity = self.grid.p_velocity_km_per_s[node_index]
            for dx, dy, dz in self.neighbor_offsets:
                nx = ix + dx
                ny = iy + dy
                nz = iz + dz
                if not (0 <= nx < self.nx and 0 <= ny < self.ny and 0 <= nz < self.nz):
                    continue
                neighbor_index = self._flat_index(nx, ny, nz)
                edge_distance = math.sqrt(
                    (self.x_coordinates[nx] - self.x_coordinates[ix]) ** 2
                    + (self.y_coordinates[ny] - self.y_coordinates[iy]) ** 2
                    + (self.z_coordinates[nz] - self.z_coordinates[iz]) ** 2
                )
                neighbor_velocity = self.grid.p_velocity_km_per_s[neighbor_index]
                edge_time = edge_distance * (
                    (1.0 / source_velocity + 1.0 / neighbor_velocity) / 2.0
                )
                candidate = current_time + edge_time
                if candidate < distances[neighbor_index]:
                    distances[neighbor_index] = candidate
                    predecessors[neighbor_index] = node_index
                    heapq.heappush(queue, (candidate, neighbor_index))
        return distances, predecessors

    def shortest_path_indices(self, source_index: int, target_index: int) -> tuple[int, ...]:
        """Return one Dijkstra shortest path between two grid nodes."""
        distances, predecessors = self.shortest_travel_times_and_predecessors_from(source_index)
        if math.isinf(distances[target_index]):
            raise ValueError("Target node is unreachable from the requested source node.")

        path = [target_index]
        current_index = target_index
        while current_index != source_index:
            predecessor = predecessors[current_index]
            if predecessor is None:
                raise ValueError(
                    "Shortest-path predecessor chain terminated before reaching source."
                )
            path.append(predecessor)
            current_index = predecessor
        path.reverse()
        return tuple(path)

    def node_point(self, index: int) -> Point3D:
        """Return the Cartesian coordinates of one flattened grid node."""
        ix, iy, iz = self._unflat_index(index)
        return Point3D(
            x_km=self.x_coordinates[ix],
            y_km=self.y_coordinates[iy],
            z_km=self.z_coordinates[iz],
        )

    def _flat_index(self, ix: int, iy: int, iz: int) -> int:
        return ix + self.nx * (iy + self.ny * iz)

    def _unflat_index(self, index: int) -> tuple[int, int, int]:
        iz, remainder = divmod(index, self.nx * self.ny)
        iy, ix = divmod(remainder, self.nx)
        return ix, iy, iz


def _neighbor_offsets(connectivity: int) -> tuple[tuple[int, int, int], ...]:
    offsets: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and manhattan != 1:
                    continue
                if connectivity == 18 and manhattan > 2:
                    continue
                offsets.append((dx, dy, dz))
    if connectivity not in (6, 18, 26):
        raise ValueError("Connectivity must be one of 6, 18, or 26.")
    return tuple(offsets)


def _nearest_axis_index(coordinates: tuple[float, ...], value: float) -> int:
    return min(range(len(coordinates)), key=lambda index: abs(coordinates[index] - value))
