"""Benchmarking helpers for the graph-based 3D first-arrival eikonal prototype."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from time import perf_counter

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import (
    Earthquake,
    EarthquakeConfiguration,
    LayeredVelocityModel,
    Station,
    StationConfiguration,
    VelocityLayer,
)
from tomobench.generation.velocity_grids import build_cartesian_velocity_grid
from tomobench.simulation.travel_times import (
    EIKONAL_3D_FIRST_ARRIVAL_METHOD,
    Eikonal3DFirstArrivalTravelTimeSimulator,
    euclidean_distance_km,
)
from tomobench.utils.paths import get_repo_root


@dataclass(frozen=True)
class EikonalBenchmarkOutputs:
    """Files produced by the eikonal benchmark workflow."""

    csv_path: Path
    json_path: Path
    note_path: Path


def run_eikonal_benchmarks(settings: BenchmarkSettings | None = None) -> EikonalBenchmarkOutputs:
    """Run bounded runtime and accuracy benchmarks for the eikonal prototype."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    benchmark_dir = repo_root / loaded_settings.outputs.benchmarks_dir
    note_path = repo_root / loaded_settings.outputs.experiment_notes_dir / "eikonal_3d_benchmark.md"
    csv_path = benchmark_dir / "eikonal_3d_prototype_benchmark.csv"
    json_path = benchmark_dir / "eikonal_3d_prototype_benchmark.json"

    rows: list[dict[str, object]] = []
    for connectivity in loaded_settings.eikonal_solver.benchmark_connectivities:
        for spacing_km in loaded_settings.eikonal_solver.benchmark_grid_spacings_km:
            benchmark_settings = replace(
                loaded_settings,
                simulation=replace(
                    loaded_settings.simulation,
                    method=EIKONAL_3D_FIRST_ARRIVAL_METHOD,
                ),
                eikonal_solver=replace(
                    loaded_settings.eikonal_solver,
                    connectivity=connectivity,
                    x_grid_spacing_km=spacing_km,
                    y_grid_spacing_km=spacing_km,
                    z_grid_spacing_km=spacing_km,
                ),
            )
            rows.extend(_run_spacing_benchmarks(benchmark_settings))

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    _write_benchmark_csv(rows, csv_path)
    _write_benchmark_json(rows, json_path)
    _write_benchmark_note(rows, note_path, loaded_settings, repo_root)
    return EikonalBenchmarkOutputs(csv_path=csv_path, json_path=json_path, note_path=note_path)


def benchmark_columns() -> tuple[str, ...]:
    """Return the benchmark CSV schema."""
    return (
        "benchmark_case",
        "velocity_scenario",
        "connectivity",
        "grid_spacing_km",
        "node_count",
        "source_count",
        "receiver_count",
        "observation_count",
        "grid_build_time_s",
        "simulation_time_s",
        "runtime_repeat_count",
        "exact_reference_type",
        "mean_absolute_error_s",
        "max_absolute_error_s",
        "mean_relative_error_percent",
    )


def _run_spacing_benchmarks(settings: BenchmarkSettings) -> list[dict[str, object]]:
    cases = benchmark_cases(settings)
    return [_run_single_case(settings, case) for case in cases]


def _run_single_case(settings: BenchmarkSettings, case: EikonalBenchmarkCase) -> dict[str, object]:
    grid_build_start = perf_counter()
    velocity_grid = build_cartesian_velocity_grid(
        settings,
        case.velocity_model,
        scenario=case.velocity_scenario,
    )
    grid_build_time_s = perf_counter() - grid_build_start

    simulator = Eikonal3DFirstArrivalTravelTimeSimulator(settings, velocity_grid=velocity_grid)
    simulation_times: list[float] = []
    observations = ()
    for _ in range(settings.eikonal_solver.runtime_repeat_count):
        run_start = perf_counter()
        observations = simulator.simulate(
            case.stations,
            case.earthquakes,
            case.velocity_model,
        )
        simulation_times.append(perf_counter() - run_start)

    observation_errors = [
        abs(observation.travel_time_s - case.exact_times_s[observation.observation_id])
        for observation in observations
    ]
    relative_errors = []
    for observation in observations:
        exact_time_s = case.exact_times_s[observation.observation_id]
        relative_errors.append(
            0.0
            if exact_time_s == 0.0
            else abs(observation.travel_time_s - exact_time_s) / exact_time_s * 100.0
        )

    return {
        "benchmark_case": case.name,
        "velocity_scenario": case.velocity_scenario,
        "connectivity": settings.eikonal_solver.connectivity,
        "grid_spacing_km": settings.eikonal_solver.grid_spacing_km,
        "node_count": velocity_grid.metadata["node_count"],
        "source_count": len(case.earthquakes.earthquakes),
        "receiver_count": len(case.stations.stations),
        "observation_count": len(observations),
        "grid_build_time_s": round(grid_build_time_s, 6),
        "simulation_time_s": round(mean(simulation_times), 6),
        "runtime_repeat_count": settings.eikonal_solver.runtime_repeat_count,
        "exact_reference_type": case.exact_reference_type,
        "mean_absolute_error_s": round(mean(observation_errors), 6),
        "max_absolute_error_s": round(max(observation_errors), 6),
        "mean_relative_error_percent": round(mean(relative_errors), 6),
    }


@dataclass(frozen=True)
class EikonalBenchmarkCase:
    name: str
    velocity_scenario: str
    velocity_model: LayeredVelocityModel
    earthquakes: EarthquakeConfiguration
    stations: StationConfiguration
    exact_times_s: dict[str, float]
    exact_reference_type: str


def benchmark_cases(settings: BenchmarkSettings) -> tuple[EikonalBenchmarkCase, ...]:
    """Return the bounded benchmark cases used for the prototype sweep."""
    return (
        _homogeneous_case(settings),
        _controlled_block_anomaly_case(settings),
        _controlled_faulted_case(settings),
    )


def _homogeneous_case(settings: BenchmarkSettings) -> EikonalBenchmarkCase:
    velocity_model = _homogeneous_velocity_model(velocity_km_per_s=5.0)
    earthquakes = EarthquakeConfiguration(
        configuration_id="benchmark_homogeneous_earthquakes",
        earthquakes=(
            Earthquake("EQH001", Point3D(20.0, 20.0, 10.0)),
            Earthquake("EQH002", Point3D(60.0, 60.0, 20.0)),
        ),
        metadata={},
    )
    stations = StationConfiguration(
        configuration_id="benchmark_homogeneous_stations",
        stations=(
            Station("STH001", Point3D(40.0, 20.0, 10.0)),
            Station("STH002", Point3D(40.0, 40.0, 20.0)),
            Station("STH003", Point3D(80.0, 60.0, 0.0)),
        ),
        metadata={},
    )
    exact_times = {
        f"{earthquake.earthquake_id}_{station.station_id}": (
            euclidean_distance_km(earthquake.hypocenter, station.location) / 5.0
        )
        for earthquake in earthquakes.earthquakes
        for station in stations.stations
    }
    return EikonalBenchmarkCase(
        name="homogeneous_on_grid_accuracy",
        velocity_scenario="layered",
        velocity_model=velocity_model,
        earthquakes=earthquakes,
        stations=stations,
        exact_times_s=exact_times,
        exact_reference_type="distance_over_velocity",
    )


def _controlled_block_anomaly_case(settings: BenchmarkSettings) -> EikonalBenchmarkCase:
    velocity_model = _homogeneous_velocity_model(velocity_km_per_s=5.0)
    source = Point3D(50.0, 50.0, 0.0)
    receiver = Point3D(50.0, 50.0, 20.0)
    earthquakes = EarthquakeConfiguration(
        configuration_id="benchmark_block_anomaly_earthquakes",
        earthquakes=(Earthquake("EQB001", source),),
        metadata={},
    )
    stations = StationConfiguration(
        configuration_id="benchmark_block_anomaly_stations",
        stations=(Station("STB001", receiver),),
        metadata={},
    )
    exact_times = {
        "EQB001_STB001": _vertical_block_anomaly_exact_time_s(
            settings=settings,
            source=source,
            receiver=receiver,
            background_velocity_km_per_s=5.0,
        )
    }
    return EikonalBenchmarkCase(
        name="controlled_block_anomaly_vertical_path",
        velocity_scenario="block_anomaly",
        velocity_model=velocity_model,
        earthquakes=earthquakes,
        stations=stations,
        exact_times_s=exact_times,
        exact_reference_type="vertical_graph_path_average_edge_slowness",
    )


def _controlled_faulted_case(settings: BenchmarkSettings) -> EikonalBenchmarkCase:
    velocity_model = _homogeneous_velocity_model(velocity_km_per_s=5.0)
    source = Point3D(40.0, 50.0, 10.0)
    receiver = Point3D(60.0, 50.0, 10.0)
    earthquakes = EarthquakeConfiguration(
        configuration_id="benchmark_faulted_earthquakes",
        earthquakes=(Earthquake("EQF001", source),),
        metadata={},
    )
    stations = StationConfiguration(
        configuration_id="benchmark_faulted_stations",
        stations=(Station("STF001", receiver),),
        metadata={},
    )
    exact_times = {
        "EQF001_STF001": _horizontal_faulted_exact_time_s(
            settings=settings,
            source=source,
            receiver=receiver,
            background_velocity_km_per_s=5.0,
        )
    }
    return EikonalBenchmarkCase(
        name="controlled_faulted_horizontal_crossing",
        velocity_scenario="faulted",
        velocity_model=velocity_model,
        earthquakes=earthquakes,
        stations=stations,
        exact_times_s=exact_times,
        exact_reference_type="horizontal_graph_path_average_edge_slowness",
    )


def _homogeneous_velocity_model(velocity_km_per_s: float) -> LayeredVelocityModel:
    return LayeredVelocityModel(
        model_id="benchmark_homogeneous_velocity",
        layers=(
            VelocityLayer(
                layer_id="LAYER01",
                top_depth_km=0.0,
                bottom_depth_km=30.0,
                p_velocity_km_per_s=velocity_km_per_s,
            ),
        ),
        metadata={"scenario": "homogeneous"},
    )


def _vertical_block_anomaly_exact_time_s(
    settings: BenchmarkSettings,
    source: Point3D,
    receiver: Point3D,
    background_velocity_km_per_s: float,
) -> float:
    spacing_km = settings.eikonal_solver.grid_spacing_km
    z_nodes = []
    current = min(source.z_km, receiver.z_km)
    while current <= max(source.z_km, receiver.z_km) + 1.0e-9:
        z_nodes.append(round(current, 10))
        current += spacing_km

    grid = build_cartesian_velocity_grid(
        settings,
        _homogeneous_velocity_model(background_velocity_km_per_s),
        scenario="block_anomaly",
    )
    z_to_velocity = {}
    x_index = grid.x_coordinates_km.index(source.x_km)
    y_index = grid.y_coordinates_km.index(source.y_km)
    nx = len(grid.x_coordinates_km)
    ny = len(grid.y_coordinates_km)
    for z_km in z_nodes:
        z_index = grid.z_coordinates_km.index(z_km)
        flat_index = x_index + nx * (y_index + ny * z_index)
        z_to_velocity[z_km] = grid.p_velocity_km_per_s[flat_index]

    total_time_s = 0.0
    for first_z_km, second_z_km in zip(z_nodes[:-1], z_nodes[1:], strict=True):
        edge_distance_km = abs(second_z_km - first_z_km)
        edge_slowness = (
            (1.0 / z_to_velocity[first_z_km]) + (1.0 / z_to_velocity[second_z_km])
        ) / 2.0
        total_time_s += edge_distance_km * edge_slowness
    return total_time_s


def _horizontal_faulted_exact_time_s(
    settings: BenchmarkSettings,
    source: Point3D,
    receiver: Point3D,
    background_velocity_km_per_s: float,
) -> float:
    spacing_km = settings.eikonal_solver.grid_spacing_km
    x_nodes = []
    current = min(source.x_km, receiver.x_km)
    while current <= max(source.x_km, receiver.x_km) + 1.0e-9:
        x_nodes.append(round(current, 10))
        current += spacing_km

    grid = build_cartesian_velocity_grid(
        settings,
        _homogeneous_velocity_model(background_velocity_km_per_s),
        scenario="faulted",
    )
    x_to_velocity = {}
    y_index = grid.y_coordinates_km.index(source.y_km)
    z_index = grid.z_coordinates_km.index(source.z_km)
    nx = len(grid.x_coordinates_km)
    ny = len(grid.y_coordinates_km)
    for x_km in x_nodes:
        x_index = grid.x_coordinates_km.index(x_km)
        flat_index = x_index + nx * (y_index + ny * z_index)
        x_to_velocity[x_km] = grid.p_velocity_km_per_s[flat_index]

    total_time_s = 0.0
    for first_x_km, second_x_km in zip(x_nodes[:-1], x_nodes[1:], strict=True):
        edge_distance_km = abs(second_x_km - first_x_km)
        edge_slowness = (
            (1.0 / x_to_velocity[first_x_km]) + (1.0 / x_to_velocity[second_x_km])
        ) / 2.0
        total_time_s += edge_distance_km * edge_slowness
    return total_time_s


def _write_benchmark_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark_columns())
        writer.writeheader()
        writer.writerows(rows)


def _write_benchmark_json(rows: list[dict[str, object]], path: Path) -> None:
    payload = {
        "benchmark_name": "eikonal_3d_first_arrival_prototype",
        "rows": rows,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_benchmark_note(
    rows: list[dict[str, object]],
    path: Path,
    settings: BenchmarkSettings,
    repo_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spacings = ", ".join(str(value) for value in settings.eikonal_solver.benchmark_grid_spacings_km)
    connectivities = ", ".join(
        str(value) for value in settings.eikonal_solver.benchmark_connectivities
    )
    lines = [
        "# Experiment: Eikonal 3D Prototype Benchmark",
        "",
        "## Objective",
        "",
        (
            "Benchmark the graph-based 3D first-arrival eikonal prototype before increasing "
            "grid resolution, using one homogeneous exact-reference case and one controlled "
            "block-anomaly exact-reference case, and one controlled fault-crossing case."
        ),
        "",
        "## Inputs Used",
        "",
        f"- Connectivities benchmarked: `{connectivities}`",
        f"- Grid spacings benchmarked: `{spacings}` km",
        f"- Interpolation: `{settings.eikonal_solver.interpolation}`",
        f"- Runtime repeats per case: `{settings.eikonal_solver.runtime_repeat_count}`",
        "",
        "## Outputs",
        "",
        f"- CSV: `{path_relative(repo_root / settings.outputs.benchmarks_dir / 'eikonal_3d_prototype_benchmark.csv', repo_root)}`",
        f"- JSON: `{path_relative(repo_root / settings.outputs.benchmarks_dir / 'eikonal_3d_prototype_benchmark.json', repo_root)}`",
        "",
        "## Summary",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['benchmark_case']}` at connectivity `{row['connectivity']}` and "
            f"`{row['grid_spacing_km']}` km: "
            f"simulation_time_s={row['simulation_time_s']}, "
            f"mean_absolute_error_s={row['mean_absolute_error_s']}, "
            f"max_absolute_error_s={row['max_absolute_error_s']}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The benchmark is intended to document prototype behavior, not final scientific accuracy.",
            "- Homogeneous exact-reference errors quantify grid discretization under on-grid source and receiver placement.",
            "- For these on-grid homogeneous test pairs, reducing spacing alone does not remove direction-discretization error because the graph connectivity still constrains propagation directions.",
            "- The controlled block-anomaly case checks that the solver remains consistent with a simple exact vertical-path reference in a bounded laterally heterogeneous setting.",
            "- The controlled faulted case checks that the solver remains consistent with a simple exact horizontal crossing of a vertical velocity-offset plane.",
            "- Any future grid-resolution increase should be justified against these runtime and error trends rather than assumed to be beneficial.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def path_relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    resolved_repo_root = repo_root.resolve()
    if resolved.is_relative_to(resolved_repo_root):
        return resolved.relative_to(resolved_repo_root).as_posix()
    return str(resolved)
