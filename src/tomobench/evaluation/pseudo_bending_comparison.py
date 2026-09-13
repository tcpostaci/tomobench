"""Compare Dijkstra graph paths and independent pseudo-bending ray tracing."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import CartesianVelocityGrid3D, LayeredVelocityModel
from tomobench.evaluation.eikonal_benchmarks import path_relative
from tomobench.generation.velocity_grids import build_cartesian_velocity_grid
from tomobench.generation.velocity_models import generate_layered_velocity_model
from tomobench.simulation.ray_tracing import (
    polyline_length_km,
    trace_pseudo_bending_ray,
)
from tomobench.simulation.travel_times import (
    PSEUDO_BENDING_3D_METHOD,
    _CartesianGridGraph,
    euclidean_distance_km,
)
from tomobench.utils.paths import get_repo_root
from tomobench.visualization.ray_path_comparison import save_ray_path_comparison_svg


@dataclass(frozen=True)
class RayMethodComparisonOutputs:
    """Files produced by the ray-method comparison workflow."""

    csv_path: Path
    json_path: Path
    figure_paths: tuple[Path, ...]
    note_path: Path

    @property
    def figure_path(self) -> Path:
        """Return the first figure path for backward-compatible callers."""
        return self.figure_paths[0]


@dataclass(frozen=True)
class RayComparisonCase:
    """One bounded diagnostic case for comparing ray-tracing methods."""

    case_id: str
    velocity_scenario: str
    source: Point3D
    receiver: Point3D
    figure_name: str
    grid_builder: Callable[[BenchmarkSettings, LayeredVelocityModel], CartesianVelocityGrid3D]


def run_ray_method_comparison(
    settings: BenchmarkSettings | None = None,
) -> RayMethodComparisonOutputs:
    """Compare Dijkstra and pseudo-bending on bounded illustrative 3D cases."""
    loaded_settings = settings or load_settings()
    comparison_settings = replace(
        loaded_settings,
        simulation=replace(
            loaded_settings.simulation,
            method=PSEUDO_BENDING_3D_METHOD,
        ),
    )
    repo_root = get_repo_root()
    output_dir = repo_root / comparison_settings.outputs.benchmarks_dir
    figure_dir = repo_root / comparison_settings.outputs.figures_dir / "ray_method_comparison"
    note_path = (
        repo_root
        / comparison_settings.outputs.experiment_notes_dir
        / "ray_method_comparison_pseudo_bending_vs_dijkstra.md"
    )
    csv_path = output_dir / "ray_method_comparison_pseudo_bending_vs_dijkstra.csv"
    json_path = output_dir / "ray_method_comparison_pseudo_bending_vs_dijkstra.json"

    velocity_model = generate_layered_velocity_model(comparison_settings)
    results = [
        _run_single_case(comparison_settings, velocity_model, figure_dir, case)
        for case in _comparison_cases()
    ]
    rows = [result.row for result in results]
    figure_paths = tuple(result.figure_path for result in results)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, csv_path)
    _write_json(rows, csv_path, figure_paths, json_path, results)
    _write_note(
        path=note_path,
        repo_root=repo_root,
        csv_path=csv_path,
        json_path=json_path,
        figure_paths=figure_paths,
        rows=rows,
    )
    return RayMethodComparisonOutputs(
        csv_path=csv_path,
        json_path=json_path,
        figure_paths=figure_paths,
        note_path=note_path,
    )


@dataclass(frozen=True)
class _CaseResult:
    row: dict[str, object]
    figure_path: Path
    dijkstra_path: tuple[Point3D, ...]
    pseudo_bending_path: tuple[Point3D, ...]


def _run_single_case(
    settings: BenchmarkSettings,
    velocity_model: LayeredVelocityModel,
    figure_dir: Path,
    case: RayComparisonCase,
) -> _CaseResult:
    grid_build_start = perf_counter()
    grid = case.grid_builder(settings, velocity_model)
    grid_build_time_s = perf_counter() - grid_build_start

    dijkstra_start = perf_counter()
    graph = _CartesianGridGraph(grid, settings.eikonal_solver.connectivity)
    source_index = graph.nearest_node_index(case.source)
    receiver_index = graph.nearest_node_index(case.receiver)
    dijkstra_times, _ = graph.shortest_travel_times_and_predecessors_from(source_index)
    dijkstra_path_indices = graph.shortest_path_indices(source_index, receiver_index)
    dijkstra_path = tuple(graph.node_point(index) for index in dijkstra_path_indices)
    dijkstra_runtime_s = perf_counter() - dijkstra_start

    pseudo_start = perf_counter()
    pseudo_ray = trace_pseudo_bending_ray(
        source=case.source,
        receiver=case.receiver,
        velocity_grid=grid,
        settings=settings.pseudo_bending_solver,
    )
    pseudo_runtime_s = perf_counter() - pseudo_start

    figure_path = figure_dir / case.figure_name
    save_ray_path_comparison_svg(
        grid=grid,
        source=case.source,
        receiver=case.receiver,
        dijkstra_path=dijkstra_path,
        pseudo_bending_path=pseudo_ray.ray_path,
        dijkstra_time_s=dijkstra_times[receiver_index],
        pseudo_bending_time_s=pseudo_ray.travel_time_s,
        path=figure_path,
    )

    straight_path_length = euclidean_distance_km(case.source, case.receiver)
    row = {
        "case_id": case.case_id,
        "velocity_scenario": case.velocity_scenario,
        "grid_spacing_km": settings.eikonal_solver.grid_spacing_label,
        "connectivity": settings.eikonal_solver.connectivity,
        "node_count": grid.metadata["node_count"],
        "source_x_km": case.source.x_km,
        "source_y_km": case.source.y_km,
        "source_z_km": case.source.z_km,
        "receiver_x_km": case.receiver.x_km,
        "receiver_y_km": case.receiver.y_km,
        "receiver_z_km": case.receiver.z_km,
        "straight_path_length_km": round(straight_path_length, 6),
        "dijkstra_travel_time_s": round(dijkstra_times[receiver_index], 6),
        "pseudo_bending_travel_time_s": round(pseudo_ray.travel_time_s, 6),
        "travel_time_difference_s": round(
            pseudo_ray.travel_time_s - dijkstra_times[receiver_index],
            6,
        ),
        "dijkstra_path_length_km": round(polyline_length_km(dijkstra_path), 6),
        "pseudo_bending_path_length_km": round(pseudo_ray.path_length_km, 6),
        "pseudo_bending_extra_length_vs_straight_km": round(
            pseudo_ray.path_length_km - straight_path_length,
            6,
        ),
        "pseudo_bending_max_deviation_from_straight_km": round(
            _max_deviation_from_straight(pseudo_ray.ray_path, case.source, case.receiver),
            6,
        ),
        "dijkstra_path_point_count": len(dijkstra_path),
        "pseudo_bending_path_point_count": len(pseudo_ray.ray_path),
        "pseudo_bending_iteration_count": pseudo_ray.iteration_count,
        "pseudo_bending_converged": pseudo_ray.converged,
        "grid_build_time_s": round(grid_build_time_s, 6),
        "dijkstra_runtime_s": round(dijkstra_runtime_s, 6),
        "pseudo_bending_runtime_s": round(pseudo_runtime_s, 6),
    }
    return _CaseResult(
        row=row,
        figure_path=figure_path,
        dijkstra_path=dijkstra_path,
        pseudo_bending_path=pseudo_ray.ray_path,
    )


def comparison_columns() -> tuple[str, ...]:
    """Return the ray-method comparison CSV schema."""
    return (
        "case_id",
        "velocity_scenario",
        "grid_spacing_km",
        "connectivity",
        "node_count",
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "receiver_x_km",
        "receiver_y_km",
        "receiver_z_km",
        "straight_path_length_km",
        "dijkstra_travel_time_s",
        "pseudo_bending_travel_time_s",
        "travel_time_difference_s",
        "dijkstra_path_length_km",
        "pseudo_bending_path_length_km",
        "pseudo_bending_extra_length_vs_straight_km",
        "pseudo_bending_max_deviation_from_straight_km",
        "dijkstra_path_point_count",
        "pseudo_bending_path_point_count",
        "pseudo_bending_iteration_count",
        "pseudo_bending_converged",
        "grid_build_time_s",
        "dijkstra_runtime_s",
        "pseudo_bending_runtime_s",
    )


def _comparison_cases() -> tuple[RayComparisonCase, ...]:
    source = Point3D(15.0, 50.0, 22.5)
    receiver = Point3D(85.0, 50.0, 0.0)
    return (
        _configured_case("layered", source, receiver),
        _configured_case("block_anomaly", source, receiver),
        _configured_case("faulted", source, receiver),
        _configured_case("salt_dome", source, receiver),
        _configured_case("dyke_intrusion", source, receiver),
    )


def _configured_case(
    scenario: str,
    source: Point3D,
    receiver: Point3D,
) -> RayComparisonCase:
    return RayComparisonCase(
        case_id=f"{scenario}_layer_snell_path",
        velocity_scenario=scenario,
        source=source,
        receiver=receiver,
        figure_name=f"pseudo_bending_vs_dijkstra_{scenario}.svg",
        grid_builder=lambda settings, velocity_model: build_cartesian_velocity_grid(
            settings,
            velocity_model,
            scenario=scenario,
        ),
    )


def _max_deviation_from_straight(
    path: tuple[Point3D, ...],
    source: Point3D,
    receiver: Point3D,
) -> float:
    sx, sy, sz = source.x_km, source.y_km, source.z_km
    vx = receiver.x_km - sx
    vy = receiver.y_km - sy
    vz = receiver.z_km - sz
    line_length_squared = vx * vx + vy * vy + vz * vz
    if line_length_squared == 0.0:
        return 0.0

    max_deviation = 0.0
    for point in path:
        wx = point.x_km - sx
        wy = point.y_km - sy
        wz = point.z_km - sz
        projection = (wx * vx + wy * vy + wz * vz) / line_length_squared
        closest = Point3D(
            x_km=sx + projection * vx,
            y_km=sy + projection * vy,
            z_km=sz + projection * vz,
        )
        max_deviation = max(max_deviation, euclidean_distance_km(point, closest))
    return max_deviation


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_columns())
        writer.writeheader()
        writer.writerows(rows)


def _write_json(
    rows: list[dict[str, object]],
    csv_path: Path,
    figure_paths: tuple[Path, ...],
    path: Path,
    results: list[_CaseResult],
) -> None:
    payload = {
        "comparison_name": "pseudo_bending_vs_dijkstra_ray_method_comparison",
        "csv_path": str(csv_path),
        "figure_paths": [str(figure_path) for figure_path in figure_paths],
        "rows": rows,
        "ray_paths": {
            str(result.row["case_id"]): {
                "dijkstra": [_point_dict(point) for point in result.dijkstra_path],
                "pseudo_bending": [_point_dict(point) for point in result.pseudo_bending_path],
            }
            for result in results
        },
        "method_independence_note": (
            "Pseudo-bending uses an interface-aware Snell-law layered backbone and does not use "
            "Dijkstra path coordinates for initialization."
        ),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_note(
    path: Path,
    repo_root: Path,
    csv_path: Path,
    json_path: Path,
    figure_paths: tuple[Path, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment: Pseudo-Bending vs Dijkstra Ray-Method Comparison",
        "",
        "## Objective",
        "",
        (
            "Compare the existing graph-based Dijkstra travel-time approximation with the "
            "independent `pseudo_bending_3d` ray tracer on the configured synthetic "
            "geological families."
        ),
        "",
        "## Outputs",
        "",
        f"- CSV: `{path_relative(csv_path, repo_root)}`",
        f"- JSON: `{path_relative(json_path, repo_root)}`",
    ]
    lines.extend(
        f"- Figure: `{path_relative(figure_path, repo_root)}`" for figure_path in figure_paths
    )
    lines.extend(
        [
            "",
            "## Cases",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['case_id']}",
                "",
                f"- Scenario: `{row['velocity_scenario']}`",
                (
                    f"- Source: `({row['source_x_km']}, {row['source_y_km']}, "
                    f"{row['source_z_km']})` km"
                ),
                (
                    f"- Receiver: `({row['receiver_x_km']}, {row['receiver_y_km']}, "
                    f"{row['receiver_z_km']})` km"
                ),
                f"- Dijkstra travel time: `{row['dijkstra_travel_time_s']}` s",
                f"- Pseudo-bending travel time: `{row['pseudo_bending_travel_time_s']}` s",
                (
                    "- Pseudo-bending maximum deviation from the straight ray: "
                    f"`{row['pseudo_bending_max_deviation_from_straight_km']}` km"
                ),
                f"- Pseudo-bending converged: `{row['pseudo_bending_converged']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "- This comparison is a bounded diagnostic case, not a final accuracy claim.",
            "- Dijkstra is evaluated as a graph-constrained shortest-time approximation.",
            "- Pseudo-bending is evaluated as an independent interface-aware ray tracer with Snell-law bending in the layered background.",
            "- Scenario interfaces are inserted as explicit crossing points and optimized while constrained to their surfaces.",
            "- The current implementation does not use smoothed trilinear velocity interpolation.",
            "- The non-horizontal interface optimizer is local and should be benchmarked before claiming global minimum travel-time behavior.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _point_dict(point: Point3D) -> dict[str, float]:
    return {
        "x_km": point.x_km,
        "y_km": point.y_km,
        "z_km": point.z_km,
    }
