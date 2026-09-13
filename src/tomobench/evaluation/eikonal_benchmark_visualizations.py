"""Workflow helpers for publication-friendly eikonal benchmark figures."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import Earthquake, Station
from tomobench.evaluation.eikonal_benchmarks import (
    EikonalBenchmarkCase,
    benchmark_cases,
    path_relative,
    run_eikonal_benchmarks,
)
from tomobench.generation.velocity_grids import build_cartesian_velocity_grid
from tomobench.simulation.travel_times import (
    EIKONAL_3D_FIRST_ARRIVAL_METHOD,
    _CartesianGridGraph,
    euclidean_distance_km,
)
from tomobench.utils.paths import get_repo_root
from tomobench.visualization.eikonal_benchmark_figures import (
    save_eikonal_connectivity_svg,
    save_eikonal_path_overlay_svg,
    save_eikonal_travel_time_slices_svg,
)


@dataclass(frozen=True)
class EikonalBenchmarkVisualizationOutputs:
    """Paths produced by the benchmark-visualization workflow."""

    connectivity_figure: Path
    path_overlay_figure: Path
    travel_time_field_figure: Path
    note_path: Path


def render_eikonal_benchmark_visualizations(
    settings: BenchmarkSettings | None = None,
) -> EikonalBenchmarkVisualizationOutputs:
    """Render a small figure set explaining the eikonal benchmark workflow."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    benchmark_csv_path = (
        repo_root / loaded_settings.outputs.benchmarks_dir / "eikonal_3d_prototype_benchmark.csv"
    )
    if not benchmark_csv_path.is_file():
        run_eikonal_benchmarks(loaded_settings)

    benchmark_rows = _read_benchmark_rows(benchmark_csv_path)
    figure_settings = replace(
        loaded_settings,
        simulation=replace(
            loaded_settings.simulation,
            method=EIKONAL_3D_FIRST_ARRIVAL_METHOD,
        ),
    )
    figure_dir = repo_root / loaded_settings.outputs.figures_dir / "eikonal_benchmark"
    note_path = (
        repo_root
        / loaded_settings.outputs.experiment_notes_dir
        / "eikonal_3d_benchmark_visualizations.md"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)

    connectivity_figure = save_eikonal_connectivity_svg(
        figure_dir / "eikonal_connectivity_6_18_26.svg"
    )

    case_by_name = {case.name: case for case in benchmark_cases(figure_settings)}
    homogeneous_case = case_by_name["homogeneous_on_grid_accuracy"]
    homogeneous_grid = build_cartesian_velocity_grid(
        figure_settings,
        homogeneous_case.velocity_model,
        scenario=homogeneous_case.velocity_scenario,
    )
    homogeneous_pair = _select_longest_pair(homogeneous_case)
    homogeneous_path = _build_case_path(
        homogeneous_grid,
        connectivity=figure_settings.eikonal_solver.connectivity,
        source=homogeneous_pair[0].hypocenter,
        receiver=homogeneous_pair[1].location,
    )
    path_overlay_figure = save_eikonal_path_overlay_svg(
        grid=homogeneous_grid,
        source=homogeneous_pair[0].hypocenter,
        receiver=homogeneous_pair[1].location,
        path_points=homogeneous_path.path_points,
        path_travel_time_s=homogeneous_path.travel_times_by_node[homogeneous_path.receiver_index],
        straight_travel_time_s=euclidean_distance_km(
            homogeneous_pair[0].hypocenter,
            homogeneous_pair[1].location,
        )
        / 5.0,
        connectivity=figure_settings.eikonal_solver.connectivity,
        grid_spacing_km=figure_settings.eikonal_solver.grid_spacing_km,
        benchmark_case_name=homogeneous_case.name,
        path=figure_dir / "eikonal_homogeneous_path_overlay.svg",
    )

    block_case = case_by_name["controlled_block_anomaly_vertical_path"]
    block_grid = build_cartesian_velocity_grid(
        figure_settings,
        block_case.velocity_model,
        scenario=block_case.velocity_scenario,
    )
    block_source, block_receiver = _illustrative_block_anomaly_pair()
    block_path = _build_case_path(
        block_grid,
        connectivity=figure_settings.eikonal_solver.connectivity,
        source=block_source,
        receiver=block_receiver,
    )
    travel_time_field_figure = save_eikonal_travel_time_slices_svg(
        grid=block_grid,
        travel_times_by_node=block_path.travel_times_by_node,
        source=block_source,
        receiver=block_receiver,
        path_points=block_path.path_points,
        connectivity=figure_settings.eikonal_solver.connectivity,
        grid_spacing_km=figure_settings.eikonal_solver.grid_spacing_km,
        benchmark_case_name="illustrative_block_anomaly_detour",
        path=figure_dir / "eikonal_block_anomaly_travel_time_field.svg",
    )

    _write_visualization_note(
        path=note_path,
        repo_root=repo_root,
        benchmark_csv_path=benchmark_csv_path,
        outputs=EikonalBenchmarkVisualizationOutputs(
            connectivity_figure=connectivity_figure,
            path_overlay_figure=path_overlay_figure,
            travel_time_field_figure=travel_time_field_figure,
            note_path=note_path,
        ),
        settings=figure_settings,
        benchmark_rows=benchmark_rows,
        homogeneous_case=homogeneous_case,
        homogeneous_source=homogeneous_pair[0].hypocenter,
        homogeneous_receiver=homogeneous_pair[1].location,
        block_case=block_case,
        block_source=block_source,
        block_receiver=block_receiver,
    )

    return EikonalBenchmarkVisualizationOutputs(
        connectivity_figure=connectivity_figure,
        path_overlay_figure=path_overlay_figure,
        travel_time_field_figure=travel_time_field_figure,
        note_path=note_path,
    )


@dataclass(frozen=True)
class _PathComputation:
    receiver_index: int
    travel_times_by_node: list[float]
    path_points: tuple[Point3D, ...]


def _build_case_path(
    grid,
    connectivity: int,
    source: Point3D,
    receiver: Point3D,
) -> _PathComputation:
    graph = _CartesianGridGraph(grid, connectivity)
    source_index = graph.nearest_node_index(source)
    receiver_index = graph.nearest_node_index(receiver)
    travel_times_by_node, _ = graph.shortest_travel_times_and_predecessors_from(source_index)
    path_indices = graph.shortest_path_indices(source_index, receiver_index)
    return _PathComputation(
        receiver_index=receiver_index,
        travel_times_by_node=travel_times_by_node,
        path_points=tuple(graph.node_point(index) for index in path_indices),
    )


def _select_longest_pair(case: EikonalBenchmarkCase) -> tuple[Earthquake, Station]:
    candidates: list[tuple[float, Earthquake, Station]] = []
    for earthquake in case.earthquakes.earthquakes:
        for station in case.stations.stations:
            candidates.append(
                (
                    euclidean_distance_km(earthquake.hypocenter, station.location),
                    earthquake,
                    station,
                )
            )
    _, earthquake, station = max(candidates, key=lambda value: value[0])
    return earthquake, station


def _illustrative_block_anomaly_pair() -> tuple[Point3D, Point3D]:
    """Return an off-axis pair that visibly bends toward the faster block anomaly."""
    return (
        Point3D(20.0, 50.0, 0.0),
        Point3D(80.0, 80.0, 20.0),
    )


def _read_benchmark_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _find_row(
    rows: list[dict[str, str]],
    benchmark_case: str,
    connectivity: int,
    grid_spacing_km: float,
) -> dict[str, str]:
    for row in rows:
        if (
            row["benchmark_case"] == benchmark_case
            and int(row["connectivity"]) == connectivity
            and float(row["grid_spacing_km"]) == grid_spacing_km
        ):
            return row
    raise ValueError("Could not find the requested benchmark row for the visualization note.")


def _write_visualization_note(
    path: Path,
    repo_root: Path,
    benchmark_csv_path: Path,
    outputs: EikonalBenchmarkVisualizationOutputs,
    settings: BenchmarkSettings,
    benchmark_rows: list[dict[str, str]],
    homogeneous_case: EikonalBenchmarkCase,
    homogeneous_source: Point3D,
    homogeneous_receiver: Point3D,
    block_case: EikonalBenchmarkCase,
    block_source: Point3D,
    block_receiver: Point3D,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    homogeneous_row = _find_row(
        benchmark_rows,
        benchmark_case=homogeneous_case.name,
        connectivity=settings.eikonal_solver.connectivity,
        grid_spacing_km=settings.eikonal_solver.grid_spacing_km,
    )
    block_row = _find_row(
        benchmark_rows,
        benchmark_case=block_case.name,
        connectivity=settings.eikonal_solver.connectivity,
        grid_spacing_km=settings.eikonal_solver.grid_spacing_km,
    )
    lines = [
        "# Experiment: Eikonal 3D Benchmark Visualizations",
        "",
        "## Objective",
        "",
        (
            "Create publication-friendly companion figures for the existing "
            "`eikonal_3d_prototype_benchmark.csv` output so the graph-based "
            "first-arrival workflow can be explained visually."
        ),
        "",
        "## Source Benchmark",
        "",
        f"- CSV: `{path_relative(benchmark_csv_path, repo_root)}`",
        f"- Selected connectivity for the figures: `{settings.eikonal_solver.connectivity}` neighbors",
        f"- Selected grid spacing for the figures: `{settings.eikonal_solver.grid_spacing_km}` km",
        "",
        "## Figure Outputs",
        "",
        f"- Connectivity schematic: `{path_relative(outputs.connectivity_figure, repo_root)}`",
        f"- Homogeneous path overlay: `{path_relative(outputs.path_overlay_figure, repo_root)}`",
        f"- Block-anomaly travel-time field: `{path_relative(outputs.travel_time_field_figure, repo_root)}`",
        "",
        "## Figure Selection Notes",
        "",
        (
            f"- The path overlay uses the benchmark case `{homogeneous_case.name}` and the "
            f"longest source-receiver pair in that control set: source "
            f"`({homogeneous_source.x_km}, {homogeneous_source.y_km}, {homogeneous_source.z_km})` km "
            f"to receiver `({homogeneous_receiver.x_km}, {homogeneous_receiver.y_km}, {homogeneous_receiver.z_km})` km."
        ),
        (
            f"- The travel-time field reuses the block-anomaly benchmark grid from `{block_case.name}` "
            "but switches to an illustrative off-axis pair so the Dijkstra detour is visible: "
            f"source `({block_source.x_km}, {block_source.y_km}, {block_source.z_km})` km "
            f"to receiver `({block_receiver.x_km}, {block_receiver.y_km}, {block_receiver.z_km})` km."
        ),
        (
            f"- In the benchmark CSV, the homogeneous control row at connectivity "
            f"`{settings.eikonal_solver.connectivity}` and `{settings.eikonal_solver.grid_spacing_km}` km "
            f"records `mean_absolute_error_s={homogeneous_row['mean_absolute_error_s']}` and "
            f"`simulation_time_s={homogeneous_row['simulation_time_s']}`."
        ),
        (
            f"- In the controlled block-anomaly row at the same settings, the exact-reference "
            f"comparison remains `mean_absolute_error_s={block_row['mean_absolute_error_s']}`."
        ),
        "",
        "## Interpretation",
        "",
        "- The connectivity schematic explains how graph neighborhood choice changes directional freedom before any travel-time field is considered.",
        "- The homogeneous path overlay isolates the geometry of Dijkstra propagation from lateral heterogeneity, making the node-to-node approximation easy to see against the straight ray.",
        "- The block-anomaly travel-time slices use an illustrative off-axis geometry so the recovered path can be compared directly against the straight ray and the benefit of Dijkstra propagation is visible.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
