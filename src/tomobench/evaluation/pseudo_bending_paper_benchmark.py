"""Paper-grade benchmark suite for the pseudo-bending 3D ray tracer."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import CartesianVelocityGrid3D, LayeredVelocityModel, VelocityLayer
from tomobench.evaluation.eikonal_benchmarks import path_relative
from tomobench.generation.velocity_grids import build_cartesian_velocity_grid
from tomobench.generation.velocity_models import generate_layered_velocity_model
from tomobench.simulation.ray_tracing import polyline_length_km, trace_pseudo_bending_ray
from tomobench.simulation.travel_times import PSEUDO_BENDING_3D_METHOD, _CartesianGridGraph
from tomobench.utils.paths import get_repo_root
from tomobench.visualization.ray_path_comparison import save_ray_path_comparison_svg


BENCHMARK_ID = "pseudo_bending_paper_benchmark_v1"
GEOLOGICAL_FAMILIES = (
    "layered",
    "block_anomaly",
    "faulted",
    "salt_dome",
    "dyke_intrusion",
)


@dataclass(frozen=True)
class PseudoBendingPaperBenchmarkOutputs:
    """Files produced by the pseudo-bending paper benchmark workflow."""

    csv_path: Path
    json_path: Path
    note_path: Path
    ray_path_dir: Path
    figure_paths: tuple[Path, ...]


@dataclass(frozen=True)
class PseudoBendingBenchmarkCase:
    """One bounded pseudo-bending benchmark case."""

    case_id: str
    scenario: str
    family: str
    source: Point3D
    receiver: Point3D
    reference_method: str
    comparison_kind: str
    velocity_model: LayeredVelocityModel | None = None


@dataclass(frozen=True)
class _ReferenceResult:
    travel_time_s: float
    path_length_km: float
    ray_path: tuple[Point3D, ...]
    runtime_s: float
    converged: bool | None
    iteration_count: int | None


@dataclass(frozen=True)
class _CaseResult:
    row: dict[str, object]
    ray_path_file: Path
    figure_path: Path
    reference_path: tuple[Point3D, ...]
    pseudo_bending_path: tuple[Point3D, ...]


def run_pseudo_bending_paper_benchmark(
    settings: BenchmarkSettings | None = None,
) -> PseudoBendingPaperBenchmarkOutputs:
    """Run bounded analytic and graph-comparison benchmarks for pseudo-bending."""
    loaded_settings = settings or load_settings()
    benchmark_settings = replace(
        loaded_settings,
        simulation=replace(loaded_settings.simulation, method=PSEUDO_BENDING_3D_METHOD),
    )
    repo_root = get_repo_root()
    benchmark_dir = repo_root / benchmark_settings.outputs.benchmarks_dir / BENCHMARK_ID
    figure_dir = repo_root / benchmark_settings.outputs.figures_dir / BENCHMARK_ID
    ray_path_dir = benchmark_dir / "ray_paths"
    note_path = repo_root / benchmark_settings.outputs.experiment_notes_dir / f"{BENCHMARK_ID}.md"
    csv_path = benchmark_dir / "summary.csv"
    json_path = benchmark_dir / "summary.json"

    ray_path_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    default_velocity_model = generate_layered_velocity_model(benchmark_settings)
    results = [
        _run_case(benchmark_settings, default_velocity_model, ray_path_dir, figure_dir, case)
        for case in benchmark_cases(benchmark_settings, default_velocity_model)
    ]
    rows = [result.row for result in results]
    figure_paths = tuple(result.figure_path for result in results)

    _write_csv(rows, csv_path)
    _write_json(rows, csv_path, figure_paths, ray_path_dir, json_path, benchmark_settings)
    summary_figures = _write_summary_figures(rows, figure_dir)
    all_figure_paths = (*figure_paths, *summary_figures)
    _write_note(
        path=note_path,
        rows=rows,
        repo_root=repo_root,
        csv_path=csv_path,
        json_path=json_path,
        ray_path_dir=ray_path_dir,
        figure_paths=all_figure_paths,
    )
    return PseudoBendingPaperBenchmarkOutputs(
        csv_path=csv_path,
        json_path=json_path,
        note_path=note_path,
        ray_path_dir=ray_path_dir,
        figure_paths=all_figure_paths,
    )


def pseudo_bending_paper_benchmark_columns() -> tuple[str, ...]:
    """Return the paper benchmark CSV schema."""
    return (
        "case_id",
        "scenario",
        "family",
        "comparison_kind",
        "reference_method",
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "receiver_x_km",
        "receiver_y_km",
        "receiver_z_km",
        "pseudo_bending_travel_time_s",
        "reference_travel_time_s",
        "travel_time_difference_s",
        "absolute_travel_time_difference_s",
        "relative_travel_time_difference",
        "pseudo_bending_path_length_km",
        "reference_path_length_km",
        "pseudo_bending_runtime_s",
        "reference_runtime_s",
        "total_runtime_s",
        "pseudo_bending_converged",
        "reference_converged",
        "pseudo_bending_iteration_count",
        "reference_iteration_count",
        "pseudo_bending_ray_point_count",
        "reference_ray_point_count",
        "grid_spacing_km",
        "connectivity",
        "node_count",
        "ray_path_file",
        "figure_file",
    )


def benchmark_cases(
    settings: BenchmarkSettings,
    default_velocity_model: LayeredVelocityModel | None = None,
) -> tuple[PseudoBendingBenchmarkCase, ...]:
    """Return the bounded benchmark cases used by the paper workflow."""
    layered_model = default_velocity_model or generate_layered_velocity_model(settings)
    source = Point3D(15.0, 50.0, 22.5)
    receiver = Point3D(85.0, 50.0, 0.0)
    homogeneous_model = _homogeneous_velocity_model(5.0)
    cases = [
        PseudoBendingBenchmarkCase(
            case_id="homogeneous_analytic_direct_ray",
            scenario="layered",
            family="layered",
            source=Point3D(20.0, 20.0, 20.0),
            receiver=Point3D(80.0, 70.0, 0.0),
            reference_method="homogeneous_distance_over_velocity",
            comparison_kind="analytic_reference",
            velocity_model=homogeneous_model,
        ),
        PseudoBendingBenchmarkCase(
            case_id="layered_analytic_snell_direct_ray",
            scenario="layered",
            family="layered",
            source=source,
            receiver=receiver,
            reference_method="layered_snell_law_direct_ray",
            comparison_kind="analytic_reference",
            velocity_model=layered_model,
        ),
    ]
    cases.extend(
        PseudoBendingBenchmarkCase(
            case_id=f"{family}_pseudo_bending_vs_dijkstra",
            scenario=family,
            family=family,
            source=source,
            receiver=receiver,
            reference_method="eikonal_dijkstra_graph_shortest_path",
            comparison_kind="eikonal_dijkstra_comparison",
            velocity_model=layered_model,
        )
        for family in GEOLOGICAL_FAMILIES
    )
    return tuple(cases)


def _run_case(
    settings: BenchmarkSettings,
    default_velocity_model: LayeredVelocityModel,
    ray_path_dir: Path,
    figure_dir: Path,
    case: PseudoBendingBenchmarkCase,
) -> _CaseResult:
    case_start = perf_counter()
    velocity_model = case.velocity_model or default_velocity_model
    grid = build_cartesian_velocity_grid(settings, velocity_model, scenario=case.scenario)

    pseudo_start = perf_counter()
    pseudo_ray = trace_pseudo_bending_ray(
        source=case.source,
        receiver=case.receiver,
        velocity_grid=grid,
        settings=settings.pseudo_bending_solver,
    )
    pseudo_runtime_s = perf_counter() - pseudo_start

    reference = _reference_result(settings, velocity_model, grid, case)
    total_runtime_s = perf_counter() - case_start
    diff_s = pseudo_ray.travel_time_s - reference.travel_time_s
    relative_diff = 0.0 if reference.travel_time_s == 0.0 else diff_s / reference.travel_time_s

    ray_path_file = ray_path_dir / f"{case.case_id}.json"
    figure_path = figure_dir / f"{case.case_id}.svg"
    _write_ray_path_file(
        path=ray_path_file,
        case=case,
        reference=reference,
        pseudo_path=pseudo_ray.ray_path,
    )
    save_ray_path_comparison_svg(
        grid=grid,
        source=case.source,
        receiver=case.receiver,
        dijkstra_path=reference.ray_path,
        pseudo_bending_path=pseudo_ray.ray_path,
        dijkstra_time_s=reference.travel_time_s,
        pseudo_bending_time_s=pseudo_ray.travel_time_s,
        path=figure_path,
        title="Reference vs pseudo-bending ray paths",
        reference_label=_short_reference_label(case.reference_method),
        pseudo_label="Pseudo-bending",
    )

    row = {
        "case_id": case.case_id,
        "scenario": case.scenario,
        "family": case.family,
        "comparison_kind": case.comparison_kind,
        "reference_method": case.reference_method,
        "source_x_km": case.source.x_km,
        "source_y_km": case.source.y_km,
        "source_z_km": case.source.z_km,
        "receiver_x_km": case.receiver.x_km,
        "receiver_y_km": case.receiver.y_km,
        "receiver_z_km": case.receiver.z_km,
        "pseudo_bending_travel_time_s": round(pseudo_ray.travel_time_s, 8),
        "reference_travel_time_s": round(reference.travel_time_s, 8),
        "travel_time_difference_s": round(diff_s, 8),
        "absolute_travel_time_difference_s": round(abs(diff_s), 8),
        "relative_travel_time_difference": round(relative_diff, 10),
        "pseudo_bending_path_length_km": round(pseudo_ray.path_length_km, 8),
        "reference_path_length_km": round(reference.path_length_km, 8),
        "pseudo_bending_runtime_s": round(pseudo_runtime_s, 8),
        "reference_runtime_s": round(reference.runtime_s, 8),
        "total_runtime_s": round(total_runtime_s, 8),
        "pseudo_bending_converged": pseudo_ray.converged,
        "reference_converged": reference.converged,
        "pseudo_bending_iteration_count": pseudo_ray.iteration_count,
        "reference_iteration_count": reference.iteration_count,
        "pseudo_bending_ray_point_count": len(pseudo_ray.ray_path),
        "reference_ray_point_count": len(reference.ray_path),
        "grid_spacing_km": settings.eikonal_solver.grid_spacing_label,
        "connectivity": settings.eikonal_solver.connectivity,
        "node_count": grid.metadata["node_count"],
        "ray_path_file": str(ray_path_file),
        "figure_file": str(figure_path),
    }
    return _CaseResult(
        row=row,
        ray_path_file=ray_path_file,
        figure_path=figure_path,
        reference_path=reference.ray_path,
        pseudo_bending_path=pseudo_ray.ray_path,
    )


def _reference_result(
    settings: BenchmarkSettings,
    velocity_model: LayeredVelocityModel,
    grid: CartesianVelocityGrid3D,
    case: PseudoBendingBenchmarkCase,
) -> _ReferenceResult:
    start = perf_counter()
    if case.reference_method == "homogeneous_distance_over_velocity":
        velocity = velocity_model.layers[0].p_velocity_km_per_s
        path = (case.source, case.receiver)
        return _ReferenceResult(
            travel_time_s=polyline_length_km(path) / velocity,
            path_length_km=polyline_length_km(path),
            ray_path=path,
            runtime_s=perf_counter() - start,
            converged=True,
            iteration_count=0,
        )
    if case.reference_method == "layered_snell_law_direct_ray":
        path = _layered_snell_reference_path(case.source, case.receiver, velocity_model)
        return _ReferenceResult(
            travel_time_s=_layered_path_time_s(path, velocity_model),
            path_length_km=polyline_length_km(path),
            ray_path=path,
            runtime_s=perf_counter() - start,
            converged=True,
            iteration_count=0,
        )
    graph = _CartesianGridGraph(grid, settings.eikonal_solver.connectivity)
    source_index = graph.nearest_node_index(case.source)
    receiver_index = graph.nearest_node_index(case.receiver)
    times, _ = graph.shortest_travel_times_and_predecessors_from(source_index)
    path_indices = graph.shortest_path_indices(source_index, receiver_index)
    path = tuple(graph.node_point(index) for index in path_indices)
    return _ReferenceResult(
        travel_time_s=times[receiver_index],
        path_length_km=polyline_length_km(path),
        ray_path=path,
        runtime_s=perf_counter() - start,
        converged=True,
        iteration_count=None,
    )


def _layered_snell_reference_path(
    source: Point3D,
    receiver: Point3D,
    velocity_model: LayeredVelocityModel,
) -> tuple[Point3D, ...]:
    horizontal_offset = math.hypot(source.x_km - receiver.x_km, source.y_km - receiver.y_km)
    vertical_span = abs(source.z_km - receiver.z_km)
    if horizontal_offset == 0.0 or vertical_span == 0.0:
        return (source, receiver)

    ux = (receiver.x_km - source.x_km) / horizontal_offset
    uy = (receiver.y_km - source.y_km) / horizontal_offset
    z_values = _crossed_depth_values(source.z_km, receiver.z_km, velocity_model)
    segments = []
    for z0, z1 in zip(z_values, z_values[1:]):
        thickness = abs(z1 - z0)
        velocity = _layer_velocity_at((z0 + z1) / 2.0, velocity_model)
        segments.append((thickness, velocity))

    p = _solve_layered_ray_parameter(horizontal_offset, tuple(segments))
    path = [source]
    current_x = source.x_km
    current_y = source.y_km
    for z_next, (thickness, velocity) in zip(z_values[1:], segments):
        segment_horizontal = (
            thickness * p * velocity / math.sqrt(max(1.0 - (p * velocity) ** 2, 1.0e-15))
        )
        current_x += ux * segment_horizontal
        current_y += uy * segment_horizontal
        path.append(Point3D(current_x, current_y, z_next))
    path[-1] = receiver
    return tuple(path)


def _crossed_depth_values(
    source_z_km: float,
    receiver_z_km: float,
    velocity_model: LayeredVelocityModel,
) -> tuple[float, ...]:
    low = min(source_z_km, receiver_z_km)
    high = max(source_z_km, receiver_z_km)
    boundaries = sorted(
        {
            boundary
            for layer in velocity_model.layers
            for boundary in (layer.top_depth_km, layer.bottom_depth_km)
            if low < boundary < high
        }
    )
    if receiver_z_km < source_z_km:
        boundaries.reverse()
    return tuple([source_z_km, *boundaries, receiver_z_km])


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


def _layered_path_time_s(
    path: tuple[Point3D, ...],
    velocity_model: LayeredVelocityModel,
) -> float:
    total = 0.0
    for start, end in zip(path, path[1:]):
        midpoint_depth = (start.z_km + end.z_km) / 2.0
        total += polyline_length_km((start, end)) / _layer_velocity_at(
            midpoint_depth,
            velocity_model,
        )
    return total


def _layer_velocity_at(depth_km: float, velocity_model: LayeredVelocityModel) -> float:
    for layer in velocity_model.layers:
        if layer.top_depth_km <= depth_km < layer.bottom_depth_km:
            return layer.p_velocity_km_per_s
    deepest = velocity_model.layers[-1]
    if math.isclose(depth_km, deepest.bottom_depth_km, rel_tol=0.0, abs_tol=1.0e-9):
        return deepest.p_velocity_km_per_s
    raise ValueError(f"Depth {depth_km} km is outside the layered velocity model.")


def _homogeneous_velocity_model(velocity_km_per_s: float) -> LayeredVelocityModel:
    return LayeredVelocityModel(
        model_id="pseudo_bending_benchmark_homogeneous_velocity",
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


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pseudo_bending_paper_benchmark_columns())
        writer.writeheader()
        writer.writerows(rows)


def _write_json(
    rows: list[dict[str, object]],
    csv_path: Path,
    figure_paths: tuple[Path, ...],
    ray_path_dir: Path,
    path: Path,
    settings: BenchmarkSettings,
) -> None:
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "purpose": "Bounded validation of pseudo_bending_3d before Phase 3 ML corpus regeneration.",
        "simulation_method": settings.simulation.method,
        "csv_path": str(csv_path),
        "ray_path_dir": str(ray_path_dir),
        "figure_paths": [str(figure_path) for figure_path in figure_paths],
        "rows": rows,
        "interpretation_boundary": (
            "Analytic cases validate homogeneous and horizontal layered direct-ray behavior. "
            "Dijkstra/eikonal cases are comparison prototypes, not globally exact references "
            "for complex geological interfaces."
        ),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_ray_path_file(
    path: Path,
    case: PseudoBendingBenchmarkCase,
    reference: _ReferenceResult,
    pseudo_path: tuple[Point3D, ...],
) -> None:
    payload = {
        "case_id": case.case_id,
        "scenario": case.scenario,
        "family": case.family,
        "reference_method": case.reference_method,
        "source": _point_dict(case.source),
        "receiver": _point_dict(case.receiver),
        "reference_ray_path": [_point_dict(point) for point in reference.ray_path],
        "pseudo_bending_ray_path": [_point_dict(point) for point in pseudo_path],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_summary_figures(rows: list[dict[str, object]], figure_dir: Path) -> tuple[Path, ...]:
    return (
        _write_bar_svg(
            rows=rows,
            field="absolute_travel_time_difference_s",
            title="Pseudo-bending benchmark travel-time differences",
            y_label="absolute difference (s)",
            path=figure_dir / "travel_time_difference_summary.svg",
        ),
        _write_bar_svg(
            rows=rows,
            field="pseudo_bending_runtime_s",
            title="Pseudo-bending benchmark runtime",
            y_label="runtime (s)",
            path=figure_dir / "runtime_summary.svg",
        ),
        _write_bar_svg(
            rows=rows,
            field="pseudo_bending_iteration_count",
            title="Pseudo-bending benchmark iterations",
            y_label="iteration count",
            path=figure_dir / "iteration_summary.svg",
        ),
    )


def _write_bar_svg(
    rows: list[dict[str, object]],
    field: str,
    title: str,
    y_label: str,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1080, 520
    left, top, chart_width, chart_height = 88.0, 72.0, 900.0, 300.0
    values = [float(row[field]) for row in rows]
    max_value = max(values) if values else 0.0
    scale_max = max_value if max_value > 0.0 else 1.0
    bar_gap = 14.0
    bar_width = (chart_width - bar_gap * (len(rows) - 1)) / len(rows)
    bars = []
    labels = []
    for index, row in enumerate(rows):
        value = float(row[field])
        bar_height = value / scale_max * chart_height
        x = left + index * (bar_width + bar_gap)
        y = top + chart_height - bar_height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            'fill="#2563eb" />'
        )
        labels.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{top + chart_height + 18:.2f}" '
            'font-family="Arial, sans-serif" font-size="10" text-anchor="middle" '
            f'fill="#111827" transform="rotate(35 {x + bar_width / 2:.2f} {top + chart_height + 18:.2f})">'
            f"{row['case_id']}</text>"
        )
        labels.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" '
            'font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#374151">'
            f"{value:.4g}</text>"
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{left:.2f}" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{title}</text>
  <rect x="{left:.2f}" y="{top:.2f}" width="{chart_width:.2f}" height="{chart_height:.2f}" fill="#f8fafc" stroke="#0f172a" />
  <text x="24" y="{top + chart_height / 2:.2f}" transform="rotate(-90 24 {top + chart_height / 2:.2f})" font-family="Arial, sans-serif" font-size="13" fill="#111827">{y_label}</text>
  {"".join(bars)}
  {"".join(labels)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def _write_note(
    path: Path,
    rows: list[dict[str, object]],
    repo_root: Path,
    csv_path: Path,
    json_path: Path,
    ray_path_dir: Path,
    figure_paths: tuple[Path, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    families = ", ".join(f"`{family}`" for family in GEOLOGICAL_FAMILIES)
    lines = [
        "# Experiment: Pseudo-Bending Paper Benchmark v1",
        "",
        "## Objective",
        "",
        (
            "Validate the configured `pseudo_bending_3d` simulator with bounded, reproducible "
            "cases before regenerating the paper-facing ML corpus."
        ),
        "",
        "## Benchmark Cases",
        "",
        "- Homogeneous analytic direct-ray reference.",
        "- Horizontal layered analytic Snell-law direct-ray reference.",
        f"- Pseudo-bending versus graph-based Dijkstra/eikonal comparison for {families}.",
        "",
        "## Outputs",
        "",
        f"- CSV summary: `{path_relative(csv_path, repo_root)}`",
        f"- JSON summary: `{path_relative(json_path, repo_root)}`",
        f"- Per-case ray paths: `{path_relative(ray_path_dir, repo_root)}`",
    ]
    lines.extend(
        f"- Figure: `{path_relative(figure_path, repo_root)}`" for figure_path in figure_paths
    )
    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            (
                "Each row records source and receiver coordinates, scenario/family, reference "
                "method, pseudo-bending and reference travel times, signed/absolute/relative "
                "travel-time differences, path lengths, runtimes, convergence status, iteration "
                "counts, and ray-path point counts."
            ),
            "",
            "## Summary",
            "",
        ]
    )
    for row in rows:
        lines.append(
            f"- `{row['case_id']}`: reference=`{row['reference_method']}`, "
            f"abs_diff_s=`{row['absolute_travel_time_difference_s']}`, "
            f"pseudo_converged=`{row['pseudo_bending_converged']}`, "
            f"iterations=`{row['pseudo_bending_iteration_count']}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- The homogeneous and horizontal layered cases are analytic direct-ray checks.",
            "- The Dijkstra/eikonal results are graph-constrained comparison values, not globally exact solutions.",
            "- Agreement in these cases supports bounded use of `pseudo_bending_3d`; it does not prove global exactness for all geological interfaces.",
            "- Results should be interpreted as validation evidence for the configured synthetic families and geometry used here.",
            "",
            "## Limitations",
            "",
            "- The pseudo-bending implementation uses piecewise-constant interface-aware travel times without smoothed trilinear interpolation.",
            "- Non-horizontal and curved geological interfaces use local constrained crossing-point optimization.",
            "- The benchmark uses a small number of deterministic source-receiver pairs to stay fast and reproducible.",
            "- The Dijkstra comparison is affected by Cartesian grid spacing and connectivity.",
            "",
            "## Next Recommended Step",
            "",
            "Proceed to Phase 3 by regenerating the pseudo-bending ML corpus after reviewing these benchmark artifacts.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _short_reference_label(reference_method: str) -> str:
    if reference_method == "homogeneous_distance_over_velocity":
        return "Homogeneous analytic"
    if reference_method == "layered_snell_law_direct_ray":
        return "Layered Snell analytic"
    return "Dijkstra/eikonal"


def _point_dict(point: Point3D) -> dict[str, float]:
    return {"x_km": point.x_km, "y_km": point.y_km, "z_km": point.z_km}
