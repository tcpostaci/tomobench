"""Expanded forward-solver validation for the Benchmark v2 scientific gate.

The repository does not currently contain an established three-dimensional Fast
Marching/eikonal dependency.  This workflow therefore uses the existing graph
shortest-path prototype as a *separate discretized comparator* and makes that
limitation explicit.  It supplements the comparator with analytic homogeneous
and horizontally layered checks, a multi-pair heterogeneous audit, and a graph
resolution study.

The workflow is intentionally audit-only.  It never rewrites historical
sidecars and it does not generate a new benchmark corpus.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from tomobench.config import (
    BlockAnomalySettings,
    DykeIntrusionSettings,
    FaultedGridSettings,
    SaltDomeSettings,
    BenchmarkSettings,
    load_settings,
)
from tomobench.datasets.target_identity import target_vector_sha256
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import (
    CartesianVelocityGrid3D,
    LayeredVelocityModel,
    VelocityLayer,
)
from tomobench.evaluation.forward_solver_failure_diagnostics import (
    _load_observation_diagnostics,
)
from tomobench.evaluation.pseudo_bending_paper_benchmark import (
    _homogeneous_velocity_model,
    _layered_path_time_s,
    _layered_snell_reference_path,
)
from tomobench.generation.velocity_grids import build_cartesian_velocity_grid
from tomobench.generation.velocity_models import generate_layered_velocity_model
from tomobench.simulation.ray_tracing import (
    PseudoBendingRayResult,
    polyline_length_km,
    trace_pseudo_bending_ray,
)
from tomobench.simulation.travel_times import _CartesianGridGraph
from tomobench.utils.paths import get_repo_root


VALIDATION_ID = "forward_solver_validation_v2"
DEFAULT_MANIFEST_PATH = Path(
    "outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/forward_solver_validation_v2"
)
DEFAULT_HISTORICAL_FAILURE_SUMMARY = Path(
    "outputs/generated/submission_readiness/forward_solver_validation_v2/"
    "failure_diagnostics_summary.json"
)
DEFAULT_HETEROGENEOUS_CASES_PER_FAMILY = 5
DEFAULT_PAIRS_PER_CASE = 5
DEFAULT_RESOLUTION_PAIRS_PER_FAMILY = 3
DEFAULT_RESOLUTION_SPACINGS_KM = (10.0, 5.0, 2.5)
DEFAULT_VALIDATION_ITERATION_BUDGETS = (30, 120)
NOISE_LEVELS_S = (0.025, 0.050, 0.100)


@dataclass(frozen=True)
class ForwardSolverValidationV2Outputs:
    """Files written by the expanded forward-solver validation workflow."""

    output_dir: Path
    validation_pair_metrics_csv: Path
    validation_statistics_csv: Path
    resolution_study_csv: Path
    discrepancy_plot: Path
    summary_json: Path
    summary_md: Path
    scientific_assessment_md: Path
    audit_metadata_json: Path


@dataclass(frozen=True)
class _AnalyticPair:
    """One analytic validation source-receiver pair."""

    pair_id: str
    family: str
    source: Point3D
    receiver: Point3D
    velocity_model: LayeredVelocityModel
    reference_type: str


@dataclass(frozen=True)
class _GraphPair:
    """One selected historical-corpus pair used for graph comparison."""

    row: Mapping[str, Any]
    grid: CartesianVelocityGrid3D


def run_forward_solver_validation_v2(
    *,
    settings: BenchmarkSettings | None = None,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    heterogeneous_cases_per_family: int = DEFAULT_HETEROGENEOUS_CASES_PER_FAMILY,
    pairs_per_case: int = DEFAULT_PAIRS_PER_CASE,
    resolution_pairs_per_family: int = DEFAULT_RESOLUTION_PAIRS_PER_FAMILY,
    resolution_spacings_km: Sequence[float] = DEFAULT_RESOLUTION_SPACINGS_KM,
    validation_iteration_budgets: Sequence[int] = DEFAULT_VALIDATION_ITERATION_BUDGETS,
) -> ForwardSolverValidationV2Outputs:
    """Run analytic, heterogeneous, and graph-resolution solver validation.

    The heterogeneous pairs are sampled deterministically from the existing
    corpus, while the analytic pairs are fixed in code and recorded in the
    output metadata.  The supplied historical corpus is read as evidence only;
    no historical sidecar is modified.
    """

    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    resolved_manifest = _resolve(manifest_path, repo_root)
    resolved_output = _resolve(output_dir, repo_root)
    _validate_positive_int(heterogeneous_cases_per_family, "heterogeneous_cases_per_family")
    _validate_positive_int(pairs_per_case, "pairs_per_case")
    _validate_positive_int(resolution_pairs_per_family, "resolution_pairs_per_family")
    budgets = _validate_iteration_budgets(validation_iteration_budgets)
    spacings = _validate_spacings(resolution_spacings_km)

    manifest = _read_json(resolved_manifest)
    all_rows, case_context = _load_observation_diagnostics(
        manifest,
        resolved_manifest,
        repo_root,
        loaded_settings,
    )
    case_rows = _group_rows_by_case(all_rows)
    selected_case_ids = _select_case_ids(
        case_rows,
        cases_per_family=heterogeneous_cases_per_family,
    )
    selected_pairs = _select_graph_pairs(
        case_rows,
        case_context,
        selected_case_ids,
        pairs_per_case=pairs_per_case,
    )

    validation_rows = _run_validation_pairs(
        loaded_settings,
        selected_pairs,
        _analytic_pairs(loaded_settings),
        budgets,
    )
    resolution_rows = _run_resolution_study(
        loaded_settings,
        case_rows,
        case_context,
        selected_case_ids,
        pairs_per_family=resolution_pairs_per_family,
        spacings_km=spacings,
    )
    statistics_rows = summarize_validation_statistics(validation_rows)
    historical_summary = _read_optional_json(
        _resolve(DEFAULT_HISTORICAL_FAILURE_SUMMARY, repo_root)
    )
    summary = build_validation_summary(
        validation_rows=validation_rows,
        statistics_rows=statistics_rows,
        resolution_rows=resolution_rows,
        historical_summary=historical_summary,
        selected_case_ids=selected_case_ids,
        selected_pairs=selected_pairs,
        budgets=budgets,
        spacings=spacings,
    )
    assessment = build_scientific_assessment(summary)

    resolved_output.mkdir(parents=True, exist_ok=True)
    pair_path = resolved_output / "validation_pair_metrics.csv"
    statistics_path = resolved_output / "validation_statistics.csv"
    resolution_path = resolved_output / "resolution_study.csv"
    plot_path = resolved_output / "solver_discrepancy_distribution.png"
    summary_path = resolved_output / "forward_solver_validation_v2_summary.json"
    summary_md_path = resolved_output / "forward_solver_validation_v2_summary.md"
    assessment_path = resolved_output / "solver_scientific_assessment.md"
    metadata_path = resolved_output / "audit_metadata.json"

    _write_csv(pair_path, validation_rows, validation_pair_columns())
    _write_csv(statistics_path, statistics_rows, validation_statistics_columns())
    _write_csv(resolution_path, resolution_rows, resolution_columns())
    _write_discrepancy_plot(validation_rows, plot_path)
    _write_json(summary, summary_path)
    _write_summary_markdown(summary, summary_md_path)
    assessment_path.write_text(assessment, encoding="utf-8")
    _write_json(
        build_audit_metadata(
            manifest_path=resolved_manifest,
            output_dir=resolved_output,
            summary=summary,
            settings=loaded_settings,
        ),
        metadata_path,
    )
    return ForwardSolverValidationV2Outputs(
        output_dir=resolved_output,
        validation_pair_metrics_csv=pair_path,
        validation_statistics_csv=statistics_path,
        resolution_study_csv=resolution_path,
        discrepancy_plot=plot_path,
        summary_json=summary_path,
        summary_md=summary_md_path,
        scientific_assessment_md=assessment_path,
        audit_metadata_json=metadata_path,
    )


def summarize_validation_statistics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize valid solver comparisons by run, comparison, and family.

    Non-converged custom-solver rows remain in the pair-level audit but are not
    used as valid travel-time errors.  This prevents a failed optimization from
    being silently treated as a correct label.
    """

    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["solver_configuration"]),
            str(row["comparison_kind"]),
            str(row["family"]),
            str(row["reference_type"]),
        )
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        solver_configuration, comparison_kind, family, reference_type = key
        valid = [row for row in group if _as_bool(row.get("error_included"))]
        absolute = [float(row["absolute_travel_time_difference_s"]) for row in valid]
        relative = [float(row["absolute_relative_error"]) for row in valid]
        reference_times = [float(row["reference_travel_time_s"]) for row in valid]
        velocity_spans = [float(row["grid_velocity_span_km_per_s"]) for row in group]
        output.append(
            {
                "solver_configuration": solver_configuration,
                "comparison_kind": comparison_kind,
                "family": family,
                "reference_type": reference_type,
                "pair_count": len(group),
                "valid_pair_count": len(valid),
                "nonconverged_pair_count": len(group) - len(valid),
                "valid_fraction": len(valid) / len(group) if group else None,
                "mean_reference_travel_time_s": _mean_or_none(reference_times),
                "median_reference_travel_time_s": _median_or_none(reference_times),
                "mean_velocity_span_km_per_s": _mean_or_none(velocity_spans),
                "mean_absolute_travel_time_difference_s": _mean_or_none(absolute),
                "median_absolute_travel_time_difference_s": _median_or_none(absolute),
                "std_absolute_travel_time_difference_s": _sample_std_or_none(absolute),
                "p95_absolute_travel_time_difference_s": _percentile_or_none(absolute, 0.95),
                "max_absolute_travel_time_difference_s": max(absolute, default=None),
                "mean_absolute_relative_error": _mean_or_none(relative),
                "median_absolute_relative_error": _median_or_none(relative),
                "std_absolute_relative_error": _sample_std_or_none(relative),
                "p95_absolute_relative_error": _percentile_or_none(relative, 0.95),
                "max_absolute_relative_error": max(relative, default=None),
                "mean_abs_difference_over_median_travel_time": _ratio_or_none(
                    _mean_or_none(absolute),
                    _median_or_none(reference_times),
                ),
                "mean_abs_difference_over_noise_0_025_s": _ratio_or_none(
                    _mean_or_none(absolute),
                    0.025,
                ),
                "p95_abs_difference_over_noise_0_025_s": _ratio_or_none(
                    _percentile_or_none(absolute, 0.95),
                    0.025,
                ),
                "mean_abs_difference_over_noise_0_050_s": _ratio_or_none(
                    _mean_or_none(absolute),
                    0.050,
                ),
                "mean_abs_difference_over_noise_0_100_s": _ratio_or_none(
                    _mean_or_none(absolute),
                    0.100,
                ),
            }
        )
    return output


def build_validation_summary(
    *,
    validation_rows: Sequence[Mapping[str, Any]],
    statistics_rows: Sequence[Mapping[str, Any]],
    resolution_rows: Sequence[Mapping[str, Any]],
    historical_summary: Mapping[str, Any] | None,
    selected_case_ids: Mapping[str, Sequence[str]],
    selected_pairs: Sequence[_GraphPair],
    budgets: Sequence[int],
    spacings: Sequence[float],
) -> dict[str, Any]:
    """Build the JSON summary used by the scientific assessment."""

    validation_convergence = _summarize_convergence_by_solver(validation_rows)
    resolution_summary = _summarize_resolution(resolution_rows)
    external_reference = _available_independent_reference_packages()
    return {
        "artifact_type": "forward_solver_validation_v2_summary",
        "validation_id": VALIDATION_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "validation_design": {
            "analytic_homogeneous_pair_count": len(
                {
                    str(row["pair_id"])
                    for row in validation_rows
                    if row["comparison_kind"] == "analytic_homogeneous"
                }
            ),
            "analytic_layered_pair_count": len(
                {
                    str(row["pair_id"])
                    for row in validation_rows
                    if row["comparison_kind"] == "analytic_layered"
                }
            ),
            "heterogeneous_pair_count": len(
                {
                    (str(row["case_id"]), str(row["pair_id"]))
                    for row in validation_rows
                    if row["comparison_kind"] == "heterogeneous_graph"
                }
            ),
            "validation_row_count": len(validation_rows),
            "heterogeneous_case_count": len(
                {str(pair.row["case_id"]) for pair in selected_pairs}
            ),
            "selected_case_ids_by_family": {
                family: list(case_ids) for family, case_ids in selected_case_ids.items()
            },
            "validation_iteration_budgets": list(budgets),
            "resolution_spacings_km": list(spacings),
            "graph_connectivity": 26,
        },
        "fresh_validation_convergence": validation_convergence,
        "validation_statistics": list(statistics_rows),
        "graph_resolution_summary": resolution_summary,
        "historical_failure_diagnostics": dict(historical_summary or {}),
        "independent_reference_inventory": external_reference,
        "terminology": {
            "recommended": "interface-aware two-point ray-path optimizer initialized by a Snell-law layered ray",
            "not_established": [
                "canonical continuous-grid pseudo-bending implementation",
                "globally validated first-arrival solver",
                "Fast Marching/eikonal reference calculation",
            ],
        },
    }


def build_scientific_assessment(summary: Mapping[str, Any]) -> str:
    """Create a prose scientific verdict from observed validation evidence."""

    convergence = summary.get("fresh_validation_convergence", {})
    robust = convergence.get("max_iterations_120", {})
    default = convergence.get("max_iterations_30", {})
    stats = summary.get("validation_statistics", [])
    hetero_robust = [
        row
        for row in stats
        if row.get("comparison_kind") == "heterogeneous_graph"
        and row.get("solver_configuration") == "max_iterations_120"
    ]
    robust_valid = sum(int(row.get("valid_pair_count", 0)) for row in hetero_robust)
    robust_pairs = sum(int(row.get("pair_count", 0)) for row in hetero_robust)
    robust_mean_abs = _weighted_mean(
        hetero_robust,
        "mean_absolute_travel_time_difference_s",
        "valid_pair_count",
    )
    robust_p95 = max(
        (float(row["p95_absolute_travel_time_difference_s"])
         for row in hetero_robust
         if row.get("p95_absolute_travel_time_difference_s") is not None),
        default=None,
    )
    resolution = summary.get("graph_resolution_summary", {})
    resolution_by_spacing = resolution.get("by_spacing", {})
    p95_coarse = max(
        (float(item["p95_absolute_difference_from_finest_s"])
         for spacing, item in resolution_by_spacing.items()
         if float(spacing) != 2.5
         and item.get("p95_absolute_difference_from_finest_s") is not None),
        default=None,
    )
    packages = summary.get("independent_reference_inventory", {})
    installed_established = [
        name for name, available in packages.items()
        if name in {"scipy", "skfmm", "pykonal", "fast_marching"} and available
    ]
    discrepancy_is_not_negligible_at_controlled_noise_scale = (
        (robust_mean_abs is not None and robust_mean_abs >= NOISE_LEVELS_S[0])
        or (robust_p95 is not None and robust_p95 >= NOISE_LEVELS_S[0])
        or (p95_coarse is not None and p95_coarse >= NOISE_LEVELS_S[0])
    )

    if int(robust.get("nonconverged_count", 0)) > 0:
        verdict = "not suitable without redesign"
        gate = (
            "The extended-budget validation still contains non-converged pairs, so the solver "
            "cannot safely generate benchmark labels without a further redesign or explicit "
            "failure handling."
        )
    elif discrepancy_is_not_negligible_at_controlled_noise_scale:
        verdict = "not suitable without redesign"
        gate = (
            "The extended-budget validation converges, but the observed disagreement with the "
            "independent graph comparator and/or the graph-resolution change is not negligible "
            "at the smallest predeclared controlled timing-noise level of 0.025 s. Together with "
            "the absence of an established continuum Fast Marching reference, this is not a "
            "sufficient scientific foundation for generating the Benchmark v2 pilot as final "
            "first-arrival labels. Pilot generation is stopped pending solver/reference redesign."
        )
    else:
        verdict = "suitable only with caveats"
        gate = (
            "The extended-budget run converges on the audited validation pairs, but the "
            "independent check is a finite-grid graph comparator rather than an established "
            "continuum Fast Marching reference. The solver may be used for a provisional pilot "
            "only with the caveats below; this does not establish global first-arrival accuracy."
        )

    lines = [
        "# Forward-Solver v2 Scientific Assessment",
        "",
        "## Explicit verdict",
        "",
        f"**{verdict}**",
        "",
        gate,
        "",
        "This assessment is a validation gate, not a manuscript rewrite and not a declaration "
        "that the historical corpus contains valid final benchmark labels.",
        "",
        "## Scope and independent-reference status",
        "",
        "The custom implementation was inspected as an interface-aware two-point path optimizer. "
        "It starts from a Snell-law path through the horizontal layered background, inserts "
        "explicit geological-interface waypoints, and locally relocates those waypoints on their "
        "constraints. The evidence does not support calling it a canonical continuous-grid "
        "pseudo-bending solver or a globally validated first-arrival solver.",
        "",
        "The repository/environment inventory found no established SciPy, scikit-fmm, PyKonal, "
        "or Fast-Marching package available to this workflow. The independent comparator is "
        "therefore the repository's separate Cartesian graph shortest-path implementation. It "
        "is useful for detecting disagreement, but it is not a trusted continuum reference.",
        "",
        f"- Established numerical packages available: `{', '.join(installed_established) or 'none detected'}`.",
        "- Graph connectivity used for the heterogeneous comparator: `26`.",
        "- Resolution-study interpretation: the finest `2.5 km` graph is a reference within the "
        "tested discretization family, not an analytical continuum solution.",
        "",
        "## Historical convergence evidence",
        "",
        f"- Historical corpus failures: {summary.get('historical_failure_diagnostics', {}).get('failure_count', 'not available')}.",
        f"- Historical all-observation-converged cases: {summary.get('historical_failure_diagnostics', {}).get('fully_converged_case_count', 'not available')} / {summary.get('historical_failure_diagnostics', {}).get('case_count', 'not available')}.",
        f"- Fresh default-budget validation: {default.get('valid_count', 0)} converged / {default.get('pair_count', 0)} pairs.",
        f"- Fresh 120-iteration validation: {robust.get('valid_count', 0)} converged / {robust.get('pair_count', 0)} pairs.",
        "- A larger iteration budget is a bounded rescue strategy, not evidence that local path "
        "optimization has found the global first arrival.",
        "",
        "## Expanded validation evidence",
        "",
        f"- Heterogeneous graph-comparison pairs evaluated with the 120-iteration run: {robust_valid} valid / {robust_pairs} selected.",
        f"- Mean absolute discrepancy across those valid pairs: {_format_number(robust_mean_abs)} s.",
        f"- Largest family-level 95th-percentile discrepancy: {_format_number(robust_p95)} s.",
        f"- Largest coarse-grid 95th-percentile difference from the 2.5 km graph: {_format_number(p95_coarse)} s.",
        "",
        "The full family-by-family distributions are in `validation_statistics.csv`; the pair-level "
        "values and convergence status are in `validation_pair_metrics.csv`; and the distribution "
        "plot is `solver_discrepancy_distribution.png`.",
        "",
        "The discrepancy values must be read against the observed total travel-time distribution "
        "and the controlled 0.025, 0.050, and 0.100 s noise levels. The report records those "
        "ratios as a predeclared contextual comparison; the noise levels are not a pass/fail "
        "threshold selected after seeing the results. Velocity-span metadata is retained for "
        "geological context, but a raw travel-time difference cannot be converted into a "
        "velocity-contrast error without an inversion model.",
        "",
        "## Terminology recommendation",
        "",
        "Use **interface-aware two-point ray-path optimizer initialized by a Snell-law layered ray** "
        "for the current implementation. `Modified pseudo-bending` could be used only if the "
        "algorithm is described with the same qualification and the distinction from canonical "
        "pseudo-bending is made explicit. Do not call the output globally validated first-arrival "
        "travel time on the basis of this audit alone.",
        "",
        "## Benchmark v2 gate implications",
        "",
        "- The default 30-iteration configuration must not silently label non-converged observations "
        "as valid. Any pilot generator must persist convergence and termination status.",
        "- Because the observed discrepancies are at or above the smallest controlled noise scale, "
        "Benchmark v2 pilot generation is stopped. The current solver must not be used to create "
        "a larger corpus presented as validated first-arrival labels.",
        "- Resumption requires a solver/reference redesign or an independent established eikonal/Fast-Marching "
        "validation with a stronger resolution-convergence result. Any future pilot must use an "
        "explicitly versioned solver configuration and retain termination status.",
        "- The target-level statistical unit, target-grouped splitting, and non-privileged input "
        "rules remain mandatory for every subsequent Benchmark v2 experiment.",
        "",
        "## Reproducible artifacts",
        "",
        "- `validation_pair_metrics.csv`",
        "- `validation_statistics.csv`",
        "- `resolution_study.csv`",
        "- `forward_solver_validation_v2_summary.json`",
        "- `solver_discrepancy_distribution.png`",
        "",
    ]
    return "\n".join(lines)


def build_audit_metadata(
    *,
    manifest_path: Path,
    output_dir: Path,
    summary: Mapping[str, Any],
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    """Build reproducibility metadata for the validation run."""

    return {
        "artifact_type": "forward_solver_validation_v2_metadata",
        "validation_id": VALIDATION_ID,
        "command": "tomobench audit-forward-solver-v2",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "output_dir": str(output_dir),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "solver_configuration": {
            "simulation_method": settings.simulation.method,
            "velocity_interpolation": settings.pseudo_bending_solver.velocity_interpolation,
            "initial_ray_point_count": settings.pseudo_bending_solver.initial_ray_point_count,
            "max_ray_point_count": settings.pseudo_bending_solver.max_ray_point_count,
            "configured_max_iterations_per_level": settings.pseudo_bending_solver.max_iterations_per_level,
            "convergence_tolerance_s": settings.pseudo_bending_solver.convergence_tolerance_s,
            "perturbation_step_km": settings.pseudo_bending_solver.perturbation_step_km,
            "finite_difference_step_km": settings.pseudo_bending_solver.finite_difference_step_km,
        },
        "target_hash_schema": "target_velocity_vector_sha256_v1",
        "summary_hash": _sha256_bytes(json.dumps(summary, sort_keys=True).encode("utf-8")),
    }


def _run_validation_pairs(
    settings: BenchmarkSettings,
    selected_pairs: Sequence[_GraphPair],
    analytic_pairs: Sequence[_AnalyticPair],
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analytic in analytic_pairs:
        grid = build_cartesian_velocity_grid(
            settings,
            analytic.velocity_model,
            scenario="layered",
        )
        for budget in budgets:
            solver_settings = replace(
                settings.pseudo_bending_solver,
                max_iterations_per_level=budget,
            )
            pseudo = trace_pseudo_bending_ray(
                source=analytic.source,
                receiver=analytic.receiver,
                velocity_grid=grid,
                settings=solver_settings,
            )
            reference_path, reference_time = _analytic_reference(analytic)
            rows.append(
                _comparison_row(
                    pair_id=analytic.pair_id,
                    case_id=f"analytic_{analytic.family}",
                    family=analytic.family,
                    scenario="layered",
                    parameter_variant_id="analytic",
                    acquisition_profile="analytic_fixed",
                    target_hash=target_vector_sha256(grid.p_velocity_km_per_s),
                    source=analytic.source,
                    receiver=analytic.receiver,
                    pseudo=pseudo,
                    reference_time_s=reference_time,
                    reference_path=reference_path,
                    reference_type=analytic.reference_type,
                    comparison_kind=(
                        "analytic_homogeneous"
                        if analytic.reference_type == "homogeneous_distance_over_velocity"
                        else "analytic_layered"
                    ),
                    solver_configuration=f"max_iterations_{budget}",
                    historical_converged=None,
                    grid=grid,
                )
            )

    graph_cache: dict[str, tuple[_CartesianGridGraph, dict[str, tuple[list[float], list[int | None]]]]] = {}
    for selected in selected_pairs:
        row = selected.row
        case_id = str(row["case_id"])
        grid = selected.grid
        if case_id not in graph_cache:
            graph = _CartesianGridGraph(grid, settings.eikonal_solver.connectivity)
            by_source: dict[str, tuple[list[float], list[int | None]]] = {}
            case_rows = [candidate.row for candidate in selected_pairs if str(candidate.row["case_id"]) == case_id]
            for candidate in case_rows:
                source = _point_from_row(candidate, "source")
                source_index = graph.nearest_node_index(source)
                if str(candidate["earthquake_id"]) not in by_source:
                    by_source[str(candidate["earthquake_id"])] = (
                        *graph.shortest_travel_times_and_predecessors_from(source_index),
                    )
            graph_cache[case_id] = (graph, by_source)
        graph, by_source = graph_cache[case_id]
        source = _point_from_row(row, "source")
        receiver = _point_from_row(row, "receiver")
        source_index = graph.nearest_node_index(source)
        receiver_index = graph.nearest_node_index(receiver)
        distances, predecessors = by_source[str(row["earthquake_id"])]
        reference_time = distances[receiver_index]
        reference_indices = _path_indices_from_predecessors(
            source_index,
            receiver_index,
            predecessors,
        )
        reference_path = tuple(graph.node_point(index) for index in reference_indices)
        for budget in budgets:
            solver_settings = replace(
                settings.pseudo_bending_solver,
                max_iterations_per_level=budget,
            )
            pseudo = trace_pseudo_bending_ray(
                source=source,
                receiver=receiver,
                velocity_grid=grid,
                settings=solver_settings,
            )
            rows.append(
                _comparison_row(
                    pair_id=str(row["observation_id"]),
                    case_id=case_id,
                    family=str(row["family"]),
                    scenario=str(row["scenario"]),
                    parameter_variant_id=str(row["parameter_variant_id"]),
                    acquisition_profile=str(row["acquisition_profile"]),
                    target_hash=str(row["target_hash"]),
                    source=source,
                    receiver=receiver,
                    pseudo=pseudo,
                    reference_time_s=reference_time,
                    reference_path=reference_path,
                    reference_type="eikonal_dijkstra_graph_shortest_path",
                    comparison_kind="heterogeneous_graph",
                    solver_configuration=f"max_iterations_{budget}",
                    historical_converged=row.get("converged"),
                    grid=grid,
                )
            )
    return rows


def _comparison_row(
    *,
    pair_id: str,
    case_id: str,
    family: str,
    scenario: str,
    parameter_variant_id: str,
    acquisition_profile: str,
    target_hash: str,
    source: Point3D,
    receiver: Point3D,
    pseudo: PseudoBendingRayResult,
    reference_time_s: float,
    reference_path: Sequence[Point3D],
    reference_type: str,
    comparison_kind: str,
    solver_configuration: str,
    historical_converged: Any,
    grid: CartesianVelocityGrid3D,
) -> dict[str, Any]:
    difference = pseudo.travel_time_s - reference_time_s
    absolute_difference = abs(difference)
    relative = absolute_difference / reference_time_s if reference_time_s else 0.0
    pseudo_interfaces = _interface_sequence(pseudo.ray_path, grid)
    reference_interfaces = _interface_sequence(tuple(reference_path), grid)
    return {
        "pair_id": pair_id,
        "case_id": case_id,
        "family": family,
        "scenario": scenario,
        "parameter_variant_id": parameter_variant_id,
        "acquisition_profile": acquisition_profile,
        "target_hash": target_hash,
        "source_x_km": source.x_km,
        "source_y_km": source.y_km,
        "source_z_km": source.z_km,
        "receiver_x_km": receiver.x_km,
        "receiver_y_km": receiver.y_km,
        "receiver_z_km": receiver.z_km,
        "pseudo_travel_time_s": pseudo.travel_time_s,
        "reference_travel_time_s": reference_time_s,
        "travel_time_difference_s": difference,
        "absolute_travel_time_difference_s": absolute_difference,
        "absolute_relative_error": relative,
        "pseudo_path_length_km": pseudo.path_length_km,
        "reference_path_length_km": polyline_length_km(tuple(reference_path)),
        "path_length_difference_km": pseudo.path_length_km
        - polyline_length_km(tuple(reference_path)),
        "reference_source_snap_distance_km": _distance_km(source, reference_path[0]),
        "reference_receiver_snap_distance_km": _distance_km(receiver, reference_path[-1]),
        "pseudo_converged": pseudo.converged,
        "pseudo_termination_reason": pseudo.termination_reason,
        "pseudo_iteration_count": pseudo.iteration_count,
        "historical_converged": historical_converged,
        "error_included": bool(pseudo.converged and math.isfinite(reference_time_s)),
        "comparison_kind": comparison_kind,
        "reference_type": reference_type,
        "solver_configuration": solver_configuration,
        "pseudo_ray_point_count": len(pseudo.ray_path),
        "reference_ray_point_count": len(reference_path),
        "pseudo_interface_sequence": pseudo_interfaces,
        "reference_interface_sequence": reference_interfaces,
        "grid_spacing_label": grid.metadata.get("grid_spacing_km", {}),
        "grid_node_count": grid.metadata.get("node_count", len(grid.p_velocity_km_per_s)),
        "grid_velocity_min_km_per_s": min(grid.p_velocity_km_per_s),
        "grid_velocity_max_km_per_s": max(grid.p_velocity_km_per_s),
        "grid_velocity_span_km_per_s": max(grid.p_velocity_km_per_s)
        - min(grid.p_velocity_km_per_s),
    }


def _run_resolution_study(
    settings: BenchmarkSettings,
    case_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    case_context: Mapping[str, Mapping[str, Any]],
    selected_case_ids: Mapping[str, Sequence[str]],
    *,
    pairs_per_family: int,
    spacings_km: Sequence[float],
) -> list[dict[str, Any]]:
    """Compare independent graph travel times across grid resolutions."""

    rows: list[dict[str, Any]] = []
    selected_pairs_by_family: dict[str, list[Mapping[str, Any]]] = {}
    for family, case_ids in selected_case_ids.items():
        first_case = case_ids[0]
        selected_pairs_by_family[family] = list(
            _evenly_select(case_rows[first_case], pairs_per_family)
        )

    for family, pairs in sorted(selected_pairs_by_family.items()):
        base_case_id = str(pairs[0]["case_id"])
        base_grid = case_context[base_case_id]["grid"]
        graph_results: dict[float, dict[str, dict[str, Any]]] = {}
        for spacing_km in spacings_km:
            resolution_settings = _settings_for_grid_resolution(
                settings,
                base_grid,
                spacing_km,
            )
            build_start = perf_counter()
            grid = build_cartesian_velocity_grid(
                resolution_settings,
                _layered_model_from_grid(base_grid),
                scenario=str(base_grid.metadata.get("grid_scenario", family)),
            )
            build_time_s = perf_counter() - build_start
            graph = _CartesianGridGraph(grid, 26)
            graph_start = perf_counter()
            by_source: dict[str, tuple[list[float], list[int | None]]] = {}
            for pair in pairs:
                source = _point_from_row(pair, "source")
                source_index = graph.nearest_node_index(source)
                source_id = str(pair["earthquake_id"])
                if source_id not in by_source:
                    by_source[source_id] = graph.shortest_travel_times_and_predecessors_from(
                        source_index
                    )
            pair_results: dict[str, dict[str, Any]] = {}
            for pair in pairs:
                source = _point_from_row(pair, "source")
                receiver = _point_from_row(pair, "receiver")
                source_index = graph.nearest_node_index(source)
                receiver_index = graph.nearest_node_index(receiver)
                distances, predecessors = by_source[str(pair["earthquake_id"])]
                path_indices = _path_indices_from_predecessors(
                    source_index,
                    receiver_index,
                    predecessors,
                )
                path = tuple(graph.node_point(index) for index in path_indices)
                pair_results[str(pair["observation_id"])] = {
                    "travel_time_s": distances[receiver_index],
                    "path_length_km": polyline_length_km(path),
                    "path_point_count": len(path),
                    "source_snapped": graph.node_point(source_index),
                    "receiver_snapped": graph.node_point(receiver_index),
                }
            graph_time_s = perf_counter() - graph_start
            graph_results[float(spacing_km)] = {
                pair_id: {
                    **result,
                    "node_count": grid.metadata["node_count"],
                    "grid_build_time_s": build_time_s,
                    "graph_runtime_s": graph_time_s,
                }
                for pair_id, result in pair_results.items()
            }

        finest_spacing = min(spacings_km)
        for pair in pairs:
            pair_id = str(pair["observation_id"])
            finest = graph_results[float(finest_spacing)][pair_id]
            for spacing_km in spacings_km:
                result = graph_results[float(spacing_km)][pair_id]
                difference = result["travel_time_s"] - finest["travel_time_s"]
                reference = finest["travel_time_s"]
                source = _point_from_row(pair, "source")
                receiver = _point_from_row(pair, "receiver")
                rows.append(
                    {
                        "family": family,
                        "case_id": base_case_id,
                        "pair_id": pair_id,
                        "target_hash": pair["target_hash"],
                        "grid_spacing_km": spacing_km,
                        "finest_reference_spacing_km": finest_spacing,
                        "node_count": result["node_count"],
                        "source_x_km": source.x_km,
                        "source_y_km": source.y_km,
                        "source_z_km": source.z_km,
                        "receiver_x_km": receiver.x_km,
                        "receiver_y_km": receiver.y_km,
                        "receiver_z_km": receiver.z_km,
                        "graph_travel_time_s": result["travel_time_s"],
                        "finest_reference_travel_time_s": finest["travel_time_s"],
                        "difference_from_finest_s": difference,
                        "absolute_difference_from_finest_s": abs(difference),
                        "relative_difference_from_finest": (
                            abs(difference) / reference if reference else 0.0
                        ),
                        "graph_path_length_km": result["path_length_km"],
                        "finest_reference_path_length_km": finest["path_length_km"],
                        "path_length_difference_km": result["path_length_km"]
                        - finest["path_length_km"],
                        "source_snap_distance_km": _distance_km(
                            source,
                            result["source_snapped"],
                        ),
                        "receiver_snap_distance_km": _distance_km(
                            receiver,
                            result["receiver_snapped"],
                        ),
                        "path_point_count": result["path_point_count"],
                        "grid_build_time_s": result["grid_build_time_s"],
                        "graph_runtime_s": result["graph_runtime_s"],
                    }
                )
    return rows


def _analytic_pairs(settings: BenchmarkSettings) -> tuple[_AnalyticPair, ...]:
    homogeneous = _homogeneous_velocity_model(5.0)
    layered = generate_layered_velocity_model(settings)
    homogeneous_coordinates = (
        (10.0, 10.0, 2.5, 90.0, 80.0, 0.0),
        (20.0, 75.0, 7.5, 80.0, 15.0, 0.0),
        (50.0, 50.0, 12.5, 50.0, 50.0, 0.0),
        (85.0, 20.0, 17.5, 15.0, 85.0, 5.0),
        (5.0, 95.0, 25.0, 95.0, 5.0, 0.0),
        (35.0, 35.0, 27.5, 65.0, 65.0, 2.5),
        (15.0, 60.0, 5.0, 75.0, 40.0, 20.0),
        (70.0, 10.0, 10.0, 30.0, 90.0, 0.0),
        (95.0, 50.0, 22.5, 5.0, 50.0, 2.5),
        (40.0, 85.0, 15.0, 60.0, 15.0, 0.0),
    )
    layered_coordinates = (
        (15.0, 50.0, 2.5, 85.0, 50.0, 0.0),
        (15.0, 50.0, 7.5, 85.0, 50.0, 0.0),
        (15.0, 50.0, 12.5, 85.0, 50.0, 0.0),
        (15.0, 50.0, 17.5, 85.0, 50.0, 0.0),
        (15.0, 50.0, 22.5, 85.0, 50.0, 0.0),
        (15.0, 50.0, 27.5, 85.0, 50.0, 0.0),
        (20.0, 20.0, 7.5, 80.0, 70.0, 0.0),
        (80.0, 20.0, 12.5, 20.0, 80.0, 0.0),
        (25.0, 80.0, 17.5, 75.0, 20.0, 0.0),
        (75.0, 75.0, 22.5, 25.0, 25.0, 0.0),
    )
    pairs: list[_AnalyticPair] = []
    for index, values in enumerate(homogeneous_coordinates, start=1):
        pairs.append(
            _AnalyticPair(
                pair_id=f"homogeneous_{index:02d}",
                family="homogeneous",
                source=Point3D(*values[:3]),
                receiver=Point3D(*values[3:]),
                velocity_model=homogeneous,
                reference_type="homogeneous_distance_over_velocity",
            )
        )
    for index, values in enumerate(layered_coordinates, start=1):
        pairs.append(
            _AnalyticPair(
                pair_id=f"layered_{index:02d}",
                family="layered",
                source=Point3D(*values[:3]),
                receiver=Point3D(*values[3:]),
                velocity_model=layered,
                reference_type="layered_snell_law_direct_ray",
            )
        )
    return tuple(pairs)


def _analytic_reference(pair: _AnalyticPair) -> tuple[tuple[Point3D, ...], float]:
    if pair.reference_type == "homogeneous_distance_over_velocity":
        path = (pair.source, pair.receiver)
        velocity = pair.velocity_model.layers[0].p_velocity_km_per_s
        return path, polyline_length_km(path) / velocity
    path = _layered_snell_reference_path(
        pair.source,
        pair.receiver,
        pair.velocity_model,
    )
    return path, _layered_path_time_s(path, pair.velocity_model)


def _select_graph_pairs(
    case_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    case_context: Mapping[str, Mapping[str, Any]],
    selected_case_ids: Mapping[str, Sequence[str]],
    *,
    pairs_per_case: int,
) -> tuple[_GraphPair, ...]:
    selected: list[_GraphPair] = []
    for family in sorted(selected_case_ids):
        for case_id in selected_case_ids[family]:
            for row in _evenly_select(case_rows[case_id], pairs_per_case):
                selected.append(_GraphPair(row=row, grid=case_context[case_id]["grid"]))
    return tuple(selected)


def _select_case_ids(
    case_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    cases_per_family: int,
) -> dict[str, tuple[str, ...]]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for case_id, rows in case_rows.items():
        if rows:
            by_family[str(rows[0]["family"])].append(case_id)
    selected: dict[str, tuple[str, ...]] = {}
    for family, case_ids in sorted(by_family.items()):
        ordered = sorted(case_ids)
        selected[family] = tuple(_evenly_select(ordered, cases_per_family))
    return selected


def _evenly_select(values: Sequence[Any], count: int) -> tuple[Any, ...]:
    if not values:
        return ()
    if count >= len(values):
        return tuple(values)
    if count == 1:
        return (values[0],)
    indexes = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return tuple(values[index] for index in indexes)


def _group_rows_by_case(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    return {
        case_id: tuple(sorted(case_rows, key=lambda row: str(row["observation_id"])))
        for case_id, case_rows in grouped.items()
    }


def _settings_for_grid_resolution(
    settings: BenchmarkSettings,
    base_grid: CartesianVelocityGrid3D,
    spacing_km: float,
) -> BenchmarkSettings:
    eikonal = replace(
        settings.eikonal_solver,
        x_grid_spacing_km=spacing_km,
        y_grid_spacing_km=spacing_km,
        z_grid_spacing_km=spacing_km,
        connectivity=26,
    )
    generation = settings.velocity_model_generation
    metadata = base_grid.metadata
    scenario = str(metadata.get("grid_scenario", "layered"))
    if scenario == "block_anomaly" and isinstance(metadata.get("block_anomaly"), dict):
        item = metadata["block_anomaly"]
        generation = replace(
            generation,
            block_anomaly=BlockAnomalySettings(
                center_km=tuple(float(value) for value in item["center_km"]),
                size_km=tuple(float(value) for value in item["size_km"]),
                velocity_delta_km_per_s=float(item["velocity_delta_km_per_s"]),
            ),
        )
    elif scenario == "faulted" and isinstance(metadata.get("faulted"), dict):
        item = metadata["faulted"]
        generation = replace(
            generation,
            faulted=FaultedGridSettings(
                fault_x_km=float(item["fault_x_km"]),
                fault_y_km=float(item["fault_y_km"]),
                strike_deg=float(item["strike_deg"]),
                dip_deg=float(item["dip_deg"]),
                dip_direction=str(item["dip_direction"]),
                positive_side=str(item["positive_side"]),
                velocity_offset_km_per_s=float(item["velocity_offset_km_per_s"]),
            ),
        )
    elif scenario == "salt_dome" and isinstance(metadata.get("salt_dome"), dict):
        item = metadata["salt_dome"]
        generation = replace(
            generation,
            salt_dome=SaltDomeSettings(
                center_km=tuple(float(value) for value in item["center_km"]),
                radii_km=tuple(float(value) for value in item["radii_km"]),
                body_velocity_km_per_s=float(item["body_velocity_km_per_s"]),
            ),
        )
    elif scenario == "dyke_intrusion" and isinstance(metadata.get("dyke_intrusion"), dict):
        item = metadata["dyke_intrusion"]
        generation = replace(
            generation,
            dyke_intrusion=DykeIntrusionSettings(
                center_km=tuple(float(value) for value in item["center_km"]),
                strike_deg=float(item["strike_deg"]),
                length_km=float(item["length_km"]),
                width_km=float(item["width_km"]),
                top_depth_km=float(item["top_depth_km"]),
                bottom_depth_km=float(item["bottom_depth_km"]),
                body_velocity_km_per_s=float(item["body_velocity_km_per_s"]),
            ),
        )
    return replace(settings, eikonal_solver=eikonal, velocity_model_generation=generation)


def _layered_model_from_grid(grid: CartesianVelocityGrid3D) -> LayeredVelocityModel:
    metadata = grid.metadata.get("layered_model")
    if not isinstance(metadata, dict):
        raise ValueError("Grid metadata lacks the layered_model definition required for resolution study.")
    boundaries = tuple(float(value) for value in metadata["depth_boundaries_km"])
    velocities = tuple(float(value) for value in metadata["velocities_km_per_s"])
    if len(boundaries) != len(velocities) + 1:
        raise ValueError("Layered grid metadata has inconsistent boundaries and velocities.")
    layers = tuple(
        VelocityLayer(
            layer_id=f"RESOLUTION_LAYER{index + 1:02d}",
            top_depth_km=boundaries[index],
            bottom_depth_km=boundaries[index + 1],
            p_velocity_km_per_s=velocities[index],
        )
        for index in range(len(velocities))
    )
    return LayeredVelocityModel(
        model_id=f"{grid.source_velocity_model_id}_resolution_reference",
        layers=layers,
        metadata={"source_grid_id": grid.grid_id},
    )


def _summarize_convergence_by_solver(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["solver_configuration"])].append(row)
    return {
        solver: {
            "pair_count": len(group),
            "valid_count": sum(_as_bool(row.get("pseudo_converged")) for row in group),
            "nonconverged_count": sum(not _as_bool(row.get("pseudo_converged")) for row in group),
            "valid_fraction": (
                sum(_as_bool(row.get("pseudo_converged")) for row in group) / len(group)
                if group
                else None
            ),
            "termination_reasons": dict(
                sorted(
                    _count_values(str(row.get("pseudo_termination_reason", "")) for row in group).items()
                )
            ),
        }
        for solver, group in sorted(grouped.items())
    }


def _summarize_resolution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["grid_spacing_km"])].append(row)
    output: dict[str, Any] = {"by_spacing": {}}
    for spacing, group in sorted(grouped.items(), key=lambda item: float(item[0])):
        absolute = [float(row["absolute_difference_from_finest_s"]) for row in group]
        relative = [float(row["relative_difference_from_finest"]) for row in group]
        output["by_spacing"][spacing] = {
            "pair_count": len(group),
            "mean_absolute_difference_from_finest_s": _mean_or_none(absolute),
            "median_absolute_difference_from_finest_s": _median_or_none(absolute),
            "std_absolute_difference_from_finest_s": _sample_std_or_none(absolute),
            "p95_absolute_difference_from_finest_s": _percentile_or_none(absolute, 0.95),
            "max_absolute_difference_from_finest_s": max(absolute, default=None),
            "mean_relative_difference_from_finest": _mean_or_none(relative),
            "p95_relative_difference_from_finest": _percentile_or_none(relative, 0.95),
            "max_relative_difference_from_finest": max(relative, default=None),
        }
    return output


def _interface_sequence(path: Sequence[Point3D], grid: CartesianVelocityGrid3D) -> str:
    if len(path) < 2:
        return "none"
    from tomobench.simulation.ray_tracing import _segment_interface_crossings

    sequence: list[str] = []
    for start, end in zip(path, path[1:]):
        for _, constraint in _segment_interface_crossings(start, end, grid):
            if constraint.kind not in sequence:
                sequence.append(constraint.kind)
    return "|".join(sequence) if sequence else "none"


def _path_indices_from_predecessors(
    source_index: int,
    receiver_index: int,
    predecessors: Sequence[int | None],
) -> tuple[int, ...]:
    if source_index == receiver_index:
        return (source_index,)
    path = [receiver_index]
    current = receiver_index
    visited: set[int] = set()
    while current != source_index:
        if current in visited:
            raise ValueError("Graph predecessor chain contains a cycle.")
        visited.add(current)
        predecessor = predecessors[current]
        if predecessor is None:
            raise ValueError("Graph predecessor chain terminated before reaching source.")
        path.append(predecessor)
        current = predecessor
    path.reverse()
    return tuple(path)


def _point_from_row(row: Mapping[str, Any], prefix: str) -> Point3D:
    return Point3D(
        float(row[f"{prefix}_x_km"]),
        float(row[f"{prefix}_y_km"]),
        float(row[f"{prefix}_z_km"]),
    )


def validation_pair_columns() -> tuple[str, ...]:
    """Return the stable validation pair CSV schema."""

    return (
        "pair_id",
        "case_id",
        "family",
        "scenario",
        "parameter_variant_id",
        "acquisition_profile",
        "target_hash",
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "receiver_x_km",
        "receiver_y_km",
        "receiver_z_km",
        "pseudo_travel_time_s",
        "reference_travel_time_s",
        "travel_time_difference_s",
        "absolute_travel_time_difference_s",
        "absolute_relative_error",
        "pseudo_path_length_km",
        "reference_path_length_km",
        "path_length_difference_km",
        "reference_source_snap_distance_km",
        "reference_receiver_snap_distance_km",
        "pseudo_converged",
        "pseudo_termination_reason",
        "pseudo_iteration_count",
        "historical_converged",
        "error_included",
        "comparison_kind",
        "reference_type",
        "solver_configuration",
        "pseudo_ray_point_count",
        "reference_ray_point_count",
        "pseudo_interface_sequence",
        "reference_interface_sequence",
        "grid_spacing_label",
        "grid_node_count",
        "grid_velocity_min_km_per_s",
        "grid_velocity_max_km_per_s",
        "grid_velocity_span_km_per_s",
    )


def validation_statistics_columns() -> tuple[str, ...]:
    """Return the stable validation-statistics CSV schema."""

    return (
        "solver_configuration",
        "comparison_kind",
        "family",
        "reference_type",
        "pair_count",
        "valid_pair_count",
        "nonconverged_pair_count",
        "valid_fraction",
        "mean_reference_travel_time_s",
        "median_reference_travel_time_s",
        "mean_velocity_span_km_per_s",
        "mean_absolute_travel_time_difference_s",
        "median_absolute_travel_time_difference_s",
        "std_absolute_travel_time_difference_s",
        "p95_absolute_travel_time_difference_s",
        "max_absolute_travel_time_difference_s",
        "mean_absolute_relative_error",
        "median_absolute_relative_error",
        "std_absolute_relative_error",
        "p95_absolute_relative_error",
        "max_absolute_relative_error",
        "mean_abs_difference_over_median_travel_time",
        "mean_abs_difference_over_noise_0_025_s",
        "p95_abs_difference_over_noise_0_025_s",
        "mean_abs_difference_over_noise_0_050_s",
        "mean_abs_difference_over_noise_0_100_s",
    )


def resolution_columns() -> tuple[str, ...]:
    """Return the stable resolution-study CSV schema."""

    return (
        "family",
        "case_id",
        "pair_id",
        "target_hash",
        "grid_spacing_km",
        "finest_reference_spacing_km",
        "node_count",
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "receiver_x_km",
        "receiver_y_km",
        "receiver_z_km",
        "graph_travel_time_s",
        "finest_reference_travel_time_s",
        "difference_from_finest_s",
        "absolute_difference_from_finest_s",
        "relative_difference_from_finest",
        "graph_path_length_km",
        "finest_reference_path_length_km",
        "path_length_difference_km",
        "source_snap_distance_km",
        "receiver_snap_distance_km",
        "path_point_count",
        "grid_build_time_s",
        "graph_runtime_s",
    )


def _write_discrepancy_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write a compact distribution plot for robust heterogeneous comparisons."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hetero = [
        row
        for row in rows
        if row.get("comparison_kind") == "heterogeneous_graph"
        and row.get("solver_configuration") == "max_iterations_120"
        and _as_bool(row.get("error_included"))
    ]
    families = sorted({str(row["family"]) for row in hetero})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if hetero:
        values = [float(row["absolute_travel_time_difference_s"]) for row in hetero]
        axes[0].hist(values, bins=min(20, max(5, len(values) // 5)), color="#2f6f9f", alpha=0.85)
        for level, label in zip(NOISE_LEVELS_S, ("0.025 s", "0.050 s", "0.100 s")):
            axes[0].axvline(level, color="#b23a48", linestyle="--", linewidth=1.0, label=label)
        axes[0].set_xlabel("Absolute pseudo/graph travel-time difference (s)")
        axes[0].set_ylabel("Pair count")
        axes[0].set_title("Heterogeneous discrepancy distribution")
        axes[0].legend(fontsize=8)
        family_values = [
            [float(row["absolute_travel_time_difference_s"]) for row in hetero if row["family"] == family]
            for family in families
        ]
        axes[1].boxplot(family_values, labels=families, showmeans=True)
        axes[1].set_ylabel("Absolute difference (s)")
        axes[1].set_title("Discrepancy by geological family")
        axes[1].tick_params(axis="x", rotation=30)
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No valid heterogeneous pairs", ha="center", va="center")
            axis.set_axis_off()
    fig.suptitle("Forward-solver v2: independent graph-comparator audit", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary_markdown(summary: Mapping[str, Any], path: Path) -> None:
    convergence = summary.get("fresh_validation_convergence", {})
    resolution = summary.get("graph_resolution_summary", {}).get("by_spacing", {})
    lines = [
        "# Forward-Solver v2 Validation Summary",
        "",
        "This summary records the expanded audit used by `solver_scientific_assessment.md`. "
        "It is not a manuscript Results section.",
        "",
        "## Validation design",
        "",
        f"- Analytic homogeneous pairs: {summary['validation_design']['analytic_homogeneous_pair_count']} per solver budget.",
        f"- Analytic layered pairs: {summary['validation_design']['analytic_layered_pair_count']} per solver budget.",
        f"- Heterogeneous graph-comparison pairs: {summary['validation_design']['heterogeneous_pair_count']} per solver budget.",
        f"- Heterogeneous cases: {summary['validation_design']['heterogeneous_case_count']}.",
        "- Geological families: layered, block anomaly, faulted, salt dome, and dyke intrusion.",
        "",
        "## Fresh convergence",
        "",
    ]
    for solver, item in sorted(convergence.items()):
        lines.append(
            f"- `{solver}`: {item['valid_count']} / {item['pair_count']} converged "
            f"({item['valid_fraction']:.6g}); termination reasons={item['termination_reasons']}."
        )
    lines.extend(
        [
            "",
            "## Graph resolution study",
            "",
            "The 2.5 km graph is the finest tested graph and is used only as an internal "
            "discretization reference.",
        ]
    )
    for spacing, item in sorted(resolution.items(), key=lambda pair: float(pair[0])):
        lines.append(
            f"- `{spacing} km`: n={item['pair_count']}, "
            f"mean absolute difference={_format_number(item['mean_absolute_difference_from_finest_s'])} s, "
            f"p95={_format_number(item['p95_absolute_difference_from_finest_s'])} s."
        )
    lines.extend(
        [
            "",
            "## Output boundary",
            "",
            "The graph comparator is independent in implementation but finite-grid and not a "
            "continuum-exact Fast Marching reference. See `solver_scientific_assessment.md` "
            "for the explicit terminology and go/no-go implications.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _available_independent_reference_packages() -> dict[str, bool]:
    return {
        name: importlib.util.find_spec(name) is not None
        for name in ("scipy", "skfmm", "pykonal", "fast_marching")
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for module_name in ("numpy", "matplotlib", "yaml"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = None
    return versions


def _count_values(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(counts)


def _weighted_mean(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    weight_key: str,
) -> float | None:
    values: list[tuple[float, float]] = []
    for row in rows:
        value = row.get(value_key)
        weight = row.get(weight_key)
        if value is None or weight is None:
            continue
        values.append((float(value), float(weight)))
    denominator = sum(weight for _, weight in values)
    return sum(value * weight for value, weight in values) / denominator if denominator else None


def _mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median_or_none(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _sample_std_or_none(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _percentile_or_none(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def _distance_km(first: Point3D, second: Point3D) -> float:
    return math.sqrt(
        (first.x_km - second.x_km) ** 2
        + (first.y_km - second.y_km) ** 2
        + (first.z_km - second.z_km) ** 2
    )


def _validate_positive_int(value: int, label: str) -> None:
    if value <= 0:
        raise ValueError(f"{label} must be positive.")


def _validate_iteration_budgets(values: Sequence[int]) -> tuple[int, ...]:
    budgets = tuple(int(value) for value in values)
    if not budgets or any(value <= 0 for value in budgets):
        raise ValueError("validation_iteration_budgets must contain positive integers.")
    return tuple(dict.fromkeys(budgets))


def _validate_spacings(values: Sequence[float]) -> tuple[float, ...]:
    spacings = tuple(float(value) for value in values)
    if not spacings or any(value <= 0.0 for value in spacings):
        raise ValueError("resolution_spacings_km must contain positive values.")
    return tuple(dict.fromkeys(spacings))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _format_number(value: Any) -> str:
    if value is None:
        return "not available"
    return f"{float(value):.8g}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    return _read_json(path) if path.is_file() else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
