"""Independent validation and corpus convergence audit for the pseudo-bending solver."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.evaluation.pseudo_bending_paper_benchmark import (
    _reference_result,
    benchmark_cases,
)
from tomobench.generation.velocity_grids import build_cartesian_velocity_grid
from tomobench.generation.velocity_models import generate_layered_velocity_model
from tomobench.simulation.ray_tracing import trace_pseudo_bending_ray
from tomobench.simulation.travel_times import PSEUDO_BENDING_3D_METHOD
from tomobench.utils.paths import get_repo_root


VALIDATION_ID = "forward_solver_validation_v1"
DEFAULT_MANIFEST_PATH = Path(
    "outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/generated/submission_readiness/forward_solver_validation_v1")


@dataclass(frozen=True)
class ForwardSolverValidationOutputs:
    """Files written by the forward-solver validation workflow."""

    validation_case_metrics_csv: Path
    corpus_convergence_case_summary_csv: Path
    corpus_solver_failure_records_csv: Path
    summary_json: Path
    summary_md: Path
    audit_metadata_json: Path


def run_forward_solver_validation(
    settings: BenchmarkSettings | None = None,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> ForwardSolverValidationOutputs:
    """Validate analytic cases and audit convergence across the generated corpus."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    resolved_manifest = _resolve_path(manifest_path, repo_root)
    resolved_output = _resolve_path(output_dir, repo_root)
    resolved_output.mkdir(parents=True, exist_ok=True)

    solver_settings = replace(
        loaded_settings,
        simulation=replace(loaded_settings.simulation, method=PSEUDO_BENDING_3D_METHOD),
    )
    validation_rows = _run_validation_cases(solver_settings)
    manifest = _read_json(resolved_manifest)
    convergence_rows, failure_rows = _audit_corpus_convergence(manifest, resolved_manifest, repo_root)

    validation_csv = resolved_output / "validation_case_metrics.csv"
    convergence_csv = resolved_output / "corpus_convergence_case_summary.csv"
    failures_csv = resolved_output / "corpus_solver_failure_records.csv"
    summary_json = resolved_output / "forward_solver_validation_summary.json"
    summary_md = resolved_output / "forward_solver_validation_summary.md"
    metadata_json = resolved_output / "audit_metadata.json"

    _write_csv(validation_rows, validation_csv, _validation_columns())
    _write_csv(convergence_rows, convergence_csv, _convergence_columns())
    _write_csv(failure_rows, failures_csv, _failure_columns())

    summary = _build_summary(
        validation_rows=validation_rows,
        convergence_rows=convergence_rows,
        failure_rows=failure_rows,
        manifest=manifest,
        manifest_path=resolved_manifest,
        repo_root=repo_root,
        settings=solver_settings,
    )
    _write_json(summary, summary_json)
    _write_summary_markdown(summary, summary_md)
    _write_json(
        _build_metadata(
            manifest_path=resolved_manifest,
            output_dir=resolved_output,
            summary=summary,
            settings=solver_settings,
        ),
        metadata_json,
    )
    return ForwardSolverValidationOutputs(
        validation_case_metrics_csv=validation_csv,
        corpus_convergence_case_summary_csv=convergence_csv,
        corpus_solver_failure_records_csv=failures_csv,
        summary_json=summary_json,
        summary_md=summary_md,
        audit_metadata_json=metadata_json,
    )


def _run_validation_cases(settings: BenchmarkSettings) -> list[dict[str, object]]:
    """Run analytic and independent graph-first-arrival comparisons."""
    default_model = generate_layered_velocity_model(settings)
    rows: list[dict[str, object]] = []
    for case in benchmark_cases(settings, default_model):
        velocity_model = case.velocity_model or default_model
        grid = build_cartesian_velocity_grid(settings, velocity_model, scenario=case.scenario)
        pseudo = trace_pseudo_bending_ray(
            source=case.source,
            receiver=case.receiver,
            velocity_grid=grid,
            settings=settings.pseudo_bending_solver,
        )
        reference = _reference_result(settings, velocity_model, grid, case)
        difference_s = pseudo.travel_time_s - reference.travel_time_s
        relative_error = (
            0.0
            if reference.travel_time_s == 0.0
            else abs(difference_s) / reference.travel_time_s
        )
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "scenario": case.scenario,
                "comparison_kind": case.comparison_kind,
                "reference_method": case.reference_method,
                "reference_scope": _reference_scope(case.reference_method),
                "pseudo_bending_travel_time_s": pseudo.travel_time_s,
                "reference_travel_time_s": reference.travel_time_s,
                "travel_time_difference_s": difference_s,
                "absolute_travel_time_difference_s": abs(difference_s),
                "relative_error": relative_error,
                "pseudo_bending_path_length_km": pseudo.path_length_km,
                "reference_path_length_km": reference.path_length_km,
                "pseudo_bending_converged": pseudo.converged,
                "reference_converged": reference.converged,
                "pseudo_bending_iteration_count": pseudo.iteration_count,
                "reference_iteration_count": reference.iteration_count,
                "pseudo_bending_ray_point_count": len(pseudo.ray_path),
                "reference_ray_point_count": len(reference.ray_path),
                "grid_spacing_km": settings.eikonal_solver.grid_spacing_label,
                "connectivity": settings.eikonal_solver.connectivity,
                "node_count": grid.metadata["node_count"],
            }
        )
    return rows


def _reference_scope(reference_method: str) -> str:
    if reference_method == "homogeneous_distance_over_velocity":
        return "analytic homogeneous straight ray"
    if reference_method == "layered_snell_law_direct_ray":
        return "analytic direct Snell-law ray; excludes head and turning branches"
    return "independent graph shortest-path comparator on the same Cartesian grid"


def _audit_corpus_convergence(
    manifest: dict[str, Any],
    manifest_path: Path,
    repo_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Summarize convergence and record every non-converged observation."""
    convergence_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    for item in manifest.get("items", []):
        case_id = str(item.get("case_id", ""))
        sidecar_value = item.get("ray_path_sidecar_path")
        sidecar_path = _resolve_manifest_path(sidecar_value, manifest_path, repo_root)
        records: list[dict[str, Any]] = []
        if sidecar_path.is_file():
            with sidecar_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        records.append(json.loads(line))

        expected_count = _as_int(item.get("input_observation_count"), len(records))
        converged_count = sum(record.get("converged") is True for record in records)
        nonconverged_records = [record for record in records if record.get("converged") is False]
        missing_status_count = sum("converged" not in record for record in records)
        iterations = [
            int(record["iteration_count"])
            for record in records
            if isinstance(record.get("iteration_count"), (int, float))
        ]
        for record in nonconverged_records:
            failure_rows.append(
                {
                    "case_id": case_id,
                    "observation_id": record.get("observation_id", ""),
                    "earthquake_id": record.get("earthquake_id", ""),
                    "station_id": record.get("station_id", ""),
                    "converged": record.get("converged"),
                    "iteration_count": record.get("iteration_count", ""),
                    "travel_time_s": record.get("travel_time_s", ""),
                    "sidecar_path": _relative_or_absolute(sidecar_path, repo_root),
                }
            )
        convergence_rows.append(
            {
                "case_id": case_id,
                "family": item.get("family", ""),
                "parameter_variant_id": item.get("parameter_variant_id", ""),
                "geometry_profile_id": item.get("geometry_profile_id", ""),
                "expected_observation_count": expected_count,
                "sidecar_record_count": len(records),
                "missing_record_count": max(expected_count - len(records), 0),
                "converged_count": converged_count,
                "nonconverged_count": len(nonconverged_records),
                "missing_convergence_status_count": missing_status_count,
                "convergence_fraction": (
                    converged_count / expected_count if expected_count else None
                ),
                "mean_iteration_count": mean(iterations) if iterations else None,
                "max_iteration_count": max(iterations) if iterations else None,
                "case_fully_converged": (
                    len(records) == expected_count
                    and converged_count == expected_count
                    and missing_status_count == 0
                ),
                "sidecar_exists": sidecar_path.is_file(),
                "sidecar_path": _relative_or_absolute(sidecar_path, repo_root),
            }
        )
    return convergence_rows, failure_rows


def _build_summary(
    validation_rows: list[dict[str, object]],
    convergence_rows: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
    manifest: dict[str, Any],
    manifest_path: Path,
    repo_root: Path,
    settings: BenchmarkSettings,
) -> dict[str, object]:
    analytic_rows = [row for row in validation_rows if row["comparison_kind"] == "analytic_reference"]
    heterogeneous_rows = [
        row for row in validation_rows if row["comparison_kind"] != "analytic_reference"
    ]
    fully_converged_cases = sum(row["case_fully_converged"] is True for row in convergence_rows)
    total_expected = sum(int(row["expected_observation_count"]) for row in convergence_rows)
    total_records = sum(int(row["sidecar_record_count"]) for row in convergence_rows)
    total_converged = sum(int(row["converged_count"]) for row in convergence_rows)
    return {
        "artifact_type": "forward_solver_validation_summary",
        "validation_id": VALIDATION_ID,
        "manifest_path": _relative_or_absolute(manifest_path, repo_root),
        "manifest_case_count": len(manifest.get("items", [])),
        "validation_case_count": len(validation_rows),
        "analytic_validation_case_count": len(analytic_rows),
        "heterogeneous_comparison_case_count": len(heterogeneous_rows),
        "analytic_validation": {
            "cases": analytic_rows,
            "max_absolute_travel_time_difference_s": max(
                (float(row["absolute_travel_time_difference_s"]) for row in analytic_rows),
                default=0.0,
            ),
            "all_within_1e-6_s": all(
                float(row["absolute_travel_time_difference_s"]) <= 1.0e-6
                for row in analytic_rows
            ),
        },
        "heterogeneous_comparison": {
            "cases": heterogeneous_rows,
            "mean_absolute_travel_time_difference_s": (
                mean(float(row["absolute_travel_time_difference_s"]) for row in heterogeneous_rows)
                if heterogeneous_rows
                else None
            ),
            "interpretation": (
                "The graph comparator is an independent first-arrival prototype on the same "
                "grid, not a continuum-exact Fast Marching reference. Differences quantify "
                "bounded method disagreement and grid/discretization effects."
            ),
        },
        "corpus_convergence": {
            "expected_observation_count": total_expected,
            "sidecar_record_count": total_records,
            "converged_observation_count": total_converged,
            "nonconverged_observation_count": len(failure_rows),
            "missing_observation_count": max(total_expected - total_records, 0),
            "convergence_fraction": total_converged / total_expected if total_expected else None,
            "fully_converged_case_count": fully_converged_cases,
            "cases_with_nonconvergence": [
                row["case_id"] for row in convergence_rows if row["nonconverged_count"]
            ],
            "cases_with_missing_records": [
                row["case_id"] for row in convergence_rows if row["missing_record_count"]
            ],
        },
        "solver_terminology": {
            "implemented_method": "interface-aware two-point path optimization",
            "initialization": "Snell-law ray-parameter solution through horizontal layered background",
            "heterogeneous_extension": "explicit geological-interface crossing waypoints constrained to slide on their surfaces",
            "recommended_manuscript_wording": (
                "interface-aware two-point ray-path optimizer initialized by a Snell-law layered ray"
            ),
            "caution": (
                "The implementation should not be described as a canonical continuous-grid "
                "pseudo-bending or globally validated Fast Marching solver without additional evidence."
            ),
        },
        "solver_configuration": {
            "simulation_method": settings.simulation.method,
            "velocity_interpolation": settings.pseudo_bending_solver.velocity_interpolation,
            "initial_ray_point_count": settings.pseudo_bending_solver.initial_ray_point_count,
            "max_ray_point_count": settings.pseudo_bending_solver.max_ray_point_count,
            "max_iterations_per_level": settings.pseudo_bending_solver.max_iterations_per_level,
            "convergence_tolerance_s": settings.pseudo_bending_solver.convergence_tolerance_s,
            "perturbation_step_km": settings.pseudo_bending_solver.perturbation_step_km,
            "finite_difference_step_km": settings.pseudo_bending_solver.finite_difference_step_km,
            "boundary_handling": settings.pseudo_bending_solver.boundary_handling,
        },
    }


def _build_metadata(
    manifest_path: Path,
    output_dir: Path,
    summary: dict[str, object],
    settings: BenchmarkSettings,
) -> dict[str, object]:
    return {
        "artifact_type": "forward_solver_validation_metadata",
        "validation_id": VALIDATION_ID,
        "command": "tomobench audit-forward-solver",
        "manifest_sha256": _file_sha256(manifest_path),
        "output_dir": str(output_dir),
        "python_version": sys.version,
        "platform": platform.platform(),
        "solver_configuration": summary["solver_configuration"],
        "validation_reference_boundary": summary["heterogeneous_comparison"]["interpretation"],
    }


def _write_summary_markdown(summary: dict[str, object], path: Path) -> None:
    analytic = summary["analytic_validation"]
    heterogeneous = summary["heterogeneous_comparison"]
    convergence = summary["corpus_convergence"]
    lines = [
        "# Forward-Solver Validation and Convergence Audit",
        "",
        "## Purpose",
        "",
        "This audit validates the configured pseudo-bending backend against bounded analytic "
        "cases, compares it with the repository's independent graph-based first-arrival "
        "prototype on representative heterogeneous cases, and counts convergence outcomes "
        "for every generated corpus observation.",
        "",
        "## Terminology",
        "",
        "The implementation is an interface-aware two-point path optimizer. It initializes a "
        "Snell-law layered path, inserts explicit geological-interface crossing waypoints, and "
        "optimizes those waypoints by local travel-time reductions. The evidence here supports "
        "that description; it does not establish equivalence to a canonical continuous-grid "
        "pseudo-bending or Fast Marching implementation.",
        "",
        "## Analytic validation",
        "",
        f"- Cases: {summary['analytic_validation_case_count']}.",
        f"- Maximum absolute travel-time difference: {analytic['max_absolute_travel_time_difference_s']:.12g} s.",
        f"- All analytic cases within 1e-6 s: {analytic['all_within_1e-6_s']}.",
        "- The homogeneous case uses distance divided by constant velocity; the layered case "
        "uses the direct Snell-law ray-parameter solution and excludes head-wave and turning-ray branches.",
        "",
        "## Heterogeneous comparison",
        "",
        f"- Representative cases: {summary['heterogeneous_comparison_case_count']}.",
        f"- Mean absolute difference from the independent graph comparator: {heterogeneous['mean_absolute_travel_time_difference_s']:.12g} s.",
        "- This comparator is a separate graph shortest-path implementation on the same Cartesian "
        "grid. It is useful as an implementation cross-check, but it is not a continuum-exact reference.",
        "",
        "## Corpus convergence",
        "",
        f"- Expected observations: {convergence['expected_observation_count']}.",
        f"- Sidecar records: {convergence['sidecar_record_count']}.",
        f"- Converged observations: {convergence['converged_observation_count']}.",
        f"- Non-converged observations: {convergence['nonconverged_observation_count']}.",
        f"- Fully converged cases: {convergence['fully_converged_case_count']} / {summary['manifest_case_count']}.",
        "",
        "## Interpretation boundary",
        "",
        "The analytic tests support the basic homogeneous and horizontally layered behavior. "
        "The heterogeneous comparisons quantify disagreement with an independent discretized "
        "first-arrival prototype, not global first-arrival correctness. The corpus convergence "
        "audit should be cited alongside any solver-based result; convergence does not by itself "
        "prove path optimality.",
        "",
        "## Generated artifacts",
        "",
        "- validation_case_metrics.csv",
        "- corpus_convergence_case_summary.csv",
        "- corpus_solver_failure_records.csv",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _validation_columns() -> tuple[str, ...]:
    return (
        "case_id",
        "family",
        "scenario",
        "comparison_kind",
        "reference_method",
        "reference_scope",
        "pseudo_bending_travel_time_s",
        "reference_travel_time_s",
        "travel_time_difference_s",
        "absolute_travel_time_difference_s",
        "relative_error",
        "pseudo_bending_path_length_km",
        "reference_path_length_km",
        "pseudo_bending_converged",
        "reference_converged",
        "pseudo_bending_iteration_count",
        "reference_iteration_count",
        "pseudo_bending_ray_point_count",
        "reference_ray_point_count",
        "grid_spacing_km",
        "connectivity",
        "node_count",
    )


def _convergence_columns() -> tuple[str, ...]:
    return (
        "case_id",
        "family",
        "parameter_variant_id",
        "geometry_profile_id",
        "expected_observation_count",
        "sidecar_record_count",
        "missing_record_count",
        "converged_count",
        "nonconverged_count",
        "missing_convergence_status_count",
        "convergence_fraction",
        "mean_iteration_count",
        "max_iteration_count",
        "case_fully_converged",
        "sidecar_exists",
        "sidecar_path",
    )


def _failure_columns() -> tuple[str, ...]:
    return (
        "case_id",
        "observation_id",
        "earthquake_id",
        "station_id",
        "converged",
        "iteration_count",
        "travel_time_s",
        "sidecar_path",
    )


def _write_csv(rows: list[dict[str, object]], path: Path, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _resolve_manifest_path(value: Any, manifest_path: Path, repo_root: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    candidate = repo_root / path
    if candidate.is_file():
        return candidate
    return manifest_path.parent / path


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback

