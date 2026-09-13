"""Failure diagnosis and bounded retry experiments for the v1 forward solver.

This module inspects every non-converged observation in the existing corpus.  It
does not rewrite the historical sidecars.  Alternative settings are evaluated on
the failed observations only and are reported as diagnostics until an explicit
Benchmark v2 solver configuration is frozen.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.datasets.target_identity import target_vector_sha256
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import CartesianVelocityGrid3D
from tomobench.simulation.ray_tracing import (
    trace_pseudo_bending_ray,
    trace_pseudo_bending_ray_with_restarts,
)
from tomobench.utils.paths import get_repo_root


DEFAULT_MANIFEST_PATH = Path(
    "outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/forward_solver_validation_v2"
)
DEFAULT_FAILURE_INPUT_PATH = Path(
    "outputs/generated/submission_readiness/forward_solver_validation_v1/"
    "corpus_solver_failure_records.csv"
)


@dataclass(frozen=True)
class FailureDiagnosticOutputs:
    """Files produced by the failure-diagnosis workflow."""

    output_dir: Path
    observation_diagnostics_csv: Path
    failure_diagnostics_csv: Path
    failure_summary_csv: Path
    strategy_results_csv: Path
    strategy_summary_csv: Path
    summary_json: Path


def run_forward_solver_failure_diagnostics(
    *,
    settings: BenchmarkSettings | None = None,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    failure_input_path: Path = DEFAULT_FAILURE_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> FailureDiagnosticOutputs:
    """Diagnose all historical failures and run bounded alternative strategies."""

    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    resolved_manifest = _resolve(manifest_path, repo_root)
    resolved_failure_input = _resolve(failure_input_path, repo_root)
    resolved_output = _resolve(output_dir, repo_root)
    manifest = _read_json(resolved_manifest)
    all_rows, case_context = _load_observation_diagnostics(
        manifest,
        resolved_manifest,
        repo_root,
        loaded_settings,
    )
    failures = [row for row in all_rows if not _as_bool(row["converged"])]
    historical_failure_ids = _read_historical_failure_ids(resolved_failure_input)
    current_failure_ids = {str(row["observation_id"]) for row in failures}
    if historical_failure_ids != current_failure_ids:
        raise ValueError(
            "Failure diagnostics do not match the historical failure artifact; "
            f"missing={sorted(historical_failure_ids - current_failure_ids)[:5]}, "
            f"extra={sorted(current_failure_ids - historical_failure_ids)[:5]}"
        )

    failure_summary = summarize_failure_rates(all_rows)
    strategy_results, strategy_summary = run_failure_strategy_experiments(
        failures,
        case_context,
        loaded_settings,
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    observation_path = resolved_output / "corpus_observation_diagnostics.csv"
    failure_path = resolved_output / "solver_failure_diagnostics.csv"
    summary_path = resolved_output / "failure_rate_summary.csv"
    strategy_results_path = resolved_output / "failure_strategy_results.csv"
    strategy_summary_path = resolved_output / "failure_strategy_summary.csv"
    json_path = resolved_output / "failure_diagnostics_summary.json"
    _write_csv(observation_path, all_rows)
    _write_csv(failure_path, failures)
    _write_csv(summary_path, failure_summary)
    _write_csv(strategy_results_path, strategy_results)
    _write_csv(strategy_summary_path, strategy_summary)
    payload = {
        "artifact_type": "forward_solver_failure_diagnostics",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "manifest_path": _relative(resolved_manifest, repo_root),
        "historical_failure_input": _relative(resolved_failure_input, repo_root),
        "observation_count": len(all_rows),
        "failure_count": len(failures),
        "failure_fraction": len(failures) / len(all_rows) if all_rows else None,
        "fully_converged_case_count": sum(
            int(row["nonconverged_count"]) == 0 for row in _case_convergence(all_rows)
        ),
        "case_count": len(case_context),
        "failure_dimensions": sorted({row["dimension"] for row in failure_summary}),
        "strategy_summary": strategy_summary,
        "termination_reason_definition": (
            "Historical sidecars predate explicit termination_reason. A failure at the "
            "configured iteration ceiling is classified as max_iterations_reached; all "
            "other missing reasons are marked historical_reason_unknown. New retry results "
            "carry the solver's explicit termination reason."
        ),
    }
    _write_json(json_path, payload)
    return FailureDiagnosticOutputs(
        output_dir=resolved_output,
        observation_diagnostics_csv=observation_path,
        failure_diagnostics_csv=failure_path,
        failure_summary_csv=summary_path,
        strategy_results_csv=strategy_results_path,
        strategy_summary_csv=strategy_summary_path,
        summary_json=json_path,
    )


def summarize_failure_rates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return failure count and rate for each requested diagnostic dimension."""

    dimensions = (
        ("family", lambda row: str(row["family"])),
        ("parameter_variant_id", lambda row: str(row["parameter_variant_id"])),
        ("source_depth_bin_km", lambda row: str(row["source_depth_bin_km"])),
        ("source_receiver_distance_bin_km", lambda row: str(row["source_receiver_distance_bin_km"])),
        ("acquisition_profile", lambda row: str(row["acquisition_profile"])),
        ("interface_type_crossed", lambda row: str(row["interface_type_crossed"])),
        ("iteration_count", lambda row: str(row["iteration_count"])),
        ("termination_reason", lambda row: str(row["termination_reason"])),
    )
    output: list[dict[str, Any]] = []
    for dimension, value_function in dimensions:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[value_function(row)].append(row)
        for level, group in sorted(grouped.items()):
            failure_count = sum(not _as_bool(row["converged"]) for row in group)
            output.append(
                {
                    "dimension": dimension,
                    "level": level,
                    "total_observation_count": len(group),
                    "failure_count": failure_count,
                    "failure_rate": failure_count / len(group) if group else None,
                }
            )
    return output


def run_failure_strategy_experiments(
    failures: Sequence[Mapping[str, Any]],
    case_context: Mapping[str, Mapping[str, Any]],
    settings: BenchmarkSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate bounded retry settings on every historical failure."""

    strategy_specs: tuple[tuple[str, Callable[..., Any], Any], ...] = (
        (
            "max_iterations_60",
            trace_pseudo_bending_ray,
            replace(settings.pseudo_bending_solver, max_iterations_per_level=60),
        ),
        (
            "max_iterations_120",
            trace_pseudo_bending_ray,
            replace(settings.pseudo_bending_solver, max_iterations_per_level=120),
        ),
        (
            "smaller_initial_step_0_125_km",
            trace_pseudo_bending_ray,
            replace(settings.pseudo_bending_solver, perturbation_step_km=0.125),
        ),
        (
            "larger_initial_step_0_5_km",
            trace_pseudo_bending_ray,
            replace(settings.pseudo_bending_solver, perturbation_step_km=0.5),
        ),
        (
            "multi_start_five_initializations",
            trace_pseudo_bending_ray_with_restarts,
            settings.pseudo_bending_solver,
        ),
        (
            "multi_start_five_initializations_max_iterations_60",
            trace_pseudo_bending_ray_with_restarts,
            replace(settings.pseudo_bending_solver, max_iterations_per_level=60),
        ),
    )
    results: list[dict[str, Any]] = []
    for failure in failures:
        context = case_context[str(failure["case_id"])]
        grid = context["grid"]
        source = Point3D(
            float(failure["source_x_km"]),
            float(failure["source_y_km"]),
            float(failure["source_z_km"]),
        )
        receiver = Point3D(
            float(failure["receiver_x_km"]),
            float(failure["receiver_y_km"]),
            float(failure["receiver_z_km"]),
        )
        baseline_time = float(failure["travel_time_s"])
        for strategy_name, tracer, strategy_settings in strategy_specs:
            candidate = tracer(
                source=source,
                receiver=receiver,
                velocity_grid=grid,
                settings=strategy_settings,
            )
            results.append(
                {
                    "case_id": failure["case_id"],
                    "observation_id": failure["observation_id"],
                    "family": failure["family"],
                    "parameter_variant_id": failure["parameter_variant_id"],
                    "acquisition_profile": failure["acquisition_profile"],
                    "interface_type_crossed": failure["interface_type_crossed"],
                    "strategy": strategy_name,
                    "baseline_converged": False,
                    "strategy_converged": candidate.converged,
                    "strategy_termination_reason": candidate.termination_reason,
                    "baseline_iteration_count": failure["iteration_count"],
                    "strategy_iteration_count": candidate.iteration_count,
                    "baseline_travel_time_s": baseline_time,
                    "strategy_travel_time_s": candidate.travel_time_s,
                    "strategy_minus_baseline_time_s": candidate.travel_time_s - baseline_time,
                    "baseline_path_length_km": failure["path_length_km"],
                    "strategy_path_length_km": candidate.path_length_km,
                }
            )

    summary: list[dict[str, Any]] = []
    for strategy_name, _, _ in strategy_specs:
        rows = [row for row in results if row["strategy"] == strategy_name]
        converged_count = sum(_as_bool(row["strategy_converged"]) for row in rows)
        time_deltas = [float(row["strategy_minus_baseline_time_s"]) for row in rows]
        summary.append(
            {
                "strategy": strategy_name,
                "baseline_failure_count": len(rows),
                "converged_after_strategy": converged_count,
                "remaining_nonconverged": len(rows) - converged_count,
                "rescue_fraction": converged_count / len(rows) if rows else None,
                "mean_strategy_iteration_count": statistics.mean(
                    float(row["strategy_iteration_count"]) for row in rows
                )
                if rows
                else None,
                "mean_strategy_minus_baseline_time_s": statistics.mean(time_deltas)
                if time_deltas
                else None,
                "max_absolute_strategy_minus_baseline_time_s": max(
                    (abs(value) for value in time_deltas), default=None
                ),
            }
        )
    return results, summary


def _load_observation_diagnostics(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    repo_root: Path,
    settings: BenchmarkSettings,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    context: dict[str, dict[str, Any]] = {}
    for item in manifest.get("items", []):
        if not isinstance(item, dict):
            raise ValueError("Manifest items must be objects.")
        case_id = str(item.get("case_id", ""))
        if not case_id:
            raise ValueError("Manifest item lacks case_id.")
        grid_path = _resolve_manifest_path(item.get("target_grid_path"), manifest_path, repo_root)
        grid = _load_grid(grid_path)
        context[case_id] = {
            "grid": grid,
            "grid_path": grid_path,
            "manifest_item": item,
        }
        sidecar_path = _resolve_manifest_path(
            item.get("ray_path_sidecar_path"), manifest_path, repo_root
        )
        for record in _read_jsonl(sidecar_path):
            source = record.get("source")
            receiver = record.get("receiver")
            if not isinstance(source, dict) or not isinstance(receiver, dict):
                raise ValueError(f"Sidecar record lacks source/receiver: {record!r}")
            source_point = _point_from_mapping(source)
            receiver_point = _point_from_mapping(receiver)
            scenario = str(item.get("scenario", grid.metadata.get("grid_scenario", "layered")))
            interface = _classify_interfaces(source_point, receiver_point, record, grid)
            max_iterations = _configured_max_iterations(item, settings)
            iteration_count = _as_int(record.get("iteration_count"), 0)
            converged_value = record.get("converged")
            termination_reason = str(record.get("termination_reason") or "")
            if not termination_reason:
                termination_reason = (
                    "max_iterations_reached"
                    if converged_value is False and iteration_count >= max_iterations
                    else "historical_reason_unknown"
                )
            distance = _distance_km(source_point, receiver_point)
            row = {
                "case_id": case_id,
                "family": str(item.get("family", scenario)),
                "scenario": scenario,
                "parameter_variant_id": str(item.get("parameter_variant_id", "")),
                "acquisition_profile": str(item.get("geometry_profile_id", "")),
                "station_seed": item.get("station_seed", ""),
                "earthquake_seed": item.get("earthquake_seed", ""),
                "target_hash": target_vector_sha256(grid.p_velocity_km_per_s),
                "target_grid_path": _relative(grid_path, repo_root),
                "observation_id": str(record.get("observation_id", "")),
                "earthquake_id": str(record.get("earthquake_id", "")),
                "station_id": str(record.get("station_id", "")),
                "converged": converged_value,
                "termination_reason": termination_reason,
                "iteration_count": iteration_count,
                "max_iterations_per_level": max_iterations,
                "source_x_km": source_point.x_km,
                "source_y_km": source_point.y_km,
                "source_z_km": source_point.z_km,
                "source_depth_km": source_point.z_km,
                "source_depth_bin_km": _depth_bin(source_point.z_km),
                "receiver_x_km": receiver_point.x_km,
                "receiver_y_km": receiver_point.y_km,
                "receiver_z_km": receiver_point.z_km,
                "source_receiver_distance_km": distance,
                "source_receiver_distance_bin_km": _distance_bin(distance),
                "interface_type_crossed": interface["all_types"],
                "geological_interface_type_crossed": interface["geological_types"],
                "horizontal_interface_count": interface["horizontal_count"],
                "interface_sequence": interface["sequence"],
                "travel_time_s": record.get("travel_time_s", ""),
                "path_length_km": record.get("path_length_km", ""),
                "ray_point_count": len(record.get("ray_path", [])),
                "sidecar_path": _relative(sidecar_path, repo_root),
            }
            rows.append(row)
    return rows, context


def _classify_interfaces(
    source: Point3D,
    receiver: Point3D,
    record: Mapping[str, Any],
    grid: CartesianVelocityGrid3D,
) -> dict[str, Any]:
    from tomobench.simulation.ray_tracing import _segment_interface_crossings

    path = tuple(
        _point_from_mapping(point)
        for point in record.get("ray_path", [])
        if isinstance(point, dict)
    )
    if len(path) < 2:
        path = (source, receiver)
    geological: set[str] = set()
    sequence: list[str] = []
    for start, end in zip(path, path[1:]):
        for _, constraint in _segment_interface_crossings(start, end, grid):
            if constraint.kind != "horizontal":
                geological.add(constraint.kind)
                sequence.append(constraint.kind)
    for _, constraint in _segment_interface_crossings(source, receiver, grid):
        if constraint.kind != "horizontal":
            geological.add(constraint.kind)
            if constraint.kind not in sequence:
                sequence.append(constraint.kind)
    boundaries = grid.metadata.get("layered_model", {}).get("depth_boundaries_km", [])
    low = min(source.z_km, receiver.z_km)
    high = max(source.z_km, receiver.z_km)
    horizontal_count = sum(low < float(boundary) < high for boundary in boundaries)
    all_types = []
    if horizontal_count:
        all_types.append("horizontal_layer")
    all_types.extend(sorted(geological))
    return {
        "all_types": "|".join(all_types) if all_types else "none",
        "geological_types": "|".join(sorted(geological)) if geological else "none",
        "horizontal_count": horizontal_count,
        "sequence": "|".join(sequence) if sequence else "none",
    }


def _case_convergence(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    return [
        {
            "case_id": case_id,
            "nonconverged_count": sum(not _as_bool(row["converged"]) for row in group),
        }
        for case_id, group in sorted(grouped.items())
    ]


def _read_historical_failure_ids(path: Path) -> set[str]:
    rows = _read_csv(path)
    return {str(row.get("observation_id", "")) for row in rows if row.get("observation_id")}


def _load_grid(path: Path) -> CartesianVelocityGrid3D:
    payload = _read_json(path)
    return CartesianVelocityGrid3D(
        grid_id=str(payload["grid_id"]),
        source_velocity_model_id=str(payload["source_velocity_model_id"]),
        x_coordinates_km=tuple(float(value) for value in payload["x_coordinates_km"]),
        y_coordinates_km=tuple(float(value) for value in payload["y_coordinates_km"]),
        z_coordinates_km=tuple(float(value) for value in payload["z_coordinates_km"]),
        p_velocity_km_per_s=tuple(float(value) for value in payload["p_velocity_km_per_s"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required ray-path sidecar is missing: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Sidecar line is not an object: {path}")
                records.append(record)
    return records


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _point_from_mapping(value: Mapping[str, Any]) -> Point3D:
    return Point3D(float(value["x_km"]), float(value["y_km"]), float(value["z_km"]))


def _configured_max_iterations(item: Mapping[str, Any], settings: BenchmarkSettings) -> int:
    solver = item.get("pseudo_bending_solver")
    if isinstance(solver, dict) and solver.get("max_iterations_per_level") is not None:
        return int(solver["max_iterations_per_level"])
    return settings.pseudo_bending_solver.max_iterations_per_level


def _depth_bin(value: float) -> str:
    lower = math.floor(max(value, 0.0) / 5.0) * 5
    return f"[{lower:g},{lower + 5:g}) km"


def _distance_bin(value: float) -> str:
    lower = math.floor(max(value, 0.0) / 25.0) * 25
    return f"[{lower:g},{lower + 25:g}) km"


def _distance_km(first: Point3D, second: Point3D) -> float:
    return math.sqrt(
        (first.x_km - second.x_km) ** 2
        + (first.y_km - second.y_km) ** 2
        + (first.z_km - second.z_km) ** 2
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _resolve_manifest_path(value: Any, manifest_path: Path, repo_root: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    candidate = repo_root / path
    return candidate if candidate.is_file() else manifest_path.parent / path


def _relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved)
