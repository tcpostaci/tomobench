"""FSM contract audit for the next authoritative-forward-solver decision.

This module compares direct analytic sampling at node-centered velocity and
cell-centered slowness locations.  It is deliberately audit-only: it does not
generate Benchmark v2 targets, fit ML models, or modify manuscript files.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
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
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import LayeredVelocityModel
from tomobench.evaluation.authoritative_forward_solver_validation import (
    DEFAULT_SEED,
    GEOLOGICAL_FAMILIES,
    ValidationPair,
    _homogeneous_model,
    _homogeneous_validation_pairs,
    _layered_validation_pairs,
    _representative_pairs,
)
from tomobench.generation.velocity_grids import (
    build_cartesian_cell_velocity_field,
    build_cartesian_forward_grid,
    fault_plane_signed_offset_km,
)
from tomobench.generation.velocity_models import generate_layered_velocity_model
from tomobench.simulation.ttcrpy_forward import (
    TtcrpyGridConfiguration,
    TtcrpyRaytraceResult,
    TtcrpyRectilinearForwardSolver,
    velocity_array_from_cartesian_grid,
)
from tomobench.simulation.travel_times import layered_first_arrival_travel_time_s
from tomobench.utils.paths import get_repo_root


VALIDATION_ID = "authoritative_forward_solver_v2"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/authoritative_forward_solver_v2"
)
REPRESENTATIONS = ("node_velocity", "cell_slowness")
DEFAULT_SPACINGS_KM = (2.5, 1.25, 0.625)
DEFAULT_DIFFICULT_SPACING_KM = 0.3125
DEFAULT_MAX_FORWARD_NODES_V2 = 12_000_000
DEFAULT_SIGNAL_EPSILON_S = 1.0e-6
DEFAULT_TRANSITION_WIDTHS_KM = (1.25, 2.5, 5.0)

RepresentationName = Literal["node_velocity", "cell_slowness"]


@dataclass(frozen=True)
class FsmFieldArtifact:
    """One direct analytic field and its configured ttcrpy FSM solver."""

    family: str
    representation: RepresentationName
    spacing_km: float
    x_coordinates_km: tuple[float, ...]
    y_coordinates_km: tuple[float, ...]
    z_coordinates_km: tuple[float, ...]
    velocity_values: np.ndarray
    solver: TtcrpyRectilinearForwardSolver
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class FsmRunResult:
    """A successful or failed batched FSM calculation."""

    status: str
    travel_times_s: tuple[float, ...]
    raytrace: TtcrpyRaytraceResult | None
    error: str | None


def run_authoritative_forward_solver_v2_audit(
    *,
    settings: BenchmarkSettings | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    spacings_km: Sequence[float] = DEFAULT_SPACINGS_KM,
    difficult_spacing_km: float = DEFAULT_DIFFICULT_SPACING_KM,
    pairs_per_family: int = 8,
    difficult_pairs_per_family: int = 2,
    analytic_fine_pairs: int = 3,
    signal_epsilon_s: float = DEFAULT_SIGNAL_EPSILON_S,
    transition_widths_km: Sequence[float] = DEFAULT_TRANSITION_WIDTHS_KM,
    runtime_spacings_km: Sequence[float] = (1.25, 0.625),
    max_forward_nodes: int = DEFAULT_MAX_FORWARD_NODES_V2,
    include_difficult: bool = True,
    run_fault_diagnostics: bool = True,
    run_smooth_sensitivity: bool = True,
    run_runtime_model: bool = True,
) -> Path:
    """Run the FSM representation and numerical-quality audit."""
    loaded_settings = settings or load_settings()
    spacings = _validate_spacings(spacings_km)
    if difficult_spacing_km <= 0.0 or difficult_spacing_km >= min(spacings):
        raise ValueError("difficult_spacing_km must be positive and finer than the regular spacings.")
    _validate_positive_int(pairs_per_family, "pairs_per_family")
    _validate_positive_int(difficult_pairs_per_family, "difficult_pairs_per_family")
    _validate_positive_int(analytic_fine_pairs, "analytic_fine_pairs")
    if signal_epsilon_s <= 0.0:
        raise ValueError("signal_epsilon_s must be positive.")
    transition_widths = tuple(float(value) for value in transition_widths_km)
    if not transition_widths or any(value <= 0.0 for value in transition_widths):
        raise ValueError("transition_widths_km must contain positive values.")

    resolved_output = _resolve(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    case_rows, refinement_rows = _run_error_budget(
        loaded_settings,
        spacings=spacings,
        difficult_spacing_km=difficult_spacing_km,
        pairs_per_family=pairs_per_family,
        difficult_pairs_per_family=difficult_pairs_per_family,
        signal_epsilon_s=signal_epsilon_s,
        max_forward_nodes=max_forward_nodes,
        include_difficult=include_difficult,
    )
    analytic_rows = _run_analytic_representation_comparison(
        loaded_settings,
        spacings=spacings,
        difficult_spacing_km=difficult_spacing_km,
        analytic_fine_pairs=analytic_fine_pairs,
        max_forward_nodes=max_forward_nodes,
        include_difficult=include_difficult,
    )
    representation_rows = _build_representation_rows(case_rows)
    if run_fault_diagnostics:
        fault_rows, fault_path_rows, fault_slice_rows = _run_fault_diagnostics(
            loaded_settings,
            case_rows=case_rows,
            refinement_rows=refinement_rows,
            spacings=spacings,
            difficult_spacing_km=difficult_spacing_km,
            max_forward_nodes=max_forward_nodes,
            include_difficult=include_difficult,
        )
    else:
        fault_rows, fault_path_rows, fault_slice_rows = [], [], []
    target_rows = _run_target_grid_resolution_audit(loaded_settings)
    smooth_rows = (
        _run_smooth_interface_sensitivity(
            loaded_settings,
            spacings=spacings,
            transition_widths_km=transition_widths,
            max_forward_nodes=max_forward_nodes,
        )
        if run_smooth_sensitivity
        else []
    )
    runtime_rows = (
        _run_runtime_model(
            loaded_settings,
            spacings=tuple(float(value) for value in runtime_spacings_km),
            max_forward_nodes=max_forward_nodes,
        )
        if run_runtime_model
        else []
    )

    _write_csv(resolved_output / "error_budget_case_metrics.csv", case_rows)
    _write_csv(resolved_output / "error_budget_refinement.csv", refinement_rows)
    _write_csv(resolved_output / "analytic_representation_comparison.csv", analytic_rows)
    _write_csv(resolved_output / "representation_comparison.csv", representation_rows)
    _write_csv(resolved_output / "fault_resolution_case_metrics.csv", fault_rows)
    _write_csv(resolved_output / "fault_ray_paths.csv", fault_path_rows)
    _write_csv(resolved_output / "fault_model_slices.csv", fault_slice_rows)
    _write_csv(resolved_output / "target_grid_resolution_audit.csv", target_rows)
    _write_csv(resolved_output / "smooth_interface_sensitivity.csv", smooth_rows)
    _write_csv(resolved_output / "runtime_model.csv", runtime_rows)
    _write_error_budget_plot(resolved_output / "error_budget_distributions.png", refinement_rows)
    _write_fault_slice_plot(resolved_output / "fault_model_slices.png", fault_slice_rows)
    (resolved_output / "fault_resolution_diagnostics.md").write_text(
        build_fault_diagnostics_markdown(fault_rows),
        encoding="utf-8",
    )
    (resolved_output / "target_grid_recommendation.md").write_text(
        build_target_grid_recommendation_markdown(target_rows),
        encoding="utf-8",
    )
    _write_json(
        resolved_output / "audit_metadata.json",
        _audit_metadata(
            loaded_settings,
            spacings=spacings,
            difficult_spacing_km=difficult_spacing_km,
            pairs_per_family=pairs_per_family,
            difficult_pairs_per_family=difficult_pairs_per_family,
            analytic_fine_pairs=analytic_fine_pairs,
            signal_epsilon_s=signal_epsilon_s,
            transition_widths_km=transition_widths,
            runtime_spacings_km=tuple(float(value) for value in runtime_spacings_km),
            max_forward_nodes=max_forward_nodes,
        ),
    )
    summary = build_v2_audit_summary(
        case_rows=case_rows,
        refinement_rows=refinement_rows,
        analytic_rows=analytic_rows,
        representation_rows=representation_rows,
        fault_rows=fault_rows,
        target_rows=target_rows,
        smooth_rows=smooth_rows,
        runtime_rows=runtime_rows,
        signal_epsilon_s=signal_epsilon_s,
        spacings=spacings,
        difficult_spacing_km=difficult_spacing_km,
    )
    _write_json(resolved_output / "audit_summary.json", summary)
    (resolved_output / "audit_summary.md").write_text(
        build_v2_audit_summary_markdown(summary),
        encoding="utf-8",
    )
    return resolved_output


def _run_error_budget(
    settings: BenchmarkSettings,
    *,
    spacings: Sequence[float],
    difficult_spacing_km: float,
    pairs_per_family: int,
    difficult_pairs_per_family: int,
    signal_epsilon_s: float,
    max_forward_nodes: int,
    include_difficult: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    case_rows: list[dict[str, object]] = []
    refinement_rows: list[dict[str, object]] = []
    all_spacings = tuple(sorted(spacings, reverse=True))
    if include_difficult:
        all_spacings = tuple(sorted((*all_spacings, difficult_spacing_km), reverse=True))
    pairs_by_family_spacing: dict[tuple[str, float], tuple[ValidationPair, ...]] = {}
    for family_index, family in enumerate(GEOLOGICAL_FAMILIES):
        full_pairs = _representative_pairs(family, pairs_per_family, DEFAULT_SEED + family_index)
        difficult_pairs = full_pairs[: min(difficult_pairs_per_family, len(full_pairs))]
        for spacing in all_spacings:
            pairs_by_family_spacing[(family, spacing)] = (
                difficult_pairs if spacing == difficult_spacing_km else full_pairs
            )
    for representation in REPRESENTATIONS:
        times: dict[tuple[str, float, str], float] = {}
        signals: dict[tuple[str, float, str], float] = {}
        for spacing in all_spacings:
            background_model = generate_layered_velocity_model(settings)
            try:
                artifact_background = _build_field_artifact(
                    settings,
                    family="layered",
                    model=background_model,
                    spacing_km=spacing,
                    representation=representation,
                    max_forward_nodes=max_forward_nodes,
                )
                background_result = None
                background_error = None
            except Exception as exc:
                artifact_background = None
                background_result = FsmRunResult(
                    "error", (), None, f"{type(exc).__name__}: {exc}"
                )
                background_error = background_result.error
            for family in GEOLOGICAL_FAMILIES:
                pairs = pairs_by_family_spacing[(family, spacing)]
                artifact_heterogeneous = None
                heterogeneous_error = None
                if artifact_background is not None:
                    background_result = _raytrace_pairs(artifact_background.solver, pairs)
                try:
                    artifact_heterogeneous = _build_field_artifact(
                        settings,
                        family=family,
                        model=background_model,
                        spacing_km=spacing,
                        representation=representation,
                        max_forward_nodes=max_forward_nodes,
                    )
                    heterogeneous_result = _raytrace_pairs(artifact_heterogeneous.solver, pairs)
                except Exception as exc:
                    heterogeneous_result = FsmRunResult(
                        "error", (), None, f"{type(exc).__name__}: {exc}"
                    )
                    heterogeneous_error = heterogeneous_result.error
                for pair_index, pair in enumerate(pairs):
                    background_time = _result_value(background_result, pair_index)
                    heterogeneous_time = _result_value(heterogeneous_result, pair_index)
                    if background_time is not None and heterogeneous_time is not None:
                        key = (family, spacing, pair.pair_id)
                        times[key] = heterogeneous_time
                        signals[key] = heterogeneous_time - background_time
                    case_rows.append(
                        _case_row(
                            family=family,
                            pair=pair,
                            representation=representation,
                            spacing_km=spacing,
                            background_time_s=background_time,
                            heterogeneous_time_s=heterogeneous_time,
                            signal_epsilon_s=signal_epsilon_s,
                            status="ok"
                            if background_time is not None and heterogeneous_time is not None
                            else "error",
                            error=(
                                background_result.error
                                if background_result is not None and background_result.error
                                else heterogeneous_result.error
                                if heterogeneous_result.error
                                else background_error or heterogeneous_error
                            ),
                        )
                    )
                del artifact_heterogeneous
                gc.collect()
            del artifact_background
            gc.collect()
        for coarse, fine in zip(all_spacings, all_spacings[1:]):
            for family in GEOLOGICAL_FAMILIES:
                pair_ids = {pair.pair_id for pair in pairs_by_family_spacing[(family, coarse)]} & {
                    pair.pair_id for pair in pairs_by_family_spacing[(family, fine)]
                }
                for pair_id in sorted(pair_ids):
                    coarse_key = (family, coarse, pair_id)
                    fine_key = (family, fine, pair_id)
                    coarse_time = times.get(coarse_key)
                    fine_time = times.get(fine_key)
                    signal = signals.get(fine_key)
                    if coarse_time is None or fine_time is None or signal is None:
                        continue
                    abs_change = abs(coarse_time - fine_time)
                    abs_signal = abs(signal)
                    signal_class = (
                        "essentially_zero_signal" if abs_signal < signal_epsilon_s else "affected_signal"
                    )
                    refinement_rows.append(
                        {
                            "family": family,
                            "pair_id": pair_id,
                            "representation": representation,
                            "coarse_spacing_km": coarse,
                            "fine_spacing_km": fine,
                            "coarse_travel_time_s": coarse_time,
                            "fine_travel_time_s": fine_time,
                            "absolute_refinement_difference_s": abs_change,
                            "relative_refinement_difference": abs_change / abs(fine_time)
                            if fine_time != 0.0
                            else None,
                            "fine_geological_signal_s": signal,
                            "absolute_geological_signal_s": abs_signal,
                            "signal_class": signal_class,
                            "numerical_error_over_signal": (
                                abs_change / abs_signal if signal_class == "affected_signal" else None
                            ),
                        }
                    )
    return case_rows, refinement_rows


def _run_analytic_representation_comparison(
    settings: BenchmarkSettings,
    *,
    spacings: Sequence[float],
    difficult_spacing_km: float,
    analytic_fine_pairs: int,
    max_forward_nodes: int,
    include_difficult: bool,
) -> list[dict[str, object]]:
    cases = (
        ("homogeneous", _homogeneous_model(5.0), _homogeneous_validation_pairs()),
        ("layered", generate_layered_velocity_model(settings), _layered_validation_pairs()),
    )
    rows: list[dict[str, object]] = []
    for family, model, all_pairs in cases:
        references = {
            pair.pair_id: (
                _euclidean_time(pair.source, pair.receiver, 5.0)
                if family == "homogeneous"
                else layered_first_arrival_travel_time_s(
                    pair.source,
                    pair.receiver,
                    model,
                    numerical_tolerance=1.0e-10,
                )
            )
            for pair in all_pairs
        }
        validation_spacings = (*spacings, difficult_spacing_km) if include_difficult else tuple(spacings)
        for spacing in validation_spacings:
            pairs = all_pairs if spacing != difficult_spacing_km else all_pairs[:analytic_fine_pairs]
            for representation in REPRESENTATIONS:
                try:
                    artifact = _build_field_artifact(
                        settings,
                        family="layered" if family in ("homogeneous", "layered") else family,
                        model=model,
                        spacing_km=spacing,
                        representation=representation,
                        max_forward_nodes=max_forward_nodes,
                    )
                    result = _raytrace_pairs(artifact.solver, pairs)
                    for pair_index, pair in enumerate(pairs):
                        travel_time = _result_value(result, pair_index)
                        reference = references[pair.pair_id]
                        rows.append(
                            {
                                "validation_kind": "analytic_representation",
                                "family": family,
                                "pair_id": pair.pair_id,
                                "representation": representation,
                                "spacing_km": spacing,
                                "reference_kind": (
                                    "homogeneous_euclidean"
                                    if family == "homogeneous"
                                    else "layered_first_arrival_snell"
                                ),
                                "reference_travel_time_s": reference,
                                "travel_time_s": travel_time,
                                "absolute_error_s": abs(travel_time - reference)
                                if travel_time is not None
                                else None,
                                "relative_error": abs(travel_time - reference) / abs(reference)
                                if travel_time is not None and reference != 0.0
                                else None,
                                "status": "ok" if travel_time is not None else result.status,
                                "error": result.error,
                            }
                        )
                    del artifact
                except Exception as exc:
                    for pair in pairs:
                        rows.append(
                            {
                                "validation_kind": "analytic_representation",
                                "family": family,
                                "pair_id": pair.pair_id,
                                "representation": representation,
                                "spacing_km": spacing,
                                "reference_kind": (
                                    "homogeneous_euclidean"
                                    if family == "homogeneous"
                                    else "layered_first_arrival_snell"
                                ),
                                "reference_travel_time_s": references[pair.pair_id],
                                "travel_time_s": None,
                                "absolute_error_s": None,
                                "relative_error": None,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                finally:
                    gc.collect()
    return rows


def _build_field_artifact(
    settings: BenchmarkSettings,
    *,
    family: str,
    model: LayeredVelocityModel,
    spacing_km: float,
    representation: RepresentationName,
    max_forward_nodes: int,
    smooth_width_km: float | None = None,
) -> FsmFieldArtifact:
    configuration = TtcrpyGridConfiguration(
        method="FSM",
        cell_slowness=representation == "cell_slowness",
        # Use the FSM arrival-time field for labels.  ``tt_from_rp=True``
        # invokes ray-path integration and can fail for otherwise valid
        # off-grid endpoint pairs on coarse grids; path extraction remains a
        # separate diagnostic where the package returns a valid path.
        tt_from_rp=False,
        maxit=100,
    )
    if smooth_width_km is not None:
        node_grid = build_cartesian_forward_grid(
            settings,
            model,
            scenario=family,
            spacing_by_axis_km=(spacing_km, spacing_km, spacing_km),
            max_grid_nodes=max_forward_nodes,
        )
        values = _smooth_velocity_values(
            settings,
            model,
            family,
            node_grid.x_coordinates_km,
            node_grid.y_coordinates_km,
            node_grid.z_coordinates_km,
            smooth_width_km,
        )
        grid = replace(node_grid, p_velocity_km_per_s=_flatten_x_fastest(values))
        solver = TtcrpyRectilinearForwardSolver.from_cartesian_grid(grid, configuration)
        return FsmFieldArtifact(
            family=family,
            representation="node_velocity",
            spacing_km=spacing_km,
            x_coordinates_km=grid.x_coordinates_km,
            y_coordinates_km=grid.y_coordinates_km,
            z_coordinates_km=grid.z_coordinates_km,
            velocity_values=values,
            solver=solver,
            metadata={**grid.metadata, "smooth_transition_width_km": smooth_width_km},
        )
    if representation == "node_velocity":
        grid = build_cartesian_forward_grid(
            settings,
            model,
            scenario=family,
            spacing_by_axis_km=(spacing_km, spacing_km, spacing_km),
            max_grid_nodes=max_forward_nodes,
        )
        values = velocity_array_from_cartesian_grid(grid)
        solver = TtcrpyRectilinearForwardSolver.from_cartesian_grid(grid, configuration)
        return FsmFieldArtifact(
            family=family,
            representation=representation,
            spacing_km=spacing_km,
            x_coordinates_km=grid.x_coordinates_km,
            y_coordinates_km=grid.y_coordinates_km,
            z_coordinates_km=grid.z_coordinates_km,
            velocity_values=values,
            solver=solver,
            metadata=grid.metadata,
        )
    field = build_cartesian_cell_velocity_field(
        settings,
        model,
        scenario=family,
        spacing_by_axis_km=(spacing_km, spacing_km, spacing_km),
        max_grid_nodes=max_forward_nodes,
    )
    shape = (
        len(field.x_coordinates_km) - 1,
        len(field.y_coordinates_km) - 1,
        len(field.z_coordinates_km) - 1,
    )
    flat = np.asarray(field.p_velocity_km_per_s, dtype=np.float64)
    values = flat.reshape((shape[2], shape[1], shape[0])).transpose(2, 1, 0)
    solver = TtcrpyRectilinearForwardSolver.from_cell_velocity_field(field, configuration)
    return FsmFieldArtifact(
        family=family,
        representation=representation,
        spacing_km=spacing_km,
        x_coordinates_km=field.x_coordinates_km,
        y_coordinates_km=field.y_coordinates_km,
        z_coordinates_km=field.z_coordinates_km,
        velocity_values=values,
        solver=solver,
        metadata=field.metadata,
    )


def _raytrace_pairs(
    solver: TtcrpyRectilinearForwardSolver,
    pairs: Sequence[ValidationPair],
    *,
    return_rays: bool = False,
) -> FsmRunResult:
    try:
        result = solver.raytrace(
            [pair.source for pair in pairs],
            [pair.receiver for pair in pairs],
            return_rays=return_rays,
        )
        return FsmRunResult("ok", result.travel_times_s, result, None)
    except Exception as exc:
        return FsmRunResult("error", (), None, f"{type(exc).__name__}: {exc}")


def _result_value(result: FsmRunResult | None, index: int) -> float | None:
    """Return one result value while preserving failed/partial batches."""
    if result is None or index >= len(result.travel_times_s):
        return None
    value = float(result.travel_times_s[index])
    return value if math.isfinite(value) else None


def _case_row(
    *,
    family: str,
    pair: ValidationPair,
    representation: str,
    spacing_km: float,
    background_time_s: float | None,
    heterogeneous_time_s: float | None,
    signal_epsilon_s: float,
    status: str,
    error: str | None,
) -> dict[str, object]:
    signal = (
        heterogeneous_time_s - background_time_s
        if background_time_s is not None and heterogeneous_time_s is not None
        else None
    )
    abs_signal = abs(signal) if signal is not None else None
    return {
        "validation_kind": "heterogeneous_error_budget",
        "family": family,
        "pair_id": pair.pair_id,
        "representation": representation,
        "spacing_km": spacing_km,
        "source_x_km": pair.source.x_km,
        "source_y_km": pair.source.y_km,
        "source_z_km": pair.source.z_km,
        "receiver_x_km": pair.receiver.x_km,
        "receiver_y_km": pair.receiver.y_km,
        "receiver_z_km": pair.receiver.z_km,
        "background_travel_time_s": background_time_s,
        "heterogeneous_travel_time_s": heterogeneous_time_s,
        "delta_t_geology_s": signal,
        "absolute_delta_t_geology_s": abs_signal,
        "signal_class": (
            "essentially_zero_signal"
            if abs_signal is not None and abs_signal < signal_epsilon_s
            else "affected_signal"
            if abs_signal is not None
            else "unavailable"
        ),
        "independent_solver_difference_s": None,
        "independent_difference_over_signal": None,
        "status": status,
        "error": error,
    }


def _build_representation_rows(case_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in case_rows:
        key = (str(row["family"]), str(row["pair_id"]), float(row["spacing_km"]))
        grouped[key][str(row["representation"])] = row
    rows: list[dict[str, object]] = []
    for (family, pair_id, spacing), values in sorted(grouped.items()):
        node = values.get("node_velocity", {})
        cell = values.get("cell_slowness", {})
        node_time = node.get("heterogeneous_travel_time_s")
        cell_time = cell.get("heterogeneous_travel_time_s")
        rows.append(
            {
                "validation_kind": "node_vs_cell_fsm",
                "family": family,
                "pair_id": pair_id,
                "spacing_km": spacing,
                "node_status": node.get("status"),
                "cell_status": cell.get("status"),
                "node_travel_time_s": node_time,
                "cell_travel_time_s": cell_time,
                "cell_minus_node_s": (
                    float(cell_time) - float(node_time)
                    if node_time is not None and cell_time is not None
                    else None
                ),
                "absolute_cell_minus_node_s": (
                    abs(float(cell_time) - float(node_time))
                    if node_time is not None and cell_time is not None
                    else None
                ),
                "node_error": node.get("error"),
                "cell_error": cell.get("error"),
            }
        )
    return rows


def _run_fault_diagnostics(
    settings: BenchmarkSettings,
    *,
    case_rows: Sequence[Mapping[str, object]],
    refinement_rows: Sequence[Mapping[str, object]],
    spacings: Sequence[float],
    difficult_spacing_km: float,
    max_forward_nodes: int,
    include_difficult: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    fault_refinement = [
        row
        for row in refinement_rows
        if row.get("family") == "faulted" and row.get("representation") == "node_velocity"
    ]
    ranked = sorted(
        fault_refinement,
        key=lambda row: float(row.get("absolute_refinement_difference_s") or -1.0),
        reverse=True,
    )
    selected_pair_ids: list[str] = []
    for row in ranked:
        pair_id = str(row["pair_id"])
        if pair_id not in selected_pair_ids:
            selected_pair_ids.append(pair_id)
        if len(selected_pair_ids) >= 4:
            break
    if not selected_pair_ids:
        selected_pair_ids = [pair.pair_id for pair in _representative_pairs("faulted", 4, DEFAULT_SEED)]
    selected_pairs = {
        pair.pair_id: pair
        for pair in _representative_pairs("faulted", max(8, len(selected_pair_ids)), DEFAULT_SEED + 2)
        if pair.pair_id in selected_pair_ids
    }
    diagnostic_spacings = (
        tuple((*spacings, difficult_spacing_km))
        if include_difficult
        else tuple(spacings)
    )
    rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    model = generate_layered_velocity_model(settings)
    faulted = settings.velocity_model_generation.faulted
    for spacing in diagnostic_spacings:
        try:
            artifact = _build_field_artifact(
                settings,
                family="faulted",
                model=model,
                spacing_km=spacing,
                representation="node_velocity",
                max_forward_nodes=max_forward_nodes,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for pair in selected_pairs.values():
                rows.append(
                    {
                        "family": "faulted",
                        "pair_id": pair.pair_id,
                        "spacing_km": spacing,
                        "travel_time_s": None,
                        "path_length_km": None,
                        "fault_side_source": None,
                        "fault_side_receiver": None,
                        "path_fault_sign_sequence": None,
                        "fault_topology_signature": None,
                        "fault_crossing_count": None,
                        "crosses_fault": None,
                        "path_topology_switch_across_resolutions": None,
                        "status": "error",
                        "error": error,
                    }
                )
            continue
        pairs = tuple(selected_pairs.values())
        result = _raytrace_pairs(artifact.solver, pairs, return_rays=True)
        rays = result.raytrace.ray_paths if result.raytrace and result.raytrace.ray_paths else ()
        for pair_index, pair in enumerate(pairs):
            travel_time = _result_value(result, pair_index)
            ray = rays[pair_index] if pair_index < len(rays) else None
            if ray is None or travel_time is None:
                rows.append(
                    {
                        "family": "faulted",
                        "pair_id": pair.pair_id,
                        "spacing_km": spacing,
                        "travel_time_s": travel_time,
                        "path_length_km": None,
                        "fault_side_source": None,
                        "fault_side_receiver": None,
                        "path_fault_sign_sequence": None,
                        "fault_topology_signature": None,
                        "fault_crossing_count": None,
                        "crosses_fault": None,
                        "path_topology_switch_across_resolutions": None,
                        "status": result.status,
                        "error": result.error or "No ray path returned for diagnostic pair.",
                    }
                )
                continue
            signs = tuple(
                _fault_sign(fault_plane_signed_offset_km(point.x_km, point.y_km, point.z_km, faulted))
                for point in ray
            )
            crossing_count = _crossing_count(signs)
            rows.append(
                {
                    "family": "faulted",
                    "pair_id": pair.pair_id,
                    "spacing_km": spacing,
                    "travel_time_s": travel_time,
                    "path_length_km": (
                        result.raytrace.path_lengths_km[pair_index]
                        if result.raytrace and result.raytrace.path_lengths_km
                        else None
                    ),
                    "fault_side_source": _fault_sign(
                        fault_plane_signed_offset_km(
                            pair.source.x_km, pair.source.y_km, pair.source.z_km, faulted
                        )
                    ),
                    "fault_side_receiver": _fault_sign(
                        fault_plane_signed_offset_km(
                            pair.receiver.x_km, pair.receiver.y_km, pair.receiver.z_km, faulted
                        )
                    ),
                    "path_fault_sign_sequence": "".join(signs),
                    "fault_topology_signature": _fault_topology_signature(signs),
                    "fault_crossing_count": crossing_count,
                    "crosses_fault": crossing_count > 0,
                    "status": result.status,
                    "error": result.error,
                }
            )
            for path_index, point in enumerate(ray):
                path_rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "spacing_km": spacing,
                        "path_index": path_index,
                        "x_km": point.x_km,
                        "y_km": point.y_km,
                        "z_km": point.z_km,
                        "fault_side": signs[path_index],
                    }
                )
        slice_rows.extend(_fault_slice_rows(artifact, faulted, spacing))
        del artifact
        gc.collect()
    topology_by_pair: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        topology_by_pair[str(row["pair_id"])].add(str(row["fault_topology_signature"]))
    for row in rows:
        sequences = topology_by_pair[str(row["pair_id"])]
        row["path_topology_switch_across_resolutions"] = len(sequences) > 1
    return rows, path_rows, slice_rows


def _run_target_grid_resolution_audit(settings: BenchmarkSettings) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target_options = {
        "historical_5x5x2p5": (5.0, 5.0, 2.5),
        "option_b_2p5x2p5x2p5": (2.5, 2.5, 2.5),
        "option_b_2p5x2p5x1p25": (2.5, 2.5, 1.25),
    }
    body_sweeps = {
        "block_anomaly": ((5.0, 7.5, 10.0, 12.5, 15.0), "width_km"),
        "salt_dome": ((10.0, 12.5, 15.0, 17.5, 20.0), "diameter_km"),
        "dyke_intrusion": ((3.0, 5.0, 7.5, 10.0, 12.5, 15.0), "width_km"),
    }
    center_offsets = {
        "block_anomaly": ((47.5, 47.5), (50.0, 50.0), (52.5, 52.5)),
        "salt_dome": ((47.5, 47.5), (50.0, 50.0), (52.5, 52.5)),
        "dyke_intrusion": ((47.5, 50.0), (50.0, 50.0), (52.5, 50.0)),
    }
    model = generate_layered_velocity_model(settings)
    for option_name, spacing in target_options.items():
        for family in ("layered", "block_anomaly", "faulted", "salt_dome", "dyke_intrusion"):
            baseline = build_cartesian_forward_grid(
                settings,
                model,
                scenario="layered",
                spacing_by_axis_km=spacing,
                max_grid_nodes=200_000,
            )
            target = build_cartesian_forward_grid(
                settings,
                model,
                scenario=family,
                spacing_by_axis_km=spacing,
                max_grid_nodes=200_000,
            )
            baseline_values = np.asarray(baseline.p_velocity_km_per_s, dtype=float)
            target_values = np.asarray(target.p_velocity_km_per_s, dtype=float)
            changed = np.flatnonzero(np.abs(target_values - baseline_values) > 1.0e-12)
            nx = len(target.x_coordinates_km)
            ny = len(target.y_coordinates_km)
            if len(changed):
                iz = changed // (nx * ny)
                remainder = changed % (nx * ny)
                iy = remainder // nx
                ix = remainder % nx
                x_count = int(len(np.unique(ix)))
                y_count = int(len(np.unique(iy)))
                z_count = int(len(np.unique(iz)))
                x_span = int(ix.max() - ix.min())
                y_span = int(iy.max() - iy.min())
                z_span = int(iz.max() - iz.min())
            else:
                x_count = y_count = z_count = x_span = y_span = z_span = 0
            rows.append(
                {
                    "audit_kind": "configured_structure_target_sampling",
                    "option": option_name,
                    "family": family,
                    "spacing_x_km": spacing[0],
                    "spacing_y_km": spacing[1],
                    "spacing_z_km": spacing[2],
                    "grid_shape": f"{len(target.x_coordinates_km)}x{len(target.y_coordinates_km)}x{len(target.z_coordinates_km)}",
                    "target_node_count": len(target_values),
                    "changed_node_count": int(len(changed)),
                    "changed_x_node_count": x_count,
                    "changed_y_node_count": y_count,
                    "changed_z_node_count": z_count,
                    "changed_x_index_span": x_span,
                    "changed_y_index_span": y_span,
                    "changed_z_index_span": z_span,
                    "minimum_lateral_index_span": min(x_span, y_span) if family != "faulted" else None,
                    "parameter_value": None,
                    "center_offset_x_km": None,
                    "center_offset_y_km": None,
                    "meets_two_lateral_intervals": (
                        min(x_span, y_span) >= 2 if family not in ("layered", "faulted") else None
                    ),
                }
            )
    for family, (values, parameter_name) in body_sweeps.items():
        for value in values:
            for center_x, center_y in center_offsets[family]:
                varied = _override_structure(settings, family, value, center_x, center_y)
                varied_model = generate_layered_velocity_model(varied)
                target = build_cartesian_forward_grid(
                    varied,
                    varied_model,
                    scenario=family,
                    spacing_by_axis_km=(5.0, 5.0, 2.5),
                    max_grid_nodes=200_000,
                )
                baseline = build_cartesian_forward_grid(
                    varied,
                    varied_model,
                    scenario="layered",
                    spacing_by_axis_km=(5.0, 5.0, 2.5),
                    max_grid_nodes=200_000,
                )
                target_values = np.asarray(target.p_velocity_km_per_s, dtype=float)
                baseline_values = np.asarray(baseline.p_velocity_km_per_s, dtype=float)
                changed = np.flatnonzero(np.abs(target_values - baseline_values) > 1.0e-12)
                nx = len(target.x_coordinates_km)
                ny = len(target.y_coordinates_km)
                if len(changed):
                    iz = changed // (nx * ny)
                    remainder = changed % (nx * ny)
                    iy = remainder // nx
                    ix = remainder % nx
                    x_span = int(ix.max() - ix.min())
                    y_span = int(iy.max() - iy.min())
                    z_span = int(iz.max() - iz.min())
                else:
                    x_span = y_span = z_span = 0
                rows.append(
                    {
                        "audit_kind": "minimum_dimension_sweep",
                        "option": "historical_5x5x2p5",
                        "family": family,
                        "spacing_x_km": 5.0,
                        "spacing_y_km": 5.0,
                        "spacing_z_km": 2.5,
                        "grid_shape": f"{len(target.x_coordinates_km)}x{len(target.y_coordinates_km)}x{len(target.z_coordinates_km)}",
                        "target_node_count": len(target_values),
                        "changed_node_count": int(len(changed)),
                        "changed_x_node_count": None,
                        "changed_y_node_count": None,
                        "changed_z_node_count": None,
                        "changed_x_index_span": x_span,
                        "changed_y_index_span": y_span,
                        "changed_z_index_span": z_span,
                        "minimum_lateral_index_span": min(x_span, y_span),
                        "parameter_name": parameter_name,
                        "parameter_value": value,
                        "center_offset_x_km": center_x,
                        "center_offset_y_km": center_y,
                        "meets_two_lateral_intervals": min(x_span, y_span) >= 2,
                    }
                )
    return rows


def _run_smooth_interface_sensitivity(
    settings: BenchmarkSettings,
    *,
    spacings: Sequence[float],
    transition_widths_km: Sequence[float],
    max_forward_nodes: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model = generate_layered_velocity_model(settings)
    for family in ("block_anomaly", "faulted", "salt_dome", "dyke_intrusion"):
        pairs = _representative_pairs(family, 2, DEFAULT_SEED + 300)
        sharp_times: dict[float, tuple[float, ...]] = {}
        for spacing in spacings:
            try:
                sharp = _build_field_artifact(
                    settings,
                    family=family,
                    model=model,
                    spacing_km=spacing,
                    representation="node_velocity",
                    max_forward_nodes=max_forward_nodes,
                )
                sharp_result = _raytrace_pairs(sharp.solver, pairs)
                sharp_times[spacing] = sharp_result.travel_times_s
                del sharp
            except Exception:
                sharp_times[spacing] = ()
            gc.collect()
        for width in transition_widths_km:
            smooth_times: dict[float, tuple[float, ...]] = {}
            for spacing in spacings:
                try:
                    smooth = _build_field_artifact(
                        settings,
                        family=family,
                        model=model,
                        spacing_km=spacing,
                        representation="node_velocity",
                        max_forward_nodes=max_forward_nodes,
                        smooth_width_km=width,
                    )
                    smooth_result = _raytrace_pairs(smooth.solver, pairs)
                    smooth_times[spacing] = smooth_result.travel_times_s
                    for pair, sharp_time, smooth_time in zip(
                        pairs,
                        sharp_times[spacing],
                        smooth_result.travel_times_s,
                    ):
                        rows.append(
                            {
                                "comparison_kind": "sharp_vs_smooth",
                                "family": family,
                                "pair_id": pair.pair_id,
                                "transition_width_km": width,
                                "spacing_km": spacing,
                                "sharp_travel_time_s": sharp_time,
                                "smooth_travel_time_s": smooth_time,
                                "smooth_minus_sharp_s": smooth_time - sharp_time,
                                "sharp_refinement_difference_s": None,
                                "smooth_refinement_difference_s": None,
                                "smooth_minus_sharp_refinement_difference_s": None,
                                "status": smooth_result.status,
                                "error": smooth_result.error,
                            }
                        )
                    del smooth
                except Exception as exc:
                    for pair in pairs:
                        rows.append(
                            {
                                "comparison_kind": "sharp_vs_smooth",
                                "family": family,
                                "pair_id": pair.pair_id,
                                "transition_width_km": width,
                                "spacing_km": spacing,
                                "sharp_travel_time_s": (
                                    sharp_times[spacing][pairs.index(pair)]
                                    if len(sharp_times[spacing]) > pairs.index(pair)
                                    else None
                                ),
                                "smooth_travel_time_s": None,
                                "smooth_minus_sharp_s": None,
                                "sharp_refinement_difference_s": None,
                                "smooth_refinement_difference_s": None,
                                "smooth_minus_sharp_refinement_difference_s": None,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                finally:
                    gc.collect()
            ordered = tuple(sorted(spacings, reverse=True))
            for coarse, fine in zip(ordered, ordered[1:]):
                for pair_index, pair in enumerate(pairs):
                    smooth_coarse = smooth_times.get(coarse, ())
                    smooth_fine = smooth_times.get(fine, ())
                    if len(smooth_coarse) <= pair_index or len(smooth_fine) <= pair_index:
                        continue
                    rows.append(
                        {
                            "comparison_kind": "refinement_difference",
                            "family": family,
                            "pair_id": pair.pair_id,
                            "transition_width_km": width,
                            "spacing_km": None,
                            "coarse_spacing_km": coarse,
                            "fine_spacing_km": fine,
                            "sharp_travel_time_s": None,
                            "smooth_travel_time_s": None,
                            "smooth_minus_sharp_s": None,
                            "sharp_refinement_difference_s": abs(
                                sharp_times[coarse][pair_index] - sharp_times[fine][pair_index]
                            ),
                            "smooth_refinement_difference_s": abs(
                                smooth_coarse[pair_index] - smooth_fine[pair_index]
                            ),
                            "smooth_minus_sharp_refinement_difference_s": abs(
                                smooth_coarse[pair_index] - smooth_fine[pair_index]
                            )
                            - abs(sharp_times[coarse][pair_index] - sharp_times[fine][pair_index]),
                            "status": "refinement_difference",
                            "error": None,
                        }
                    )
    return rows


def _run_runtime_model(
    settings: BenchmarkSettings,
    *,
    spacings: Sequence[float],
    max_forward_nodes: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model = generate_layered_velocity_model(settings)
    stations = _runtime_stations(settings)
    sources = _runtime_sources(settings)
    pairs = tuple(
        ValidationPair(
            f"runtime_{source_index:02d}_{station_index:02d}",
            source,
            receiver,
        )
        for source_index, source in enumerate(sources)
        for station_index, receiver in enumerate(stations)
    )
    for spacing in spacings:
        process_rss_before_mb = _working_set_mb()
        grid_started = perf_counter()
        try:
            grid = build_cartesian_forward_grid(
                settings,
                model,
                scenario="block_anomaly",
                spacing_by_axis_km=(spacing, spacing, spacing),
                max_grid_nodes=max_forward_nodes,
            )
        except Exception as exc:
            rows.append(
                {
                    "spacing_km": spacing,
                    "grid_shape": None,
                    "grid_node_or_cell_count": None,
                    "grid_construction_s": perf_counter() - grid_started,
                    "solver_construction_s": None,
                    "one_source_16_receivers_s": None,
                    "twenty_four_source_384_receiver_s": None,
                    "process_rss_before_mb": process_rss_before_mb,
                    "process_rss_after_grid_mb": _working_set_mb(),
                    "process_rss_after_solver_mb": None,
                    "process_rss_after_travel_times_mb": None,
                    "projected_50_targets_one_geometry_h": None,
                    "projected_50_targets_two_geometries_h": None,
                    "projected_50_targets_five_geometries_h": None,
                    "projected_250_targets_one_geometry_h": None,
                    "projected_250_targets_two_geometries_h": None,
                    "projected_250_targets_five_geometries_h": None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        grid_construction_s = perf_counter() - grid_started
        process_rss_after_grid_mb = _working_set_mb()
        solver_started = perf_counter()
        try:
            solver = TtcrpyRectilinearForwardSolver.from_cartesian_grid(
                grid,
                TtcrpyGridConfiguration(method="FSM", tt_from_rp=True, maxit=100),
            )
        except Exception as exc:
            rows.append(
                {
                    "spacing_km": spacing,
                    "grid_shape": "x".join(
                        str(value)
                        for value in (
                            len(grid.x_coordinates_km),
                            len(grid.y_coordinates_km),
                            len(grid.z_coordinates_km),
                        )
                    ),
                    "grid_node_or_cell_count": len(grid.p_velocity_km_per_s),
                    "grid_construction_s": grid_construction_s,
                    "solver_construction_s": perf_counter() - solver_started,
                    "one_source_16_receivers_s": None,
                    "twenty_four_source_384_receiver_s": None,
                    "process_rss_before_mb": process_rss_before_mb,
                    "process_rss_after_grid_mb": process_rss_after_grid_mb,
                    "process_rss_after_solver_mb": _working_set_mb(),
                    "process_rss_after_travel_times_mb": None,
                    "projected_50_targets_one_geometry_h": None,
                    "projected_50_targets_two_geometries_h": None,
                    "projected_50_targets_five_geometries_h": None,
                    "projected_250_targets_one_geometry_h": None,
                    "projected_250_targets_two_geometries_h": None,
                    "projected_250_targets_five_geometries_h": None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            del grid
            gc.collect()
            continue
        solver_construction_s = perf_counter() - solver_started
        solve_start = perf_counter()
        one_source_result = _raytrace_pairs(
            solver,
            tuple(pairs[: len(stations)]),
        )
        one_source_s = perf_counter() - solve_start
        all_start = perf_counter()
        all_status = "ok"
        all_error = None
        for source_index, source in enumerate(sources):
            receiver_points = stations
            try:
                solver.raytrace(
                    [source],
                    receiver_points,
                    aggregate_src=True,
                    return_rays=False,
                )
            except Exception as exc:
                all_status = "error"
                all_error = f"{type(exc).__name__}: {exc}"
                break
        all_s = perf_counter() - all_start
        after_travel_times_mb = _working_set_mb()
        projection = {
            f"projected_{target_count}_targets_{geometry_count}_geometries_h": (
                all_s * target_count * geometry_count / 3600.0
            )
            for target_count in (50, 250)
            for geometry_count in (1, 2, 5)
        }
        rows.append(
            {
                "spacing_km": spacing,
                "grid_shape": "x".join(str(value) for value in solver.grid_shape),
                "grid_node_or_cell_count": int(np.prod(solver.grid_shape)),
                "grid_construction_s": grid_construction_s,
                "solver_construction_s": solver_construction_s,
                "one_source_16_receivers_s": one_source_s,
                "twenty_four_source_384_receiver_s": all_s,
                "process_rss_before_mb": process_rss_before_mb,
                "process_rss_after_grid_mb": process_rss_after_grid_mb,
                "process_rss_after_solver_mb": _working_set_mb(),
                "process_rss_after_travel_times_mb": after_travel_times_mb,
                "one_source_status": one_source_result.status,
                "status": all_status,
                "error": all_error or one_source_result.error,
                **projection,
            }
        )
        del solver, grid
        gc.collect()
    return rows


def build_v2_audit_summary(
    *,
    case_rows: Sequence[Mapping[str, object]],
    refinement_rows: Sequence[Mapping[str, object]],
    analytic_rows: Sequence[Mapping[str, object]],
    representation_rows: Sequence[Mapping[str, object]],
    fault_rows: Sequence[Mapping[str, object]],
    target_rows: Sequence[Mapping[str, object]],
    smooth_rows: Sequence[Mapping[str, object]],
    runtime_rows: Sequence[Mapping[str, object]],
    signal_epsilon_s: float,
    spacings: Sequence[float],
    difficult_spacing_km: float,
) -> dict[str, object]:
    return {
        "artifact_type": "authoritative_forward_solver_v2_audit_summary",
        "validation_id": VALIDATION_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": [
            "no Benchmark v2 target generation",
            "no ML fitting or rerun",
            "no manuscript modification",
            "FSM-only numerical-contract audit after the v1 NO-GO gate",
        ],
        "representations": list(REPRESENTATIONS),
        "regular_spacings_km": list(spacings),
        "difficult_spacing_km": difficult_spacing_km,
        "signal_epsilon_s": signal_epsilon_s,
        "error_budget": {
            "case_rows": len(case_rows),
            "refinement_rows": len(refinement_rows),
            "case_status_counts": _status_counts(case_rows),
            "refinement_by_family_representation": _group_stats(
                refinement_rows,
                ("family", "representation", "fine_spacing_km"),
                "absolute_refinement_difference_s",
            ),
            "numerical_error_over_signal_by_family_representation": _group_stats(
                [
                    row
                    for row in refinement_rows
                    if row.get("numerical_error_over_signal") is not None
                ],
                ("family", "representation", "fine_spacing_km"),
                "numerical_error_over_signal",
            ),
            "geological_signal_by_family_representation": _group_stats(
                case_rows,
                ("family", "representation", "spacing_km"),
                "absolute_delta_t_geology_s",
            ),
        },
        "analytic_representation": {
            "rows": len(analytic_rows),
            "by_family_representation_spacing": _group_stats(
                analytic_rows,
                ("family", "representation", "spacing_km"),
                "absolute_error_s",
            ),
        },
        "node_vs_cell": {
            "rows": len(representation_rows),
            "by_family_spacing": _group_stats(
                representation_rows,
                ("family", "spacing_km"),
                "absolute_cell_minus_node_s",
            ),
        },
        "fault_diagnostics": {
            "rows": len(fault_rows),
            "by_spacing": _group_stats(fault_rows, ("spacing_km",), "travel_time_s"),
            "topology_switch_rows": sum(
                _as_bool(row.get("path_topology_switch_across_resolutions"))
                for row in fault_rows
            ),
        },
        "target_grid_resolution": {
            "rows": len(target_rows),
            "configured_sampling": _group_stats(
                [row for row in target_rows if row.get("audit_kind") == "configured_structure_target_sampling"],
                ("option", "family"),
                "changed_node_count",
            ),
        },
        "smooth_interface": {
            "rows": len(smooth_rows),
            "refinement_rows": sum(row.get("status") == "refinement_difference" for row in smooth_rows),
            "by_family_width": _group_stats(
                [row for row in smooth_rows if row.get("status") != "refinement_difference"],
                ("family", "transition_width_km"),
                "smooth_minus_sharp_s",
            ),
        },
        "runtime": {
            "rows": len(runtime_rows),
            "status_counts": _status_counts(runtime_rows),
        },
    }


def build_v2_audit_summary_markdown(summary: Mapping[str, object]) -> str:
    error_budget = summary.get("error_budget", {})
    analytic = summary.get("analytic_representation", {})
    representation = summary.get("node_vs_cell", {})
    runtime = summary.get("runtime", {})
    lines = [
        "# Authoritative Forward-Solver v2 FSM Contract Audit",
        "",
        "This is a numerical-contract audit only. It does not generate Benchmark v2 targets, fit ML models, or modify the manuscript.",
        "",
        "## Scope",
        "",
        f"- Representations: `{summary.get('representations')}`.",
        f"- Regular spacings: `{summary.get('regular_spacings_km')}` km; difficult-pair spacing: `{summary.get('difficult_spacing_km')}` km.",
        f"- Geological-signal near-zero classification threshold used for ratio reporting: `{summary.get('signal_epsilon_s')}` s. This is a classification epsilon, not an acceptance criterion.",
        "",
        "## Error budget",
        "",
        f"- Heterogeneous case rows: `{error_budget.get('case_rows')}`; refinement rows: `{error_budget.get('refinement_rows')}`.",
        "- Refinement statistics are in `error_budget_refinement.csv`; geological signal is `delta_t_geology = t_heterogeneous - t_background` in `error_budget_case_metrics.csv`.",
        "- Ratios are omitted for essentially zero geological signal rather than interpreted as meaningful.",
        "",
        "## Analytical representation comparison",
        "",
        f"- Rows: `{analytic.get('rows')}`. Grouped statistics: `{analytic.get('by_family_representation_spacing')}`.",
        "- The layered reference includes the direct branch and admissible critical-interface head-wave candidates; these are analytical controls for the monotonic layered model, not a general heterogeneous reference.",
        "",
        "## Node versus cell",
        "",
        f"- Rows: `{representation.get('rows')}`. Grouped absolute cell-minus-node differences: `{representation.get('by_family_spacing')}`.",
        "- Cell values are sampled directly at cell centers and converted to slowness before `ttcrpy` receives them. They are not interpolated from node values.",
        "",
        "## Diagnostics and target representation",
        "",
        "- Fault path/slice outputs are in `fault_resolution_diagnostics.md`, `fault_ray_paths.csv`, and `fault_model_slices.csv`.",
        "- Target-grid sampling and minimum-dimension sweeps are in `target_grid_resolution_audit.csv` and `target_grid_recommendation.md`.",
        "- Sharp/smooth sensitivity is diagnostic only; smoothed models are not silently substituted for sharp benchmark geology.",
        "",
        "## Runtime",
        "",
        f"- Runtime rows: `{runtime.get('rows')}`; status counts: `{runtime.get('status_counts')}`. Detailed measurements are in `runtime_model.csv`.",
        "",
        "The scientific decision is issued separately in `solver_gate_v2.md` after the output distributions and independent FSM-only PyKonal cross-check are reviewed.",
        "",
    ]
    return "\n".join(lines)


def build_fault_diagnostics_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    """Describe the fault-resolution diagnostic without hiding failed pairs."""
    successful = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    switches = sorted(
        {
            str(row["pair_id"])
            for row in successful
            if _as_bool(row.get("path_topology_switch_across_resolutions"))
        }
    )
    crossing_counts = [
        int(row["fault_crossing_count"])
        for row in successful
        if row.get("fault_crossing_count") is not None
    ]
    lines = [
        "# Fault-resolution diagnostics",
        "",
        "This is a targeted FSM diagnostic for the sharp analytic fault model. It is not a solver-selection target and does not alter Benchmark v2 geology.",
        "",
        f"- Diagnostic rows: `{len(rows)}`; successful rows: `{len(successful)}`; explicit failures: `{len(failed)}`.",
        f"- Pair IDs with more than one sampled fault-side sign sequence across resolutions: `{switches or 'none observed'}`.",
        f"- Successful path crossing-count range: `{min(crossing_counts) if crossing_counts else None}` to `{max(crossing_counts) if crossing_counts else None}`.",
        "",
        "The complete travel-time/path records are in `fault_resolution_case_metrics.csv` and `fault_ray_paths.csv`; sampled y≈50 km slices are in `fault_model_slices.csv` and `fault_model_slices.png`.",
        "",
        (
            "Interpretation: no compressed path-topology switch was observed in this "
            "sample, although this does not establish topology stability for the full "
            "benchmark. Path changes near discontinuities remain a diagnostic risk."
            if not switches
            else "Interpretation: topology switches are retained as observations requiring "
            "scientific review. They may represent genuine changes between competing "
            "first-arrival paths near a discontinuity or grid/interface-assignment effects; "
            "they are not silently classified as solver failures."
        ),
        "",
    ]
    if failed:
        lines.extend(
            [
                "## Explicit failures",
                "",
                *[
                    f"- `{row.get('pair_id')}` at `{row.get('spacing_km')}` km: `{row.get('error')}`"
                    for row in failed
                ],
                "",
            ]
        )
    return "\n".join(lines)


def build_target_grid_recommendation_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    """Summarize target-grid sampling evidence and a bounded recommendation."""
    configured = [row for row in rows if row.get("audit_kind") == "configured_structure_target_sampling"]
    sweep = [row for row in rows if row.get("audit_kind") == "minimum_dimension_sweep"]
    first_supported: dict[str, float | None] = {}
    for family in ("block_anomaly", "salt_dome", "dyke_intrusion"):
        family_rows = [row for row in sweep if row.get("family") == family]
        values = sorted(
            {
                float(row["parameter_value"])
                for row in family_rows
                if row.get("parameter_value") is not None
            }
        )
        supported_values = [
            value
            for value in values
            if all(
                _as_bool(row.get("meets_two_lateral_intervals"))
                for row in family_rows
                if row.get("parameter_value") is not None
                and float(row["parameter_value"]) == value
            )
        ]
        first_supported[family] = supported_values[0] if supported_values else None
    lines = [
        "# Geological target-grid resolution assessment",
        "",
        "This audit samples the analytic structures directly on candidate ML target grids. It does not generate Benchmark v2 targets and does not select parameters using ML performance.",
        "",
        "## Candidate grids",
        "",
        *[
            f"- `{row.get('option')}`: `{row.get('grid_shape')}` nodes, `{row.get('target_node_count')}` total nodes."
            for row in configured
            if row.get("family") == "layered"
        ],
        "",
        "The two-interval criterion used in the dimension sweep means that the smallest changed lateral index span is at least two grid intervals at all tested center offsets. This is a representability diagnostic, not a claim that the body boundary is accurately reconstructed.",
        "",
        "## First dimension meeting the diagnostic criterion on the historical grid",
        "",
        *[
            f"- `{family}` `{value:g} km`"
            if value is not None
            else f"- `{family}`: no tested value met the criterion"
            for family, value in first_supported.items()
        ],
        "",
        "## Recommendation",
        "",
        "Retain the forward grid as an independent, finer grid. If the historical 5×5×2.5 km ML target is retained, restrict important lateral body dimensions using the measured family-specific minimums above and describe the resulting fields as coarse effective representations. The 3 km historical dyke should not be retained as a resolved body on that target grid.",
        "",
        "Recommendation for pilot planning: use the `option_b_2p5x2p5x2p5` target grid (21,853 nodes). It improves lateral sampling by a factor of two while retaining the existing 2.5 km depth interval and has about 3.8 times the historical target dimensionality. The 2.5×2.5×1.25 km option remains a sensitivity alternative, not the primary recommendation, because it increases target dimensionality to 42,025 nodes without being required by the lateral body-resolution diagnostic. This recommendation must be frozen before Benchmark v2 generation and must not be made from test performance.",
        "",
        "The detailed rows, including offsets, changed-node counts, and index spans, are in `target_grid_resolution_audit.csv`.",
        "",
    ]
    return "\n".join(lines)


def _override_structure(
    settings: BenchmarkSettings,
    family: str,
    value: float,
    center_x: float,
    center_y: float,
) -> BenchmarkSettings:
    generation = settings.velocity_model_generation
    if family == "block_anomaly":
        block = replace(
            generation.block_anomaly,
            center_km=(center_x, center_y, generation.block_anomaly.center_km[2]),
            size_km=(value, value, generation.block_anomaly.size_km[2]),
        )
        generation = replace(generation, block_anomaly=block)
    elif family == "salt_dome":
        salt = replace(
            generation.salt_dome,
            center_km=(center_x, center_y, generation.salt_dome.center_km[2]),
            radii_km=(value / 2.0, value / 2.0, generation.salt_dome.radii_km[2]),
        )
        generation = replace(generation, salt_dome=salt)
    elif family == "dyke_intrusion":
        dyke = replace(
            generation.dyke_intrusion,
            center_km=(center_x, center_y, generation.dyke_intrusion.center_km[2]),
            width_km=value,
        )
        generation = replace(generation, dyke_intrusion=dyke)
    else:
        raise ValueError(f"Unsupported dimension sweep family: {family}")
    return replace(settings, velocity_model_generation=generation)


def _smooth_velocity_values(
    settings: BenchmarkSettings,
    model: LayeredVelocityModel,
    family: str,
    x_coordinates: Sequence[float],
    y_coordinates: Sequence[float],
    z_coordinates: Sequence[float],
    transition_width_km: float,
) -> np.ndarray:
    x = np.asarray(x_coordinates, dtype=float)
    y = np.asarray(y_coordinates, dtype=float)
    z = np.asarray(z_coordinates, dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    boundaries = np.asarray([layer.top_depth_km for layer in model.layers] + [model.layers[-1].bottom_depth_km])
    velocities = np.asarray([layer.p_velocity_km_per_s for layer in model.layers], dtype=float)
    layer_index = np.searchsorted(boundaries[1:], zz, side="right")
    base = velocities[np.minimum(layer_index, len(velocities) - 1)]
    if family == "block_anomaly":
        anomaly = settings.velocity_model_generation.block_anomaly
        half = np.asarray(anomaly.size_km, dtype=float) / 2.0
        signed = np.minimum.reduce(
            (
                half[0] - np.abs(xx - anomaly.center_km[0]),
                half[1] - np.abs(yy - anomaly.center_km[1]),
                half[2] - np.abs(zz - anomaly.center_km[2]),
            )
        )
        weight = _smooth_step(signed, transition_width_km)
        return base + anomaly.velocity_delta_km_per_s * weight
    if family == "salt_dome":
        salt = settings.velocity_model_generation.salt_dome
        radii = np.asarray(salt.radii_km, dtype=float)
        normalized = np.sqrt(
            ((xx - salt.center_km[0]) / radii[0]) ** 2
            + ((yy - salt.center_km[1]) / radii[1]) ** 2
            + ((zz - salt.center_km[2]) / radii[2]) ** 2
        )
        signed = min(radii) * (1.0 - normalized)
        weight = _smooth_step(signed, transition_width_km)
        return base + weight * (salt.body_velocity_km_per_s - base)
    if family == "dyke_intrusion":
        dyke = settings.velocity_model_generation.dyke_intrusion
        strike = math.radians(dyke.strike_deg)
        along = (xx - dyke.center_km[0]) * math.cos(strike) + (yy - dyke.center_km[1]) * math.sin(strike)
        normal = -(xx - dyke.center_km[0]) * math.sin(strike) + (yy - dyke.center_km[1]) * math.cos(strike)
        signed = np.minimum.reduce(
            (
                dyke.length_km / 2.0 - np.abs(along),
                dyke.width_km / 2.0 - np.abs(normal),
                zz - dyke.top_depth_km,
                dyke.bottom_depth_km - zz,
            )
        )
        weight = _smooth_step(signed, transition_width_km)
        return base + weight * (dyke.body_velocity_km_per_s - base)
    if family == "faulted":
        faulted = settings.velocity_model_generation.faulted
        strike = math.radians(faulted.strike_deg)
        normal_x = math.sin(strike)
        normal_y = -math.cos(strike)
        horizontal_offset = np.zeros_like(zz)
        if not math.isclose(faulted.dip_deg, 90.0, rel_tol=0.0, abs_tol=1.0e-9):
            horizontal_offset = zz / math.tan(math.radians(faulted.dip_deg))
        signed = (
            (xx - faulted.fault_x_km) * normal_x
            + (yy - faulted.fault_y_km) * normal_y
            - (1.0 if faulted.dip_direction == "positive_normal" else -1.0) * horizontal_offset
        )
        if faulted.positive_side == "less_equal":
            signed = -signed
        weight = _smooth_step(signed, transition_width_km)
        return base + faulted.velocity_offset_km_per_s * weight
    return base


def _smooth_step(signed_distance_km: np.ndarray, width_km: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(signed_distance_km / (0.5 * width_km)))


def _fault_slice_rows(
    artifact: FsmFieldArtifact,
    faulted: Any,
    spacing_km: float,
) -> list[dict[str, object]]:
    values = artifact.velocity_values
    y_index = int(np.argmin(np.abs(np.asarray(artifact.y_coordinates_km) - 50.0)))
    rows: list[dict[str, object]] = []
    for iz, z in enumerate(artifact.z_coordinates_km):
        for ix, x in enumerate(artifact.x_coordinates_km):
            rows.append(
                {
                    "spacing_km": spacing_km,
                    "slice": "nearest_y_50km",
                    "x_km": x,
                    "y_km": artifact.y_coordinates_km[y_index],
                    "z_km": z,
                    "velocity_km_per_s": float(values[ix, y_index, iz]),
                    "fault_signed_offset_km": fault_plane_signed_offset_km(x, artifact.y_coordinates_km[y_index], z, faulted),
                }
            )
    return rows


def _fault_sign(value: float) -> str:
    if abs(value) < 1.0e-8:
        return "0"
    return "+" if value > 0.0 else "-"


def _crossing_count(signs: Sequence[str]) -> int:
    nonzero = [sign for sign in signs if sign != "0"]
    return sum(left != right for left, right in zip(nonzero, nonzero[1:]))


def _fault_topology_signature(signs: Sequence[str]) -> str:
    """Compress sampled signs so different path discretizations are comparable."""
    signature: list[str] = []
    for sign in signs:
        if sign == "0":
            continue
        if not signature or signature[-1] != sign:
            signature.append(sign)
    return "".join(signature)


def _runtime_stations(settings: BenchmarkSettings) -> tuple[Point3D, ...]:
    x_min, x_max = settings.study_area.station_x_bounds_km
    y_min, y_max = settings.study_area.station_y_bounds_km
    x_values = np.linspace(x_min, x_max, 4)
    y_values = np.linspace(y_min, y_max, 4)
    return tuple(
        Point3D(float(x), float(y), settings.station_generation.station_elevation_km)
        for y in y_values
        for x in x_values
    )


def _runtime_sources(settings: BenchmarkSettings) -> tuple[Point3D, ...]:
    rng = np.random.default_rng(DEFAULT_SEED + 901)
    x_min, x_max = settings.study_area.station_x_bounds_km
    y_min, y_max = settings.study_area.station_y_bounds_km
    z_min = max(settings.study_area.z_range_km[0] + settings.study_area.margin_km, 1.0)
    z_max = min(settings.study_area.z_range_km[1] - settings.study_area.margin_km, 29.0)
    return tuple(
        Point3D(float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)), float(rng.uniform(z_min, z_max)))
        for _ in range(24)
    )


def _working_set_mb() -> float | None:
    """Return current Windows working-set memory without an extra dependency."""
    if sys.platform != "win32":
        return None

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_memory_info.restype = ctypes.c_int
        succeeded = get_memory_info(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
    except Exception:
        return None
    if not succeeded:
        return None
    return float(counters.WorkingSetSize) / (1024.0 * 1024.0)


def _euclidean_time(first: Point3D, second: Point3D, velocity_km_per_s: float) -> float:
    return math.dist(
        (first.x_km, first.y_km, first.z_km),
        (second.x_km, second.y_km, second.z_km),
    ) / velocity_km_per_s


def _flatten_x_fastest(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values.transpose(2, 1, 0).reshape(-1))


def _group_stats(
    rows: Iterable[Mapping[str, object]],
    group_fields: Sequence[str],
    value_field: str,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        try:
            value = float(row[value_field])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        key = tuple(str(row.get(field)) for field in group_fields)
        grouped[key].append(value)
    output: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        output.append(
            {
                **{field: value for field, value in zip(group_fields, key)},
                "field": value_field,
                **_stats(values),
            }
        )
    return output


def _stats(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "maximum": None, "std": None}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(max(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _status_counts(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("status", "missing"))] += 1
    return dict(sorted(counts.items()))


def _as_bool(value: object) -> bool:
    """Parse booleans from both in-memory rows and CSV round-trips."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _audit_metadata(
    settings: BenchmarkSettings,
    *,
    spacings: Sequence[float],
    difficult_spacing_km: float,
    pairs_per_family: int,
    difficult_pairs_per_family: int,
    analytic_fine_pairs: int,
    signal_epsilon_s: float,
    transition_widths_km: Sequence[float],
    runtime_spacings_km: Sequence[float],
    max_forward_nodes: int,
) -> dict[str, object]:
    package_names = ("ttcrpy", "numpy", "scipy", "vtk")
    versions: dict[str, str | None] = {}
    from importlib import metadata

    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "artifact_type": "authoritative_forward_solver_v2_audit_metadata",
        "validation_id": VALIDATION_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": "python -m tomobench.evaluation.authoritative_forward_solver_v2",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "seed": DEFAULT_SEED,
        "regular_spacings_km": list(spacings),
        "difficult_spacing_km": difficult_spacing_km,
        "pairs_per_family": pairs_per_family,
        "difficult_pairs_per_family": difficult_pairs_per_family,
        "analytic_fine_pairs": analytic_fine_pairs,
        "signal_epsilon_s": signal_epsilon_s,
        "transition_widths_km": list(transition_widths_km),
        "runtime_spacings_km": list(runtime_spacings_km),
        "max_forward_nodes": max_forward_nodes,
        "scope_exclusions": [
            "no Benchmark v2 target generation",
            "no ML rerun",
            "no manuscript modification",
            "no legacy-solver selection or tuning",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_error_budget_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    values_by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("absolute_refinement_difference_s")
        if value is None:
            continue
        values_by_family[str(row["family"])].append(float(value))
    if not values_by_family:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.boxplot(
        [values_by_family[key] for key in sorted(values_by_family)],
        showmeans=True,
    )
    axis.set_xticks(range(1, len(values_by_family) + 1))
    axis.set_xticklabels(sorted(values_by_family))
    axis.set_ylabel("Absolute refinement difference (s)")
    axis.set_title("FSM refinement differences by geological family")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_fault_slice_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    spacings = sorted({float(row["spacing_km"]) for row in rows}, reverse=True)
    if not spacings:
        return
    figure, axes = plt.subplots(1, len(spacings), figsize=(5 * len(spacings), 4), squeeze=False)
    for axis, spacing in zip(axes[0], spacings):
        selected = [row for row in rows if float(row["spacing_km"]) == spacing]
        x_values = sorted({float(row["x_km"]) for row in selected})
        z_values = sorted({float(row["z_km"]) for row in selected})
        grid = np.full((len(z_values), len(x_values)), np.nan)
        x_index = {value: index for index, value in enumerate(x_values)}
        z_index = {value: index for index, value in enumerate(z_values)}
        for row in selected:
            grid[z_index[float(row["z_km"])], x_index[float(row["x_km"])]] = float(row["velocity_km_per_s"])
        image = axis.imshow(
            grid,
            extent=(min(x_values), max(x_values), max(z_values), min(z_values)),
            aspect="auto",
            cmap="viridis",
        )
        axis.set_title(f"Fault slice, {spacing:g} km")
        axis.set_xlabel("x (km)")
        axis.set_ylabel("z (km)")
        figure.colorbar(image, ax=axis, label="Velocity (km/s)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _validate_spacings(values: Sequence[float]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if len(parsed) < 2 or any(value <= 0.0 for value in parsed) or len(set(parsed)) != len(parsed):
        raise ValueError("spacings_km must contain at least two unique positive values.")
    return parsed


def _validate_positive_int(value: int, label: str) -> None:
    if int(value) < 1:
        raise ValueError(f"{label} must be positive.")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else get_repo_root() / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairs-per-family", type=int, default=8)
    parser.add_argument("--difficult-pairs-per-family", type=int, default=2)
    parser.add_argument("--analytic-fine-pairs", type=int, default=3)
    parser.add_argument("--difficult-spacing-km", type=float, default=DEFAULT_DIFFICULT_SPACING_KM)
    parser.add_argument("--signal-epsilon-s", type=float, default=DEFAULT_SIGNAL_EPSILON_S)
    parser.add_argument(
        "--spacing-km",
        type=float,
        nargs="+",
        default=list(DEFAULT_SPACINGS_KM),
    )
    parser.add_argument(
        "--runtime-spacing-km",
        type=float,
        nargs="+",
        default=[1.25, 0.625],
    )
    parser.add_argument("--max-forward-nodes", type=int, default=DEFAULT_MAX_FORWARD_NODES_V2)
    parser.add_argument("--skip-difficult", action="store_true")
    parser.add_argument("--skip-fault-diagnostics", action="store_true")
    parser.add_argument("--skip-smooth-sensitivity", action="store_true")
    parser.add_argument("--skip-runtime-model", action="store_true")
    args = parser.parse_args(argv)
    output = run_authoritative_forward_solver_v2_audit(
        output_dir=args.output_dir,
        spacings_km=args.spacing_km,
        difficult_spacing_km=args.difficult_spacing_km,
        pairs_per_family=args.pairs_per_family,
        difficult_pairs_per_family=args.difficult_pairs_per_family,
        analytic_fine_pairs=args.analytic_fine_pairs,
        signal_epsilon_s=args.signal_epsilon_s,
        runtime_spacings_km=args.runtime_spacing_km,
        max_forward_nodes=args.max_forward_nodes,
        include_difficult=not args.skip_difficult,
        run_fault_diagnostics=not args.skip_fault_diagnostics,
        run_smooth_sensitivity=not args.skip_smooth_sensitivity,
        run_runtime_model=not args.skip_runtime_model,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
