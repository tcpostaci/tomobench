"""Robustness analyses for the frozen Benchmark v2 evidence.

This module is intentionally an evidence consumer.  It does not regenerate the
250-target corpus, alter the frozen split, or select a model from test results.
The only forward solves performed here are the pre-specified reference-model
FSM checks needed to audit the fixed-ray operator contract.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tomobench.config import load_settings
from tomobench.config.settings import FaultedGridSettings
from tomobench.domain.schemas import CartesianVelocityGrid3D
from tomobench.evaluation.benchmark_v2_final_analysis import (
    BOOTSTRAP_RESAMPLES,
    FEATURE_NAMES,
    ML_METHODS,
    PCG_MAX_ITERATIONS,
    PCG_TOLERANCE,
    TIE_TOLERANCE,
    ProductionCase,
    RayGeometry,
    _prediction_metrics_from_arrays,
    _read_json,
    _stable_seed,
    _solve_pcg,
    build_feature_matrices,
    build_reference_ray_geometry,
    build_structure_masks,
    cell_centered_values_from_node_vector,
    fit_observation_pca_model,
    fit_target_pca,
    load_production_cases,
    ray_matvec,
    solve_fixed_ray_case,
)
from tomobench.generation.velocity_grids import fault_plane_signed_offset_km
from tomobench.tomography.reference import build_layered_reference_velocity_grid
from tomobench.simulation.ttcrpy_forward import (
    TtcrpyGridConfiguration,
    TtcrpyRectilinearForwardSolver,
)
from tomobench.utils.paths import get_repo_root


HARDENING_RELATIVE = Path("outputs/generated/submission_readiness/robustness_checks")
PAPER_HARDENING_RELATIVE = Path("paper/robustness_checks")
PRODUCTION_RELATIVE = Path(
    "outputs/generated/submission_readiness/benchmark_v2_production_250_v1"
)
EVIDENCE_RELATIVE = Path(
    "outputs/generated/submission_readiness/benchmark_v2_final_evidence_v1"
)
TEST_METHODS = (*ML_METHODS, "reference_prior", "fixed_ray")
PRIMARY_COMPARISONS = (
    ("realistic_vs_fixed_ray", "realistic_full", "fixed_ray"),
    ("realistic_vs_travel_time_only", "realistic_full", "travel_time_only"),
    ("realistic_vs_training_target_mean", "realistic_full", "training_target_mean"),
    ("realistic_vs_no_travel_time", "realistic_full", "no_travel_time"),
    ("realistic_vs_shuffled_travel_time", "realistic_full", "shuffled_travel_time"),
    ("realistic_vs_reference_prior", "realistic_full", "reference_prior"),
)
FAMILY_ORDER = ("layered", "block_anomaly", "faulted", "salt_dome", "dyke_intrusion")
NOISE_LEVELS = (0.0, 0.025, 0.05, 0.1)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty schema to {path}.")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps(_json_default(value), separators=(",", ":"))
    return value


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}g}"


def _stats(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
    }


def _bootstrap_mean_ci(values: Sequence[float], seed: int, repetitions: int = 10_000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(repetitions, array.size), replace=True), axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _family_stratified_ci(rows: Sequence[Mapping[str, Any]], seed: int) -> tuple[float, float]:
    grouped: dict[str, np.ndarray] = {}
    for family in FAMILY_ORDER:
        values = [float(row["delta_rmse_km_per_s"]) for row in rows if row["family"] == family]
        if values:
            grouped[family] = np.asarray(values, dtype=float)
    if not grouped:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        blocks = [rng.choice(values, size=values.size, replace=True) for values in grouped.values()]
        samples.append(np.concatenate(blocks))
    means = np.asarray([np.mean(sample) for sample in samples])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _node_shape(case: ProductionCase) -> tuple[int, int, int]:
    return (
        len(case.target_grid.x_coordinates_km),
        len(case.target_grid.y_coordinates_km),
        len(case.target_grid.z_coordinates_km),
    )


def _cell_shape(case: ProductionCase) -> tuple[int, int, int]:
    nx, ny, nz = _node_shape(case)
    return nx - 1, ny - 1, nz - 1


def _case_target_cells(case: ProductionCase) -> np.ndarray:
    return cell_centered_values_from_node_vector(case.target_vector, _node_shape(case))


def _family_parameters(case: ProductionCase) -> dict[str, Any]:
    sampled = case.physical_parameters.get("sampled_family_parameters", {})
    parameters = sampled.get(case.family)
    if not isinstance(parameters, dict):
        raise ValueError(f"{case.target_id}: missing sampled parameters for {case.family}.")
    return dict(parameters)


def direct_analytic_cell_truth(case: ProductionCase) -> np.ndarray:
    """Evaluate the saved analytic construction at target-cell centers.

    This is intentionally separate from node-to-cell averaging.  The exact
    parameter registry and the target-grid layered metadata are the only inputs.
    """

    x_axis = np.asarray(case.target_grid.x_coordinates_km, dtype=float)
    y_axis = np.asarray(case.target_grid.y_coordinates_km, dtype=float)
    z_axis = np.asarray(case.target_grid.z_coordinates_km, dtype=float)
    x = (x_axis[:-1] + x_axis[1:]) / 2.0
    y = (y_axis[:-1] + y_axis[1:]) / 2.0
    z = (z_axis[:-1] + z_axis[1:]) / 2.0
    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    x_grid = xx.transpose(2, 0, 1)
    y_grid = yy.transpose(2, 0, 1)
    z_grid = zz.transpose(2, 0, 1)
    layered = case.target_grid.metadata.get("layered_model", {})
    boundaries = np.asarray(layered.get("depth_boundaries_km", []), dtype=float)
    velocities = np.asarray(layered.get("velocities_km_per_s", []), dtype=float)
    if boundaries.size != velocities.size + 1:
        raise ValueError(f"{case.target_id}: invalid layered metadata for direct cell truth.")
    layer_index = np.searchsorted(boundaries[1:-1], z_grid, side="right")
    truth = velocities[layer_index].astype(float, copy=True)
    if case.family == "layered":
        return truth.reshape(-1)

    parameters = _family_parameters(case)
    if case.family == "block_anomaly":
        center = np.asarray(parameters["center_km"], dtype=float)
        size = np.asarray(parameters["size_km"], dtype=float)
        body = (
            (np.abs(x_grid - center[0]) <= size[0] / 2.0)
            & (np.abs(y_grid - center[1]) <= size[1] / 2.0)
            & (np.abs(z_grid - center[2]) <= size[2] / 2.0)
        )
        truth[body] += float(parameters["velocity_delta_km_per_s"])
    elif case.family == "salt_dome":
        center = np.asarray(parameters["center_km"], dtype=float)
        radii = np.asarray(parameters["radii_km"], dtype=float)
        body = (
            ((x_grid - center[0]) / radii[0]) ** 2
            + ((y_grid - center[1]) / radii[1]) ** 2
            + ((z_grid - center[2]) / radii[2]) ** 2
            <= 1.0
        )
        truth[body] = float(parameters["body_velocity_km_per_s"])
    elif case.family == "dyke_intrusion":
        center = np.asarray(parameters["center_km"], dtype=float)
        strike = np.deg2rad(float(parameters["strike_deg"]))
        along = np.abs((x_grid - center[0]) * np.cos(strike) + (y_grid - center[1]) * np.sin(strike))
        normal = np.abs(-(x_grid - center[0]) * np.sin(strike) + (y_grid - center[1]) * np.cos(strike))
        body = (
            (along <= float(parameters["length_km"]) / 2.0)
            & (normal <= float(parameters["width_km"]) / 2.0)
            & (z_grid >= float(parameters["top_depth_km"]))
            & (z_grid <= float(parameters["bottom_depth_km"]))
        )
        truth[body] = float(parameters["body_velocity_km_per_s"])
    elif case.family == "faulted":
        fault = FaultedGridSettings(
            fault_x_km=float(parameters["fault_x_km"]),
            fault_y_km=float(parameters["fault_y_km"]),
            strike_deg=float(parameters["strike_deg"]),
            dip_deg=float(parameters["dip_deg"]),
            dip_direction=str(parameters["dip_direction"]),
            positive_side=str(parameters["positive_side"]),
            velocity_offset_km_per_s=float(parameters["velocity_offset_km_per_s"]),
        )
        signed = np.asarray(
            [
                fault_plane_signed_offset_km(float(xv), float(yv), float(zv), fault)
                for xv, yv, zv in zip(x_grid.flat, y_grid.flat, z_grid.flat, strict=True)
            ]
        ).reshape(x_grid.shape)
        positive = signed >= 0.0 if fault.positive_side == "greater_equal" else signed <= 0.0
        truth[positive] += fault.velocity_offset_km_per_s
    else:
        raise ValueError(f"Unsupported analytic family: {case.family}.")
    return truth.reshape(-1)


def _load_final_test_artifacts(
    cases: Sequence[ProductionCase], evidence_dir: Path
) -> dict[str, dict[str, np.ndarray]]:
    test_cases = [case for case in cases if case.split == "test"]
    test_ids = [case.target_id for case in test_cases]
    result: dict[str, dict[str, np.ndarray]] = {method: {} for method in TEST_METHODS}
    for method in ML_METHODS:
        path = evidence_dir / "ml" / f"{method}_test_predictions.npz"
        with np.load(path) as artifact:
            ids = [str(value) for value in artifact["target_ids"].tolist()]
            values = np.asarray(artifact["predicted_node_velocity_km_per_s"], dtype=float)
        if ids != test_ids:
            raise ValueError(f"{method}: final prediction target ordering disagrees with frozen test set.")
        for target_id, value in zip(ids, values, strict=True):
            result[method][target_id] = value
    for case in test_cases:
        path = evidence_dir / "classical" / "predictions" / f"{case.target_id}.npz"
        with np.load(path) as artifact:
            result["reference_prior"][case.target_id] = np.asarray(artifact["reference_cells"], dtype=float)
            result["fixed_ray"][case.target_id] = np.asarray(artifact["fixed_cells"], dtype=float)
            result.setdefault("coverage", {})[case.target_id] = np.asarray(artifact["coverage_km"], dtype=float)
    return result


def _load_all_observation_rows(production_dir: Path, case: ProductionCase) -> list[dict[str, str]]:
    return _read_csv(production_dir / "observations" / f"{case.target_id}_observations.csv")


def _load_common_background_grid(production_dir: Path) -> CartesianVelocityGrid3D:
    payload = _read_json(production_dir / "forward_grids" / "common_layered_background.json")
    with np.load(production_dir / "forward_grids" / "common_layered_background.npz") as artifact:
        x = tuple(float(value) for value in artifact["x_km"].tolist())
        y = tuple(float(value) for value in artifact["y_km"].tolist())
        z = tuple(float(value) for value in artifact["z_km"].tolist())
        values = tuple(float(value) for value in artifact["p_velocity_km_per_s"].tolist())
    return CartesianVelocityGrid3D(
        grid_id="common_layered_background_forward_grid",
        source_velocity_model_id="common_layered_background",
        x_coordinates_km=x,
        y_coordinates_km=y,
        z_coordinates_km=z,
        p_velocity_km_per_s=values,
        metadata=dict(payload["grid_metadata"]),
    )


def _load_geometry(path: Path) -> RayGeometry:
    with np.load(path) as artifact:
        row_ptr = np.asarray(artifact["row_ptr"], dtype=np.int64)
        cell_indices = np.asarray(artifact["cell_indices"], dtype=np.int64)
        path_lengths = np.asarray(artifact["path_lengths_km"], dtype=float)
        coverage = np.asarray(artifact["coverage_km"], dtype=float)
    return RayGeometry(
        row_ptr=row_ptr,
        cell_indices=cell_indices,
        path_lengths_km=path_lengths,
        coverage_km=coverage,
        converged_count=384,
        observation_count=384,
        runtime_s=math.nan,
    )


def _reference_cells(case: ProductionCase, settings: Any) -> np.ndarray:
    reference = build_layered_reference_velocity_grid(case.target_grid, settings)
    return cell_centered_values_from_node_vector(np.asarray(reference.p_velocity_km_per_s), _node_shape(case))


def run_operator_consistency(
    production_dir: Path,
    output_dir: Path,
    *,
    cases: Sequence[ProductionCase] | None = None,
    settings: Any | None = None,
    force: bool = False,
) -> Path:
    """Audit FSM reference times against the fixed-ray ``G_ref s0`` operator."""

    cases = list(cases or load_production_cases(production_dir))
    settings = settings or load_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "operator_consistency_arrays.npz"
    target_ids = [case.target_id for case in cases]
    if cache_path.exists() and not force:
        with np.load(cache_path) as artifact:
            cached_ids = [str(value) for value in artifact["target_ids"].tolist()]
        if cached_ids == target_ids:
            return cache_path

    geometry_dir = output_dir / "reference_ray_geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    common_grid = _load_common_background_grid(production_dir)
    solver_config = TtcrpyGridConfiguration(
        method="FSM",
        cell_slowness=False,
        tt_from_rp=False,
        interp_vel=False,
        eps=1.0e-5,
        maxit=100,
        weno=True,
        nsnx=5,
        nsny=5,
        nsnz=5,
        n_secondary=2,
        n_tertiary=2,
        radius_factor_tertiary=3.0,
        translate_grid=False,
        n_threads=1,
    )
    solver = TtcrpyRectilinearForwardSolver.from_cartesian_grid(common_grid, solver_config)
    fsm_times = np.empty((len(cases), 384), dtype=float)
    ray_times = np.empty_like(fsm_times)
    observation_times = np.empty_like(fsm_times)
    source_depths = np.empty_like(fsm_times)
    offsets = np.empty_like(fsm_times)
    case_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        observations = case.observations
        observation_times[case_index] = observations[:, 7]
        source_depths[case_index] = observations[:, 2]
        offsets[case_index] = np.linalg.norm(observations[:, 0:3] - observations[:, 3:6], axis=1)
        raw_rows = _load_all_observation_rows(production_dir, case)
        background_by_id = {
            str(row["observation_id"]): float(row["background_travel_time_s"])
            for row in raw_rows
            if row.get("background_travel_time_s") not in (None, "")
        }
        if case.family != "layered" and len(background_by_id) == 384:
            fsm = np.asarray([background_by_id[obs_id] for obs_id in case.observation_ids], dtype=float)
            fsm_source = "frozen production background_travel_time_s from common FSM solver"
        else:
            fsm = np.empty(384, dtype=float)
            for start in range(0, 384, 16):
                result = solver.raytrace(
                    observations[start : start + 1, 0:3],
                    observations[start : start + 16, 3:6],
                    aggregate_src=True,
                )
                fsm[start : start + 16] = np.asarray(result.travel_times_s, dtype=float)
            fsm_source = "recomputed common-background FSM reference time"
        fsm_times[case_index] = fsm

        geometry_path = geometry_dir / f"{case.target_id}.npz"
        if geometry_path.exists() and not force:
            geometry = _load_geometry(geometry_path)
        else:
            geometry = build_reference_ray_geometry(case, settings)
            np.savez_compressed(
                geometry_path,
                row_ptr=geometry.row_ptr,
                cell_indices=geometry.cell_indices,
                path_lengths_km=geometry.path_lengths_km,
                coverage_km=geometry.coverage_km,
            )
        reference_cells = _reference_cells(case, settings)
        ray = ray_matvec(geometry, 1.0 / reference_cells)
        ray_times[case_index] = ray
        delta = fsm - ray
        residual = observations[:, 7] - fsm
        abs_delta = np.abs(delta)
        abs_time = np.abs(observations[:, 7])
        abs_residual = np.abs(residual)
        for obs_index, observation_id in enumerate(case.observation_ids):
            row = {
                "target_id": case.target_id,
                "family": case.family,
                "split": case.split,
                "observation_id": observation_id,
                "observation_index": obs_index,
                "source_depth_km": observations[obs_index, 2],
                "source_receiver_offset_km": offsets[case_index, obs_index],
                "observed_time_s": observations[obs_index, 7],
                "t_fsm_0_s": fsm[obs_index],
                "t_ray_0_s": ray[obs_index],
                "operator_difference_s": delta[obs_index],
                "abs_operator_difference_s": abs_delta[obs_index],
                "r_fsm_s": residual[obs_index],
                "abs_observed_time_s": abs_time[obs_index],
                "abs_r_fsm_s": abs_residual[obs_index],
                "operator_over_abs_observed": _safe_ratio(abs_delta[obs_index], abs_time[obs_index]),
                "operator_over_abs_r_fsm": _safe_ratio(abs_delta[obs_index], abs_residual[obs_index]),
                "operator_over_noise_0p025": abs_delta[obs_index] / 0.025,
                "operator_over_noise_0p05": abs_delta[obs_index] / 0.05,
                "operator_over_noise_0p10": abs_delta[obs_index] / 0.10,
                "fsm_source": fsm_source,
            }
            observation_rows.append(row)
        case_rows.append(
            {
                "target_id": case.target_id,
                "family": case.family,
                "split": case.split,
                "fsm_source": fsm_source,
                "operator_mean_signed_s": float(np.mean(delta)),
                "operator_mae_s": float(np.mean(abs_delta)),
                "operator_rmse_s": float(np.sqrt(np.mean(delta**2))),
                "operator_median_abs_s": float(np.median(abs_delta)),
                "operator_p90_abs_s": float(np.percentile(abs_delta, 90)),
                "operator_p95_abs_s": float(np.percentile(abs_delta, 95)),
                "operator_p99_abs_s": float(np.percentile(abs_delta, 99)),
                "operator_max_abs_s": float(np.max(abs_delta)),
                "median_abs_observed_s": float(np.median(abs_time)),
                "median_abs_r_fsm_s": float(np.median(abs_residual)),
                "operator_over_median_abs_observed": _safe_ratio(np.median(abs_delta), np.median(abs_time)),
                "operator_over_median_abs_r_fsm": _safe_ratio(np.median(abs_delta), np.median(abs_residual)),
                "positive_coverage_cells": int(np.count_nonzero(geometry.coverage_km > 0.0)),
            }
        )
    np.savez_compressed(
        cache_path,
        target_ids=np.asarray(target_ids),
        fsm_times_s=fsm_times,
        ray_times_s=ray_times,
        observation_times_s=observation_times,
        source_depths_km=source_depths,
        offsets_km=offsets,
    )
    _write_csv(output_dir / "operator_consistency_observation_metrics.csv", observation_rows)
    _write_csv(output_dir / "operator_consistency_target_metrics.csv", case_rows)
    summary_rows = _group_operator_rows(observation_rows)
    _write_csv(output_dir / "operator_consistency_group_summary.csv", summary_rows)
    _write_json(
        {
            "artifact_type": "robustness_checks_fixed_ray_operator_consistency",
            "target_count": len(cases),
            "observation_count": len(cases) * 384,
            "fsm_contract": solver_config.constructor_kwargs(),
            "fsm_reference_source": "stored common-background production values for non-layered cases; recomputed common-background values for layered cases whose production background column was target-equal",
            "ray_operator_source": "reference-model pseudo-bending rays discretized into 40x40x12 target cells",
            "reference_model_is_independent_of_case_geology": True,
            "denominator_epsilon_s": 1.0e-6,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        },
        output_dir / "operator_consistency_metadata.json",
    )
    return cache_path


def _safe_ratio(numerator: float, denominator: float, epsilon: float = 1.0e-6) -> float | None:
    if abs(denominator) <= epsilon:
        return None
    return float(numerator / denominator)


def _group_operator_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[("family", str(row["family"]))].append(row)
        depth = float(row["source_depth_km"])
        offset = float(row["source_receiver_offset_km"])
        depth_bin = "0-10 km" if depth < 10 else "10-20 km" if depth < 20 else "20-30 km"
        offset_bin = "0-30 km" if offset < 30 else "30-60 km" if offset < 60 else "60-90 km" if offset < 90 else ">=90 km"
        groups[("source_depth", depth_bin)].append(row)
        groups[("offset", offset_bin)].append(row)
        groups[("all", "all")].append(row)
    output: list[dict[str, Any]] = []
    for (group_type, group_label), group in sorted(groups.items()):
        differences = np.asarray([float(row["operator_difference_s"]) for row in group])
        absolute = np.abs(differences)
        output.append(
            {
                "group_type": group_type,
                "group_label": group_label,
                "n": len(group),
                "mean_signed_difference_s": float(np.mean(differences)),
                "mae_s": float(np.mean(absolute)),
                "rmse_s": float(np.sqrt(np.mean(differences**2))),
                "median_abs_s": float(np.median(absolute)),
                "p90_abs_s": float(np.percentile(absolute, 90)),
                "p95_abs_s": float(np.percentile(absolute, 95)),
                "p99_abs_s": float(np.percentile(absolute, 99)),
                "max_abs_s": float(np.max(absolute)),
                "median_abs_r_fsm_s": float(np.median([float(row["abs_r_fsm_s"]) for row in group])),
            }
        )
    return output


def _acquisition_stats(production_dir: Path, cases: Sequence[ProductionCase]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    for case in cases:
        stations_payload = _read_json(production_dir / "acquisitions" / f"{case.acquisition_id}_stations.json")
        earthquakes_payload = _read_json(production_dir / "acquisitions" / f"{case.acquisition_id}_earthquakes.json")
        stations = np.asarray(
            [[item["location"][axis] for axis in ("x_km", "y_km", "z_km")] for item in stations_payload["stations"]]
        )
        earthquakes = np.asarray(
            [[item["hypocenter"][axis] for axis in ("x_km", "y_km", "z_km")] for item in earthquakes_payload["earthquakes"]]
        )
        station_distances = _pairwise_distances(stations)
        source_distances = _pairwise_distances(earthquakes)
        offsets = np.linalg.norm(case.observations[:, 0:3] - case.observations[:, 3:6], axis=1)
        times = case.observations[:, 7]
        case_rows.append(
            {
                "target_id": case.target_id,
                "family": case.family,
                "split": case.split,
                "station_count": len(stations),
                "earthquake_count": len(earthquakes),
                "station_x_min_km": np.min(stations[:, 0]),
                "station_x_max_km": np.max(stations[:, 0]),
                "station_y_min_km": np.min(stations[:, 1]),
                "station_y_max_km": np.max(stations[:, 1]),
                "station_z_unique": json.dumps(sorted(set(stations[:, 2].tolist()))),
                "source_x_min_km": np.min(earthquakes[:, 0]),
                "source_x_max_km": np.max(earthquakes[:, 0]),
                "source_y_min_km": np.min(earthquakes[:, 1]),
                "source_y_max_km": np.max(earthquakes[:, 1]),
                "source_z_min_km": np.min(earthquakes[:, 2]),
                "source_z_max_km": np.max(earthquakes[:, 2]),
                "minimum_station_pair_distance_km": np.min(station_distances) if station_distances.size else None,
                "minimum_source_pair_distance_km": np.min(source_distances) if source_distances.size else None,
                "offset_min_km": np.min(offsets),
                "offset_p05_km": np.percentile(offsets, 5),
                "offset_median_km": np.median(offsets),
                "offset_p95_km": np.percentile(offsets, 95),
                "offset_max_km": np.max(offsets),
                "travel_time_min_s": np.min(times),
                "travel_time_mean_s": np.mean(times),
                "travel_time_median_s": np.median(times),
                "travel_time_p95_s": np.percentile(times, 95),
                "travel_time_max_s": np.max(times),
            }
        )
        for row_index, (offset, time) in enumerate(zip(offsets, times, strict=True)):
            observation_rows.append(
                {
                    "target_id": case.target_id,
                    "family": case.family,
                    "split": case.split,
                    "observation_index": row_index,
                    "offset_km": offset,
                    "travel_time_s": time,
                }
            )
    return case_rows, observation_rows


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.empty(0, dtype=float)
    distances = points[:, None, :] - points[None, :, :]
    upper = np.triu_indices(len(points), k=1)
    return np.linalg.norm(distances, axis=2)[upper]


def write_acquisition_report(
    production_dir: Path, paper_dir: Path, output_dir: Path, cases: Sequence[ProductionCase]
) -> None:
    case_rows, observation_rows = _acquisition_stats(production_dir, cases)
    _write_csv(output_dir / "acquisition_case_metrics.csv", case_rows)
    _write_csv(output_dir / "acquisition_observation_metrics.csv", observation_rows)
    all_offsets = np.asarray([float(row["offset_km"]) for row in observation_rows])
    all_times = np.asarray([float(row["travel_time_s"]) for row in observation_rows])
    _write_json(
        {
            "all_production_observation_count": len(observation_rows),
            "all_production_target_count": len(cases),
            "stations": {
                "count": 16,
                "distribution": "uniform random x/y",
                "allowable_range_km": {"x": [5.0, 95.0], "y": [5.0, 95.0], "z": [0.0, 0.0]},
                "independent_per_target": True,
                "minimum_separation_enforced": False,
            },
            "sources": {
                "count": 24,
                "distribution": "uniform random x/y/z hypocenters",
                "allowable_range_km": {"x": [5.0, 95.0], "y": [5.0, 95.0], "z": [5.0, 25.0]},
                "configured_depth_range_km": [1.0, 29.0],
                "minimum_source_source_or_source_station_separation_enforced": False,
            },
            "offset_statistics_km": _stats(all_offsets),
            "travel_time_statistics_s": _stats(all_times),
            "geometry_seeds_independent_of_geology": True,
        },
        output_dir / "acquisition_geometry_summary.json",
    )
    _write_acquisition_figure(production_dir, paper_dir / "figures", cases[0])
    summary = _read_json(output_dir / "acquisition_geometry_summary.json")
    _write_markdown(
        paper_dir / "acquisition_geometry_report.md",
        f"""# Acquisition geometry report

## Scope

This report audits the frozen production acquisition files. It does not create a new
experiment. There are 16 stations, 24 earthquake sources, and one randomized acquisition
realization per target, giving 384 observations per target and 96,000 production observations.
Station and source seeds are independent of geological-model seeds.

## Stations

Stations are drawn independently from a uniform lateral distribution over x=[5,95] km and
y=[5,95] km, with z=0 km. The 5-km margin is explicit in every station artifact. No minimum
station separation is enforced; the observed minimum-pair distribution is retained in
`acquisition_case_metrics.csv`.

## Sources

Sources are drawn independently from uniform x/y/z distributions over x=[5,95] km,
y=[5,95] km, and z=[5,25] km. The configured study-area depth range is [1,29] km, but
the production hypocenter generator uses the stricter [5,25] km interval. No minimum
source-source or source-station separation constraint is enforced.

## Production distributions

Across all production observations, source-receiver Euclidean offset is min/ p05/ median /
p95 / max = **{_fmt(summary['offset_statistics_km']['min'])} / {_fmt(summary['offset_statistics_km']['p05'])} /
{_fmt(summary['offset_statistics_km']['median'])} / {_fmt(summary['offset_statistics_km']['p95'])} /
{_fmt(summary['offset_statistics_km']['max'])} km**. Travel time is min/mean/median/p95/max =
**{_fmt(summary['travel_time_statistics_s']['min'])} / {_fmt(summary['travel_time_statistics_s']['mean'])} /
{_fmt(summary['travel_time_statistics_s']['median'])} / {_fmt(summary['travel_time_statistics_s']['p95'])} /
{_fmt(summary['travel_time_statistics_s']['max'])} s**.

The complete per-case and per-observation records are in the hardening output directory. The
schematic is a geometry description only; it is not an inversion result.
""",
    )


def _write_acquisition_figure(production_dir: Path, figure_dir: Path, case: ProductionCase) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stations_payload = _read_json(production_dir / "acquisitions" / f"{case.acquisition_id}_stations.json")
    earthquakes_payload = _read_json(production_dir / "acquisitions" / f"{case.acquisition_id}_earthquakes.json")
    stations = np.asarray([[item["location"][axis] for axis in ("x_km", "y_km", "z_km")] for item in stations_payload["stations"]])
    sources = np.asarray([[item["hypocenter"][axis] for axis in ("x_km", "y_km", "z_km")] for item in earthquakes_payload["earthquakes"]])
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(stations[:, 0], stations[:, 1], marker="^", s=42, label="Stations", color="#1f77b4")
    axes[0].scatter(sources[:, 0], sources[:, 1], marker=".", s=42, label="Sources", color="#d62728")
    axes[0].set(xlim=(0, 100), ylim=(0, 100), xlabel="x (km)", ylabel="y (km)", title="Plan view")
    axes[0].set_aspect("equal")
    axes[0].legend(frameon=False, loc="best")
    axes[1].scatter(stations[:, 0], stations[:, 2], marker="^", s=42, label="Stations", color="#1f77b4")
    axes[1].scatter(sources[:, 0], sources[:, 2], marker=".", s=42, label="Sources", color="#d62728")
    axes[1].set(xlim=(0, 100), ylim=(30, 0), xlabel="x (km)", ylabel="depth z (km)", title="x–depth view")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"Canonical acquisition geometry: {case.target_id}")
    for extension in ("png", "svg"):
        fig.savefig(figure_dir / f"acquisition_geometry.{extension}", dpi=240 if extension == "png" else None)
    plt.close(fig)


def write_reference_provenance(paper_dir: Path, production_dir: Path, settings: Any) -> None:
    del settings
    _read_json(production_dir / "benchmark_v2_production_config.json")
    _write_markdown(
        paper_dir / "reference_model_provenance.md",
        """# 1-D reference-model provenance

## Recovered model

The fixed-ray comparator uses the independently configured common layered P-wave reference:

| layer | depth interval (km) | velocity (km/s) |
|---:|---:|---:|
| 1 | 0–5 | 4.0 |
| 2 | 5–12 | 5.0 |
| 3 | 12–18 | 6.0 |
| 4 | 18–24 | 6.8 |
| 5 | 24–30 | 7.4 |

The reference is sampled at the 41×41×13 target-grid nodes for the fixed-ray path construction;
the path segments are then accumulated in the 40×40×12 target cells. The separate FSM operator
audit samples the same reference profile at the 81×81×25, 1.25-km forward grid.

## Provenance and information boundary

The model is recorded in `config/benchmark_config.yaml`, the production configuration, and the
saved common-background forward-grid metadata. The production scope records that ML fitting was
not performed during corpus construction and that the test set was created and frozen before
model selection. The generator stores the common background in each target's physical-parameter
record. The fixed-ray implementation itself calls the central layered reference settings; it does
not read target velocities or family-specific anomaly parameters.

The reference therefore uses a preconfigured layered background, not a profile fitted from the
training targets, validation targets, or test targets. It does not use the target-specific sampled
layer velocities of the layered family, nor any block, fault, salt, or dyke parameters. The
reference prior is consequently a deliberately strong but explicit background comparator, not an
estimated prior.

## What is and is not recoverable

The exact numerical definition and its presence in the central configuration are recoverable.
The artifact timestamps show that the common background and production corpus were materialized
on 2026-08-29 before this hardening pass. A separate source-control event proving the original
authoring time of every configuration line is not present in the current evidence bundle; that
historical timestamp is therefore not claimed. No training-only fairness sensitivity is mandatory
because the recovered prior is independent of the complete target distribution and contains no
validation/test-specific fitted information. It remains a visible sensitivity/comparator rather
than a claim of operational prior availability.
""",
    )


def write_target_space_report(
    cases: Sequence[ProductionCase], paper_dir: Path, output_dir: Path
) -> None:
    cell_targets = {case.target_id: _case_target_cells(case) for case in cases}
    node_targets = {case.target_id: np.asarray(case.target_vector, dtype=float) for case in cases}
    rows: list[dict[str, Any]] = []
    for case in cases:
        within = [other for other in cases if other.target_id != case.target_id and other.family == case.family]
        if within:
            cell_distances = [_distance_metrics(cell_targets[case.target_id], cell_targets[other.target_id]) for other in within]
            node_distances = [_distance_metrics(node_targets[case.target_id], node_targets[other.target_id]) for other in within]
            nearest_cell = min(cell_distances, key=lambda value: value["rmse"])
            nearest_node = min(node_distances, key=lambda value: value["rmse"])
        else:
            nearest_cell = nearest_node = {"rmse": None, "mae": None, "max_abs": None}
        train = [other for other in cases if other.split == "train"]
        cell_cross = [_distance_metrics(cell_targets[case.target_id], cell_targets[other.target_id]) for other in train]
        node_cross = [_distance_metrics(node_targets[case.target_id], node_targets[other.target_id]) for other in train]
        nearest_cell_cross = min(cell_cross, key=lambda value: value["rmse"])
        nearest_node_cross = min(node_cross, key=lambda value: value["rmse"])
        nearest_cross_case = min(
            train,
            key=lambda other: _distance_metrics(cell_targets[case.target_id], cell_targets[other.target_id])["rmse"],
        )
        target_std = float(np.std(cell_targets[case.target_id]))
        rows.append(
            {
                "target_id": case.target_id,
                "family": case.family,
                "split": case.split,
                "within_family_nearest_cell_rmse_km_per_s": nearest_cell["rmse"],
                "within_family_nearest_cell_mae_km_per_s": nearest_cell["mae"],
                "within_family_nearest_cell_max_abs_km_per_s": nearest_cell["max_abs"],
                "within_family_nearest_node_rmse_km_per_s": nearest_node["rmse"],
                "within_family_nearest_node_mae_km_per_s": nearest_node["mae"],
                "within_family_nearest_node_max_abs_km_per_s": nearest_node["max_abs"],
                "closest_train_target_id": nearest_cross_case.target_id if case.split != "train" else "NA",
                "closest_train_target_family": nearest_cross_case.family if case.split != "train" else "NA",
                "closest_train_cell_rmse_km_per_s": nearest_cell_cross["rmse"] if case.split != "train" else None,
                "closest_train_cell_mae_km_per_s": nearest_cell_cross["mae"] if case.split != "train" else None,
                "closest_train_cell_max_abs_km_per_s": nearest_cell_cross["max_abs"] if case.split != "train" else None,
                "closest_train_node_rmse_km_per_s": nearest_node_cross["rmse"] if case.split != "train" else None,
                "closest_train_node_mae_km_per_s": nearest_node_cross["mae"] if case.split != "train" else None,
                "closest_train_node_max_abs_km_per_s": nearest_node_cross["max_abs"] if case.split != "train" else None,
                "target_standard_deviation_km_per_s": target_std,
                "within_family_nearest_cell_rmse_over_target_std": _safe_ratio(nearest_cell["rmse"] or math.nan, target_std),
                "configured_velocity_range_normalized_cell_rmse": _safe_ratio(nearest_cell["rmse"] or math.nan, 5.0),
            }
        )
    _write_csv(output_dir / "target_space_proximity.csv", rows)
    cross_rows = [row for row in rows if row["split"] in {"validation", "test"}]
    field_groups = {
        "within_family_cell_rmse": "within_family_nearest_cell_rmse_km_per_s",
        "within_family_node_rmse": "within_family_nearest_node_rmse_km_per_s",
        "within_family_cell_mae": "within_family_nearest_cell_mae_km_per_s",
        "within_family_node_mae": "within_family_nearest_node_mae_km_per_s",
        "within_family_cell_max": "within_family_nearest_cell_max_abs_km_per_s",
        "within_family_node_max": "within_family_nearest_node_max_abs_km_per_s",
        "cross_split_cell_rmse": "closest_train_cell_rmse_km_per_s",
        "cross_split_node_rmse": "closest_train_node_rmse_km_per_s",
        "cross_split_cell_mae": "closest_train_cell_mae_km_per_s",
        "cross_split_node_mae": "closest_train_node_mae_km_per_s",
        "cross_split_cell_max": "closest_train_cell_max_abs_km_per_s",
        "cross_split_node_max": "closest_train_node_max_abs_km_per_s",
    }
    summary_rows: list[dict[str, Any]] = []
    for scope, values_field in field_groups.items():
        for group_label, group in [("all", rows if scope.startswith("within") else cross_rows), *[(family, [r for r in rows if r["family"] == family and (scope.startswith("within") or r["split"] in {"validation", "test"})]) for family in FAMILY_ORDER]]:
            values = [float(row[values_field]) for row in group if row.get(values_field) not in (None, "")]
            summary = _stats(values)
            summary.update({"scope": scope, "group": group_label})
            summary_rows.append(summary)
    _write_csv(output_dir / "target_space_proximity_summary.csv", summary_rows)
    train_targets = np.vstack([case.target_vector for case in cases if case.split == "train"])
    pca = fit_target_pca(train_targets)
    centered = train_targets
    centered = centered - np.mean(centered, axis=0)
    _, all_singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * np.finfo(float).eps * float(all_singular_values[0])
    _write_json(
        {
            "centered_training_matrix_shape": list(centered.shape),
            "singular_values": all_singular_values.tolist(),
            "singular_value_count_retained": int(np.count_nonzero(all_singular_values > tolerance)),
            "singular_value_count_computed": len(all_singular_values),
            "singular_value_smallest_retained": float(all_singular_values[np.count_nonzero(all_singular_values > tolerance) - 1]),
            "numerical_rank": pca.rank,
            "svd_tolerance": tolerance,
            "rank_criterion": "singular_value > max(matrix_shape)*machine_epsilon*largest_singular_value",
            "rank_interpretation": "numerical rank after centering; not an exact algebraic claim about an uncentered target family",
        },
        output_dir / "pca_rank_diagnostic.json",
    )
    _write_csv(
        output_dir / "pca_singular_values.csv",
        [{"component": index + 1, "singular_value": value, "above_tolerance": bool(value > tolerance)} for index, value in enumerate(all_singular_values)],
    )
    _write_markdown(
        paper_dir / "target_space_proximity_report.md",
        f"""# Target-space proximity and split wording

The frozen corpus contains 250 target arrays with unique SHA-256 hashes and a target-atomic
175/37/38 split. Exact hash uniqueness rules out exact duplicate target arrays across splits; it
does not imply that nearby geological fields are absent.

The hardening calculation compares the harmonized 40×40×12 cell vectors descriptively. For each
target it records its closest same-family target using RMSE, MAE, and maximum absolute difference.
For validation and test targets it also records the closest training target and its family. The
complete rows and family summaries are in `target_space_proximity.csv` and
`target_space_proximity_summary.csv`; no proximity threshold was used to remove targets or alter
the split.

The centered training matrix has shape 175×21853 and numerical rank **{pca.rank}**. The SVD
tolerance is **{_fmt(tolerance, 8)}**, using the explicit criterion
`max(shape) * eps * largest_singular_value`; the singular values are retained in
`pca_singular_values.csv`. The rank is therefore a numerical tolerance result after centering,
not evidence of exact algebraic degeneracy.

Manuscript wording should be: **“a target-atomic split with no exact target-hash overlap across
train, validation, and test sets.”** The proximity numbers are a diversity diagnostic, not a
post-hoc safety threshold or a claim of independence from all forms of target-space similarity.
""",
    )


def _distance_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    difference = np.asarray(first) - np.asarray(second)
    return {
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae": float(np.mean(np.abs(difference))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def write_coverage_report(
    cases: Sequence[ProductionCase], artifacts: Mapping[str, Mapping[str, np.ndarray]], paper_dir: Path, output_dir: Path
) -> None:
    test_cases = [case for case in cases if case.split == "test"]
    rows: list[dict[str, Any]] = []
    methods = ("realistic_full", "fixed_ray", "reference_prior", "training_target_mean")
    for case in test_cases:
        target = _case_target_cells(case)
        coverage = artifacts["coverage"][case.target_id]
        for threshold in (0.0, 0.1, 1.0, 5.0, 10.0):
            mask = coverage > 0.0 if threshold == 0.0 else coverage >= threshold
            for method in methods:
                if method == "realistic_full":
                    prediction = cell_centered_values_from_node_vector(artifacts[method][case.target_id], _node_shape(case))
                elif method == "training_target_mean":
                    prediction = cell_centered_values_from_node_vector(artifacts[method][case.target_id], _node_shape(case))
                else:
                    prediction = artifacts[method][case.target_id]
                metrics = _prediction_metrics_from_arrays(target[mask], prediction[mask]) if np.any(mask) else {"rmse": None, "mae": None, "bias": None}
                rows.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "method_id": method,
                        "coverage_threshold_label": ">0" if threshold == 0.0 else str(threshold),
                        "coverage_threshold_km": threshold,
                        "cell_count": int(np.count_nonzero(mask)),
                        "total_cell_count": int(coverage.size),
                        "coverage_fraction": float(np.mean(mask)),
                        "rmse_km_per_s": metrics["rmse"],
                        "mae_km_per_s": metrics["mae"],
                        "bias_km_per_s": metrics["bias"],
                    }
                )
    _write_csv(output_dir / "coverage_domain_metrics.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for threshold in (0.0, 0.1, 1.0, 5.0, 10.0):
        eligible = [row for row in rows if float(row["coverage_threshold_km"]) == threshold and row["method_id"] == "realistic_full"]
        fractions = np.asarray([float(row["coverage_fraction"]) for row in eligible])
        usable = [row for row in eligible if int(row["cell_count"]) >= 192]
        for method in methods:
            method_rows = [row for row in rows if float(row["coverage_threshold_km"]) == threshold and row["method_id"] == method]
            metric_values = [float(row["rmse_km_per_s"]) for row in method_rows if row["rmse_km_per_s"] not in (None, "")]
            summary_rows.append(
                {
                    "coverage_threshold_label": ">0" if threshold == 0.0 else str(threshold),
                    "coverage_threshold_km": threshold,
                    "method_id": method,
                    "test_target_count": len(eligible),
                    "metric_contributing_target_count": len(metric_values),
                    "total_qualifying_cells_across_targets": int(sum(int(row["cell_count"]) for row in eligible)),
                    "fraction_of_all_cells_across_targets": float(sum(int(row["cell_count"]) for row in eligible) / (len(eligible) * 19200)),
                    "mean_qualifying_cell_fraction": float(np.mean(fractions)),
                    "median_qualifying_cell_fraction": float(np.median(fractions)),
                    "p05_qualifying_cell_fraction": float(np.percentile(fractions, 5)),
                    "p95_qualifying_cell_fraction": float(np.percentile(fractions, 95)),
                    "targets_with_at_least_one_cell": int(sum(int(row["cell_count"]) > 0 for row in eligible)),
                    "targets_with_usable_cells_ge_1pct": len(usable),
                    "rmse_mean_km_per_s": float(np.mean(metric_values)) if metric_values else None,
                    "rmse_median_km_per_s": float(np.median(metric_values)) if metric_values else None,
                }
            )
    _write_csv(output_dir / "coverage_domain_summary.csv", summary_rows)
    family_rows: list[dict[str, Any]] = []
    for threshold in (0.0, 0.1, 1.0, 5.0, 10.0):
        for family in FAMILY_ORDER:
            selected = [row for row in rows if float(row["coverage_threshold_km"]) == threshold and row["method_id"] == "realistic_full" and row["family"] == family]
            fractions = [float(row["coverage_fraction"]) for row in selected]
            family_rows.append(
                {
                    "coverage_threshold_label": ">0" if threshold == 0.0 else str(threshold),
                    "coverage_threshold_km": threshold,
                    "family": family,
                    "test_target_count": len(selected),
                    "targets_with_at_least_one_cell": sum(int(row["cell_count"]) > 0 for row in selected),
                    "targets_with_usable_cells_ge_1pct": sum(int(row["cell_count"]) >= 192 for row in selected),
                    "mean_fraction": float(np.mean(fractions)) if fractions else None,
                    "median_fraction": float(np.median(fractions)) if fractions else None,
                }
            )
    _write_csv(output_dir / "coverage_family_summary.csv", family_rows)
    depth_rows: list[dict[str, Any]] = []
    for case in test_cases:
        coverage = artifacts["coverage"][case.target_id].reshape((_cell_shape(case)[2], _cell_shape(case)[1], _cell_shape(case)[0]))
        for iz in range(coverage.shape[0]):
            for threshold in (0.0, 0.1, 1.0, 5.0, 10.0):
                mask = coverage[iz] > 0 if threshold == 0.0 else coverage[iz] >= threshold
                depth_rows.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "depth_index": iz,
                        "top_depth_km": case.target_grid.z_coordinates_km[iz],
                        "bottom_depth_km": case.target_grid.z_coordinates_km[iz + 1],
                        "coverage_threshold_label": ">0" if threshold == 0.0 else str(threshold),
                        "coverage_threshold_km": threshold,
                        "qualifying_cell_count": int(np.count_nonzero(mask)),
                        "depth_cell_count": int(mask.size),
                        "qualifying_fraction": float(np.mean(mask)),
                    }
                )
    _write_csv(output_dir / "coverage_depth_distribution.csv", depth_rows)
    # Paired target deltas are reported only where both methods have a non-empty domain.
    paired_rows: list[dict[str, Any]] = []
    for threshold in (0.0, 0.1, 1.0, 5.0, 10.0):
        for comparison_id, first, second in (("realistic_vs_fixed_ray", "realistic_full", "fixed_ray"),):
            first_rows = {row["target_id"]: row for row in rows if float(row["coverage_threshold_km"]) == threshold and row["method_id"] == first}
            second_rows = {row["target_id"]: row for row in rows if float(row["coverage_threshold_km"]) == threshold and row["method_id"] == second}
            for target_id in sorted(first_rows):
                left = first_rows[target_id]
                right = second_rows[target_id]
                if left["rmse_km_per_s"] in (None, "") or right["rmse_km_per_s"] in (None, ""):
                    continue
                paired_rows.append(
                    {
                        "comparison_id": comparison_id,
                        "target_id": target_id,
                        "family": left["family"],
                        "coverage_threshold_label": ">0" if threshold == 0.0 else str(threshold),
                        "coverage_threshold_km": threshold,
                        "delta_rmse_km_per_s": float(left["rmse_km_per_s"]) - float(right["rmse_km_per_s"]),
                    }
                )
    _write_csv(output_dir / "coverage_paired_deltas.csv", paired_rows)
    paired_summary: list[dict[str, Any]] = []
    for threshold in (0.0, 0.1, 1.0, 5.0, 10.0):
        selected = [row for row in paired_rows if float(row["coverage_threshold_km"]) == threshold]
        deltas = np.asarray([float(row["delta_rmse_km_per_s"]) for row in selected])
        ci = _bootstrap_mean_ci(deltas, _stable_seed(f"hardening-coverage-{threshold}")) if len(deltas) else (math.nan, math.nan)
        paired_summary.append(
            {
                "coverage_threshold_label": ">0" if threshold == 0.0 else str(threshold),
                "coverage_threshold_km": threshold,
                "test_target_count": len(test_cases),
                "contributing_target_count": len(selected),
                "delta_definition": "realistic_full RMSE minus fixed_ray RMSE; negative favors realistic_full",
                "delta_mean_km_per_s": float(np.mean(deltas)) if len(deltas) else None,
                "delta_median_km_per_s": float(np.median(deltas)) if len(deltas) else None,
                "ci95_lower_km_per_s": ci[0],
                "ci95_upper_km_per_s": ci[1],
                "wins": int(np.count_nonzero(deltas < -TIE_TOLERANCE)),
                "losses": int(np.count_nonzero(deltas > TIE_TOLERANCE)),
                "ties": int(np.count_nonzero(np.abs(deltas) <= TIE_TOLERANCE)),
            }
        )
    _write_csv(output_dir / "coverage_paired_summary.csv", paired_summary)
    _write_json(
        {
            "coverage_source": "accumulated reference-model fixed-ray path length",
            "thresholds_reported": [">0", 0.1, 1.0, 5.0, 10.0],
            "positive_coverage_rule": "coverage_km > 0.0",
            "scientifically_usable_cell_rule": "at least 1% of 19,200 target cells (192 cells), declared as a reporting convention and never used for model selection",
            "zero_cell_policy": "retain all 38 test targets in domain summaries; metric contributing counts are reported separately and empty-domain RMSE is NA",
            "independent_diagnostic_status": "not available from the frozen production corpus: FSM labels were generated without validated path sidecars, and the existing pseudo-bending paths are reference-model diagnostics rather than true-model first-arrival paths",
        },
        output_dir / "coverage_domain_metadata.json",
    )
    ten = next(row for row in paired_summary if row["coverage_threshold_km"] == 10.0)
    _write_markdown(
        paper_dir / "coverage_domain_report.md",
        f"""# Coverage-domain report

Coverage is reported as a **reference-model illumination diagnostic**: accumulated path length
from the fixed rays traced through the configured 1-D prior. It is not ground-truth sensitivity
and was not supplied as an ML feature.

The report uses thresholds >0, 0.1, 1, 5, and 10 km. The >0 rule is strictly `coverage_km > 0`;
the other rules are inclusive `coverage_km >= threshold`. All 38 test targets remain in the
denominators. When a target has no qualifying cells, its method/domain RMSE is NA and the number
of contributing targets is shown explicitly. A reporting-only “usable” domain is defined before
interpretation as at least 192 cells (1% of 19,200); this is not a model-selection criterion.

The complete total, fraction, family, and depth distributions are in the CSV files beside this
report. At the pre-specified 10-km accumulated reference-ray path-length threshold, the paired
realistic-full minus fixed-ray RMSE summary has **{ten['contributing_target_count']}** contributing
targets, mean delta **{_fmt(ten['delta_mean_km_per_s'])} km/s**, and percentile-bootstrap 95% CI
**[{_fmt(ten['ci95_lower_km_per_s'])}, {_fmt(ten['ci95_upper_km_per_s'])}] km/s**. This result
must be read together with the retained domain size and the fact that the domain comes from the
reference comparator.

An independent true-model/FSM path-coverage sensitivity is not claimed. The production FSM labels
did not retain validated ray-path sidecars, while the repository's existing pseudo-bending paths
are generated through the reference model and therefore cannot be relabeled as true-model first
arrivals without changing the path contract. The manuscript should call the result reference-ray
illumination, not “coverage of the true model.”
""",
    )


def write_structure_report(
    cases: Sequence[ProductionCase], artifacts: Mapping[str, Mapping[str, np.ndarray]], paper_dir: Path, output_dir: Path
) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.split != "test":
            continue
        target = _case_target_cells(case)
        predictions: dict[str, np.ndarray] = {
            "realistic_full": cell_centered_values_from_node_vector(artifacts["realistic_full"][case.target_id], _node_shape(case)),
            "travel_time_only": cell_centered_values_from_node_vector(artifacts["travel_time_only"][case.target_id], _node_shape(case)),
            "fixed_ray": artifacts["fixed_ray"][case.target_id],
            "reference_prior": artifacts["reference_prior"][case.target_id],
            "training_target_mean": cell_centered_values_from_node_vector(artifacts["training_target_mean"][case.target_id], _node_shape(case)),
        }
        masks = build_structure_masks(case)
        for method, prediction in predictions.items():
            for scope, mask_map in masks.items():
                for label, mask in mask_map.items():
                    metrics = _prediction_metrics_from_arrays(target[mask], prediction[mask]) if np.any(mask) else {"rmse": None, "mae": None, "bias": None, "true_mean": None, "predicted_mean": None}
                    rows.append(
                        {
                            "target_id": case.target_id,
                            "family": case.family,
                            "method_id": method,
                            "metric_scope": scope,
                            "mask_label": label,
                            "cell_count": int(np.count_nonzero(mask)),
                            "rmse_km_per_s": metrics["rmse"],
                            "mae_km_per_s": metrics["mae"],
                            "bias_km_per_s": metrics["bias"],
                            "true_mean_km_per_s": metrics.get("true_mean"),
                            "predicted_mean_km_per_s": metrics.get("predicted_mean"),
                        }
                    )
                    if scope == "layer":
                        rows[-1]["layer_bias_km_per_s"] = metrics.get("bias")
            if case.family in {"block_anomaly", "salt_dome", "dyke_intrusion"}:
                body = masks["structure"]["body"]
                background = masks["structure"]["background"]
                true_contrast = float(np.mean(target[body]) - np.mean(target[background]))
                predicted_contrast = float(np.mean(prediction[body]) - np.mean(prediction[background]))
                rows.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "method_id": method,
                        "metric_scope": "contrast",
                        "mask_label": "body_minus_background",
                        "cell_count": int(np.count_nonzero(body)),
                        "true_contrast_km_per_s": true_contrast,
                        "predicted_contrast_km_per_s": predicted_contrast,
                        "contrast_error_km_per_s": predicted_contrast - true_contrast,
                    }
                )
            if case.family == "faulted":
                positive = masks["fault"]["positive_side"]
                negative = masks["fault"]["negative_side"]
                true_contrast = float(np.mean(target[positive]) - np.mean(target[negative]))
                predicted_contrast = float(np.mean(prediction[positive]) - np.mean(prediction[negative]))
                rows.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "method_id": method,
                        "metric_scope": "contrast",
                        "mask_label": "positive_minus_negative_side",
                        "cell_count": int(np.count_nonzero(positive)),
                        "true_contrast_km_per_s": true_contrast,
                        "predicted_contrast_km_per_s": predicted_contrast,
                        "contrast_error_km_per_s": predicted_contrast - true_contrast,
                    }
                )
    _write_csv(output_dir / "structure_method_metrics.csv", rows)
    summary_rows = _aggregate_structure_rows(rows)
    _write_csv(output_dir / "structure_method_summary.csv", summary_rows)
    _write_markdown(
        paper_dir / "structure_method_comparison.md",
        """# Structure-specific method comparison

The same generator-defined target-cell masks are applied to all reported methods: realistic/full
PCA-ridge, travel-time-only PCA-ridge, reference-model fixed-ray inversion, the 1-D reference
prior, and the training-target mean. No predicted localization detector or method-specific mask
was introduced. Metrics are computed within each target and then summarized over targets, so the
384 observations are not treated as independent structural replicates.

For block, dyke, and salt families, the table reports body/background RMSE and body-minus-background
contrast recovery. For faulted targets it reports positive/negative-side RMSE and side contrast.
For layered targets it reports layer-wise RMSE and bias. The complete target-level table and the
compact aggregated table are `structure_method_metrics.csv` and `structure_method_summary.csv`.
""",
    )


def _aggregate_structure_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row["method_id"]), str(row["metric_scope"]), str(row["mask_label"]))].append(row)
    output: list[dict[str, Any]] = []
    for (family, method, scope, label), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "family": family,
            "method_id": method,
            "metric_scope": scope,
            "mask_label": label,
            "target_count": len({str(row["target_id"]) for row in group}),
            "mean_cell_count": float(np.mean([int(row["cell_count"]) for row in group])),
        }
        for field in ("rmse_km_per_s", "mae_km_per_s", "bias_km_per_s", "true_mean_km_per_s", "predicted_mean_km_per_s", "true_contrast_km_per_s", "predicted_contrast_km_per_s", "contrast_error_km_per_s"):
            values = [float(row[field]) for row in group if row.get(field) not in (None, "")]
            item[f"mean_{field}"] = float(np.mean(values)) if values else None
            item[f"median_{field}"] = float(np.median(values)) if values else None
        output.append(item)
    return output


def write_statistical_report(evidence_dir: Path, paper_dir: Path, output_dir: Path) -> None:
    rows = _read_csv(evidence_dir / "statistics" / "target_level_paired_deltas.csv")
    selected = [row for row in rows if row["metric_domain"] == "cell_all"]
    output_rows: list[dict[str, Any]] = []
    for comparison_id, first, second in PRIMARY_COMPARISONS:
        group = [row for row in selected if row["comparison_id"] == comparison_id]
        if not group:
            continue
        deltas = np.asarray([float(row["delta_rmse_km_per_s"]) for row in group])
        pooled = _bootstrap_mean_ci(deltas, _stable_seed(f"hardening-pooled-{comparison_id}"))
        stratified = _family_stratified_ci(group, _stable_seed(f"hardening-stratified-{comparison_id}"))
        output_rows.append(
            {
                "comparison_id": comparison_id,
                "first_method_id": first,
                "second_method_id": second,
                "metric_domain": "cell_all",
                "test_target_count": len(group),
                "mean_delta_km_per_s": np.mean(deltas),
                "median_delta_km_per_s": np.median(deltas),
                "pooled_ci95_lower_km_per_s": pooled[0],
                "pooled_ci95_upper_km_per_s": pooled[1],
                "family_stratified_ci95_lower_km_per_s": stratified[0],
                "family_stratified_ci95_upper_km_per_s": stratified[1],
                "wins": int(np.count_nonzero(deltas < -TIE_TOLERANCE)),
                "losses": int(np.count_nonzero(deltas > TIE_TOLERANCE)),
                "ties": int(np.count_nonzero(np.abs(deltas) <= TIE_TOLERANCE)),
                "win_fraction_excluding_ties": _safe_ratio(np.count_nonzero(deltas < -TIE_TOLERANCE), np.count_nonzero(np.abs(deltas) > TIE_TOLERANCE), 0.0),
                "endpoint_role": "primary" if comparison_id == "realistic_vs_fixed_ray" else "secondary hypothesis-driven",
            }
        )
    travel_vs_fixed = [row for row in selected if row["comparison_id"] == "travel_time_only_vs_fixed_ray"]
    if travel_vs_fixed:
        deltas = np.asarray([float(row["delta_rmse_km_per_s"]) for row in travel_vs_fixed])
        output_rows.append(
            {
                "comparison_id": "travel_time_only_vs_fixed_ray",
                "first_method_id": "travel_time_only",
                "second_method_id": "fixed_ray",
                "metric_domain": "cell_all",
                "test_target_count": len(deltas),
                "mean_delta_km_per_s": np.mean(deltas),
                "median_delta_km_per_s": np.median(deltas),
                "pooled_ci95_lower_km_per_s": _bootstrap_mean_ci(deltas, _stable_seed("hardening-pooled-travel-fixed"))[0],
                "pooled_ci95_upper_km_per_s": _bootstrap_mean_ci(deltas, _stable_seed("hardening-pooled-travel-fixed"))[1],
                "family_stratified_ci95_lower_km_per_s": _family_stratified_ci(travel_vs_fixed, _stable_seed("hardening-stratified-travel-fixed"))[0],
                "family_stratified_ci95_upper_km_per_s": _family_stratified_ci(travel_vs_fixed, _stable_seed("hardening-stratified-travel-fixed"))[1],
                "wins": int(np.count_nonzero(deltas < -TIE_TOLERANCE)),
                "losses": int(np.count_nonzero(deltas > TIE_TOLERANCE)),
                "ties": int(np.count_nonzero(np.abs(deltas) <= TIE_TOLERANCE)),
                "endpoint_role": "secondary hypothesis-driven; sign diagnostic retained because mean and wins can differ",
            }
        )
    _write_csv(output_dir / "family_stratified_paired_bootstrap.csv", output_rows)
    _write_markdown(
        paper_dir / "statistical_sensitivity_report.md",
        """# Statistical sensitivity report

The predeclared primary endpoint is the full-input selected PCA-ridge estimator versus the
validation-selected reference-model fixed-ray regularized perturbation inversion on all-cell RMSE
over the 38 held-out geological targets. All other comparisons are secondary, hypothesis-driven
controls, or exploratory diagnostics.

The existing pooled paired bootstrap is retained. This report adds a family-stratified paired
bootstrap that resamples targets with replacement within each of the five test families and then
combines the blocks using their observed composition. It also reports mean/median deltas and
wins/losses/ties. Negative deltas favor the first method. The complete table is
`family_stratified_paired_bootstrap.csv`.

The training-target mean is an explicit no-observation comparison, not a feature-disruption
control. The shuffled and no-time controls answer different questions from the prior comparison.
The sign counts are robustness diagnostics, not replacements for effect sizes or confidence
intervals. Secondary comparisons are not mechanically multiplicity-adjusted because they were
defined as distinct controls and diagnostics rather than a single family of confirmatory claims;
the primary endpoint remains singular.
""",
    )


def _predict_details(model: Any, feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(feature_matrix, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    standardized = (x - model.feature_mean) / model.feature_std
    coefficients = model.coefficient_mean + (standardized - model.standardized_feature_mean) @ model.coefficient_weights
    prediction = model.pca_mean + coefficients @ model.pca_components
    return np.clip(prediction, model.velocity_bounds[0], model.velocity_bounds[1]), coefficients


def _full_feature_from_times(case: ProductionCase, times: np.ndarray) -> np.ndarray:
    observations = case.observations.copy()
    observations[:, 7] = times
    return observations.reshape(1, -1)


def write_travel_time_report(
    cases: Sequence[ProductionCase], artifacts: Mapping[str, Mapping[str, np.ndarray]], settings: Any, production_dir: Path, evidence_dir: Path, output_dir: Path, paper_dir: Path, operator_arrays: Path | None = None
) -> None:
    train = [case for case in cases if case.split == "train"]
    test = [case for case in cases if case.split == "test"]
    target_train = np.vstack([case.target_vector for case in train])
    pca = fit_target_pca(target_train)
    full_features = build_feature_matrices(cases)["realistic_full"]
    velocity_bounds = tuple(float(value) for value in settings.velocity_model_generation.velocity_bounds_km_per_s)
    train_indices = [index for index, case in enumerate(cases) if case.split == "train"]
    model = fit_observation_pca_model(full_features[train_indices], target_train, pca, 4, "pca_ridge", 100.0, velocity_bounds)
    weights = model.coefficient_weights.reshape((384, len(FEATURE_NAMES), 4))
    coefficient_rows: list[dict[str, Any]] = []
    for feature_index, name in enumerate(FEATURE_NAMES):
        group = weights[:, feature_index, :]
        coefficient_rows.append(
            {
                "feature_group": name,
                "standardized_coefficient_frobenius_norm": np.linalg.norm(group),
                "standardized_coefficient_mean_abs": np.mean(np.abs(group)),
                "component_1_frobenius_norm": np.linalg.norm(group[:, 0]),
                "component_2_frobenius_norm": np.linalg.norm(group[:, 1]),
                "component_3_frobenius_norm": np.linalg.norm(group[:, 2]),
                "component_4_frobenius_norm": np.linalg.norm(group[:, 3]),
                "component_mean_abs": json.dumps(np.mean(np.abs(group), axis=0).tolist()),
            }
        )
    _write_csv(output_dir / "travel_time_coefficient_diagnostics.csv", coefficient_rows)
    test_times = {case.target_id: case.observations[:, 7].copy() for case in test}
    if operator_arrays is not None and operator_arrays.exists():
        with np.load(operator_arrays) as artifact:
            ids = [str(value) for value in artifact["target_ids"].tolist()]
            fsm_arrays = {target_id: np.asarray(values, dtype=float) for target_id, values in zip(ids, artifact["fsm_times_s"], strict=True)}
    else:
        fsm_arrays = {target_id: None for target_id in test_times}
    train_time_mean = np.mean(np.stack([case.observations[:, 7] for case in train]), axis=0)
    test_ids = [case.target_id for case in test]
    split_rng = np.random.default_rng(_stable_seed("hardening-split-level-time-shuffle"))
    split_perm = split_rng.permutation(len(test))
    split_times = {case.target_id: test_times[test_ids[int(split_perm[index])]] for index, case in enumerate(test)}
    rows: list[dict[str, Any]] = []
    for case in test:
        within_rng = np.random.default_rng(_stable_seed(f"hardening-within-time-shuffle:{case.target_id}"))
        within_times = case.observations[:, 7][within_rng.permutation(384)]
        variants: dict[str, np.ndarray] = {
            "true_travel_time": test_times[case.target_id],
            "within_case_shuffled": within_times,
            "split_level_shuffled": split_times[case.target_id],
            "training_feature_mean": train_time_mean,
        }
        if fsm_arrays[case.target_id] is not None:
            variants["fsm_background_reference"] = fsm_arrays[case.target_id]
        baseline_prediction, baseline_coefficients = _predict_details(model, _full_feature_from_times(case, variants["true_travel_time"]))
        target = _case_target_cells(case)
        for variant, times in variants.items():
            predicted_nodes, coefficients = _predict_details(model, _full_feature_from_times(case, times))
            predicted_cells = cell_centered_values_from_node_vector(predicted_nodes[0], _node_shape(case))
            metrics = _prediction_metrics_from_arrays(target, predicted_cells)
            rows.append(
                {
                    "target_id": case.target_id,
                    "family": case.family,
                    "variant": variant,
                    "cell_rmse_km_per_s": metrics["rmse"],
                    "cell_mae_km_per_s": metrics["mae"],
                    "coefficient_l2_distance_from_true": np.linalg.norm(coefficients[0] - baseline_coefficients[0]),
                    "coefficient_mean_abs_difference_from_true": np.mean(np.abs(coefficients[0] - baseline_coefficients[0])),
                    "travel_time_l2_distance_from_true_s": np.linalg.norm(times - variants["true_travel_time"]),
                    "travel_time_mean_abs_difference_from_true_s": np.mean(np.abs(times - variants["true_travel_time"])),
                }
            )
    _write_csv(output_dir / "travel_time_counterfactual_metrics.csv", rows)
    observations = np.concatenate([case.observations[:, 7] for case in test])
    geo = []
    if fsm_arrays and all(value is not None for value in fsm_arrays.values()):
        geo = np.concatenate([test_times[case.target_id] - fsm_arrays[case.target_id] for case in test])
    else:
        for case in test:
            raw = _load_all_observation_rows(production_dir, case)
            background = [float(row["background_travel_time_s"]) for row in raw if row.get("background_travel_time_s") not in (None, "")]
            if background:
                geo.extend(test_times[case.target_id] - np.asarray(background))
    scale_rows: list[dict[str, Any]] = []
    for level in NOISE_LEVELS:
        scale_rows.append(
            {
                "noise_std_s": level,
                "median_abs_travel_time_s": np.median(np.abs(observations)),
                "median_abs_geological_perturbation_s": np.median(np.abs(geo)) if len(geo) else None,
                "noise_as_percent_of_median_abs_travel_time": 100.0 * level / np.median(np.abs(observations)),
                "noise_as_percent_of_median_abs_geological_perturbation": 100.0 * level / np.median(np.abs(geo)) if len(geo) and np.median(np.abs(geo)) > 1.0e-6 else None,
            }
        )
    _write_csv(output_dir / "travel_time_scale_context.csv", scale_rows)
    noise_rows = _read_csv(evidence_dir / "noise" / "noise_replication_metrics.csv")
    noise_summary: list[dict[str, Any]] = []
    for method in ("realistic_full", "travel_time_only", "fixed_ray"):
        for level in NOISE_LEVELS:
            values = [float(row["cell_rmse_km_per_s"]) for row in noise_rows if row["method_id"] == method and float(row["noise_std_s"]) == level]
            degradations = [float(row["degradation_from_zero_rmse_km_per_s"]) for row in noise_rows if row["method_id"] == method and float(row["noise_std_s"]) == level]
            noise_summary.append(
                {
                    "method_id": method,
                    "noise_std_s": level,
                    "n_target_replication_rows": len(values),
                    "mean_rmse": np.mean(values),
                    "std_rmse": np.std(values),
                    "median_rmse": np.median(values),
                    "p05_rmse": np.percentile(values, 5),
                    "p95_rmse": np.percentile(values, 95),
                    "mean_degradation": np.mean(degradations),
                    "std_degradation": np.std(degradations),
                    "p05_degradation": np.percentile(degradations, 5),
                    "p95_degradation": np.percentile(degradations, 95),
                }
            )
    _write_csv(output_dir / "noise_replication_variability.csv", noise_summary)
    learning_rows = _read_csv(evidence_dir / "learning_curve" / "learning_curve_repetitions.csv")
    _write_csv(output_dir / "learning_curve_variability.csv", learning_rows)
    _write_markdown(
        paper_dir / "travel_time_dependence_report.md",
        """# Frozen-model travel-time dependence

The selected full-input PCA-ridge estimator is reconstructed from the frozen train-only PCA and
the preselected configuration (4 components, ridge alpha 100). It is not retrained for the
counterfactuals. Standardized coefficient magnitudes are descriptive diagnostics; they are not
causal importance measures.

The counterfactual table replaces only the travel-time channel with true times, within-case
shuffles, split-level shuffles, the training-feature mean, and the exact common-background FSM
reference when the operator audit is available. It reports coefficient changes and velocity RMSE
for the same held-out targets. The coefficient-group table and counterfactual table are
`travel_time_coefficient_diagnostics.csv` and `travel_time_counterfactual_metrics.csv`.

The separate scale table compares 0.025/0.05/0.10 s with observed travel-time and geological
perturbation scales. The noise table retains the ten-seed repetitions and reports mean, SD,
median, p05/p95, and degradation from zero noise across target/replication rows. A weak response
to small zero-mean perturbations would not by itself imply that the model ignores travel time;
association-destruction counterfactuals are the direct test of that claim.
""",
    )


def write_noise_learning_report(evidence_dir: Path, paper_dir: Path, output_dir: Path) -> None:
    del evidence_dir, output_dir
    _write_markdown(
        paper_dir / "noise_learning_variability_report.md",
        """# Noise and learning-curve variability

The frozen noise analysis uses ten seeds at each nonzero level and aggregates repetitions within
target before the across-target summary. The hardening table retains the raw target/replication
rows and reports mean, SD, median, p05/p95, and degradation from zero noise for realistic/full,
travel-time-only, and fixed-ray methods. No repetition was added or selected after inspecting test
performance.

The frozen learning curve uses family-balanced training subsets of 25, 50, 100, 150, and 175
targets with three subset seeds. The complete repetition table is copied into the hardening
directory as `learning_curve_variability.csv`; the 175-target rows are expected to coincide because
the full training set is used at that size.

These analyses are sensitivity descriptions. They do not turn the ten perturbation seeds or three
subset seeds into independent geological targets.
""",
    )


def _fixed_ray_raw_diagnostics(
    case: ProductionCase,
    geometry: RayGeometry,
    reference_cells: np.ndarray,
    settings: Any,
    evidence_prediction: np.ndarray,
    *,
    reference_travel_times_s: np.ndarray | None = None,
    damping: float = 0.1,
    smoothing: float = 10.0,
) -> dict[str, Any]:
    del settings
    cell_shape = _cell_shape(case)
    reference_slowness = 1.0 / reference_cells
    reference_times = (
        ray_matvec(geometry, reference_slowness)
        if reference_travel_times_s is None
        else np.asarray(reference_travel_times_s, dtype=float)
    )
    residual = case.observations[:, 7] - reference_times
    rhs = _ray_transpose_local(geometry, residual, reference_cells.size)
    delta, iterations, converged = _solve_pcg(
        geometry,
        rhs,
        damping=damping,
        smoothing=smoothing,
        cell_shape=cell_shape,
        max_iterations=PCG_MAX_ITERATIONS,
        tolerance=PCG_TOLERANCE,
    )
    slowness = reference_slowness + delta
    positive = np.isfinite(slowness) & (slowness > 0.0)
    finite = np.where(positive, slowness, 1.0 / 8.0)
    preclip = 1.0 / finite
    clipped = np.clip(preclip, 3.0, 8.0)
    return {
        "target_id": case.target_id,
        "method_id": "fixed_ray",
        "raw_slowness_zero_or_negative_fraction": np.mean(~positive),
        "raw_slowness_min_s_per_km": np.min(slowness),
        "raw_slowness_max_s_per_km": np.max(slowness),
        "preclip_velocity_min_km_per_s": np.min(preclip),
        "preclip_velocity_max_km_per_s": np.max(preclip),
        "preclip_velocity_below_3_fraction": np.mean(preclip < 3.0),
        "preclip_velocity_above_8_fraction": np.mean(preclip > 8.0),
        "postclip_lower_boundary_fraction": np.mean(clipped <= 3.0 + 1.0e-12),
        "postclip_upper_boundary_fraction": np.mean(clipped >= 8.0 - 1.0e-12),
        "postclip_changed_fraction": np.mean(np.abs(clipped - preclip) > 1.0e-12),
        "pcg_iterations": iterations,
        "pcg_converged": converged,
        "evidence_prediction_lower_boundary_fraction": np.mean(evidence_prediction <= 3.0 + 1.0e-12),
        "evidence_prediction_upper_boundary_fraction": np.mean(evidence_prediction >= 8.0 - 1.0e-12),
    }


def _ray_transpose_local(geometry: RayGeometry, vector: np.ndarray, cell_count: int) -> np.ndarray:
    result = np.zeros(cell_count, dtype=float)
    for row_index in range(geometry.observation_count):
        start = int(geometry.row_ptr[row_index])
        end = int(geometry.row_ptr[row_index + 1])
        np.add.at(result, geometry.cell_indices[start:end], geometry.path_lengths_km[start:end] * vector[row_index])
    return result


def write_fixed_ray_report(
    cases: Sequence[ProductionCase], artifacts: Mapping[str, Mapping[str, np.ndarray]], settings: Any, evidence_dir: Path, paper_dir: Path, output_dir: Path
) -> None:
    rows: list[dict[str, Any]] = []
    geometry_dir = evidence_dir / "classical" / "reference_ray_geometry"
    for case in cases:
        if case.split != "test":
            continue
        geometry = _load_geometry(geometry_dir / f"{case.target_id}.npz")
        reference_cells = artifacts["reference_prior"][case.target_id]
        fixed = artifacts["fixed_ray"][case.target_id]
        rows.append(_fixed_ray_raw_diagnostics(case, geometry, reference_cells, settings, fixed))
    for method in ML_METHODS:
        for case in cases:
            if case.split != "test":
                continue
            prediction = cell_centered_values_from_node_vector(artifacts[method][case.target_id], _node_shape(case))
            rows.append(
                {
                    "target_id": case.target_id,
                    "method_id": method,
                    "ml_lower_boundary_fraction": np.mean(prediction <= 3.0 + 1.0e-12),
                    "ml_upper_boundary_fraction": np.mean(prediction >= 8.0 - 1.0e-12),
                }
            )
    _write_csv(output_dir / "fixed_ray_physical_diagnostics.csv", rows)
    corrected_selection_path = output_dir.parent / "corrected_classical" / "selected_classical_configuration.json"
    selected = _read_json(
        corrected_selection_path
        if corrected_selection_path.exists()
        else evidence_dir / "classical" / "selected_classical_configuration.json"
    )
    candidates = _read_csv(evidence_dir / "classical" / "classical_validation_candidate_summary.csv")
    pseudo = settings.pseudo_bending_solver
    _write_csv(output_dir / "fixed_ray_validation_grid_copy.csv", candidates)
    algorithm_contract = f"""# Fixed-ray algorithm contract

The reference-ray implementation is **`tomobench.simulation.ray_tracing.trace_pseudo_bending_ray`**
in `code/src/tomobench/simulation/ray_tracing.py`, called by
`tomobench.evaluation.benchmark_v2_final_analysis.build_reference_ray_geometry`. Algorithmically,
it is an **interface-aware constrained waypoint local search with piecewise-straight path segments**.
The name `pseudo_bending` is retained because it is the repository function name, but this code is not
a reimplementation of the published two-point algorithm in [7] or the spherical-earth method in [8].
Those references provide methodological context only.

Reference rays were calculated using an initial Snell-law ray-parameter solution through the
piecewise-constant horizontal layered background, initialized by bisection on the layered ray
parameter for 120 iterations. The initial path is then augmented with explicit crossing waypoints
for the configured block, ellipsoid, dyke, and fault interfaces. Interior interface waypoints are
constrained to their surfaces and relocated by deterministic coordinate search: both signs of the
two-dimensional tangent directions are tested, the candidate is projected back to the interface,
and a move is accepted only when it decreases the piecewise-constant polyline travel time by more
than 1e-12 s. The step is halved from `max(0.25 km, minimum grid spacing)` to 0.001 km. The outer
loop stops on no improvement, on an improvement no larger than 1e-5 s, or after 30 iterations.

Endpoints are passed as the saved source and receiver coordinates. The ray tracer clamps points to
the model domain internally; all production endpoints are already inside that domain, so this clamp
does not change the saved production coordinates. The cell operator subsequently intersects the
piecewise-straight polyline with the 40 x 40 x 12 target-cell grid, assigns each segment midpoint to
one cell using the documented lower-inclusive convention, and accumulates path length. No adaptive
point insertion or path densification is performed after interface crossings. The configured
`initial_ray_point_count` (**{pseudo.initial_ray_point_count}**) and `max_ray_point_count`
(**{pseudo.max_ray_point_count}**) are retained in metadata but are not active refinement loops in
`trace_pseudo_bending_ray`; likewise, `max_point_move_km` and `finite_difference_step_km` are
configuration metadata rather than operations used by this reference-ray call.

The fixed-ray sensitivity is therefore a sparse path-length operator, not an exact Jacobian of the
FSM solver. The corrected inversion uses this fixed reference geometry with the residual
`t_obs - t_FSM(s0)` and does not update the rays after recovering a model.
"""
    _write_markdown(paper_dir / "fixed_ray_algorithm_contract.md", algorithm_contract)
    _write_markdown(
        paper_dir / "fixed_ray_implementation_report.md",
        algorithm_contract
        + f"""

## Inversion selection and physical diagnostics

Validation selection used damping candidates **{selected.get('damping_grid')}** and smoothing
candidates **{selected.get('smoothing_grid')}**. Each candidate was evaluated on the 37 validation
targets using mean all-cell velocity RMSE; ties were ordered by positive-coverage RMSE, travel-time
RMSE, damping, and smoothing. The selected values were damping **{selected.get('selected_damping')}**
and smoothing **{selected.get('selected_smoothing')}**. The 38-target test set was not used in this
selection.

The physical diagnostics are in `fixed_ray_physical_diagnostics.csv`. Before clipping, the report
records zero/negative slowness, slowness extrema, reciprocal velocities outside [3,8] km/s, and
post-clipping boundary fractions. The same boundary fractions are reported for ML fields. These
diagnostics are descriptive and do not remove or reselect targets.
""",
    )


def run_corrected_classical_reanalysis(
    cases: Sequence[ProductionCase],
    settings: Any,
    operator_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Re-evaluate only the fixed-ray comparator with ``t_obs-t_FSM(s0)``.

    The validation grid and test artifacts are written to a new directory.  The
    historical classical evidence is never overwritten.
    """

    with np.load(operator_dir / "operator_consistency_arrays.npz") as artifact:
        target_ids = [str(value) for value in artifact["target_ids"].tolist()]
        fsm_by_target = {
            target_id: np.asarray(values, dtype=float)
            for target_id, values in zip(target_ids, artifact["fsm_times_s"], strict=True)
        }
    geometry_dir = operator_dir / "reference_ray_geometry"
    validation = [case for case in cases if case.split == "validation"]
    test = [case for case in cases if case.split == "test"]
    if not all(case.target_id in fsm_by_target for case in [*validation, *test]):
        raise ValueError("Corrected classical reanalysis requires FSM reference times for validation and test.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    lower, upper = (float(value) for value in settings.velocity_model_generation.velocity_bounds_km_per_s)
    cell_shape = _cell_shape(test[0])
    validation_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    case_cache: dict[str, tuple[ProductionCase, RayGeometry, np.ndarray, np.ndarray]] = {}
    for case in [*validation, *test]:
        geometry = _load_geometry(geometry_dir / f"{case.target_id}.npz")
        reference_cells = _reference_cells(case, settings)
        # The corrected run uses the direct cell-center truth that is reported as
        # the primary cell-domain metric. The historical eight-node contract
        # remains available in the preserved final-evidence directory.
        target_cells = direct_analytic_cell_truth(case)
        case_cache[case.target_id] = (case, geometry, reference_cells, target_cells)
    for damping in (0.001, 0.01, 0.1, 1.0):
        for smoothing in (0.0, 0.1, 1.0, 10.0):
            solved_rows = []
            for case in validation:
                _, geometry, reference_cells, target_cells = case_cache[case.target_id]
                solved = solve_fixed_ray_case(
                    case=case,
                    geometry=geometry,
                    reference_cells=reference_cells,
                    target_cells=target_cells,
                    damping=damping,
                    smoothing=smoothing,
                    cell_shape=cell_shape,
                    velocity_bounds=(lower, upper),
                    reference_travel_times_s=fsm_by_target[case.target_id],
                )
                solved_rows.append(solved)
                validation_rows.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "damping": damping,
                        "smoothing": smoothing,
                        "all_cell_rmse_km_per_s": solved["all_cell_rmse"],
                        "positive_coverage_rmse_km_per_s": solved["positive_coverage_rmse"],
                        "travel_time_rmse_s": solved["travel_time_rmse"],
                        "clipped_fraction": solved["clipped_fraction"],
                        "pcg_iterations": solved["pcg_iterations"],
                        "pcg_converged": solved["pcg_converged"],
                    }
                )
            candidates.append(
                {
                    "damping": damping,
                    "smoothing": smoothing,
                    "validation_all_cell_rmse_mean": np.mean([row["all_cell_rmse"] for row in solved_rows]),
                    "validation_positive_coverage_rmse_mean": np.mean([row["positive_coverage_rmse"] for row in solved_rows]),
                    "validation_travel_time_rmse_mean": np.mean([row["travel_time_rmse"] for row in solved_rows]),
                    "all_pcg_converged": all(row["pcg_converged"] for row in solved_rows),
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            float(row["validation_all_cell_rmse_mean"]),
            float(row["validation_positive_coverage_rmse_mean"]),
            float(row["validation_travel_time_rmse_mean"]),
            float(row["damping"]),
            float(row["smoothing"]),
        ),
    )
    _write_csv(output_dir / "classical_validation_grid.csv", validation_rows)
    _write_csv(output_dir / "classical_validation_candidate_summary.csv", candidates)
    _write_json(
        {
            "artifact_type": "robustness_checks_corrected_fixed_ray_reanalysis",
            "residual_formula": "t_obs - t_FSM(s0)",
            "sensitivity_operator": "G_ref retained from reference-model pseudo-bending rays",
            "selected_damping": selected["damping"],
            "selected_smoothing": selected["smoothing"],
            "damping_grid": [0.001, 0.01, 0.1, 1.0],
            "smoothing_grid": [0.0, 0.1, 1.0, 10.0],
            "selection_metric": "validation mean all-cell velocity RMSE",
            "selection_unit": "37 validation targets",
            "test_evaluation_policy": "selected corrected configuration evaluated once on frozen 38-target test set",
            "historical_evidence_preserved": True,
        },
        output_dir / "selected_classical_configuration.json",
    )
    test_rows: list[dict[str, Any]] = []
    test_predictions: dict[str, np.ndarray] = {}
    for case in test:
        _, geometry, reference_cells, target_cells = case_cache[case.target_id]
        solved = solve_fixed_ray_case(
            case=case,
            geometry=geometry,
            reference_cells=reference_cells,
            target_cells=target_cells,
            damping=float(selected["damping"]),
            smoothing=float(selected["smoothing"]),
            cell_shape=cell_shape,
            velocity_bounds=(lower, upper),
            reference_travel_times_s=fsm_by_target[case.target_id],
        )
        test_predictions[case.target_id] = np.asarray(solved["fixed_cells"], dtype=float)
        np.savez_compressed(
            output_dir / "predictions" / f"{case.target_id}.npz",
            target_cells=target_cells,
            reference_cells=reference_cells,
            fixed_cells=solved["fixed_cells"],
            coverage_km=geometry.coverage_km,
            fixed_predicted_times=solved["predicted_times"],
            fixed_residual_s=solved["residuals"],
        )
        test_rows.append(
            {
                "target_id": case.target_id,
                "family": case.family,
                "cell_rmse_km_per_s": solved["all_cell_rmse"],
                "cell_mae_km_per_s": solved["all_cell_mae"],
                "cell_bias_km_per_s": float(np.mean(solved["fixed_cells"] - target_cells)),
                "positive_coverage_rmse_km_per_s": solved["positive_coverage_rmse"],
                "travel_time_rmse_s": solved["travel_time_rmse"],
                "travel_time_mae_s": solved["travel_time_mae"],
                "clipped_fraction": solved["clipped_fraction"],
                "pcg_iterations": solved["pcg_iterations"],
                "pcg_converged": solved["pcg_converged"],
            }
        )
    _write_csv(output_dir / "classical_test_metrics.csv", test_rows)
    return {"selected": selected, "test_rows": test_rows, "test_predictions": test_predictions, "fsm_by_target": fsm_by_target}


def write_operator_consistency_report(
    operator_dir: Path, paper_dir: Path, cases: Sequence[ProductionCase]
) -> dict[str, Any]:
    target_rows = _read_csv(operator_dir / "operator_consistency_target_metrics.csv")
    observation_rows = _read_csv(operator_dir / "operator_consistency_observation_metrics.csv")
    differences = np.asarray([float(row["operator_difference_s"]) for row in observation_rows])
    absolute = np.abs(differences)
    stats = _stats(differences)
    stats.update(
        {
            "mae": float(np.mean(absolute)),
            "rmse": float(np.sqrt(np.mean(differences**2))),
            "median_abs": float(np.median(absolute)),
            "p90_abs": float(np.percentile(absolute, 90)),
            "p95_abs": float(np.percentile(absolute, 95)),
            "p99_abs": float(np.percentile(absolute, 99)),
            "max_abs": float(np.max(absolute)),
        }
    )
    _write_json(stats, operator_dir / "operator_consistency_overall_summary.json")
    target_abs = np.asarray([float(row["operator_mae_s"]) for row in target_rows])
    target_residual = np.asarray([float(row["median_abs_r_fsm_s"]) for row in target_rows])
    target_ratio = target_abs / np.maximum(target_residual, 1.0e-6)
    _write_json(
        {
            "target_count": len(target_rows),
            "operator_mae_per_target_stats": _stats(target_abs),
            "median_abs_r_fsm_per_target_stats": _stats(target_residual),
            "operator_mae_over_median_abs_r_fsm_stats_with_1e-6_floor": _stats(target_ratio),
            "noise_scales_s": [0.025, 0.05, 0.1],
            "decision_rule": "interpret the operator difference by direct comparison with observed-time, residual, geological-signal, and noise-scale distributions; no single post-hoc pass/fail threshold is imposed",
            "decision": "FIXED-RAY FORMULATION CORRECTED AND RE-EVALUATED",
            "production_target_count_verified": len(cases),
        },
        operator_dir / "operator_consistency_scale_summary.json",
    )
    first = next(row for row in target_rows if row["target_id"] == "layered_target_01")
    report = f"""# Fixed-ray forward-operator consistency

## Audit performed

For all **{len(cases)}** production acquisition geometries and 96,000 source-receiver records,
the audit compares the exact common-background FSM reference time (t_{{FSM}}(s_0)) with the
reference-ray operator prediction (G_{{ref}}s_0). The 1.25-km finite-grid contract is FSM,
`tt_from_rp=False`, node-centered velocity, direct analytic sampling, and the same source/receiver
coordinates with no snapping. Non-layered cases reuse their stored common-background production
times, which were generated by this solver contract; layered cases are recomputed because the
production background column was explicitly target-equal for that family.

The operator-level signed difference (t_{{FSM}}(s_0)-G_{{ref}}s_0) has n={stats['n']}, mean signed
difference **{_fmt(stats['mean'])} s**, MAE **{_fmt(stats['mae'])} s**, RMSE **{_fmt(stats['rmse'])} s**,
median absolute difference **{_fmt(stats['median_abs'])} s**, p90 **{_fmt(stats['p90_abs'])} s**,
p95 **{_fmt(stats['p95_abs'])} s**, p99 **{_fmt(stats['p99_abs'])} s**, and maximum absolute
difference **{_fmt(stats['max_abs'])} s**. Source-depth, source-receiver-offset, family, and
case-level distributions are in the CSV files beside this report.

## Scale comparison and decision

The audit also records (r_{{FSM}}=t_{{obs}}-t_{{FSM}}(s_0)), observed-time magnitude, robust
ratios with a 1e-6-s denominator guard, and the 0.025/0.05/0.10-s noise scales. The guard is a
reporting epsilon only; it is not an acceptance threshold. For example, the first recovered target
row has median absolute (r_{{FSM}}) **{first['median_abs_r_fsm_s']} s** and operator MAE
**{first['operator_mae_s']} s**; the complete target distribution is the authoritative scale
comparison.

The offset is not treated as a cosmetic difference: (G_{{ref}}s_0) is a path-integral prediction
from a separate reference-ray construction, whereas (t_{{FSM}}(s_0)) is the adopted finite-grid
forward operator used for labels. Across the completed audit, the discrepancy is material relative
to the 0.025/0.05/0.10-s noise scales and to the reference-model residual scale, so the historical
residual formula is not treated as interchangeable with the adopted forward operator. The fixed-ray
residual is therefore corrected to (r=t_{{obs}}-t_{{FSM}}(s_0)) while (G_{{ref}}) is retained as the
fixed linearized sensitivity operator. The corrected validation/test reanalysis is stored separately
and supersedes the historical fixed-ray result for manuscript claims; the historical result remains
preserved as provenance only.

**FIXED-RAY FORMULATION CORRECTED AND RE-EVALUATED**
"""
    _write_markdown(paper_dir / "fixed_ray_operator_consistency.md", report)
    _write_markdown(operator_dir / "fixed_ray_operator_consistency.md", report)
    return {"stats": stats, "target_rows": target_rows, "observation_rows": observation_rows}


def _corrected_method_cells(
    case: ProductionCase,
    artifacts: Mapping[str, Mapping[str, np.ndarray]],
    corrected_fixed: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for method in ML_METHODS:
        predictions[method] = cell_centered_values_from_node_vector(
            artifacts[method][case.target_id], _node_shape(case)
        )
    predictions["reference_prior"] = artifacts["reference_prior"][case.target_id]
    predictions["fixed_ray"] = corrected_fixed[case.target_id]
    return predictions


def _paired_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for comparison_id, first, second in comparisons:
        selected = [row for row in rows if row["comparison_id"] == comparison_id]
        if not selected:
            continue
        deltas = np.asarray([float(row["delta_rmse_km_per_s"]) for row in selected])
        pooled = _bootstrap_mean_ci(deltas, _stable_seed(f"corrected-pooled-{comparison_id}"))
        stratified = _family_stratified_ci(
            selected, _stable_seed(f"corrected-stratified-{comparison_id}")
        )
        output.append(
            {
                "comparison_id": comparison_id,
                "first_method_id": first,
                "second_method_id": second,
                "metric_domain": "direct_analytic_cell_center_all",
                "test_target_count": len(selected),
                "mean_delta_km_per_s": float(np.mean(deltas)),
                "median_delta_km_per_s": float(np.median(deltas)),
                "pooled_ci95_lower_km_per_s": pooled[0],
                "pooled_ci95_upper_km_per_s": pooled[1],
                "family_stratified_ci95_lower_km_per_s": stratified[0],
                "family_stratified_ci95_upper_km_per_s": stratified[1],
                "wins": int(np.count_nonzero(deltas < -TIE_TOLERANCE)),
                "losses": int(np.count_nonzero(deltas > TIE_TOLERANCE)),
                "ties": int(np.count_nonzero(np.abs(deltas) <= TIE_TOLERANCE)),
                "delta_definition": "first method RMSE minus second method RMSE; negative favors first",
                "endpoint_role": (
                    "primary"
                    if comparison_id == "realistic_vs_corrected_fixed_ray"
                    else "secondary hypothesis-driven"
                ),
            }
        )
    return output


def _write_corrected_primary_figure(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "realistic_full": "Full-input PCA-ridge",
        "travel_time_only": "Travel-time only",
        "no_travel_time": "No travel time",
        "shuffled_travel_time": "Shuffled time",
        "reference_prior": "Reference prior",
        "fixed_ray": "Reference-ray baseline",
        "training_target_mean": "Training target mean",
    }
    methods = tuple(labels)
    values = {
        method: np.asarray(
            [
                float(row["direct_cell_center_rmse_km_per_s"])
                for row in rows
                if row["method_id"] == method
            ]
        )
        for method in methods
    }
    figure, axis = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    axis.boxplot(
        [values[method] for method in methods],
        labels=[labels[method] for method in methods],
        showmeans=True,
    )
    rng = np.random.default_rng(_stable_seed("corrected-direct-primary-figure-jitter"))
    for index, method in enumerate(methods, start=1):
        jitter = rng.uniform(-0.09, 0.09, size=values[method].size)
        axis.scatter(
            np.full(values[method].size, index) + jitter,
            values[method],
            s=12,
            alpha=0.55,
        )
    axis.set_ylabel("Per-target cell RMSE (km/s)")
    axis.set_title("Held-out-target comparison using direct cell-center truth (38 targets)")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", labelrotation=24)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / "figure_d_primary_method_comparison_corrected_direct.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "figure_d_primary_method_comparison_corrected_direct.svg",
        bbox_inches="tight",
    )
    plt.close(figure)


def _write_corrected_coverage_bundle(
    cases: Sequence[ProductionCase],
    predictions_by_target: Mapping[str, Mapping[str, np.ndarray]],
    paper_dir: Path,
    output_dir: Path,
) -> None:
    methods = (
        "realistic_full",
        "travel_time_only",
        "no_travel_time",
        "shuffled_travel_time",
        "reference_prior",
        "fixed_ray",
        "training_target_mean",
    )
    test_cases = [case for case in cases if case.split == "test"]
    rows: list[dict[str, Any]] = []
    for case in test_cases:
        direct_truth = direct_analytic_cell_truth(case)
        coverage = predictions_by_target[case.target_id]["coverage"]
        for threshold in (0.0, 0.1, 1.0, 5.0, 10.0):
            mask = coverage > 0.0 if threshold == 0.0 else coverage >= threshold
            for method in methods:
                prediction = predictions_by_target[case.target_id][method]
                metrics = (
                    _prediction_metrics_from_arrays(direct_truth[mask], prediction[mask])
                    if np.any(mask)
                    else {"rmse": None, "mae": None, "bias": None}
                )
                rows.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "method_id": method,
                        "coverage_threshold_label": ">0" if threshold == 0.0 else str(threshold),
                        "coverage_threshold_km": threshold,
                        "cell_count": int(np.count_nonzero(mask)),
                        "total_cell_count": int(coverage.size),
                        "coverage_fraction": float(np.mean(mask)),
                        "rmse_km_per_s": metrics["rmse"],
                        "mae_km_per_s": metrics["mae"],
                        "bias_km_per_s": metrics["bias"],
                    }
                )
    _write_csv(output_dir / "corrected_coverage_domain_metrics.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for threshold in (0.0, 0.1, 1.0, 5.0, 10.0):
        domain_rows = [
            row
            for row in rows
            if float(row["coverage_threshold_km"]) == threshold
            and row["method_id"] == "realistic_full"
        ]
        fractions = np.asarray([float(row["coverage_fraction"]) for row in domain_rows])
        usable = [row for row in domain_rows if int(row["cell_count"]) >= 192]
        for method in methods:
            method_rows = [
                row
                for row in rows
                if float(row["coverage_threshold_km"]) == threshold
                and row["method_id"] == method
            ]
            values = [
                float(row["rmse_km_per_s"])
                for row in method_rows
                if row["rmse_km_per_s"] not in (None, "")
            ]
            summary_rows.append(
                {
                    "coverage_threshold_label": ">0" if threshold == 0.0 else str(threshold),
                    "coverage_threshold_km": threshold,
                    "method_id": method,
                    "test_target_count": len(domain_rows),
                    "metric_contributing_target_count": len(values),
                    "total_qualifying_cells_across_targets": int(
                        sum(int(row["cell_count"]) for row in domain_rows)
                    ),
                    "fraction_of_all_cells_across_targets": float(
                        sum(int(row["cell_count"]) for row in domain_rows)
                        / (len(domain_rows) * 19200)
                    ),
                    "mean_qualifying_cell_fraction": float(np.mean(fractions)),
                    "median_qualifying_cell_fraction": float(np.median(fractions)),
                    "p05_qualifying_cell_fraction": float(np.percentile(fractions, 5)),
                    "p95_qualifying_cell_fraction": float(np.percentile(fractions, 95)),
                    "targets_with_at_least_one_cell": int(
                        sum(int(row["cell_count"]) > 0 for row in domain_rows)
                    ),
                    "targets_with_usable_cells_ge_1pct": len(usable),
                    "rmse_mean_km_per_s": float(np.mean(values)) if values else None,
                    "rmse_median_km_per_s": float(np.median(values)) if values else None,
                }
            )
    _write_csv(output_dir / "corrected_coverage_domain_summary.csv", summary_rows)
    paired_rows: list[dict[str, Any]] = []
    for threshold in (0.0, 0.1, 1.0, 5.0, 10.0):
        left = {
            row["target_id"]: row
            for row in rows
            if float(row["coverage_threshold_km"]) == threshold
            and row["method_id"] == "realistic_full"
        }
        right = {
            row["target_id"]: row
            for row in rows
            if float(row["coverage_threshold_km"]) == threshold
            and row["method_id"] == "fixed_ray"
        }
        for target_id in sorted(left):
            if left[target_id]["rmse_km_per_s"] in (None, "") or right[target_id]["rmse_km_per_s"] in (None, ""):
                continue
            paired_rows.append(
                {
                    "comparison_id": "realistic_vs_corrected_fixed_ray",
                    "target_id": target_id,
                    "family": left[target_id]["family"],
                    "coverage_threshold_label": ">0" if threshold == 0.0 else str(threshold),
                    "coverage_threshold_km": threshold,
                    "delta_rmse_km_per_s": float(left[target_id]["rmse_km_per_s"])
                    - float(right[target_id]["rmse_km_per_s"]),
                }
            )
    _write_csv(output_dir / "corrected_coverage_paired_deltas.csv", paired_rows)
    paired_summary: list[dict[str, Any]] = []
    for threshold in (0.0, 0.1, 1.0, 5.0, 10.0):
        selected = [
            row
            for row in paired_rows
            if float(row["coverage_threshold_km"]) == threshold
        ]
        deltas = np.asarray([float(row["delta_rmse_km_per_s"]) for row in selected])
        pooled = _bootstrap_mean_ci(
            deltas, _stable_seed(f"corrected-coverage-{threshold}")
        ) if len(deltas) else (math.nan, math.nan)
        stratified = _family_stratified_ci(
            selected, _stable_seed(f"corrected-coverage-stratified-{threshold}")
        ) if len(deltas) else (math.nan, math.nan)
        paired_summary.append(
            {
                "coverage_threshold_label": ">0" if threshold == 0.0 else str(threshold),
                "coverage_threshold_km": threshold,
                "test_target_count": len(test_cases),
                "contributing_target_count": len(selected),
                "delta_definition": "realistic_full RMSE minus corrected fixed_ray RMSE; negative favors realistic_full",
                "delta_mean_km_per_s": float(np.mean(deltas)) if len(deltas) else None,
                "delta_median_km_per_s": float(np.median(deltas)) if len(deltas) else None,
                "pooled_ci95_lower_km_per_s": pooled[0],
                "pooled_ci95_upper_km_per_s": pooled[1],
                "family_stratified_ci95_lower_km_per_s": stratified[0],
                "family_stratified_ci95_upper_km_per_s": stratified[1],
                "wins": int(np.count_nonzero(deltas < -TIE_TOLERANCE)),
                "losses": int(np.count_nonzero(deltas > TIE_TOLERANCE)),
                "ties": int(np.count_nonzero(np.abs(deltas) <= TIE_TOLERANCE)),
            }
        )
    _write_csv(output_dir / "corrected_coverage_paired_summary.csv", paired_summary)
    _write_json(
        {
            "truth_definition": "direct analytic cell-center evaluation",
            "coverage_source": "accumulated reference-model fixed-ray path length",
            "thresholds_reported": [">0", 0.1, 1.0, 5.0, 10.0],
            "zero_cell_policy": "retain all test targets in domain summaries and report contributing counts separately",
            "independent_diagnostic_status": "unavailable: production FSM labels have no validated path sidecars",
        },
        output_dir / "corrected_coverage_metadata.json",
    )
    ten = next(row for row in paired_summary if row["coverage_threshold_km"] == 10.0)
    _write_markdown(
        paper_dir / "corrected_coverage_domain_report.md",
        f"""# Corrected fixed-ray coverage-domain sensitivity

This report repeats the coverage-domain diagnostic after the fixed-ray residual correction and
uses direct analytic cell-center truth. Coverage remains accumulated path length along the
reference-model rays; it is an illumination diagnostic, not ground-truth sensitivity or an ML
input. All test targets remain in the denominator, and the 1% usable-cell convention is reporting
only.

At the pre-specified 10-km accumulated reference-ray path-length threshold, the paired
realistic-full minus corrected-fixed-ray delta is **{_fmt(ten['delta_mean_km_per_s'])} km/s** with
pooled percentile-bootstrap CI **[{_fmt(ten['pooled_ci95_lower_km_per_s'])},
{_fmt(ten['pooled_ci95_upper_km_per_s'])}] km/s** and **{ten['contributing_target_count']}**
contributing targets. Complete domain, family, depth, and target-level records are in the CSV
files beside this report.

An independent true-model/FSM path diagnostic remains unavailable because the frozen FSM labels did
not retain validated path sidecars. The reference-ray origin is therefore kept explicit.
""",
    )


def _write_corrected_structure_bundle(
    cases: Sequence[ProductionCase],
    predictions_by_target: Mapping[str, Mapping[str, np.ndarray]],
    paper_dir: Path,
    output_dir: Path,
) -> None:
    test_cases = [case for case in cases if case.split == "test"]
    methods = (
        "realistic_full",
        "travel_time_only",
        "no_travel_time",
        "shuffled_travel_time",
        "reference_prior",
        "fixed_ray",
        "training_target_mean",
    )
    rows: list[dict[str, Any]] = []
    for case in test_cases:
        target = direct_analytic_cell_truth(case)
        masks = build_structure_masks(case)
        for method in methods:
            prediction = predictions_by_target[case.target_id][method]
            for scope, mask_map in masks.items():
                for label, mask in mask_map.items():
                    metrics = (
                        _prediction_metrics_from_arrays(target[mask], prediction[mask])
                        if np.any(mask)
                        else {"rmse": None, "mae": None, "bias": None}
                    )
                    row: dict[str, Any] = {
                        "target_id": case.target_id,
                        "family": case.family,
                        "method_id": method,
                        "metric_scope": scope,
                        "mask_label": label,
                        "cell_count": int(np.count_nonzero(mask)),
                        "rmse_km_per_s": metrics["rmse"],
                        "mae_km_per_s": metrics["mae"],
                        "bias_km_per_s": metrics["bias"],
                        "true_mean_km_per_s": metrics.get("true_mean"),
                        "predicted_mean_km_per_s": metrics.get("predicted_mean"),
                    }
                    rows.append(row)
            for family in ("block_anomaly", "salt_dome", "dyke_intrusion"):
                if case.family != family:
                    continue
                body = masks["structure"]["body"]
                background = masks["structure"]["background"]
                true_contrast = float(np.mean(target[body]) - np.mean(target[background]))
                predicted_contrast = float(
                    np.mean(prediction[body]) - np.mean(prediction[background])
                )
                rows.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "method_id": method,
                        "metric_scope": "contrast",
                        "mask_label": "body_minus_background",
                        "cell_count": int(np.count_nonzero(body)),
                        "true_contrast_km_per_s": true_contrast,
                        "predicted_contrast_km_per_s": predicted_contrast,
                        "contrast_error_km_per_s": predicted_contrast - true_contrast,
                    }
                )
            if case.family == "faulted":
                positive = masks["fault"]["positive_side"]
                negative = masks["fault"]["negative_side"]
                true_contrast = float(np.mean(target[positive]) - np.mean(target[negative]))
                predicted_contrast = float(
                    np.mean(prediction[positive]) - np.mean(prediction[negative])
                )
                rows.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "method_id": method,
                        "metric_scope": "contrast",
                        "mask_label": "positive_minus_negative_side",
                        "cell_count": int(np.count_nonzero(positive)),
                        "true_contrast_km_per_s": true_contrast,
                        "predicted_contrast_km_per_s": predicted_contrast,
                        "contrast_error_km_per_s": predicted_contrast - true_contrast,
                    }
                )
    _write_csv(output_dir / "corrected_structure_method_metrics.csv", rows)
    _write_csv(output_dir / "corrected_structure_method_summary.csv", _aggregate_structure_rows(rows))
    _write_markdown(
        paper_dir / "corrected_structure_method_comparison.md",
        """# Corrected fixed-ray structure comparison

The generator-defined masks are applied unchanged to all methods after replacing the historical
fixed-ray output with the validation-selected corrected reanalysis. All velocity metrics use direct
analytic cell-center truth. Body/background, fault-side, layer-wise, and contrast rows are
aggregated at the target level; the source-receiver observations are not treated as independent
structural replicates. The complete rows and summaries are in the adjacent CSV files.
""",
    )


def _write_corrected_noise_bundle(
    cases: Sequence[ProductionCase],
    settings: Any,
    operator_dir: Path,
    corrected_result: Mapping[str, Any],
    output_dir: Path,
) -> None:
    with np.load(operator_dir / "operator_consistency_arrays.npz") as artifact:
        ids = [str(value) for value in artifact["target_ids"].tolist()]
        fsm_by_target = {
            target_id: np.asarray(values, dtype=float)
            for target_id, values in zip(ids, artifact["fsm_times_s"], strict=True)
        }
    del corrected_result
    train = [case for case in cases if case.split == "train"]
    test = [case for case in cases if case.split == "test"]
    train_targets = np.vstack([case.target_vector for case in train])
    pca = fit_target_pca(train_targets)
    feature_matrices = build_feature_matrices(cases)
    train_indices = [index for index, case in enumerate(cases) if case.split == "train"]
    bounds = tuple(float(value) for value in settings.velocity_model_generation.velocity_bounds_km_per_s)
    models = {
        method: fit_observation_pca_model(
            feature_matrices[method][train_indices],
            train_targets,
            pca,
            4,
            "pca_ridge",
            100.0,
            bounds,
        )
        for method in ("realistic_full", "travel_time_only")
    }
    geometry_dir = operator_dir / "reference_ray_geometry"
    selected = _read_json(output_dir.parent / "selected_classical_configuration.json")
    damping = float(selected["selected_damping"])
    smoothing = float(selected["selected_smoothing"])
    rows: list[dict[str, Any]] = []
    zero_metrics: dict[tuple[str, str], float] = {}
    for noise_level in NOISE_LEVELS:
        seeds = (0,) if noise_level == 0.0 else tuple(range(1, 11))
        for seed in seeds:
            for case in test:
                rng = np.random.default_rng(
                    _stable_seed(f"noise:{int(round(noise_level * 1000))}:{seed}:{case.target_id}")
                )
                noisy_observations = case.observations.copy()
                if noise_level > 0.0:
                    noisy_observations[:, 7] += rng.normal(
                        0.0, noise_level, size=noisy_observations.shape[0]
                    )
                noisy_case = replace(case, observations=noisy_observations)
                predictions = {
                    method: cell_centered_values_from_node_vector(
                        models[method].predict(build_feature_matrices([noisy_case])[method])[0],
                        _node_shape(case),
                    )
                    for method in ("realistic_full", "travel_time_only")
                }
                geometry = _load_geometry(geometry_dir / f"{case.target_id}.npz")
                reference_cells = _reference_cells(case, settings)
                target = direct_analytic_cell_truth(case)
                fixed = solve_fixed_ray_case(
                    case=noisy_case,
                    geometry=geometry,
                    reference_cells=reference_cells,
                    target_cells=target,
                    damping=damping,
                    smoothing=smoothing,
                    cell_shape=_cell_shape(case),
                    velocity_bounds=bounds,
                    reference_travel_times_s=fsm_by_target[case.target_id],
                )
                predictions["fixed_ray"] = fixed["fixed_cells"]
                for method, prediction in predictions.items():
                    metrics = _prediction_metrics_from_arrays(target, prediction)
                    zero_key = (case.target_id, method)
                    if noise_level == 0.0:
                        zero_metrics[zero_key] = metrics["rmse"]
                    rows.append(
                        {
                            "target_id": case.target_id,
                            "family": case.family,
                            "method_id": method,
                            "noise_std_s": noise_level,
                            "noise_seed": seed,
                            "cell_rmse_km_per_s": metrics["rmse"],
                            "cell_mae_km_per_s": metrics["mae"],
                            "cell_bias_km_per_s": metrics["bias"],
                        }
                    )
    for row in rows:
        row["degradation_from_zero_rmse_km_per_s"] = float(
            row["cell_rmse_km_per_s"]
        ) - zero_metrics[(str(row["target_id"]), str(row["method_id"]))]
    _write_csv(output_dir / "corrected_noise_replication_metrics.csv", rows)
    summary: list[dict[str, Any]] = []
    for method in ("realistic_full", "travel_time_only", "fixed_ray"):
        for level in NOISE_LEVELS:
            selected_rows = [
                row
                for row in rows
                if row["method_id"] == method and float(row["noise_std_s"]) == level
            ]
            rmse = np.asarray([float(row["cell_rmse_km_per_s"]) for row in selected_rows])
            degradation = np.asarray(
                [float(row["degradation_from_zero_rmse_km_per_s"]) for row in selected_rows]
            )
            summary.append(
                {
                    "method_id": method,
                    "noise_std_s": level,
                    "target_replication_row_count": len(selected_rows),
                    "rmse_mean_km_per_s": float(np.mean(rmse)),
                    "rmse_sd_km_per_s": float(np.std(rmse)),
                    "rmse_median_km_per_s": float(np.median(rmse)),
                    "rmse_p05_km_per_s": float(np.percentile(rmse, 5)),
                    "rmse_p95_km_per_s": float(np.percentile(rmse, 95)),
                    "degradation_mean_km_per_s": float(np.mean(degradation)),
                    "degradation_sd_km_per_s": float(np.std(degradation)),
                    "degradation_p05_km_per_s": float(np.percentile(degradation, 5)),
                    "degradation_p95_km_per_s": float(np.percentile(degradation, 95)),
                }
            )
    _write_csv(output_dir / "corrected_noise_summary.csv", summary)
    _write_json(
        {
            "truth_definition": "direct analytic cell-center evaluation",
            "noise_seeds": "identical deterministic seeds to frozen noise analysis",
            "fixed_ray_residual": "t_obs_noisy - t_FSM(s_0), with G_ref retained",
            "models_retrained": False,
            "test_selection": False,
        },
        output_dir / "corrected_noise_metadata.json",
    )


def run_corrected_classical_hardening(
    cases: Sequence[ProductionCase],
    settings: Any,
    evidence_dir: Path,
    operator_dir: Path,
    corrected_result: Mapping[str, Any],
    paper_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Regenerate fixed-ray-dependent diagnostics after the operator correction."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _load_final_test_artifacts(cases, evidence_dir)
    corrected_fixed = corrected_result["test_predictions"]
    predictions_by_target: dict[str, dict[str, np.ndarray]] = {}
    metric_rows: list[dict[str, Any]] = []
    for case in cases:
        if case.split != "test":
            continue
        predictions = _corrected_method_cells(case, artifacts, corrected_fixed)
        predictions["coverage"] = artifacts["coverage"][case.target_id]
        predictions_by_target[case.target_id] = predictions
        old_truth = _case_target_cells(case)
        direct_truth = direct_analytic_cell_truth(case)
        for method, prediction in predictions.items():
            if method == "coverage":
                continue
            old = _prediction_metrics_from_arrays(old_truth, prediction)
            direct = _prediction_metrics_from_arrays(direct_truth, prediction)
            metric_rows.append(
                {
                    "target_id": case.target_id,
                    "family": case.family,
                    "method_id": method,
                    "old_eight_corner_rmse_km_per_s": old["rmse"],
                    "direct_cell_center_rmse_km_per_s": direct["rmse"],
                    "old_eight_corner_mae_km_per_s": old["mae"],
                    "direct_cell_center_mae_km_per_s": direct["mae"],
                    "direct_minus_old_rmse_km_per_s": direct["rmse"] - old["rmse"],
                    "direct_bias_km_per_s": direct["bias"],
                }
            )
    _write_csv(output_dir / "corrected_test_method_metrics.csv", metric_rows)
    comparison_specs = (
        ("realistic_vs_corrected_fixed_ray", "realistic_full", "fixed_ray"),
        ("realistic_vs_travel_time_only", "realistic_full", "travel_time_only"),
        ("realistic_vs_training_target_mean", "realistic_full", "training_target_mean"),
        ("realistic_vs_no_travel_time", "realistic_full", "no_travel_time"),
        ("realistic_vs_shuffled_travel_time", "realistic_full", "shuffled_travel_time"),
        ("realistic_vs_reference_prior", "realistic_full", "reference_prior"),
        ("travel_time_only_vs_corrected_fixed_ray", "travel_time_only", "fixed_ray"),
    )
    paired_rows: list[dict[str, Any]] = []
    by_target = {
        target_id: {row["method_id"]: row for row in target_rows}
        for target_id, target_rows in _group_metric_rows_by_target(metric_rows).items()
    }
    for comparison_id, first, second in comparison_specs:
        for target_id, rows_by_method in sorted(by_target.items()):
            paired_rows.append(
                {
                    "comparison_id": comparison_id,
                    "target_id": target_id,
                    "family": rows_by_method[first]["family"],
                    "delta_rmse_km_per_s": float(
                        rows_by_method[first]["direct_cell_center_rmse_km_per_s"]
                    )
                    - float(rows_by_method[second]["direct_cell_center_rmse_km_per_s"]),
                }
            )
    _write_csv(output_dir / "corrected_paired_deltas.csv", paired_rows)
    paired_summary = _paired_summary_rows(paired_rows, comparison_specs)
    _write_csv(output_dir / "corrected_paired_summary.csv", paired_summary)
    write_corrected_statistical_report(output_dir, paper_dir)
    primary = next(
        row for row in paired_summary if row["comparison_id"] == "realistic_vs_corrected_fixed_ray"
    )
    _write_json(
        {
            "primary_endpoint": "realistic_full versus corrected fixed_ray on direct analytic cell-center all-cell RMSE",
            "primary": primary,
            "truth_definition": "direct analytic cell-center evaluation",
            "historical_result_preserved": True,
            "test_selection": False,
        },
        output_dir / "corrected_primary_result.json",
    )
    _write_corrected_primary_figure(metric_rows, output_dir / "figures")
    _write_corrected_coverage_bundle(
        cases, predictions_by_target, paper_dir, output_dir / "coverage"
    )
    _write_corrected_structure_bundle(
        cases, predictions_by_target, paper_dir, output_dir / "structure"
    )
    _write_corrected_noise_bundle(
        cases, settings, operator_dir, corrected_result, output_dir / "noise"
    )
    diagnostics: list[dict[str, Any]] = []
    for case in cases:
        if case.split != "test":
            continue
        geometry = _load_geometry(operator_dir / "reference_ray_geometry" / f"{case.target_id}.npz")
        reference_cells = _reference_cells(case, settings)
        diagnostic = _fixed_ray_raw_diagnostics(
            case,
            geometry,
            reference_cells,
            settings,
            corrected_fixed[case.target_id],
            reference_travel_times_s=corrected_result["fsm_by_target"][case.target_id],
            damping=float(corrected_result["selected"]["damping"]),
            smoothing=float(corrected_result["selected"]["smoothing"]),
        )
        diagnostics.append(diagnostic)
    _write_csv(output_dir / "corrected_fixed_ray_physical_diagnostics.csv", diagnostics)
    write_corrected_public_diagnostic_figures(output_dir)
    _write_markdown(
        paper_dir / "corrected_classical_reanalysis.md",
        f"""# Corrected fixed-ray reanalysis

The operator audit found a separate FSM/reference-ray background-time discrepancy. The fixed-ray
comparator was therefore re-evaluated with residual
`r = t_obs - t_FSM(s0)`, retaining the reference-ray `G_ref` sensitivity matrix. Damping and
smoothing were selected on the 37 validation targets only; the selected pair was then evaluated
once on the frozen 38-target test set. Historical results remain under the original final-evidence
directory and are not overwritten.

The primary corrected metric uses direct analytic cell-center truth. The full-input minus corrected
fixed-ray direct-cell RMSE delta is **{_fmt(primary['mean_delta_km_per_s'])} km/s**, with pooled
95% CI **[{_fmt(primary['pooled_ci95_lower_km_per_s'])}, {_fmt(primary['pooled_ci95_upper_km_per_s'])}]**,
family-stratified 95% CI **[{_fmt(primary['family_stratified_ci95_lower_km_per_s'])},
{_fmt(primary['family_stratified_ci95_upper_km_per_s'])}]**, and wins/losses/ties
**{primary['wins']}/{primary['losses']}/{primary['ties']}**. Complete corrected paired,
coverage, structure, noise, figure, and physical-diagnostic artifacts are in this directory.
""",
    )
    return {"primary": primary, "metric_rows": metric_rows, "predictions_by_target": predictions_by_target}


def _group_metric_rows_by_target(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_id"])].append(row)
    return grouped


def write_corrected_statistical_report(output_dir: Path, paper_dir: Path) -> None:
    """Write direct-truth method and paired uncertainty summaries from cached rows."""

    metric_rows = _read_csv(output_dir / "corrected_test_method_metrics.csv")
    method_order = (
        "realistic_full",
        "travel_time_only",
        "no_travel_time",
        "geometry_only",
        "distance_only",
        "shuffled_travel_time",
        "training_target_mean",
        "reference_prior",
        "fixed_ray",
    )
    method_labels = {
        "realistic_full": "Full-input PCA-ridge",
        "travel_time_only": "Travel-time-only",
        "no_travel_time": "No travel time",
        "geometry_only": "Geometry-only",
        "distance_only": "Distance-only",
        "shuffled_travel_time": "Shuffled travel time",
        "training_target_mean": "Training-target mean",
        "reference_prior": "Reference 1-D prior",
        "fixed_ray": "Reference-ray baseline",
    }
    method_summary: list[dict[str, Any]] = []
    for method in method_order:
        selected = [row for row in metric_rows if row["method_id"] == method]
        rmse = np.asarray(
            [float(row["direct_cell_center_rmse_km_per_s"]) for row in selected], dtype=float
        )
        mae = np.asarray(
            [float(row["direct_cell_center_mae_km_per_s"]) for row in selected], dtype=float
        )
        rmse_ci = _bootstrap_mean_ci(rmse, _stable_seed(f"corrected-method-rmse-{method}"))
        mae_ci = _bootstrap_mean_ci(mae, _stable_seed(f"corrected-method-mae-{method}"))
        method_summary.append(
            {
                "method_id": method,
                "method_label": method_labels[method],
                "target_count": len(selected),
                "mean_rmse_km_per_s": float(np.mean(rmse)),
                "rmse_ci95_lower_km_per_s": rmse_ci[0],
                "rmse_ci95_upper_km_per_s": rmse_ci[1],
                "mean_mae_km_per_s": float(np.mean(mae)),
                "mae_ci95_lower_km_per_s": mae_ci[0],
                "mae_ci95_upper_km_per_s": mae_ci[1],
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "metric_domain": "direct_analytic_cell_center_all",
            }
        )
    _write_csv(output_dir / "corrected_method_summary.csv", method_summary)
    paired_summary = _read_csv(output_dir / "corrected_paired_summary.csv")
    comparison_labels = {
        "realistic_vs_corrected_fixed_ray": "Realistic full − corrected fixed-ray",
        "realistic_vs_travel_time_only": "Realistic full − travel-time-only",
        "realistic_vs_training_target_mean": "Realistic full − training-target mean",
        "realistic_vs_no_travel_time": "Realistic full − no travel time",
        "realistic_vs_shuffled_travel_time": "Realistic full − shuffled travel time",
        "realistic_vs_reference_prior": "Realistic full − reference prior",
        "travel_time_only_vs_corrected_fixed_ray": "Travel-time-only − corrected fixed-ray",
    }
    lines = [
        "# Corrected direct-truth statistical sensitivity",
        "",
        "The active primary endpoint is target-level all-cell velocity RMSE evaluated against direct",
        "analytic cell-center truth. Each of the 38 test targets contributes one metric per method;",
        "source-receiver rows are not resampled as independent observations. The method intervals are",
        "10,000-resample percentile bootstrap intervals, and the paired intervals use the same target-level",
        "unit. Family-stratified intervals resample within the five configured families before pooling",
        "the resampled target values.",
        "",
        "## Direct-truth method summary",
        "",
        "| Method | Mean RMSE (km/s) | 95% CI | Mean MAE (km/s) | 95% CI |",
        "|---|---:|---|---:|---|",
    ]
    for row in method_summary:
        lines.append(
            f"| {row['method_label']} | {_fmt(row['mean_rmse_km_per_s'], 6)} | "
            f"[{_fmt(row['rmse_ci95_lower_km_per_s'], 6)}, {_fmt(row['rmse_ci95_upper_km_per_s'], 6)}] | "
            f"{_fmt(row['mean_mae_km_per_s'], 6)} | "
            f"[{_fmt(row['mae_ci95_lower_km_per_s'], 6)}, {_fmt(row['mae_ci95_upper_km_per_s'], 6)}] |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "Negative deltas favor the first method. Wins, losses, and ties use the declared target-level",
            "tie tolerance; no source-receiver observation is treated as an independent replicate.",
            "",
            "| Comparison | Mean delta (km/s) | Pooled 95% CI | Family-stratified 95% CI | Wins/losses/ties | Role |",
            "|---|---:|---|---|---:|---|",
        ]
    )
    for row in paired_summary:
        lines.append(
            f"| {comparison_labels.get(row['comparison_id'], row['comparison_id'])} | "
            f"{_fmt(row['mean_delta_km_per_s'], 6)} | "
            f"[{_fmt(row['pooled_ci95_lower_km_per_s'], 6)}, {_fmt(row['pooled_ci95_upper_km_per_s'], 6)}] | "
            f"[{_fmt(row['family_stratified_ci95_lower_km_per_s'], 6)}, "
            f"{_fmt(row['family_stratified_ci95_upper_km_per_s'], 6)}] | "
            f"{row['wins']}/{row['losses']}/{row['ties']} | {row['endpoint_role']} |"
        )
    lines.extend(
        [
            "",
            "The corrected fixed-ray conclusion is unchanged under family stratification: the primary",
            "interval remains positive, so the corrected fixed-ray method has lower mean all-cell RMSE",
            "than realistic full-input PCA-ridge on this test set. The no-time and shuffled-time contrasts",
            "also retain intervals below zero, but they are secondary hypothesis-driven contrasts rather",
            "than a claim of universal statistical superiority. The full-input versus travel-time-only and",
            "full-input versus training-target-mean intervals include zero. This wording treats the primary",
            "comparison as confirmatory and the remaining comparisons as bounded, multiplicity-aware",
            "descriptive evidence; no family or observation-level pseudo-replication is used.",
            "",
            "The source rows are `corrected_test_method_metrics.csv`, `corrected_paired_deltas.csv`, and",
            "`corrected_paired_summary.csv` in the companion output directory. The historical eight-corner",
            "analysis remains preserved separately and is not the active endpoint.",
        ]
    )
    _write_markdown(paper_dir / "corrected_statistical_sensitivity.md", "\n".join(lines))


def write_corrected_public_diagnostic_figures(output_dir: Path) -> None:
    """Create active structure and coverage/noise figures from cached corrected summaries."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    structure_rows = _read_csv(output_dir / "structure" / "corrected_structure_method_summary.csv")
    body_specs = (
        ("block_anomaly", "body", "Block body"),
        ("dyke_intrusion", "body", "Dyke body"),
        ("salt_dome", "body", "Salt body"),
    )
    methods = ("realistic_full", "travel_time_only", "fixed_ray")
    labels = {
        "realistic_full": "Full-input PCA-ridge",
        "travel_time_only": "Travel-time-only",
        "fixed_ray": "Reference-ray baseline",
    }
    colors = {"realistic_full": "#4472C4", "travel_time_only": "#70AD47", "fixed_ray": "#ED7D31"}
    structure_values = {
        method: [
            float(
                next(
                    row["mean_rmse_km_per_s"]
                    for row in structure_rows
                    if row["family"] == family
                    and row["method_id"] == method
                    and row["metric_scope"] == "structure"
                    and row["mask_label"] == mask
                )
            )
            for family, mask, _ in body_specs
        ]
        for method in methods
    }
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    x = np.arange(len(body_specs), dtype=float)
    width = 0.24
    for index, method in enumerate(methods):
        axis.bar(
            x + (index - 1) * width,
            structure_values[method],
            width,
            label=labels[method],
            color=colors[method],
        )
    axis.set_xticks(x, [label for _, _, label in body_specs])
    axis.set_ylabel("Mean direct-cell RMSE (km/s)")
    axis.set_title("Generator-defined body recovery")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(figure_dir / "figure_4_structural_recovery_corrected.png", dpi=300)
    figure.savefig(figure_dir / "figure_4_structural_recovery_corrected.svg")
    plt.close(figure)

    coverage_rows = _read_csv(
        output_dir / "coverage" / "corrected_coverage_paired_summary.csv"
    )
    noise_rows = _read_csv(output_dir / "noise" / "corrected_noise_summary.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    thresholds = np.asarray([float(row["coverage_threshold_km"]) for row in coverage_rows])
    deltas = np.asarray([float(row["delta_mean_km_per_s"]) for row in coverage_rows])
    lower = np.asarray([float(row["pooled_ci95_lower_km_per_s"]) for row in coverage_rows])
    upper = np.asarray([float(row["pooled_ci95_upper_km_per_s"]) for row in coverage_rows])
    axes[0].errorbar(
        thresholds,
        deltas,
        yerr=np.vstack((deltas - lower, upper - deltas)),
        fmt="o-",
        color="#C55A11",
        capsize=3,
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Reference-ray coverage threshold (km)")
    axes[0].set_ylabel("Full-input minus reference-ray baseline RMSE (km/s)")
    axes[0].set_title("Coverage-domain paired delta")
    axes[0].grid(alpha=0.25)
    noise_methods = ("realistic_full", "travel_time_only", "fixed_ray")
    for method in noise_methods:
        selected = [row for row in noise_rows if row["method_id"] == method]
        selected.sort(key=lambda row: float(row["noise_std_s"]))
        axes[1].plot(
            [float(row["noise_std_s"]) for row in selected],
            [float(row["degradation_mean_km_per_s"]) for row in selected],
            "o-",
            label=labels[method],
            color=colors[method],
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Timing-noise standard deviation (s)")
    axes[1].set_ylabel("RMSE degradation (km/s)")
    axes[1].set_title("Matched Gaussian-noise sensitivity")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.25)
    figure.savefig(figure_dir / "figure_5_coverage_noise_corrected.png", dpi=300)
    figure.savefig(figure_dir / "figure_5_coverage_noise_corrected.svg")
    plt.close(figure)


def write_metric_harmonization_report(
    cases: Sequence[ProductionCase], artifacts: Mapping[str, Mapping[str, np.ndarray]], paper_dir: Path, output_dir: Path
) -> None:
    rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    for case in cases:
        if case.split != "test":
            continue
        node_truth = _case_target_cells(case)
        direct_truth = direct_analytic_cell_truth(case)
        predictions: dict[str, np.ndarray] = {
            "realistic_full": cell_centered_values_from_node_vector(artifacts["realistic_full"][case.target_id], _node_shape(case)),
            "travel_time_only": cell_centered_values_from_node_vector(artifacts["travel_time_only"][case.target_id], _node_shape(case)),
            "fixed_ray": artifacts["fixed_ray"][case.target_id],
            "reference_prior": artifacts["reference_prior"][case.target_id],
            "training_target_mean": cell_centered_values_from_node_vector(artifacts["training_target_mean"][case.target_id], _node_shape(case)),
        }
        for method, prediction in predictions.items():
            old = _prediction_metrics_from_arrays(node_truth, prediction)
            new = _prediction_metrics_from_arrays(direct_truth, prediction)
            rows.append(
                {
                    "target_id": case.target_id,
                    "family": case.family,
                    "method_id": method,
                    "truth_definition": "eight_corner_node_average",
                    "old_rmse_km_per_s": old["rmse"],
                    "old_mae_km_per_s": old["mae"],
                    "direct_cell_center_rmse_km_per_s": new["rmse"],
                    "direct_cell_center_mae_km_per_s": new["mae"],
                    "rmse_change_direct_minus_old_km_per_s": new["rmse"] - old["rmse"],
                    "mae_change_direct_minus_old_km_per_s": new["mae"] - old["mae"],
                }
            )
            masks = build_structure_masks(case)
            for scope, mapping in masks.items():
                for label, mask in mapping.items():
                    if np.any(mask):
                        old_s = _prediction_metrics_from_arrays(node_truth[mask], prediction[mask])
                        new_s = _prediction_metrics_from_arrays(direct_truth[mask], prediction[mask])
                        structural_rows.append(
                            {
                                "target_id": case.target_id,
                                "family": case.family,
                                "method_id": method,
                                "metric_scope": scope,
                                "mask_label": label,
                                "old_rmse": old_s["rmse"],
                                "direct_rmse": new_s["rmse"],
                                "rmse_change": new_s["rmse"] - old_s["rmse"],
                                "old_bias": old_s["bias"],
                                "direct_bias": new_s["bias"],
                                "bias_change": new_s["bias"] - old_s["bias"],
                            }
                        )
    _write_csv(output_dir / "metric_harmonization_target_metrics.csv", rows)
    _write_csv(output_dir / "metric_harmonization_structure_metrics.csv", structural_rows)
    summary: list[dict[str, Any]] = []
    for method in ("realistic_full", "travel_time_only", "fixed_ray", "reference_prior", "training_target_mean"):
        selected = [row for row in rows if row["method_id"] == method]
        summary.append(
            {
                "method_id": method,
                "old_mean_rmse": np.mean([float(row["old_rmse_km_per_s"]) for row in selected]),
                "direct_mean_rmse": np.mean([float(row["direct_cell_center_rmse_km_per_s"]) for row in selected]),
                "old_median_rmse": np.median([float(row["old_rmse_km_per_s"]) for row in selected]),
                "direct_median_rmse": np.median([float(row["direct_cell_center_rmse_km_per_s"]) for row in selected]),
                "mean_change": np.mean([float(row["rmse_change_direct_minus_old_km_per_s"]) for row in selected]),
            }
        )
    _write_csv(output_dir / "metric_harmonization_summary.csv", summary)
    old_primary = [row for row in rows if row["method_id"] == "realistic_full"]
    old_fixed = [row for row in rows if row["method_id"] == "fixed_ray"]
    old_delta = np.asarray([float(left["old_rmse_km_per_s"]) - float(right["old_rmse_km_per_s"]) for left, right in zip(sorted(old_primary, key=lambda row: row["target_id"]), sorted(old_fixed, key=lambda row: row["target_id"]), strict=True)])
    direct_delta = np.asarray([float(left["direct_cell_center_rmse_km_per_s"]) - float(right["direct_cell_center_rmse_km_per_s"]) for left, right in zip(sorted(old_primary, key=lambda row: row["target_id"]), sorted(old_fixed, key=lambda row: row["target_id"]), strict=True)])
    _write_json(
        {
            "old_primary_mean_delta_realistic_minus_fixed": float(np.mean(old_delta)),
            "direct_primary_mean_delta_realistic_minus_fixed": float(np.mean(direct_delta)),
            "old_primary_wins": int(np.count_nonzero(old_delta < 0.0)),
            "direct_primary_wins": int(np.count_nonzero(direct_delta < 0.0)),
            "old_primary_ci95": _bootstrap_mean_ci(old_delta, _stable_seed("metric-old-primary")),
            "direct_primary_ci95": _bootstrap_mean_ci(direct_delta, _stable_seed("metric-direct-primary")),
            "truth_contracts_compared": ["eight-corner node average", "direct analytic cell-center evaluation"],
            "truth_definition_not_selected_by_method_performance": True,
        },
        output_dir / "metric_harmonization_primary_comparison.json",
    )
    _write_markdown(
        paper_dir / "metric_harmonization_sensitivity.md",
        """# Metric-harmonization sensitivity

The historical primary metric converts node-centered truth and ML predictions to 40×40×12 cells
by averaging eight target-node corners; fixed-ray output is already cell-valued. This report adds
an independent cell-center truth obtained by evaluating each saved analytic construction directly
at cell centers. It does not derive the alternative truth by averaging nodes and it is not chosen
because it favors a method.

The target-level absolute metrics, structure-mask metrics, method summaries, and paired
full-input/fixed-ray comparison are in the CSV/JSON files beside this report. The direct analytic
cell-center definition is conceptually the more direct truth for a cell-domain comparison; the
historical eight-node result is retained as a sensitivity so the effect of changing the metric
contract remains visible. The report records whether mean errors, rankings, paired conclusions,
and structure-mask values change.
""",
    )


def write_forward_validation_report(paper_dir: Path, solver_dir: Path) -> None:
    _write_markdown(
        paper_dir / "final_forward_model_validation_summary.md",
        f"""# Final forward-model numerical validation summary

The final production labels use `ttcrpy==1.4.2`, 3-D rectilinear FSM, `tt_from_rp=False`, direct
node-centered analytic sampling on an 81×81×25 grid with 1.25-km spacing, and unsnapped endpoints.
The existing authoritative v2 solver audit was reused; no open-ended solver search was restarted.

## Existing evidence

The audit contains finite homogeneous and monotonic layered analytical controls, node-versus-cell
representation comparisons, staged heterogeneous 2.5→1.25-km refinement, a limited 0.625-km
cross-check, targeted 0.3125-km feasibility diagnostics, and an independent grid-aligned PyKonal
finite-grid comparison. Exact numerical tables remain in `{_relative(get_repo_root(), solver_dir)}`.

The 1.25-km analytic controls are finite-grid errors, not continuum-exact validation. The audit
reports a 1.25-km PyKonal/FSM absolute difference with mean 0.0689 s, median 0.0537 s, p95
0.0996 s, and maximum 0.1036 s for its 15 comparable grid-aligned pairs; this is a cross-check,
not ground truth. The heterogeneous 2.5→1.25-km refinement table is finite but family- and
representation-dependent, and full family-wide 0.625-km refinement was not completed. The audit
also found affected-signal numerical error/signal ratios that can be comparable to or larger than
one for sampled narrow/fault structures.

## Interpretation for this manuscript

The labels are documented finite-grid 3-D FSM first-arrival approximations. Solver convergence and
finite output do not establish continuum-exact travel times, globally exact first arrivals, or
stable ray paths in all heterogeneous cases. The forward-model limitations remain part of the
validity statement and limit external generalization. The target-grid resolution audit and the
sharp-interface/fault/dyke limitations should be cited in Supplementary Methods, not omitted when
reporting ML or fixed-ray results.

## Final claim level

The manuscript is classified as **LEVEL 1 — DISCRETE-OPERATOR BENCHMARK**. The family-resolved
signal/error analysis separates the 96,000-row geological-signal distribution from the limited
finite-grid refinement sample and the independent finite-grid solver cross-check. The evidence
supports a reproducible computational benchmark, not a continuum-physical accuracy claim. The
separate local FSM/fixed-ray diagnostic also does not support treating the saved path lengths as a
validated FSM Jacobian.
""",
    )


def write_framing_report(paper_dir: Path) -> None:
    _write_markdown(
        paper_dir / "novelty_framing_reassessment.md",
        """# Novelty and framing reassessment

Three bounded title variants were considered:

1. **A Controlled Synthetic Benchmark for Learning-Based and Fixed-Ray 3-D P-Wave Velocity Reconstruction from First-Arrival Travel Times**
2. **A Controlled Synthetic Benchmark of PCA-Ridge and Fixed-Ray 3-D P-Wave Velocity Reconstruction from First-Arrival Travel Times**
3. **Controlled Evaluation of Travel-Time Information, PCA-Ridge Reconstruction, and Fixed-Ray Inversion in a Synthetic 3-D Velocity Benchmark**

The evidence supports an evaluation-centric contribution: a controlled synthetic benchmark,
frozen target-atomic split, travel-time information controls, and comparison of one selected
PCA-ridge estimator with a bounded reference-model fixed-ray comparator. It does not support a
generic claim about “machine learning for tomography,” nonlinear tomography, or superiority of
learning methods as a class. The method-specific title is the most precise if the paper continues
to report only PCA-ridge; the evaluation-centric title is the safest match to the actual evidence
because much of the contribution is the control and metric design.

No stronger learning baseline is preregistered at this stage. The final decision should be made
after the primary validity checks: the current study is not an architecture benchmark, and adding a
model only to improve the test result would be outside the frozen design. If editors require a
nonlinear comparator, it must be a separately preregistered follow-up with validation-only tuning.
""",
    )


def _load_operator_arrays(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None or not path.exists():
        return None
    with np.load(path) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def run_hardening_analysis(
    *,
    production_dir: Path,
    evidence_dir: Path,
    output_dir: Path,
    paper_dir: Path,
    run_operator: bool = False,
) -> None:
    """Generate the non-destructive hardening reports from frozen evidence."""

    settings = load_settings()
    cases = load_production_cases(production_dir)
    artifacts = _load_final_test_artifacts(cases, evidence_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    write_reference_provenance(paper_dir, production_dir, settings)
    write_acquisition_report(production_dir, paper_dir, output_dir / "acquisition", cases)
    write_target_space_report(cases, paper_dir, output_dir / "target_space")
    write_coverage_report(cases, artifacts, paper_dir, output_dir / "coverage")
    write_structure_report(cases, artifacts, paper_dir, output_dir / "structure")
    write_statistical_report(evidence_dir, paper_dir, output_dir / "statistics")
    write_metric_harmonization_report(cases, artifacts, paper_dir, output_dir / "metric_harmonization")
    operator_arrays = output_dir / "operator_consistency" / "operator_consistency_arrays.npz"
    if run_operator:
        run_operator_consistency(production_dir, operator_arrays.parent, cases=cases, settings=settings)
    write_travel_time_report(cases, artifacts, settings, production_dir, evidence_dir, output_dir / "travel_time", paper_dir, operator_arrays if operator_arrays.exists() else None)
    write_noise_learning_report(evidence_dir, paper_dir, output_dir / "travel_time")
    write_fixed_ray_report(cases, artifacts, settings, evidence_dir, paper_dir, output_dir / "fixed_ray")
    write_forward_validation_report(paper_dir, get_repo_root() / "outputs/generated/submission_readiness/authoritative_forward_solver_v2")
    write_framing_report(paper_dir)
    _write_json(
        {
            "artifact_type": "robustness_checks_analysis_bundle",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "production_target_count": len(cases),
            "test_target_count": sum(case.split == "test" for case in cases),
            "source_evidence_dir": _relative(get_repo_root(), evidence_dir),
            "operator_consistency_present": operator_arrays.exists(),
            "test_set_modified": False,
            "model_selection_on_test": False,
        },
        output_dir / "hardening_analysis_metadata.json",
    )
