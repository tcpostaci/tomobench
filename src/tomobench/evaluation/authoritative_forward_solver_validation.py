"""Validation workflow for selecting an authoritative Benchmark v2 forward solver.

This module is deliberately audit-only.  It validates package behavior and the
legacy solver on existing analytic/frozen cases, but it does not generate a new
Benchmark v2 corpus or fit any ML model.

The primary candidate is ttcrpy on an independently sampled forward grid.  PyKonal
and scikit-fmm are treated as independent/secondary checks where their numerical
contracts permit a fair comparison.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import CartesianVelocityGrid3D, LayeredVelocityModel, VelocityLayer
from tomobench.generation.velocity_grids import (
    build_cartesian_forward_grid,
    build_cartesian_velocity_grid,
)
from tomobench.generation.velocity_models import generate_layered_velocity_model
from tomobench.simulation.ray_tracing import trace_pseudo_bending_ray
from tomobench.simulation.ttcrpy_forward import (
    TtcrpyGridConfiguration,
    TtcrpyRectilinearForwardSolver,
    velocity_array_from_cartesian_grid,
)
from tomobench.simulation.travel_times import layered_first_arrival_travel_time_s
from tomobench.utils.paths import get_repo_root


VALIDATION_ID = "authoritative_forward_solver_validation_v1"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/authoritative_forward_solver_v1"
)
GEOLOGICAL_FAMILIES = (
    "layered",
    "block_anomaly",
    "faulted",
    "salt_dome",
    "dyke_intrusion",
)
DEFAULT_RESOLUTION_SPACINGS_KM = (2.5, 1.25, 0.625)
DEFAULT_ALGORITHM_SPACING_KM = 2.5
DEFAULT_ANALYTIC_SPACING_KM = 1.25
DEFAULT_MAX_FORWARD_NODES = 5_000_000
DEFAULT_SEED = 20260828


@dataclass(frozen=True)
class AuthoritativeValidationOutputs:
    """Files produced by the authoritative-solver validation workflow."""

    output_dir: Path
    summary_json: Path
    summary_md: Path
    analytic_csv: Path
    algorithm_csv: Path
    resolution_csv: Path
    pykonal_csv: Path
    scikit_fmm_csv: Path
    legacy_csv: Path
    discrepancy_plot: Path
    metadata_json: Path


@dataclass(frozen=True)
class ValidationPair:
    """One source-receiver pair used by a validation block."""

    pair_id: str
    source: Point3D
    receiver: Point3D


@dataclass(frozen=True)
class _GridSpec:
    family: str
    spacing_km: float
    grid: CartesianVelocityGrid3D
    values: np.ndarray


def run_authoritative_forward_solver_validation(
    *,
    settings: BenchmarkSettings | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    resolution_spacings_km: Sequence[float] = DEFAULT_RESOLUTION_SPACINGS_KM,
    pairs_per_family: int = 8,
    algorithm_pairs_per_family: int = 5,
    pykonal_pairs_per_family: int = 4,
    legacy_cases_per_profile: int = 1,
    legacy_pairs_per_case: int = 3,
) -> AuthoritativeValidationOutputs:
    """Run the package, analytic, convergence, and legacy diagnostic audits."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    resolved_output = _resolve(output_dir, repo_root)
    spacings = _validate_spacings(resolution_spacings_km)
    for name, value in (
        ("pairs_per_family", pairs_per_family),
        ("algorithm_pairs_per_family", algorithm_pairs_per_family),
        ("pykonal_pairs_per_family", pykonal_pairs_per_family),
        ("legacy_cases_per_profile", legacy_cases_per_profile),
        ("legacy_pairs_per_case", legacy_pairs_per_case),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive.")

    analytic_rows = _run_analytic_validation(
        loaded_settings,
        spacing_km=min(spacings),
    )
    algorithm_rows = _run_algorithm_comparison(
        loaded_settings,
        pairs_per_family=algorithm_pairs_per_family,
    )
    resolution_rows = _run_resolution_study(
        loaded_settings,
        spacings_km=spacings,
        pairs_per_family=pairs_per_family,
    )
    pykonal_rows = _run_pykonal_crosscheck(
        loaded_settings,
        pairs_per_family=pykonal_pairs_per_family,
    )
    scikit_fmm_rows = _run_scikit_fmm_smoke(loaded_settings)
    legacy_rows = _run_legacy_comparison(
        loaded_settings,
        repo_root=repo_root,
        cases_per_profile=legacy_cases_per_profile,
        pairs_per_case=legacy_pairs_per_case,
    )

    summary = build_validation_summary(
        settings=loaded_settings,
        analytic_rows=analytic_rows,
        algorithm_rows=algorithm_rows,
        resolution_rows=resolution_rows,
        pykonal_rows=pykonal_rows,
        scikit_fmm_rows=scikit_fmm_rows,
        legacy_rows=legacy_rows,
        resolution_spacings_km=spacings,
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    analytic_path = resolved_output / "analytic_validation.csv"
    algorithm_path = resolved_output / "algorithm_comparison.csv"
    resolution_path = resolved_output / "resolution_convergence.csv"
    pykonal_path = resolved_output / "pykonal_crosscheck.csv"
    scikit_path = resolved_output / "scikit_fmm_smoke.csv"
    legacy_path = resolved_output / "legacy_solver_comparison.csv"
    summary_path = resolved_output / "authoritative_forward_solver_validation_summary.json"
    summary_md_path = resolved_output / "authoritative_forward_solver_validation_summary.md"
    discrepancy_path = resolved_output / "travel_time_discrepancy_distribution.png"
    metadata_path = resolved_output / "validation_metadata.json"

    _write_csv(analytic_path, analytic_rows)
    _write_csv(algorithm_path, algorithm_rows)
    _write_csv(resolution_path, resolution_rows)
    _write_csv(pykonal_path, pykonal_rows)
    _write_csv(scikit_path, scikit_fmm_rows)
    _write_csv(legacy_path, legacy_rows)
    _write_json(summary_path, summary)
    summary_md_path.write_text(build_validation_summary_markdown(summary), encoding="utf-8")
    _write_discrepancy_plot(analytic_rows, algorithm_rows, legacy_rows, discrepancy_path)
    _write_json(
        metadata_path,
        build_validation_metadata(
            settings=loaded_settings,
            output_dir=resolved_output,
            resolution_spacings_km=spacings,
            pairs_per_family=pairs_per_family,
            algorithm_pairs_per_family=algorithm_pairs_per_family,
            pykonal_pairs_per_family=pykonal_pairs_per_family,
            legacy_cases_per_profile=legacy_cases_per_profile,
            legacy_pairs_per_case=legacy_pairs_per_case,
        ),
    )
    return AuthoritativeValidationOutputs(
        output_dir=resolved_output,
        summary_json=summary_path,
        summary_md=summary_md_path,
        analytic_csv=analytic_path,
        algorithm_csv=algorithm_path,
        resolution_csv=resolution_path,
        pykonal_csv=pykonal_path,
        scikit_fmm_csv=scikit_path,
        legacy_csv=legacy_path,
        discrepancy_plot=discrepancy_path,
        metadata_json=metadata_path,
    )


def _run_analytic_validation(
    settings: BenchmarkSettings,
    *,
    spacing_km: float = DEFAULT_ANALYTIC_SPACING_KM,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    spacing = float(spacing_km)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Analytic-validation spacing must be finite and positive.")
    configurations = (
        ("FSM_rp0", TtcrpyGridConfiguration(method="FSM", tt_from_rp=False, maxit=100)),
        ("FSM_rp1", TtcrpyGridConfiguration(method="FSM", tt_from_rp=True, maxit=100)),
    )
    homogeneous = _homogeneous_model(5.0)
    homogeneous_pairs = _homogeneous_validation_pairs()
    layered = generate_layered_velocity_model(settings)
    layered_pairs = _layered_validation_pairs()
    for model, family, pairs, reference_kind in (
        (homogeneous, "homogeneous", homogeneous_pairs, "homogeneous_euclidean"),
        (layered, "layered", layered_pairs, "layered_first_arrival_snell"),
    ):
        grid = build_cartesian_forward_grid(
            settings,
            model,
            scenario="layered",
            spacing_by_axis_km=(spacing, spacing, spacing),
            max_grid_nodes=DEFAULT_MAX_FORWARD_NODES,
        )
        references = {
            pair.pair_id: (
                _euclidean_time(pair.source, pair.receiver, 5.0)
                if reference_kind == "homogeneous_euclidean"
                else layered_first_arrival_travel_time_s(
                    pair.source,
                    pair.receiver,
                    model,
                    numerical_tolerance=1.0e-10,
                )
            )
            for pair in pairs
        }
        for configuration_name, configuration in configurations:
            rows.extend(
                _ttcrpy_pair_rows(
                    grid=grid,
                    pairs=pairs,
                    configuration=configuration,
                    validation_kind="analytic",
                    family=family,
                    reference_times=references,
                    reference_kind=reference_kind,
                    configuration_name=configuration_name,
                )
            )
    return rows


def _run_algorithm_comparison(
    settings: BenchmarkSettings,
    *,
    pairs_per_family: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    algorithm_configurations = (
        ("FSM_rp0", TtcrpyGridConfiguration(method="FSM", tt_from_rp=False, maxit=100)),
        ("FSM_rp1", TtcrpyGridConfiguration(method="FSM", tt_from_rp=True, maxit=100)),
        (
            "SPM_ns3",
            TtcrpyGridConfiguration(
                method="SPM",
                tt_from_rp=False,
                nsnx=3,
                nsny=3,
                nsnz=3,
            ),
        ),
        (
            "SPM_ns5",
            TtcrpyGridConfiguration(
                method="SPM",
                tt_from_rp=False,
                nsnx=5,
                nsny=5,
                nsnz=5,
            ),
        ),
        (
            "DSPM",
            TtcrpyGridConfiguration(
                method="DSPM",
                tt_from_rp=True,
                n_secondary=2,
                n_tertiary=2,
            ),
        ),
    )
    for family in GEOLOGICAL_FAMILIES:
        model = generate_layered_velocity_model(settings)
        grid = build_cartesian_forward_grid(
            settings,
            model,
            scenario=family,
            spacing_by_axis_km=(
                DEFAULT_ALGORITHM_SPACING_KM,
                DEFAULT_ALGORITHM_SPACING_KM,
                DEFAULT_ALGORITHM_SPACING_KM,
            ),
            max_grid_nodes=DEFAULT_MAX_FORWARD_NODES,
        )
        pairs = _representative_pairs(family, pairs_per_family, DEFAULT_SEED + 11)
        for configuration_name, configuration in algorithm_configurations:
            rows.extend(
                _ttcrpy_pair_rows(
                    grid=grid,
                    pairs=pairs,
                    configuration=configuration,
                    validation_kind="algorithm_comparison",
                    family=family,
                    reference_times={},
                    reference_kind="none",
                    configuration_name=configuration_name,
                )
            )
    return _add_algorithm_pair_deltas(rows)


def _run_resolution_study(
    settings: BenchmarkSettings,
    *,
    spacings_km: Sequence[float],
    pairs_per_family: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    configuration = TtcrpyGridConfiguration(method="FSM", tt_from_rp=True, maxit=100)
    for family_index, family in enumerate(GEOLOGICAL_FAMILIES):
        model = generate_layered_velocity_model(settings)
        pairs = _representative_pairs(
            family,
            pairs_per_family,
            DEFAULT_SEED + 101 + family_index,
        )
        family_rows: list[dict[str, object]] = []
        for spacing in spacings_km:
            grid = build_cartesian_forward_grid(
                settings,
                model,
                scenario=family,
                spacing_by_axis_km=(spacing, spacing, spacing),
                max_grid_nodes=DEFAULT_MAX_FORWARD_NODES,
            )
            family_rows.extend(
                _ttcrpy_pair_rows(
                    grid=grid,
                    pairs=pairs,
                    configuration=configuration,
                    validation_kind="resolution",
                    family=family,
                    reference_times={},
                    reference_kind="none",
                    configuration_name="FSM_rp1",
                )
            )
        rows.extend(_add_resolution_deltas(family_rows, spacings_km))
    return rows


def _ttcrpy_pair_rows(
    *,
    grid: CartesianVelocityGrid3D,
    pairs: Sequence[ValidationPair],
    configuration: TtcrpyGridConfiguration,
    validation_kind: str,
    family: str,
    reference_times: Mapping[str, float],
    reference_kind: str,
    configuration_name: str,
    return_rays: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    solver: TtcrpyRectilinearForwardSolver | None = None
    error: str | None = None
    try:
        solver = TtcrpyRectilinearForwardSolver.from_cartesian_grid(grid, configuration)
        result = solver.raytrace(
            [pair.source for pair in pairs],
            [pair.receiver for pair in pairs],
            return_rays=return_rays,
        )
    except Exception as exc:  # preserve failures as explicit audit rows
        error = f"{type(exc).__name__}: {exc}"
        result = None

    spacing = float(grid.metadata["grid_spacing_km"]["x"])
    for index, pair in enumerate(pairs):
        row: dict[str, object] = {
            "validation_kind": validation_kind,
            "family": family,
            "pair_id": pair.pair_id,
            "configuration": configuration_name,
            "method": configuration.method,
            "tt_from_rp": configuration.tt_from_rp,
            "grid_spacing_km": spacing,
            "source_x_km": pair.source.x_km,
            "source_y_km": pair.source.y_km,
            "source_z_km": pair.source.z_km,
            "receiver_x_km": pair.receiver.x_km,
            "receiver_y_km": pair.receiver.y_km,
            "receiver_z_km": pair.receiver.z_km,
            "euclidean_distance_km": _distance(pair.source, pair.receiver),
            "reference_kind": reference_kind,
            "reference_travel_time_s": reference_times.get(pair.pair_id),
            "status": "error" if error else "ok",
            "error": error,
            "travel_time_s": None,
            "absolute_error_s": None,
            "relative_error": None,
            "path_length_km": None,
            "source_endpoint_error_km": None,
            "receiver_endpoint_error_km": None,
            "runtime_s": None,
        }
        if result is not None:
            travel_time = result.travel_times_s[index]
            reference_time = reference_times.get(pair.pair_id)
            row.update(
                {
                    "travel_time_s": travel_time,
                    "absolute_error_s": (
                        abs(travel_time - reference_time) if reference_time is not None else None
                    ),
                    "relative_error": (
                        abs(travel_time - reference_time) / abs(reference_time)
                        if reference_time not in (None, 0.0)
                        else None
                    ),
                    "path_length_km": (
                        result.path_lengths_km[index] if result.path_lengths_km is not None else None
                    ),
                    "source_endpoint_error_km": (
                        result.source_endpoint_errors_km[index]
                        if result.source_endpoint_errors_km is not None
                        else None
                    ),
                    "receiver_endpoint_error_km": (
                        result.receiver_endpoint_errors_km[index]
                        if result.receiver_endpoint_errors_km is not None
                        else None
                    ),
                    "runtime_s": result.runtime_s / len(pairs),
                }
            )
        rows.append(row)
    return rows


def _add_algorithm_pair_deltas(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    reference_by_pair: dict[tuple[str, str, float], float] = {}
    for row in rows:
        if row["configuration"] == "FSM_rp1" and row["status"] == "ok":
            reference_by_pair[(str(row["family"]), str(row["pair_id"]), float(row["grid_spacing_km"]))] = float(
                row["travel_time_s"]
            )
    output: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        reference = reference_by_pair.get(
            (str(row["family"]), str(row["pair_id"]), float(row["grid_spacing_km"]))
        )
        item["delta_from_FSM_rp1_s"] = (
            float(row["travel_time_s"]) - reference
            if reference is not None and row["travel_time_s"] is not None
            else None
        )
        item["absolute_delta_from_FSM_rp1_s"] = (
            abs(float(item["delta_from_FSM_rp1_s"]))
            if item["delta_from_FSM_rp1_s"] is not None
            else None
        )
        output.append(item)
    return output


def _add_resolution_deltas(
    rows: Sequence[Mapping[str, object]],
    spacings_km: Sequence[float],
) -> list[dict[str, object]]:
    ordered = tuple(sorted(float(value) for value in spacings_km))
    by_pair_spacing = {
        (str(row["pair_id"]), float(row["grid_spacing_km"])): row for row in rows
    }
    output: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        spacing = float(row["grid_spacing_km"])
        next_finer = next((value for value in ordered if value < spacing), None)
        finer = (
            by_pair_spacing.get((str(row["pair_id"]), next_finer))
            if next_finer is not None
            else None
        )
        item["next_finer_spacing_km"] = next_finer
        item["absolute_change_from_next_finer_s"] = (
            abs(float(row["travel_time_s"]) - float(finer["travel_time_s"]))
            if finer is not None
            and row["status"] == "ok"
            and finer["status"] == "ok"
            else None
        )
        item["relative_change_from_next_finer"] = (
            item["absolute_change_from_next_finer_s"] / abs(float(finer["travel_time_s"]))
            if item["absolute_change_from_next_finer_s"] is not None
            and float(finer["travel_time_s"]) != 0.0
            else None
        )
        output.append(item)
    return output


def _homogeneous_model(velocity_km_per_s: float) -> LayeredVelocityModel:
    return LayeredVelocityModel(
        model_id=f"homogeneous_{velocity_km_per_s:g}",
        layers=(
            VelocityLayer(
                layer_id="HOMOGENEOUS",
                top_depth_km=0.0,
                bottom_depth_km=30.0,
                p_velocity_km_per_s=velocity_km_per_s,
            ),
        ),
        metadata={"construction": "analytic_constant_velocity"},
    )


def _homogeneous_validation_pairs() -> tuple[ValidationPair, ...]:
    coordinates = (
        ((1.3, 2.7, 3.1), (1.3, 2.7, 20.9)),
        ((2.5, 3.5, 15.0), (85.0, 3.5, 15.0)),
        ((4.1, 8.2, 4.7), (86.4, 72.3, 0.0)),
        ((12.5, 25.0, 27.5), (87.5, 75.0, 2.5)),
        ((50.0, 50.0, 1.25), (50.0, 50.0, 28.75)),
        ((6.25, 6.25, 6.25), (93.75, 93.75, 23.75)),
        ((18.8, 79.4, 11.3), (91.2, 17.6, 0.0)),
        ((48.7, 12.4, 8.6), (2.6, 88.1, 2.5)),
    )
    return tuple(
        ValidationPair(f"homogeneous_{index:02d}", Point3D(*source), Point3D(*receiver))
        for index, (source, receiver) in enumerate(coordinates, start=1)
    )


def _layered_validation_pairs() -> tuple[ValidationPair, ...]:
    coordinates = (
        ((20.0, 20.0, 2.0), (22.5, 20.0, 0.0)),
        ((20.0, 20.0, 4.0), (30.0, 20.0, 0.0)),
        ((20.0, 20.0, 8.0), (27.5, 20.0, 0.0)),
        ((20.0, 20.0, 15.0), (35.0, 20.0, 0.0)),
        ((20.0, 20.0, 21.0), (40.0, 20.0, 0.0)),
        ((20.0, 20.0, 28.0), (45.0, 20.0, 0.0)),
        ((35.0, 65.0, 8.0), (45.0, 65.0, 0.0)),
        ((35.0, 65.0, 15.0), (55.0, 65.0, 0.0)),
        ((35.0, 65.0, 21.0), (60.0, 65.0, 0.0)),
        ((35.0, 65.0, 28.0), (65.0, 65.0, 0.0)),
    )
    return tuple(
        ValidationPair(f"layered_{index:02d}", Point3D(*source), Point3D(*receiver))
        for index, (source, receiver) in enumerate(coordinates, start=1)
    )


def _representative_pairs(
    family: str,
    count: int,
    seed: int,
) -> tuple[ValidationPair, ...]:
    """Return deterministic pairs, including paths that cross and miss structures."""
    fixed = (
        ((15.0, 50.0, 22.5), (85.0, 50.0, 0.0)),
        ((35.0, 50.0, 17.5), (35.0, 80.0, 0.0)),
        ((70.0, 30.0, 27.5), (20.0, 30.0, 0.0)),
        ((50.0, 15.0, 7.5), (50.0, 85.0, 0.0)),
        ((50.0, 50.0, 12.5), (50.0, 50.0, 0.0)),
        ((20.0, 80.0, 4.5), (80.0, 20.0, 0.0)),
        ((80.0, 70.0, 19.0), (20.0, 70.0, 0.0)),
        ((25.0, 25.0, 25.0), (75.0, 75.0, 0.0)),
    )
    coordinates = list(fixed)
    if count > len(coordinates):
        rng = np.random.default_rng(seed)
        for _ in range(count - len(coordinates)):
            source = (
                float(rng.choice(np.arange(5.0, 96.0, 2.5))),
                float(rng.choice(np.arange(5.0, 96.0, 2.5))),
                float(rng.choice(np.arange(2.5, 30.0, 2.5))),
            )
            receiver = (
                float(rng.choice(np.arange(5.0, 96.0, 2.5))),
                float(rng.choice(np.arange(5.0, 96.0, 2.5))),
                0.0,
            )
            coordinates.append((source, receiver))
    return tuple(
        ValidationPair(f"{family}_{index:02d}", Point3D(*source), Point3D(*receiver))
        for index, (source, receiver) in enumerate(coordinates[:count], start=1)
    )


def _run_pykonal_crosscheck(
    settings: BenchmarkSettings,
    *,
    pairs_per_family: int,
) -> list[dict[str, object]]:
    """Attempt a PyKonal check in the current interpreter.

    The normal ttcrpy probe uses NumPy 1.26 for wheel compatibility, while the
    published PyKonal extension requires a separate NumPy 2 environment on
    Windows.  A separate worker can replace these diagnostic rows later; keeping
    the failure rows here prevents an unavailable check from being hidden.
    """
    try:
        import pykonal
    except Exception as exc:
        return [_pykonal_status_row(f"import_error:{type(exc).__name__}: {exc}")]

    rows: list[dict[str, object]] = []
    for family_index, family in enumerate(GEOLOGICAL_FAMILIES):
        model = generate_layered_velocity_model(settings)
        grid = build_cartesian_forward_grid(
            settings,
            model,
            scenario=family,
            spacing_by_axis_km=(2.5, 2.5, 2.5),
            max_grid_nodes=DEFAULT_MAX_FORWARD_NODES,
        )
        values = velocity_array_from_cartesian_grid(grid)
        pairs = _representative_pairs(family, pairs_per_family, DEFAULT_SEED + family_index)
        for pair in pairs:
            try:
                start = perf_counter()
                travel_time = _pykonal_travel_time(
                    pykonal,
                    values,
                    np.asarray(grid.x_coordinates_km),
                    np.asarray(grid.y_coordinates_km),
                    np.asarray(grid.z_coordinates_km),
                    pair.source,
                    pair.receiver,
                )
                rows.append(
                    {
                        "family": family,
                        "pair_id": pair.pair_id,
                        "status": "ok",
                        "pykonal_travel_time_s": travel_time,
                        "runtime_s": perf_counter() - start,
                        "source_snap_error_km": 0.0,
                        "receiver_snap_error_km": 0.0,
                        "error": None,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "family": family,
                        "pair_id": pair.pair_id,
                        "status": "error",
                        "pykonal_travel_time_s": None,
                        "runtime_s": None,
                        "source_snap_error_km": None,
                        "receiver_snap_error_km": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return rows


def _pykonal_status_row(error: str) -> dict[str, object]:
    return {
        "family": None,
        "pair_id": None,
        "status": "unavailable",
        "pykonal_travel_time_s": None,
        "runtime_s": None,
        "source_snap_error_km": None,
        "receiver_snap_error_km": None,
        "error": error,
    }


def _pykonal_travel_time(
    pykonal: Any,
    velocity: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
    source: Point3D,
    receiver: Point3D,
) -> float:
    solver = pykonal.EikonalSolver(coord_sys="cartesian")
    solver.velocity.min_coords = (
        float(x_coordinates[0]),
        float(y_coordinates[0]),
        float(z_coordinates[0]),
    )
    solver.velocity.node_intervals = (
        float(x_coordinates[1] - x_coordinates[0]),
        float(y_coordinates[1] - y_coordinates[0]),
        float(z_coordinates[1] - z_coordinates[0]),
    )
    solver.velocity.npts = velocity.shape
    solver.velocity.values = np.asarray(velocity, dtype=np.float64)
    source_idx = tuple(
        int(np.argmin(np.abs(axis - coordinate)))
        for axis, coordinate in zip(
            (x_coordinates, y_coordinates, z_coordinates),
            (source.x_km, source.y_km, source.z_km),
        )
    )
    source_node = np.array(
        [x_coordinates[source_idx[0]], y_coordinates[source_idx[1]], z_coordinates[source_idx[2]]]
    )
    source_error = np.linalg.norm(source_node - np.array([source.x_km, source.y_km, source.z_km]))
    if source_error > 1.0e-10:
        raise ValueError(
            "PyKonal cross-check requires grid-aligned sources; "
            f"snap error was {source_error:g} km."
        )
    solver.traveltime.values[source_idx] = 0.0
    solver.unknown[source_idx] = False
    solver.trial.push(*source_idx)
    if not solver.solve():
        raise RuntimeError("PyKonal returned a false solve status.")
    return float(
        solver.traveltime.value(np.array([receiver.x_km, receiver.y_km, receiver.z_km]))
    )


def _run_scikit_fmm_smoke(settings: BenchmarkSettings) -> list[dict[str, object]]:
    """Record a bounded API smoke test; scikit-fmm is not a primary validator."""
    del settings
    try:
        import skfmm

        shape = (9, 9, 9)
        phi = np.ones(shape, dtype=np.float64)
        source_idx = (4, 4, 4)
        phi[source_idx] = -1.0
        speed = np.full(shape, 2.0, dtype=np.float64)
        start = perf_counter()
        travel_times = skfmm.travel_time(phi, speed=speed, dx=(1.0, 1.0, 1.0))
        runtime_s = perf_counter() - start
        return [
            {
                "status": "ok",
                "shape": shape,
                "finite_fraction": float(np.isfinite(travel_times).mean()),
                "source_value_s": float(travel_times[source_idx]),
                "runtime_s": runtime_s,
                "interpretation": "api_smoke_only_zero_contour_not_a_point_source_reference",
                "error": None,
            }
        ]
    except Exception as exc:
        return [
            {
                "status": "error",
                "shape": None,
                "finite_fraction": None,
                "source_value_s": None,
                "runtime_s": None,
                "interpretation": "api_smoke_only",
                "error": f"{type(exc).__name__}: {exc}",
            }
        ]


def _run_legacy_comparison(
    settings: BenchmarkSettings,
    *,
    repo_root: Path,
    cases_per_profile: int,
    pairs_per_case: int,
) -> list[dict[str, object]]:
    manifest_path = repo_root / "outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json"
    if not manifest_path.exists():
        return [{"status": "unavailable", "error": f"missing manifest: {manifest_path}"}]
    manifest = _read_json(manifest_path)
    selected = _select_legacy_items(
        manifest.get("items", []),
        cases_per_profile=cases_per_profile,
    )
    rows: list[dict[str, object]] = []
    configuration = TtcrpyGridConfiguration(method="FSM", tt_from_rp=True, maxit=100)
    for item in selected:
        try:
            target_path = repo_root / str(item["target_velocity_grid_path"])
            grid = CartesianVelocityGrid3D(**_read_json(target_path))
            base_grid = _base_grid_from_target_metadata(settings, grid)
            model = _layered_model_from_grid_metadata(grid)
            historical_settings = _settings_with_historical_scenario(settings, grid)
            authoritative_grid = build_cartesian_forward_grid(
                historical_settings,
                model,
                scenario=str(item["family"]),
                spacing_by_axis_km=(2.5, 2.5, 2.5),
                max_grid_nodes=DEFAULT_MAX_FORWARD_NODES,
            )
            obs_path = repo_root / str(item["input_dataset_csv_path"])
            observations = _read_csv_rows(obs_path)[:pairs_per_case]
            pairs = tuple(
                ValidationPair(
                    f"{item['case_id']}_{index:03d}",
                    Point3D(
                        float(row["source_x_km"]),
                        float(row["source_y_km"]),
                        float(row["source_z_km"]),
                    ),
                    Point3D(
                        float(row["receiver_x_km"]),
                        float(row["receiver_y_km"]),
                        float(row["receiver_z_km"]),
                    ),
                )
                for index, row in enumerate(observations, start=1)
            )
            solver = TtcrpyRectilinearForwardSolver.from_cartesian_grid(
                authoritative_grid,
                configuration,
            )
            authoritative = solver.raytrace(
                [pair.source for pair in pairs],
                [pair.receiver for pair in pairs],
                return_rays=True,
            )
            for index, (pair, observation) in enumerate(zip(pairs, observations)):
                legacy = trace_pseudo_bending_ray(
                    pair.source,
                    pair.receiver,
                    grid,
                    settings.pseudo_bending_solver,
                )
                authoritative_time = authoritative.travel_times_s[index]
                legacy_time = legacy.travel_time_s
                rows.append(
                    {
                        "status": "ok",
                        "family": item.get("family"),
                        "case_id": item.get("case_id"),
                        "parameter_variant_id": item.get("parameter_variant_id"),
                        "geometry_profile_id": item.get("geometry_profile_id"),
                        "station_seed": item.get("station_seed"),
                        "earthquake_seed": item.get("earthquake_seed"),
                        "pair_id": pair.pair_id,
                        "source_depth_km": pair.source.z_km,
                        "source_receiver_distance_km": _distance(pair.source, pair.receiver),
                        "anomaly_intersection": _line_intersects_anomaly(
                            pair.source,
                            pair.receiver,
                            grid,
                            base_grid,
                        ),
                        "historical_label_travel_time_s": float(observation["travel_time_s"]),
                        "legacy_travel_time_s": legacy_time,
                        "ttcrpy_travel_time_s": authoritative_time,
                        "legacy_minus_ttcrpy_s": legacy_time - authoritative_time,
                        "legacy_absolute_difference_s": abs(legacy_time - authoritative_time),
                        "legacy_converged": legacy.converged,
                        "legacy_iteration_count": legacy.iteration_count,
                        "ttcrpy_runtime_s": authoritative.runtime_s / len(pairs),
                        "legacy_grid_spacing_x_km": float(
                            grid.metadata["grid_spacing_km"]["x"]
                        ),
                        "legacy_grid_spacing_y_km": float(
                            grid.metadata["grid_spacing_km"]["y"]
                        ),
                        "legacy_grid_spacing_z_km": float(
                            grid.metadata["grid_spacing_km"]["z"]
                        ),
                        "authoritative_grid_spacing_km": 2.5,
                        "error": None,
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "status": "error",
                    "family": item.get("family"),
                    "case_id": item.get("case_id"),
                    "parameter_variant_id": item.get("parameter_variant_id"),
                    "geometry_profile_id": item.get("geometry_profile_id"),
                    "station_seed": item.get("station_seed"),
                    "earthquake_seed": item.get("earthquake_seed"),
                    "pair_id": None,
                    "source_depth_km": None,
                    "source_receiver_distance_km": None,
                    "anomaly_intersection": None,
                    "historical_label_travel_time_s": None,
                    "legacy_travel_time_s": None,
                    "ttcrpy_travel_time_s": None,
                    "legacy_minus_ttcrpy_s": None,
                    "legacy_absolute_difference_s": None,
                    "legacy_converged": None,
                    "legacy_iteration_count": None,
                    "ttcrpy_runtime_s": None,
                    "legacy_grid_spacing_x_km": None,
                    "legacy_grid_spacing_y_km": None,
                    "legacy_grid_spacing_z_km": None,
                    "authoritative_grid_spacing_km": 2.5,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def _select_legacy_items(
    items: Sequence[Mapping[str, object]],
    *,
    cases_per_profile: int,
) -> list[Mapping[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for item in items:
        groups[(str(item.get("family")), str(item.get("geometry_profile_id")))].append(item)
    selected: list[Mapping[str, object]] = []
    for key in sorted(groups):
        selected.extend(sorted(groups[key], key=lambda item: str(item.get("case_id")))[:cases_per_profile])
    return selected


def _base_grid_from_target_metadata(
    settings: BenchmarkSettings,
    target_grid: CartesianVelocityGrid3D,
) -> CartesianVelocityGrid3D:
    model = _layered_model_from_grid_metadata(target_grid)
    spacing = tuple(
        float(np.diff(axis)[0])
        for axis in (
            target_grid.x_coordinates_km,
            target_grid.y_coordinates_km,
            target_grid.z_coordinates_km,
        )
    )
    return build_cartesian_velocity_grid(
        settings,
        model,
        scenario="layered",
        spacing_by_axis_km=spacing,
        max_grid_nodes=DEFAULT_MAX_FORWARD_NODES,
    )


def _layered_model_from_grid_metadata(
    grid: CartesianVelocityGrid3D,
) -> LayeredVelocityModel:
    layered = grid.metadata.get("layered_model", {})
    boundaries = tuple(float(value) for value in layered.get("depth_boundaries_km", []))
    velocities = tuple(float(value) for value in layered.get("velocities_km_per_s", []))
    if len(boundaries) != len(velocities) + 1 or len(velocities) == 0:
        raise ValueError("Target-grid metadata does not contain a valid layered model.")
    return LayeredVelocityModel(
        model_id=f"{grid.source_velocity_model_id}_layered_metadata_model",
        layers=tuple(
            VelocityLayer(
                layer_id=f"LAYER{index + 1:02d}",
                top_depth_km=boundaries[index],
                bottom_depth_km=boundaries[index + 1],
                p_velocity_km_per_s=velocities[index],
            )
            for index in range(len(velocities))
        ),
        metadata={"construction": "target_metadata_layered_model"},
    )


def _settings_with_historical_scenario(
    settings: BenchmarkSettings,
    target_grid: CartesianVelocityGrid3D,
) -> BenchmarkSettings:
    """Recreate legacy scenario parameters for a finer diagnostic grid.

    The historical grid metadata is the source of truth for this diagnostic;
    the current central-config defaults are not substituted for old variants.
    """
    from dataclasses import replace

    scenario = str(target_grid.metadata.get("grid_scenario", "layered"))
    metadata = target_grid.metadata.get(scenario, {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Missing historical scenario metadata for {scenario!r}.")
    velocity_settings = settings.velocity_model_generation
    if scenario == "block_anomaly":
        scenario_value = replace(
            velocity_settings.block_anomaly,
            center_km=tuple(float(value) for value in metadata["center_km"]),
            size_km=tuple(float(value) for value in metadata["size_km"]),
            velocity_delta_km_per_s=float(metadata["velocity_delta_km_per_s"]),
        )
        velocity_settings = replace(velocity_settings, block_anomaly=scenario_value)
    elif scenario == "faulted":
        scenario_value = replace(
            velocity_settings.faulted,
            fault_x_km=float(metadata["fault_x_km"]),
            fault_y_km=float(metadata["fault_y_km"]),
            strike_deg=float(metadata["strike_deg"]),
            dip_deg=float(metadata["dip_deg"]),
            dip_direction=str(metadata["dip_direction"]),
            positive_side=str(metadata["positive_side"]),
            velocity_offset_km_per_s=float(metadata["velocity_offset_km_per_s"]),
        )
        velocity_settings = replace(velocity_settings, faulted=scenario_value)
    elif scenario == "salt_dome":
        scenario_value = replace(
            velocity_settings.salt_dome,
            center_km=tuple(float(value) for value in metadata["center_km"]),
            radii_km=tuple(float(value) for value in metadata["radii_km"]),
            body_velocity_km_per_s=float(metadata["body_velocity_km_per_s"]),
        )
        velocity_settings = replace(velocity_settings, salt_dome=scenario_value)
    elif scenario == "dyke_intrusion":
        scenario_value = replace(
            velocity_settings.dyke_intrusion,
            center_km=tuple(float(value) for value in metadata["center_km"]),
            strike_deg=float(metadata["strike_deg"]),
            length_km=float(metadata["length_km"]),
            width_km=float(metadata["width_km"]),
            top_depth_km=float(metadata["top_depth_km"]),
            bottom_depth_km=float(metadata["bottom_depth_km"]),
            body_velocity_km_per_s=float(metadata["body_velocity_km_per_s"]),
        )
        velocity_settings = replace(velocity_settings, dyke_intrusion=scenario_value)
    elif scenario != "layered":
        raise ValueError(f"Unsupported historical scenario {scenario!r}.")
    return replace(settings, velocity_model_generation=velocity_settings)


def _line_intersects_anomaly(
    source: Point3D,
    receiver: Point3D,
    target_grid: CartesianVelocityGrid3D,
    base_grid: CartesianVelocityGrid3D,
) -> bool:
    target_values = velocity_array_from_cartesian_grid(target_grid)
    base_values = velocity_array_from_cartesian_grid(base_grid)
    points = np.linspace(
        np.array([source.x_km, source.y_km, source.z_km]),
        np.array([receiver.x_km, receiver.y_km, receiver.z_km]),
        101,
    )
    axes = tuple(
        np.asarray(axis, dtype=float)
        for axis in (
            target_grid.x_coordinates_km,
            target_grid.y_coordinates_km,
            target_grid.z_coordinates_km,
        )
    )
    for point in points:
        indices = tuple(int(np.argmin(np.abs(axis - coordinate))) for axis, coordinate in zip(axes, point))
        if abs(target_values[indices] - base_values[indices]) > 1.0e-10:
            return True
    return False


def _distance(first: Point3D, second: Point3D) -> float:
    """Return the Euclidean source-receiver distance in kilometres."""
    return math.sqrt(
        (first.x_km - second.x_km) ** 2
        + (first.y_km - second.y_km) ** 2
        + (first.z_km - second.z_km) ** 2
    )


def _euclidean_time(first: Point3D, second: Point3D, velocity_km_per_s: float) -> float:
    """Return the analytical homogeneous-medium travel time."""
    if velocity_km_per_s <= 0.0:
        raise ValueError("Homogeneous velocity must be positive.")
    return _distance(first, second) / velocity_km_per_s


def _validate_spacings(spacings_km: Sequence[float]) -> tuple[float, ...]:
    """Validate and canonicalize the nested resolution-study spacings."""
    values = tuple(float(value) for value in spacings_km)
    if len(values) < 2:
        raise ValueError("At least two grid spacings are required for a resolution study.")
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("Resolution spacings must be finite and positive.")
    if len(set(values)) != len(values):
        raise ValueError("Resolution spacings must be unique.")
    return tuple(sorted(values, reverse=True))


def _resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_values(rows: Iterable[Mapping[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def _summary_statistics(rows: Iterable[Mapping[str, object]], field: str) -> dict[str, object]:
    """Summarize finite values without treating failed rows as zero."""
    values = _finite_values(rows, field)
    if not values:
        return {
            "field": field,
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "field": field,
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "p95": float(np.percentile(np.asarray(values, dtype=float), 95.0)),
        "maximum": max(values),
    }


def _group_summary(
    rows: Sequence[Mapping[str, object]],
    group_fields: Sequence[str],
    metric_field: str,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    summaries: list[dict[str, object]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        summary = _summary_statistics(group_rows, metric_field)
        summary.update({field: value for field, value in zip(group_fields, key)})
        summaries.append(summary)
    return summaries


def _status_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("status", "missing"))] += 1
    return dict(sorted(counts.items()))


def _package_versions() -> dict[str, str | None]:
    names = ("ttcrpy", "pykonal", "scikit-fmm", "numpy", "scipy", "vtk", "PyYAML")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_validation_summary(
    *,
    settings: BenchmarkSettings,
    analytic_rows: Sequence[Mapping[str, object]],
    algorithm_rows: Sequence[Mapping[str, object]],
    resolution_rows: Sequence[Mapping[str, object]],
    pykonal_rows: Sequence[Mapping[str, object]],
    scikit_fmm_rows: Sequence[Mapping[str, object]],
    legacy_rows: Sequence[Mapping[str, object]],
    resolution_spacings_km: Sequence[float],
) -> dict[str, object]:
    """Build a machine-readable summary for the authoritative-solver gate."""
    del settings
    resolution_deltas = _summary_statistics(
        resolution_rows,
        "absolute_change_from_next_finer_s",
    )
    algorithm_delta = _summary_statistics(algorithm_rows, "delta_from_FSM_rp1_s")
    algorithm_absolute_delta = _summary_statistics(
        algorithm_rows,
        "absolute_delta_from_FSM_rp1_s",
    )
    legacy_delta = _summary_statistics(legacy_rows, "legacy_absolute_difference_s")
    analytic_absolute = _summary_statistics(analytic_rows, "absolute_error_s")
    return {
        "artifact_type": "authoritative_forward_solver_validation_summary",
        "validation_id": VALIDATION_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": {
            "audit_only": True,
            "new_benchmark_v2_corpus_generated": False,
            "ml_models_fitted": False,
            "manuscript_modified": False,
            "test_set_used": False,
        },
        "package_versions": _package_versions(),
        "forward_grid_architecture": {
            "analytic_sampling": "scenario evaluated directly at requested forward-grid nodes",
            "historical_ml_target_grid_spacing_km": {"x": 5.0, "y": 5.0, "z": 2.5},
            "analytic_validation_spacing_km": sorted(
                {float(row["grid_spacing_km"]) for row in analytic_rows}
            ),
            "resolution_study_spacings_km": list(resolution_spacings_km),
            "target_grid_and_forward_grid_are_distinct": True,
            "velocity_representation": "node_centered_p_velocity",
        },
        "numerical_contract": {
            "ttcrpy_grid": "3D rectilinear node-coordinate grid",
            "ttcrpy_primary_candidate": "FSM with tt_from_rp enabled for path-integrated output",
            "endpoint_policy": "adapter rejects points outside the grid and records path endpoint residuals; no silent snapping",
            "convergence_interpretation": "finite successful return plus cross-resolution and algorithm stability; not a claim of continuum convergence",
        },
        "row_counts": {
            "analytic": len(analytic_rows),
            "algorithm_comparison": len(algorithm_rows),
            "resolution": len(resolution_rows),
            "pykonal": len(pykonal_rows),
            "scikit_fmm": len(scikit_fmm_rows),
            "legacy": len(legacy_rows),
        },
        "status_counts": {
            "analytic": _status_counts(analytic_rows),
            "algorithm_comparison": _status_counts(algorithm_rows),
            "resolution": _status_counts(resolution_rows),
            "pykonal": _status_counts(pykonal_rows),
            "scikit_fmm": _status_counts(scikit_fmm_rows),
            "legacy": _status_counts(legacy_rows),
        },
        "analytic_validation": {
            "overall_absolute_error_s": analytic_absolute,
            "by_reference_kind": _group_summary(
                analytic_rows, ("reference_kind", "configuration"), "absolute_error_s"
            ),
            "by_family_configuration": _group_summary(
                analytic_rows, ("family", "configuration"), "absolute_error_s"
            ),
        },
        "algorithm_comparison": {
            "delta_from_FSM_rp1_s": algorithm_delta,
            "absolute_delta_from_FSM_rp1_s": algorithm_absolute_delta,
            "by_family_configuration": _group_summary(
                algorithm_rows,
                ("family", "configuration"),
                "absolute_delta_from_FSM_rp1_s",
            ),
            "path_length_by_family_configuration": _group_summary(
                algorithm_rows,
                ("family", "configuration"),
                "path_length_km",
            ),
        },
        "resolution_convergence": {
            "absolute_change_from_next_finer_s": resolution_deltas,
            "by_family_spacing": _group_summary(
                resolution_rows,
                ("family", "grid_spacing_km"),
                "absolute_change_from_next_finer_s",
            ),
            "finest_spacing_km": min(resolution_spacings_km),
        },
        "pykonal_crosscheck": {
            "travel_time_difference_summary_s": _summary_statistics(
                pykonal_rows, "absolute_difference_s"
            ),
            "by_family": _group_summary(
                pykonal_rows, ("family",), "absolute_difference_s"
            ),
            "by_family_spacing": _group_summary(
                pykonal_rows,
                ("family", "grid_spacing_km"),
                "absolute_difference_s",
            ),
            "independent_implementation_status": _status_counts(pykonal_rows),
        },
        "scikit_fmm": {
            "status": _status_counts(scikit_fmm_rows),
            "interpretation": "secondary API smoke check only; not used as the authoritative reference",
        },
        "legacy_solver_comparison": {
            "absolute_difference_summary_s": legacy_delta,
            "by_family": _group_summary(legacy_rows, ("family",), "legacy_absolute_difference_s"),
            "by_anomaly_intersection": _group_summary(
                legacy_rows, ("anomaly_intersection",), "legacy_absolute_difference_s"
            ),
            "diagnostic_only": True,
        },
        "terminology": {
            "legacy_solver": "interface-aware two-point ray-path optimizer initialized by a Snell-law layered ray",
            "first_arrival_claim": "not granted by this audit until independent validation supports it",
        },
    }


def _format_stat_block(summary: Mapping[str, object]) -> str:
    return (
        f"n={summary.get('count', 0)}, mean={_format_number(summary.get('mean'))}, "
        f"median={_format_number(summary.get('median'))}, "
        f"p95={_format_number(summary.get('p95'))}, "
        f"max={_format_number(summary.get('maximum'))}"
    )


def _format_number(value: object) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def build_validation_summary_markdown(summary: Mapping[str, object]) -> str:
    """Render a concise human-readable companion to the JSON audit summary."""
    analytic = summary.get("analytic_validation", {})
    algorithm = summary.get("algorithm_comparison", {})
    resolution = summary.get("resolution_convergence", {})
    pykonal = summary.get("pykonal_crosscheck", {})
    legacy = summary.get("legacy_solver_comparison", {})
    lines = [
        "# Authoritative Forward-Solver Validation v1",
        "",
        "This is an audit artifact. It does not generate Benchmark v2 labels, fit ML models, "
        "or modify the manuscript.",
        "",
        "## Scope",
        "",
        f"- Package versions: `{summary.get('package_versions', {})}`",
        "- Forward fields are sampled directly from the analytic scenario on an independent "
        "forward grid; the historical ML target grid is not used as the authoritative numerical grid.",
        "- Successful finite returns and refinement/algorithm comparisons are reported; these "
        "are not described as continuum convergence.",
        "",
        "## Analytic validation",
        "",
        f"Overall absolute travel-time error: {_format_stat_block(analytic.get('overall_absolute_error_s', {}))} s.",
        "",
        "| Reference | Configuration | n | mean abs. error (s) | p95 (s) | max (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in analytic.get("by_reference_kind", []):
        lines.append(
            f"| {row.get('reference_kind')} | {row.get('configuration')} | {row.get('count', 0)} | "
            f"{_format_number(row.get('mean'))} | {_format_number(row.get('p95'))} | "
            f"{_format_number(row.get('maximum'))} |"
        )
    lines.extend(
        [
            "",
            "## Algorithm comparison",
            "",
            "Differences below are relative to ttcrpy FSM with `tt_from_rp=True` on the same grid and pairs; "
            "they are not comparisons against the legacy solver.",
            "",
            f"Absolute pairwise difference summary: {_format_stat_block(algorithm.get('absolute_delta_from_FSM_rp1_s', {}))} s.",
            "",
            "| Family | Configuration | n | mean signed delta (s) | p95 absolute delta (s) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in algorithm.get("by_family_configuration", []):
        lines.append(
            f"| {row.get('family')} | {row.get('configuration')} | {row.get('count', 0)} | "
            f"{_format_number(row.get('mean'))} | {_format_number(row.get('p95'))} |"
        )
    lines.extend(
        [
            "",
            "## Resolution study",
            "",
            f"Change to the next finer grid summary: {_format_stat_block(resolution.get('absolute_change_from_next_finer_s', {}))} s.",
            "",
            "| Family | Spacing (km) | n | mean change (s) | p95 (s) | max (s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in resolution.get("by_family_spacing", []):
        lines.append(
            f"| {row.get('family')} | {row.get('grid_spacing_km')} | {row.get('count', 0)} | "
            f"{_format_number(row.get('mean'))} | {_format_number(row.get('p95'))} | "
            f"{_format_number(row.get('maximum'))} |"
        )
    lines.extend(
        [
            "",
            "## Independent and legacy diagnostics",
            "",
            f"PyKonal travel-time difference: {_format_stat_block(pykonal.get('travel_time_difference_summary_s', {}))} s.",
            f"Legacy absolute difference: {_format_stat_block(legacy.get('absolute_difference_summary_s', {}))} s.",
            "",
            "PyKonal and scikit-fmm availability/runtime details are retained in their CSV/JSON outputs. "
            "PyKonal source-point seeding and its grid contract are recorded separately from ttcrpy.",
            "",
            "## Status",
            "",
            "The authoritative gate is issued separately in `solver_gate.md` after the validation "
            "artifacts and any independent cross-check outputs are reviewed.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_validation_metadata(
    *,
    settings: BenchmarkSettings,
    output_dir: Path,
    resolution_spacings_km: Sequence[float],
    pairs_per_family: int,
    algorithm_pairs_per_family: int,
    pykonal_pairs_per_family: int,
    legacy_cases_per_profile: int,
    legacy_pairs_per_case: int,
) -> dict[str, object]:
    """Record the exact audit design without serializing the full settings object."""
    del settings
    return {
        "artifact_type": "authoritative_forward_solver_validation_metadata",
        "validation_id": VALIDATION_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": "python -m tomobench.evaluation.authoritative_forward_solver_validation",
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "output_dir": str(output_dir),
        "seed": DEFAULT_SEED,
        "resolution_spacings_km": list(resolution_spacings_km),
        "pairs_per_family": pairs_per_family,
        "algorithm_pairs_per_family": algorithm_pairs_per_family,
        "pykonal_pairs_per_family": pykonal_pairs_per_family,
        "legacy_cases_per_profile": legacy_cases_per_profile,
        "legacy_pairs_per_case": legacy_pairs_per_case,
        "software_versions": _package_versions(),
        "scope_exclusions": [
            "no Benchmark v2 corpus generation",
            "no ML model fitting or test evaluation",
            "no manuscript modification",
        ],
    }


def rebuild_validation_summary_from_outputs(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Rebuild the summary after an external PyKonal worker has run.

    The primary ttcrpy run and the independent PyKonal runs deliberately use
    separate environments.  This helper keeps the raw per-run CSVs intact and
    replaces only the summary/plot companions with a merged, auditable view.
    """
    resolved = Path(output_dir)
    analytic_rows = _read_csv_rows(resolved / "analytic_validation.csv")
    algorithm_rows = _read_csv_rows(resolved / "algorithm_comparison.csv")
    resolution_rows = _read_csv_rows(resolved / "resolution_convergence.csv")
    scikit_rows = _read_csv_rows(resolved / "scikit_fmm_smoke.csv")
    legacy_rows = _read_csv_rows(resolved / "legacy_solver_comparison.csv")
    independent_paths = sorted(
        path
        for path in resolved.glob("pykonal_crosscheck_*.csv")
        if path.name != "pykonal_crosscheck.csv"
    )
    if independent_paths:
        pykonal_rows: list[dict[str, object]] = []
        for path in independent_paths:
            source_rows = _read_csv_rows(path)
            for row in source_rows:
                row["comparison_file"] = path.name
            pykonal_rows.extend(source_rows)
    else:
        pykonal_rows = _read_csv_rows(resolved / "pykonal_crosscheck.csv")
    metadata = _read_json(resolved / "validation_metadata.json")
    spacings = tuple(float(value) for value in metadata["resolution_spacings_km"])
    summary = build_validation_summary(
        settings=load_settings(),
        analytic_rows=analytic_rows,
        algorithm_rows=algorithm_rows,
        resolution_rows=resolution_rows,
        pykonal_rows=pykonal_rows,
        scikit_fmm_rows=scikit_rows,
        legacy_rows=legacy_rows,
        resolution_spacings_km=spacings,
    )
    summary["pykonal_crosscheck"]["comparison_files"] = [
        path.name for path in independent_paths
    ]
    summary["pykonal_crosscheck"]["environment_summary_files"] = [
        path.name for path in sorted(resolved.glob("pykonal_crosscheck_*_summary.json"))
    ]
    summary_path = resolved / "authoritative_forward_solver_validation_summary.json"
    _write_json(summary_path, summary)
    (resolved / "authoritative_forward_solver_validation_summary.md").write_text(
        build_validation_summary_markdown(summary),
        encoding="utf-8",
    )
    _write_discrepancy_plot(
        analytic_rows,
        algorithm_rows,
        legacy_rows,
        resolved / "travel_time_discrepancy_distribution.png",
    )
    return summary_path


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nempty\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_discrepancy_plot(
    analytic_rows: Sequence[Mapping[str, object]],
    algorithm_rows: Sequence[Mapping[str, object]],
    legacy_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    """Write a compact diagnostic plot without requiring pandas."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
        analytic_values = _finite_values(analytic_rows, "absolute_error_s")
        algorithm_values = [
            abs(value) for value in _finite_values(algorithm_rows, "delta_from_FSM_rp1_s")
        ]
        legacy_values = _finite_values(legacy_rows, "legacy_absolute_difference_s")
        series = (
            (axes[0], analytic_values, "Analytic reference", "|ttcrpy - reference| (s)"),
            (axes[1], algorithm_values, "Algorithm spread", "|delta from FSM rp1| (s)"),
            (axes[2], legacy_values, "Legacy diagnostic", "|legacy - ttcrpy| (s)"),
        )
        for axis, values, title, label in series:
            if values:
                axis.hist(values, bins=min(20, max(5, len(values))), color="#35618f", alpha=0.85)
                axis.set_xlabel(label)
                axis.set_ylabel("count")
            else:
                axis.text(0.5, 0.5, "No finite rows", ha="center", va="center")
                axis.set_xticks([])
                axis.set_yticks([])
            axis.set_title(title)
            axis.grid(alpha=0.25)
        figure.suptitle("Authoritative forward-solver validation diagnostics")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160)
        plt.close(figure)
    except Exception as exc:  # preserve the validation data even if plotting is unavailable
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".plot_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairs-per-family", type=int, default=8)
    parser.add_argument("--algorithm-pairs-per-family", type=int, default=5)
    parser.add_argument("--pykonal-pairs-per-family", type=int, default=4)
    parser.add_argument("--legacy-cases-per-profile", type=int, default=1)
    parser.add_argument("--legacy-pairs-per-case", type=int, default=3)
    parser.add_argument(
        "--resolution-spacing-km",
        type=float,
        nargs="+",
        default=list(DEFAULT_RESOLUTION_SPACINGS_KM),
    )
    args = parser.parse_args(argv)
    outputs = run_authoritative_forward_solver_validation(
        output_dir=args.output_dir,
        resolution_spacings_km=args.resolution_spacing_km,
        pairs_per_family=args.pairs_per_family,
        algorithm_pairs_per_family=args.algorithm_pairs_per_family,
        pykonal_pairs_per_family=args.pykonal_pairs_per_family,
        legacy_cases_per_profile=args.legacy_cases_per_profile,
        legacy_pairs_per_case=args.legacy_pairs_per_case,
    )
    print(outputs.summary_json)
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
