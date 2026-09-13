"""Focused final robustness diagnostics for the frozen benchmark.

The routines in this module consume the frozen production corpus and existing
solver-audit artifacts.  They do not retrain an estimator, change a split, or
select a result from the test set.  The only new forward calculations are the
small, predeclared local FSM finite-difference diagnostic around the reference
model.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tomobench.config import load_settings
from tomobench.domain.geometry import Point3D
from tomobench.evaluation.benchmark_v2_final_analysis import (
    ProductionCase,
    RayGeometry,
    load_production_cases,
)
from tomobench.evaluation.robustness_checks import (
    _bootstrap_mean_ci,
    _family_stratified_ci,
    _fmt,
    _load_geometry,
    _read_csv,
    _reference_cells,
    _stable_seed,
    _write_csv,
    _write_json,
    _write_markdown,
)
from tomobench.simulation.ttcrpy_forward import (
    TtcrpyGridConfiguration,
    TtcrpyRectilinearForwardSolver,
)


DISPLAY_FAMILIES = (
    "block_anomaly",
    "dyke_intrusion",
    "faulted",
    "layered",
    "salt_dome",
)
FAMILY_LABELS = {
    "block_anomaly": "Block anomaly",
    "dyke_intrusion": "Dyke intrusion",
    "faulted": "Faulted",
    "layered": "Layered",
    "salt_dome": "Salt dome",
}
JACOBIAN_EPSILONS_S_PER_KM = (0.005, 0.010)
JACOBIAN_ZERO_TOLERANCE_KM = 1.0e-10
GEOLOGICAL_SIGNAL_FLOOR_S = 0.010
SIGNAL_FRACTIONS_S = (0.010, 0.025, 0.050, 0.100)


def _cell_shape_from_case(case: ProductionCase) -> tuple[int, int, int]:
    return tuple(len(axis) - 1 for axis in (
        case.target_grid.x_coordinates_km,
        case.target_grid.y_coordinates_km,
        case.target_grid.z_coordinates_km,
    ))


def _cell_coordinates(
    cell_index: int,
    cell_shape: tuple[int, int, int],
    target_grid: Any,
) -> tuple[int, int, int, float, float, float]:
    nx, ny, nz = cell_shape
    plane = nx * ny
    iz, remainder = divmod(int(cell_index), plane)
    iy, ix = divmod(remainder, nx)
    x_axis = np.asarray(target_grid.x_coordinates_km, dtype=float)
    y_axis = np.asarray(target_grid.y_coordinates_km, dtype=float)
    z_axis = np.asarray(target_grid.z_coordinates_km, dtype=float)
    return (
        ix,
        iy,
        iz,
        float((x_axis[ix] + x_axis[ix + 1]) / 2.0),
        float((y_axis[iy] + y_axis[iy + 1]) / 2.0),
        float((z_axis[iz] + z_axis[iz + 1]) / 2.0),
    )


def _xfast_to_ttcr_field(values: np.ndarray, cell_shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = cell_shape
    return np.asarray(values, dtype=float).reshape((nz, ny, nx)).transpose(2, 1, 0)


def _local_fsm_configuration() -> TtcrpyGridConfiguration:
    """Return the adopted FSM numerical settings with cell slowness enabled."""

    return TtcrpyGridConfiguration(
        method="FSM",
        n_threads=1,
        cell_slowness=True,
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
    )


def _observation_selection_records(cases: Sequence[ProductionCase]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in sorted((item for item in cases if item.split == "test"), key=lambda item: item.target_id):
        for index, (observation, observation_id) in enumerate(
            zip(case.observations, case.observation_ids, strict=True)
        ):
            midpoint_x = (observation[0] + observation[3]) / 2.0
            midpoint_y = (observation[1] + observation[4]) / 2.0
            midpoint_radius = float(np.hypot(midpoint_x - 50.0, midpoint_y - 50.0))
            records.append(
                {
                    "target_id": case.target_id,
                    "family": case.family,
                    "observation_index": index,
                    "observation_id": observation_id,
                    "source_x_km": float(observation[0]),
                    "source_y_km": float(observation[1]),
                    "source_z_km": float(observation[2]),
                    "receiver_x_km": float(observation[3]),
                    "receiver_y_km": float(observation[4]),
                    "receiver_z_km": float(observation[5]),
                    "offset_km": float(observation[6]),
                    "midpoint_radius_km": midpoint_radius,
                }
            )
    if not records:
        raise ValueError("The local Jacobian diagnostic requires test observations.")
    offsets = np.asarray([row["offset_km"] for row in records], dtype=float)
    depths = np.asarray([row["source_z_km"] for row in records], dtype=float)
    radii = np.asarray([row["midpoint_radius_km"] for row in records], dtype=float)
    q1, q2 = np.percentile(offsets, (100.0 / 3.0, 200.0 / 3.0))
    depth_median = float(np.median(depths))
    radius_median = float(np.median(radii))
    for row in records:
        offset = float(row["offset_km"])
        row["path_length_class"] = (
            "short" if offset <= q1 else "medium" if offset <= q2 else "long"
        )
        row["source_depth_class"] = "shallow" if row["source_z_km"] <= depth_median else "deep"
        row["geometry_class"] = (
            "central" if row["midpoint_radius_km"] <= radius_median else "boundary"
        )
    return records


def _select_jacobian_pairs(
    cases: Sequence[ProductionCase],
    geometry_dir: Path,
) -> list[dict[str, Any]]:
    """Select 3 x 2 x 2 geometry categories without using travel-time results."""

    records = _observation_selection_records(cases)
    eligible: list[dict[str, Any]] = []
    for row in records:
        geometry_path = geometry_dir / f"{row['target_id']}.npz"
        if not geometry_path.exists():
            continue
        with np.load(geometry_path) as artifact:
            row_index = int(row["observation_index"])
            start = int(artifact["row_ptr"][row_index])
            end = int(artifact["row_ptr"][row_index + 1])
            positive_count = int(
                np.count_nonzero(np.asarray(artifact["path_lengths_km"][start:end]) > 0.0)
            )
        if positive_count >= 5:
            row = dict(row)
            row["positive_reference_cell_count"] = positive_count
            eligible.append(row)
    records = eligible
    if not records:
        raise ValueError("No geometry-only candidate has five positive reference-path cells.")
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()
    categories = (
        (path_class, depth_class, geometry_class)
        for path_class in ("short", "medium", "long")
        for depth_class in ("shallow", "deep")
        for geometry_class in ("central", "boundary")
    )
    for path_class, depth_class, geometry_class in categories:
        exact = [
            row
            for row in records
            if row["path_length_class"] == path_class
            and row["source_depth_class"] == depth_class
            and row["geometry_class"] == geometry_class
            and (row["target_id"], row["observation_index"]) not in used
        ]
        candidates = exact or [
            row
            for row in records
            if (row["target_id"], row["observation_index"]) not in used
        ]
        if not candidates:
            raise ValueError(f"No candidate remains for {path_class}/{depth_class}/{geometry_class}.")

        def mismatch(row: Mapping[str, Any]) -> tuple[int, float, str, int]:
            category_mismatch = sum(
                (
                    row["path_length_class"] != path_class,
                    row["source_depth_class"] != depth_class,
                    row["geometry_class"] != geometry_class,
                )
            )
            return (
                int(category_mismatch),
                float(row["offset_km"]),
                str(row["target_id"]),
                int(row["observation_index"]),
            )

        chosen = min(candidates, key=mismatch)
        chosen = dict(chosen)
        chosen["selection_category"] = f"{path_class}_{depth_class}_{geometry_class}"
        chosen["selection_exact_category_available"] = bool(exact)
        selected.append(chosen)
        used.add((str(chosen["target_id"]), int(chosen["observation_index"])))
    return selected


def _select_jacobian_cells(
    geometry: RayGeometry,
    row_index: int,
    case: ProductionCase,
) -> list[dict[str, Any]]:
    """Select depth and path-contribution cells using only reference geometry."""

    start = int(geometry.row_ptr[row_index])
    end = int(geometry.row_ptr[row_index + 1])
    cells = [
        (int(cell), float(length))
        for cell, length in zip(
            geometry.cell_indices[start:end],
            geometry.path_lengths_km[start:end],
            strict=True,
        )
        if float(length) > 0.0
    ]
    if len(cells) < 5:
        raise ValueError(f"{case.target_id} observation {row_index} has fewer than five positive cells.")
    cell_shape = _cell_shape_from_case(case)
    enriched = []
    for cell_index, path_length in cells:
        ix, iy, iz, x, y, z = _cell_coordinates(cell_index, cell_shape, case.target_grid)
        enriched.append(
            {
                "cell_index": cell_index,
                "ix": ix,
                "iy": iy,
                "iz": iz,
                "cell_x_km": x,
                "cell_y_km": y,
                "cell_z_km": z,
                "reference_path_length_km": path_length,
            }
        )
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for role, target_depth in (
        ("shallow_depth", 5.0),
        ("mid_depth", 15.0),
        ("deep_depth", 25.0),
    ):
        candidate = min(
            (row for row in enriched if row["cell_index"] not in used),
            key=lambda row: (abs(row["cell_z_km"] - target_depth), row["cell_index"]),
        )
        selected.append({**candidate, "selection_role": role})
        used.add(int(candidate["cell_index"]))
    high = max(
        (row for row in enriched if row["cell_index"] not in used),
        key=lambda row: (-row["reference_path_length_km"], row["cell_index"]),
    )
    selected.append({**high, "selection_role": "high_path_length"})
    used.add(int(high["cell_index"]))
    low = min(
        (row for row in enriched if row["cell_index"] not in used),
        key=lambda row: (row["reference_path_length_km"], row["cell_index"]),
    )
    selected.append({**low, "selection_role": "low_nonzero_path_length"})
    return selected


def _sign_class(value: float) -> int:
    if abs(value) <= JACOBIAN_ZERO_TOLERANCE_KM:
        return 0
    return 1 if value > 0.0 else -1


def _jacobian_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fsm = np.asarray([float(row["fsm_fd_sensitivity_km"]) for row in rows], dtype=float)
    reference = np.asarray([float(row["fixed_ray_path_sensitivity_km"]) for row in rows], dtype=float)
    absolute_difference = np.abs(fsm - reference)
    relative = np.asarray(
        [
            float(row["relative_difference"])
            for row in rows
            if row.get("relative_difference") not in (None, "")
        ],
        dtype=float,
    )
    sign_matches = sum(_sign_class(a) == _sign_class(b) for a, b in zip(fsm, reference, strict=True))
    zero_matches = sum(
        (abs(a) <= JACOBIAN_ZERO_TOLERANCE_KM) == (abs(b) <= JACOBIAN_ZERO_TOLERANCE_KM)
        for a, b in zip(fsm, reference, strict=True)
    )
    correlation = (
        float(np.corrcoef(reference, fsm)[0, 1])
        if len(rows) > 1 and np.std(reference) > 0.0 and np.std(fsm) > 0.0
        else math.nan
    )
    return {
        "n_entries": len(rows),
        "correlation": correlation,
        "mae_km": float(np.mean(absolute_difference)),
        "median_relative_difference": float(np.median(relative)) if relative.size else math.nan,
        "p95_relative_difference": float(np.percentile(relative, 95)) if relative.size else math.nan,
        "relative_entry_count": int(relative.size),
        "sign_agreement_fraction": sign_matches / len(rows),
        "zero_nonzero_agreement_fraction": zero_matches / len(rows),
    }


def plot_fsm_fixed_ray_jacobian_scatter(
    entries: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot the local FSM versus fixed-ray sensitivity scatter (Supplementary Figure S2).

    Reads only the stored diagnostic entries, so the figure can be replotted from the
    frozen records without re-running the FSM finite-difference diagnostic.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.0, 5.0))
    for epsilon in JACOBIAN_EPSILONS_S_PER_KM:
        values = [row for row in entries if float(row["epsilon_s_per_km"]) == epsilon]
        axis.scatter(
            [float(row["fixed_ray_path_sensitivity_km"]) for row in values],
            [float(row["fsm_fd_sensitivity_km"]) for row in values],
            s=24,
            alpha=0.7,
            label=rf"$\varepsilon$ = {epsilon:g} s/km",
        )
    limits = [
        min(float(row["fixed_ray_path_sensitivity_km"]) for row in entries),
        max(float(row["fixed_ray_path_sensitivity_km"]) for row in entries),
    ]
    axis.plot(limits, limits, "k--", linewidth=1.0, label="1:1")
    axis.set_xlabel(r"$G_{\mathrm{ref}}$ path sensitivity (km)")
    axis.set_ylabel("FSM finite-difference sensitivity (km)")
    axis.set_title("Local FSM versus fixed-ray sensitivity")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output_dir / "fsm_fixed_ray_jacobian_scatter.png"
    # 600 dpi keeps the 6.0 x 5.0 in figure above 300 dpi at MDPI full text width.
    figure.savefig(path, dpi=600)
    figure.savefig(output_dir / "fsm_fixed_ray_jacobian_scatter.svg")
    plt.close(figure)
    return path


def run_fsm_fixed_ray_jacobian_diagnostic(
    production_dir: Path,
    evidence_dir: Path,
    output_dir: Path,
    paper_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Run the predeclared 12-pair/5-cell/two-epsilon local diagnostic."""

    summary_path = output_dir / "fsm_fixed_ray_jacobian_summary.json"
    if summary_path.exists() and not force:
        return summary_path
    settings = load_settings()
    cases = load_production_cases(production_dir)
    case_by_id = {case.target_id: case for case in cases}
    geometry_dir = evidence_dir / "classical" / "reference_ray_geometry"
    pairs = _select_jacobian_pairs(cases, geometry_dir)
    pair_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        case = case_by_id[str(pair["target_id"])]
        geometry_path = geometry_dir / f"{case.target_id}.npz"
        if not geometry_path.exists():
            raise FileNotFoundError(f"Missing frozen reference geometry: {geometry_path}")
        geometry = _load_geometry(geometry_path)
        row_index = int(pair["observation_index"])
        source = Point3D(
            float(pair["source_x_km"]),
            float(pair["source_y_km"]),
            float(pair["source_z_km"]),
        )
        receiver = Point3D(
            float(pair["receiver_x_km"]),
            float(pair["receiver_y_km"]),
            float(pair["receiver_z_km"]),
        )
        reference_cells = _reference_cells(case, settings)
        cell_shape = _cell_shape_from_case(case)
        base_slowness = _xfast_to_ttcr_field(1.0 / reference_cells, cell_shape)
        solver = TtcrpyRectilinearForwardSolver(
            case.target_grid.x_coordinates_km,
            case.target_grid.y_coordinates_km,
            case.target_grid.z_coordinates_km,
            base_slowness,
            _local_fsm_configuration(),
            values_kind="slowness",
        )
        base_time = float(solver.raytrace([source], [receiver]).travel_times_s[0])
        cells = _select_jacobian_cells(geometry, row_index, case)
        pair_rows.append(
            {
                **pair,
                "base_fsm_time_s": base_time,
                "selected_cell_count": len(cells),
            }
        )
        for cell in cells:
            cell_rows.append(
                {
                    "target_id": case.target_id,
                    "family": case.family,
                    "observation_index": row_index,
                    "observation_id": pair["observation_id"],
                    **cell,
                }
            )
            for epsilon in JACOBIAN_EPSILONS_S_PER_KM:
                perturbed = base_slowness.copy()
                perturbed[cell["ix"], cell["iy"], cell["iz"]] += epsilon
                solver.grid.set_slowness(perturbed)
                perturbed_time = float(solver.raytrace([source], [receiver]).travel_times_s[0])
                fsm_sensitivity = (perturbed_time - base_time) / epsilon
                fixed_sensitivity = float(cell["reference_path_length_km"])
                signed_difference = fsm_sensitivity - fixed_sensitivity
                relative_difference = (
                    abs(signed_difference) / abs(fixed_sensitivity)
                    if abs(fixed_sensitivity) > JACOBIAN_ZERO_TOLERANCE_KM
                    else None
                )
                entries.append(
                    {
                        "target_id": case.target_id,
                        "family": case.family,
                        "observation_index": row_index,
                        "observation_id": pair["observation_id"],
                        "selection_category": pair["selection_category"],
                        "cell_index": cell["cell_index"],
                        "cell_selection_role": cell["selection_role"],
                        "cell_x_km": cell["cell_x_km"],
                        "cell_y_km": cell["cell_y_km"],
                        "cell_z_km": cell["cell_z_km"],
                        "reference_path_length_km": cell["reference_path_length_km"],
                        "epsilon_s_per_km": epsilon,
                        "base_fsm_time_s": base_time,
                        "perturbed_fsm_time_s": perturbed_time,
                        "fsm_fd_sensitivity_km": fsm_sensitivity,
                        "fixed_ray_path_sensitivity_km": fixed_sensitivity,
                        "signed_difference_fsm_minus_fixed_km": signed_difference,
                        "absolute_difference_km": abs(signed_difference),
                        "relative_difference": relative_difference,
                        "fsm_sign": _sign_class(fsm_sensitivity),
                        "fixed_ray_sign": _sign_class(fixed_sensitivity),
                    }
                )
        solver.grid.set_slowness(base_slowness)

    summary_by_epsilon = {}
    for epsilon in JACOBIAN_EPSILONS_S_PER_KM:
        summary_by_epsilon[str(epsilon)] = _jacobian_summary(
            [row for row in entries if float(row["epsilon_s_per_km"]) == epsilon]
        )
    combined = _jacobian_summary(entries)
    _write_csv(output_dir / "fsm_fixed_ray_jacobian_selected_pairs.csv", pair_rows)
    _write_csv(output_dir / "fsm_fixed_ray_jacobian_selected_cells.csv", cell_rows)
    _write_csv(output_dir / "fsm_fixed_ray_jacobian_entries.csv", entries)
    _write_json(
        {
            "artifact_type": "local_fsm_fixed_ray_jacobian_diagnostic",
            "selection_scope": "38 frozen test targets; geometry-only selection",
            "pair_count": len(pair_rows),
            "cells_per_pair": 5,
            "entry_count": len(entries),
            "epsilon_s_per_km": list(JACOBIAN_EPSILONS_S_PER_KM),
            "zero_tolerance_km": JACOBIAN_ZERO_TOLERANCE_KM,
            "selection_rule": {
                "path_classes": "global test-observation offset tertiles",
                "source_depth_classes": "global test-observation source-depth median",
                "geometry_classes": "global test-observation horizontal midpoint-radius median",
                "cells": "nearest positive-path cells to 5/15/25 km depth, then longest remaining and shortest remaining positive path length",
                "uses_finite_difference_results": False,
            },
            "fsm_operator": "auxiliary target-grid 40x40x12 cell-slowness ttcrpy FSM, matching the G_ref cell parameterization",
            "summary_by_epsilon": summary_by_epsilon,
            "combined_summary": combined,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        },
        summary_path,
    )
    try:
        plot_fsm_fixed_ray_jacobian_scatter(entries, output_dir)
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        _write_json({"plot_error": repr(exc)}, output_dir / "fsm_fixed_ray_jacobian_plot_error.json")

    _write_markdown(
        paper_dir / "fsm_fixed_ray_jacobian_diagnostic.md",
        _jacobian_markdown(pair_rows, cell_rows, entries, summary_by_epsilon, combined),
    )
    return summary_path


def _jacobian_markdown(
    pairs: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    summary_by_epsilon: Mapping[str, Mapping[str, Any]],
    combined: Mapping[str, Any],
) -> str:
    lines = [
        "# FSM/fixed-ray local Jacobian consistency diagnostic",
        "",
        "## Predeclared diagnostic contract",
        "",
        "This is a small local consistency diagnostic, not a full 384 x 19,200 Jacobian. It was",
        "selected before finite-difference results were evaluated. The path subset contains one",
        "source-receiver pair for every combination of short/medium/long offset (global test-observation",
        "tertiles), shallow/deep source (global source-depth median), and central/boundary horizontal",
        "midpoint (global midpoint-radius median): 3 x 2 x 2 = 12 pairs. The selection uses only saved",
        "coordinates, distances, target IDs, and deterministic tie ordering; it does not use travel times",
        "or finite-difference results. A candidate was eligible only when its saved G_ref row contained",
        "at least five positive cells, so the requested five cell roles could be assigned without",
        "reusing a cell; this is a geometry-only eligibility condition.",
        "",
        "For every selected path, five positive reference-ray cells were selected: the positive-path cell",
        "nearest 5 km, 15 km, and 25 km depth, followed by the longest remaining path contribution and",
        "the shortest remaining nonzero contribution. This rule is applied to the saved G_ref geometry",
        "before perturbation. The two one-sided slowness perturbations were epsilon = 0.005 and 0.010",
        "s/km. They are finite-difference stability checks, not tuned values.",
        "",
        "The FSM calculation is an auxiliary target-grid 40 x 40 x 12 cell-slowness ttcrpy FSM solve",
        "with the adopted FSM controls. This parameterization aligns the perturbed cell with the",
        "fixed-ray column. It is a local alignment diagnostic; it is not a replacement for or a claim",
        "that the production 1.25-km node-centered label operator has an explicitly calculated full",
        "Jacobian.",
        "",
        "## Selected pairs",
        "",
        "| Category | Target | Family | Observation | Offset (km) | Source depth (km) | Midpoint radius (km) | Exact category available |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in pairs:
        lines.append(
            f"| {row['selection_category']} | {row['target_id']} | {FAMILY_LABELS.get(str(row['family']), row['family'])} | "
            f"{row['observation_id']} | {_fmt(row['offset_km'], 5)} | {_fmt(row['source_z_km'], 5)} | "
            f"{_fmt(row['midpoint_radius_km'], 5)} | {row['selection_exact_category_available']} |"
        )
    lines.extend(
        [
            "",
            "## Entry-level results",
            "",
            "The complete entry table, including both travel times and signed differences, is in",
            "`fsm_fixed_ray_jacobian_entries.csv`. The fixed-ray sensitivity is the saved path length",
            "for the selected cell. The FSM sensitivity is `(t_FSM(s0 + epsilon e_j) -",
            "t_FSM(s0))/epsilon`; signed differences below are FSM minus fixed-ray.",
            "",
            "| Epsilon (s/km) | FSM finite difference (km) | G_ref (km) | Signed difference (km) | Absolute difference (km) | Relative difference |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in entries:
        lines.append(
            f"| {_fmt(row['epsilon_s_per_km'], 5)} | {_fmt(row['fsm_fd_sensitivity_km'], 7)} | "
            f"{_fmt(row['fixed_ray_path_sensitivity_km'], 7)} | {_fmt(row['signed_difference_fsm_minus_fixed_km'], 7)} | "
            f"{_fmt(row['absolute_difference_km'], 7)} | {_fmt(row['relative_difference'], 7)} |"
        )
    lines.extend(
        [
            "",
            "## Agreement summary",
            "",
            "| Epsilon | n | Correlation | MAE (km) | Median relative difference | p95 relative difference | Sign agreement | Zero/nonzero agreement |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for epsilon, summary in summary_by_epsilon.items():
        lines.append(
            f"| {epsilon} | {summary['n_entries']} | {_fmt(summary['correlation'], 6)} | "
            f"{_fmt(summary['mae_km'], 6)} | {_fmt(summary['median_relative_difference'], 6)} | "
            f"{_fmt(summary['p95_relative_difference'], 6)} | {_fmt(summary['sign_agreement_fraction'], 6)} | "
            f"{_fmt(summary['zero_nonzero_agreement_fraction'], 6)} |"
        )
    lines.append(
        f"| combined | {combined['n_entries']} | {_fmt(combined['correlation'], 6)} | "
        f"{_fmt(combined['mae_km'], 6)} | {_fmt(combined['median_relative_difference'], 6)} | "
        f"{_fmt(combined['p95_relative_difference'], 6)} | {_fmt(combined['sign_agreement_fraction'], 6)} | "
        f"{_fmt(combined['zero_nonzero_agreement_fraction'], 6)} |"
    )
    lines.extend(
        [
            "",
            "![G_ref sensitivity versus FSM finite-difference sensitivity](../../outputs/generated/submission_readiness/robustness_checks/fsm_fixed_ray_jacobian/fsm_fixed_ray_jacobian_scatter.png)",
            "",
            "The interpretation is deliberately limited to the reported local subset and the",
            "auxiliary cell-slowness FSM parameterization. It does not establish a globally exact FSM",
            "Jacobian or nonlinear equivalence of the fixed-ray inversion.",
        ]
    )
    return "\n".join(lines)


def _distribution_row(values: Sequence[float], family: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "family": family,
        "n": int(array.size),
        "median_abs_delta_t_geo_s": float(np.percentile(array, 50)),
        "p25_abs_delta_t_geo_s": float(np.percentile(array, 25)),
        "p75_abs_delta_t_geo_s": float(np.percentile(array, 75)),
        "p90_abs_delta_t_geo_s": float(np.percentile(array, 90)),
        "p95_abs_delta_t_geo_s": float(np.percentile(array, 95)),
        "maximum_abs_delta_t_geo_s": float(np.max(array)),
        **{
            f"fraction_abs_delta_t_geo_below_{str(limit).replace('.', 'p')}_s": float(
                np.mean(array < limit)
            )
            for limit in SIGNAL_FRACTIONS_S
        },
    }


def _audit_stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "n": 0,
            "median": math.nan,
            "p75": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "maximum": math.nan,
        }
    return {
        "n": int(array.size),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def write_forward_error_signal_report(
    production_dir: Path,
    solver_dir: Path,
    output_dir: Path,
    paper_dir: Path,
) -> Path:
    """Summarize production geological signal and existing numerical diagnostics separately."""

    production_values: dict[str, list[float]] = defaultdict(list)
    for case in load_production_cases(production_dir):
        for row in _read_csv(production_dir / "observations" / f"{case.target_id}_observations.csv"):
            signal = row.get("delta_t_geology_s")
            if signal in (None, ""):
                signal = float(row["travel_time_s"]) - float(row["background_travel_time_s"])
            production_values[case.family].append(abs(float(signal)))
    production_rows = [
        _distribution_row(production_values[family], family)
        for family in DISPLAY_FAMILIES
    ]
    refinement = _read_csv(solver_dir / "error_budget_refinement.csv")
    refinement = [
        row
        for row in refinement
        if row.get("representation") == "node_velocity"
        and abs(float(row["coarse_spacing_km"]) - 2.5) < 1.0e-12
        and abs(float(row["fine_spacing_km"]) - 1.25) < 1.0e-12
    ]
    refinement_detail: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []
    for family in DISPLAY_FAMILIES:
        family_rows = [row for row in refinement if row["family"] == family]
        errors = [float(row["absolute_refinement_difference_s"]) for row in family_rows]
        ratios = []
        for row in family_rows:
            signal = abs(float(row["absolute_geological_signal_s"]))
            error = abs(float(row["absolute_refinement_difference_s"]))
            ratio = error / signal if signal >= GEOLOGICAL_SIGNAL_FLOOR_S else None
            ratios.append(ratio) if ratio is not None else None
            refinement_detail.append(
                {
                    "family": family,
                    "pair_id": row["pair_id"],
                    "representation": row["representation"],
                    "coarse_spacing_km": row["coarse_spacing_km"],
                    "fine_spacing_km": row["fine_spacing_km"],
                    "absolute_refinement_error_s": error,
                    "absolute_geological_signal_s": signal,
                    "signal_floor_s": GEOLOGICAL_SIGNAL_FLOOR_S,
                    "error_over_signal": ratio,
                    "ratio_is_reportable": ratio is not None,
                }
            )
        error_stats = _audit_stats(errors)
        ratio_array = np.asarray(ratios, dtype=float)
        refinement_rows.append(
            {
                "family": family,
                "audit_n": len(family_rows),
                "reportable_ratio_n": int(ratio_array.size),
                "signal_floor_s": GEOLOGICAL_SIGNAL_FLOOR_S,
                "median_absolute_refinement_error_s": error_stats["median"],
                "p95_absolute_refinement_error_s": error_stats["p95"],
                "maximum_absolute_refinement_error_s": error_stats["maximum"],
                "median_error_over_signal": float(np.percentile(ratio_array, 50)) if ratio_array.size else None,
                "p75_error_over_signal": float(np.percentile(ratio_array, 75)) if ratio_array.size else None,
                "p90_error_over_signal": float(np.percentile(ratio_array, 90)) if ratio_array.size else None,
                "p95_error_over_signal": float(np.percentile(ratio_array, 95)) if ratio_array.size else None,
                "maximum_error_over_signal": float(np.max(ratio_array)) if ratio_array.size else None,
                **{
                    f"fraction_error_over_signal_{label}": (
                        float(np.mean(ratio_array < limit)) if ratio_array.size else None
                    )
                    for label, limit in (("lt_0p1", 0.1), ("lt_0p25", 0.25), ("lt_0p5", 0.5), ("lt_1", 1.0))
                },
                "fraction_error_over_signal_gt_1": (
                    float(np.mean(ratio_array > 1.0)) if ratio_array.size else None
                ),
            }
        )

    pykonal = _read_csv(solver_dir / "pykonal_fsm_crosscheck_rp0_1p25_v1.csv")
    pykonal_rows: list[dict[str, Any]] = []
    for family in DISPLAY_FAMILIES:
        family_rows = [row for row in pykonal if row["family"] == family]
        errors = np.asarray([abs(float(row["absolute_difference_s"])) for row in family_rows], dtype=float)
        pykonal_rows.append(
            {
                "family": family,
                "n": int(errors.size),
                "median_absolute_difference_s": float(np.median(errors)),
                "p95_absolute_difference_s": float(np.percentile(errors, 95)),
                "maximum_absolute_difference_s": float(np.max(errors)),
                "definition": "independent PyKonal versus ttcrpy FSM finite-grid cross-check; not continuum truth",
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "production_geological_signal_by_family.csv", production_rows)
    _write_csv(output_dir / "refinement_error_signal_by_family.csv", refinement_rows)
    _write_csv(output_dir / "refinement_error_signal_detail.csv", refinement_detail)
    _write_csv(output_dir / "pykonal_crosscheck_by_family.csv", pykonal_rows)
    path = paper_dir / "forward_error_signal_by_family.md"
    _write_markdown(path, _forward_error_signal_markdown(production_rows, refinement_rows, pykonal_rows))
    return path


def _forward_error_signal_markdown(
    production_rows: Sequence[Mapping[str, Any]],
    refinement_rows: Sequence[Mapping[str, Any]],
    pykonal_rows: Sequence[Mapping[str, Any]],
) -> str:
    production_by_family = {str(row["family"]): row for row in production_rows}
    refinement_by_family = {str(row["family"]): row for row in refinement_rows}
    lines = [
        "# Forward-model error versus geological signal by family",
        "",
        "## Definitions and separation of quantities",
        "",
        "The geological signal is defined for every one of the 96,000 production observation rows as",
        "`Delta t_geo = t_FSM(s_target) - t_FSM(s_background/reference)` using the adopted 1.25-km",
        "node-centered finite-grid operator. The report summarizes its absolute value. For the layered",
        "family, the adopted target and common background are the same construction, so the operator",
        "signal is exactly zero by design; this is not a claim that natural layered geology has no",
        "travel-time contrast.",
        "",
        "Numerical discrepancy is not inferred for every production row. The existing authoritative",
        "2.5-to-1.25-km node-velocity refinement audit supplies three diagnostic paths per family.",
        "Its absolute refinement difference is labeled below as a finite-grid diagnostic, not as",
        "continuum error. The independent PyKonal comparison is reported separately for the same reason.",
        "No incompatible numerical-error definitions are combined into a single production error.",
        "",
        "The ratio `R = |e_num|/|Delta t_geo|` is reported only when the audit-path denominator is at",
        f"least {GEOLOGICAL_SIGNAL_FLOOR_S:g} s. This floor was declared before summarizing ratios: it",
        "is below the 0.025-s smallest tested timing-noise level but excludes near-zero denominators",
        "whose ratios are numerically unstable. Absolute signal and absolute discrepancy remain",
        "reported for all paths.",
        "",
        "## Production geological-signal distribution",
        "",
        "| Family | n | Median |Delta t_geo| (s) | p25 (s) | p75 (s) | p90 (s) | p95 (s) | Maximum (s) | <0.01 s | <0.025 s | <0.05 s | <0.10 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in production_rows:
        lines.append(
            f"| {FAMILY_LABELS[row['family']]} | {row['n']} | {_fmt(row['median_abs_delta_t_geo_s'], 6)} | "
            f"{_fmt(row['p25_abs_delta_t_geo_s'], 6)} | {_fmt(row['p75_abs_delta_t_geo_s'], 6)} | "
            f"{_fmt(row['p90_abs_delta_t_geo_s'], 6)} | {_fmt(row['p95_abs_delta_t_geo_s'], 6)} | "
            f"{_fmt(row['maximum_abs_delta_t_geo_s'], 6)} | "
            f"{_fmt(row['fraction_abs_delta_t_geo_below_0p01_s'], 5)} | "
            f"{_fmt(row['fraction_abs_delta_t_geo_below_0p025_s'], 5)} | "
            f"{_fmt(row['fraction_abs_delta_t_geo_below_0p05_s'], 5)} | "
            f"{_fmt(row['fraction_abs_delta_t_geo_below_0p1_s'], 5)} |"
        )
    lines.extend(
        [
            "",
            "## Existing 2.5-to-1.25-km refinement diagnostic",
            "",
            "The following table uses three node-velocity audit paths per family. `R` is conditional",
            "on the 0.010-s signal floor; an `NA` ratio means that no audit path in that family exceeded",
            "the declared floor.",
            "",
            "| Family | Audit n | Reportable R n | Median absolute error (s) | p95 absolute error (s) | Median R | p75 R | p90 R | p95 R | R<0.1 | R<0.25 | R<0.5 | R<1 | R>1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in refinement_rows:
        lines.append(
            f"| {FAMILY_LABELS[row['family']]} | {row['audit_n']} | {row['reportable_ratio_n']} | "
            f"{_fmt(row['median_absolute_refinement_error_s'], 6)} | {_fmt(row['p95_absolute_refinement_error_s'], 6)} | "
            f"{_fmt(row['median_error_over_signal'], 6)} | {_fmt(row['p75_error_over_signal'], 6)} | "
            f"{_fmt(row['p90_error_over_signal'], 6)} | {_fmt(row['p95_error_over_signal'], 6)} | "
            f"{_fmt(row['fraction_error_over_signal_lt_0p1'], 5)} | {_fmt(row['fraction_error_over_signal_lt_0p25'], 5)} | "
            f"{_fmt(row['fraction_error_over_signal_lt_0p5'], 5)} | {_fmt(row['fraction_error_over_signal_lt_1'], 5)} | "
            f"{_fmt(row['fraction_error_over_signal_gt_1'], 5)} |"
        )
    lines.extend(
        [
            "",
            "## Suggested manuscript-ready separation",
            "",
            "| Family | Median |Delta t_geo| (s) | p95 |Delta t_geo| (s) | Numerical diagnostic | Median diagnostic error (s) | Median error/signal | Fraction error/signal >1 |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for family in DISPLAY_FAMILIES:
        signal = production_by_family[family]
        audit = refinement_by_family[family]
        diagnostic = "2.5 to 1.25 km node-grid refinement; n=3 audit paths"
        lines.append(
            f"| {FAMILY_LABELS[family]} | {_fmt(signal['median_abs_delta_t_geo_s'], 6)} | "
            f"{_fmt(signal['p95_abs_delta_t_geo_s'], 6)} | {diagnostic} | "
            f"{_fmt(audit['median_absolute_refinement_error_s'], 6)} | {_fmt(audit['median_error_over_signal'], 6)} | "
            f"{_fmt(audit['fraction_error_over_signal_gt_1'], 5)} |"
        )
    lines.extend(
        [
            "",
            "## Independent PyKonal cross-check",
            "",
            "This is an independent finite-grid implementation comparison at 1.25 km, not a validated",
            "continuum reference.",
            "",
            "| Family | n | Median absolute difference (s) | p95 (s) | Maximum (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in pykonal_rows:
        lines.append(
            f"| {FAMILY_LABELS[row['family']]} | {row['n']} | {_fmt(row['median_absolute_difference_s'], 6)} | "
            f"{_fmt(row['p95_absolute_difference_s'], 6)} | {_fmt(row['maximum_absolute_difference_s'], 6)} |"
        )
    lines.extend(
        [
            "",
            "The diagnostic supports a Level 1 interpretation: numerical uncertainty is often",
            "comparable to the target-induced timing signal for the affected structures and is not",
            "uniformly bounded below that signal. `R > 1` does not mean that a label is wrong; it means",
            "that, under this finite-grid diagnostic, numerical uncertainty is at least as large as the",
            "target-induced timing perturbation for that sampled path.",
        ]
    )
    return "\n".join(lines)


def _paired_direct_summary(
    rows: Sequence[Mapping[str, Any]],
    first_method: str,
    second_method: str,
    comparison_id: str,
) -> dict[str, Any]:
    by_target: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_target[str(row["target_id"])][str(row["method_id"])] = row
    deltas: list[dict[str, Any]] = []
    for target_id, methods in sorted(by_target.items()):
        if first_method not in methods or second_method not in methods:
            continue
        deltas.append(
            {
                "comparison_id": comparison_id,
                "target_id": target_id,
                "family": methods[first_method]["family"],
                "delta_rmse_km_per_s": float(methods[first_method]["direct_cell_center_rmse_km_per_s"])
                - float(methods[second_method]["direct_cell_center_rmse_km_per_s"]),
            }
        )
    values = np.asarray([float(row["delta_rmse_km_per_s"]) for row in deltas], dtype=float)
    tolerance = 1.0e-12
    wins = int(np.sum(values < -tolerance))
    losses = int(np.sum(values > tolerance))
    ties = int(values.size - wins - losses)
    pooled = _bootstrap_mean_ci(values, _stable_seed(f"focused-paired-{comparison_id}"))
    stratified = _family_stratified_ci(
        deltas,
        _stable_seed(f"focused-paired-family-{comparison_id}"),
    )
    return {
        "comparison_id": comparison_id,
        "first_method_id": first_method,
        "second_method_id": second_method,
        "n": int(values.size),
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "pooled_ci_lower": pooled[0],
        "pooled_ci_upper": pooled[1],
        "family_stratified_ci_lower": stratified[0],
        "family_stratified_ci_upper": stratified[1],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_p_two_sided": _exact_sign_test_p(wins, losses),
        "deltas": deltas,
    }


def _exact_sign_test_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = sum(math.comb(n, index) for index in range(min(wins, losses) + 1))
    return float(min(1.0, 2.0 * tail / (2.0**n)))


def write_final_travel_time_information_decision(
    corrected_dir: Path,
    travel_dir: Path,
    paper_dir: Path,
) -> Path:
    """Write the final paired and frozen-model travel-time interpretation."""

    method_rows = _read_csv(corrected_dir / "corrected_test_method_metrics.csv")
    summaries = {
        row["comparison_id"]: row
        for row in _read_csv(corrected_dir / "corrected_paired_summary.csv")
    }
    comparison_specs = (
        ("realistic_vs_no_travel_time", "realistic_full", "no_travel_time"),
        ("realistic_vs_shuffled_travel_time", "realistic_full", "shuffled_travel_time"),
        ("realistic_vs_travel_time_only", "realistic_full", "travel_time_only"),
        ("realistic_vs_training_target_mean", "realistic_full", "training_target_mean"),
    )
    direct_summaries: list[dict[str, Any]] = []
    for comparison_id, first, second in comparison_specs:
        row = summaries[comparison_id]
        direct_summaries.append(
            {
                "comparison_id": comparison_id,
                "label": f"{first} - {second}",
                "n": int(row["test_target_count"]),
                "mean_delta": float(row["mean_delta_km_per_s"]),
                "median_delta": float(row["median_delta_km_per_s"]),
                "pooled_ci_lower": float(row["pooled_ci95_lower_km_per_s"]),
                "pooled_ci_upper": float(row["pooled_ci95_upper_km_per_s"]),
                "family_stratified_ci_lower": float(row["family_stratified_ci95_lower_km_per_s"]),
                "family_stratified_ci_upper": float(row["family_stratified_ci95_upper_km_per_s"]),
                "wins": int(row["wins"]),
                "losses": int(row["losses"]),
                "ties": int(row["ties"]),
                "sign_test_p_two_sided": _exact_sign_test_p(int(row["wins"]), int(row["losses"])),
            }
        )
    travel_mean = _paired_direct_summary(
        method_rows,
        "travel_time_only",
        "training_target_mean",
        "travel_time_only_vs_training_target_mean",
    )
    travel_mean["label"] = "travel_time_only - training_target_mean"
    travel_mean.pop("deltas", None)
    direct_summaries.append(travel_mean)
    coefficients = {
        row["feature_group"]: float(row["standardized_coefficient_frobenius_norm"])
        for row in _read_csv(travel_dir / "travel_time_coefficient_diagnostics.csv")
    }
    counterfactual_rows = _read_csv(travel_dir / "travel_time_counterfactual_metrics.csv")
    variants = sorted({str(row["variant"]) for row in counterfactual_rows})
    counterfactual_summary = []
    true_mean = float(
        np.mean(
            [
                float(row["cell_rmse_km_per_s"])
                for row in counterfactual_rows
                if row["variant"] == "true_travel_time"
            ]
        )
    )
    for variant in variants:
        values = np.asarray(
            [float(row["cell_rmse_km_per_s"]) for row in counterfactual_rows if row["variant"] == variant],
            dtype=float,
        )
        counterfactual_summary.append(
            {
                "variant": variant,
                "n": int(values.size),
                "mean_historical_averaged_cell_rmse": float(np.mean(values)),
                "mean_delta_from_true_travel_time": float(np.mean(values) - true_mean),
                "median_delta_from_true_travel_time": float(
                    np.median(values - np.asarray(
                        [
                            float(row["cell_rmse_km_per_s"])
                            for row in counterfactual_rows
                            if row["variant"] == "true_travel_time"
                        ],
                        dtype=float,
                    ))
                    if variant != "true_travel_time"
                    else 0.0
                ),
            }
        )
    _write_csv(travel_dir / "final_travel_time_paired_summary.csv", direct_summaries)
    _write_csv(travel_dir / "final_travel_time_counterfactual_summary.csv", counterfactual_summary)
    path = paper_dir / "final_travel_time_information_decision.md"
    _write_markdown(path, _travel_time_markdown(direct_summaries, coefficients, counterfactual_summary))
    return path


def _travel_time_markdown(
    summaries: Sequence[Mapping[str, Any]],
    coefficients: Mapping[str, float],
    counterfactual_summary: Sequence[Mapping[str, Any]],
) -> str:
    label_map = {
        "realistic_vs_no_travel_time": "Full input - no time",
        "realistic_vs_shuffled_travel_time": "Full input - shuffled time",
        "realistic_vs_travel_time_only": "Full input - travel-time-only",
        "realistic_vs_training_target_mean": "Full input - training-target mean",
        "travel_time_only_vs_training_target_mean": "Travel-time-only - training-target mean",
    }
    lines = [
        "# Final travel-time information decision",
        "",
        "The active comparisons use direct analytic cell-center truth, target-level RMSE, the frozen",
        "38-target test set, and the existing paired bootstrap contract. Negative deltas favor the first",
        "method. A two-sided exact sign-test p-value is included as a frequency diagnostic; it is not",
        "used as a replacement for effect sizes or confidence intervals.",
        "",
        "| Comparison | Mean delta (km/s) | Median delta (km/s) | Pooled 95% CI | Family-stratified 95% CI | Wins/losses/ties | Sign-test p |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {label_map.get(row['comparison_id'], row['label'])} | {_fmt(row['mean_delta'], 6)} | "
            f"{_fmt(row['median_delta'], 6)} | [{_fmt(row['pooled_ci_lower'], 6)}, {_fmt(row['pooled_ci_upper'], 6)}] | "
            f"[{_fmt(row['family_stratified_ci_lower'], 6)}, {_fmt(row['family_stratified_ci_upper'], 6)}] | "
            f"{row['wins']}/{row['losses']}/{row['ties']} | {_fmt(row['sign_test_p_two_sided'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## Frozen-model dependence diagnostics",
            "",
            "The frozen selected full-input PCA-ridge model was not retrained for these substitutions.",
            "The standardized coefficient-group norms are descriptive. The travel-time group norm is",
            f"{_fmt(coefficients.get('travel_time_s'), 6)}, compared with {_fmt(coefficients.get('euclidean_distance_km'), 6)} for",
            "Euclidean distance. Frozen counterfactual summaries below use the historical eight-corner",
            "averaged-cell metric stored by the dependence diagnostic; they are sensitivity evidence, not",
            "a replacement for the active direct-cell endpoint.",
            "",
            "| Frozen substitution | n | Mean historical averaged-cell RMSE (km/s) | Mean delta from true time (km/s) | Median delta from true time (km/s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in counterfactual_summary:
        lines.append(
            f"| {row['variant']} | {row['n']} | {_fmt(row['mean_historical_averaged_cell_rmse'], 6)} | "
            f"{_fmt(row['mean_delta_from_true_travel_time'], 6)} | {_fmt(row['median_delta_from_true_travel_time'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "**A TRAVEL-TIME ASSOCIATION EFFECT IS SUPPORTED.** The full input has lower direct-cell",
            "RMSE than the no-time and shuffled-time controls, with pooled and family-stratified intervals",
            "below zero. The direct full-input versus travel-time-only interval includes zero, and the",
            "full-input versus training-target-mean interval also includes zero. The travel-time-only",
            "versus training-target-mean paired comparison is reported explicitly in the table rather than",
            "being inferred from separate method means.",
            "",
            "The supported statement is association-level and conditional on the selected PCA-ridge",
            "representation, synthetic corpus, and finite-grid operator. It is not a causal attribution",
            "of physical information to the travel-time column, nor evidence that travel-time-only is",
            "superior to the full input or that PCA-ridge resolves the full inverse problem.",
        ]
    )
    return "\n".join(lines)


def write_reference_prior_interpretation(corrected_dir: Path, paper_dir: Path) -> Path:
    """Describe the heterogeneous full-input versus reference-prior comparison."""

    paired = {
        row["comparison_id"]: row
        for row in _read_csv(corrected_dir / "corrected_paired_summary.csv")
    }["realistic_vs_reference_prior"]
    deltas = [
        row
        for row in _read_csv(corrected_dir / "corrected_paired_deltas.csv")
        if row["comparison_id"] == "realistic_vs_reference_prior"
    ]
    family_rows: list[dict[str, Any]] = []
    for family in DISPLAY_FAMILIES:
        values = np.asarray(
            [float(row["delta_rmse_km_per_s"]) for row in deltas if row["family"] == family],
            dtype=float,
        )
        family_rows.append(
            {
                "family": family,
                "n": int(values.size),
                "mean_delta": float(np.mean(values)),
                "median_delta": float(np.median(values)),
                "wins": int(np.sum(values < -1.0e-12)),
                "losses": int(np.sum(values > 1.0e-12)),
                "ties": int(np.sum(np.abs(values) <= 1.0e-12)),
            }
        )
    _write_csv(corrected_dir / "reference_prior_family_effects.csv", family_rows)
    path = paper_dir / "reference_prior_interpretation.md"
    lines = [
        "# Reference-prior comparison interpretation",
        "",
        "The active direct-cell comparison is realistic full-input PCA-ridge minus the reference",
        "1-D prior; negative values favor the full-input estimator.",
        "",
        f"The pooled mean paired delta is **{_fmt(paired['mean_delta_km_per_s'], 6)} km/s**, while the",
        f"median delta is **{_fmt(paired['median_delta_km_per_s'], 6)} km/s**. The pooled 95% CI is",
        f"**[{_fmt(paired['pooled_ci95_lower_km_per_s'], 6)}, {_fmt(paired['pooled_ci95_upper_km_per_s'], 6)}] km/s**,",
        f"and the family-stratified interval is **[{_fmt(paired['family_stratified_ci95_lower_km_per_s'], 6)},",
        f"{_fmt(paired['family_stratified_ci95_upper_km_per_s'], 6)}] km/s**. Target-level wins/losses/ties",
        f"are **{paired['wins']}/{paired['losses']}/{paired['ties']}** for the full-input method, with exact",
        f"two-sided sign-test p = **{_fmt(_exact_sign_test_p(int(paired['wins']), int(paired['losses'])), 6)}**.",
        "",
        "| Family | n | Mean delta (km/s) | Median delta (km/s) | Full-input wins | Full-input losses | Ties |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in family_rows:
        lines.append(
            f"| {FAMILY_LABELS[row['family']]} | {row['n']} | {_fmt(row['mean_delta'], 6)} | "
            f"{_fmt(row['median_delta'], 6)} | {row['wins']} | {row['losses']} | {row['ties']} |"
        )
    lines.extend(
        [
            "",
            "The comparison is heterogeneous and does not support a blanket superiority claim. Mean",
            "error magnitude and win/loss frequency answer different questions: the mean delta weights",
            "the size of each target-level difference, whereas wins count how often the first method has",
            "the smaller RMSE. A smaller number of large positive deltas can therefore coexist with a",
            "larger number of small negative deltas, producing a positive median or a different mean",
            "direction. The pooled and family-stratified intervals can also differ because stratified",
            "resampling preserves the family structure and within-family variability rather than treating",
            "the observed mixture as an undifferentiated block.",
            "",
            "Allowed wording: **the full-input versus reference-prior comparison was heterogeneous, with",
            "the pooled interval including zero; no uniform superiority is established**. Stronger wording",
            "that either method is generally or physically superior is not supported.",
        ]
    )
    _write_markdown(path, "\n".join(lines))
    return path


def write_clipping_summary_report(
    diagnostics_path: Path,
    paper_dir: Path,
    *,
    ml_diagnostics_path: Path | None = None,
) -> Path:
    """Summarize boundary clipping without inventing unavailable ML pre-clip values."""

    rows = _read_csv(diagnostics_path)
    if ml_diagnostics_path is not None:
        rows.extend(
            row
            for row in _read_csv(ml_diagnostics_path)
            if row["method_id"] != "fixed_ray"
        )
    summaries: list[dict[str, Any]] = []
    fixed = [row for row in rows if row["method_id"] == "fixed_ray"]
    summaries.append(
        {
            "method_id": "fixed_ray",
            "n_targets": len(fixed),
            "lower_bound_fraction_mean": float(np.mean([float(row["postclip_lower_boundary_fraction"]) for row in fixed])),
            "upper_bound_fraction_mean": float(np.mean([float(row["postclip_upper_boundary_fraction"]) for row in fixed])),
            "total_clipping_fraction_mean": float(np.mean([float(row["postclip_changed_fraction"]) for row in fixed])),
            "total_clipping_fraction_max_per_target": float(np.max([float(row["postclip_changed_fraction"]) for row in fixed])),
            "preclip_velocity_minimum_km_per_s": float(np.min([float(row["preclip_velocity_min_km_per_s"]) for row in fixed])),
            "preclip_velocity_maximum_km_per_s": float(np.max([float(row["preclip_velocity_max_km_per_s"]) for row in fixed])),
            "preclip_range_available": True,
        }
    )
    for method in sorted({row["method_id"] for row in rows if row["method_id"] != "fixed_ray"}):
        selected = [row for row in rows if row["method_id"] == method]
        lower = np.asarray([float(row["ml_lower_boundary_fraction"]) for row in selected], dtype=float)
        upper = np.asarray([float(row["ml_upper_boundary_fraction"]) for row in selected], dtype=float)
        summaries.append(
            {
                "method_id": method,
                "n_targets": len(selected),
                "lower_bound_fraction_mean": float(np.mean(lower)),
                "upper_bound_fraction_mean": float(np.mean(upper)),
                "total_clipping_fraction_mean": float(np.mean(lower + upper)),
                "total_clipping_fraction_max_per_target": float(np.max(lower + upper)),
                "preclip_velocity_minimum_km_per_s": None,
                "preclip_velocity_maximum_km_per_s": None,
                "preclip_range_available": False,
            }
        )
    output_path = diagnostics_path.parent / "clipping_summary.csv"
    _write_csv(output_path, summaries)
    path = paper_dir / "clipping_summary.md"
    fixed_summary = summaries[0]
    realistic_summary = next(
        row for row in summaries if row["method_id"] == "realistic_full"
    )
    lines = [
        "# Reference-ray and PCA-ridge clipping summary",
        "",
        "The reference-ray rows retain raw slowness and pre-clipped reciprocal velocity, so their changed",
        "fraction is directly measurable. All 38 reference-ray solves converged, and the raw",
        "slowness zero/negative fraction was zero for every target. PCA-ridge prediction artifacts store",
        "the post-clip node velocities rather than raw unclipped predictions; consequently, their table",
        "entries are bound-hit fractions, reported as the available clipping diagnostic. An ML pre-clip",
        "velocity range and a changed-from-raw fraction cannot be recovered from the released arrays.",
        "",
        "| Method | Targets | Mean lower-bound fraction | Mean upper-bound fraction | Mean total clipping/bound-hit fraction | Max per-target fraction | Pre-clipped velocity range (km/s) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    clipping_labels = {"fixed_ray": "Reference-ray baseline", "realistic_full": "Full-input PCA-ridge"}
    for row in summaries:
        preclip = (
            f"[{_fmt(row['preclip_velocity_minimum_km_per_s'], 6)}, {_fmt(row['preclip_velocity_maximum_km_per_s'], 6)}]"
            if row["preclip_range_available"]
            else "not persisted"
        )
        lines.append(
            f"| {clipping_labels.get(row['method_id'], row['method_id'])} | {row['n_targets']} | {_fmt(row['lower_bound_fraction_mean'], 6)} | "
            f"{_fmt(row['upper_bound_fraction_mean'], 6)} | {_fmt(row['total_clipping_fraction_mean'], 6)} | "
            f"{_fmt(row['total_clipping_fraction_max_per_target'], 6)} | {preclip} |"
        )
    lines.extend(
        [
            "",
            f"The reference-ray mean total changed fraction is approximately {_fmt(fixed_summary['total_clipping_fraction_mean'], 6)} "
            f"and the maximum per-target fraction is approximately {_fmt(fixed_summary['total_clipping_fraction_max_per_target'], 6)}. "
            "For the full-input PCA-ridge method,",
            f"the mean post-clip bound-hit fraction is approximately {_fmt(realistic_summary['total_clipping_fraction_mean'], 6)} and the maximum is",
            f"approximately {_fmt(realistic_summary['total_clipping_fraction_max_per_target'], 6)}. These small fractions do not justify omitting the diagnostic, but",
            "they also do not appear large enough to explain the primary RMSE ordering by themselves.",
        ]
    )
    _write_markdown(path, "\n".join(lines))
    return path
