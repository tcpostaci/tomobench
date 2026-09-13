"""Final Phase 6 classical-versus-ML comparison helpers."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from tomobench.domain.schemas import CartesianVelocityGrid3D
from tomobench.evaluation.metrics import mean_absolute_error, root_mean_squared_error
from tomobench.tomography.corpus import (
    DEFAULT_KNOWN_RAY_CORPUS_OUTPUT_DIR,
    DEFAULT_ML_SUITE_SUMMARY,
)
from tomobench.tomography.reference import DEFAULT_REFERENCE_RAY_CORPUS_OUTPUT_DIR
from tomobench.utils.paths import get_repo_root

DEFAULT_FINAL_PHASE6_COMPARISON_OUTPUT_DIR = Path(
    "outputs/generated/classical_tomography/final_phase6_comparison_v1"
)


@dataclass(frozen=True)
class FinalPhase6ComparisonOutputs:
    """Artifacts written by the final Phase 6 comparison audit."""

    output_dir: Path
    summary_json: Path
    summary_csv: Path
    ml_case_metrics_csv: Path
    clipping_csv: Path


def run_final_phase6_comparison(
    output_dir: Path = DEFAULT_FINAL_PHASE6_COMPARISON_OUTPUT_DIR,
    known_ray_summary_path: Path = DEFAULT_KNOWN_RAY_CORPUS_OUTPUT_DIR
    / "known_ray_corpus_summary.json",
    known_ray_case_summary_path: Path = DEFAULT_KNOWN_RAY_CORPUS_OUTPUT_DIR
    / "known_ray_corpus_case_summary.csv",
    reference_ray_summary_path: Path = DEFAULT_REFERENCE_RAY_CORPUS_OUTPUT_DIR
    / "reference_ray_corpus_summary.json",
    reference_ray_case_summary_path: Path = DEFAULT_REFERENCE_RAY_CORPUS_OUTPUT_DIR
    / "reference_ray_corpus_case_summary.csv",
    ml_metrics_path: Path = DEFAULT_ML_SUITE_SUMMARY.parent
    / "models/pca-linear/fixed_split/metrics.json",
) -> FinalPhase6ComparisonOutputs:
    """Write a harmonized final Phase 6 comparison artifact."""
    repo_root = get_repo_root()
    output_dir = _resolve_path(output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    known_summary = _read_json(_resolve_path(known_ray_summary_path, repo_root))
    reference_summary = _read_json(_resolve_path(reference_ray_summary_path, repo_root))
    known_case_rows = _read_csv_rows(_resolve_path(known_ray_case_summary_path, repo_root))
    reference_case_rows = _read_csv_rows(_resolve_path(reference_ray_case_summary_path, repo_root))
    ml_case_rows = compute_ml_cell_centered_case_metrics(
        _resolve_path(ml_metrics_path, repo_root),
        repo_root,
        reference_case_rows=reference_case_rows,
    )
    clipping_rows = (
        _classical_clipping_rows(
            known_case_rows,
            cell_csv_name="known_ray_coverage_smoothing_cell_velocity_predictions.csv",
            method="known_ray",
        )
        + _classical_clipping_rows(
            reference_case_rows,
            cell_csv_name="reference_ray_cell_velocity_predictions.csv",
            method="reference_ray",
        )
    )
    summary = _summary_payload(
        known_summary=known_summary,
        reference_summary=reference_summary,
        ml_case_rows=ml_case_rows,
        clipping_rows=clipping_rows,
        source_paths={
            "known_ray_summary": known_ray_summary_path.as_posix(),
            "known_ray_case_summary": known_ray_case_summary_path.as_posix(),
            "reference_ray_summary": reference_ray_summary_path.as_posix(),
            "reference_ray_case_summary": reference_ray_case_summary_path.as_posix(),
            "ml_metrics": ml_metrics_path.as_posix(),
        },
    )
    summary_json = output_dir / "final_phase6_comparison_summary.json"
    summary_csv = output_dir / "final_phase6_comparison_summary.csv"
    ml_case_metrics_csv = output_dir / "ml_pca_linear_cell_centered_case_metrics.csv"
    clipping_csv = output_dir / "classical_clipping_diagnostics.csv"
    _write_json(summary_json, summary)
    _write_summary_csv(summary_csv, summary)
    _write_csv(ml_case_metrics_csv, ml_case_rows)
    _write_csv(clipping_csv, clipping_rows)
    return FinalPhase6ComparisonOutputs(
        output_dir=output_dir,
        summary_json=summary_json,
        summary_csv=summary_csv,
        ml_case_metrics_csv=ml_case_metrics_csv,
        clipping_csv=clipping_csv,
    )


def node_centered_values_to_cell_centered(
    node_values: Sequence[float],
    nx: int,
    ny: int,
    nz: int,
) -> tuple[float, ...]:
    """Average node-centered values onto cells using x-fastest grid order."""
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("Grid shape must contain at least two nodes on each axis.")
    if len(node_values) != nx * ny * nz:
        raise ValueError("Node value length does not match grid shape.")
    cells: list[float] = []
    for iz in range(nz - 1):
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                cells.append(
                    sum(
                        float(node_values[_node_index(ix + dx, iy + dy, iz + dz, nx, ny)])
                        for dz in (0, 1)
                        for dy in (0, 1)
                        for dx in (0, 1)
                    )
                    / 8.0
                )
    return tuple(cells)


def compute_ml_cell_centered_case_metrics(
    ml_metrics_path: Path,
    repo_root: Path | None = None,
    reference_case_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert selected ML node predictions to cell-centered case metrics."""
    root = repo_root or get_repo_root()
    metrics = _read_json(ml_metrics_path)
    reference_coverage_by_case = _reference_coverage_by_case(reference_case_rows or [])
    rows: list[dict[str, Any]] = []
    split_metrics = metrics.get("split_metrics")
    if not isinstance(split_metrics, dict):
        raise ValueError("ML metrics JSON must contain split_metrics.")
    for split, split_payload in split_metrics.items():
        if not isinstance(split_payload, dict):
            continue
        case_metrics = split_payload.get("case_metrics")
        if not isinstance(case_metrics, list):
            continue
        for case in case_metrics:
            if not isinstance(case, dict):
                continue
            prediction_path = _resolve_path(Path(str(case["prediction_path"])), root)
            prediction = _read_json(prediction_path)
            target_grid_path = _resolve_path(Path(str(case["target_grid_path"])), root)
            grid = _load_velocity_grid(target_grid_path)
            predicted_nodes = tuple(float(value) for value in prediction["predicted_p_velocity_km_per_s"])
            nx = len(grid.x_coordinates_km)
            ny = len(grid.y_coordinates_km)
            nz = len(grid.z_coordinates_km)
            predicted_cells = node_centered_values_to_cell_centered(predicted_nodes, nx, ny, nz)
            target_cells = node_centered_values_to_cell_centered(
                grid.p_velocity_km_per_s,
                nx,
                ny,
                nz,
            )
            coverage_mask = reference_coverage_by_case.get(str(case["case_id"]))
            covered_metrics = _covered_metrics(target_cells, predicted_cells, coverage_mask)
            rows.append(
                {
                    "case_id": str(case["case_id"]),
                    "family": str(case.get("scenario") or "unknown"),
                    "split": str(split),
                    "all_cell_velocity_rmse_km_per_s": root_mean_squared_error(
                        target_cells,
                        predicted_cells,
                    ),
                    "all_cell_velocity_mae_km_per_s": mean_absolute_error(
                        target_cells,
                        predicted_cells,
                    ),
                    "reference_covered_cell_count": covered_metrics["covered_cell_count"],
                    "reference_covered_cell_velocity_rmse_km_per_s": covered_metrics[
                        "velocity_rmse_km_per_s"
                    ],
                    "reference_covered_cell_velocity_mae_km_per_s": covered_metrics[
                        "velocity_mae_km_per_s"
                    ],
                    "node_centered_original_rmse_km_per_s": float(case["rmse"]),
                    "node_centered_original_mae_km_per_s": float(case["mae"]),
                    "prediction_path": prediction_path.as_posix(),
                    "target_grid_path": target_grid_path.as_posix(),
                }
            )
    return rows


def _summary_payload(
    known_summary: dict[str, Any],
    reference_summary: dict[str, Any],
    ml_case_rows: list[dict[str, Any]],
    clipping_rows: list[dict[str, Any]],
    source_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "artifact_type": "final_phase6_classical_ml_comparison_audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_files": source_paths,
        "metric_contract": {
            "classical_velocity_metrics": "cell-centered velocity over Cartesian cells between adjacent nodes",
            "ml_cell_centered_metrics": "ML node-centered predictions converted to cell-centered values by averaging each cell's eight surrounding nodes",
            "ml_original_metrics": "node-centered full-grid velocity metrics retained only for traceability",
            "covered_cell_compatibility": "ML covered-cell-compatible metrics reuse the reference-ray coverage mask per case",
        },
        "audit_findings": [
            "The earlier reference-ray implementation used laterally averaged case target grids; it has been corrected to use configured layered background velocities sampled on each case coordinate grid.",
            "The corrected reference-ray outputs still have high clipping counts, so low velocity RMSE should be reviewed alongside clipping and coverage diagnostics.",
            "Known-ray and reference-ray classical metrics are cell-centered; ML metrics in this artifact have been harmonized to the same all-cell cell-centered contract.",
            "This remains synthetic-only and does not constitute real-data validation or full nonlinear tomography.",
        ],
        "known_ray_classical": _compact_classical_summary(known_summary),
        "reference_ray_classical": _compact_classical_summary(reference_summary),
        "ml_pca_linear_cell_centered": _ml_summary(ml_case_rows),
        "clipping_diagnostics": _clipping_summary(clipping_rows),
        "comparison_caution": [
            "Known-ray inversion is optimistic because ray geometry comes from the true synthetic model.",
            "Reference-ray inversion is fairer but still uses a fixed configured layered background and one fixed regularization pair.",
            "The selected ML model is compared through a harmonized cell-centered conversion, but it is not a classical iterative inversion.",
            "Do not claim ML improves classical tomography unless these harmonized metrics and clipping diagnostics are reviewed and accepted.",
        ],
    }


def _ml_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "splits": sorted({str(row["split"]) for row in rows}),
        "all_splits": _aggregate_metrics(rows),
        "split_wise_metrics": {
            split: _aggregate_metrics([row for row in rows if row["split"] == split])
            for split in sorted({str(row["split"]) for row in rows})
        },
        "family_wise_metrics": {
            family: _aggregate_metrics([row for row in rows if row["family"] == family])
            for family in sorted({str(row["family"]) for row in rows})
        },
    }


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_cell_velocity_rmse_km_per_s": _aggregate(rows, "all_cell_velocity_rmse_km_per_s"),
        "all_cell_velocity_mae_km_per_s": _aggregate(rows, "all_cell_velocity_mae_km_per_s"),
        "reference_covered_cell_velocity_rmse_km_per_s": _aggregate(
            rows,
            "reference_covered_cell_velocity_rmse_km_per_s",
        ),
        "reference_covered_cell_velocity_mae_km_per_s": _aggregate(
            rows,
            "reference_covered_cell_velocity_mae_km_per_s",
        ),
        "node_centered_original_rmse_km_per_s": _aggregate(
            rows,
            "node_centered_original_rmse_km_per_s",
        ),
        "node_centered_original_mae_km_per_s": _aggregate(
            rows,
            "node_centered_original_mae_km_per_s",
        ),
    }


def _clipping_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {}
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        by_method[method] = {
            "sample_count": len(method_rows),
            "clipped_cell_count": _aggregate(method_rows, "clipped_cell_count"),
            "clipped_cell_percentage": _aggregate(method_rows, "clipped_cell_percentage"),
            "covered_clipped_cell_count": _aggregate(method_rows, "covered_clipped_cell_count"),
            "uncovered_clipped_cell_count": _aggregate(
                method_rows,
                "uncovered_clipped_cell_count",
            ),
            "family_wise": {
                family: {
                    "sample_count": len(family_rows),
                    "clipped_cell_count": _aggregate(family_rows, "clipped_cell_count"),
                    "clipped_cell_percentage": _aggregate(
                        family_rows,
                        "clipped_cell_percentage",
                    ),
                    "covered_clipped_cell_count": _aggregate(
                        family_rows,
                        "covered_clipped_cell_count",
                    ),
                    "uncovered_clipped_cell_count": _aggregate(
                        family_rows,
                        "uncovered_clipped_cell_count",
                    ),
                }
                for family, family_rows in _group_by(method_rows, "family").items()
            },
        }
    return by_method


def _classical_clipping_rows(
    case_rows: list[dict[str, Any]],
    cell_csv_name: str,
    method: str,
) -> list[dict[str, Any]]:
    rows = []
    for case in case_rows:
        summary_path = Path(str(case["summary_json"]))
        summary = _read_json(summary_path)
        lower, upper = (float(value) for value in summary["velocity_bounds_km_per_s"])
        min_slowness = 1.0 / upper
        max_slowness = 1.0 / lower
        cell_rows = _read_csv_rows(summary_path.parent / cell_csv_name)
        clipped_count = 0
        covered_clipped_count = 0
        uncovered_clipped_count = 0
        for row in cell_rows:
            slowness = float(row["predicted_slowness_s_per_km"])
            clipped = not math.isclose(
                slowness,
                min(max(slowness, min_slowness), max_slowness),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            if not clipped:
                continue
            clipped_count += 1
            if _as_bool(row["covered_by_threshold"]):
                covered_clipped_count += 1
            else:
                uncovered_clipped_count += 1
        cell_count = len(cell_rows)
        rows.append(
            {
                "method": method,
                "case_id": str(case["case_id"]),
                "family": str(case["family"]),
                "cell_count": cell_count,
                "clipped_cell_count": clipped_count,
                "clipped_cell_percentage": clipped_count / cell_count if cell_count else 0.0,
                "covered_clipped_cell_count": covered_clipped_count,
                "uncovered_clipped_cell_count": uncovered_clipped_count,
                "covered_clipped_fraction_of_clipped": (
                    covered_clipped_count / clipped_count if clipped_count else 0.0
                ),
            }
        )
    return rows


def _reference_coverage_by_case(case_rows: list[dict[str, Any]]) -> dict[str, tuple[bool, ...]]:
    coverage_by_case = {}
    for row in case_rows:
        summary_path = Path(str(row["summary_json"]))
        coverage_path = summary_path.parent / "reference_ray_cell_coverage.csv"
        if not coverage_path.is_file():
            continue
        coverage_by_case[str(row["case_id"])] = tuple(
            _as_bool(item["covered_by_threshold"]) for item in _read_csv_rows(coverage_path)
        )
    return coverage_by_case


def _covered_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
    coverage_mask: tuple[bool, ...] | None,
) -> dict[str, float | int | None]:
    if coverage_mask is None:
        return {
            "covered_cell_count": None,
            "velocity_rmse_km_per_s": None,
            "velocity_mae_km_per_s": None,
        }
    actual_covered = [value for value, keep in zip(actual, coverage_mask, strict=True) if keep]
    predicted_covered = [
        value for value, keep in zip(predicted, coverage_mask, strict=True) if keep
    ]
    if not actual_covered:
        return {
            "covered_cell_count": 0,
            "velocity_rmse_km_per_s": None,
            "velocity_mae_km_per_s": None,
        }
    return {
        "covered_cell_count": len(actual_covered),
        "velocity_rmse_km_per_s": root_mean_squared_error(actual_covered, predicted_covered),
        "velocity_mae_km_per_s": mean_absolute_error(actual_covered, predicted_covered),
    }


def _compact_classical_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": summary["sample_count"],
        "diagnostic_type": summary["diagnostic_type"],
        "travel_time_rmse_s": summary["travel_time_rmse_s"],
        "travel_time_mae_s": summary["travel_time_mae_s"],
        "all_cell_velocity_rmse_km_per_s": summary["all_cell_velocity_rmse_km_per_s"],
        "all_cell_velocity_mae_km_per_s": summary["all_cell_velocity_mae_km_per_s"],
        "covered_cell_velocity_rmse_km_per_s": summary[
            "covered_cell_velocity_rmse_km_per_s"
        ],
        "covered_cell_velocity_mae_km_per_s": summary[
            "covered_cell_velocity_mae_km_per_s"
        ],
        "coverage_fraction": summary["coverage_fraction"],
        "clipped_velocity_cell_count": summary["clipped_velocity_cell_count"],
        "family_wise_metrics": summary["family_wise_metrics"],
    }


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for method_key in ("known_ray_classical", "reference_ray_classical"):
        method = summary[method_key]
        rows.extend(_summary_method_rows(method_key, method))
    ml_all = summary["ml_pca_linear_cell_centered"]["all_splits"]
    for metric, aggregate in ml_all.items():
        rows.extend(
            {
                "section": "ml_pca_linear_cell_centered_all_splits",
                "metric": f"{metric}_{name}",
                "value": value,
            }
            for name, value in aggregate.items()
        )
    for method, payload in summary["clipping_diagnostics"].items():
        for metric in (
            "clipped_cell_count",
            "clipped_cell_percentage",
            "covered_clipped_cell_count",
            "uncovered_clipped_cell_count",
        ):
            rows.extend(
                {
                    "section": f"clipping:{method}",
                    "metric": f"{metric}_{name}",
                    "value": value,
                }
                for name, value in payload[metric].items()
            )
    _write_csv(path, rows)


def _summary_method_rows(method_key: str, method: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"section": method_key, "metric": "sample_count", "value": method["sample_count"]},
    ]
    for metric in (
        "travel_time_rmse_s",
        "travel_time_mae_s",
        "all_cell_velocity_rmse_km_per_s",
        "all_cell_velocity_mae_km_per_s",
        "covered_cell_velocity_rmse_km_per_s",
        "covered_cell_velocity_mae_km_per_s",
        "coverage_fraction",
        "clipped_velocity_cell_count",
    ):
        rows.extend(
            {
                "section": method_key,
                "metric": f"{metric}_{name}",
                "value": value,
            }
            for name, value in method[metric].items()
        )
    return rows


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and row.get(key) != "" and not math.isnan(float(row[key]))
    ]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(values), "mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(sorted(grouped.items()))


def _load_velocity_grid(path: Path) -> CartesianVelocityGrid3D:
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


def _node_index(ix: int, iy: int, iz: int, nx: int, ny: int) -> int:
    return ix + nx * (iy + ny * iz)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _as_bool(value: Any) -> bool:
    return str(value).lower() == "true"
