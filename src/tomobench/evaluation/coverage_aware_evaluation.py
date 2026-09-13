"""Coverage-aware evaluation for grouped-split ML and fixed-ray results."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tomobench.evaluation.metrics import mean_absolute_error, root_mean_squared_error
from tomobench.tomography.comparison import node_centered_values_to_cell_centered
from tomobench.utils.paths import get_repo_root


DEFAULT_COVERAGE_MANIFEST = Path(
    "outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json"
)
DEFAULT_COVERAGE_SPLIT = Path(
    "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/target_group_split_assignments.csv"
)
DEFAULT_COVERAGE_ML_METRICS = Path(
    "outputs/generated/ml_baselines/phase9_grouped_split_ablation_v1/cell_centered_metrics.csv"
)
DEFAULT_COVERAGE_CLASSICAL_DIR = Path(
    "outputs/generated/classical_tomography/fair_reference_ray_corpus_v1"
)
DEFAULT_COVERAGE_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/coverage_aware_evaluation_v1"
)
DEFAULT_COVERAGE_THRESHOLDS_KM = (0.0, 1.0e-9, 1.0e-3, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0)
DEFAULT_COVERAGE_METHODS = (
    "reference_prior",
    "optimized_fixed_ray",
    "full_input_euclidean_path",
    "full_input",
    "travel_time_only",
)


@dataclass(frozen=True)
class CoverageAwareEvaluationOutputs:
    """Artifacts written by the coverage-aware evaluation."""

    output_dir: Path
    case_metrics_csv: Path
    depth_metrics_csv: Path
    distribution_csv: Path
    summary_json: Path
    summary_md: Path
    metadata_json: Path


def run_coverage_aware_evaluation(
    *,
    manifest_path: Path = DEFAULT_COVERAGE_MANIFEST,
    grouped_split_path: Path = DEFAULT_COVERAGE_SPLIT,
    ml_metrics_path: Path = DEFAULT_COVERAGE_ML_METRICS,
    classical_corpus_dir: Path = DEFAULT_COVERAGE_CLASSICAL_DIR,
    output_dir: Path = DEFAULT_COVERAGE_OUTPUT_DIR,
    thresholds_km: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS_KM,
    method_ids: Sequence[str] = DEFAULT_COVERAGE_METHODS,
) -> CoverageAwareEvaluationOutputs:
    """Evaluate fixed-test reconstructions under increasingly strict coverage masks.

    Coverage is taken from the fair reference-ray geometry, so every method is
    evaluated under one explicitly shared cell mask.  ``threshold=0`` means
    strictly positive accumulated path length; positive thresholds use ``>=``.
    """
    if not thresholds_km:
        raise ValueError("thresholds_km must contain at least one value.")
    thresholds = tuple(float(value) for value in thresholds_km)
    if any(value < 0.0 for value in thresholds):
        raise ValueError("Coverage thresholds must be non-negative.")
    thresholds = tuple(sorted(set(thresholds)))
    methods = tuple(dict.fromkeys(str(value) for value in method_ids))
    if not methods:
        raise ValueError("method_ids must contain at least one method.")

    repo_root = get_repo_root()
    manifest_resolved = _resolve_path(Path(manifest_path), repo_root)
    split_resolved = _resolve_path(Path(grouped_split_path), repo_root)
    ml_metrics_resolved = _resolve_path(Path(ml_metrics_path), repo_root)
    classical_resolved = _resolve_path(Path(classical_corpus_dir), repo_root)
    output_resolved = _resolve_path(Path(output_dir), repo_root)
    manifest = _read_json(manifest_resolved)
    split_by_case, target_hash_by_case = _read_grouped_split(split_resolved)
    ml_rows = _read_csv(ml_metrics_resolved)
    ml_by_variant_case = _index_ml_rows(ml_rows, methods)
    test_items = [
        item
        for item in _manifest_items(manifest)
        if split_by_case.get(str(item["case_id"])) == "test"
    ]
    if not test_items:
        raise ValueError("The grouped split contains no test cases.")

    case_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    path_length_values: list[float] = []
    for item in sorted(test_items, key=lambda value: str(value["case_id"])):
        case_id = str(item["case_id"])
        family = str(item.get("family") or item.get("scenario") or "unknown")
        target_grid_path = _resolve_manifest_path(
            item.get("target_velocity_grid_path") or item.get("target_grid_path"),
            repo_root,
        )
        target_grid = _read_json(target_grid_path)
        nx = len(target_grid["x_coordinates_km"])
        ny = len(target_grid["y_coordinates_km"])
        nz = len(target_grid["z_coordinates_km"])
        target_cells = node_centered_values_to_cell_centered(
            target_grid["p_velocity_km_per_s"], nx, ny, nz
        )

        classical_case_dir = classical_resolved / "cases" / case_id
        coverage_path = classical_case_dir / "reference_ray_cell_coverage.csv"
        coverage_rows = _read_csv(coverage_path)
        coverage_values = np.array(
            [float(row["total_path_length_coverage_km"]) for row in coverage_rows],
            dtype=float,
        )
        expected_cell_count = (nx - 1) * (ny - 1) * (nz - 1)
        if coverage_values.size != expected_cell_count:
            raise ValueError(
                f"{case_id}: coverage count {coverage_values.size} does not match "
                f"target cell count {expected_cell_count}."
            )
        path_length_values.extend(float(value) for value in coverage_values if value > 0.0)

        reference_summary_path = classical_case_dir / "reference_ray_perturbation_summary.json"
        reference_summary = _read_json(reference_summary_path)
        reference_grid_path = _resolve_path(
            Path(str(reference_summary["source_files"]["reference_velocity_grid"])),
            repo_root,
        )
        reference_grid = _read_json(reference_grid_path)
        reference_prior_cells = node_centered_values_to_cell_centered(
            reference_grid["p_velocity_km_per_s"],
            len(reference_grid["x_coordinates_km"]),
            len(reference_grid["y_coordinates_km"]),
            len(reference_grid["z_coordinates_km"]),
        )
        methods_by_case = {
            "reference_prior": tuple(float(value) for value in reference_prior_cells),
            "optimized_fixed_ray": _read_classical_prediction(
                classical_case_dir / "reference_ray_cell_velocity_predictions.csv",
                expected_cell_count,
            ),
        }
        for method in methods:
            if method in {"reference_prior", "optimized_fixed_ray"}:
                continue
            ml_row = ml_by_variant_case.get((method, case_id))
            if ml_row is None:
                raise ValueError(f"Missing ML metrics row for method={method}, case={case_id}.")
            methods_by_case[method] = _read_ml_prediction_cells(
                ml_row,
                target_grid,
                expected_cell_count,
                repo_root,
            )

        for threshold in thresholds:
            covered_mask = _coverage_mask(coverage_values, threshold)
            covered_count = int(np.count_nonzero(covered_mask))
            for method in methods:
                metrics = _masked_metrics(
                    target_cells,
                    methods_by_case[method],
                    covered_mask,
                )
                case_rows.append(
                    {
                        "case_id": case_id,
                        "family": family,
                        "split": "test",
                        "target_hash": target_hash_by_case[case_id],
                        "method_id": method,
                        "coverage_threshold_km": threshold,
                        "coverage_rule": "strictly_positive" if threshold == 0.0 else "greater_than_or_equal",
                        "cell_count": expected_cell_count,
                        "covered_cell_count": covered_count,
                        "coverage_fraction": covered_count / expected_cell_count,
                        "mean_covered_path_length_km": _mean_masked(
                            coverage_values, covered_mask
                        ),
                        "max_covered_path_length_km": _max_masked(
                            coverage_values, covered_mask
                        ),
                        "all_cell_velocity_rmse_km_per_s": metrics["all_rmse"],
                        "all_cell_velocity_mae_km_per_s": metrics["all_mae"],
                        "covered_cell_velocity_rmse_km_per_s": metrics["covered_rmse"],
                        "covered_cell_velocity_mae_km_per_s": metrics["covered_mae"],
                    }
                )

            depth_rows.extend(
                _depth_coverage_rows(
                    case_id=case_id,
                    family=family,
                    target_hash=target_hash_by_case[case_id],
                    threshold=threshold,
                    coverage_values=coverage_values,
                    nx_cells=nx - 1,
                    ny_cells=ny - 1,
                    z_coordinates=target_grid["z_coordinates_km"],
                )
            )

    output_resolved.mkdir(parents=True, exist_ok=True)
    case_metrics_csv = output_resolved / "coverage_case_metrics.csv"
    depth_metrics_csv = output_resolved / "coverage_depth_metrics.csv"
    distribution_csv = output_resolved / "coverage_distribution_summary.csv"
    summary_json = output_resolved / "coverage_aware_evaluation_summary.json"
    summary_md = output_resolved / "coverage_aware_evaluation_summary.md"
    metadata_json = output_resolved / "audit_metadata.json"
    _write_csv(case_metrics_csv, case_rows)
    _write_csv(depth_metrics_csv, depth_rows)
    distribution_rows = _distribution_rows(path_length_values, thresholds)
    _write_csv(distribution_csv, distribution_rows)
    summary = _summary_payload(
        manifest=manifest,
        split_by_case=split_by_case,
        target_hash_by_case=target_hash_by_case,
        case_rows=case_rows,
        depth_rows=depth_rows,
        thresholds=thresholds,
        methods=methods,
        source_paths={
            "manifest": _relative_or_absolute(manifest_resolved, repo_root),
            "grouped_split": _relative_or_absolute(split_resolved, repo_root),
            "ml_metrics": _relative_or_absolute(ml_metrics_resolved, repo_root),
            "fair_classical_corpus": _relative_or_absolute(classical_resolved, repo_root),
        },
    )
    _write_json(summary_json, summary)
    _write_markdown(summary_md, summary, case_metrics_csv, depth_metrics_csv, repo_root)
    _write_json(
        metadata_json,
        {
            "artifact_type": "coverage_aware_evaluation_metadata_v1",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "command": "tomobench audit-coverage-aware-evaluation",
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "thresholds_km": list(thresholds),
            "methods": list(methods),
            "source_file_sha256": {
                str(key): _sha256_file(path)
                for key, path in (
                    ("manifest", manifest_resolved),
                    ("grouped_split", split_resolved),
                    ("ml_metrics", ml_metrics_resolved),
                )
            },
            "coverage_definition": "reference-ray accumulated path length per cell; threshold 0 uses strictly positive coverage and positive thresholds use greater-than-or-equal",
            "test_case_count": len(test_items),
            "test_unique_target_count": len(
                {target_hash_by_case[str(item["case_id"])] for item in test_items}
            ),
        },
    )
    return CoverageAwareEvaluationOutputs(
        output_dir=output_resolved,
        case_metrics_csv=case_metrics_csv,
        depth_metrics_csv=depth_metrics_csv,
        distribution_csv=distribution_csv,
        summary_json=summary_json,
        summary_md=summary_md,
        metadata_json=metadata_json,
    )


def _masked_metrics(
    target: Sequence[float],
    prediction: Sequence[float],
    mask: np.ndarray,
) -> dict[str, float | None]:
    target_values = np.asarray(target, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    if target_values.shape != prediction_values.shape or target_values.shape[0] != mask.size:
        raise ValueError("Target, prediction, and coverage mask lengths must match.")
    covered_target = target_values[mask]
    covered_prediction = prediction_values[mask]
    return {
        "all_rmse": root_mean_squared_error(target_values.tolist(), prediction_values.tolist()),
        "all_mae": mean_absolute_error(target_values.tolist(), prediction_values.tolist()),
        "covered_rmse": (
            root_mean_squared_error(covered_target.tolist(), covered_prediction.tolist())
            if covered_target.size
            else None
        ),
        "covered_mae": (
            mean_absolute_error(covered_target.tolist(), covered_prediction.tolist())
            if covered_target.size
            else None
        ),
    }


def _coverage_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    return values > 0.0 if threshold == 0.0 else values >= threshold


def _depth_coverage_rows(
    *,
    case_id: str,
    family: str,
    target_hash: str,
    threshold: float,
    coverage_values: np.ndarray,
    nx_cells: int,
    ny_cells: int,
    z_coordinates: Sequence[float],
) -> list[dict[str, Any]]:
    cells_per_depth = nx_cells * ny_cells
    rows: list[dict[str, Any]] = []
    for iz in range(len(z_coordinates) - 1):
        start = iz * cells_per_depth
        end = start + cells_per_depth
        depth_values = coverage_values[start:end]
        mask = _coverage_mask(depth_values, threshold)
        rows.append(
            {
                "case_id": case_id,
                "family": family,
                "target_hash": target_hash,
                "coverage_threshold_km": threshold,
                "coverage_rule": "strictly_positive" if threshold == 0.0 else "greater_than_or_equal",
                "iz": iz,
                "top_depth_km": float(z_coordinates[iz]),
                "bottom_depth_km": float(z_coordinates[iz + 1]),
                "cell_count": int(depth_values.size),
                "covered_cell_count": int(np.count_nonzero(mask)),
                "coverage_fraction": float(np.mean(mask)) if depth_values.size else 0.0,
                "mean_positive_path_length_km": _mean_masked(depth_values, depth_values > 0.0),
                "mean_covered_path_length_km": _mean_masked(depth_values, mask),
            }
        )
    return rows


def _distribution_rows(values: Sequence[float], thresholds: Sequence[float]) -> list[dict[str, Any]]:
    positive = np.asarray(values, dtype=float)
    if positive.size == 0:
        raise ValueError("No positive coverage values were found.")
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        mask = _coverage_mask(positive, threshold)
        selected = positive[mask]
        rows.append(
            {
                "coverage_threshold_km": threshold,
                "coverage_rule": "strictly_positive" if threshold == 0.0 else "greater_than_or_equal",
                "positive_cell_count": int(positive.size),
                "selected_cell_count": int(selected.size),
                "selected_fraction_of_positive_cells": float(selected.size / positive.size),
                "min_selected_path_length_km": float(np.min(selected)) if selected.size else None,
                "p05_selected_path_length_km": _percentile(selected, 5.0),
                "median_selected_path_length_km": _percentile(selected, 50.0),
                "mean_selected_path_length_km": float(np.mean(selected)) if selected.size else None,
                "p95_selected_path_length_km": _percentile(selected, 95.0),
                "max_selected_path_length_km": float(np.max(selected)) if selected.size else None,
            }
        )
    return rows


def _summary_payload(
    *,
    manifest: dict[str, Any],
    split_by_case: dict[str, str],
    target_hash_by_case: dict[str, str],
    case_rows: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
    thresholds: Sequence[float],
    methods: Sequence[str],
    source_paths: dict[str, str],
) -> dict[str, Any]:
    test_cases = [case_id for case_id, split in split_by_case.items() if split == "test"]
    method_summary: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        threshold_key = _float_key(threshold)
        rows = [
            row for row in case_rows if float(row["coverage_threshold_km"]) == threshold
        ]
        method_summary[threshold_key] = {
            method: {
                "case_count": len(method_rows := [
                    row for row in rows if row["method_id"] == method
                ]),
                "unique_target_count": len({row["target_hash"] for row in method_rows}),
                "mean_coverage_fraction": _mean_rows(method_rows, "coverage_fraction"),
                "mean_all_cell_rmse_km_per_s": _mean_rows(
                    method_rows, "all_cell_velocity_rmse_km_per_s"
                ),
                "mean_all_cell_mae_km_per_s": _mean_rows(
                    method_rows, "all_cell_velocity_mae_km_per_s"
                ),
                "mean_covered_cell_rmse_km_per_s": _mean_rows(
                    method_rows, "covered_cell_velocity_rmse_km_per_s"
                ),
                "mean_covered_cell_mae_km_per_s": _mean_rows(
                    method_rows, "covered_cell_velocity_mae_km_per_s"
                ),
            }
            for method in methods
        }
    depth_summary = _aggregate_depth_rows(depth_rows)
    return {
        "artifact_type": "coverage_aware_grouped_test_evaluation_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_files": source_paths,
        "corpus_id": manifest.get("corpus_id") or manifest.get("batch_id"),
        "test_case_count": len(test_cases),
        "test_unique_target_count": len({target_hash_by_case[case_id] for case_id in test_cases}),
        "test_case_ids": sorted(test_cases),
        "thresholds_km": list(thresholds),
        "methods": list(methods),
        "coverage_geometry": "fair_reference_ray_fixed_geometry",
        "coverage_definition": "accumulated reference-ray path length per cell; threshold 0 is strictly positive and positive thresholds use >=",
        "method_summary_by_threshold": method_summary,
        "depth_coverage_summary": depth_summary,
        "interpretation": [
            "All methods use the same reference-ray coverage masks; this does not make the ML model a tomography algorithm.",
            "The all-cell metric is repeated for each threshold for comparison; the covered-cell metric changes with the mask.",
            "The reference prior and optimized fixed-ray rows use the validation-selected lambda=1.0, alpha=10.0 configuration frozen before test evaluation.",
            "Coverage thresholds are sensitivity diagnostics, not claims that a single path-length cutoff is physically universal.",
        ],
    }


def _aggregate_depth_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["coverage_threshold_km"]), int(row["iz"]))].append(row)
    output: list[dict[str, Any]] = []
    for (threshold, iz), group in sorted(grouped.items()):
        output.append(
            {
                "coverage_threshold_km": threshold,
                "iz": iz,
                "top_depth_km": group[0]["top_depth_km"],
                "bottom_depth_km": group[0]["bottom_depth_km"],
                "case_count": len(group),
                "mean_coverage_fraction": _mean_rows(group, "coverage_fraction"),
                "mean_covered_cell_count": _mean_rows(group, "covered_cell_count"),
                "mean_positive_path_length_km": _mean_rows(
                    group, "mean_positive_path_length_km"
                ),
                "mean_covered_path_length_km": _mean_rows(
                    group, "mean_covered_path_length_km"
                ),
            }
        )
    return output


def _read_ml_prediction_cells(
    row: dict[str, Any],
    target_grid: dict[str, Any],
    expected_cell_count: int,
    repo_root: Path,
) -> tuple[float, ...]:
    prediction = _read_json(_resolve_path(Path(str(row["prediction_path"])), repo_root))
    nodes = tuple(float(value) for value in prediction["predicted_p_velocity_km_per_s"])
    cells = node_centered_values_to_cell_centered(
        nodes,
        len(target_grid["x_coordinates_km"]),
        len(target_grid["y_coordinates_km"]),
        len(target_grid["z_coordinates_km"]),
    )
    if len(cells) != expected_cell_count:
        raise ValueError("ML cell prediction length does not match target cell count.")
    return cells


def _read_classical_prediction(path: Path, expected_cell_count: int) -> tuple[float, ...]:
    rows = _read_csv(path)
    values = tuple(float(row["predicted_velocity_km_per_s"]) for row in rows)
    if len(values) != expected_cell_count:
        raise ValueError(f"Classical prediction length does not match target cell count: {path}")
    return values


def _index_ml_rows(
    rows: list[dict[str, Any]],
    methods: Sequence[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    selected = {
        (str(row.get("variant_id", "")), str(row["case_id"])): row
        for row in rows
        if str(row.get("split", "")) == "test"
        and str(row.get("variant_id", "")) in set(methods)
    }
    return selected


def _manifest_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a non-empty items list.")
    if not all(isinstance(item, dict) for item in items):
        raise TypeError("Every manifest item must be an object.")
    return [item for item in items if isinstance(item, dict)]


def _read_grouped_split(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    rows = _read_csv(path)
    split_by_case: dict[str, str] = {}
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    target_hash_by_case: dict[str, str] = {}
    for row in rows:
        case_id = str(row["case_id"])
        split = str(row["split"])
        target_hash = str(row["target_hash"])
        split_by_case[case_id] = split
        target_hash_by_case[case_id] = target_hash
        hash_to_splits[target_hash].add(split)
    crossing = [target_hash for target_hash, splits in hash_to_splits.items() if len(splits) > 1]
    if crossing:
        raise ValueError("Grouped split has target groups crossing splits.")
    return split_by_case, target_hash_by_case


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, summary: dict[str, Any], case_path: Path, depth_path: Path, repo_root: Path) -> None:
    lines = [
        "# Coverage-Aware Grouped-Test Evaluation",
        "",
        "The fixed test cases are evaluated using accumulated path length from the fair reference-ray geometry. Threshold `0` denotes strictly positive coverage; positive thresholds use `>=`.",
        "",
        f"- Test cases: `{summary['test_case_count']}`",
        f"- Unique test targets: `{summary['test_unique_target_count']}`",
        f"- Thresholds (km): `{summary['thresholds_km']}`",
        "",
        "## Outputs",
        "",
        f"- Case metrics: `{_relative_or_absolute(case_path, repo_root)}`",
        f"- Depth metrics: `{_relative_or_absolute(depth_path, repo_root)}`",
        "",
        "## Caution",
        "",
        "Coverage masks are shared evaluation masks, not additional information supplied to the ML models. The threshold sweep is a sensitivity analysis, not a universal physical cutoff.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _mean_rows(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return sum(values) / len(values) if values else None


def _mean_masked(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    return float(np.mean(selected)) if selected.size else None


def _max_masked(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    return float(np.max(selected)) if selected.size else None


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def _resolve_manifest_path(path_value: Any, repo_root: Path) -> Path:
    if not path_value:
        raise ValueError("Manifest item is missing a target grid path.")
    path = _resolve_path(Path(str(path_value)), repo_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_key(value: float) -> str:
    return f"{value:g}"


__all__ = ["CoverageAwareEvaluationOutputs", "run_coverage_aware_evaluation"]
