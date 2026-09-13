"""Target-level uncertainty reporting for grouped synthetic test evaluations.

The corrected Phase 9 comparison keeps acquisition cases grouped in the split, but
its original uncertainty report still treated the 25 acquisition cases as the
resampling units.  This module makes the geological target velocity model the
independent unit: acquisition-case metrics are first averaged within exact target
hash, and only those target-level values enter bootstrap or sign-flip calculations.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from tomobench.evaluation.phase8_paired_test_comparison import (
    PAIRED_METRIC_CONTRACT,
    validate_paired_metric_contract,
)
from tomobench.evaluation.phase8_uncertainty_report import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_PERMUTATION_RESAMPLES,
    DEFAULT_PERMUTATION_SEED,
    bootstrap_mean_confidence_interval,
    paired_sign_flip_p_value,
)
from tomobench.utils.paths import get_repo_root


TARGET_LEVEL_UNCERTAINTY_REPORT_ID = "phase9_grouped_split_target_level_uncertainty_v1"
DEFAULT_INPUT_DIR = Path(
    "outputs/generated/submission_readiness/phase9_grouped_split_paired_comparison_v1"
)
DEFAULT_PAIRED_DELTAS_DIR = Path(
    "outputs/generated/submission_readiness/phase9_grouped_split_uncertainty_v1"
)
DEFAULT_GROUPED_SPLIT_PATH = Path(
    "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/"
    "target_group_split_assignments.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/"
    "phase9_grouped_split_target_level_uncertainty_v1"
)
DEFAULT_TIE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class TargetLevelUncertaintyOutputs:
    """Files written by the target-level uncertainty audit."""

    output_dir: Path
    acquisition_case_metrics_csv: Path
    target_level_metrics_csv: Path
    target_level_paired_deltas_csv: Path
    summary_md: Path
    summary_json: Path
    metadata_json: Path
    experiment_note: Path


def run_target_level_uncertainty_report(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    paired_deltas_dir: Path = DEFAULT_PAIRED_DELTAS_DIR,
    grouped_split_path: Path = DEFAULT_GROUPED_SPLIT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    permutation_resamples: int = DEFAULT_PERMUTATION_RESAMPLES,
    permutation_seed: int = DEFAULT_PERMUTATION_SEED,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    experiment_note_path: Path | None = None,
) -> TargetLevelUncertaintyOutputs:
    """Create target-level method and paired uncertainty summaries.

    The input case metrics and paired case deltas are not recomputed.  They are
    enriched with the exact target hash from the frozen grouped split, aggregated
    within target, and then summarized using target-level resampling.
    """

    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance cannot be negative.")
    repo_root = get_repo_root()
    resolved_input = _resolve(input_dir, repo_root)
    resolved_paired_deltas = _resolve(paired_deltas_dir, repo_root)
    resolved_split = _resolve(grouped_split_path, repo_root)
    resolved_output = _resolve(output_dir, repo_root)
    case_metrics_path = resolved_input / "paired_test_case_metrics.csv"
    case_deltas_path = resolved_paired_deltas / "paired_case_deltas.csv"
    fixed_ids_path = resolved_input / "fixed_test_case_ids.json"

    case_rows = _read_csv(case_metrics_path)
    validate_paired_metric_contract(case_rows)
    delta_rows = _read_csv(case_deltas_path)
    fixed_case_ids = _read_fixed_case_ids(fixed_ids_path)
    target_by_case = _load_target_by_case(resolved_split, fixed_case_ids)

    acquisition_rows = enrich_acquisition_case_metrics(case_rows, target_by_case)
    _validate_case_metric_coverage(acquisition_rows, fixed_case_ids)
    target_metric_rows = aggregate_target_level_metrics(acquisition_rows)

    enriched_delta_rows = enrich_case_delta_rows(delta_rows, target_by_case)
    target_delta_rows = aggregate_target_level_paired_deltas(
        enriched_delta_rows,
        tie_tolerance=tie_tolerance,
    )

    method_summary = summarize_target_level_methods(
        target_metric_rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    paired_summary = summarize_target_level_paired_deltas(
        target_delta_rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        permutation_resamples=permutation_resamples,
        permutation_seed=permutation_seed,
        tie_tolerance=tie_tolerance,
    )

    output_dir_path = resolved_output
    output_dir_path.mkdir(parents=True, exist_ok=True)
    acquisition_path = output_dir_path / "acquisition_case_metrics.csv"
    target_metrics_path = output_dir_path / "target_level_metrics.csv"
    target_deltas_path = output_dir_path / "target_level_paired_deltas.csv"
    summary_md_path = output_dir_path / "target_level_uncertainty_summary.md"
    summary_json_path = output_dir_path / "target_level_uncertainty_summary.json"
    metadata_path = output_dir_path / "audit_metadata.json"
    note_path = _resolve(
        experiment_note_path
        if experiment_note_path is not None
        else repo_root / "wiki/experiments/phase9_grouped_split_target_level_uncertainty_v1.md",
        repo_root,
    )

    _write_csv(acquisition_path, acquisition_rows)
    _write_csv(target_metrics_path, target_metric_rows)
    _write_csv(target_deltas_path, target_delta_rows)

    target_hashes = sorted({str(row["target_hash"]) for row in target_metric_rows})
    method_ids = sorted({str(row["method_id"]) for row in target_metric_rows})
    comparison_ids = sorted({str(row["comparison_id"]) for row in target_delta_rows})
    summary_payload = {
        "artifact_type": "target_level_uncertainty_summary",
        "experiment_id": TARGET_LEVEL_UNCERTAINTY_REPORT_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_case_metrics": _relative(case_metrics_path, repo_root),
        "input_case_deltas": _relative(case_deltas_path, repo_root),
        "input_grouped_split": _relative(resolved_split, repo_root),
        "fixed_test_case_count": len(fixed_case_ids),
        "unique_test_target_count": len(target_hashes),
        "target_hashes": target_hashes,
        "method_ids": method_ids,
        "comparison_ids": comparison_ids,
        "resampling_unit": "unique target velocity model identified by exact target hash",
        "aggregation_rule": "arithmetic mean across acquisition geometries within target",
        "method_summary": method_summary,
        "paired_summary": paired_summary,
        "inference_caveat": (
            "The corrected test set has five independent target models. Bootstrap and "
            "sign-flip calculations are descriptive and highly unstable as population "
            "inference at this sample size; acquisition cases are not resampled independently."
        ),
    }
    metadata = {
        "artifact_type": "target_level_uncertainty_metadata",
        "experiment_id": TARGET_LEVEL_UNCERTAINTY_REPORT_ID,
        "generated_at_utc": summary_payload["generated_at_utc"],
        "input_artifacts": {
            "case_metrics": _relative(case_metrics_path, repo_root),
            "case_deltas": _relative(case_deltas_path, repo_root),
            "grouped_split": _relative(resolved_split, repo_root),
            "fixed_case_ids": _relative(fixed_ids_path, repo_root),
        },
        "fixed_test_case_count": len(fixed_case_ids),
        "unique_test_target_count": len(target_hashes),
        "bootstrap": {
            "type": "percentile bootstrap over target-level summaries",
            "resamples": bootstrap_resamples,
            "base_seed": bootstrap_seed,
            "confidence_level": 0.95,
            "percentile_interpolation": "linear",
        },
        "permutation": {
            "type": "paired sign-flip Monte Carlo over target-level paired deltas",
            "resamples": permutation_resamples,
            "base_seed": permutation_seed,
        },
        "tie_tolerance_km_per_s": tie_tolerance,
        "case_level_comparison_warning": (
            "The prior phase9_grouped_split_uncertainty_v1 report resampled 25 acquisition "
            "cases. This artifact resamples only five target-level values per method or comparison."
        ),
    }
    _write_json(summary_json_path, summary_payload)
    _write_json(metadata_path, metadata)
    _write_summary_markdown(
        summary_md_path,
        summary_payload=summary_payload,
        method_summary=method_summary,
        paired_summary=paired_summary,
        output_dir=output_dir_path,
        repo_root=repo_root,
    )
    _write_experiment_note(
        note_path,
        summary_payload=summary_payload,
        output_dir=output_dir_path,
        repo_root=repo_root,
    )
    return TargetLevelUncertaintyOutputs(
        output_dir=output_dir_path,
        acquisition_case_metrics_csv=acquisition_path,
        target_level_metrics_csv=target_metrics_path,
        target_level_paired_deltas_csv=target_deltas_path,
        summary_md=summary_md_path,
        summary_json=summary_json_path,
        metadata_json=metadata_path,
        experiment_note=note_path,
    )


def enrich_acquisition_case_metrics(
    rows: Sequence[Mapping[str, Any]],
    target_by_case: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Attach target identity to every acquisition-case metric row."""

    enriched: list[dict[str, Any]] = []
    seen_method_case: set[tuple[str, str]] = set()
    for source in rows:
        case_id = str(source.get("case_id", ""))
        method_id = str(source.get("method_id", ""))
        if not case_id or not method_id:
            raise ValueError("Acquisition metrics require non-empty case_id and method_id.")
        key = (method_id, case_id)
        if key in seen_method_case:
            raise ValueError(f"Duplicate acquisition metric row for method/case: {key}.")
        seen_method_case.add(key)
        if case_id not in target_by_case:
            raise ValueError(f"Case {case_id!r} is absent from the frozen test split.")
        target = target_by_case[case_id]
        row = dict(source)
        row["target_hash"] = target["target_hash"]
        row["target_id"] = target["target_hash"]
        row["target_family"] = target["family"]
        row["target_split"] = target["split"]
        row["acquisition_id"] = case_id
        enriched.append(row)
    return enriched


def enrich_case_delta_rows(
    rows: Sequence[Mapping[str, Any]],
    target_by_case: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Attach target identity to each paired acquisition-case delta."""

    enriched: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in rows:
        case_id = str(source.get("case_id", ""))
        comparison_id = str(source.get("comparison_id", ""))
        if not case_id or not comparison_id:
            raise ValueError("Paired case deltas require case_id and comparison_id.")
        key = (comparison_id, case_id)
        if key in seen:
            raise ValueError(f"Duplicate paired acquisition delta row: {key}.")
        seen.add(key)
        if case_id not in target_by_case:
            raise ValueError(f"Paired delta case {case_id!r} is absent from the frozen test split.")
        row = dict(source)
        target = target_by_case[case_id]
        row["target_hash"] = target["target_hash"]
        row["target_id"] = target["target_hash"]
        row["target_family"] = target["family"]
        row["target_split"] = target["split"]
        enriched.append(row)
    return enriched


def aggregate_target_level_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Average each method's metrics over acquisition geometries within target."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        method_id = str(row.get("method_id", ""))
        target_hash = str(row.get("target_hash", ""))
        if not method_id or not target_hash:
            raise ValueError("Target-level aggregation requires method_id and target_hash.")
        if str(row.get("target_split", row.get("split", ""))) != "test":
            raise ValueError("Target-level metrics must come from the fixed test split.")
        grouped[(method_id, target_hash)].append(row)

    output: list[dict[str, Any]] = []
    for (method_id, target_hash), group in sorted(grouped.items()):
        families = {str(row.get("target_family", row.get("family", ""))) for row in group}
        if len(families) != 1:
            raise ValueError(f"Target {target_hash} has inconsistent family labels: {families}.")
        rmse_values = [_metric(row, "rmse_km_per_s") for row in group]
        mae_values = [_metric(row, "mae_km_per_s") for row in group]
        first = group[0]
        output.append(
            {
                "target_hash": target_hash,
                "target_id": target_hash,
                "family": next(iter(families)),
                "split": "test",
                "method_id": method_id,
                "method_label": str(first.get("method_label", method_id)),
                "method_family": str(first.get("method_family", "")),
                "comparison_role": str(first.get("comparison_role", "")),
                "diagnostic_only": first.get("diagnostic_only", False),
                "is_primary": first.get("is_primary", True),
                "acquisition_case_count": len(group),
                "acquisition_case_ids": "|".join(
                    sorted(str(row["case_id"]) for row in group)
                ),
                "rmse_km_per_s": statistics.mean(rmse_values),
                "mae_km_per_s": statistics.mean(mae_values),
                "rmse_acquisition_case_std_km_per_s": statistics.pstdev(rmse_values),
                "mae_acquisition_case_std_km_per_s": statistics.pstdev(mae_values),
                "aggregation_unit": "unique_target_velocity_model",
                "metric_contract": PAIRED_METRIC_CONTRACT,
            }
        )
    return output


def aggregate_target_level_paired_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> list[dict[str, Any]]:
    """Average paired acquisition-case differences within exact target hash."""

    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance cannot be negative.")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        comparison_id = str(row.get("comparison_id", ""))
        target_hash = str(row.get("target_hash", ""))
        if not comparison_id or not target_hash:
            raise ValueError("Target-level deltas require comparison_id and target_hash.")
        if str(row.get("target_split", row.get("split", ""))) != "test":
            raise ValueError("Target-level deltas must come from the fixed test split.")
        grouped[(comparison_id, target_hash)].append(row)

    output: list[dict[str, Any]] = []
    for (comparison_id, target_hash), group in sorted(grouped.items()):
        first = group[0]
        first_method = str(first.get("first_method_id", ""))
        second_method = str(first.get("second_method_id", ""))
        families = {str(row.get("target_family", row.get("family", ""))) for row in group}
        if len(families) != 1:
            raise ValueError(f"Paired target {target_hash} has inconsistent families: {families}.")
        if any(
            str(row.get("first_method_id", "")) != first_method
            or str(row.get("second_method_id", "")) != second_method
            for row in group
        ):
            raise ValueError(f"Comparison {comparison_id} changes method IDs within a target.")
        rmse_first = [_metric(row, "first_rmse_km_per_s") for row in group]
        rmse_second = [_metric(row, "second_rmse_km_per_s") for row in group]
        mae_first = [_metric(row, "first_mae_km_per_s") for row in group]
        mae_second = [_metric(row, "second_mae_km_per_s") for row in group]
        rmse_deltas = [_metric(row, "delta_rmse_km_per_s") for row in group]
        mae_deltas = [_metric(row, "delta_mae_km_per_s") for row in group]
        rmse_delta = statistics.mean(rmse_deltas)
        mae_delta = statistics.mean(mae_deltas)
        output.append(
            {
                "comparison_id": comparison_id,
                "first_method_id": first_method,
                "second_method_id": second_method,
                "first_method_label": str(first.get("first_method_label", first_method)),
                "second_method_label": str(first.get("second_method_label", second_method)),
                "target_hash": target_hash,
                "target_id": target_hash,
                "family": next(iter(families)),
                "split": "test",
                "acquisition_case_count": len(group),
                "acquisition_case_ids": "|".join(
                    sorted(str(row["case_id"]) for row in group)
                ),
                "first_rmse_target_mean_km_per_s": statistics.mean(rmse_first),
                "second_rmse_target_mean_km_per_s": statistics.mean(rmse_second),
                "delta_rmse_km_per_s": rmse_delta,
                "first_mae_target_mean_km_per_s": statistics.mean(mae_first),
                "second_mae_target_mean_km_per_s": statistics.mean(mae_second),
                "delta_mae_km_per_s": mae_delta,
                "rmse_case_win_count": sum(value < -tie_tolerance for value in rmse_deltas),
                "rmse_case_loss_count": sum(value > tie_tolerance for value in rmse_deltas),
                "rmse_case_tie_count": sum(abs(value) <= tie_tolerance for value in rmse_deltas),
                "aggregation_unit": "unique_target_velocity_model",
                "metric_contract": str(first.get("metric_contract", PAIRED_METRIC_CONTRACT)),
            }
        )
    return output


def summarize_target_level_methods(
    target_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Summarize methods with one resampling value per unique target."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        grouped[str(row["method_id"])].append(row)
    output: list[dict[str, Any]] = []
    for method_id, rows in sorted(grouped.items()):
        rmse_values = [_metric(row, "rmse_km_per_s") for row in rows]
        mae_values = [_metric(row, "mae_km_per_s") for row in rows]
        rmse_ci = bootstrap_mean_confidence_interval(
            rmse_values,
            resamples=bootstrap_resamples,
            seed=_stable_seed(bootstrap_seed, f"method:{method_id}:rmse"),
        )
        mae_ci = bootstrap_mean_confidence_interval(
            mae_values,
            resamples=bootstrap_resamples,
            seed=_stable_seed(bootstrap_seed, f"method:{method_id}:mae"),
        )
        output.append(
            {
                "method_id": method_id,
                "method_label": str(rows[0].get("method_label", method_id)),
                "target_count": len(rows),
                "acquisition_case_count": sum(
                    int(row["acquisition_case_count"]) for row in rows
                ),
                "rmse_mean_km_per_s": rmse_ci["mean"],
                "rmse_std_km_per_s": statistics.pstdev(rmse_values),
                "rmse_median_km_per_s": statistics.median(rmse_values),
                "rmse_ci95_lower_km_per_s": rmse_ci["lower"],
                "rmse_ci95_upper_km_per_s": rmse_ci["upper"],
                "mae_mean_km_per_s": mae_ci["mean"],
                "mae_std_km_per_s": statistics.pstdev(mae_values),
                "mae_median_km_per_s": statistics.median(mae_values),
                "mae_ci95_lower_km_per_s": mae_ci["lower"],
                "mae_ci95_upper_km_per_s": mae_ci["upper"],
                "bootstrap_unit": "unique_target_velocity_model",
                "bootstrap_sample_size": len(rmse_values),
                "bootstrap_resamples": bootstrap_resamples,
                "rmse_bootstrap_seed": _stable_seed(bootstrap_seed, f"method:{method_id}:rmse"),
                "mae_bootstrap_seed": _stable_seed(bootstrap_seed, f"method:{method_id}:mae"),
            }
        )
    return output


def summarize_target_level_paired_deltas(
    target_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    permutation_resamples: int = DEFAULT_PERMUTATION_RESAMPLES,
    permutation_seed: int = DEFAULT_PERMUTATION_SEED,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> list[dict[str, Any]]:
    """Summarize paired differences with target-level bootstrap and sign-flips."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        grouped[str(row["comparison_id"])].append(row)
    output: list[dict[str, Any]] = []
    for comparison_id, rows in sorted(grouped.items()):
        rmse_values = [_metric(row, "delta_rmse_km_per_s") for row in rows]
        mae_values = [_metric(row, "delta_mae_km_per_s") for row in rows]
        rmse_seed = _stable_seed(bootstrap_seed, f"comparison:{comparison_id}:rmse")
        mae_seed = _stable_seed(bootstrap_seed, f"comparison:{comparison_id}:mae")
        rmse_perm_seed = _stable_seed(permutation_seed, f"comparison:{comparison_id}:rmse")
        mae_perm_seed = _stable_seed(permutation_seed, f"comparison:{comparison_id}:mae")
        rmse_ci = bootstrap_mean_confidence_interval(
            rmse_values, resamples=bootstrap_resamples, seed=rmse_seed
        )
        mae_ci = bootstrap_mean_confidence_interval(
            mae_values, resamples=bootstrap_resamples, seed=mae_seed
        )
        first = rows[0]
        output.append(
            {
                "comparison_id": comparison_id,
                "first_method_id": str(first["first_method_id"]),
                "second_method_id": str(first["second_method_id"]),
                "first_method_label": str(first.get("first_method_label", first["first_method_id"])),
                "second_method_label": str(first.get("second_method_label", first["second_method_id"])),
                "target_count": len(rows),
                "acquisition_case_count": sum(
                    int(row["acquisition_case_count"]) for row in rows
                ),
                "delta_definition": "first method target-level metric minus second; negative favors first",
                "rmse_delta_mean_km_per_s": rmse_ci["mean"],
                "rmse_delta_std_km_per_s": statistics.pstdev(rmse_values),
                "rmse_delta_median_km_per_s": statistics.median(rmse_values),
                "rmse_delta_ci95_lower_km_per_s": rmse_ci["lower"],
                "rmse_delta_ci95_upper_km_per_s": rmse_ci["upper"],
                "rmse_target_win_count": sum(value < -tie_tolerance for value in rmse_values),
                "rmse_target_loss_count": sum(value > tie_tolerance for value in rmse_values),
                "rmse_target_tie_count": sum(abs(value) <= tie_tolerance for value in rmse_values),
                "mae_delta_mean_km_per_s": mae_ci["mean"],
                "mae_delta_std_km_per_s": statistics.pstdev(mae_values),
                "mae_delta_median_km_per_s": statistics.median(mae_values),
                "mae_delta_ci95_lower_km_per_s": mae_ci["lower"],
                "mae_delta_ci95_upper_km_per_s": mae_ci["upper"],
                "mae_target_win_count": sum(value < -tie_tolerance for value in mae_values),
                "mae_target_loss_count": sum(value > tie_tolerance for value in mae_values),
                "mae_target_tie_count": sum(abs(value) <= tie_tolerance for value in mae_values),
                "rmse_exploratory_sign_flip_p_value": paired_sign_flip_p_value(
                    rmse_values,
                    resamples=permutation_resamples,
                    seed=rmse_perm_seed,
                ),
                "mae_exploratory_sign_flip_p_value": paired_sign_flip_p_value(
                    mae_values,
                    resamples=permutation_resamples,
                    seed=mae_perm_seed,
                ),
                "bootstrap_unit": "unique_target_velocity_model",
                "bootstrap_sample_size": len(rmse_values),
                "bootstrap_resamples": bootstrap_resamples,
                "permutation_resamples": permutation_resamples,
                "rmse_bootstrap_seed": rmse_seed,
                "mae_bootstrap_seed": mae_seed,
                "rmse_permutation_seed": rmse_perm_seed,
                "mae_permutation_seed": mae_perm_seed,
                "rmse_ci_contains_zero": rmse_ci["lower"] <= 0.0 <= rmse_ci["upper"],
                "mae_ci_contains_zero": mae_ci["lower"] <= 0.0 <= mae_ci["upper"],
            }
        )
    return output


def _load_target_by_case(
    split_path: Path,
    fixed_case_ids: Sequence[str],
) -> dict[str, dict[str, str]]:
    rows = _read_csv(split_path)
    by_case: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        target_hash = str(row.get("target_hash", ""))
        split = str(row.get("split", ""))
        family = str(row.get("scenario", ""))
        if not case_id or not target_hash or not split:
            raise ValueError(f"Grouped split row lacks case_id, target_hash, or split: {row!r}")
        if case_id in by_case and by_case[case_id] != {
            "target_hash": target_hash,
            "split": split,
            "family": family,
        }:
            raise ValueError(f"Grouped split contains conflicting rows for case {case_id}.")
        by_case[case_id] = {
            "target_hash": target_hash,
            "split": split,
            "family": family,
        }
    expected = set(fixed_case_ids)
    actual = set(by_case)
    if actual != expected and not expected.issubset(actual):
        raise ValueError(
            "Grouped split does not contain all fixed test cases; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
        )
    selected = {case_id: by_case[case_id] for case_id in fixed_case_ids}
    non_test = {case_id: row["split"] for case_id, row in selected.items() if row["split"] != "test"}
    if non_test:
        raise ValueError(f"Fixed test IDs are not all assigned to test: {non_test}")
    return selected


def _validate_case_metric_coverage(
    rows: Sequence[Mapping[str, Any]],
    fixed_case_ids: Sequence[str],
) -> None:
    expected = set(fixed_case_ids)
    by_method: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_method[str(row["method_id"])].add(str(row["case_id"]))
    incomplete = {
        method: sorted(expected - case_ids)
        for method, case_ids in by_method.items()
        if case_ids != expected
    }
    if incomplete:
        raise ValueError(f"Acquisition metrics do not cover fixed test cases: {incomplete}")


def _write_summary_markdown(
    path: Path,
    *,
    summary_payload: Mapping[str, Any],
    method_summary: Sequence[Mapping[str, Any]],
    paired_summary: Sequence[Mapping[str, Any]],
    output_dir: Path,
    repo_root: Path,
) -> None:
    target_count = int(summary_payload["unique_test_target_count"])
    case_count = int(summary_payload["fixed_test_case_count"])
    lines = [
        "# Target-Level Uncertainty Summary",
        "",
        "## Audit finding",
        "",
        "The preceding `phase9_grouped_split_uncertainty_v1` artifact used a case-level "
        "percentile bootstrap and case-level paired sign-flip calculations. Its nominal "
        "sample size was 25 acquisition cases, although those cases represent only five "
        "independent target velocity models.",
        "",
        f"This corrected report contains {case_count} acquisition cases and {target_count} "
        "unique target hashes. Acquisition geometries are averaged within target before "
        "any resampling. The resampling unit is therefore the unique target velocity model, "
        "not the acquisition case.",
        "",
        "## Method summaries",
        "",
        "| Method | Targets | Acquisition cases | Mean RMSE | Target-level 95% CI | Mean MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_summary:
        lines.append(
            f"| `{row['method_id']}` | {row['target_count']} | {row['acquisition_case_count']} | "
            f"{float(row['rmse_mean_km_per_s']):.6f} | "
            f"[{float(row['rmse_ci95_lower_km_per_s']):.6f}, {float(row['rmse_ci95_upper_km_per_s']):.6f}] | "
            f"{float(row['mae_mean_km_per_s']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Target-level paired comparisons",
            "",
            "Negative deltas favor the first method. Wins and losses below are counted over "
            "target-level deltas, not over acquisition geometries.",
            "",
            "| Comparison | Targets | Mean RMSE delta | Target-level 95% CI | Wins/Losses/Ties | Exploratory sign-flip p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired_summary:
        lines.append(
            f"| `{row['comparison_id']}` | {row['target_count']} | "
            f"{float(row['rmse_delta_mean_km_per_s']):.6f} | "
            f"[{float(row['rmse_delta_ci95_lower_km_per_s']):.6f}, "
            f"{float(row['rmse_delta_ci95_upper_km_per_s']):.6f}] | "
            f"{row['rmse_target_win_count']}/{row['rmse_target_loss_count']}/{row['rmse_target_tie_count']} | "
            f"{float(row['rmse_exploratory_sign_flip_p_value']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            f"The target-level sample size is only {target_count}. The intervals and sign-flip "
            "values are descriptive sensitivity summaries, not strong population inference. "
            "The five target models also represent one target per geological family in this "
            "corrected test set, so they cannot support family-general conclusions.",
            "",
            "The target-level aggregation rule will be mandatory for Benchmark v2: calculate "
            "case metrics, average them within target, and treat unique target models as the "
            "independent observations.",
            "",
            "## Outputs",
            "",
            f"- Acquisition metrics: `{_relative(output_dir / 'acquisition_case_metrics.csv', repo_root)}`",
            f"- Target-level metrics: `{_relative(output_dir / 'target_level_metrics.csv', repo_root)}`",
            f"- Target-level paired deltas: `{_relative(output_dir / 'target_level_paired_deltas.csv', repo_root)}`",
            f"- JSON metadata and summary: `{_relative(output_dir / 'audit_metadata.json', repo_root)}`, "
            f"`{_relative(output_dir / 'target_level_uncertainty_summary.json', repo_root)}`",
            "",
        ]
    )
    _write_text(path, "\n".join(lines))


def _write_experiment_note(
    path: Path,
    *,
    summary_payload: Mapping[str, Any],
    output_dir: Path,
    repo_root: Path,
) -> None:
    lines = [
        "# Experiment: Target-Level Uncertainty for the Corrected Grouped Test Set",
        "",
        f"Experiment ID: {TARGET_LEVEL_UNCERTAINTY_REPORT_ID}",
        "",
        "## Objective",
        "",
        "Correct the statistical experimental unit after target-level leakage was removed. "
        "The geological target velocity model, identified by exact canonical target hash, "
        "is the independent unit; acquisition geometries are repeated measurements within target.",
        "",
        "## Protocol",
        "",
        f"- Fixed test cases: `{summary_payload['fixed_test_case_count']}`.",
        f"- Unique target models: `{summary_payload['unique_test_target_count']}`.",
        "- Case RMSE/MAE is first averaged arithmetically within target.",
        "- Percentile bootstrap resamples target-level values only.",
        "- Paired sign-flip calculations use target-level paired deltas only.",
        "- The five-target results are descriptive; no strong population inference is claimed.",
        "",
        "## Outputs",
        "",
        f"- `{_relative(output_dir / 'acquisition_case_metrics.csv', repo_root)}`",
        f"- `{_relative(output_dir / 'target_level_metrics.csv', repo_root)}`",
        f"- `{_relative(output_dir / 'target_level_paired_deltas.csv', repo_root)}`",
        f"- `{_relative(output_dir / 'target_level_uncertainty_summary.md', repo_root)}`",
        "",
    ]
    _write_text(path, "\n".join(lines))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV artifact is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_fixed_case_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required fixed-case JSON is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    case_ids = payload.get("case_ids") if isinstance(payload, dict) else None
    if not isinstance(case_ids, list) or not all(isinstance(value, str) for value in case_ids):
        raise ValueError("fixed_test_case_ids.json must contain a string case_ids list.")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("fixed_test_case_ids.json contains duplicate case IDs.")
    return list(case_ids)


def _metric(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid metric {key!r}: {row!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Metric {key!r} must be finite.")
    return value


def _stable_seed(base_seed: int, label: str) -> int:
    return int(base_seed) + sum((index + 1) * ord(character) for index, character in enumerate(label))


def _resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _write_text(path, "")
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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("\n" if text and not text.endswith("\n") else ""), encoding="utf-8")
