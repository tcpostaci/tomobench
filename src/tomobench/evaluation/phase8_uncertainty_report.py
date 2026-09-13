"""Phase 8B uncertainty and paired statistical reporting."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tomobench.evaluation.phase8_paired_test_comparison import (
    PAIRED_METRIC_CONTRACT,
    validate_paired_metric_contract,
)
from tomobench.utils.paths import get_repo_root

PHASE8_UNCERTAINTY_REPORT_ID = "phase8_uncertainty_report_v1"
DEFAULT_PHASE8_PAIRED_INPUT_DIR = Path(
    "outputs/generated/submission_readiness/phase8_paired_test_comparison_v1"
)
DEFAULT_PHASE8_UNCERTAINTY_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/phase8_uncertainty_report_v1"
)
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 8_201
DEFAULT_PERMUTATION_RESAMPLES = 10_000
DEFAULT_PERMUTATION_SEED = 8_202
DEFAULT_TIE_TOLERANCE = 1.0e-12
NEAR_ZERO_DELTA_THRESHOLD_KM_PER_S = 0.005

FULL_ML_METHOD_ID = "pca_linear_observation_regressor"
REFERENCE_RAY_METHOD_ID = "reference_ray_fixed_ray_phase6"
KNOWN_RAY_METHOD_ID = "known_ray_inversion_diagnostic"
TRAVEL_TIME_ONLY_METHOD_ID = "travel_time_only"

_REQUIRED_PAIRINGS = (
    ("full_ml_vs_reference_ray", FULL_ML_METHOD_ID, REFERENCE_RAY_METHOD_ID),
    ("full_ml_vs_known_ray", FULL_ML_METHOD_ID, KNOWN_RAY_METHOD_ID),
    ("full_ml_vs_mean_target", FULL_ML_METHOD_ID, "mean_target"),
    ("full_ml_vs_no_travel_time", FULL_ML_METHOD_ID, "no_travel_time"),
    ("full_ml_vs_shuffled_travel_time", FULL_ML_METHOD_ID, "shuffled_travel_time"),
    ("full_ml_vs_travel_time_only", FULL_ML_METHOD_ID, TRAVEL_TIME_ONLY_METHOD_ID),
)
_OPTIONAL_PAIRINGS = (
    ("travel_time_only_vs_reference_ray", TRAVEL_TIME_ONLY_METHOD_ID, REFERENCE_RAY_METHOD_ID),
)


@dataclass(frozen=True)
class Phase8UncertaintyReportOutputs:
    """Paths written by the Phase 8B uncertainty report."""

    output_dir: Path
    method_uncertainty_summary_csv: Path
    method_uncertainty_summary_json: Path
    paired_delta_summary_csv: Path
    paired_delta_summary_json: Path
    paired_case_deltas_csv: Path
    bootstrap_samples_metadata_json: Path
    experiment_note: Path


def bootstrap_mean_confidence_interval(
    values: Sequence[float] | Iterable[float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Return a deterministic percentile bootstrap interval for a sample mean.

    The bootstrap resamples cases with replacement. The percentile calculation uses
    linear interpolation between sorted bootstrap means, so the result is independent
    of NumPy availability and reproducible across supported Python versions.
    """
    sample = [float(value) for value in values]
    _validate_resampling_arguments(sample, resamples, confidence_level)
    rng = random.Random(seed)
    sample_size = len(sample)
    bootstrap_means = [
        statistics.mean(sample[rng.randrange(sample_size)] for _ in range(sample_size))
        for _ in range(resamples)
    ]
    bootstrap_means.sort()
    tail_probability = (1.0 - confidence_level) / 2.0
    return {
        "mean": statistics.mean(sample),
        "lower": _percentile(bootstrap_means, tail_probability),
        "upper": _percentile(bootstrap_means, 1.0 - tail_probability),
    }


def paired_sign_flip_p_value(
    deltas: Sequence[float] | Iterable[float],
    *,
    resamples: int = DEFAULT_PERMUTATION_RESAMPLES,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> float:
    """Return an exploratory two-sided paired sign-flip p-value for mean delta.

    The null distribution independently flips the sign of each observed per-case
    delta. A plus-one correction keeps the Monte Carlo p-value valid at finite
    resample counts. This is descriptive/exploratory evidence for the small fixed
    test set, not a definitive hypothesis test.
    """
    sample = [float(value) for value in deltas]
    if not sample:
        raise ValueError("At least one paired delta is required for a sign-flip test.")
    if resamples < 1:
        raise ValueError("resamples must be at least 1.")
    if not all(math.isfinite(value) for value in sample):
        raise ValueError("Sign-flip deltas must be finite.")
    observed = abs(statistics.mean(sample))
    if observed == 0.0:
        return 1.0
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(resamples):
        null_mean = statistics.mean(
            value if rng.getrandbits(1) else -value for value in sample
        )
        if abs(null_mean) >= observed:
            exceedances += 1
    return (exceedances + 1) / (resamples + 1)


def compute_paired_case_deltas(
    rows: Sequence[Mapping[str, Any]],
    first_method_id: str,
    second_method_id: str,
    *,
    comparison_id: str | None = None,
    case_ids: Sequence[str] | None = None,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> list[dict[str, Any]]:
    """Join two methods by exact case ID and calculate first-minus-second deltas."""
    if first_method_id == second_method_id:
        raise ValueError("Paired comparison methods must be different.")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance cannot be negative.")
    by_method: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        method_id = str(row.get("method_id", ""))
        case_id = str(row.get("case_id", ""))
        if method_id in {first_method_id, second_method_id}:
            if not case_id:
                raise ValueError("Paired delta rows require non-empty case_id values.")
            if case_id in by_method[method_id]:
                raise ValueError(f"Duplicate row for method={method_id}, case={case_id}.")
            by_method[method_id][case_id] = row
    first_rows = by_method.get(first_method_id, {})
    second_rows = by_method.get(second_method_id, {})
    if not first_rows:
        raise ValueError(f"Paired comparison method is missing: {first_method_id}.")
    if not second_rows:
        raise ValueError(f"Paired comparison method is missing: {second_method_id}.")
    expected_ids = list(case_ids) if case_ids is not None else sorted(first_rows)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("case_ids contains duplicates.")
    first_ids = set(first_rows)
    second_ids = set(second_rows)
    expected_set = set(expected_ids)
    if first_ids != expected_set:
        raise ValueError(
            f"{first_method_id} does not cover the requested exact case IDs; "
            f"missing={sorted(expected_set - first_ids)}, extra={sorted(first_ids - expected_set)}."
        )
    if second_ids != expected_set:
        raise ValueError(
            f"{second_method_id} does not cover the requested exact case IDs; "
            f"missing={sorted(expected_set - second_ids)}, extra={sorted(second_ids - expected_set)}."
        )
    deltas: list[dict[str, Any]] = []
    for case_id in expected_ids:
        first = first_rows[case_id]
        second = second_rows[case_id]
        first_family = str(first.get("family", ""))
        second_family = str(second.get("family", ""))
        if first_family != second_family:
            raise ValueError(
                f"Paired methods have inconsistent family labels for case {case_id}: "
                f"{first_method_id}={first_family!r}, {second_method_id}={second_family!r}."
            )
        first_rmse = _metric_value(first, "rmse_km_per_s")
        second_rmse = _metric_value(second, "rmse_km_per_s")
        first_mae = _metric_value(first, "mae_km_per_s")
        second_mae = _metric_value(second, "mae_km_per_s")
        rmse_delta = first_rmse - second_rmse
        mae_delta = first_mae - second_mae
        deltas.append(
            {
                "comparison_id": comparison_id or f"{first_method_id}_vs_{second_method_id}",
                "first_method_id": first_method_id,
                "second_method_id": second_method_id,
                "case_id": case_id,
                "family": str(first.get("family", "")),
                "split": "test",
                "metric_contract": PAIRED_METRIC_CONTRACT,
                "first_rmse_km_per_s": first_rmse,
                "second_rmse_km_per_s": second_rmse,
                "delta_rmse_km_per_s": rmse_delta,
                "first_mae_km_per_s": first_mae,
                "second_mae_km_per_s": second_mae,
                "delta_mae_km_per_s": mae_delta,
                "first_improves_rmse": rmse_delta < -tie_tolerance,
                "first_improves_mae": mae_delta < -tie_tolerance,
            }
        )
    return deltas


def aggregate_paired_delta_summary(
    delta_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    permutation_resamples: int = DEFAULT_PERMUTATION_RESAMPLES,
    permutation_seed: int = DEFAULT_PERMUTATION_SEED,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> list[dict[str, Any]]:
    """Aggregate paired case deltas with bootstrap intervals and sign-flip p-values."""
    if not delta_rows:
        raise ValueError("Paired delta rows are empty.")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance cannot be negative.")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        if row.get("metric_contract") != PAIRED_METRIC_CONTRACT:
            raise ValueError("Paired delta rows do not use the shared cell-centered contract.")
        if str(row.get("split")) != "test":
            raise ValueError("Paired delta rows must have split='test'.")
        grouped[str(row.get("comparison_id", ""))].append(row)
    if "" in grouped:
        raise ValueError("Every paired delta row needs comparison_id.")

    summaries: list[dict[str, Any]] = []
    for comparison_id in sorted(grouped):
        rows = grouped[comparison_id]
        first_method_id = str(rows[0].get("first_method_id", ""))
        second_method_id = str(rows[0].get("second_method_id", ""))
        if not first_method_id or not second_method_id:
            raise ValueError(f"Comparison {comparison_id} needs both method IDs.")
        case_ids = [str(row.get("case_id", "")) for row in rows]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError(f"Comparison {comparison_id} contains duplicate case IDs.")
        if any(
            str(row.get("first_method_id")) != first_method_id
            or str(row.get("second_method_id")) != second_method_id
            for row in rows
        ):
            raise ValueError(f"Comparison {comparison_id} changes method IDs across cases.")
        rmse_deltas = [_metric_value(row, "delta_rmse_km_per_s") for row in rows]
        mae_deltas = [_metric_value(row, "delta_mae_km_per_s") for row in rows]
        rmse_bootstrap_seed = _stable_seed(bootstrap_seed, f"{comparison_id}:rmse")
        mae_bootstrap_seed = _stable_seed(bootstrap_seed, f"{comparison_id}:mae")
        rmse_permutation_seed = _stable_seed(permutation_seed, f"{comparison_id}:rmse")
        mae_permutation_seed = _stable_seed(permutation_seed, f"{comparison_id}:mae")
        rmse_ci = bootstrap_mean_confidence_interval(
            rmse_deltas, resamples=bootstrap_resamples, seed=rmse_bootstrap_seed
        )
        mae_ci = bootstrap_mean_confidence_interval(
            mae_deltas, resamples=bootstrap_resamples, seed=mae_bootstrap_seed
        )
        rmse_counts = _win_loss_tie_counts(rmse_deltas, tie_tolerance)
        mae_counts = _win_loss_tie_counts(mae_deltas, tie_tolerance)
        first = rows[0]
        summaries.append(
            {
                "comparison_id": comparison_id,
                "first_method_id": first_method_id,
                "second_method_id": second_method_id,
                "first_method_label": str(first.get("first_method_label", first_method_id)),
                "second_method_label": str(first.get("second_method_label", second_method_id)),
                "case_count": len(rows),
                "delta_definition": "first method metric minus second method metric; negative favors first method",
                "rmse_delta_mean_km_per_s": rmse_ci["mean"],
                "rmse_delta_median_km_per_s": statistics.median(rmse_deltas),
                "rmse_delta_std_km_per_s": statistics.pstdev(rmse_deltas),
                "rmse_delta_iqr_km_per_s": _interquartile_range(rmse_deltas),
                "rmse_delta_ci95_lower_km_per_s": rmse_ci["lower"],
                "rmse_delta_ci95_upper_km_per_s": rmse_ci["upper"],
                "rmse_win_count": rmse_counts["win_count"],
                "rmse_loss_count": rmse_counts["loss_count"],
                "rmse_tie_count": rmse_counts["tie_count"],
                "rmse_improvement_fraction": rmse_counts["win_count"] / len(rows),
                "mae_delta_mean_km_per_s": mae_ci["mean"],
                "mae_delta_median_km_per_s": statistics.median(mae_deltas),
                "mae_delta_std_km_per_s": statistics.pstdev(mae_deltas),
                "mae_delta_iqr_km_per_s": _interquartile_range(mae_deltas),
                "mae_delta_ci95_lower_km_per_s": mae_ci["lower"],
                "mae_delta_ci95_upper_km_per_s": mae_ci["upper"],
                "mae_win_count": mae_counts["win_count"],
                "mae_loss_count": mae_counts["loss_count"],
                "mae_tie_count": mae_counts["tie_count"],
                "mae_improvement_fraction": mae_counts["win_count"] / len(rows),
                "rmse_exploratory_sign_flip_p_value": paired_sign_flip_p_value(
                    rmse_deltas,
                    resamples=permutation_resamples,
                    seed=rmse_permutation_seed,
                ),
                "mae_exploratory_sign_flip_p_value": paired_sign_flip_p_value(
                    mae_deltas,
                    resamples=permutation_resamples,
                    seed=mae_permutation_seed,
                ),
                "rmse_bootstrap_seed": rmse_bootstrap_seed,
                "mae_bootstrap_seed": mae_bootstrap_seed,
                "rmse_permutation_seed": rmse_permutation_seed,
                "mae_permutation_seed": mae_permutation_seed,
                "bootstrap_resamples": bootstrap_resamples,
                "permutation_resamples": permutation_resamples,
                "rmse_ci_contains_zero": _interval_contains_zero(rmse_ci["lower"], rmse_ci["upper"]),
                "mae_ci_contains_zero": _interval_contains_zero(mae_ci["lower"], mae_ci["upper"]),
            }
        )
    return summaries


def run_phase8_uncertainty_report(
    *,
    output_dir: Path = DEFAULT_PHASE8_UNCERTAINTY_OUTPUT_DIR,
    input_dir: Path = DEFAULT_PHASE8_PAIRED_INPUT_DIR,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    permutation_resamples: int = DEFAULT_PERMUTATION_RESAMPLES,
    permutation_seed: int = DEFAULT_PERMUTATION_SEED,
    include_travel_time_only_reference: bool = True,
    experiment_note_path: Path | None = None,
    pairings: Sequence[tuple[str, str, str]] | None = None,
    report_id: str = PHASE8_UNCERTAINTY_REPORT_ID,
) -> Phase8UncertaintyReportOutputs:
    """Run Phase 8B from the existing Phase 8A paired case-metrics artifact."""
    repo_root = get_repo_root()
    resolved_input_dir = _resolve_path(input_dir, repo_root)
    resolved_output_dir = _resolve_path(output_dir, repo_root)
    case_metrics_path = resolved_input_dir / "paired_test_case_metrics.csv"
    fixed_ids_path = resolved_input_dir / "fixed_test_case_ids.json"
    source_summary_path = resolved_input_dir / "paired_test_summary.json"
    rows = _read_csv_rows(case_metrics_path)
    validate_paired_metric_contract(rows)
    fixed_payload = _read_json(fixed_ids_path)
    fixed_case_ids = _fixed_case_ids(fixed_payload)
    _validate_fixed_case_coverage(rows, fixed_case_ids)
    if not fixed_case_ids:
        raise ValueError("Phase 8A fixed test case list is empty.")
    method_ids = sorted({str(row["method_id"]) for row in rows})
    method_summaries, method_seed_metadata = _summarize_methods(
        rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )

    pair_rows: list[dict[str, Any]] = []
    included_pairings: list[tuple[str, str, str]] = []
    missing_data_notes: list[str] = []
    active_pairings = list(pairings) if pairings is not None else list(_REQUIRED_PAIRINGS)
    if pairings is None and include_travel_time_only_reference:
        active_pairings.extend(_OPTIONAL_PAIRINGS)
    for comparison_id, first_method_id, second_method_id in active_pairings:
        available = set(method_ids)
        if first_method_id not in available or second_method_id not in available:
            missing_data_notes.append(
                f"Omitted {comparison_id}: missing method(s) among "
                f"{first_method_id} and {second_method_id}; no metrics were invented."
            )
            continue
        deltas = compute_paired_case_deltas(
            rows,
            first_method_id,
            second_method_id,
            comparison_id=comparison_id,
            case_ids=fixed_case_ids,
        )
        labels = _method_labels(rows)
        for delta in deltas:
            delta["first_method_label"] = labels[first_method_id]
            delta["second_method_label"] = labels[second_method_id]
        pair_rows.extend(deltas)
        included_pairings.append((comparison_id, first_method_id, second_method_id))

    pair_summaries = aggregate_paired_delta_summary(
        pair_rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        permutation_resamples=permutation_resamples,
        permutation_seed=permutation_seed,
    )
    method_summary_json = {
        "artifact_type": "phase8_method_uncertainty_summary",
        "experiment_id": report_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_phase8a_case_metrics": _relative_to_repo(case_metrics_path, repo_root),
        "input_phase8a_summary": _relative_to_repo(source_summary_path, repo_root),
        "fixed_test_case_ids_path": _relative_to_repo(fixed_ids_path, repo_root),
        "paired_test_case_count": len(fixed_case_ids),
        "metric_contract": _metric_contract_description(),
        "bootstrap": _bootstrap_metadata(
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            confidence_level=0.95,
            method_seed_metadata=method_seed_metadata,
        ),
        "methods_included": method_ids,
        "method_summary": method_summaries,
        "claim_caveats": _claim_caveats(),
    }
    pair_summaries_with_interpretations = _add_pair_interpretations(
        pair_summaries, method_summaries
    )
    paired_summary_json = {
        "artifact_type": "phase8_paired_delta_summary",
        "experiment_id": report_id,
        "generated_at_utc": method_summary_json["generated_at_utc"],
        "input_phase8a_case_metrics": _relative_to_repo(case_metrics_path, repo_root),
        "fixed_test_case_ids_path": _relative_to_repo(fixed_ids_path, repo_root),
        "paired_test_case_count": len(fixed_case_ids),
        "metric_contract": _metric_contract_description(),
        "bootstrap": _bootstrap_metadata(
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            confidence_level=0.95,
            method_seed_metadata={},
        ),
        "permutation_test": {
            "type": "paired sign-flip Monte Carlo test for mean delta",
            "resamples": permutation_resamples,
            "seed": permutation_seed,
            "p_value_note": "Exploratory only; p-values are not definitive with 25 paired synthetic cases.",
        },
        "delta_definition": "first method metric minus second method metric; negative favors first method",
        "near_zero_delta_threshold_km_per_s": NEAR_ZERO_DELTA_THRESHOLD_KM_PER_S,
        "comparisons_included": [
            {
                "comparison_id": comparison_id,
                "first_method_id": first_method_id,
                "second_method_id": second_method_id,
            }
            for comparison_id, first_method_id, second_method_id in included_pairings
        ],
        "paired_delta_summary": pair_summaries_with_interpretations,
        "missing_data_notes": missing_data_notes,
        "claim_caveats": _claim_caveats(),
    }
    metadata = {
        "artifact_type": "phase8_bootstrap_samples_metadata",
        "experiment_id": report_id,
        "input_artifact": _relative_to_repo(case_metrics_path, repo_root),
        "paired_test_case_count": len(fixed_case_ids),
        "bootstrap": method_summary_json["bootstrap"],
        "paired_comparison_bootstrap_seeds": {
            row["comparison_id"]: {
                "rmse": row["rmse_bootstrap_seed"],
                "mae": row["mae_bootstrap_seed"],
            }
            for row in pair_summaries
        },
        "paired_comparison_permutation_seeds": {
            row["comparison_id"]: {
                "rmse": row["rmse_permutation_seed"],
                "mae": row["mae_permutation_seed"],
            }
            for row in pair_summaries
        },
        "permutation": paired_summary_json["permutation_test"],
        "note": "Bootstrap sample arrays are intentionally not persisted; seeds and algorithm metadata make the intervals reproducible.",
    }

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    method_csv_path = resolved_output_dir / "method_uncertainty_summary.csv"
    method_json_path = resolved_output_dir / "method_uncertainty_summary.json"
    pair_csv_path = resolved_output_dir / "paired_delta_summary.csv"
    pair_json_path = resolved_output_dir / "paired_delta_summary.json"
    case_delta_path = resolved_output_dir / "paired_case_deltas.csv"
    metadata_path = resolved_output_dir / "bootstrap_samples_metadata.json"
    note_path = _resolve_path(
        experiment_note_path
        if experiment_note_path is not None
        else repo_root / "wiki/experiments/phase8_uncertainty_report_v1.md",
        repo_root,
    )
    _write_csv(method_csv_path, method_summaries)
    _write_csv(pair_csv_path, pair_summaries_with_interpretations)
    _write_csv(case_delta_path, pair_rows)
    _write_json(method_json_path, method_summary_json)
    _write_json(pair_json_path, paired_summary_json)
    _write_json(metadata_path, metadata)
    _write_experiment_note(
        note_path,
        method_summary=method_summary_json,
        paired_summary=paired_summary_json,
        output_dir=resolved_output_dir,
        repo_root=repo_root,
        report_id=report_id,
    )
    return Phase8UncertaintyReportOutputs(
        output_dir=resolved_output_dir,
        method_uncertainty_summary_csv=method_csv_path,
        method_uncertainty_summary_json=method_json_path,
        paired_delta_summary_csv=pair_csv_path,
        paired_delta_summary_json=pair_json_path,
        paired_case_deltas_csv=case_delta_path,
        bootstrap_samples_metadata_json=metadata_path,
        experiment_note=note_path,
    )


def summarize_method_uncertainty(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Summarize per-case RMSE/MAE distributions by method."""
    validate_paired_metric_contract(rows)
    summaries, _ = _summarize_methods(
        rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    return summaries


def _summarize_methods(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method_id"])].append(row)
    labels = _method_labels(rows)
    summaries: list[dict[str, Any]] = []
    seed_metadata: dict[str, dict[str, int]] = {}
    for method_id in sorted(by_method):
        method_rows = by_method[method_id]
        rmse_values = [_metric_value(row, "rmse_km_per_s") for row in method_rows]
        mae_values = [_metric_value(row, "mae_km_per_s") for row in method_rows]
        rmse_seed = _stable_seed(bootstrap_seed, f"method:{method_id}:rmse")
        mae_seed = _stable_seed(bootstrap_seed, f"method:{method_id}:mae")
        rmse_ci = bootstrap_mean_confidence_interval(
            rmse_values, resamples=bootstrap_resamples, seed=rmse_seed
        )
        mae_ci = bootstrap_mean_confidence_interval(
            mae_values, resamples=bootstrap_resamples, seed=mae_seed
        )
        first = method_rows[0]
        summaries.append(
            {
                "method_id": method_id,
                "method_label": labels[method_id],
                "method_family": str(first.get("method_family", "")),
                "comparison_role": str(first.get("comparison_role", "")),
                "diagnostic_only": _as_bool(first.get("diagnostic_only", False)),
                "is_primary": _as_bool(first.get("is_primary", True)),
                "is_post_hoc_selection": _as_bool(first.get("is_post_hoc_selection", False)),
                "case_count": len(method_rows),
                "rmse_mean_km_per_s": rmse_ci["mean"],
                "rmse_std_km_per_s": statistics.pstdev(rmse_values),
                "rmse_median_km_per_s": statistics.median(rmse_values),
                "rmse_iqr_km_per_s": _interquartile_range(rmse_values),
                "rmse_ci95_lower_km_per_s": rmse_ci["lower"],
                "rmse_ci95_upper_km_per_s": rmse_ci["upper"],
                "mae_mean_km_per_s": mae_ci["mean"],
                "mae_std_km_per_s": statistics.pstdev(mae_values),
                "mae_median_km_per_s": statistics.median(mae_values),
                "mae_iqr_km_per_s": _interquartile_range(mae_values),
                "mae_ci95_lower_km_per_s": mae_ci["lower"],
                "mae_ci95_upper_km_per_s": mae_ci["upper"],
                "rmse_bootstrap_seed": rmse_seed,
                "mae_bootstrap_seed": mae_seed,
                "bootstrap_resamples": bootstrap_resamples,
            }
        )
        seed_metadata[method_id] = {"rmse": rmse_seed, "mae": mae_seed}
    return summaries, seed_metadata


def _add_pair_interpretations(
    summaries: Sequence[Mapping[str, Any]],
    method_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    method_by_id = {str(row["method_id"]): row for row in method_summaries}
    output: list[dict[str, Any]] = []
    for source in summaries:
        row = dict(source)
        first_id = str(row["first_method_id"])
        second_id = str(row["second_method_id"])
        first_method = method_by_id[first_id]
        second_method = method_by_id[second_id]
        rmse_intervals_overlap = _intervals_overlap(
            float(first_method["rmse_ci95_lower_km_per_s"]),
            float(first_method["rmse_ci95_upper_km_per_s"]),
            float(second_method["rmse_ci95_lower_km_per_s"]),
            float(second_method["rmse_ci95_upper_km_per_s"]),
        )
        delta_near_zero = (
            abs(float(row["rmse_delta_mean_km_per_s"]))
            <= NEAR_ZERO_DELTA_THRESHOLD_KM_PER_S
        )
        if row["comparison_id"] == "full_ml_vs_travel_time_only":
            if rmse_intervals_overlap or delta_near_zero or row["rmse_ci_contains_zero"]:
                interpretation = (
                    "The full-ML versus travel-time-only difference is not meaningfully "
                    "established on this fixed synthetic test set."
                )
            else:
                interpretation = (
                    "The paired fixed-test results separate full ML and travel-time-only "
                    "descriptively, but the result remains benchmark-specific."
                )
        elif float(row["rmse_delta_ci95_upper_km_per_s"]) < 0.0:
            if first_id == FULL_ML_METHOD_ID and second_id in {
                "no_travel_time",
                "shuffled_travel_time",
            }:
                interpretation = (
                    "Full ML has lower paired RMSE than this control under the configured "
                    "synthetic benchmark, consistent with a contribution from travel-time signal."
                )
            elif first_id == FULL_ML_METHOD_ID and second_id == REFERENCE_RAY_METHOD_ID:
                interpretation = (
                    "Full ML has lower paired RMSE than the Phase 6 fixed-ray reference "
                    "baseline under this configured synthetic benchmark."
                )
            else:
                interpretation = (
                    f"{first_method['method_label']} has lower paired RMSE than "
                    f"{second_method['method_label']} under this configured synthetic benchmark."
                )
        else:
            interpretation = (
                "The paired result does not clearly establish a lower full-method RMSE "
                "under the reported interval rule."
            )
        row["rmse_method_ci_intervals_overlap"] = rmse_intervals_overlap
        row["rmse_delta_is_near_zero"] = delta_near_zero
        row["interpretation"] = interpretation
        output.append(row)
    return output


def _validate_fixed_case_coverage(rows: Sequence[Mapping[str, Any]], case_ids: Sequence[str]) -> None:
    expected = set(case_ids)
    row_ids = {str(row.get("case_id", "")) for row in rows}
    if row_ids != expected:
        raise ValueError(
            "Phase 8A case metrics do not match fixed_test_case_ids.json exactly; "
            f"missing={sorted(expected - row_ids)}, extra={sorted(row_ids - expected)}."
        )
    by_method: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_method[str(row["method_id"])].add(str(row["case_id"]))
    incomplete = {
        method_id: sorted(expected - method_case_ids)
        for method_id, method_case_ids in by_method.items()
        if method_case_ids != expected
    }
    if incomplete:
        raise ValueError(f"Phase 8A method coverage is incomplete for fixed cases: {incomplete}.")


def _fixed_case_ids(payload: Mapping[str, Any]) -> list[str]:
    raw_ids = payload.get("case_ids")
    if not isinstance(raw_ids, list) or not all(isinstance(case_id, str) for case_id in raw_ids):
        raise ValueError("fixed_test_case_ids.json must contain a string case_ids list.")
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("fixed_test_case_ids.json contains duplicate case IDs.")
    return list(raw_ids)


def _method_labels(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in rows:
        method_id = str(row["method_id"])
        label = str(row.get("method_label", method_id))
        if method_id in labels and labels[method_id] != label:
            raise ValueError(f"Method {method_id} has inconsistent method labels.")
        labels[method_id] = label
    return labels


def _win_loss_tie_counts(values: Sequence[float], tie_tolerance: float) -> dict[str, int]:
    wins = sum(value < -tie_tolerance for value in values)
    losses = sum(value > tie_tolerance for value in values)
    ties = len(values) - wins - losses
    return {"win_count": wins, "loss_count": losses, "tie_count": ties}


def _metric_contract_description() -> dict[str, Any]:
    return {
        "name": PAIRED_METRIC_CONTRACT,
        "reported_metrics": ["rmse_km_per_s", "mae_km_per_s"],
        "case_unit": "one fixed synthetic test case",
        "ml_and_target": "Phase 8A cell-centered conversion from eight-node Cartesian-cell averages",
        "classical": "Phase 8A classical outputs are already cell-centered",
        "aggregation_std": "Population standard deviation across per-case metrics (ddof=0)",
    }


def _claim_caveats() -> list[str]:
    return [
        "Synthetic-only benchmark; no real-data validation is claimed.",
        "Known-ray inversion is an optimistic diagnostic because it uses true-model ray geometry.",
        "Reference-ray inversion is a fixed-ray regularized baseline, not full nonlinear tomography.",
        "Permutation p-values are exploratory only because the paired sample has 25 fixed cases.",
        "The uncertainty report does not establish universal ML superiority.",
    ]


def _bootstrap_metadata(
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    confidence_level: float,
    method_seed_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": "case-level percentile bootstrap for the arithmetic mean",
        "confidence_level": confidence_level,
        "resamples": bootstrap_resamples,
        "base_seed": bootstrap_seed,
        "percentile_interpolation": "linear",
        "method_metric_seeds": method_seed_metadata,
    }


def _validate_resampling_arguments(
    sample: Sequence[float], resamples: int, confidence_level: float
) -> None:
    if not sample:
        raise ValueError("At least one value is required for a bootstrap interval.")
    if resamples < 1:
        raise ValueError("resamples must be at least 1.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    if not all(math.isfinite(value) for value in sample):
        raise ValueError("Bootstrap values must be finite.")


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty sample.")
    position = (len(sorted_values) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    weight = position - lower_index
    return float(
        sorted_values[lower_index]
        + weight * (sorted_values[upper_index] - sorted_values[lower_index])
    )


def _interquartile_range(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    return _percentile(ordered, 0.75) - _percentile(ordered, 0.25)


def _interval_contains_zero(lower: float, upper: float) -> bool:
    return lower <= 0.0 <= upper


def _intervals_overlap(first_lower: float, first_upper: float, second_lower: float, second_upper: float) -> bool:
    return max(first_lower, second_lower) <= min(first_upper, second_upper)


def _metric_value(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid metric {key!r} in row {row!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Metric {key!r} must be finite.")
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _stable_seed(base_seed: int, label: str) -> int:
    return int(base_seed) + sum((index + 1) * ord(character) for index, character in enumerate(label))


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 8A CSV artifact is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 8A JSON artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_experiment_note(
    path: Path,
    *,
    method_summary: Mapping[str, Any],
    paired_summary: Mapping[str, Any],
    output_dir: Path,
    repo_root: Path,
    report_id: str = PHASE8_UNCERTAINTY_REPORT_ID,
) -> None:
    method_rows = method_summary["method_summary"]
    pair_rows = paired_summary["paired_delta_summary"]
    lines = [
        "# Experiment: Phase 8B Uncertainty and Paired Statistical Reporting",
        "",
        "Experiment ID: " + report_id,
        "",
        "## Objective",
        "",
        "Add case-level uncertainty intervals and paired method-difference reporting to the "
        "Phase 8A fixed synthetic test-set comparison.",
        "",
        "## Input and statistical protocol",
        "",
        f"- Input: `{method_summary['input_phase8a_case_metrics']}` from Phase 8A; no model predictions were recomputed.",
        f"- Paired cases: `{method_summary['paired_test_case_count']}` exact fixed test cases.",
        "- RMSE and MAE retain the Phase 8A cell-centered metric contract.",
        "- Method intervals are percentile bootstrap 95% CIs for the per-case arithmetic mean.",
        "- Standard deviation uses population `ddof=0`; IQR uses linearly interpolated quartiles.",
        "- Paired deltas are first-method minus second-method; negative values favor the first method.",
        f"- The descriptive near-zero RMSE-delta flag uses `{NEAR_ZERO_DELTA_THRESHOLD_KM_PER_S:.3f}` km/s; it is not a significance threshold.",
        "- Sign-flip p-values are deterministic, two-sided, and exploratory only.",
        "",
        "## Method uncertainty summaries",
        "",
        "| Method | Mean RMSE | RMSE 95% CI | Mean MAE | MAE 95% CI | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| `{row['method_id']}` | {float(row['rmse_mean_km_per_s']):.6f} | "
            f"[{float(row['rmse_ci95_lower_km_per_s']):.6f}, {float(row['rmse_ci95_upper_km_per_s']):.6f}] | "
            f"{float(row['mae_mean_km_per_s']):.6f} | "
            f"[{float(row['mae_ci95_lower_km_per_s']):.6f}, {float(row['mae_ci95_upper_km_per_s']):.6f}] | "
            f"{row['case_count']} |"
        )
    lines.extend(["", "## Paired differences", ""])
    for row in pair_rows:
        lines.extend(
            [
                f"### `{row['comparison_id']}`",
                "",
                f"- Mean RMSE delta: `{float(row['rmse_delta_mean_km_per_s']):.6f}` km/s; "
                f"95% CI `[{float(row['rmse_delta_ci95_lower_km_per_s']):.6f}, {float(row['rmse_delta_ci95_upper_km_per_s']):.6f}]`; "
                f"wins/losses/ties `{row['rmse_win_count']}/{row['rmse_loss_count']}/{row['rmse_tie_count']}`; "
                f"first-method improvement fraction `{float(row['rmse_improvement_fraction']):.3f}`.",
                f"- Mean MAE delta: `{float(row['mae_delta_mean_km_per_s']):.6f}` km/s; "
                f"95% CI `[{float(row['mae_delta_ci95_lower_km_per_s']):.6f}, {float(row['mae_delta_ci95_upper_km_per_s']):.6f}]`; "
                f"wins/losses/ties `{row['mae_win_count']}/{row['mae_loss_count']}/{row['mae_tie_count']}`; "
                f"first-method improvement fraction `{float(row['mae_improvement_fraction']):.3f}`.",
                f"- Exploratory sign-flip p-values: RMSE `{float(row['rmse_exploratory_sign_flip_p_value']):.4f}`, "
                f"MAE `{float(row['mae_exploratory_sign_flip_p_value']):.4f}`.",
                f"- Interpretation: {row['interpretation']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation and claim boundaries",
            "",
            "The full-ML and travel-time-only comparison is treated cautiously: overlapping "
            "method-level intervals, a near-zero paired delta, or a paired interval containing "
            "zero does not establish a meaningful difference. Lower errors for full ML against "
            "the no-travel-time and shuffled-travel-time controls, when their paired RMSE interval "
            "is wholly below zero, are interpreted as synthetic-benchmark evidence that the travel-time "
            "signal contributes beyond those geometry/prior controls.",
            "",
            "A full-ML advantage over reference-ray is limited to the configured synthetic benchmark "
            "and the implemented Phase 6 fixed-ray regularized baseline. Known-ray remains an "
            "optimistic diagnostic using true-model ray geometry. These results do not provide real-data "
            "validation, do not demonstrate full nonlinear tomography, and do not establish universal "
            "ML superiority.",
            "",
            "## Outputs",
            "",
            f"- Method summary CSV/JSON: `{_relative_to_repo(output_dir / 'method_uncertainty_summary.csv', repo_root)}`; "
            f"`{_relative_to_repo(output_dir / 'method_uncertainty_summary.json', repo_root)}`",
            f"- Paired delta CSV/JSON: `{_relative_to_repo(output_dir / 'paired_delta_summary.csv', repo_root)}`; "
            f"`{_relative_to_repo(output_dir / 'paired_delta_summary.json', repo_root)}`",
            f"- Case deltas: `{_relative_to_repo(output_dir / 'paired_case_deltas.csv', repo_root)}`",
            f"- Reproducibility metadata: `{_relative_to_repo(output_dir / 'bootstrap_samples_metadata.json', repo_root)}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
