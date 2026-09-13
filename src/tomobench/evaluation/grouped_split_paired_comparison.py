"""Paired fixed-test comparison for the corrected target-grouped evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from tomobench.evaluation.phase8_paired_test_comparison import (
    PAIRED_METRIC_CONTRACT,
    aggregate_paired_test_summary,
    validate_paired_metric_contract,
)
from tomobench.tomography.comparison import compute_ml_cell_centered_case_metrics
from tomobench.utils.paths import get_repo_root


COMPARISON_ID = "phase9_grouped_split_paired_comparison_v1"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/phase9_grouped_split_paired_comparison_v1"
)
DEFAULT_ML_METRICS_PATH = Path(
    "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/"
    "models/pca-linear/fixed_split/metrics.json"
)
DEFAULT_ABLATION_METRICS_PATH = Path(
    "outputs/generated/ml_baselines/phase9_grouped_split_ablation_v1/cell_centered_metrics.csv"
)
DEFAULT_FAIR_BASELINE_PATH = Path(
    "outputs/generated/submission_readiness/fair_reference_ray_baseline_v1/"
    "fair_reference_baseline_case_metrics.csv"
)
DEFAULT_KNOWN_RAY_PATH = Path(
    "outputs/generated/classical_tomography/known_ray_perturbation_v1/"
    "known_ray_perturbation_case_summary.csv"
)
DEFAULT_GROUPED_SPLIT_PATH = Path(
    "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/"
    "target_group_split_assignments.csv"
)
DEFAULT_UNCERTAINTY_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/phase9_grouped_split_uncertainty_v1"
)


@dataclass(frozen=True)
class GroupedSplitPairedComparisonOutputs:
    """Files written by the corrected paired comparison."""

    output_dir: Path
    paired_test_case_metrics_csv: Path
    paired_test_summary_csv: Path
    paired_test_summary_json: Path
    fixed_test_case_ids_json: Path
    paired_family_summary_csv: Path
    experiment_note: Path


def run_grouped_split_uncertainty_report(
    *,
    input_dir: Path = DEFAULT_OUTPUT_DIR,
    output_dir: Path = DEFAULT_UNCERTAINTY_OUTPUT_DIR,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 8_201,
    permutation_resamples: int = 10_000,
    permutation_seed: int = 8_202,
):
    """Run bootstrap and paired-delta reporting against the corrected paired artifact."""
    from tomobench.evaluation.phase8_uncertainty_report import (
        run_phase8_uncertainty_report,
    )

    return run_phase8_uncertainty_report(
        input_dir=input_dir,
        output_dir=output_dir,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        permutation_resamples=permutation_resamples,
        permutation_seed=permutation_seed,
        include_travel_time_only_reference=False,
        pairings=(
            (
                "full_ml_vs_fixed_ray",
                "pca_linear_observation_regressor",
                "optimized_fixed_ray",
            ),
            (
                "full_ml_vs_reference_prior",
                "pca_linear_observation_regressor",
                "reference_prior",
            ),
            (
                "full_ml_vs_known_ray",
                "pca_linear_observation_regressor",
                "known_ray_inversion_diagnostic",
            ),
            (
                "full_ml_vs_realistic_full_input",
                "pca_linear_observation_regressor",
                "full_input_euclidean_path",
            ),
            (
                "full_ml_vs_travel_time_only",
                "pca_linear_observation_regressor",
                "travel_time_only",
            ),
            (
                "full_ml_vs_no_travel_time",
                "pca_linear_observation_regressor",
                "no_travel_time",
            ),
            (
                "full_ml_vs_shuffled_travel_time",
                "pca_linear_observation_regressor",
                "shuffled_travel_time",
            ),
            (
                "full_ml_vs_mean_target",
                "pca_linear_observation_regressor",
                "mean_target",
            ),
            (
                "travel_time_only_vs_fixed_ray",
                "travel_time_only",
                "optimized_fixed_ray",
            ),
        ),
        report_id="phase9_grouped_split_uncertainty_v1",
        experiment_note_path=Path(
            "wiki/experiments/phase9_grouped_split_uncertainty_v1.md"
        ),
    )


def run_grouped_split_paired_comparison(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ml_metrics_path: Path = DEFAULT_ML_METRICS_PATH,
    ablation_metrics_path: Path = DEFAULT_ABLATION_METRICS_PATH,
    fair_baseline_path: Path = DEFAULT_FAIR_BASELINE_PATH,
    known_ray_path: Path = DEFAULT_KNOWN_RAY_PATH,
    grouped_split_path: Path = DEFAULT_GROUPED_SPLIT_PATH,
) -> GroupedSplitPairedComparisonOutputs:
    """Create a same-case comparison from the corrected grouped outputs."""
    repo_root = get_repo_root()
    resolved_output = _resolve(output_dir, repo_root)
    resolved_output.mkdir(parents=True, exist_ok=True)
    resolved_ml = _resolve(ml_metrics_path, repo_root)
    resolved_ablation = _resolve(ablation_metrics_path, repo_root)
    resolved_fair = _resolve(fair_baseline_path, repo_root)
    resolved_known = _resolve(known_ray_path, repo_root)
    resolved_split = _resolve(grouped_split_path, repo_root)

    split_rows = _read_csv(resolved_split)
    fixed_test_case_ids = _fixed_test_case_ids(split_rows)
    ml_case_rows = compute_ml_cell_centered_case_metrics(resolved_ml, repo_root)
    ablation_rows = _read_csv(resolved_ablation)
    fair_rows = _read_csv(resolved_fair)
    known_rows = _read_csv(resolved_known)

    paired_rows: list[dict[str, Any]] = []
    ml_by_case = _index_test_rows(ml_case_rows, fixed_test_case_ids, "grouped PCA-linear ML")
    for case_id in fixed_test_case_ids:
        row = ml_by_case[case_id]
        paired_rows.append(
            _metric_row(
                case_id=case_id,
                family=str(row["family"]),
                method_id="pca_linear_observation_regressor",
                method_label="PCA-linear observation regressor",
                method_family="ml",
                comparison_role="primary_ml",
                diagnostic_only=False,
                is_primary=True,
                source_artifact=resolved_ml,
                repo_root=repo_root,
                rmse=row["all_cell_velocity_rmse_km_per_s"],
                mae=row["all_cell_velocity_mae_km_per_s"],
                source_split=str(row.get("split", "test")),
                prediction_path=row.get("prediction_path"),
            )
        )

    ablation_metadata = {
        "full_input": (
            "privileged_full_input",
            "Full-input PCA-linear model (true-model ray path)",
            "privileged_ml_diagnostic",
            True,
            False,
        ),
        "full_input_euclidean_path": (
            "full_input_euclidean_path",
            "Full-input PCA-linear model (Euclidean path)",
            "realistic_ml_alternative",
            False,
            True,
        ),
        "travel_time_only": (
            "travel_time_only",
            "Travel-time-only ablation",
            "observation_ablation",
            False,
            True,
        ),
        "no_travel_time": (
            "no_travel_time",
            "No-travel-time ablation",
            "observation_ablation",
            False,
            True,
        ),
        "shuffled_travel_time": (
            "shuffled_travel_time",
            "Shuffled-travel-time ablation",
            "observation_ablation",
            False,
            True,
        ),
        "mean_target": (
            "mean_target",
            "Mean-target baseline",
            "sanity_baseline",
            False,
            True,
        ),
        "geometry_only": (
            "geometry_only",
            "Geometry-only baseline",
            "sanity_baseline",
            False,
            True,
        ),
        "path_length_only": (
            "privileged_path_length_only",
            "Path-length-only diagnostic (true-model ray path)",
            "privileged_ml_diagnostic",
            True,
            False,
        ),
        "euclidean_path_only": (
            "euclidean_path_only",
            "Euclidean-path-only baseline",
            "sanity_baseline",
            False,
            True,
        ),
        "family_mean_target_diagnostic": (
            "family_mean_target_diagnostic",
            "Family-mean-target diagnostic",
            "sanity_baseline",
            True,
            False,
        ),
    }
    ablation_by_variant = _index_ablation_rows(ablation_rows)
    for variant_id, metadata in ablation_metadata.items():
        method_id, label, role, diagnostic_only, is_primary = metadata
        variant_rows = _index_test_rows(
            ablation_by_variant.get(variant_id, []),
            fixed_test_case_ids,
            f"ablation {variant_id}",
        )
        for case_id in fixed_test_case_ids:
            row = variant_rows[case_id]
            paired_rows.append(
                _metric_row(
                    case_id=case_id,
                    family=str(row["family"]),
                    method_id=method_id,
                    method_label=label,
                    method_family="ml" if not diagnostic_only else "diagnostic",
                    comparison_role=role,
                    diagnostic_only=diagnostic_only,
                    is_primary=is_primary,
                    source_artifact=resolved_ablation,
                    repo_root=repo_root,
                    rmse=row["all_cell_velocity_rmse_km_per_s"],
                    mae=row["all_cell_velocity_mae_km_per_s"],
                    source_split=str(row.get("split", "test")),
                    prediction_path=row.get("prediction_path"),
                )
            )

    fair_by_case = _index_test_rows(fair_rows, fixed_test_case_ids, "fair reference baseline")
    for method_id, label, role, rmse_key, mae_key in (
        (
            "reference_prior",
            "Reference 1D prior",
            "classical_reference_prior",
            "reference_prior_all_cell_rmse_km_per_s",
            "reference_prior_all_cell_mae_km_per_s",
        ),
        (
            "optimized_fixed_ray",
            "Validation-selected fixed-ray inversion",
            "primary_classical",
            "optimized_fixed_ray_all_cell_rmse_km_per_s",
            "optimized_fixed_ray_all_cell_mae_km_per_s",
        ),
    ):
        for case_id in fixed_test_case_ids:
            row = fair_by_case[case_id]
            paired_rows.append(
                _metric_row(
                    case_id=case_id,
                    family=str(row["family"]),
                    method_id=method_id,
                    method_label=label,
                    method_family="classical_reference_ray",
                    comparison_role=role,
                    diagnostic_only=False,
                    is_primary=True,
                    source_artifact=resolved_fair,
                    repo_root=repo_root,
                    rmse=row[rmse_key],
                    mae=row[mae_key],
                    source_split=str(row.get("split", "test")),
                    regularization_damping=row.get("damping"),
                    regularization_smoothing=row.get("smoothing"),
                )
            )

    known_by_case = _index_rows_by_case(known_rows, "corrected known-ray perturbation")
    for case_id in fixed_test_case_ids:
        row = known_by_case[case_id]
        family = str(next(item["family"] for item in fair_rows if item["case_id"] == case_id))
        paired_rows.append(
            _metric_row(
                case_id=case_id,
                family=family,
                method_id="known_ray_inversion_diagnostic",
                method_label="Known-ray perturbation diagnostic (true geometry)",
                method_family="classical_known_ray",
                comparison_role="privileged_classical_diagnostic",
                diagnostic_only=True,
                is_primary=False,
                source_artifact=resolved_known,
                repo_root=repo_root,
                rmse=row["known_ray_perturbation_all_cell_rmse_km_per_s"],
                mae=row["known_ray_perturbation_all_cell_mae_km_per_s"],
                source_split="test",
                regularization_damping=row.get("damping"),
                regularization_smoothing=row.get("smoothing"),
            )
        )

    validate_paired_metric_contract(paired_rows)
    aggregate = aggregate_paired_test_summary(paired_rows)
    split_counts = _split_counts(split_rows)
    split_unique_targets = _split_unique_target_counts(split_rows)
    summary = {
        "artifact_type": "grouped_split_paired_comparison_summary",
        "experiment_id": COMPARISON_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "paired_test_case_count": len(fixed_test_case_ids),
        "fixed_test_case_ids": fixed_test_case_ids,
        "split_case_counts": split_counts,
        "split_unique_target_counts": split_unique_targets,
        "split_strategy": "target_grouped_family_stratified",
        "metric_contract": {
            "name": PAIRED_METRIC_CONTRACT,
            "case_unit": "one fixed target-grouped synthetic test case",
            "ml": "Node-centered predictions and targets are converted to cell-centered values by averaging each cell's eight surrounding nodes.",
            "classical": "Classical rows are already reported as cell-centered metrics.",
            "aggregation_std": "Population standard deviation across per-case metrics (ddof=0).",
        },
        "method_summary": aggregate["method_summary"],
        "family_wise_summary": aggregate["family_wise_summary"],
        "method_ranking": aggregate["method_ranking"],
        "source_files": {
            "grouped_split": _relative(resolved_split, repo_root),
            "ml_metrics": _relative(resolved_ml, repo_root),
            "ablation_metrics": _relative(resolved_ablation, repo_root),
            "fair_reference_baseline": _relative(resolved_fair, repo_root),
            "known_ray_perturbation": _relative(resolved_known, repo_root),
        },
        "protocol_notes": [
            "All methods are joined by the exact 25 case IDs in the grouped fixed test split.",
            "The validation-selected fixed-ray regularization pair is frozen before test rows are read.",
            "The true-model path-length and true-geometry known-ray methods are retained only as explicitly labeled privileged diagnostics.",
            "The effective independent target count is the unique target-group count, not the case count.",
        ],
        "claim_caveats": [
            "Synthetic-only benchmark; no real-data validation is claimed.",
            "The test set contains five unique target hashes, one per family, repeated across acquisition geometries.",
            "A paired ranking over 25 cases should not be interpreted as universal ML superiority.",
        ],
    }

    case_csv = resolved_output / "paired_test_case_metrics.csv"
    summary_csv = resolved_output / "paired_test_summary.csv"
    family_csv = resolved_output / "paired_family_summary.csv"
    summary_json = resolved_output / "paired_test_summary.json"
    fixed_ids_json = resolved_output / "fixed_test_case_ids.json"
    note_path = repo_root / "wiki/experiments/phase9_grouped_split_paired_comparison_v1.md"
    _write_csv(case_csv, paired_rows)
    _write_csv(summary_csv, aggregate["method_summary"])
    _write_csv(family_csv, aggregate["family_wise_summary"])
    _write_json(summary_json, summary)
    _write_json(
        fixed_ids_json,
        {
            "artifact_type": "grouped_split_fixed_test_case_ids",
            "experiment_id": COMPARISON_ID,
            "case_count": len(fixed_test_case_ids),
            "case_ids": fixed_test_case_ids,
            "split_unique_target_count": split_unique_targets["test"],
        },
    )
    _write_note(note_path, summary, repo_root)
    return GroupedSplitPairedComparisonOutputs(
        output_dir=resolved_output,
        paired_test_case_metrics_csv=case_csv,
        paired_test_summary_csv=summary_csv,
        paired_test_summary_json=summary_json,
        fixed_test_case_ids_json=fixed_ids_json,
        paired_family_summary_csv=family_csv,
        experiment_note=note_path,
    )


def _metric_row(
    *,
    case_id: str,
    family: str,
    method_id: str,
    method_label: str,
    method_family: str,
    comparison_role: str,
    diagnostic_only: bool,
    is_primary: bool,
    source_artifact: Path,
    repo_root: Path,
    rmse: Any,
    mae: Any,
    source_split: str,
    prediction_path: Any = None,
    regularization_damping: Any = None,
    regularization_smoothing: Any = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "family": family,
        "split": "test",
        "source_split": source_split,
        "method_id": method_id,
        "method_label": method_label,
        "method_family": method_family,
        "comparison_role": comparison_role,
        "diagnostic_only": diagnostic_only,
        "is_primary": is_primary,
        "is_post_hoc_selection": False,
        "metric_contract": PAIRED_METRIC_CONTRACT,
        "rmse_km_per_s": float(rmse),
        "mae_km_per_s": float(mae),
        "regularization_damping": _optional_float(regularization_damping),
        "regularization_smoothing": _optional_float(regularization_smoothing),
        "source_artifact": _relative(source_artifact, repo_root),
        "prediction_path": (
            _relative(Path(str(prediction_path)), repo_root)
            if prediction_path not in (None, "")
            else ""
        ),
    }


def _fixed_test_case_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    ids = sorted(str(row["case_id"]) for row in rows if str(row.get("split")) == "test")
    if not ids:
        raise ValueError("Grouped split assignment does not contain test cases.")
    if len(ids) != len(set(ids)):
        raise ValueError("Grouped split assignment contains duplicate test case IDs.")
    return ids


def _index_ablation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("variant_id", "")), []).append(dict(row))
    return grouped


def _index_test_rows(
    rows: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed = _index_rows_by_case(rows, label)
    expected = set(case_ids)
    observed = set(indexed)
    if observed != expected:
        raise ValueError(
            f"{label} does not cover the fixed test cases; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}."
        )
    return indexed


def _index_rows_by_case(
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("split", "test")) != "test" and label != "corrected known-ray perturbation":
            continue
        case_id = str(row.get("case_id", ""))
        if not case_id:
            raise ValueError(f"{label} contains a row without case_id.")
        if case_id in indexed:
            raise ValueError(f"{label} contains duplicate case_id: {case_id}.")
        indexed[case_id] = row
    return indexed


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split", ""))
        counts[split] = counts.get(split, 0) + 1
    return dict(sorted(counts.items()))


def _split_unique_target_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    groups: dict[str, set[str]] = {}
    for row in rows:
        split = str(row.get("split", ""))
        target_hash = str(row.get("target_hash", ""))
        if not target_hash:
            raise ValueError("Grouped split rows require target_vector_sha256.")
        groups.setdefault(split, set()).add(target_hash)
    return {split: len(values) for split, values in sorted(groups.items())}


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _write_note(path: Path, summary: Mapping[str, Any], repo_root: Path) -> None:
    lines = [
        "# Experiment: Corrected Grouped-Split Paired Comparison",
        "",
        "## Objective",
        "",
        "Compare the corrected ML, ablation, classical-prior, fixed-ray, and known-ray "
        "diagnostic results on exactly the same target-grouped fixed test cases.",
        "",
        "## Independence and metric contract",
        "",
        f"- Test cases: {summary['paired_test_case_count']}; unique target groups in test: "
        f"{summary['split_unique_target_counts']['test']}.",
        "- All rows use the cell-centered velocity RMSE/MAE contract.",
        "- Grouped assignment keeps every exact target hash in one split.",
        "- The validation-selected fixed-ray pair is frozen before test evaluation.",
        "",
        "## Primary ranking",
        "",
    ]
    for row in summary["method_ranking"]:
        lines.append(
            f"{row['rank']}. {row['method_id']}: "
            f"mean RMSE {float(row['rmse_mean_km_per_s']):.6f} km/s; "
            f"mean MAE {float(row['mae_mean_km_per_s']):.6f} km/s."
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The true-model ray-path and true-geometry known-ray rows are privileged diagnostics "
            "and are not realistic held-out inputs. The five unique test targets make the case "
            "count larger than the effective target-level sample size. Results remain synthetic "
            "and interpolation-oriented within the configured families.",
            "",
            "## Outputs",
            "",
            "- paired_test_case_metrics.csv",
            "- paired_test_summary.csv",
            "- paired_family_summary.csv",
            "- paired_test_summary.json",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required grouped comparison CSV is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved)
