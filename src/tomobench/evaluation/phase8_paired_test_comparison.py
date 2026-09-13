"""Paired Phase 8A comparison on the selected ML fixed test cases."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tomobench.tomography.comparison import compute_ml_cell_centered_case_metrics
from tomobench.utils.paths import get_repo_root

PHASE8_PAIRED_TEST_COMPARISON_ID = "phase8_paired_test_comparison_v1"
DEFAULT_PHASE8_PAIRED_TEST_COMPARISON_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/phase8_paired_test_comparison_v1"
)
DEFAULT_SELECTED_ML_METRICS_PATH = Path(
    "outputs/generated/ml_baselines/paper_pseudo_bending_ml_suite_v1/"
    "models/pca-linear/fixed_split/metrics.json"
)
DEFAULT_KNOWN_RAY_CASE_SUMMARY_PATH = Path(
    "outputs/generated/classical_tomography/known_ray_corpus_v1/"
    "known_ray_corpus_case_summary.csv"
)
DEFAULT_REFERENCE_RAY_CASE_SUMMARY_PATH = Path(
    "outputs/generated/classical_tomography/reference_ray_corpus_v1/"
    "reference_ray_corpus_case_summary.csv"
)
DEFAULT_PHASE7A_CELL_CENTERED_METRICS_PATH = Path(
    "outputs/generated/ml_baselines/phase7_observation_signal_ablation_v1/"
    "cell_centered_metrics.csv"
)
DEFAULT_PHASE7B_SUMMARY_PATH = Path(
    "outputs/generated/classical_tomography/reference_ray_regularization_sensitivity_v1/"
    "regularization_sensitivity_summary.json"
)
DEFAULT_PHASE7B_CASE_METRICS_PATH = Path(
    "outputs/generated/classical_tomography/reference_ray_regularization_sensitivity_v1/"
    "regularization_sensitivity_case_metrics.csv"
)
DEFAULT_CORPUS_MANIFEST_PATH = Path(
    "outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json"
)
DEFAULT_PHASE6_REFERENCE_DAMPING = 0.01
DEFAULT_PHASE6_REFERENCE_SMOOTHING = 1.0
PAIRED_METRIC_CONTRACT = "cell_centered_velocity_rmse_mae_km_per_s"

_REQUIRED_ABLATION_VARIANTS = (
    "mean_target",
    "no_travel_time",
    "shuffled_travel_time",
)
_OPTIONAL_ABLATION_VARIANTS = ("travel_time_only",)


@dataclass(frozen=True)
class Phase8PairedTestComparisonOutputs:
    """Paths written by the Phase 8A paired comparison."""

    output_dir: Path
    paired_test_case_metrics_csv: Path
    paired_test_summary_csv: Path
    paired_test_summary_json: Path
    fixed_test_case_ids_json: Path
    paired_family_summary_csv: Path
    experiment_note: Path


def run_phase8_paired_test_comparison(
    *,
    output_dir: Path = DEFAULT_PHASE8_PAIRED_TEST_COMPARISON_OUTPUT_DIR,
    selected_ml_metrics_path: Path = DEFAULT_SELECTED_ML_METRICS_PATH,
    known_ray_case_summary_path: Path = DEFAULT_KNOWN_RAY_CASE_SUMMARY_PATH,
    reference_ray_case_summary_path: Path = DEFAULT_REFERENCE_RAY_CASE_SUMMARY_PATH,
    phase7a_cell_centered_metrics_path: Path = DEFAULT_PHASE7A_CELL_CENTERED_METRICS_PATH,
    phase7b_summary_path: Path = DEFAULT_PHASE7B_SUMMARY_PATH,
    phase7b_case_metrics_path: Path = DEFAULT_PHASE7B_CASE_METRICS_PATH,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST_PATH,
    experiment_note_path: Path | None = None,
    include_reference_sensitivity_audit: bool = False,
) -> Phase8PairedTestComparisonOutputs:
    """Create a same-test-set comparison without rerunning training or tomography.

    The selected ML metrics JSON is the authority for the fixed test IDs. Classical and
    Phase 7A rows are then joined to those IDs; no filename-derived case selection is used.
    """
    repo_root = get_repo_root()
    resolved_output_dir = _resolve_path(Path(output_dir), repo_root)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    selected_metrics_path = _resolve_path(Path(selected_ml_metrics_path), repo_root)
    known_case_path = _resolve_path(Path(known_ray_case_summary_path), repo_root)
    reference_case_path = _resolve_path(Path(reference_ray_case_summary_path), repo_root)
    phase7a_path = _resolve_path(Path(phase7a_cell_centered_metrics_path), repo_root)
    phase7b_summary_file = _resolve_path(Path(phase7b_summary_path), repo_root)
    phase7b_case_file = _resolve_path(Path(phase7b_case_metrics_path), repo_root)
    manifest_path = _resolve_path(Path(corpus_manifest_path), repo_root)

    selected_metrics = _read_json(selected_metrics_path)
    fixed_test_case_ids = resolve_fixed_test_case_ids(selected_metrics_path, repo_root)
    manifest_cases = _load_manifest_cases(manifest_path)
    _validate_manifest_case_ids(fixed_test_case_ids, manifest_cases)
    known_rows = _read_csv_rows(known_case_path)
    reference_rows = _read_csv_rows(reference_case_path)
    phase7a_rows = _read_csv_rows(phase7a_path) if phase7a_path.is_file() else []
    phase7b_summary = _read_json(phase7b_summary_file) if phase7b_summary_file.is_file() else {}
    phase7b_rows = _read_csv_rows(phase7b_case_file) if phase7b_case_file.is_file() else []
    missing_data_notes: list[str] = []

    _validate_rows_against_manifest(
        known_rows, manifest_cases, "known-ray case summary", fixed_test_case_ids, repo_root
    )
    _validate_rows_against_manifest(
        reference_rows, manifest_cases, "reference-ray case summary", fixed_test_case_ids, repo_root
    )
    _validate_rows_against_manifest(phase7a_rows, manifest_cases, "Phase 7A metrics", repo_root=repo_root)
    phase6_reference = _phase6_reference_parameters(phase7b_summary, reference_rows)
    _validate_reference_baseline_parameters(reference_rows, phase6_reference)

    case_metrics_rows: list[dict[str, Any]] = []
    ml_case_rows = compute_ml_cell_centered_case_metrics(
        selected_metrics_path,
        repo_root,
        reference_case_rows=reference_rows,
    )
    ml_test_rows = _rows_by_case(
        [row for row in ml_case_rows if str(row.get("split")) == "test"],
        method="pca_linear_observation_regressor",
    )
    _require_case_ids(
        fixed_test_case_ids,
        ml_test_rows,
        "pca_linear_observation_regressor",
    )
    _validate_rows_against_manifest(
        ml_test_rows.values(), manifest_cases, "selected ML metrics", fixed_test_case_ids, repo_root
    )
    for case_id in fixed_test_case_ids:
        row = ml_test_rows[case_id]
        case_metrics_rows.append(
            _paired_row(
                case_id=case_id,
                manifest_case=manifest_cases[case_id],
                method_id="pca_linear_observation_regressor",
                method_label="PCA-linear observation regressor",
                method_family="ml",
                comparison_role="primary_ml",
                diagnostic_only=False,
                is_primary=True,
                is_post_hoc_selection=False,
                source_split="test",
                source_artifact=selected_metrics_path,
                target_grid_path=Path(str(manifest_cases[case_id]["target_grid_path"])),
                rmse=row["all_cell_velocity_rmse_km_per_s"],
                mae=row["all_cell_velocity_mae_km_per_s"],
                repo_root=repo_root,
                prediction_path=Path(str(row["prediction_path"])),
            )
        )

    case_metrics_rows.extend(
        _classical_rows(
            fixed_test_case_ids,
            manifest_cases,
            _rows_by_case(known_rows, method="known_ray_inversion_diagnostic"),
            method_id="known_ray_inversion_diagnostic",
            method_label="Known-ray inversion diagnostic",
            method_family="classical_known_ray",
            comparison_role="diagnostic_classical",
            diagnostic_only=True,
            is_primary=True,
            source_artifact=known_case_path,
            source_split="not_defined_in_corpus_output",
            repo_root=repo_root,
        )
    )
    case_metrics_rows.extend(
        _classical_rows(
            fixed_test_case_ids,
            manifest_cases,
            _rows_by_case(reference_rows, method="reference_ray_fixed_ray_phase6"),
            method_id="reference_ray_fixed_ray_phase6",
            method_label="Reference-ray fixed-ray inversion (Phase 6)",
            method_family="classical_reference_ray",
            comparison_role="primary_classical",
            diagnostic_only=False,
            is_primary=True,
            source_artifact=reference_case_path,
            source_split="not_defined_in_corpus_output",
            repo_root=repo_root,
            regularization_damping=phase6_reference["damping"],
            regularization_smoothing=phase6_reference["smoothing"],
        )
    )

    ablation_by_variant = _group_variant_rows(phase7a_rows)
    ablation_metadata = {
        "mean_target": (
            "Mean-target baseline",
            "sanity_baseline",
            "sanity_baseline",
        ),
        "no_travel_time": (
            "No-travel-time ablation",
            "sanity_baseline",
            "observation_ablation",
        ),
        "shuffled_travel_time": (
            "Shuffled-travel-time ablation",
            "sanity_baseline",
            "observation_ablation",
        ),
        "travel_time_only": (
            "Travel-time-only ablation",
            "sanity_baseline",
            "observation_ablation",
        ),
    }
    for variant_id in _REQUIRED_ABLATION_VARIANTS + _OPTIONAL_ABLATION_VARIANTS:
        variant_rows = ablation_by_variant.get(variant_id, {})
        if not variant_rows:
            missing_data_notes.append(
                f"Phase 7A variant '{variant_id}' was unavailable and was omitted from the paired table."
            )
            continue
        if not set(fixed_test_case_ids) <= set(variant_rows):
            missing = sorted(set(fixed_test_case_ids) - set(variant_rows))
            missing_data_notes.append(
                f"Phase 7A variant '{variant_id}' was omitted because it lacks fixed-test case IDs: "
                + ", ".join(missing)
            )
            continue
        label, method_family, comparison_role = ablation_metadata[variant_id]
        for case_id in fixed_test_case_ids:
            row = variant_rows[case_id]
            case_metrics_rows.append(
                _paired_row(
                    case_id=case_id,
                    manifest_case=manifest_cases[case_id],
                    method_id=variant_id,
                    method_label=label,
                    method_family=method_family,
                    comparison_role=comparison_role,
                    diagnostic_only=False,
                    is_primary=True,
                    is_post_hoc_selection=False,
                    source_split=str(row.get("split") or "test"),
                    source_artifact=phase7a_path,
                    target_grid_path=Path(str(manifest_cases[case_id]["target_grid_path"])),
                    rmse=row["all_cell_velocity_rmse_km_per_s"],
                    mae=row["all_cell_velocity_mae_km_per_s"],
                    repo_root=repo_root,
                    prediction_path=(
                        Path(str(row["prediction_path"]))
                        if row.get("prediction_path")
                        else None
                    ),
                )
            )

    if include_reference_sensitivity_audit:
        _validate_rows_against_manifest(
            phase7b_rows, manifest_cases, "Phase 7B sensitivity metrics", repo_root=repo_root
        )
        case_metrics_rows.extend(
            _reference_sensitivity_audit_rows(
                fixed_test_case_ids=fixed_test_case_ids,
                manifest_cases=manifest_cases,
                phase7b_summary=phase7b_summary,
                phase7b_rows=phase7b_rows,
                source_artifact=phase7b_case_file,
                repo_root=repo_root,
            )
        )
    else:
        if phase7b_summary:
            missing_data_notes.append(
                "The Phase 7B best covered-cell setting is retained as a post-hoc sensitivity "
                "audit in metadata and does not replace the Phase 6 reference-ray row."
            )
        else:
            missing_data_notes.append(
                "Phase 7B sensitivity metadata was unavailable; no sensitivity row was included."
            )

    validate_paired_metric_contract(case_metrics_rows)
    aggregate = aggregate_paired_test_summary(case_metrics_rows)
    methods = _method_metadata(case_metrics_rows)
    source_files = {
        "paper_candidate_v3_artifact_index": _relative_to_repo(
            repo_root / "outputs/generated/paper_candidate_v3/artifact_index.json", repo_root
        ),
        "paper_candidate_v3_summary": _relative_to_repo(
            repo_root / "outputs/generated/paper_candidate_v3/notes/package_summary.md", repo_root
        ),
        "selected_ml_metrics": _relative_to_repo(selected_metrics_path, repo_root),
        "known_ray_case_summary": _relative_to_repo(known_case_path, repo_root),
        "reference_ray_case_summary": _relative_to_repo(reference_case_path, repo_root),
        "phase7a_cell_centered_metrics": _relative_to_repo(phase7a_path, repo_root),
        "phase7b_summary": _relative_to_repo(phase7b_summary_file, repo_root),
        "phase7b_case_metrics": _relative_to_repo(phase7b_case_file, repo_root),
        "corpus_manifest": _relative_to_repo(manifest_path, repo_root),
    }
    sensitivity_metadata = _sensitivity_metadata(
        phase7b_summary,
        included=include_reference_sensitivity_audit,
    )
    summary = {
        "artifact_type": "phase8_paired_test_comparison_summary",
        "experiment_id": PHASE8_PAIRED_TEST_COMPARISON_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "paired_test_case_count": len(fixed_test_case_ids),
        "fixed_test_case_ids": fixed_test_case_ids,
        "fixed_test_case_ids_path": "fixed_test_case_ids.json",
        "methods_included": methods,
        "ranking_metric": "rmse_mean_km_per_s ascending; ties by mae_mean_km_per_s then method_id",
        "metric_contract": {
            "name": PAIRED_METRIC_CONTRACT,
            "reported_metrics": ["rmse_km_per_s", "mae_km_per_s"],
            "ml": "Node-centered predictions and true target grids are each averaged over the eight nodes of every Cartesian cell before RMSE/MAE computation.",
            "classical": "Phase 6 and Phase 7B classical outputs are already cell-centered and are compared against cell-centered target values.",
            "aggregation_std": "Population standard deviation across per-case metrics (ddof=0).",
        },
        "method_summary": aggregate["method_summary"],
        "family_wise_summary": aggregate["family_wise_summary"],
        "method_ranking": aggregate["method_ranking"],
        "regularization_policy": {
            "primary_reference_ray": {
                "setting_source": "Phase 6 documented baseline, retained as primary",
                "damping": phase6_reference["damping"],
                "smoothing": phase6_reference["smoothing"],
                "method_id": "reference_ray_fixed_ray_phase6",
            },
            "phase7b_best_audit": sensitivity_metadata,
        },
        "source_files": source_files,
        "missing_data_notes": missing_data_notes,
        "claim_caveats": [
            "Synthetic-only benchmark; no real-data validation is claimed.",
            "Known-ray inversion is an optimistic diagnostic because it uses true-model ray geometry.",
            "Reference-ray inversion is a fixed-ray regularized baseline, not full nonlinear tomography.",
            "The paired ranking applies only to these fixed synthetic test cases and does not establish universal ML superiority.",
            "Any Phase 7B best-setting row is a post-hoc sensitivity/audit result and must not silently replace the Phase 6 primary reference-ray baseline.",
        ],
    }

    fixed_ids_path = resolved_output_dir / "fixed_test_case_ids.json"
    case_metrics_path = resolved_output_dir / "paired_test_case_metrics.csv"
    summary_csv_path = resolved_output_dir / "paired_test_summary.csv"
    family_summary_path = resolved_output_dir / "paired_family_summary.csv"
    summary_json_path = resolved_output_dir / "paired_test_summary.json"
    _write_json(
        fixed_ids_path,
        {
            "artifact_type": "phase8_fixed_test_case_ids",
            "experiment_id": PHASE8_PAIRED_TEST_COMPARISON_ID,
            "source_metrics_path": _relative_to_repo(selected_metrics_path, repo_root),
            "source_split": "test",
            "case_count": len(fixed_test_case_ids),
            "case_ids": fixed_test_case_ids,
            "cases": [
                {
                    "case_id": case_id,
                    "family": manifest_cases[case_id]["family"],
                    "prediction_path": _relative_to_repo(
                        _resolve_path(
                            Path(str(_test_case_by_id(selected_metrics, case_id)["prediction_path"])),
                            repo_root,
                        ),
                        repo_root,
                    ),
                    "target_grid_path": _relative_to_repo(
                        _resolve_path(Path(str(manifest_cases[case_id]["target_grid_path"])), repo_root),
                        repo_root,
                    ),
                }
                for case_id in fixed_test_case_ids
            ],
        },
    )
    _write_csv(case_metrics_path, case_metrics_rows)
    _write_csv(summary_csv_path, aggregate["method_summary"])
    _write_csv(family_summary_path, aggregate["family_wise_summary"])
    _write_json(summary_json_path, summary)
    note_path = _resolve_path(
        Path(experiment_note_path)
        if experiment_note_path is not None
        else repo_root / "wiki/experiments/phase8_paired_test_comparison_v1.md",
        repo_root,
    )
    _write_experiment_note(
        note_path,
        summary=summary,
        case_metrics_path=case_metrics_path,
        summary_csv_path=summary_csv_path,
        family_summary_path=family_summary_path,
        repo_root=repo_root,
    )
    return Phase8PairedTestComparisonOutputs(
        output_dir=resolved_output_dir,
        paired_test_case_metrics_csv=case_metrics_path,
        paired_test_summary_csv=summary_csv_path,
        paired_test_summary_json=summary_json_path,
        fixed_test_case_ids_json=fixed_ids_path,
        paired_family_summary_csv=family_summary_path,
        experiment_note=note_path,
    )


def resolve_fixed_test_case_ids(
    metrics_path_or_payload: str | Path | Mapping[str, Any],
    repo_root: Path | None = None,
) -> list[str]:
    """Resolve fixed test IDs from the metrics case records, validating prediction links."""
    root = repo_root or get_repo_root()
    metrics = (
        _read_json(_resolve_path(Path(metrics_path_or_payload), root))
        if isinstance(metrics_path_or_payload, (str, Path))
        else dict(metrics_path_or_payload)
    )
    model_name = metrics.get("model_name")
    if model_name is not None and str(model_name) != "pca_linear_observation_regressor":
        raise ValueError(f"Selected ML metrics use unexpected model_name: {model_name!r}.")
    split_metrics = metrics.get("split_metrics")
    if not isinstance(split_metrics, dict) or not isinstance(split_metrics.get("test"), dict):
        raise ValueError("Selected ML metrics must contain split_metrics.test.")
    case_metrics = split_metrics["test"].get("case_metrics")
    if not isinstance(case_metrics, list) or not case_metrics:
        raise ValueError("Selected ML metrics split_metrics.test.case_metrics is empty or invalid.")
    case_ids: list[str] = []
    for case in case_metrics:
        if not isinstance(case, dict) or not str(case.get("case_id", "")):
            raise ValueError("Every fixed-test ML case record must contain a case_id.")
        case_id = str(case["case_id"])
        if case_id in case_ids:
            raise ValueError(f"Duplicate fixed-test case_id in ML metrics: {case_id}.")
        prediction_path_value = case.get("prediction_path")
        if not prediction_path_value:
            raise ValueError(f"Fixed-test ML case {case_id} has no prediction_path.")
        prediction_path = _resolve_path(Path(str(prediction_path_value)), root)
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Fixed-test ML prediction for {case_id} is missing: {prediction_path}")
        prediction = _read_json(prediction_path)
        prediction_case_id = prediction.get("case_id")
        if prediction_case_id is not None and str(prediction_case_id) != case_id:
            raise ValueError(
                f"Prediction case_id {prediction_case_id!r} does not match metrics case_id {case_id!r}."
            )
        case_ids.append(case_id)
    return case_ids


def join_methods_by_case_id(
    fixed_test_case_ids: Sequence[str],
    method_rows: Mapping[str, Iterable[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Join per-method metric rows by exact case ID into long-form paired rows.

    This helper intentionally accepts only explicit case IDs and metric values; callers can
    attach method metadata afterward. Every method must cover the same case-ID set.
    """
    expected_ids = list(fixed_test_case_ids)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("fixed_test_case_ids contains duplicates.")
    joined: list[dict[str, Any]] = []
    for method_id, rows in method_rows.items():
        indexed = _rows_by_case(list(rows), method=str(method_id))
        _require_case_ids(expected_ids, indexed, str(method_id))
        for case_id in expected_ids:
            row = indexed[case_id]
            joined.append(
                {
                    "case_id": case_id,
                    "method_id": str(method_id),
                    "split": "test",
                    "metric_contract": PAIRED_METRIC_CONTRACT,
                    "rmse_km_per_s": _metric_from_row(row, "rmse"),
                    "mae_km_per_s": _metric_from_row(row, "mae"),
                }
            )
    return joined


def validate_paired_metric_contract(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate that paired rows use one explicit, finite cell-centered metric contract."""
    if not rows:
        raise ValueError("Paired metric rows are empty.")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("method_id", "")), str(row.get("case_id", "")))
        if not key[0] or not key[1]:
            raise ValueError("Every paired metric row needs method_id and case_id.")
        if key in seen:
            raise ValueError(f"Duplicate paired metric row: method={key[0]}, case={key[1]}.")
        seen.add(key)
        if row.get("metric_contract") != PAIRED_METRIC_CONTRACT:
            raise ValueError(
                f"Paired row {key} does not use {PAIRED_METRIC_CONTRACT!r}."
            )
        if str(row.get("split")) != "test":
            raise ValueError(f"Paired row {key} does not have split='test'.")
        for metric in ("rmse_km_per_s", "mae_km_per_s"):
            value = _finite_float(row.get(metric), f"{key} {metric}")
            if value < 0.0:
                raise ValueError(f"{key} {metric} cannot be negative.")


def aggregate_paired_test_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Aggregate paired rows by method and family and rank primary methods by mean RMSE."""
    validate_paired_metric_contract(rows)
    rows_by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    all_case_ids: set[str] = set()
    for row in rows:
        method_id = str(row["method_id"])
        rows_by_method[method_id].append(row)
        all_case_ids.add(str(row["case_id"]))
    method_summary: list[dict[str, Any]] = []
    for method_id in sorted(rows_by_method):
        method_rows = rows_by_method[method_id]
        method_case_ids = {str(row["case_id"]) for row in method_rows}
        if method_case_ids != all_case_ids:
            missing = sorted(all_case_ids - method_case_ids)
            extra = sorted(method_case_ids - all_case_ids)
            raise ValueError(
                f"Method {method_id} does not cover the exact paired case IDs; "
                f"missing={missing}, extra={extra}."
            )
        first = method_rows[0]
        rmse_values = [float(row["rmse_km_per_s"]) for row in method_rows]
        mae_values = [float(row["mae_km_per_s"]) for row in method_rows]
        rmse_stats = _statistics(rmse_values)
        mae_stats = _statistics(mae_values)
        method_summary.append(
            {
                "method_id": method_id,
                "method_label": first.get("method_label", method_id),
                "method_family": first.get("method_family", ""),
                "comparison_role": first.get("comparison_role", ""),
                "diagnostic_only": bool(first.get("diagnostic_only", False)),
                "is_primary": bool(first.get("is_primary", True)),
                "is_post_hoc_selection": bool(first.get("is_post_hoc_selection", False)),
                "case_count": len(method_rows),
                "rmse_mean_km_per_s": rmse_stats["mean"],
                "rmse_median_km_per_s": rmse_stats["median"],
                "rmse_std_km_per_s": rmse_stats["std"],
                "rmse_min_km_per_s": rmse_stats["min"],
                "rmse_max_km_per_s": rmse_stats["max"],
                "mae_mean_km_per_s": mae_stats["mean"],
                "mae_median_km_per_s": mae_stats["median"],
                "mae_std_km_per_s": mae_stats["std"],
                "mae_min_km_per_s": mae_stats["min"],
                "mae_max_km_per_s": mae_stats["max"],
                "rank": None,
            }
        )

    primary = [row for row in method_summary if row["is_primary"]]
    primary.sort(
        key=lambda row: (
            float(row["rmse_mean_km_per_s"]),
            float(row["mae_mean_km_per_s"]),
            str(row["method_id"]),
        )
    )
    for rank, row in enumerate(primary, start=1):
        row["rank"] = rank
    ranking = [
        {
            "rank": row["rank"],
            "method_id": row["method_id"],
            "method_label": row["method_label"],
            "method_family": row["method_family"],
            "comparison_role": row["comparison_role"],
            "rmse_mean_km_per_s": row["rmse_mean_km_per_s"],
            "mae_mean_km_per_s": row["mae_mean_km_per_s"],
            "case_count": row["case_count"],
        }
        for row in primary
    ]

    family_summary: list[dict[str, Any]] = []
    for method_id in sorted(rows_by_method):
        method_rows = rows_by_method[method_id]
        for family in sorted({str(row["family"]) for row in method_rows}):
            family_rows = [row for row in method_rows if str(row["family"]) == family]
            rmse_stats = _statistics([float(row["rmse_km_per_s"]) for row in family_rows])
            mae_stats = _statistics([float(row["mae_km_per_s"]) for row in family_rows])
            first = family_rows[0]
            family_summary.append(
                {
                    "method_id": method_id,
                    "method_label": first.get("method_label", method_id),
                    "family": family,
                    "case_count": len(family_rows),
                    "rmse_mean_km_per_s": rmse_stats["mean"],
                    "rmse_median_km_per_s": rmse_stats["median"],
                    "rmse_std_km_per_s": rmse_stats["std"],
                    "mae_mean_km_per_s": mae_stats["mean"],
                    "mae_median_km_per_s": mae_stats["median"],
                    "mae_std_km_per_s": mae_stats["std"],
                }
            )
    return {
        "method_summary": method_summary,
        "family_wise_summary": family_summary,
        "method_ranking": ranking,
    }


def _classical_rows(
    fixed_test_case_ids: Sequence[str],
    manifest_cases: Mapping[str, Mapping[str, Any]],
    rows_by_case: Mapping[str, Mapping[str, Any]],
    *,
    method_id: str,
    method_label: str,
    method_family: str,
    comparison_role: str,
    diagnostic_only: bool,
    is_primary: bool,
    source_artifact: Path,
    source_split: str,
    repo_root: Path,
    regularization_damping: float | None = None,
    regularization_smoothing: float | None = None,
) -> list[dict[str, Any]]:
    _require_case_ids(fixed_test_case_ids, rows_by_case, method_id)
    rows = []
    for case_id in fixed_test_case_ids:
        row = rows_by_case[case_id]
        rows.append(
            _paired_row(
                case_id=case_id,
                manifest_case=manifest_cases[case_id],
                method_id=method_id,
                method_label=method_label,
                method_family=method_family,
                comparison_role=comparison_role,
                diagnostic_only=diagnostic_only,
                is_primary=is_primary,
                is_post_hoc_selection=False,
                source_split=source_split,
                source_artifact=source_artifact,
                target_grid_path=Path(str(manifest_cases[case_id]["target_grid_path"])),
                rmse=row["all_cell_velocity_rmse_km_per_s"],
                mae=row["all_cell_velocity_mae_km_per_s"],
                repo_root=repo_root,
                regularization_damping=regularization_damping,
                regularization_smoothing=regularization_smoothing,
            )
        )
    return rows


def _reference_sensitivity_audit_rows(
    *,
    fixed_test_case_ids: Sequence[str],
    manifest_cases: Mapping[str, Mapping[str, Any]],
    phase7b_summary: Mapping[str, Any],
    phase7b_rows: Sequence[Mapping[str, Any]],
    source_artifact: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    selection = phase7b_summary.get("selection_audit", {})
    best = selection.get("best_covered_cell_rmse")
    if not isinstance(best, dict):
        raise ValueError("Phase 7B best covered-cell setting is unavailable for the requested audit row.")
    damping = _finite_float(best.get("damping"), "Phase 7B best damping")
    smoothing = _finite_float(best.get("smoothing"), "Phase 7B best smoothing")
    selected_rows = {
        str(row["case_id"]): row
        for row in phase7b_rows
        if math.isclose(float(row["damping"]), damping, rel_tol=1.0e-12, abs_tol=1.0e-12)
        and math.isclose(float(row["smoothing"]), smoothing, rel_tol=1.0e-12, abs_tol=1.0e-12)
    }
    _require_case_ids(fixed_test_case_ids, selected_rows, "reference_ray_phase7b_best_covered_audit")
    return [
        _paired_row(
            case_id=case_id,
            manifest_case=manifest_cases[case_id],
            method_id="reference_ray_phase7b_best_covered_audit",
            method_label="Reference-ray Phase 7B best-covered sensitivity audit",
            method_family="classical_reference_ray",
            comparison_role="regularization_sensitivity_audit",
            diagnostic_only=False,
            is_primary=False,
            is_post_hoc_selection=True,
            source_split="all_corpus_sensitivity_output",
            source_artifact=source_artifact,
            target_grid_path=Path(str(manifest_cases[case_id]["target_grid_path"])),
            rmse=selected_rows[case_id]["all_cell_velocity_rmse_km_per_s"],
            mae=selected_rows[case_id]["all_cell_velocity_mae_km_per_s"],
            repo_root=repo_root,
            regularization_damping=damping,
            regularization_smoothing=smoothing,
            selection_note=(
                "Post hoc Phase 7B all-corpus covered-cell selection; sensitivity/audit only, "
                "not the primary Phase 6 reference-ray setting."
            ),
        )
        for case_id in fixed_test_case_ids
    ]


def _paired_row(
    *,
    case_id: str,
    manifest_case: Mapping[str, Any],
    method_id: str,
    method_label: str,
    method_family: str,
    comparison_role: str,
    diagnostic_only: bool,
    is_primary: bool,
    is_post_hoc_selection: bool,
    source_split: str,
    source_artifact: Path,
    target_grid_path: Path,
    rmse: Any,
    mae: Any,
    repo_root: Path,
    prediction_path: Path | None = None,
    regularization_damping: float | None = None,
    regularization_smoothing: float | None = None,
    selection_note: str = "",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "family": str(manifest_case["family"]),
        "split": "test",
        "source_split": source_split,
        "method_id": method_id,
        "method_label": method_label,
        "method_family": method_family,
        "comparison_role": comparison_role,
        "diagnostic_only": diagnostic_only,
        "is_primary": is_primary,
        "is_post_hoc_selection": is_post_hoc_selection,
        "metric_contract": PAIRED_METRIC_CONTRACT,
        "rmse_km_per_s": _finite_float(rmse, f"{method_id}/{case_id} RMSE"),
        "mae_km_per_s": _finite_float(mae, f"{method_id}/{case_id} MAE"),
        "regularization_damping": regularization_damping,
        "regularization_smoothing": regularization_smoothing,
        "selection_note": selection_note,
        "source_artifact": _relative_to_repo(source_artifact, repo_root),
        "target_grid_path": _relative_to_repo(
            _resolve_path(target_grid_path, repo_root), repo_root
        ),
        "prediction_path": (
            _relative_to_repo(_resolve_path(prediction_path, repo_root), repo_root)
            if prediction_path is not None
            else ""
        ),
    }


def _phase6_reference_parameters(
    summary: Mapping[str, Any],
    reference_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, float]:
    selection_audit = summary.get("selection_audit", {})
    phase6_reference = (
        selection_audit.get("phase6_reference")
        if isinstance(selection_audit, Mapping)
        else None
    )
    if isinstance(phase6_reference, dict):
        return {
            "damping": _finite_float(
                phase6_reference.get("damping"), "Phase 6 reference damping"
            ),
                "smoothing": _finite_float(
                    phase6_reference.get("smoothing"), "Phase 6 reference smoothing"
                ),
        }
    documented_pairs = {
        (
            _finite_float(row["selected_damping"], "reference selected damping"),
            _finite_float(row["selected_smoothing"], "reference selected smoothing"),
        )
        for row in reference_rows
        if row.get("selected_damping") not in (None, "")
        and row.get("selected_smoothing") not in (None, "")
    }
    if len(documented_pairs) == 1:
        damping, smoothing = documented_pairs.pop()
        return {"damping": damping, "smoothing": smoothing}
    return {
        "damping": DEFAULT_PHASE6_REFERENCE_DAMPING,
        "smoothing": DEFAULT_PHASE6_REFERENCE_SMOOTHING,
    }


def _validate_reference_baseline_parameters(
    rows: Sequence[Mapping[str, Any]],
    phase6_reference: Mapping[str, float],
) -> None:
    for row in rows:
        if not row.get("case_id"):
            raise ValueError("Reference-ray case summary contains a row without case_id.")
        if row.get("selected_damping") not in (None, "") and not math.isclose(
            float(row["selected_damping"]),
            float(phase6_reference["damping"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"Reference-ray case {row['case_id']} is not using the documented Phase 6 damping."
            )
        if row.get("selected_smoothing") not in (None, "") and not math.isclose(
            float(row["selected_smoothing"]),
            float(phase6_reference["smoothing"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"Reference-ray case {row['case_id']} is not using the documented Phase 6 smoothing."
            )


def _load_manifest_cases(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Corpus manifest must contain a non-empty items list.")
    cases: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not item.get("case_id"):
            raise ValueError("Every corpus manifest item must contain case_id.")
        case_id = str(item["case_id"])
        if case_id in cases:
            raise ValueError(f"Corpus manifest contains duplicate case_id: {case_id}.")
        if not item.get("family") or not item.get("target_grid_path"):
            raise ValueError(f"Manifest case {case_id} lacks family or target_grid_path.")
        cases[case_id] = item
    return cases


def _validate_rows_against_manifest(
    rows: Iterable[Mapping[str, Any]],
    manifest_cases: Mapping[str, Mapping[str, Any]],
    source: str,
    expected_case_ids: Sequence[str] | None = None,
    repo_root: Path | None = None,
) -> None:
    """Check source family and target-grid provenance for the cases being compared."""
    root = repo_root or get_repo_root()
    expected = set(expected_case_ids) if expected_case_ids is not None else None
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if expected is not None and case_id not in expected:
            continue
        if case_id not in manifest_cases:
            raise ValueError(f"{source} contains case_id absent from the corpus manifest: {case_id}.")
        manifest_case = manifest_cases[case_id]
        row_family = row.get("family") or row.get("scenario")
        if row_family not in (None, "") and str(row_family) != str(manifest_case["family"]):
            raise ValueError(
                f"{source} family for {case_id} does not match the corpus manifest: "
                f"{row_family!r} != {manifest_case['family']!r}."
            )
        row_target_path = row.get("target_grid_path")
        if row_target_path not in (None, ""):
            manifest_target = _resolve_path(
                Path(str(manifest_case["target_grid_path"])), root
            ).resolve()
            source_target = _resolve_path(Path(str(row_target_path)), root).resolve()
            if source_target != manifest_target:
                raise ValueError(
                    f"{source} target grid for {case_id} does not match the corpus manifest."
                )


def _validate_manifest_case_ids(
    fixed_test_case_ids: Sequence[str],
    manifest_cases: Mapping[str, Mapping[str, Any]],
) -> None:
    missing = [case_id for case_id in fixed_test_case_ids if case_id not in manifest_cases]
    if missing:
        raise ValueError("Fixed-test case IDs are missing from the corpus manifest: " + ", ".join(missing))


def _group_variant_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        variant_id = str(row.get("variant_id", ""))
        case_id = str(row.get("case_id", ""))
        if not variant_id or not case_id:
            raise ValueError("Phase 7A cell-centered rows require variant_id and case_id.")
        if case_id in grouped[variant_id]:
            raise ValueError(f"Duplicate Phase 7A row for variant={variant_id}, case={case_id}.")
        grouped[variant_id][case_id] = row
    return dict(grouped)


def _rows_by_case(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id:
            raise ValueError(f"{method} contains a row without case_id.")
        if case_id in indexed:
            raise ValueError(f"{method} contains duplicate case_id: {case_id}.")
        indexed[case_id] = row
    return indexed


def _require_case_ids(
    expected_case_ids: Sequence[str],
    indexed_rows: Mapping[str, Any],
    method: str,
) -> None:
    expected = set(expected_case_ids)
    observed = set(indexed_rows)
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"{method} lacks per-case metrics for fixed test IDs: {', '.join(missing)}")


def _test_case_by_id(metrics: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    for case in metrics["split_metrics"]["test"]["case_metrics"]:
        if str(case["case_id"]) == case_id:
            return case
    raise KeyError(case_id)


def _method_metadata(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        method_id = str(row["method_id"])
        seen.setdefault(
            method_id,
            {
                "method_id": method_id,
                "method_label": row["method_label"],
                "method_family": row["method_family"],
                "comparison_role": row["comparison_role"],
                "diagnostic_only": bool(row["diagnostic_only"]),
                "is_primary": bool(row["is_primary"]),
                "is_post_hoc_selection": bool(row["is_post_hoc_selection"]),
                "regularization_damping": row.get("regularization_damping"),
                "regularization_smoothing": row.get("regularization_smoothing"),
            },
        )
    return [seen[key] for key in sorted(seen)]


def _sensitivity_metadata(
    summary: Mapping[str, Any],
    *,
    included: bool,
) -> dict[str, Any]:
    selection = summary.get("selection_audit", {})
    best = selection.get("best_covered_cell_rmse") if isinstance(selection, dict) else None
    if not isinstance(best, dict):
        return {
            "available": False,
            "included_as_paired_row": False,
            "selection_status": "unavailable",
        }
    return {
        "available": True,
        "included_as_paired_row": included,
        "selection_status": (
            "post_hoc_full_corpus_sensitivity_audit"
            if included
            else "available_but_not_included_in_primary_comparison"
        ),
        "damping": best.get("damping"),
        "smoothing": best.get("smoothing"),
        "covered_cell_rmse_full_corpus_km_per_s": best.get(
            "covered_cell_velocity_rmse_km_per_s_mean"
        ),
        "source": "Phase 7B bounded all-corpus regularization sweep",
    }


def _metric_from_row(row: Mapping[str, Any], metric: str) -> float:
    candidates = (f"{metric}_km_per_s", f"all_cell_velocity_{metric}_km_per_s", metric)
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            return _finite_float(row[key], key)
    raise ValueError(f"Metric row lacks {metric} value.")


def _statistics(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {number!r}.")
    return number


def _write_experiment_note(
    path: Path,
    *,
    summary: Mapping[str, Any],
    case_metrics_path: Path,
    summary_csv_path: Path,
    family_summary_path: Path,
    repo_root: Path,
) -> None:
    ranking = summary["method_ranking"]
    lines = [
        "# Experiment: Phase 8A Paired Same-Test-Set Comparison",
        "",
        f"Experiment ID: `{PHASE8_PAIRED_TEST_COMPARISON_ID}`",
        "",
        "## Objective",
        "",
        "Evaluate the selected ML model, classical diagnostics, and Phase 7A sanity baselines "
        "on exactly the same fixed test cases under one cell-centered RMSE/MAE contract.",
        "",
        "## Scope and protocol",
        "",
        f"- Paired test cases: `{summary['paired_test_case_count']}`; IDs are resolved from the selected ML `metrics.json` test case records.",
        "- ML predictions and true node-centered targets are converted to cell-centered values by eight-node cell averaging.",
        "- Classical outputs are already cell-centered; only the fixed test-case subset is retained.",
        "- Population standard deviation (`ddof=0`) is reported across per-case metrics.",
        "",
        "## Primary method ranking",
        "",
    ]
    for row in ranking:
        lines.append(
            f"{row['rank']}. `{row['method_id']}`: mean RMSE "
            f"`{float(row['rmse_mean_km_per_s']):.6f}` km/s, mean MAE "
            f"`{float(row['mae_mean_km_per_s']):.6f}` km/s."
        )
    lines.extend(
        [
            "",
            "## Regularization handling",
            "",
            "The primary reference-ray row retains the Phase 6 documented baseline "
            "(`lambda=0.01`, `alpha=1.0`). The Phase 7B best-covered setting is retained "
            "as a sensitivity/audit record only; if materialized as a row, it is explicitly "
            "marked post hoc and is excluded from the primary ranking.",
            "",
            "## Claim boundaries",
            "",
            "- Synthetic-only benchmark; no real-data validation is claimed.",
            "- Known-ray inversion is an optimistic diagnostic using true-model ray geometry.",
            "- Reference-ray is a fixed-ray regularized baseline, not full nonlinear tomography.",
            "- The ranking is limited to this fixed synthetic test set; no universal ML superiority claim is made.",
            "",
            "## Outputs",
            "",
            f"- Case metrics: `{_relative_to_repo(case_metrics_path, repo_root)}`",
            f"- Method summary: `{_relative_to_repo(summary_csv_path, repo_root)}`",
            f"- Family summary: `{_relative_to_repo(family_summary_path, repo_root)}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV artifact is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
