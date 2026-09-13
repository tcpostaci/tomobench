"""Fair, validation-selected reference-ray baseline evaluation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from tomobench.evaluation.metrics import mean_absolute_error, root_mean_squared_error
from tomobench.tomography.inversion import _load_velocity_grid, cell_centered_velocity_target
from tomobench.utils.paths import get_repo_root


FAIR_CLASSICAL_BASELINE_VERSION = "fair_reference_ray_baseline_v1"
DEFAULT_CLASSICAL_SENSITIVITY_CASE_METRICS = Path(
    "outputs/generated/classical_tomography/reference_ray_regularization_sensitivity_v1/"
    "regularization_sensitivity_case_metrics.csv"
)
DEFAULT_CLASSICAL_GROUPED_SPLIT = Path(
    "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/"
    "target_group_split_assignments.csv"
)
DEFAULT_CLASSICAL_ABLATION_METRICS = Path(
    "outputs/generated/ml_baselines/phase9_grouped_split_ablation_v1/"
    "cell_centered_metrics.csv"
)
DEFAULT_CLASSICAL_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/fair_reference_ray_baseline_v1"
)


@dataclass(frozen=True)
class FairClassicalBaselineOutputs:
    """Paths written by the fair classical-baseline selection protocol."""

    output_dir: Path
    parameter_grid_csv: Path
    case_metrics_csv: Path
    summary_json: Path
    summary_md: Path


def run_fair_reference_ray_baseline(
    *,
    sensitivity_case_metrics_path: Path = DEFAULT_CLASSICAL_SENSITIVITY_CASE_METRICS,
    grouped_split_path: Path = DEFAULT_CLASSICAL_GROUPED_SPLIT,
    ablation_metrics_path: Path = DEFAULT_CLASSICAL_ABLATION_METRICS,
    output_dir: Path = DEFAULT_CLASSICAL_OUTPUT_DIR,
    minimum_coverage_km: float = 1.0e-9,
) -> FairClassicalBaselineOutputs:
    """Select lambda/alpha on validation cases and evaluate once on fixed test cases.

    The sensitivity case metrics are precomputed forward/inversion candidates.
    This workflow uses only rows labelled ``validation`` by the independent
    target-grouped split to select one global parameter pair.  It then reports
    the frozen pair on train, validation, and test rows without reselecting on
    test RMSE.
    """
    if minimum_coverage_km < 0.0:
        raise ValueError("minimum_coverage_km must be non-negative.")
    repo_root = get_repo_root()
    sensitivity_path = _resolve_path(Path(sensitivity_case_metrics_path), repo_root)
    split_path = _resolve_path(Path(grouped_split_path), repo_root)
    ablation_path = _resolve_path(Path(ablation_metrics_path), repo_root)
    resolved_output_dir = _resolve_path(Path(output_dir), repo_root)
    _require_file(sensitivity_path, "reference-ray sensitivity case metrics")
    _require_file(split_path, "grouped split assignment")

    sensitivity_rows = _read_csv(sensitivity_path)
    split_by_case_id = _read_grouped_split(split_path)
    _validate_case_split_coverage(sensitivity_rows, split_by_case_id)
    parameter_grid = sorted(
        {(float(row["damping"]), float(row["smoothing"])) for row in sensitivity_rows}
    )
    if not parameter_grid:
        raise ValueError("Reference-ray sensitivity metrics contain no parameter pairs.")

    validation_rows = [
        row
        for row in sensitivity_rows
        if split_by_case_id[str(row["case_id"])] == "validation"
    ]
    if not validation_rows:
        raise ValueError("The grouped split contains no validation sensitivity rows.")
    validation_grid_rows = _aggregate_grid_rows(validation_rows, parameter_grid)
    selected_grid_row = min(
        validation_grid_rows,
        key=lambda row: (
            float(row["validation_covered_cell_rmse_km_per_s_mean"]),
            float(row["validation_all_cell_rmse_km_per_s_mean"]),
            float(row["validation_travel_time_rmse_s_mean"]),
            float(row["validation_clipped_velocity_cell_percentage_mean"]),
            float(row["damping"]),
            float(row["smoothing"]),
        ),
    )
    selected_damping = float(selected_grid_row["damping"])
    selected_smoothing = float(selected_grid_row["smoothing"])
    selected_rows = [
        row
        for row in sensitivity_rows
        if float(row["damping"]) == selected_damping
        and float(row["smoothing"]) == selected_smoothing
    ]
    prior_rows = _reference_prior_metrics(selected_rows, repo_root)
    ablation_rows = _read_ablation_rows(ablation_path, repo_root) if ablation_path.is_file() else {}
    case_rows = _selected_case_rows(
        selected_rows,
        prior_rows,
        split_by_case_id,
        ablation_rows,
    )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    parameter_grid_csv = resolved_output_dir / "reference_parameter_grid_validation_metrics.csv"
    case_metrics_csv = resolved_output_dir / "fair_reference_baseline_case_metrics.csv"
    summary_json = resolved_output_dir / "fair_reference_baseline_summary.json"
    summary_md = resolved_output_dir / "fair_reference_baseline_summary.md"
    _write_csv(parameter_grid_csv, validation_grid_rows)
    _write_csv(case_metrics_csv, case_rows)

    summary_payload = {
        "artifact_type": FAIR_CLASSICAL_BASELINE_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "sensitivity_case_metrics_path": _relative_or_absolute(sensitivity_path, repo_root),
        "grouped_split_path": _relative_or_absolute(split_path, repo_root),
        "ablation_metrics_path": _relative_or_absolute(ablation_path, repo_root),
        "selection_scope": "validation_only",
        "selection_metric_priority": [
            "validation_covered_cell_rmse_km_per_s_mean",
            "validation_all_cell_rmse_km_per_s_mean",
            "validation_travel_time_rmse_s_mean",
            "validation_clipped_velocity_cell_percentage_mean",
            "damping_ascending",
            "smoothing_ascending",
        ],
        "minimum_coverage_threshold_km": minimum_coverage_km,
        "parameter_grid": [
            {"damping": damping, "smoothing": smoothing}
            for damping, smoothing in parameter_grid
        ],
        "selected_damping": selected_damping,
        "selected_smoothing": selected_smoothing,
        "split_case_counts": _split_counts(split_by_case_id),
        "split_unique_target_counts": _split_unique_target_counts(split_path),
        "validation_selection_row": selected_grid_row,
        "selected_parameter_metrics": _selected_parameter_summary(case_rows),
        "comparison_scope": "fixed_test_cases",
        "split_comparisons": {
            split: {
                "reference_prior": _aggregate_method_rows(
                    [row for row in case_rows if row["split"] == split],
                    "reference_prior",
                ),
                "optimized_fixed_ray": _aggregate_method_rows(
                    [row for row in case_rows if row["split"] == split],
                    "optimized_fixed_ray",
                ),
                "pca_linear_ml": _aggregate_method_rows(
                    [row for row in case_rows if row["split"] == split],
                    "pca_linear_ml",
                ),
                "travel_time_only": _aggregate_method_rows(
                    [row for row in case_rows if row["split"] == split],
                    "travel_time_only",
                ),
            }
            for split in ("train", "validation", "test")
        },
        "reference_prior_comparison": _aggregate_method_rows(
            [row for row in case_rows if row["split"] == "test"], "reference_prior"
        ),
        "optimized_fixed_ray_comparison": _aggregate_method_rows(
            [row for row in case_rows if row["split"] == "test"], "optimized_fixed_ray"
        ),
        "ml_comparison": _aggregate_method_rows(
            [row for row in case_rows if row["split"] == "test"], "pca_linear_ml"
        ),
        "travel_time_only_comparison": _aggregate_method_rows(
            [row for row in case_rows if row["split"] == "test"], "travel_time_only"
        ),
        "paired_difference_definition": "method metric minus reference_prior metric; negative means improvement over the prior",
        "fairness_statement": "The selected lambda/alpha pair is frozen after validation selection and its test metrics are read once from the fixed test cases.",
        "limitations": [
            "The classical candidates come from the existing reference-ray sensitivity sidecars; this protocol does not retrace them.",
            "The reference-ray geometry remains a fixed configured layered-model diagnostic and is not a fully nonlinear inversion.",
            "Selection uses the validation covered-cell metric first; all-cell metrics, clipping, and prior performance are reported alongside it.",
        ],
    }
    _write_json(summary_payload, summary_json)
    _write_markdown_summary(summary_md, summary_payload, parameter_grid_csv, case_metrics_csv, repo_root)
    return FairClassicalBaselineOutputs(
        output_dir=resolved_output_dir,
        parameter_grid_csv=parameter_grid_csv,
        case_metrics_csv=case_metrics_csv,
        summary_json=summary_json,
        summary_md=summary_md,
    )


def _aggregate_grid_rows(
    validation_rows: list[dict[str, Any]],
    parameter_grid: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for damping, smoothing in parameter_grid:
        selected = [
            row
            for row in validation_rows
            if float(row["damping"]) == damping and float(row["smoothing"]) == smoothing
        ]
        if not selected:
            raise ValueError(f"Missing validation rows for damping={damping}, smoothing={smoothing}.")
        rows.append(
            {
                "damping": damping,
                "smoothing": smoothing,
                "validation_case_count": len(selected),
                "validation_covered_cell_rmse_km_per_s_mean": _mean(
                    selected,
                    "covered_cell_velocity_rmse_km_per_s",
                ),
                "validation_covered_cell_mae_km_per_s_mean": _mean(
                    selected,
                    "covered_cell_velocity_mae_km_per_s",
                ),
                "validation_all_cell_rmse_km_per_s_mean": _mean(
                    selected,
                    "all_cell_velocity_rmse_km_per_s",
                ),
                "validation_all_cell_mae_km_per_s_mean": _mean(
                    selected,
                    "all_cell_velocity_mae_km_per_s",
                ),
                "validation_travel_time_rmse_s_mean": _mean(selected, "travel_time_rmse_s"),
                "validation_clipped_velocity_cell_percentage_mean": _mean(
                    selected,
                    "clipped_velocity_cell_percentage",
                ),
                "validation_covered_clipped_cell_count_mean": _mean(
                    selected,
                    "covered_clipped_cell_count",
                ),
                "validation_uncovered_clipped_cell_count_mean": _mean(
                    selected,
                    "uncovered_clipped_cell_count",
                ),
                "validation_nonphysical_slowness_count_mean": _mean(
                    selected,
                    "nonphysical_slowness_count",
                ),
            }
        )
    return rows


def _selected_case_rows(
    selected_rows: list[dict[str, Any]],
    prior_rows: dict[str, dict[str, float | int | None]],
    split_by_case_id: dict[str, str],
    ablation_rows: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in sorted(selected_rows, key=lambda item: str(item["case_id"])):
        case_id = str(row["case_id"])
        prior = prior_rows[case_id]
        result = {
            "case_id": case_id,
            "family": str(row["family"]),
            "split": split_by_case_id[case_id],
            "damping": float(row["damping"]),
            "smoothing": float(row["smoothing"]),
            "cell_count": int(row["cell_count"]),
            "covered_cell_count": int(row["covered_cell_count"]),
            "coverage_fraction": float(row["coverage_fraction"]),
            "travel_time_rmse_s": float(row["travel_time_rmse_s"]),
            "travel_time_mae_s": float(row["travel_time_mae_s"]),
            "optimized_fixed_ray_all_cell_rmse_km_per_s": float(
                row["all_cell_velocity_rmse_km_per_s"]
            ),
            "optimized_fixed_ray_all_cell_mae_km_per_s": float(
                row["all_cell_velocity_mae_km_per_s"]
            ),
            "optimized_fixed_ray_covered_cell_rmse_km_per_s": float(
                row["covered_cell_velocity_rmse_km_per_s"]
            ),
            "optimized_fixed_ray_covered_cell_mae_km_per_s": float(
                row["covered_cell_velocity_mae_km_per_s"]
            ),
            "clipped_velocity_cell_count": int(row["clipped_velocity_cell_count"]),
            "clipped_velocity_cell_percentage": float(row["clipped_velocity_cell_percentage"]),
            "covered_clipped_cell_count": int(row["covered_clipped_cell_count"]),
            "uncovered_clipped_cell_count": int(row["uncovered_clipped_cell_count"]),
            "nonphysical_slowness_count": int(row["nonphysical_slowness_count"]),
            "reference_prior_all_cell_rmse_km_per_s": prior["all_cell_rmse"],
            "reference_prior_all_cell_mae_km_per_s": prior["all_cell_mae"],
            "reference_prior_covered_cell_rmse_km_per_s": prior["covered_cell_rmse"],
            "reference_prior_covered_cell_mae_km_per_s": prior["covered_cell_mae"],
            "optimized_minus_prior_all_cell_rmse_km_per_s": _difference(
                row["all_cell_velocity_rmse_km_per_s"], prior["all_cell_rmse"]
            ),
            "optimized_minus_prior_covered_cell_rmse_km_per_s": _difference(
                row["covered_cell_velocity_rmse_km_per_s"], prior["covered_cell_rmse"]
            ),
        }
        ml = ablation_rows.get((case_id, "full_input"))
        travel_time_only = ablation_rows.get((case_id, "travel_time_only"))
        result.update(
            {
                "pca_linear_ml_all_cell_rmse_km_per_s": _optional_float(
                    ml.get("all_cell_velocity_rmse_km_per_s") if ml else None
                ),
                "pca_linear_ml_all_cell_mae_km_per_s": _optional_float(
                    ml.get("all_cell_velocity_mae_km_per_s") if ml else None
                ),
                "travel_time_only_all_cell_rmse_km_per_s": _optional_float(
                    travel_time_only.get("all_cell_velocity_rmse_km_per_s")
                    if travel_time_only
                    else None
                ),
                "travel_time_only_all_cell_mae_km_per_s": _optional_float(
                    travel_time_only.get("all_cell_velocity_mae_km_per_s")
                    if travel_time_only
                    else None
                ),
            }
        )
        result["ml_minus_prior_all_cell_rmse_km_per_s"] = _difference(
            result["pca_linear_ml_all_cell_rmse_km_per_s"],
            prior["all_cell_rmse"],
        )
        result["travel_time_only_minus_prior_all_cell_rmse_km_per_s"] = _difference(
            result["travel_time_only_all_cell_rmse_km_per_s"],
            prior["all_cell_rmse"],
        )
        output.append(result)
    return output


def _reference_prior_metrics(
    selected_rows: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for row in selected_rows:
        case_id = str(row["case_id"])
        summary_path = _resolve_path(Path(str(row["phase6_summary_json"])), repo_root)
        summary = _read_json(summary_path)
        source_files = summary.get("source_files")
        if not isinstance(source_files, dict):
            raise ValueError(f"Missing source_files in Phase 6 summary for {case_id}.")
        target_grid = _load_velocity_grid(
            _resolve_path(Path(str(source_files["true_target_velocity_grid"])), repo_root)
        )
        reference_grid = _load_velocity_grid(
            _resolve_path(Path(str(source_files["reference_velocity_grid"])), repo_root)
        )
        target = cell_centered_velocity_target(target_grid)
        reference = cell_centered_velocity_target(reference_grid)
        coverage_path = summary_path.parent / "reference_ray_cell_coverage.csv"
        covered_mask = _read_coverage_mask(coverage_path)
        covered_target = [value for value, covered in zip(target, covered_mask, strict=True) if covered]
        covered_reference = [
            value for value, covered in zip(reference, covered_mask, strict=True) if covered
        ]
        output[case_id] = {
            "all_cell_rmse": root_mean_squared_error(target, reference),
            "all_cell_mae": mean_absolute_error(target, reference),
            "covered_cell_rmse": root_mean_squared_error(covered_target, covered_reference)
            if covered_target
            else None,
            "covered_cell_mae": mean_absolute_error(covered_target, covered_reference)
            if covered_target
            else None,
        }
    return output


def _read_ablation_rows(path: Path, repo_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    del repo_root
    rows = _read_csv(path)
    return {
        (str(row["case_id"]), str(row.get("variant_id", ""))): row
        for row in rows
        if row.get("variant_id") in {"full_input", "travel_time_only"}
        and str(row.get("split", "")) == "test"
    }


def _read_grouped_split(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    split_by_case_id: dict[str, str] = {}
    target_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        case_id = str(row["case_id"])
        split = str(row["split"])
        target_hash = str(row["target_hash"])
        split_by_case_id[case_id] = split
        target_splits[target_hash].add(split)
    crossing = sorted(target_hash for target_hash, splits in target_splits.items() if len(splits) > 1)
    if crossing:
        raise ValueError("Grouped split contains target groups crossing splits: " + ", ".join(crossing))
    return split_by_case_id


def _validate_case_split_coverage(
    sensitivity_rows: list[dict[str, Any]],
    split_by_case_id: dict[str, str],
) -> None:
    sensitivity_cases = {str(row["case_id"]) for row in sensitivity_rows}
    split_cases = set(split_by_case_id)
    if sensitivity_cases != split_cases:
        raise ValueError(
            "Grouped split and sensitivity metrics have different case IDs. "
            f"Only in sensitivity: {sorted(sensitivity_cases - split_cases)}; "
            f"only in split: {sorted(split_cases - sensitivity_cases)}"
        )


def _read_coverage_mask(path: Path) -> list[bool]:
    _require_file(path, "reference-ray coverage")
    rows = _read_csv(path)
    return [str(row["covered_by_threshold"]).strip().lower() == "true" for row in rows]


def _selected_parameter_summary(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in case_rows if row["split"] == split]
        output[split] = _aggregate_method_rows(rows, "optimized_fixed_ray")
    return output


def _aggregate_method_rows(case_rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    if not case_rows:
        return {"case_count": 0}
    prefix = {
        "reference_prior": "reference_prior",
        "optimized_fixed_ray": "optimized_fixed_ray",
        "pca_linear_ml": "pca_linear_ml",
        "travel_time_only": "travel_time_only",
    }[method]
    rmse_key = f"{prefix}_all_cell_rmse_km_per_s"
    mae_key = f"{prefix}_all_cell_mae_km_per_s"
    covered_rmse_key = f"{prefix}_covered_cell_rmse_km_per_s"
    covered_mae_key = f"{prefix}_covered_cell_mae_km_per_s"
    rmse_values = [
        float(row[rmse_key]) for row in case_rows if row.get(rmse_key) not in (None, "")
    ]
    mae_values = [
        float(row[mae_key]) for row in case_rows if row.get(mae_key) not in (None, "")
    ]
    covered_rmse_values = [
        float(row[covered_rmse_key])
        for row in case_rows
        if row.get(covered_rmse_key) not in (None, "")
    ]
    covered_mae_values = [
        float(row[covered_mae_key])
        for row in case_rows
        if row.get(covered_mae_key) not in (None, "")
    ]
    coverage_values = [
        float(row["coverage_fraction"])
        for row in case_rows
        if row.get("coverage_fraction") not in (None, "")
    ]
    clipping_values = [
        float(row["clipped_velocity_cell_percentage"])
        for row in case_rows
        if row.get("clipped_velocity_cell_percentage") not in (None, "")
    ]
    summary = {
        "case_count": len(rmse_values),
        "mean_all_cell_rmse_km_per_s": sum(rmse_values) / len(rmse_values)
        if rmse_values
        else None,
        "mean_all_cell_mae_km_per_s": sum(mae_values) / len(mae_values) if mae_values else None,
        "covered_case_count": len(covered_rmse_values),
        "mean_covered_cell_rmse_km_per_s": (
            sum(covered_rmse_values) / len(covered_rmse_values)
            if covered_rmse_values
            else None
        ),
        "mean_covered_cell_mae_km_per_s": (
            sum(covered_mae_values) / len(covered_mae_values)
            if covered_mae_values
            else None
        ),
        "mean_coverage_fraction": (
            sum(coverage_values) / len(coverage_values) if coverage_values else None
        ),
        "mean_clipped_velocity_cell_percentage": (
            sum(clipping_values) / len(clipping_values) if clipping_values else None
        ),
    }
    return summary


def _split_counts(split_by_case_id: dict[str, str]) -> dict[str, int]:
    return {
        split: sum(value == split for value in split_by_case_id.values())
        for split in ("train", "validation", "test")
    }


def _split_unique_target_counts(path: Path) -> dict[str, int]:
    rows = _read_csv(path)
    return {
        split: len({str(row["target_hash"]) for row in rows if str(row["split"]) == split})
        for split in ("train", "validation", "test")
    }


def _difference(left: Any, right: Any) -> float | None:
    if left in (None, "") or right in (None, ""):
        return None
    return float(left) - float(right)


def _optional_float(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown_summary(
    path: Path,
    payload: dict[str, Any],
    parameter_grid_csv: Path,
    case_metrics_csv: Path,
    repo_root: Path,
) -> None:
    lines = [
        "# Fair Reference-Ray Baseline Selection",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Protocol",
        "",
        "A single damping/smoothing pair is selected using validation covered-cell RMSE, with deterministic tie-breaks. The selected pair is then frozen and evaluated on the fixed test cases.",
        "",
        f"- Selected damping: `{payload['selected_damping']}`",
        f"- Selected smoothing: `{payload['selected_smoothing']}`",
        f"- Split case counts: `{payload['split_case_counts']}`",
        f"- Split unique-target counts: `{payload['split_unique_target_counts']}`",
        "",
        "## Frozen Test Comparison",
        "",
    ]
    for method_key, label in (
        ("reference_prior", "Reference prior"),
        ("optimized_fixed_ray", "Optimized fixed-ray"),
        ("pca_linear_ml", "PCA-linear ML"),
        ("travel_time_only", "Travel-time-only"),
    ):
        comparison_key = {
            "reference_prior": "reference_prior_comparison",
            "optimized_fixed_ray": "optimized_fixed_ray_comparison",
            "pca_linear_ml": "ml_comparison",
            "travel_time_only": "travel_time_only_comparison",
        }[method_key]
        summary = payload[comparison_key]
        lines.append(
            f"- {label}: all-cell RMSE `{summary.get('mean_all_cell_rmse_km_per_s')}` km/s over `{summary.get('case_count')}` cases."
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Validation parameter grid: `{_relative_or_absolute(parameter_grid_csv, repo_root)}`",
            f"- Frozen case metrics: `{_relative_or_absolute(case_metrics_csv, repo_root)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


__all__ = ["FairClassicalBaselineOutputs", "run_fair_reference_ray_baseline"]
