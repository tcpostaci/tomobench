"""Audit generated Phase 4 paper pseudo-bending ML results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tomobench.utils.paths import get_repo_root


@dataclass(frozen=True)
class PaperMLAuditOutputs:
    """Artifacts written by the Phase 4.5 ML audit workflow."""

    audit_summary_json: Path
    audit_note: Path


def audit_paper_pseudo_bending_ml_suite(
    suite_dir: Path | None = None,
    audit_note_path: Path | None = None,
) -> PaperMLAuditOutputs:
    """Audit the generated Phase 4 ML suite and write traceable summaries."""
    repo_root = get_repo_root()
    resolved_suite_dir = _resolve_repo_path(
        suite_dir or Path("outputs/generated/ml_baselines/paper_pseudo_bending_ml_suite_v1"),
        repo_root,
    )
    suite_summary_path = resolved_suite_dir / "suite_summary.json"
    repeated_summary_path = resolved_suite_dir / "repeated_split_summary.json"
    loo_summary_path = resolved_suite_dir / "leave_one_family_out_summary.json"
    noise_summary_path = resolved_suite_dir / "noise_robustness_summary.json"
    audit_summary_path = resolved_suite_dir / "audit_summary.json"
    resolved_audit_note_path = _resolve_repo_path(
        audit_note_path or Path("wiki/experiments/paper_pseudo_bending_ml_suite_v1_audit.md"),
        repo_root,
    )

    suite_summary = _read_json(suite_summary_path)
    repeated_summary = _read_json(repeated_summary_path)
    loo_summary = _read_json(loo_summary_path)
    noise_summary = _read_json(noise_summary_path)
    manifest_path = _resolve_repo_path(Path(str(suite_summary["input_manifest_path"])), repo_root)
    manifest = _read_json(manifest_path)

    best_model_name = str(suite_summary["best_model"]["model_name"])
    best_metrics_path = _resolve_repo_path(
        Path(str(suite_summary["best_model"]["metrics_json"])),
        repo_root,
    )
    best_metrics = _read_json(best_metrics_path)
    fixed_ranking = _fixed_split_ranking(suite_summary)
    repeated_record = _model_record(repeated_summary, best_model_name)
    loo_record = _model_record(loo_summary, best_model_name)
    noise_record = _model_record(noise_summary, best_model_name)
    fixed_family_weaknesses = _rank_family_metrics(
        best_metrics["split_metrics"]["test"]["per_family_metrics"],
    )
    loo_family_weaknesses = _rank_loo_records(loo_record["held_out_records"])
    worst_cases = _rank_case_metrics(best_metrics)
    noise_trend = _noise_trend(noise_record["noise_level_records"])

    fixed_validation_rmse = float(suite_summary["best_model"]["fixed_validation_rmse"])
    fixed_test_rmse = float(suite_summary["best_model"]["fixed_test_rmse"])
    repeated_validation_rmse = float(repeated_record["aggregate_metrics"]["mean_validation_rmse"])
    repeated_to_fixed_ratio = repeated_validation_rmse / fixed_validation_rmse
    worst_loo = loo_family_weaknesses[0]
    loo_to_fixed_test_ratio = float(worst_loo["mean_rmse"]) / fixed_test_rmse
    rankings_consistent = _top_model(repeated_summary, "mean_validation_rmse") == best_model_name
    noise_degrades_smoothly = bool(noise_trend["is_monotonic_non_decreasing"])

    key_risks: list[str] = []
    if repeated_to_fixed_ratio > 1.25:
        key_risks.append(
            "Repeated-split mean validation RMSE is materially higher than the fixed validation RMSE."
        )
    if loo_to_fixed_test_ratio > 3.0 or float(worst_loo["mean_rmse"]) > 0.15:
        key_risks.append(
            f"Leave-one-family-out is weak for {worst_loo['held_out_family']} "
            f"(RMSE {float(worst_loo['mean_rmse']):.6f} km/s)."
        )
    if not rankings_consistent:
        key_risks.append("The fixed-split winner is not the repeated-split validation winner.")
    if not noise_degrades_smoothly:
        key_risks.append("Noise robustness does not degrade smoothly for the best model.")

    proceed_to_phase_5 = not key_risks
    recommended_next_phase = "Phase 5" if proceed_to_phase_5 else "Phase 4B"
    audit_payload = {
        "artifact_type": "paper_pseudo_bending_ml_suite_audit",
        "audit_status": "completed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "suite_id": suite_summary["experiment_id"],
        "input_manifest_path": _relative_to_repo_or_absolute(manifest_path, repo_root),
        "input_corpus_id": manifest.get("corpus_id", suite_summary.get("input_corpus_id")),
        "simulator": manifest.get("simulator_name", suite_summary.get("simulator")),
        "sample_count": suite_summary["sample_count"],
        "families": suite_summary["families"],
        "best_model": suite_summary["best_model"],
        "fixed_split_ranking": fixed_ranking,
        "stability_indicators": {
            "fixed_validation_rmse": fixed_validation_rmse,
            "fixed_test_rmse": fixed_test_rmse,
            "repeated_mean_validation_rmse": repeated_validation_rmse,
            "repeated_std_validation_rmse": repeated_record["aggregate_metrics"][
                "std_validation_rmse"
            ],
            "repeated_to_fixed_validation_rmse_ratio": repeated_to_fixed_ratio,
            "repeated_top_model": _top_model(repeated_summary, "mean_validation_rmse"),
            "fixed_and_repeated_winner_match": rankings_consistent,
        },
        "leave_one_family_out": {
            "worst_family": worst_loo,
            "loo_to_fixed_test_rmse_ratio": loo_to_fixed_test_ratio,
            "held_out_family_ranking": loo_family_weaknesses,
        },
        "noise_robustness": noise_trend,
        "per_family_weaknesses": {
            "fixed_test_family_ranking": fixed_family_weaknesses,
            "loo_family_ranking": loo_family_weaknesses,
        },
        "worst_cases": worst_cases,
        "linear_model_assessment": _linear_model_assessment(
            rankings_consistent=rankings_consistent,
            repeated_to_fixed_ratio=repeated_to_fixed_ratio,
            loo_to_fixed_test_ratio=loo_to_fixed_test_ratio,
        ),
        "key_risks": key_risks,
        "proceed_to_phase_5": proceed_to_phase_5,
        "recommended_next_phase": recommended_next_phase,
        "recommendation": _recommendation_text(
            proceed_to_phase_5=proceed_to_phase_5,
            worst_family=str(worst_loo["held_out_family"]),
        ),
    }
    _save_json(audit_payload, audit_summary_path)
    _write_audit_note(resolved_audit_note_path, audit_payload, repo_root)
    return PaperMLAuditOutputs(
        audit_summary_json=audit_summary_path,
        audit_note=resolved_audit_note_path,
    )


def _fixed_split_ranking(suite_summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "rank": record["rank"],
            "model_name": record["model_name"],
            "fixed_validation_rmse": record["fixed_validation_rmse"],
            "fixed_test_rmse": record["fixed_test_rmse"],
            "fixed_validation_mae": record["fixed_validation_mae"],
            "fixed_test_mae": record["fixed_test_mae"],
        }
        for record in suite_summary["model_ranking"]
    ]


def _rank_family_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "family": family,
                "case_count": values["case_count"],
                "mean_mae": values["mean_mae"],
                "mean_rmse": values["mean_rmse"],
            }
            for family, values in metrics.items()
        ),
        key=lambda item: float(item["mean_rmse"]),
        reverse=True,
    )


def _rank_loo_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "held_out_family": record["held_out_family"],
                "case_count": record["held_out"]["case_count"],
                "mean_mae": record["held_out"]["mean_mae"],
                "mean_rmse": record["held_out"]["mean_rmse"],
            }
            for record in records
        ),
        key=lambda item: float(item["mean_rmse"]),
        reverse=True,
    )


def _rank_case_metrics(best_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for split_name, split_metrics in best_metrics["split_metrics"].items():
        for case in split_metrics["case_metrics"]:
            cases.append(
                {
                    "split": split_name,
                    "case_id": case["case_id"],
                    "scenario": case["scenario"],
                    "mae": case["mae"],
                    "rmse": case["rmse"],
                    "prediction_path": case.get("prediction_path"),
                }
            )
    return sorted(cases, key=lambda item: float(item["rmse"]), reverse=True)[:10]


def _noise_trend(records: list[dict[str, Any]]) -> dict[str, Any]:
    points = [
        {
            "noise_std_s": record["noise_std_s"],
            "validation_rmse": record["validation"]["mean_rmse"],
            "test_rmse": record["test"]["mean_rmse"],
        }
        for record in records
    ]
    test_values = [float(point["test_rmse"]) for point in points]
    return {
        "points": points,
        "is_monotonic_non_decreasing": all(
            right >= left - 1.0e-12 for left, right in zip(test_values, test_values[1:])
        ),
        "test_rmse_delta_0_to_max_noise": test_values[-1] - test_values[0],
    }


def _linear_model_assessment(
    *,
    rankings_consistent: bool,
    repeated_to_fixed_ratio: float,
    loo_to_fixed_test_ratio: float,
) -> str:
    if rankings_consistent and repeated_to_fixed_ratio < 1.35:
        assessment = (
            "The linear winner is plausible for this synthetic PCA-target setup because linear and "
            "ridge models are close and remain strongest across repeated splits."
        )
    else:
        assessment = (
            "The linear winner may be split-sensitive because repeated-split behavior does not "
            "fully agree with the fixed-split result."
        )
    if loo_to_fixed_test_ratio > 3.0:
        assessment += (
            " However, leave-one-family-out performance shows that the result should be treated "
            "as interpolation within represented families, not robust family extrapolation."
        )
    return assessment


def _recommendation_text(*, proceed_to_phase_5: bool, worst_family: str) -> str:
    if proceed_to_phase_5:
        return "Proceed to Phase 5 with conservative reporting based on generated metrics."
    return (
        "Perform Phase 4B before Phase 5, focused on the weak leave-one-family-out behavior "
        f"for {worst_family} and on explaining family-specific failure modes."
    )


def _top_model(summary: dict[str, Any], metric_name: str) -> str:
    records = sorted(
        summary["model_records"],
        key=lambda record: float(record["aggregate_metrics"][metric_name]),
    )
    return str(records[0]["model_name"])


def _model_record(summary: dict[str, Any], model_name: str) -> dict[str, Any]:
    for record in summary["model_records"]:
        if record["model_name"] == model_name:
            return dict(record)
    raise ValueError(f"Model record not found: {model_name}")


def _write_audit_note(path: Path, payload: dict[str, Any], repo_root: Path) -> Path:
    best_model = payload["best_model"]
    lines = [
        "# Experiment: Paper Pseudo-Bending ML Suite v1 Audit",
        "",
        f"Suite ID: `{payload['suite_id']}`",
        "",
        "## Audit Recommendation",
        "",
        f"- Proceed to Phase 5: `{str(payload['proceed_to_phase_5']).lower()}`",
        f"- Recommended next phase: `{payload['recommended_next_phase']}`",
        f"- Recommendation: {payload['recommendation']}",
        "",
        "## Evidence Source",
        "",
        f"- Input manifest: `{payload['input_manifest_path']}`",
        f"- Input corpus ID: `{payload['input_corpus_id']}`",
        f"- Simulator: `{payload['simulator']}`",
        f"- Sample count: `{payload['sample_count']}`",
        f"- Families: `{', '.join(payload['families'])}`",
        "",
        "## Fixed-Split Model Ranking",
        "",
    ]
    for record in payload["fixed_split_ranking"]:
        lines.append(
            f"- Rank {record['rank']}: `{record['model_name']}` "
            f"(validation RMSE `{record['fixed_validation_rmse']:.6f}`, "
            f"test RMSE `{record['fixed_test_rmse']:.6f}` km/s)"
        )
    lines.extend(
        [
            "",
            "## Repeated-Split Stability",
            "",
            f"- Best fixed model: `{best_model['model_name']}`",
            f"- Fixed validation RMSE: `{payload['stability_indicators']['fixed_validation_rmse']:.6f}` km/s",
            f"- Repeated mean validation RMSE: `{payload['stability_indicators']['repeated_mean_validation_rmse']:.6f}` km/s",
            f"- Repeated validation RMSE std: `{payload['stability_indicators']['repeated_std_validation_rmse']:.6f}` km/s",
            f"- Repeated/fixed validation RMSE ratio: `{payload['stability_indicators']['repeated_to_fixed_validation_rmse_ratio']:.3f}`",
            f"- Fixed and repeated winner match: `{str(payload['stability_indicators']['fixed_and_repeated_winner_match']).lower()}`",
            "",
            "## Leave-One-Family-Out",
            "",
        ]
    )
    for record in payload["leave_one_family_out"]["held_out_family_ranking"]:
        lines.append(
            f"- `{record['held_out_family']}`: RMSE `{record['mean_rmse']:.6f}` km/s, "
            f"MAE `{record['mean_mae']:.6f}` km/s"
        )
    lines.extend(
        [
            "",
            "## Noise Robustness",
            "",
            f"- Smooth non-decreasing test RMSE trend: `{str(payload['noise_robustness']['is_monotonic_non_decreasing']).lower()}`",
            f"- Test RMSE change from no noise to max configured noise: `{payload['noise_robustness']['test_rmse_delta_0_to_max_noise']:.6f}` km/s",
        ]
    )
    for point in payload["noise_robustness"]["points"]:
        lines.append(
            f"- Noise `{point['noise_std_s']}` s: validation RMSE `{point['validation_rmse']:.6f}`, "
            f"test RMSE `{point['test_rmse']:.6f}` km/s"
        )
    lines.extend(
        [
            "",
            "## Per-Family Weaknesses",
            "",
        ]
    )
    for record in payload["per_family_weaknesses"]["fixed_test_family_ranking"]:
        lines.append(
            f"- Fixed test `{record['family']}`: RMSE `{record['mean_rmse']:.6f}` km/s, "
            f"MAE `{record['mean_mae']:.6f}` km/s"
        )
    lines.extend(["", "## Worst Cases", ""])
    for case in payload["worst_cases"][:10]:
        lines.append(
            f"- `{case['case_id']}` ({case['scenario']}, {case['split']}): "
            f"RMSE `{case['rmse']:.6f}` km/s, MAE `{case['mae']:.6f}` km/s"
        )
    lines.extend(
        [
            "",
            "## Linear Model Assessment",
            "",
            payload["linear_model_assessment"],
            "",
            "## Key Risks",
            "",
        ]
    )
    lines.extend(f"- {risk}" for risk in payload["key_risks"])
    lines.extend(
        [
            "",
            "## Machine-Readable Summary",
            "",
            "- Audit summary JSON: "
            "`outputs/generated/ml_baselines/paper_pseudo_bending_ml_suite_v1/audit_summary.json`",
            "",
            "## Limitation",
            "",
            "This audit uses only generated repository artifacts. It does not add new scientific "
            "claims, external validation, or real-data evidence.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return dict(payload)


def _save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_to_repo_or_absolute(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()
