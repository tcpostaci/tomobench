"""Package Phase 6 and Phase 7 evidence for paper drafting."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.evaluation.paper_candidate_v2 import (
    audit_phase6_metric_contract,
    load_phase6_packaging_evidence,
)
from tomobench.utils.paths import get_repo_root

PAPER_CANDIDATE_V3_ID = "paper_candidate_v3"
PHASE7A_ID = "phase7_observation_signal_ablation_v1"
PHASE7B_ID = "reference_ray_regularization_sensitivity_v1"
PHASE7C_ID = "phase7_learning_curve_v1"


@dataclass(frozen=True)
class PaperCandidateV3PackageOutputs:
    """Paths produced by the final Phase 6/7 evidence package."""

    output_dir: Path
    artifact_index_json: Path
    summary_markdown: Path
    experiment_note: Path
    table_paths: tuple[Path, ...]
    figure_references: tuple[Path, ...]


def package_paper_candidate_v3(
    settings: BenchmarkSettings | None = None,
    *,
    output_dir: Path | None = None,
) -> PaperCandidateV3PackageOutputs:
    """Create the Phase 6/7 evidence package without rerunning experiments."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    package_dir = _resolve_path(
        output_dir
        or (repo_root / loaded_settings.outputs.generated_base_dir / PAPER_CANDIDATE_V3_ID),
        repo_root,
    )
    tables_dir = package_dir / "tables"
    notes_dir = package_dir / "notes"
    tables_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    evidence = load_phase7_packaging_evidence(repo_root)
    audit_phase6_metric_contract(evidence["phase6"])
    audit_phase7_packaging_evidence(evidence)
    table_paths = _write_tables(evidence, tables_dir)
    summary_path = notes_dir / "package_summary.md"
    _write_package_summary(evidence, summary_path, table_paths, repo_root)
    experiment_note = repo_root / "wiki" / "experiments" / f"{PAPER_CANDIDATE_V3_ID}.md"
    _write_experiment_note(evidence, experiment_note, table_paths, repo_root)
    artifact_index = _build_artifact_index(
        evidence,
        table_paths,
        summary_path,
        experiment_note,
        repo_root,
    )
    artifact_index_path = package_dir / "artifact_index.json"
    _write_json(artifact_index_path, artifact_index)
    return PaperCandidateV3PackageOutputs(
        output_dir=package_dir,
        artifact_index_json=artifact_index_path,
        summary_markdown=summary_path,
        experiment_note=experiment_note,
        table_paths=tuple(table_paths),
        figure_references=tuple(evidence["v2_figure_paths"]),
    )


def load_phase7_packaging_evidence(repo_root: Path | None = None) -> dict[str, Any]:
    """Load existing Phase 6/7 outputs and validate package inputs are present."""
    root = repo_root or get_repo_root()
    phase7a_dir = root / "outputs" / "generated" / "ml_baselines" / PHASE7A_ID
    phase7b_dir = (
        root
        / "outputs"
        / "generated"
        / "classical_tomography"
        / "reference_ray_regularization_sensitivity_v1"
    )
    phase7c_dir = root / "outputs" / "generated" / "ml_baselines" / PHASE7C_ID
    v2_dir = root / "outputs" / "generated" / "paper_candidate_v2"
    paths = {
        "phase7a_summary": phase7a_dir / "ablation_summary.json",
        "phase7a_cell_centered_metrics": phase7a_dir / "cell_centered_metrics.csv",
        "phase7a_leave_one_family_out": phase7a_dir / "leave_one_family_out_summary.json",
        "phase7b_summary": phase7b_dir / "regularization_sensitivity_summary.json",
        "phase7b_summary_csv": phase7b_dir / "regularization_sensitivity_summary.csv",
        "phase7b_case_metrics": phase7b_dir / "regularization_sensitivity_case_metrics.csv",
        "phase7b_family_wise": phase7b_dir / "family_wise_sensitivity.csv",
        "phase7b_tradeoff": phase7b_dir / "rmse_clipping_tradeoff.csv",
        "phase7c_summary": phase7c_dir / "learning_curve_summary.json",
        "phase7c_summary_csv": phase7c_dir / "learning_curve_summary.csv",
        "phase7c_repetition_metrics": phase7c_dir / "learning_curve_repetition_metrics.csv",
        "phase7c_family_wise": phase7c_dir / "family_wise_learning_curve.csv",
        "paper_candidate_v2_index": v2_dir / "artifact_index.json",
        "paper_candidate_v2_summary": v2_dir / "notes" / "package_summary.md",
    }
    phase6 = load_phase6_packaging_evidence(root)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        missing_text = "\n".join(_relative_to_repo(path, root) for path in missing)
        raise FileNotFoundError(f"Cannot package {PAPER_CANDIDATE_V3_ID}; missing:\n{missing_text}")
    v2_index = _read_json(paths["paper_candidate_v2_index"])
    figure_paths = [
        _resolve_path(Path(str(path)), root) for path in v2_index.get("generated_figures", [])
    ]
    missing_figures = [path for path in figure_paths if not path.is_file()]
    if missing_figures:
        missing_text = "\n".join(_relative_to_repo(path, root) for path in missing_figures)
        raise FileNotFoundError(f"Cannot reference missing v2 figures:\n{missing_text}")
    return {
        "root": root,
        "phase6": phase6,
        "phase7a": _read_json(paths["phase7a_summary"]),
        "phase7b": _read_json(paths["phase7b_summary"]),
        "phase7c": _read_json(paths["phase7c_summary"]),
        "v2_index": v2_index,
        "paths": paths,
        "v2_figure_paths": figure_paths,
    }


def audit_phase7_packaging_evidence(evidence: dict[str, Any]) -> None:
    """Fail fast when a Phase 7 package input does not support its stated audit."""
    phase7a = evidence["phase7a"]
    if phase7a.get("input_corpus_id") != "paper_pseudo_bending_observation_pairs_v1":
        raise ValueError("Phase 7A package input corpus is not the paper pseudo-bending corpus.")
    interpretation = phase7a.get("interpretation", {})
    if not interpretation.get("full_beats_no_travel_time"):
        raise ValueError("Phase 7A does not support the full-vs-no-travel-time claim.")
    if not interpretation.get("full_beats_shuffled_travel_time"):
        raise ValueError("Phase 7A does not support the full-vs-shuffled control claim.")

    phase7b = evidence["phase7b"]
    if phase7b.get("sample_scope", {}).get("case_count") != 100:
        raise ValueError("Phase 7B package input must cover all 100 cases.")
    selection_audit = phase7b.get("selection_audit", {})
    required_audit_keys = {
        "phase6_reference",
        "best_covered_cell_rmse",
        "lower_clipping_with_minor_rmse_cost",
        "lowest_clipping",
    }
    if not required_audit_keys <= set(selection_audit):
        raise ValueError("Phase 7B selection audit is incomplete.")

    phase7c = evidence["phase7c"]
    if phase7c.get("sample_count") != 100:
        raise ValueError("Phase 7C package input must cover all 100 cases.")
    if phase7c.get("interpretation", {}).get("saturation_at_100_supported") is not False:
        raise ValueError("Phase 7C must retain the unresolved 100-case saturation boundary.")
    if int(phase7c.get("interpretation", {}).get("largest_training_size_evaluated", 0)) != 70:
        raise ValueError("Phase 7C package must document the 70-case leakage-free limit.")


def _write_tables(evidence: dict[str, Any], tables_dir: Path) -> list[Path]:
    table_specs = (
        ("final_claims_and_evidence", _final_claim_rows(evidence)),
        ("phase7_ablation_summary", _phase7a_rows(evidence["phase7a"])),
        ("reference_ray_sensitivity_summary", _phase7b_rows(evidence["phase7b"])),
        ("learning_curve_summary", _phase7c_rows(evidence["phase7c"])),
    )
    paths: list[Path] = []
    for name, rows in table_specs:
        csv_path = tables_dir / f"{name}.csv"
        markdown_path = tables_dir / f"{name}.md"
        _write_csv(csv_path, rows)
        markdown_path.write_text(_markdown_table(rows), encoding="utf-8")
        paths.extend((markdown_path, csv_path))
    return paths


def _final_claim_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    phase6 = evidence["phase6"]["final_comparison"]
    ml = phase6["ml_pca_linear_cell_centered"]
    phase7a = evidence["phase7a"]
    phase7b = evidence["phase7b"]["selection_audit"]
    phase7c = evidence["phase7c"]
    root = evidence["root"]

    def source_path(group: str, key: str) -> str:
        source = evidence[group]["paths"][key] if group == "phase6" else evidence[group][key]
        return _relative_to_repo(source, root)

    return [
        {
            "claim_id": "phase6_ml_vs_reference_ray",
            "claim": "ML reconstruction outperformed the implemented fixed-ray tomography baseline under the configured synthetic benchmark and harmonized cell-centered metric contract.",
            "evidence": source_path("phase6", "final_comparison_summary"),
            "quantitative_context": (
                f"ML fixed-test RMSE={_format_float(ml['split_wise_metrics']['test']['all_cell_velocity_rmse_km_per_s']['mean'])} km/s; "
                f"reference-ray all-100-case RMSE={_format_float(phase6['reference_ray_classical']['all_cell_velocity_rmse_km_per_s']['mean'])} km/s"
            ),
            "boundary": "Reported scopes differ; reference-ray remains fixed-ray regularized tomography.",
        },
        {
            "claim_id": "phase7a_travel_time_signal",
            "claim": "Travel-time observations contributed measurable reconstruction signal beyond acquisition geometry and target-distribution priors.",
            "evidence": source_path("paths", "phase7a_summary"),
            "quantitative_context": (
                f"full_input={_format_float(phase7a['interpretation']['full_input_fixed_test_cell_rmse'])}; "
                f"no_travel_time={_format_float(phase7a['interpretation']['no_travel_time_fixed_test_cell_rmse'])}; "
                f"shuffled={_format_float(phase7a['interpretation']['shuffled_travel_time_fixed_test_cell_rmse'])} km/s"
            ),
            "boundary": "Selected PCA-linear model on the synthetic corpus only.",
        },
        {
            "claim_id": "phase7b_regularization_audit",
            "claim": "Phase 7B audited the reference-ray baseline across regularization choices; Phase 6 settings were not globally optimal, but the comparison is no longer based on a single unchecked setting.",
            "evidence": source_path("paths", "phase7b_summary"),
            "quantitative_context": (
                f"Phase 6 covered RMSE={_format_float(phase7b['phase6_reference']['covered_cell_velocity_rmse_km_per_s_mean'])}; "
                f"best bounded covered RMSE={_format_float(phase7b['best_covered_cell_rmse']['covered_cell_velocity_rmse_km_per_s_mean'])}; "
                f"lowest clipping={_format_float(phase7b['lowest_clipping']['clipped_velocity_cell_percentage_mean'])} fraction"
            ),
            "boundary": "Bounded sensitivity audit, not global classical optimization.",
        },
        {
            "claim_id": "phase7c_learning_curve",
            "claim": "Performance was still improving up to 70 leakage-free training cases; data saturation is not established.",
            "evidence": source_path("paths", "phase7c_summary"),
            "quantitative_context": (
                "fixed-test cell RMSE: "
                + ", ".join(
                    f"{row['training_size']}={_format_float(row['mean_test_cell_rmse'])}"
                    for row in phase7c["summary_rows"]
                )
                + " km/s"
            ),
            "boundary": "The fixed Phase 4/6 holdout leaves 70 eligible training cases; 100-case saturation was not tested leakage-free.",
        },
        {
            "claim_id": "study_scope_boundary",
            "claim": "The study remains synthetic-only and does not claim real-data validation or full nonlinear tomography.",
            "evidence": "paper_candidate_v3 artifact index and package summary",
            "quantitative_context": "Synthetic pseudo-bending corpus with five configured geological families.",
            "boundary": "Known-ray is optimistic diagnostic evidence; reference-ray is fixed-ray regularized evidence.",
        },
    ]


def _phase7a_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "variant_id": row["variant_id"],
            "diagnostic_only": row["diagnostic_only"],
            "fixed_test_node_rmse_km_per_s": row["fixed_test_node_rmse"],
            "fixed_test_node_mae_km_per_s": row["fixed_test_node_mae"],
            "fixed_test_cell_rmse_km_per_s": row["fixed_test_cell_rmse"],
            "fixed_test_cell_mae_km_per_s": row["fixed_test_cell_mae"],
            "repeated_mean_validation_rmse_km_per_s": row["repeated_mean_validation_rmse"],
            "loo_worst_family": row["leave_one_family_out_worst_family"],
            "loo_worst_rmse_km_per_s": row["leave_one_family_out_worst_rmse"],
        }
        for row in summary["summary_rows"]
    ]


def _phase7b_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    audit = summary["selection_audit"]
    rows = []
    for label, item in (
        ("phase6_reference", audit["phase6_reference"]),
        ("best_covered_cell_rmse", audit["best_covered_cell_rmse"]),
        ("lower_clipping_with_minor_rmse_cost", audit["lower_clipping_with_minor_rmse_cost"]),
        ("lowest_clipping", audit["lowest_clipping"]),
    ):
        rows.append(
            {
                "setting": label,
                "damping": item["damping"],
                "smoothing": item["smoothing"],
                "travel_time_rmse_s_mean": item["travel_time_rmse_s_mean"],
                "all_cell_rmse_km_per_s_mean": item["all_cell_velocity_rmse_km_per_s_mean"],
                "covered_cell_rmse_km_per_s_mean": item[
                    "covered_cell_velocity_rmse_km_per_s_mean"
                ],
                "clipped_cell_count_mean": item["clipped_velocity_cell_count_mean"],
                "clipped_cell_percentage_mean": item["clipped_velocity_cell_percentage_mean"],
                "audit_note": (
                    "Phase 6 setting"
                    if label == "phase6_reference"
                    else "Bounded sensitivity alternative"
                ),
            }
        )
    return rows


def _phase7c_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "training_size": row["training_size"],
            "repetition_count": row["repetition_count"],
            "mean_test_node_rmse_km_per_s": row["mean_test_node_rmse"],
            "mean_test_node_mae_km_per_s": row["mean_test_node_mae"],
            "mean_test_cell_rmse_km_per_s": row["mean_test_cell_rmse"],
            "mean_test_cell_mae_km_per_s": row["mean_test_cell_mae"],
            "std_test_cell_rmse_km_per_s": row["std_test_cell_rmse"],
            "mean_validation_node_rmse_km_per_s": row["mean_validation_node_rmse"],
            "mean_validation_node_mae_km_per_s": row["mean_validation_node_mae"],
            "mean_train_node_rmse_km_per_s": row["mean_train_node_rmse"],
            "mean_train_node_mae_km_per_s": row["mean_train_node_mae"],
        }
        for row in summary["summary_rows"]
    ]


def _write_package_summary(
    evidence: dict[str, Any],
    path: Path,
    table_paths: list[Path],
    repo_root: Path,
) -> Path:
    phase6 = evidence["phase6"]["final_comparison"]
    ml = phase6["ml_pca_linear_cell_centered"]
    phase7a = evidence["phase7a"]["interpretation"]
    phase7b = evidence["phase7b"]["selection_audit"]
    phase7c = evidence["phase7c"]
    lines = [
        "# Paper Candidate v3 Package Summary",
        "",
        f"Package ID: `{PAPER_CANDIDATE_V3_ID}`",
        "",
        "## Scope",
        "",
        (
            "This package consolidates the final Phase 6 comparison and Phase 7 audit evidence "
            "for paper writing. It creates a new package and preserves both "
            "`paper_candidate_v1` and `paper_candidate_v2`."
        ),
        "",
        "## Defensible Claims",
        "",
        (
            "ML reconstruction outperformed the implemented fixed-ray tomography baseline "
            "under the configured synthetic benchmark and harmonized cell-centered metric "
            "contract. The selected ML model has fixed-test cell-centered RMSE "
            f"`{_format_float(ml['split_wise_metrics']['test']['all_cell_velocity_rmse_km_per_s']['mean'])}` km/s, "
            "while the reference-ray all-case RMSE is "
            f"`{_format_float(phase6['reference_ray_classical']['all_cell_velocity_rmse_km_per_s']['mean'])}` km/s; "
            "these scopes are reported explicitly in the evidence tables."
        ),
        "",
        (
            "Phase 7A indicates that travel-time observations contributed measurable signal "
            "beyond geometry and target-distribution priors: full input, no-travel-time, and "
            "shuffled-travel-time fixed-test cell RMSE values are "
            f"`{_format_float(phase7a['full_input_fixed_test_cell_rmse'])}`, "
            f"`{_format_float(phase7a['no_travel_time_fixed_test_cell_rmse'])}`, and "
            f"`{_format_float(phase7a['shuffled_travel_time_fixed_test_cell_rmse'])}` km/s."
        ),
        "",
        (
            "Phase 7B audited the reference-ray baseline across regularization choices. The "
            "Phase 6 setting was not globally optimal within the bounded sweep, but the "
            "comparison is no longer based on a single unchecked setting: its covered-cell "
            f"RMSE was `{_format_float(phase7b['phase6_reference']['covered_cell_velocity_rmse_km_per_s_mean'])}` km/s "
            f"versus `{_format_float(phase7b['best_covered_cell_rmse']['covered_cell_velocity_rmse_km_per_s_mean'])}` km/s "
            "for the best bounded covered-cell setting."
        ),
        "",
        (
            "Phase 7C shows performance was still improving up to 70 leakage-free training "
            "cases; data saturation is not established. Fixed-test cell RMSE decreased from "
            f"`{_format_float(phase7c['summary_rows'][0]['mean_test_cell_rmse'])}` km/s at 20 cases "
            f"to `{_format_float(phase7c['summary_rows'][-1]['mean_test_cell_rmse'])}` km/s at 70 cases."
        ),
        "",
        "## Claim Boundaries",
        "",
        "- The study is synthetic-only and does not validate real-data performance.",
        "- Known-ray inversion is an optimistic diagnostic because it uses true-model ray geometry.",
        "- Reference-ray inversion is fixed-ray regularized tomography, not full nonlinear iterative tomography.",
        "- The fixed Phase 4/6 holdout leaves 70 eligible training cases; 100-case saturation is unresolved.",
        "- The Phase 7A family-mean diagnostic is not a fair deployable model.",
        "- No claim of universal ML superiority is made.",
        "",
        "## Generated Tables",
        "",
    ]
    lines.extend(f"- `{_relative_to_repo(item, repo_root)}`" for item in table_paths)
    lines.extend(["", "## Referenced Phase 6 Visual Figures", ""])
    lines.extend(
        f"- `{_relative_to_repo(path, repo_root)}`" for path in evidence["v2_figure_paths"]
    )
    lines.extend(
        [
            "",
            "The visual figures are referenced from the preserved `paper_candidate_v2` package "
            "and are not regenerated or overwritten by this package.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_experiment_note(
    evidence: dict[str, Any],
    path: Path,
    table_paths: list[Path],
    repo_root: Path,
) -> Path:
    phase7b = evidence["phase7b"]["selection_audit"]
    phase7c = evidence["phase7c"]
    lines = [
        "# Experiment: Paper Candidate v3",
        "",
        f"Experiment ID: `{PAPER_CANDIDATE_V3_ID}`",
        "",
        "## Objective",
        "",
        "Package Phase 6 and Phase 7 evidence into a final paper writing artifact "
        "without modifying earlier candidate packages or rerunning experiments.",
        "",
        "## Included Evidence",
        "",
        f"- Phase 6 final comparison: `{_relative_to_repo(evidence['phase6']['paths']['final_comparison_summary'], repo_root)}`",
        f"- Phase 7A observation ablation: `{_relative_to_repo(evidence['paths']['phase7a_summary'], repo_root)}`",
        f"- Phase 7B regularization sensitivity: `{_relative_to_repo(evidence['paths']['phase7b_summary'], repo_root)}`",
        f"- Phase 7C learning curve: `{_relative_to_repo(evidence['paths']['phase7c_summary'], repo_root)}`",
        f"- Preserved Phase 6 visual figures: `{_relative_to_repo(evidence['paths']['paper_candidate_v2_index'], repo_root)}`",
        "",
        "## Interpretation",
        "",
        "The package supports the cautious statement that ML reconstruction outperformed the "
        "implemented fixed-ray tomography baseline under the configured synthetic benchmark "
        "and harmonized cell-centered metric contract.",
        "",
        "Travel-time observations contributed measurable reconstruction signal beyond geometry "
        "and target-distribution priors in the Phase 7A PCA-linear ablation.",
        "",
        "Phase 7B showed that the Phase 6 reference-ray regularization pair was not globally "
        "optimal within the bounded sweep, while also showing why RMSE and clipping must be "
        "reported together. The Phase 6 pair remains preserved as the historical comparison.",
        "",
        f"Phase 7C reached `{phase7c['interpretation']['largest_training_size_evaluated']}` "
        "leakage-free training cases and still showed improvement, so data saturation is not "
        "established.",
        "",
        "## Boundaries",
        "",
        "This is synthetic-only evidence. It does not validate real-data deployment, claim "
        "universal ML superiority, or represent full nonlinear iterative tomography. Known-ray "
        "is optimistic diagnostic evidence, and reference-ray is a fixed-ray regularized baseline.",
        "",
        "## Audit References",
        "",
        f"- Phase 7B best bounded covered-cell RMSE: `{_format_float(phase7b['best_covered_cell_rmse']['covered_cell_velocity_rmse_km_per_s_mean'])}` km/s",
        f"- Phase 7B lowest clipping fraction: `{_format_float(phase7b['lowest_clipping']['clipped_velocity_cell_percentage_mean'])}`",
        "",
        "## Package Tables",
        "",
    ]
    lines.extend(f"- `{_relative_to_repo(item, repo_root)}`" for item in table_paths)
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_artifact_index(
    evidence: dict[str, Any],
    table_paths: list[Path],
    summary_path: Path,
    experiment_note: Path,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "artifact_type": "paper_candidate_v3_artifact_index",
        "package_id": PAPER_CANDIDATE_V3_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "does_not_rerun_training_or_tomography": True,
        "preserves_historical_packages": [
            "outputs/generated/paper_candidate_v1/",
            "outputs/generated/paper_candidate_v2/",
        ],
        "source_evidence": {
            key: _relative_to_repo(path, repo_root) for key, path in evidence["paths"].items()
        },
        "phase6_source_evidence": {
            key: _relative_to_repo(path, repo_root)
            for key, path in evidence["phase6"]["paths"].items()
        },
        "generated_tables": [_relative_to_repo(path, repo_root) for path in table_paths],
        "referenced_phase6_figures": [
            _relative_to_repo(path, repo_root) for path in evidence["v2_figure_paths"]
        ],
        "referenced_phase6_figure_metadata": evidence["v2_index"].get("figure_metadata_json"),
        "summary_note": _relative_to_repo(summary_path, repo_root),
        "experiment_note": _relative_to_repo(experiment_note, repo_root),
        "claim_boundary": [
            "Synthetic-only benchmark; no real-data validation claim.",
            "Known-ray inversion is optimistic diagnostic evidence using true-model ray geometry.",
            "Reference-ray inversion is fixed-ray regularized tomography, not full nonlinear tomography.",
            "Phase 7B is a bounded regularization sensitivity audit, not global classical optimization.",
            "Phase 7C does not establish saturation at 100 cases because the fixed holdout leaves 70 eligible training cases.",
            "No universal ML superiority claim.",
        ],
        "final_claims": _final_claim_rows(evidence),
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot write an empty Markdown table.")
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows({_key: _format_cell(value) for _key, value in row.items()} for row in rows)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_float(value: Any) -> str:
    return f"{float(value):.6f}"


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


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()
