"""Package the final manuscript-facing evidence package for Phase 8D."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from tomobench.utils.paths import get_repo_root

PAPER_CANDIDATE_V4_ID = "paper_candidate_v4"
DEFAULT_PAPER_CANDIDATE_V4_OUTPUT_DIR = Path("outputs/generated/paper_candidate_v4")


@dataclass(frozen=True)
class PaperCandidateV4PackageOutputs:
    """Paths written by the Phase 8D evidence package."""

    output_dir: Path
    artifact_index_json: Path
    summary_markdown: Path
    final_claims_markdown: Path
    final_claims_csv: Path
    paired_method_markdown: Path
    paired_method_csv: Path
    paired_family_markdown: Path
    paired_family_csv: Path
    uncertainty_method_markdown: Path
    uncertainty_method_csv: Path
    paired_delta_markdown: Path
    paired_delta_csv: Path
    commands_to_regenerate: Path
    data_availability_draft: Path
    software_availability_draft: Path
    experiment_note: Path


def package_paper_candidate_v4(
    *,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
    generated_at_utc: str | None = None,
    experiment_note_path: Path | None = None,
) -> PaperCandidateV4PackageOutputs:
    """Create the Phase 8D package without rerunning training or tomography."""
    root = (repo_root or get_repo_root()).resolve()
    package_dir = _resolve_path(output_dir or DEFAULT_PAPER_CANDIDATE_V4_OUTPUT_DIR, root)
    _assert_historical_packages_are_preserved(root, package_dir)
    sources = _load_sources(root)
    phase8a = _read_json(sources["phase8a_summary"])
    phase8b_methods = _read_json(sources["phase8b_method_summary"])
    phase8b_deltas = _read_json(sources["phase8b_delta_summary"])
    repro = _read_json(sources["repro_manifest"])
    v3_index = _read_json(sources["v3_index"])

    tables_dir = package_dir / "tables"
    notes_dir = package_dir / "notes"
    tables_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    paired_method_csv = tables_dir / "paired_test_method_summary.csv"
    paired_family_csv = tables_dir / "paired_test_family_summary.csv"
    shutil.copyfile(sources["phase8a_method_csv"], paired_method_csv)
    shutil.copyfile(sources["phase8a_family_csv"], paired_family_csv)
    paired_method_rows = _read_csv(paired_method_csv)
    paired_family_rows = _read_csv(paired_family_csv)
    paired_method_markdown = tables_dir / "paired_test_method_summary.md"
    paired_family_markdown = tables_dir / "paired_test_family_summary.md"
    _write_paired_method_markdown(paired_method_markdown, paired_method_rows, root)
    _write_paired_family_markdown(paired_family_markdown, paired_family_rows, root)

    uncertainty_method_rows = list(phase8b_methods.get("method_summary", []))
    paired_delta_rows = list(phase8b_deltas.get("paired_delta_summary", []))
    if not uncertainty_method_rows or not paired_delta_rows:
        raise ValueError("Phase 8B summaries must contain method and paired-delta rows.")
    uncertainty_method_csv = tables_dir / "uncertainty_method_summary.csv"
    uncertainty_method_markdown = tables_dir / "uncertainty_method_summary.md"
    paired_delta_csv = tables_dir / "paired_delta_summary.csv"
    paired_delta_markdown = tables_dir / "paired_delta_summary.md"
    _write_csv(uncertainty_method_csv, uncertainty_method_rows)
    _write_csv(paired_delta_csv, paired_delta_rows)
    _write_uncertainty_method_markdown(uncertainty_method_markdown, uncertainty_method_rows, root)
    _write_paired_delta_markdown(paired_delta_markdown, paired_delta_rows, root)

    final_claim_rows = _final_claim_rows(
        phase8a=phase8a,
        phase8b_deltas=paired_delta_rows,
        v3_index=v3_index,
        sources=sources,
        root=root,
    )
    final_claims_csv = tables_dir / "final_submission_claims_and_evidence.csv"
    final_claims_markdown = tables_dir / "final_submission_claims_and_evidence.md"
    _write_csv(final_claims_csv, final_claim_rows)
    _write_final_claims_markdown(final_claims_markdown, final_claim_rows)

    figure_table_plan = package_dir / "tables" / "manuscript_figure_table_plan.md"
    _write_figure_table_plan(figure_table_plan, v3_index, root)

    commands_to_regenerate = package_dir / "commands_to_regenerate.md"
    data_availability_draft = package_dir / "data_availability_draft.md"
    software_availability_draft = package_dir / "software_availability_draft.md"
    repro_manifest_copy = package_dir / "reproducibility_manifest.json"
    environment_summary_copy = package_dir / "environment_summary.json"
    _write_v4_commands(sources["commands"], commands_to_regenerate)
    shutil.copyfile(sources["data_draft"], data_availability_draft)
    shutil.copyfile(sources["software_draft"], software_availability_draft)
    shutil.copyfile(sources["repro_manifest"], repro_manifest_copy)
    shutil.copyfile(sources["environment"], environment_summary_copy)

    summary_markdown = notes_dir / "package_summary.md"
    _write_package_summary(
        summary_markdown,
        phase8a=phase8a,
        phase8b_deltas=paired_delta_rows,
        v3_index=v3_index,
        repro=repro,
        package_dir=package_dir,
        root=root,
        table_paths=(
            final_claims_markdown,
            final_claims_csv,
            paired_method_markdown,
            paired_method_csv,
            paired_family_markdown,
            paired_family_csv,
            uncertainty_method_markdown,
            uncertainty_method_csv,
            paired_delta_markdown,
            paired_delta_csv,
            figure_table_plan,
        ),
    )

    note_path = _resolve_path(
        experiment_note_path
        or (root / "wiki" / "experiments" / f"{PAPER_CANDIDATE_V4_ID}.md"),
        root,
    )
    _write_experiment_note(
        note_path,
        phase8a=phase8a,
        phase8b_deltas=paired_delta_rows,
        v3_index=v3_index,
        repro=repro,
        package_dir=package_dir,
        root=root,
    )

    generated_at = generated_at_utc or datetime.now(UTC).isoformat()
    artifact_index_path = package_dir / "artifact_index.json"
    artifact_index = _build_artifact_index(
        package_dir=package_dir,
        root=root,
        generated_at_utc=generated_at,
        sources=sources,
        package_files=(
            summary_markdown,
            final_claims_markdown,
            final_claims_csv,
            paired_method_markdown,
            paired_method_csv,
            paired_family_markdown,
            paired_family_csv,
            uncertainty_method_markdown,
            uncertainty_method_csv,
            paired_delta_markdown,
            paired_delta_csv,
            figure_table_plan,
            commands_to_regenerate,
            data_availability_draft,
            software_availability_draft,
            repro_manifest_copy,
            environment_summary_copy,
        ),
        experiment_note=note_path,
        v3_index=v3_index,
    )
    _write_json(artifact_index_path, artifact_index)
    return PaperCandidateV4PackageOutputs(
        output_dir=package_dir,
        artifact_index_json=artifact_index_path,
        summary_markdown=summary_markdown,
        final_claims_markdown=final_claims_markdown,
        final_claims_csv=final_claims_csv,
        paired_method_markdown=paired_method_markdown,
        paired_method_csv=paired_method_csv,
        paired_family_markdown=paired_family_markdown,
        paired_family_csv=paired_family_csv,
        uncertainty_method_markdown=uncertainty_method_markdown,
        uncertainty_method_csv=uncertainty_method_csv,
        paired_delta_markdown=paired_delta_markdown,
        paired_delta_csv=paired_delta_csv,
        commands_to_regenerate=commands_to_regenerate,
        data_availability_draft=data_availability_draft,
        software_availability_draft=software_availability_draft,
        experiment_note=note_path,
    )


def _load_sources(root: Path) -> dict[str, Path]:
    sources = {
        "v3_dir": root / "outputs/generated/paper_candidate_v3",
        "v3_index": root / "outputs/generated/paper_candidate_v3/artifact_index.json",
        "v3_summary": root / "outputs/generated/paper_candidate_v3/notes/package_summary.md",
        "v3_claims": root / "outputs/generated/paper_candidate_v3/tables/final_claims_and_evidence.md",
        "v4_source": root / "code/src/tomobench/evaluation/paper_candidate_v4.py",
        "v4_tests": root / "code/tests/unit/test_paper_candidate_v4.py",
        "cli_source": root / "code/src/tomobench/cli.py",
        "phase6_final_dir": root / "outputs/generated/classical_tomography/final_phase6_comparison_v1",
        "phase6_known_dir": root / "outputs/generated/classical_tomography/known_ray_corpus_v1",
        "phase6_reference_dir": root / "outputs/generated/classical_tomography/reference_ray_corpus_v1",
        "phase7a_dir": root / "outputs/generated/ml_baselines/phase7_observation_signal_ablation_v1",
        "phase7b_dir": root / "outputs/generated/classical_tomography/reference_ray_regularization_sensitivity_v1",
        "phase7c_dir": root / "outputs/generated/ml_baselines/phase7_learning_curve_v1",
        "phase6_summary": root / "outputs/generated/classical_tomography/final_phase6_comparison_v1/final_phase6_comparison_summary.json",
        "phase7a_summary": root / "outputs/generated/ml_baselines/phase7_observation_signal_ablation_v1/ablation_summary.json",
        "phase7b_summary": root / "outputs/generated/classical_tomography/reference_ray_regularization_sensitivity_v1/regularization_sensitivity_summary.json",
        "phase7c_summary": root / "outputs/generated/ml_baselines/phase7_learning_curve_v1/learning_curve_summary.json",
        "phase8a_dir": root / "outputs/generated/submission_readiness/phase8_paired_test_comparison_v1",
        "phase8a_summary": root / "outputs/generated/submission_readiness/phase8_paired_test_comparison_v1/paired_test_summary.json",
        "phase8a_method_csv": root / "outputs/generated/submission_readiness/phase8_paired_test_comparison_v1/paired_test_summary.csv",
        "phase8a_case_csv": root / "outputs/generated/submission_readiness/phase8_paired_test_comparison_v1/paired_test_case_metrics.csv",
        "phase8a_family_csv": root / "outputs/generated/submission_readiness/phase8_paired_test_comparison_v1/paired_family_summary.csv",
        "phase8a_ids": root / "outputs/generated/submission_readiness/phase8_paired_test_comparison_v1/fixed_test_case_ids.json",
        "phase8b_dir": root / "outputs/generated/submission_readiness/phase8_uncertainty_report_v1",
        "phase8b_method_summary": root / "outputs/generated/submission_readiness/phase8_uncertainty_report_v1/method_uncertainty_summary.json",
        "phase8b_delta_summary": root / "outputs/generated/submission_readiness/phase8_uncertainty_report_v1/paired_delta_summary.json",
        "phase8b_case_deltas": root / "outputs/generated/submission_readiness/phase8_uncertainty_report_v1/paired_case_deltas.csv",
        "repro_dir": root / "outputs/generated/submission_readiness/reproducibility_manifest_v1",
        "repro_manifest": root / "outputs/generated/submission_readiness/reproducibility_manifest_v1/reproducibility_manifest.json",
        "commands": root / "outputs/generated/submission_readiness/reproducibility_manifest_v1/commands_to_regenerate.md",
        "data_draft": root / "outputs/generated/submission_readiness/reproducibility_manifest_v1/data_availability_draft.md",
        "software_draft": root / "outputs/generated/submission_readiness/reproducibility_manifest_v1/software_availability_draft.md",
        "environment": root / "outputs/generated/submission_readiness/reproducibility_manifest_v1/environment_summary.json",
    }
    directory_keys = {
        "v3_dir",
        "phase6_final_dir",
        "phase6_known_dir",
        "phase6_reference_dir",
        "phase7a_dir",
        "phase7b_dir",
        "phase7c_dir",
        "phase8a_dir",
        "phase8b_dir",
        "repro_dir",
    }
    missing = [
        path for key, path in sources.items() if key not in directory_keys and not path.is_file()
    ]
    missing.extend(
        path for key, path in sources.items() if key in directory_keys and not path.is_dir()
    )
    if missing:
        missing_text = "\n".join(_relative_to_repo(path, root) for path in missing)
        raise FileNotFoundError(f"Cannot package {PAPER_CANDIDATE_V4_ID}; missing:\n{missing_text}")
    return sources


def _assert_historical_packages_are_preserved(root: Path, package_dir: Path) -> None:
    if package_dir.resolve() == (root / "outputs/generated/paper_candidate_v3").resolve():
        raise ValueError("Phase 8D cannot use paper_candidate_v3 as its output directory.")
    missing = [
        root / f"outputs/generated/paper_candidate_v{i}"
        for i in (1, 2, 3)
        if not (root / f"outputs/generated/paper_candidate_v{i}").is_dir()
    ]
    if missing:
        missing_text = ", ".join(_relative_to_repo(path, root) for path in missing)
        raise FileNotFoundError(f"Historical paper candidate packages are missing: {missing_text}")


def _final_claim_rows(
    *,
    phase8a: Mapping[str, Any],
    phase8b_deltas: Sequence[Mapping[str, Any]],
    v3_index: Mapping[str, Any],
    sources: Mapping[str, Path],
    root: Path,
) -> list[dict[str, Any]]:
    for comparison_id in (
        "full_ml_vs_reference_ray",
        "full_ml_vs_known_ray",
        "full_ml_vs_mean_target",
        "full_ml_vs_no_travel_time",
        "full_ml_vs_shuffled_travel_time",
    ):
        _find_delta(phase8b_deltas, comparison_id)
    full_time_only = _find_delta(phase8b_deltas, "full_ml_vs_travel_time_only")
    phase7b_claim = _find_v3_claim(v3_index, "phase7b_regularization_audit")
    phase7c_claim = _find_v3_claim(v3_index, "phase7c_learning_curve")
    phase8a_path = _relative_to_repo(sources["phase8a_summary"], root)
    phase8b_path = _relative_to_repo(sources["phase8b_delta_summary"], root)
    rows = [
        {
            "claim_id": "synthetic_benchmark_scope",
            "claim_status": "use",
            "claim": "The evidence concerns a configured synthetic seismic benchmark.",
            "evidence": _relative_to_repo(sources["v3_summary"], root),
            "scope": "Synthetic five-family pseudo-bending corpus.",
            "caveat": "No real-data validation is claimed.",
        },
        {
            "claim_id": "paired_cell_centered_comparison",
            "claim_status": "use",
            "claim": (
                f"ML, classical, and sanity methods were compared on the same "
                f"{phase8a['paired_test_case_count']} fixed test cases under a harmonized "
                "cell-centered RMSE/MAE contract."
            ),
            "evidence": f"{phase8a_path}; { _relative_to_repo(sources['phase8a_case_csv'], root) }",
            "scope": "Phase 8A fixed test cases only.",
            "caveat": "The ranking is descriptive for this fixed synthetic test set.",
        },
        {
            "claim_id": "travel_time_signal_beyond_controls",
            "claim_status": "use",
            "claim": (
                "Travel-time-bearing ML variants outperform the no-travel-time, "
                "shuffled-travel-time, and mean-target controls on the paired comparison."
            ),
            "evidence": phase8b_path,
            "scope": "Selected PCA-linear observation regressor and Phase 7A controls.",
            "caveat": (
                "This is evidence within the configured synthetic benchmark; exploratory "
                "p-values are not definitive with 25 paired cases."
            ),
        },
        {
            "claim_id": "full_ml_vs_reference_ray",
            "claim_status": "use",
            "claim": (
                "Full ML has lower paired error than the implemented Phase 6 reference-ray "
                "fixed-ray baseline under the configured synthetic benchmark."
            ),
            "evidence": phase8b_path,
            "scope": "Phase 6 reference-ray setting retained as primary.",
            "caveat": (
                "Reference-ray is a fixed-ray regularized baseline, not full nonlinear "
                "tomography or a universal classical benchmark."
            ),
        },
        {
            "claim_id": "full_ml_vs_travel_time_only_unresolved",
            "claim_status": "use",
            "claim": (
                "Full input and travel-time-only are nearly tied; the uncertainty analysis "
                "does not establish a meaningful difference."
            ),
            "evidence": phase8b_path,
            "scope": "Full ML versus travel-time-only on 25 paired cases.",
            "caveat": (
                f"RMSE mean delta={_format_float(full_time_only['rmse_delta_mean_km_per_s'])} km/s; "
                f"95% CI=[{_format_float(full_time_only['rmse_delta_ci95_lower_km_per_s'])}, "
                f"{_format_float(full_time_only['rmse_delta_ci95_upper_km_per_s'])}] km/s."
            ),
        },
        {
            "claim_id": "phase7b_classical_audit",
            "claim_status": "use",
            "claim": phase7b_claim["claim"],
            "evidence": _relative_to_repo(sources["phase7b_summary"], root),
            "scope": "Phase 7B bounded regularization/clipping sensitivity sweep.",
            "caveat": "Sensitivity/audit evidence only; it does not silently replace the Phase 6 primary row.",
        },
        {
            "claim_id": "phase7c_learning_curve",
            "claim_status": "use",
            "claim": phase7c_claim["claim"],
            "evidence": _relative_to_repo(sources["phase7c_summary"], root),
            "scope": "Leakage-free fixed-holdout training sizes through 70 cases.",
            "caveat": "Data saturation is not established.",
        },
        {
            "claim_id": "reproducibility_materials",
            "claim_status": "use",
            "claim": (
                "A reproducibility manifest, ordered command list, configuration hash, "
                "environment metadata, and availability drafts are included."
            ),
            "evidence": _relative_to_repo(sources["repro_manifest"], root),
            "scope": "Phase 8C submission-readiness materials.",
            "caveat": "The availability drafts do not assert public release or deposition.",
        },
        {
            "claim_id": "avoid_real_data_validation",
            "claim_status": "avoid",
            "claim": "Real-data validation or field deployment has been demonstrated.",
            "evidence": "Not supported by this evidence package.",
            "scope": "Outside the synthetic benchmark.",
            "caveat": "Do not use this claim.",
        },
        {
            "claim_id": "avoid_full_nonlinear_tomography",
            "claim_status": "avoid",
            "claim": "The workflow constitutes full nonlinear iterative tomography.",
            "evidence": "Not supported by this evidence package.",
            "scope": "Reference-ray is fixed-ray regularized inversion.",
            "caveat": "Do not use this claim.",
        },
        {
            "claim_id": "avoid_universal_ml_superiority",
            "claim_status": "avoid",
            "claim": "ML is universally superior to classical tomography.",
            "evidence": "Not supported by a synthetic fixed-test comparison.",
            "scope": "Configured benchmark only.",
            "caveat": "Do not generalize beyond the implemented baselines and test set.",
        },
        {
            "claim_id": "avoid_public_deposition",
            "claim_status": "avoid",
            "claim": "The code or data have already been publicly released or deposited.",
            "evidence": _relative_to_repo(sources["data_draft"], root),
            "scope": "Availability wording is a draft.",
            "caveat": "Do not claim public release, repository URL, DOI, or deposition without evidence.",
        },
        {
            "claim_id": "avoid_family_mean_deployment",
            "claim_status": "avoid",
            "claim": "The Phase 7A family-mean diagnostic is a deployable prediction model.",
            "evidence": _relative_to_repo(sources["v3_claims"], root),
            "scope": "Family-label diagnostic only.",
            "caveat": "Do not present it as a fair deployable model.",
        },
    ]
    return rows


def _write_package_summary(
    path: Path,
    *,
    phase8a: Mapping[str, Any],
    phase8b_deltas: Sequence[Mapping[str, Any]],
    v3_index: Mapping[str, Any],
    repro: Mapping[str, Any],
    package_dir: Path,
    root: Path,
    table_paths: Sequence[Path],
) -> None:
    ranking = sorted(
        _read_csv(
            root
            / "outputs/generated/submission_readiness/phase8_paired_test_comparison_v1/paired_test_summary.csv"
        ),
        key=lambda row: int(row["rank"]),
    )
    full_time_only = _find_delta(phase8b_deltas, "full_ml_vs_travel_time_only")
    phase7b_claim = _find_v3_claim(v3_index, "phase7b_regularization_audit")
    phase7c_claim = _find_v3_claim(v3_index, "phase7c_learning_curve")
    repository = repro.get("repository", {})
    runtime = repro.get("runtime_environment", {})
    central_config = repro.get("central_config", {})
    lines = [
        "# Paper Candidate v4 Package Summary",
        "",
        f"Package ID: `{PAPER_CANDIDATE_V4_ID}`",
        "",
        "## Purpose",
        "",
        (
            "This package is the final manuscript-facing evidence layer for the current "
            "Applied Sciences preparation. It rolls together the preserved Phase 6/7 "
            "technical evidence, the Phase 8A harmonized paired comparison, the Phase 8B "
            "uncertainty report, and the Phase 8C reproducibility materials. It reads existing "
            "artifacts and does not rerun training, simulation, or tomography."
        ),
        "",
        (
            "The earlier `paper_candidate_v1`, `paper_candidate_v2`, and `paper_candidate_v3` "
            "directories are preserved and are not overwritten by this package."
        ),
        "",
        "## Paired Fixed-Test Evidence",
        "",
        (
            f"Phase 8A contains {phase8a['paired_test_case_count']} exact fixed test cases and "
            "seven method rows per case under the shared cell-centered velocity RMSE/MAE "
            "contract. The descriptive RMSE ranking is:"
        ),
        "",
    ]
    lines.extend(
        f"{row['rank']}. `{row['method_id']}` — RMSE `{_format_float(row['rmse_mean_km_per_s'])}` km/s; "
        f"MAE `{_format_float(row['mae_mean_km_per_s'])}` km/s"
        for row in ranking
    )
    lines.extend(
        [
            "",
            (
                "Travel-time-only and full ML are nearly tied descriptively. Phase 8B does not "
                "meaningfully establish a difference: the full-ML minus travel-time-only RMSE "
                f"delta is `{_format_float(full_time_only['rmse_delta_mean_km_per_s'])}` km/s with "
                f"95% CI `[{_format_float(full_time_only['rmse_delta_ci95_lower_km_per_s'])}, "
                f"{_format_float(full_time_only['rmse_delta_ci95_upper_km_per_s'])}]` km/s, and "
                "the paired interval contains zero."
            ),
            "",
            (
                "The stronger supported comparison is against the controls and the implemented "
                "Phase 6 reference-ray fixed-ray baseline. The paired RMSE intervals for full ML "
                "versus no-travel-time, shuffled-travel-time, mean-target, and reference-ray all "
                "exclude zero in the favorable direction in the Phase 8B report."
            ),
            "",
            "## Phase 7 Evidence Boundaries",
            "",
            (
                f"Phase 7B: {phase7b_claim['claim']} Its source quantitative context is "
                f"`{phase7b_claim.get('quantitative_context', '')}`. This remains a bounded "
                "sensitivity/audit result; the Phase 6 regularization pair remains the primary "
                "reference-ray row in the paired comparison."
            ),
            "",
            (
                f"Phase 7C: {phase7c_claim['claim']} Its source quantitative context is "
                f"`{phase7c_claim.get('quantitative_context', '')}`. Data saturation is not "
                "established."
            ),
            "",
            "## Reproducibility",
            "",
            (
                f"The copied Phase 8C manifest records Python `{runtime.get('python_version', 'not recorded')}`, "
                f"configuration SHA-256 `{central_config.get('sha256', 'not recorded')}`, "
                f"commit `{repository.get('commit_hash', 'not recorded')}`, and dirty-worktree="
                f"`{repository.get('dirty_worktree', 'not recorded')}`. The ordered command list, "
                "environment metadata, and conservative data/software availability drafts are "
                "included at the package root."
            ),
            "",
            "The availability drafts use conditional wording and do not claim that code or data have been publicly released or deposited.",
            "",
            "## Claims to Use",
            "",
            "- Synthetic benchmark and synthetic-only scope.",
            "- Implemented Phase 6 reference-ray fixed-ray baseline.",
            "- Harmonized cell-centered paired comparison on the same fixed test cases.",
            "- Travel-time signal contributes beyond the geometry/prior controls in this benchmark.",
            "- Full-input versus travel-time-only remains unresolved by the uncertainty analysis.",
            "",
            "## Claims to Avoid",
            "",
            "- Real-data validation or field deployment.",
            "- Full nonlinear iterative tomography.",
            "- Universal ML superiority.",
            "- Public release, repository deposition, DOI, or repository URL unless completed and documented.",
            "- Treating the family-mean diagnostic as a deployable model.",
            "",
            "## Package Limitations",
            "",
            "- The comparison is synthetic-only and uses 25 paired test cases.",
            "- Known-ray inversion is an optimistic diagnostic using true-model ray geometry.",
            "- Reference-ray inversion is fixed-ray regularized evidence, not full nonlinear tomography.",
            "- Phase 7B does not establish a globally optimal classical regularization choice.",
            "- Data saturation is not established by the available leakage-free 70-case learning curve.",
            "- The small full-ML versus travel-time-only difference should not be presented as established.",
            "",
            "## Included Manuscript-Facing Tables",
            "",
        ]
    )
    lines.extend(f"- `{_relative_to_repo(path, root)}`" for path in table_paths)
    lines.extend(
        [
            "",
            f"Package directory: `{_relative_to_repo(package_dir, root)}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_final_claims_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Final Submission Claims and Evidence",
        "",
        "This table is a manuscript-control aid. `use` rows are bounded claims supported by the packaged evidence; `avoid` rows are explicitly unsupported formulations.",
        "",
        _markdown_table(rows),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_paired_method_markdown(path: Path, rows: Sequence[Mapping[str, Any]], root: Path) -> None:
    fields = (
        "rank",
        "method_id",
        "method_label",
        "case_count",
        "rmse_mean_km_per_s",
        "rmse_median_km_per_s",
        "rmse_std_km_per_s",
        "mae_mean_km_per_s",
        "mae_median_km_per_s",
        "mae_std_km_per_s",
    )
    _write_table_markdown(
        path,
        "Paired Fixed-Test Method Summary",
        "Source: Phase 8A paired_test_summary.csv. All rows use the same fixed test cases and cell-centered metric contract.",
        rows,
        fields,
        root,
    )


def _write_paired_family_markdown(path: Path, rows: Sequence[Mapping[str, Any]], root: Path) -> None:
    fields = (
        "method_id",
        "method_label",
        "family",
        "case_count",
        "rmse_mean_km_per_s",
        "rmse_median_km_per_s",
        "rmse_std_km_per_s",
        "mae_mean_km_per_s",
        "mae_median_km_per_s",
        "mae_std_km_per_s",
    )
    _write_table_markdown(
        path,
        "Paired Fixed-Test Family Summary",
        "Source: Phase 8A paired_family_summary.csv. Family means remain descriptive synthetic-family diagnostics.",
        rows,
        fields,
        root,
    )


def _write_uncertainty_method_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    root: Path,
) -> None:
    fields = (
        "method_id",
        "method_label",
        "case_count",
        "rmse_mean_km_per_s",
        "rmse_std_km_per_s",
        "rmse_median_km_per_s",
        "rmse_iqr_km_per_s",
        "rmse_ci95_lower_km_per_s",
        "rmse_ci95_upper_km_per_s",
        "mae_mean_km_per_s",
        "mae_std_km_per_s",
        "mae_median_km_per_s",
        "mae_iqr_km_per_s",
        "mae_ci95_lower_km_per_s",
        "mae_ci95_upper_km_per_s",
    )
    _write_table_markdown(
        path,
        "Method Uncertainty Summary",
        "Source: Phase 8B method_uncertainty_summary.json. Intervals are deterministic percentile-bootstrap 95% CIs for the mean across 25 cases.",
        rows,
        fields,
        root,
    )


def _write_paired_delta_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    root: Path,
) -> None:
    fields = (
        "comparison_id",
        "first_method_label",
        "second_method_label",
        "case_count",
        "rmse_delta_mean_km_per_s",
        "rmse_delta_median_km_per_s",
        "rmse_delta_ci95_lower_km_per_s",
        "rmse_delta_ci95_upper_km_per_s",
        "rmse_win_count",
        "rmse_loss_count",
        "rmse_tie_count",
        "rmse_improvement_fraction",
        "rmse_exploratory_sign_flip_p_value",
        "interpretation",
    )
    _write_table_markdown(
        path,
        "Paired Method-Difference Summary",
        "Source: Phase 8B paired_delta_summary.json. Delta is first method minus second; negative values favor the first method. P-values are exploratory only.",
        rows,
        fields,
        root,
    )


def _write_table_markdown(
    path: Path,
    title: str,
    description: str,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    root: Path,
) -> None:
    selected = [{field: row.get(field, "") for field in fields} for row in rows]
    lines = [f"# {title}", "", description, "", _markdown_table(selected)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    _ = root


def _write_figure_table_plan(path: Path, v3_index: Mapping[str, Any], root: Path) -> None:
    figures = list(v3_index.get("referenced_phase6_figures", []))
    lines = [
        "# Manuscript Figure and Table Plan",
        "",
        "This is a planning aid; it does not create new scientific results or figures.",
        "",
        "## Candidate Figures",
        "",
        "- Preserve representative Phase 6 cell-centered comparison figures for the configured geological families.",
        "- Cite the original preserved `paper_candidate_v2` figure paths listed below; v4 does not overwrite or regenerate them.",
        "",
    ]
    lines.extend(f"- `{figure}`" for figure in figures)
    lines.extend(
        [
            "",
            "## Candidate Tables",
            "",
            "- Final claims and evidence.",
            "- Paired fixed-test method and family summaries.",
            "- Bootstrap method uncertainty summary.",
            "- Paired method-difference summary with wins/losses/ties and exploratory p-values.",
            "- Reproducibility commands and availability drafts as supplementary submission material.",
            "",
            "## Editing Note",
            "",
            "Keep the full-input versus travel-time-only uncertainty caveat adjacent to any descriptive ranking table.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    _ = root


def _write_v4_commands(source_path: Path, target_path: Path) -> None:
    """Extend the Phase 8C command sequence with the final Phase 8D step."""
    source_text = source_path.read_text(encoding="utf-8").rstrip()
    addition = """

## Phase 8D final package

Run after the Phase 8C command above:

| Order | Phase | Command | Main outputs |
|---:|---|---|---|
| 20 | `phase8d` | `uv run tomobench package-paper-candidate-v4` | `outputs/generated/paper_candidate_v4` |
"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(source_text + addition, encoding="utf-8")


def _write_experiment_note(
    path: Path,
    *,
    phase8a: Mapping[str, Any],
    phase8b_deltas: Sequence[Mapping[str, Any]],
    v3_index: Mapping[str, Any],
    repro: Mapping[str, Any],
    package_dir: Path,
    root: Path,
) -> None:
    full_time_only = _find_delta(phase8b_deltas, "full_ml_vs_travel_time_only")
    repository = repro.get("repository", {})
    lines = [
        "# Experiment: Paper Candidate v4",
        "",
        f"Experiment ID: `{PAPER_CANDIDATE_V4_ID}`",
        "",
        "## Objective",
        "",
        "Roll Phase 6/7 technical evidence, Phase 8A paired comparison, Phase 8B uncertainty, and Phase 8C reproducibility material into a final manuscript-facing package without modifying earlier candidate packages or rerunning experiments.",
        "",
        "## Inputs",
        "",
        "- Preserved `paper_candidate_v3` artifact index, summary, and claims table.",
        "- Phase 8A fixed-test case, method, and family summaries.",
        "- Phase 8B bootstrap method uncertainty and paired delta summaries.",
        "- Phase 8C reproducibility manifest, commands, environment, and availability drafts.",
        "",
        "## Results Packaged",
        "",
        f"- Paired test cases: `{phase8a['paired_test_case_count']}`.",
        "- Full ML and travel-time-only are descriptively nearly tied; their paired RMSE uncertainty interval contains zero.",
        f"- Full ML minus travel-time-only RMSE mean delta: `{_format_float(full_time_only['rmse_delta_mean_km_per_s'])}` km/s.",
        "- Travel-time-bearing variants outperform the configured no-travel-time, shuffled-travel-time, and mean-target controls in this synthetic benchmark.",
        "- Full ML is lower than the implemented Phase 6 reference-ray fixed-ray baseline on the paired comparison.",
        "",
        "## Boundaries",
        "",
        "- Synthetic-only; no real-data validation.",
        "- Known-ray inversion is an optimistic diagnostic.",
        "- Reference-ray inversion is a fixed-ray regularized baseline, not full nonlinear tomography.",
        "- Phase 7B remains a bounded sensitivity/audit record; its alternatives do not silently replace Phase 6.",
        "- Phase 7C does not establish data saturation.",
        "- No universal ML superiority or public deposition claim is made.",
        "",
        "## Provenance",
        "",
        f"- Git commit recorded by Phase 8C: `{repository.get('commit_hash', 'not recorded')}`.",
        f"- Dirty-worktree flag recorded by Phase 8C: `{repository.get('dirty_worktree', 'not recorded')}`.",
        f"- Package output: `{_relative_to_repo(package_dir, root)}`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    _ = v3_index


def _build_artifact_index(
    *,
    package_dir: Path,
    root: Path,
    generated_at_utc: str,
    sources: Mapping[str, Path],
    package_files: Sequence[Path],
    experiment_note: Path,
    v3_index: Mapping[str, Any],
) -> dict[str, Any]:
    source_specs = _source_specs(sources)
    package_records = [
        _artifact_record(
            artifact_id=f"v4_{path.relative_to(package_dir).as_posix().replace('/', '_').replace('.', '_')}",
            path=path,
            root=root,
            artifact_type="package_output",
            phase="phase8d",
            role="generated",
        )
        for path in package_files
    ]
    package_records.append(
        _artifact_record(
            artifact_id="v4_experiment_note",
            path=experiment_note,
            root=root,
            artifact_type="documentation",
            phase="phase8d",
            role="documentation",
        )
    )
    source_records = [
        _artifact_record(
            artifact_id=spec["artifact_id"],
            path=spec["path"],
            root=root,
            artifact_type=spec["artifact_type"],
            phase=spec["phase"],
            role=spec["role"],
            description=spec["description"],
            source_artifact=spec.get("source_artifact"),
        )
        for spec in source_specs
    ]
    generated_paths = [_relative_to_repo(path, root) for path in package_files]
    source_evidence = {
        spec["artifact_id"]: _relative_to_repo(spec["path"], root) for spec in source_specs
    }
    reproducibility_materials = {
        _relative_to_repo(path, root): _relative_to_repo(sources[source_key], root)
        for path, source_key in (
            (package_dir / "commands_to_regenerate.md", "commands"),
            (package_dir / "data_availability_draft.md", "data_draft"),
            (package_dir / "software_availability_draft.md", "software_draft"),
            (package_dir / "reproducibility_manifest.json", "repro_manifest"),
            (package_dir / "environment_summary.json", "environment"),
        )
    }
    preservation = {
        f"paper_candidate_v{i}": {
            "path": f"outputs/generated/paper_candidate_v{i}",
            "exists": (root / f"outputs/generated/paper_candidate_v{i}").is_dir(),
        }
        for i in (1, 2, 3)
    }
    return {
        "artifact_type": "paper_candidate_v4_artifact_index",
        "package_id": PAPER_CANDIDATE_V4_ID,
        "generated_at_utc": generated_at_utc,
        "does_not_rerun_training_or_tomography": True,
        "preserves_historical_packages": [
            "outputs/generated/paper_candidate_v1/",
            "outputs/generated/paper_candidate_v2/",
            "outputs/generated/paper_candidate_v3/",
        ],
        "preservation_audit": preservation,
        "source_evidence": source_evidence,
        "phase8_sources": {
            key: value
            for key, value in source_evidence.items()
            if key.startswith("phase8") or key.startswith("reproducibility")
        },
        "reproducibility_materials": reproducibility_materials,
        "generated_tables": [path for path in generated_paths if "/tables/" in path],
        "package_outputs": [
            _relative_to_repo(package_dir / "artifact_index.json", root),
            *generated_paths,
        ],
        "summary_note": _relative_to_repo(package_dir / "notes/package_summary.md", root),
        "experiment_note": _relative_to_repo(experiment_note, root),
        "artifact_map": source_records + package_records,
        "final_claims": _find_claims_for_index(package_dir / "tables/final_submission_claims_and_evidence.csv"),
        "claim_boundaries": [
            "Synthetic-only benchmark; no real-data validation claim.",
            "Known-ray inversion is optimistic diagnostic evidence using true-model ray geometry.",
            "Reference-ray inversion is a fixed-ray regularized baseline, not full nonlinear tomography.",
            "Phase 7B alternatives are sensitivity/audit results and do not silently replace the Phase 6 primary setting.",
            "Phase 7C does not establish data saturation.",
            "The full-input versus travel-time-only difference is not meaningfully established.",
            "No universal ML superiority or public-deposition claim is made.",
        ],
        "source_package_indexes": {
            "paper_candidate_v3": _relative_to_repo(sources["v3_index"], root),
            "phase8a": _relative_to_repo(sources["phase8a_summary"], root),
            "phase8b_method_uncertainty": _relative_to_repo(sources["phase8b_method_summary"], root),
            "phase8b_paired_delta": _relative_to_repo(sources["phase8b_delta_summary"], root),
            "phase8c": _relative_to_repo(sources["repro_manifest"], root),
        },
        "v3_source_claim_count": len(v3_index.get("final_claims", [])),
    }


def _source_specs(sources: Mapping[str, Path]) -> tuple[dict[str, Any], ...]:
    specs = (
        ("v3_directory", "v3_dir", "generated_directory", "paper_candidate_v3", "evidence", "Preserved Phase 7D evidence package."),
        ("v3_artifact_index", "v3_index", "generated_manifest", "paper_candidate_v3", "evidence", "Phase 7D artifact and claim index."),
        ("v3_summary", "v3_summary", "generated_documentation", "paper_candidate_v3", "documentation", "Phase 7D package summary."),
        ("v3_final_claims", "v3_claims", "generated_table", "paper_candidate_v3", "evidence", "Phase 7D claims/evidence table."),
        ("v4_source", "v4_source", "source_code", "phase8d", "source", "Phase 8D packaging implementation."),
        ("v4_tests", "v4_tests", "test_code", "phase8d", "verification", "Focused Phase 8D package tests."),
        ("cli_source", "cli_source", "source_code", "software", "source", "CLI command definitions, including Phase 8D."),
        ("phase6_final_directory", "phase6_final_dir", "generated_directory", "phase6", "evidence", "Phase 6 final comparison artifacts."),
        ("phase6_known_ray_directory", "phase6_known_dir", "generated_directory", "phase6", "evidence", "Phase 6 known-ray diagnostic artifacts."),
        ("phase6_reference_ray_directory", "phase6_reference_dir", "generated_directory", "phase6", "evidence", "Phase 6 reference-ray artifacts."),
        ("phase7a_directory", "phase7a_dir", "generated_directory", "phase7a", "evidence", "Phase 7A ablation artifacts."),
        ("phase7b_directory", "phase7b_dir", "generated_directory", "phase7b", "evidence", "Phase 7B sensitivity artifacts."),
        ("phase7c_directory", "phase7c_dir", "generated_directory", "phase7c", "evidence", "Phase 7C learning-curve artifacts."),
        ("phase6_final_comparison", "phase6_summary", "generated_summary", "phase6", "evidence", "Phase 6 final comparison summary."),
        ("phase7a_ablation_summary", "phase7a_summary", "generated_summary", "phase7a", "evidence", "Phase 7A observation-signal ablation summary."),
        ("phase7b_sensitivity_summary", "phase7b_summary", "generated_summary", "phase7b", "evidence", "Phase 7B bounded reference-ray sensitivity summary."),
        ("phase7c_learning_curve_summary", "phase7c_summary", "generated_summary", "phase7c", "evidence", "Phase 7C fixed-holdout learning-curve summary."),
        ("phase8a_directory", "phase8a_dir", "generated_directory", "phase8a", "evidence", "Phase 8A paired fixed-test package."),
        ("phase8a_summary", "phase8a_summary", "generated_summary", "phase8a", "evidence", "Phase 8A method ranking and metric contract."),
        ("phase8a_method_summary", "phase8a_method_csv", "generated_table", "phase8a", "evidence", "Phase 8A per-method paired summary."),
        ("phase8a_case_metrics", "phase8a_case_csv", "generated_case_metrics", "phase8a", "evidence", "Phase 8A exact paired case metrics."),
        ("phase8a_family_summary", "phase8a_family_csv", "generated_table", "phase8a", "evidence", "Phase 8A family-wise paired summary."),
        ("phase8a_fixed_test_ids", "phase8a_ids", "generated_manifest", "phase8a", "evidence", "Phase 8A exact fixed test-case ID manifest."),
        ("phase8b_directory", "phase8b_dir", "generated_directory", "phase8b", "evidence", "Phase 8B uncertainty package."),
        ("phase8b_method_uncertainty", "phase8b_method_summary", "generated_summary", "phase8b", "evidence", "Phase 8B bootstrap method uncertainty summary."),
        ("phase8b_paired_delta", "phase8b_delta_summary", "generated_summary", "phase8b", "evidence", "Phase 8B paired delta and exploratory sign-flip summary."),
        ("phase8b_case_deltas", "phase8b_case_deltas", "generated_case_metrics", "phase8b", "evidence", "Phase 8B per-case method deltas."),
        ("reproducibility_manifest", "repro_manifest", "generated_manifest", "phase8c", "reproducibility", "Phase 8C provenance and artifact manifest."),
        ("reproducibility_commands", "commands", "documentation", "phase8c", "reproducibility", "Phase 8C dependency-ordered regeneration commands."),
        ("reproducibility_data_draft", "data_draft", "documentation", "phase8c", "reproducibility", "Phase 8C conservative data availability draft."),
        ("reproducibility_software_draft", "software_draft", "documentation", "phase8c", "reproducibility", "Phase 8C conservative software availability draft."),
        ("reproducibility_environment", "environment", "generated_summary", "phase8c", "reproducibility", "Phase 8C environment metadata."),
    )
    return tuple(
        {
            "artifact_id": artifact_id,
            "path": sources[source_key],
            "artifact_type": artifact_type,
            "phase": phase,
            "role": role,
            "description": description,
        }
        for artifact_id, source_key, artifact_type, phase, role, description in specs
    )


def _artifact_record(
    *,
    artifact_id: str,
    path: Path,
    root: Path,
    artifact_type: str,
    phase: str,
    role: str,
    description: str | None = None,
    source_artifact: str | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    record: dict[str, Any] = {
        "artifact_id": artifact_id,
        "path": _relative_to_repo(resolved, root),
        "artifact_type": artifact_type,
        "phase": phase,
        "role": role,
        "exists": resolved.exists(),
        "kind": "directory" if resolved.is_dir() else "file",
    }
    if description:
        record["description"] = description
    if source_artifact:
        record["source_artifact"] = source_artifact
    if resolved.is_file():
        record["size_bytes"] = resolved.stat().st_size
        record["sha256"] = _sha256(resolved)
    return record


def _find_claims_for_index(path: Path) -> list[dict[str, str]]:
    return _read_csv(path)


def _find_v3_claim(index: Mapping[str, Any], claim_id: str) -> Mapping[str, Any]:
    for claim in index.get("final_claims", []):
        if claim.get("claim_id") == claim_id:
            return claim
    raise KeyError(f"v3 artifact index has no claim {claim_id!r}.")


def _find_delta(rows: Sequence[Mapping[str, Any]], comparison_id: str) -> Mapping[str, Any]:
    for row in rows:
        if row.get("comparison_id") == comparison_id:
            return row
    raise KeyError(f"Phase 8B summary has no comparison {comparison_id!r}.")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fields = _ordered_fields(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: _format_cell(row.get(field, "")) for field in fields} for row in rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _ordered_fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    first_fields = list(rows[0].keys())
    remaining = sorted({key for row in rows for key in row} - set(first_fields))
    return first_fields + remaining


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot render an empty Markdown table.")
    fields = _ordered_fields(rows)
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_format_cell(row.get(field, "")) for field in fields) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_float(value: Any) -> str:
    return f"{float(value):.6f}"


def _read_json(path: Path) -> dict[str, Any]:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _relative_to_repo(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
