"""Package existing Phase 2-4B evidence for paper drafting."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.utils.paths import get_repo_root


PAPER_CANDIDATE_ID = "paper_candidate_v1"
MAIN_MODEL_NAME = "pca_linear_observation_regressor"
PHASE4B_INTERVENTION_ID = "directional_slowness_v1"


@dataclass(frozen=True)
class PaperCandidatePackageOutputs:
    """Paths produced by the paper-candidate packaging workflow."""

    output_dir: Path
    artifact_index_json: Path
    summary_markdown: Path
    experiment_note: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]


def package_paper_candidate_v1(
    settings: BenchmarkSettings | None = None,
) -> PaperCandidatePackageOutputs:
    """Create the final paper evidence package without rerunning training."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    package_dir = repo_root / loaded_settings.outputs.generated_base_dir / PAPER_CANDIDATE_ID
    tables_dir = package_dir / "tables"
    figures_dir = package_dir / "figures"
    notes_dir = package_dir / "notes"
    for directory in (tables_dir, figures_dir, notes_dir):
        directory.mkdir(parents=True, exist_ok=True)

    evidence = _load_evidence(repo_root)
    _validate_final_decision(evidence)

    table_paths = _write_tables(evidence, tables_dir, repo_root)
    figure_paths = _write_figures(evidence, figures_dir)
    package_note = _write_package_summary(
        evidence, notes_dir / "package_summary.md", table_paths, figure_paths, repo_root
    )
    experiment_note = _write_experiment_note(
        evidence,
        repo_root / "wiki" / "experiments" / f"{PAPER_CANDIDATE_ID}.md",
        table_paths,
        figure_paths,
        repo_root,
    )
    artifact_index = _build_artifact_index(
        evidence, table_paths, figure_paths, package_note, experiment_note, repo_root
    )

    index_path = package_dir / "artifact_index.json"
    index_path.write_text(json.dumps(artifact_index, indent=2) + "\n", encoding="utf-8")

    return PaperCandidatePackageOutputs(
        output_dir=package_dir,
        artifact_index_json=index_path,
        summary_markdown=package_note,
        experiment_note=experiment_note,
        table_paths=tuple(table_paths),
        figure_paths=tuple(figure_paths),
    )


def _load_evidence(repo_root: Path) -> dict[str, Any]:
    ml_suite_dir = (
        repo_root / "outputs" / "generated" / "ml_baselines" / "paper_pseudo_bending_ml_suite_v1"
    )
    strengthened_dir = (
        repo_root
        / "outputs"
        / "generated"
        / "ml_baselines"
        / "paper_pseudo_bending_ml_strengthened_v1"
    )
    faulted_dir = (
        repo_root
        / "outputs"
        / "generated"
        / "ml_baselines"
        / "paper_pseudo_bending_faulted_generalization_v1"
    )
    benchmark_dir = repo_root / "outputs" / "benchmarks" / "pseudo_bending_paper_benchmark_v1"
    corpus_manifest = (
        repo_root
        / "outputs"
        / "datasets"
        / "ml_supervised"
        / "paper_pseudo_bending_observation_pairs_v1"
        / "manifest.json"
    )

    paths = {
        "benchmark_summary": benchmark_dir / "summary.json",
        "benchmark_csv": benchmark_dir / "summary.csv",
        "corpus_manifest": corpus_manifest,
        "suite_summary": ml_suite_dir / "suite_summary.json",
        "repeated_split_summary": ml_suite_dir / "repeated_split_summary.json",
        "leave_one_family_out_summary": ml_suite_dir / "leave_one_family_out_summary.json",
        "noise_robustness_summary": ml_suite_dir / "noise_robustness_summary.json",
        "audit_summary": ml_suite_dir / "audit_summary.json",
        "strengthened_summary": strengthened_dir / "strengthened_summary.json",
        "faulted_generalization_summary": faulted_dir / "faulted_generalization_summary.json",
        "faulted_diagnosis": faulted_dir / "faulted_diagnosis.json",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(_relative_to_repo(path, repo_root) for path in missing)
        raise FileNotFoundError(
            f"Cannot package {PAPER_CANDIDATE_ID}; missing evidence:\n{missing_text}"
        )

    return {
        "paths": paths,
        "benchmark": _read_json(paths["benchmark_summary"]),
        "corpus": _read_json(paths["corpus_manifest"]),
        "suite": _read_json(paths["suite_summary"]),
        "repeated": _read_json(paths["repeated_split_summary"]),
        "loo": _read_json(paths["leave_one_family_out_summary"]),
        "noise": _read_json(paths["noise_robustness_summary"]),
        "audit": _read_json(paths["audit_summary"]),
        "strengthened": _read_json(paths["strengthened_summary"]),
        "faulted_generalization": _read_json(paths["faulted_generalization_summary"]),
        "faulted_diagnosis": _read_json(paths["faulted_diagnosis"]),
    }


def _validate_final_decision(evidence: dict[str, Any]) -> None:
    suite = evidence["suite"]
    faulted_generalization = evidence["faulted_generalization"]
    if suite["best_model"]["model_name"] != MAIN_MODEL_NAME:
        raise ValueError(
            f"Expected main model {MAIN_MODEL_NAME}; found {suite['best_model']['model_name']}."
        )
    recommendation = faulted_generalization["recommendation"]
    if recommendation["recommended_main_model"] != MAIN_MODEL_NAME:
        raise ValueError("Phase 4B recommendation does not preserve the Phase 4 linear main model.")
    if faulted_generalization["intervention"]["intervention_id"] != PHASE4B_INTERVENTION_ID:
        raise ValueError("Unexpected Phase 4B intervention identifier.")
    if not recommendation["proceed_to_phase_5"]:
        raise ValueError("Phase 4B evidence does not recommend proceeding to Phase 5.")


def _write_tables(evidence: dict[str, Any], tables_dir: Path, repo_root: Path) -> list[Path]:
    tables: list[Path] = []
    tables.append(
        _write_table_pair(
            tables_dir / "simulator_benchmark_summary",
            _benchmark_rows(evidence["benchmark"], repo_root),
        )
    )
    tables.append(
        _write_table_pair(
            tables_dir / "fixed_split_model_ranking",
            _fixed_split_rows(evidence["suite"]),
        )
    )
    tables.append(
        _write_table_pair(
            tables_dir / "repeated_split_stability",
            _repeated_split_rows(evidence["repeated"]),
        )
    )
    tables.append(
        _write_table_pair(
            tables_dir / "leave_one_family_out",
            _loo_rows(evidence["loo"], MAIN_MODEL_NAME),
        )
    )
    tables.append(
        _write_table_pair(
            tables_dir / "noise_robustness",
            _noise_rows(evidence["noise"], MAIN_MODEL_NAME),
        )
    )
    tables.append(
        _write_table_pair(
            tables_dir / "final_model_selection_and_limitations",
            _selection_rows(evidence),
        )
    )
    return [path for pair in tables for path in pair]


def _write_figures(evidence: dict[str, Any], figures_dir: Path) -> list[Path]:
    svg_paths = [
        figures_dir / "fixed_split_model_ranking.svg",
        figures_dir / "leave_one_family_out_main_model.svg",
        figures_dir / "noise_robustness_main_model.svg",
        figures_dir / "fixed_test_family_errors_main_model.svg",
    ]
    _bar_chart(
        _fixed_split_rows(evidence["suite"]),
        "model_name",
        "fixed_validation_rmse_km_per_s",
        svg_paths[0],
        "Fixed-split validation RMSE",
        "RMSE (km/s)",
    )
    _bar_chart(
        _loo_rows(evidence["loo"], MAIN_MODEL_NAME),
        "held_out_family",
        "mean_rmse_km_per_s",
        svg_paths[1],
        "Leave-one-family-out RMSE",
        "RMSE (km/s)",
    )
    _line_chart(
        _noise_rows(evidence["noise"], MAIN_MODEL_NAME),
        "noise_std_s",
        "test_rmse_km_per_s",
        svg_paths[2],
        "Noise robustness, main model",
        "Travel-time noise std (s)",
        "Test RMSE (km/s)",
    )
    _bar_chart(
        _fixed_test_family_rows(evidence["suite"], MAIN_MODEL_NAME),
        "family",
        "mean_rmse_km_per_s",
        svg_paths[3],
        "Fixed test RMSE by family",
        "RMSE (km/s)",
    )
    return [path for svg_path in svg_paths for path in (svg_path, svg_path.with_suffix(".png"))]


def _benchmark_rows(benchmark: dict[str, Any], repo_root: Path) -> list[dict[str, object]]:
    rows = []
    for row in benchmark["rows"]:
        rows.append(
            {
                "case_id": row["case_id"],
                "family": row["family"],
                "comparison_kind": row["comparison_kind"],
                "reference_method": row["reference_method"],
                "pseudo_bending_travel_time_s": row["pseudo_bending_travel_time_s"],
                "reference_travel_time_s": row["reference_travel_time_s"],
                "absolute_travel_time_difference_s": row["absolute_travel_time_difference_s"],
                "relative_travel_time_difference": row["relative_travel_time_difference"],
                "pseudo_bending_runtime_s": row["pseudo_bending_runtime_s"],
                "reference_runtime_s": row["reference_runtime_s"],
                "pseudo_bending_iteration_count": row["pseudo_bending_iteration_count"],
                "pseudo_bending_converged": row["pseudo_bending_converged"],
                "figure_file": _relative_to_repo(
                    _resolve_repo_path(row["figure_file"], repo_root), repo_root
                ),
            }
        )
    return rows


def _fixed_split_rows(suite: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "rank": row["rank"],
            "model_name": row["model_name"],
            "fixed_validation_rmse_km_per_s": row["fixed_validation_rmse"],
            "fixed_validation_mae_km_per_s": row["fixed_validation_mae"],
            "fixed_test_rmse_km_per_s": row["fixed_test_rmse"],
            "fixed_test_mae_km_per_s": row["fixed_test_mae"],
            "metrics_json": row["metrics_json"],
        }
        for row in suite["model_ranking"]
    ]


def _repeated_split_rows(repeated: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for record in repeated["model_records"]:
        metrics = record["aggregate_metrics"]
        rows.append(
            {
                "model_name": record["model_name"],
                "mean_validation_rmse_km_per_s": metrics["mean_validation_rmse"],
                "std_validation_rmse_km_per_s": metrics["std_validation_rmse"],
                "mean_test_rmse_km_per_s": metrics["mean_test_rmse"],
                "std_test_rmse_km_per_s": metrics["std_test_rmse"],
                "split_seeds": ";".join(str(seed) for seed in repeated["split_random_seeds"]),
            }
        )
    return sorted(rows, key=lambda row: float(row["mean_validation_rmse_km_per_s"]))


def _loo_rows(loo: dict[str, Any], model_name: str) -> list[dict[str, object]]:
    record = _model_record(loo["model_records"], model_name)
    rows = []
    for held_out in record["held_out_records"]:
        metrics = held_out["held_out"]
        rows.append(
            {
                "model_name": model_name,
                "held_out_family": held_out["held_out_family"],
                "train_case_count": held_out["train_case_count"],
                "held_out_case_count": metrics["case_count"],
                "mean_rmse_km_per_s": metrics["mean_rmse"],
                "mean_mae_km_per_s": metrics["mean_mae"],
            }
        )
    return sorted(rows, key=lambda row: float(row["mean_rmse_km_per_s"]), reverse=True)


def _noise_rows(noise: dict[str, Any], model_name: str) -> list[dict[str, object]]:
    record = _model_record(noise["model_records"], model_name)
    return [
        {
            "model_name": model_name,
            "noise_std_s": row["noise_std_s"],
            "validation_rmse_km_per_s": row["validation"]["mean_rmse"],
            "validation_mae_km_per_s": row["validation"]["mean_mae"],
            "test_rmse_km_per_s": row["test"]["mean_rmse"],
            "test_mae_km_per_s": row["test"]["mean_mae"],
        }
        for row in record["noise_level_records"]
    ]


def _fixed_test_family_rows(
    suite: dict[str, Any], model_name: str, repo_root: Path | None = None
) -> list[dict[str, object]]:
    ranking_record = next(row for row in suite["model_ranking"] if row["model_name"] == model_name)
    metrics_path = _resolve_repo_path(ranking_record["metrics_json"], repo_root or get_repo_root())
    metrics = _read_json(metrics_path)["split_metrics"]["test"]
    rows = [
        {
            "family": family,
            "case_count": family_metrics["case_count"],
            "mean_rmse_km_per_s": family_metrics["mean_rmse"],
            "mean_mae_km_per_s": family_metrics["mean_mae"],
        }
        for family, family_metrics in metrics["per_family_metrics"].items()
    ]
    return sorted(rows, key=lambda row: float(row["mean_rmse_km_per_s"]), reverse=True)


def _selection_rows(evidence: dict[str, Any]) -> list[dict[str, object]]:
    suite = evidence["suite"]
    strengthened = evidence["strengthened"]
    faulted = evidence["faulted_generalization"]
    comparison = faulted["before_after_comparison"]
    return [
        {
            "item": "Main model",
            "status": "selected",
            "evidence": MAIN_MODEL_NAME,
            "reason": "Best fixed validation RMSE and cleanest repeated-split behavior in Phase 4.",
        },
        {
            "item": "Fixed validation RMSE",
            "status": "reported",
            "evidence": _format_float(suite["best_model"]["fixed_validation_rmse"]),
            "reason": "Primary Phase 4 model-selection metric.",
        },
        {
            "item": "Fixed test RMSE",
            "status": "reported",
            "evidence": _format_float(suite["best_model"]["fixed_test_rmse"]),
            "reason": "Held-out fixed split estimate for the selected model.",
        },
        {
            "item": "Repeated validation RMSE",
            "status": "reported",
            "evidence": _format_float(
                _model_record(evidence["repeated"]["model_records"], MAIN_MODEL_NAME)[
                    "aggregate_metrics"
                ]["mean_validation_rmse"]
            ),
            "reason": "Seeded split-stability check.",
        },
        {
            "item": "Faulted leave-one-family-out RMSE",
            "status": "limitation",
            "evidence": _format_float(comparison["phase4_faulted_loo_rmse"]),
            "reason": "Weak family extrapolation; report honestly in paper.",
        },
        {
            "item": "Phase 4.5B strengthened model",
            "status": "secondary",
            "evidence": f"{strengthened['best_model']['model_name']}; faulted LOO RMSE {_format_float(comparison['phase45b_faulted_loo_rmse'])}",
            "reason": "Slight faulted LOO improvement but worse fixed/repeated metrics.",
        },
        {
            "item": "Directional-feature intervention",
            "status": "rejected as main",
            "evidence": f"{PHASE4B_INTERVENTION_ID}; faulted LOO RMSE {_format_float(comparison['directional_features_faulted_loo_rmse'])}",
            "reason": "Worse than the Phase 4 main model on the target faulted LOO criterion.",
        },
        {
            "item": "Validation boundary",
            "status": "limitation",
            "evidence": "synthetic-only",
            "reason": "No real-data validation or classical inversion comparison has been completed yet.",
        },
    ]


def _write_table_pair(base_path: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    if not rows:
        raise ValueError(f"No rows for table {base_path.name}.")
    csv_path = base_path.with_suffix(".csv")
    md_path = base_path.with_suffix(".md")
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(_markdown_table(rows), encoding="utf-8")
    return csv_path, md_path


def _markdown_table(rows: list[dict[str, object]]) -> str:
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row[header]) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _bar_chart(
    rows: list[dict[str, object]],
    label_key: str,
    value_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    labels = [
        str(row[label_key]).replace("_observation_regressor", "").replace("_", "\n") for row in rows
    ]
    values = [float(row[value_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(labels, values, color="#3B82A0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)
    for index, value in enumerate(values):
        ax.text(index, value, _format_float(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save_svg_and_png(fig, output_path)


def _line_chart(
    rows: list[dict[str, object]],
    x_key: str,
    y_key: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(xs, ys, marker="o", color="#2F7D5F", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for x, y in zip(xs, ys, strict=True):
        ax.text(x, y, _format_float(y), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save_svg_and_png(fig, output_path)


def _save_svg_and_png(fig: plt.Figure, svg_path: Path) -> None:
    fig.savefig(svg_path)
    fig.savefig(svg_path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def _write_package_summary(
    evidence: dict[str, Any],
    path: Path,
    table_paths: list[Path],
    figure_paths: list[Path],
    repo_root: Path,
) -> Path:
    best = evidence["suite"]["best_model"]
    lines = [
        "# Paper Candidate v1 Package Summary",
        "",
        f"Package ID: `{PAPER_CANDIDATE_ID}`",
        "",
        "## Final Recommendation",
        "",
        (
            f"Use `{MAIN_MODEL_NAME}` from `paper_pseudo_bending_ml_suite_v1` as the main "
            "machine-learning result. Treat Phase 4.5B and Phase 4B as secondary diagnostic "
            "experiments rather than main results."
        ),
        "",
        "## Key Metrics",
        "",
        f"- Fixed validation RMSE: `{_format_float(best['fixed_validation_rmse'])}` km/s",
        f"- Fixed test RMSE: `{_format_float(best['fixed_test_rmse'])}` km/s",
        f"- Repeated validation RMSE: `{_format_float(_model_record(evidence['repeated']['model_records'], MAIN_MODEL_NAME)['aggregate_metrics']['mean_validation_rmse'])}` km/s",
        f"- Faulted leave-one-family-out RMSE: `{_format_float(evidence['faulted_generalization']['before_after_comparison']['phase4_faulted_loo_rmse'])}` km/s",
        "",
        "## Generated Tables",
        "",
    ]
    lines.extend(f"- `{_relative_to_repo(path, repo_root)}`" for path in table_paths)
    lines.extend(["", "## Generated Figures", ""])
    lines.extend(f"- `{_relative_to_repo(path, repo_root)}`" for path in figure_paths)
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This package indexes existing generated evidence and does not rerun training.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_experiment_note(
    evidence: dict[str, Any],
    path: Path,
    table_paths: list[Path],
    figure_paths: list[Path],
    repo_root: Path,
) -> Path:
    best = evidence["suite"]["best_model"]
    comparison = evidence["faulted_generalization"]["before_after_comparison"]
    lines = [
        "# Experiment: Paper Candidate v1",
        "",
        f"Experiment ID: `{PAPER_CANDIDATE_ID}`",
        "",
        "## Objective",
        "",
        (
            "Package the validated pseudo-bending benchmark, observation corpus, ML suite, "
            "audit, bounded strengthening attempt, and faulted-generalization intervention "
            "for paper preparation."
        ),
        "",
        "## Final Simulator Choice",
        "",
        (
            "`pseudo_bending_3d` is the default paper simulator. The eikonal/Dijkstra "
            "implementation is retained as a benchmark and prototype comparison only."
        ),
        "",
        "## Final Corpus",
        "",
        (
            f"The final corpus is `{evidence['corpus']['corpus_id']}` with "
            f"`{evidence['corpus']['case_count']}` cases, `{evidence['corpus']['station_count']}` "
            f"stations, `{evidence['corpus']['earthquake_count']}` earthquakes per case, "
            f"and `{evidence['corpus']['input_simulation_method']}` observations."
        ),
        "",
        "## Final Selected Model",
        "",
        (
            f"The selected main model is `{MAIN_MODEL_NAME}`. Its fixed validation RMSE is "
            f"`{_format_float(best['fixed_validation_rmse'])}` km/s and fixed test RMSE is "
            f"`{_format_float(best['fixed_test_rmse'])}` km/s."
        ),
        "",
        "## Benchmark Evidence",
        "",
        (
            "Phase 2 benchmark artifacts provide analytic direct-ray checks for homogeneous "
            "and horizontally layered cases and Dijkstra/eikonal comparison cases for the "
            "configured synthetic geological families."
        ),
        "",
        "## ML Evidence",
        "",
        (
            f"The repeated-split mean validation RMSE for the selected model is "
            f"`{_format_float(_model_record(evidence['repeated']['model_records'], MAIN_MODEL_NAME)['aggregate_metrics']['mean_validation_rmse'])}` km/s. "
            f"The faulted leave-one-family-out RMSE is "
            f"`{_format_float(comparison['phase4_faulted_loo_rmse'])}` km/s."
        ),
        "",
        "## Rejected or Secondary Strengthening Attempts",
        "",
        (
            f"Phase 4.5B improved the faulted leave-one-family-out RMSE to "
            f"`{_format_float(comparison['phase45b_faulted_loo_rmse'])}` km/s, but it worsened "
            "fixed and repeated-split metrics. It is therefore secondary evidence."
        ),
        (
            f"The `{PHASE4B_INTERVENTION_ID}` feature intervention produced faulted "
            f"leave-one-family-out RMSE `{_format_float(comparison['directional_features_faulted_loo_rmse'])}` km/s, "
            "which is worse than Phase 4. It is recorded as a rejected main-result path."
        ),
        "",
        "## Key Limitations",
        "",
        "- Validation is synthetic-only at this stage.",
        "- Weak `faulted` leave-one-family-out generalization indicates limited family extrapolation.",
        "- The PCA target representation and configured geological families remain modeling assumptions.",
        "- No completed real-data validation or classical inversion comparison is claimed.",
        "",
        "## Recommended Usage",
        "",
        (
            "Use the generated tables and figures to support the Methods and Results chapters. "
            "Frame the ML result as supervised reconstruction within a controlled synthetic "
            "pseudo-bending corpus, with explicit limitations for unseen geological families."
        ),
        "",
        "## Recommended Paper Framing",
        "",
        (
            "Frame the paper as a reproducible synthetic-data pipeline and baseline ML inversion "
            "study. Present the Phase 4 linear model as the main clean result, and discuss Phase "
            "4.5B/4B as transparent negative or secondary evidence."
        ),
        "",
        "## Generated Tables",
        "",
    ]
    lines.extend(f"- `{_relative_to_repo(path, repo_root)}`" for path in table_paths)
    lines.extend(["", "## Generated Figures", ""])
    lines.extend(f"- `{_relative_to_repo(path, repo_root)}`" for path in figure_paths)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_artifact_index(
    evidence: dict[str, Any],
    table_paths: list[Path],
    figure_paths: list[Path],
    package_note: Path,
    experiment_note: Path,
    repo_root: Path,
) -> dict[str, Any]:
    benchmark_figures = [
        _relative_to_repo(Path(path), repo_root) for path in evidence["benchmark"]["figure_paths"]
    ]
    benchmark_ray_paths = sorted(
        _relative_to_repo(path, repo_root)
        for path in (
            repo_root / "outputs" / "benchmarks" / "pseudo_bending_paper_benchmark_v1" / "ray_paths"
        ).glob("*.json")
    )
    paths = evidence["paths"]
    return {
        "artifact_type": "paper_candidate_artifact_index",
        "package_id": PAPER_CANDIDATE_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "does_not_rerun_training": True,
        "final_selected_model": {
            "model_name": MAIN_MODEL_NAME,
            "source_experiment_id": "paper_pseudo_bending_ml_suite_v1",
            "selection_basis": "Phase 4 fixed validation RMSE with repeated-split stability check.",
            "metrics": {
                "fixed_validation_rmse_km_per_s": evidence["suite"]["best_model"][
                    "fixed_validation_rmse"
                ],
                "fixed_test_rmse_km_per_s": evidence["suite"]["best_model"]["fixed_test_rmse"],
                "repeated_validation_rmse_km_per_s": _model_record(
                    evidence["repeated"]["model_records"], MAIN_MODEL_NAME
                )["aggregate_metrics"]["mean_validation_rmse"],
                "faulted_leave_one_family_out_rmse_km_per_s": evidence["faulted_generalization"][
                    "before_after_comparison"
                ]["phase4_faulted_loo_rmse"],
            },
        },
        "simulator": {
            "default_for_benchmark": "pseudo_bending_3d",
            "benchmark_or_prototype_only": [
                "eikonal_3d_first_arrival_prototype",
                "Dijkstra graph shortest-path comparisons",
            ],
        },
        "source_evidence": {
            "phase_2_pseudo_bending_benchmark": {
                "summary_json": _relative_to_repo(paths["benchmark_summary"], repo_root),
                "summary_csv": _relative_to_repo(paths["benchmark_csv"], repo_root),
                "ray_path_json_files": benchmark_ray_paths,
                "figures": benchmark_figures,
            },
            "phase_3_pseudo_bending_corpus": {
                "manifest_json": _relative_to_repo(paths["corpus_manifest"], repo_root),
                "corpus_id": evidence["corpus"]["corpus_id"],
                "case_count": evidence["corpus"]["case_count"],
            },
            "phase_4_ml_suite": {
                "suite_summary_json": _relative_to_repo(paths["suite_summary"], repo_root),
                "repeated_split_summary_json": _relative_to_repo(
                    paths["repeated_split_summary"], repo_root
                ),
                "leave_one_family_out_summary_json": _relative_to_repo(
                    paths["leave_one_family_out_summary"], repo_root
                ),
                "noise_robustness_summary_json": _relative_to_repo(
                    paths["noise_robustness_summary"], repo_root
                ),
            },
            "phase_4_5_audit": {
                "audit_summary_json": _relative_to_repo(paths["audit_summary"], repo_root),
            },
            "phase_4_5b_strengthened": {
                "strengthened_summary_json": _relative_to_repo(
                    paths["strengthened_summary"], repo_root
                ),
                "status": "secondary_not_main",
            },
            "phase_4b_faulted_generalization": {
                "faulted_generalization_summary_json": _relative_to_repo(
                    paths["faulted_generalization_summary"], repo_root
                ),
                "faulted_diagnosis_json": _relative_to_repo(paths["faulted_diagnosis"], repo_root),
                "directional_feature_intervention_status": "rejected_as_main_result",
            },
        },
        "generated_tables": [_relative_to_repo(path, repo_root) for path in table_paths],
        "generated_figures": [_relative_to_repo(path, repo_root) for path in figure_paths],
        "paper_notes": [
            _relative_to_repo(package_note, repo_root),
            _relative_to_repo(experiment_note, repo_root),
        ],
        "limitations_to_report": [
            "Synthetic-only validation.",
            "Weak faulted leave-one-family-out generalization.",
            "No real-data validation or classical inversion comparison claimed.",
        ],
        "rejected_or_secondary_model_experiments": [
            {
                "experiment_id": "paper_pseudo_bending_ml_strengthened_v1",
                "status": "secondary_not_main",
                "reason": "Slight faulted LOO improvement but worsened other metrics.",
            },
            {
                "experiment_id": "paper_pseudo_bending_faulted_generalization_v1",
                "intervention_id": PHASE4B_INTERVENTION_ID,
                "status": "rejected_as_main_result",
                "reason": "Directional features worsened faulted LOO relative to Phase 4.",
            },
        ],
    }


def _model_record(records: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    return next(record for record in records if record["model_name"] == model_name)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _cell(value: object) -> str:
    if isinstance(value, float):
        return _format_float(value)
    return str(value).replace("|", "\\|")


def _format_float(value: float) -> str:
    return f"{value:.6f}"
