"""Package final Phase 6 evidence for paper drafting."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.domain.schemas import CartesianVelocityGrid3D
from tomobench.tomography.comparison import node_centered_values_to_cell_centered
from tomobench.utils.paths import get_repo_root

PAPER_CANDIDATE_V2_ID = "paper_candidate_v2"
MAIN_MODEL_NAME = "pca_linear_observation_regressor"
REPRESENTATIVE_FAMILIES = (
    "layered",
    "block_anomaly",
    "faulted",
    "dyke_intrusion",
    "salt_dome",
)


@dataclass(frozen=True)
class PaperCandidateV2PackageOutputs:
    """Paths produced by the Phase 6-ready evidence package."""

    output_dir: Path
    artifact_index_json: Path
    summary_markdown: Path
    experiment_note: Path
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    figure_metadata_json: Path


def package_paper_candidate_v2(
    settings: BenchmarkSettings | None = None,
) -> PaperCandidateV2PackageOutputs:
    """Create a Phase 6-ready paper evidence package without rerunning training."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    package_dir = repo_root / loaded_settings.outputs.generated_base_dir / PAPER_CANDIDATE_V2_ID
    tables_dir = package_dir / "tables"
    figures_dir = package_dir / "figures"
    notes_dir = package_dir / "notes"
    for directory in (tables_dir, figures_dir, notes_dir):
        directory.mkdir(parents=True, exist_ok=True)

    evidence = load_phase6_packaging_evidence(repo_root)
    audit_phase6_metric_contract(evidence)
    table_paths = _write_tables(evidence, tables_dir)
    figure_paths, figure_metadata_path = _write_visual_comparison_figures(
        evidence,
        figures_dir,
        repo_root,
    )
    summary_note = _write_package_summary(
        evidence,
        notes_dir / "package_summary.md",
        table_paths,
        figure_paths,
        repo_root,
    )
    experiment_note = _write_experiment_note(
        evidence,
        repo_root / "wiki" / "experiments" / f"{PAPER_CANDIDATE_V2_ID}.md",
        table_paths,
        figure_paths,
        figure_metadata_path,
        repo_root,
    )
    artifact_index = _build_artifact_index(
        evidence,
        table_paths,
        figure_paths,
        figure_metadata_path,
        summary_note,
        experiment_note,
        repo_root,
    )
    artifact_index_json = package_dir / "artifact_index.json"
    _write_json(artifact_index_json, artifact_index)
    return PaperCandidateV2PackageOutputs(
        output_dir=package_dir,
        artifact_index_json=artifact_index_json,
        summary_markdown=summary_note,
        experiment_note=experiment_note,
        table_paths=tuple(table_paths),
        figure_paths=tuple(figure_paths),
        figure_metadata_json=figure_metadata_path,
    )


def load_phase6_packaging_evidence(repo_root: Path | None = None) -> dict[str, Any]:
    """Load Phase 6 comparison and source evidence needed by the v2 package."""
    root = repo_root or get_repo_root()
    final_comparison_dir = (
        root / "outputs" / "generated" / "classical_tomography" / "final_phase6_comparison_v1"
    )
    known_dir = root / "outputs" / "generated" / "classical_tomography" / "known_ray_corpus_v1"
    reference_dir = (
        root / "outputs" / "generated" / "classical_tomography" / "reference_ray_corpus_v1"
    )
    ml_suite_dir = (
        root / "outputs" / "generated" / "ml_baselines" / "paper_pseudo_bending_ml_suite_v1"
    )
    corpus_manifest = (
        root
        / "outputs"
        / "datasets"
        / "ml_supervised"
        / "paper_pseudo_bending_observation_pairs_v1"
        / "manifest.json"
    )
    paper_v1_dir = root / "outputs" / "generated" / "paper_candidate_v1"
    paths = {
        "final_comparison_summary": final_comparison_dir / "final_phase6_comparison_summary.json",
        "final_comparison_summary_csv": final_comparison_dir
        / "final_phase6_comparison_summary.csv",
        "ml_cell_case_metrics": final_comparison_dir
        / "ml_pca_linear_cell_centered_case_metrics.csv",
        "classical_clipping_diagnostics": final_comparison_dir
        / "classical_clipping_diagnostics.csv",
        "known_case_summary": known_dir / "known_ray_corpus_case_summary.csv",
        "reference_case_summary": reference_dir / "reference_ray_corpus_case_summary.csv",
        "suite_summary": ml_suite_dir / "suite_summary.json",
        "leave_one_family_out_summary": ml_suite_dir / "leave_one_family_out_summary.json",
        "corpus_manifest": corpus_manifest,
        "paper_candidate_v1_index": paper_v1_dir / "artifact_index.json",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(_relative_to_repo(path, root) for path in missing)
        raise FileNotFoundError(f"Cannot package {PAPER_CANDIDATE_V2_ID}; missing:\n{missing_text}")
    return {
        "paths": paths,
        "final_comparison": _read_json(paths["final_comparison_summary"]),
        "ml_case_rows": _read_csv_rows(paths["ml_cell_case_metrics"]),
        "clipping_rows": _read_csv_rows(paths["classical_clipping_diagnostics"]),
        "known_case_rows": _read_csv_rows(paths["known_case_summary"]),
        "reference_case_rows": _read_csv_rows(paths["reference_case_summary"]),
        "suite": _read_json(paths["suite_summary"]),
        "loo": _read_json(paths["leave_one_family_out_summary"]),
        "corpus": _read_json(paths["corpus_manifest"]),
        "paper_candidate_v1": _read_json(paths["paper_candidate_v1_index"]),
    }


def audit_phase6_metric_contract(evidence: dict[str, Any]) -> None:
    """Fail fast if the final comparison mixes metric contracts ambiguously."""
    final = evidence["final_comparison"]
    ml = final["ml_pca_linear_cell_centered"]
    if ml["sample_count"] != 100:
        raise ValueError("Expected all-case ML cell-centered metrics for 100 cases.")
    if set(ml["splits"]) != {"test", "train", "validation"}:
        raise ValueError("Expected ML cell-centered metrics to retain fixed split labels.")
    if ml["split_wise_metrics"]["test"]["all_cell_velocity_rmse_km_per_s"]["count"] <= 0:
        raise ValueError("Expected fixed-test-only ML cell-centered metrics.")
    if final["known_ray_classical"]["sample_count"] != 100:
        raise ValueError("Expected 100 known-ray classical cases.")
    if final["reference_ray_classical"]["sample_count"] != 100:
        raise ValueError("Expected 100 reference-ray classical cases.")
    if "leave_one_family_out_summary" not in evidence["paths"]:
        raise ValueError("Leave-one-family-out ML evidence is required.")
    contract = final["metric_contract"]
    if "cell-centered" not in contract["ml_cell_centered_metrics"]:
        raise ValueError("ML cell-centered contract is not explicit.")
    if "node-centered" not in contract["ml_original_metrics"]:
        raise ValueError("Original ML node-centered metric note is not explicit.")
    cautions = " ".join(final["comparison_caution"] + final["audit_findings"]).lower()
    for required in ("known-ray", "reference-ray", "clipping", "synthetic"):
        if required not in cautions:
            raise ValueError(f"Final comparison caveats do not mention {required}.")


def load_case_visual_comparison(
    case_id: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Load cell-centered true, ML, known-ray, and reference-ray arrays for one case."""
    ml_row = _row_by_case(evidence["ml_case_rows"], case_id)
    known_row = _row_by_case(evidence["known_case_rows"], case_id)
    reference_row = _row_by_case(evidence["reference_case_rows"], case_id)
    target_grid = _load_velocity_grid(Path(str(ml_row["target_grid_path"])))
    prediction = _read_json(Path(str(ml_row["prediction_path"])))
    nx = len(target_grid.x_coordinates_km)
    ny = len(target_grid.y_coordinates_km)
    nz = len(target_grid.z_coordinates_km)
    cell_shape = (nx - 1, ny - 1, nz - 1)
    true_cells = node_centered_values_to_cell_centered(
        target_grid.p_velocity_km_per_s,
        nx,
        ny,
        nz,
    )
    ml_cells = node_centered_values_to_cell_centered(
        tuple(float(value) for value in prediction["predicted_p_velocity_km_per_s"]),
        nx,
        ny,
        nz,
    )
    known_cells = _read_cell_prediction_values(
        Path(str(known_row["summary_json"])).parent
        / "known_ray_coverage_smoothing_cell_velocity_predictions.csv"
    )
    reference_cells = _read_cell_prediction_values(
        Path(str(reference_row["summary_json"])).parent / "reference_ray_cell_velocity_predictions.csv"
    )
    expected_cell_count = cell_shape[0] * cell_shape[1] * cell_shape[2]
    for name, values in (
        ("true", true_cells),
        ("ml", ml_cells),
        ("known_ray", known_cells),
        ("reference_ray", reference_cells),
    ):
        if len(values) != expected_cell_count:
            raise ValueError(f"{case_id} {name} cell count does not match grid shape.")
    return {
        "case_id": case_id,
        "family": ml_row["family"],
        "split": ml_row["split"],
        "cell_shape": cell_shape,
        "true_cells": true_cells,
        "ml_cells": ml_cells,
        "known_ray_cells": known_cells,
        "reference_ray_cells": reference_cells,
        "metrics": {
            "ml_all_cell_rmse": float(ml_row["all_cell_velocity_rmse_km_per_s"]),
            "ml_reference_covered_rmse": float(
                ml_row["reference_covered_cell_velocity_rmse_km_per_s"]
            ),
            "known_all_cell_rmse": float(known_row["all_cell_velocity_rmse_km_per_s"]),
            "known_covered_cell_rmse": float(known_row["covered_cell_velocity_rmse_km_per_s"]),
            "reference_all_cell_rmse": float(reference_row["all_cell_velocity_rmse_km_per_s"]),
            "reference_covered_cell_rmse": float(
                reference_row["covered_cell_velocity_rmse_km_per_s"]
            ),
            "reference_clipped_cell_count": int(float(reference_row["clipped_velocity_cell_count"])),
        },
        "target_grid_path": str(ml_row["target_grid_path"]),
        "prediction_path": str(ml_row["prediction_path"]),
    }


def _write_tables(evidence: dict[str, Any], tables_dir: Path) -> list[Path]:
    table_specs = (
        ("phase6_metric_contract_summary", _phase6_metric_rows(evidence["final_comparison"])),
        ("ml_cell_centered_split_summary", _ml_split_rows(evidence["final_comparison"])),
        ("classical_clipping_summary", _clipping_rows(evidence["final_comparison"])),
        ("leave_one_family_out_main_model", _loo_rows(evidence["loo"])),
    )
    paths: list[Path] = []
    for name, rows in table_specs:
        paths.extend(_write_table_pair(tables_dir / name, rows))
    return paths


def _write_visual_comparison_figures(
    evidence: dict[str, Any],
    figures_dir: Path,
    repo_root: Path,
) -> tuple[list[Path], Path]:
    selected_rows = select_representative_ml_cases(evidence["ml_case_rows"])
    metadata_cases = []
    figure_paths: list[Path] = []
    for row in selected_rows:
        case = load_case_visual_comparison(str(row["case_id"]), evidence)
        slice_index = case["cell_shape"][2] // 2
        basename = f"{case['case_id']}_cell_slice_z{slice_index:02d}_comparison"
        svg_path = figures_dir / f"{basename}.svg"
        _plot_case_comparison(case, slice_index, svg_path)
        png_path = svg_path.with_suffix(".png")
        figure_paths.extend((svg_path, png_path))
        metadata_cases.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "split": case["split"],
                "slice_axis": "z",
                "slice_index": slice_index,
                "cell_shape": {
                    "nx": case["cell_shape"][0],
                    "ny": case["cell_shape"][1],
                    "nz": case["cell_shape"][2],
                },
                "metrics": case["metrics"],
                "figure_svg": _relative_to_repo(svg_path, repo_root),
                "figure_png": _relative_to_repo(png_path, repo_root),
                "target_grid_path": _relative_to_repo(Path(case["target_grid_path"]), repo_root),
                "ml_prediction_path": _relative_to_repo(Path(case["prediction_path"]), repo_root),
            }
        )
    metadata = {
        "artifact_type": "paper_candidate_v2_visual_comparison_metadata",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection_rule": "fixed-test split median all-cell ML cell-centered RMSE per family where available",
        "velocity_contract": "all panels use cell-centered velocities; node-centered true and ML grids are averaged over each cell's eight nodes",
        "families": [row["family"] for row in selected_rows],
        "cases": metadata_cases,
    }
    metadata_path = figures_dir / "visual_comparison_metadata.json"
    _write_json(metadata_path, metadata)
    return figure_paths, metadata_path


def select_representative_ml_cases(
    ml_case_rows: list[dict[str, Any]],
    families: Sequence[str] = REPRESENTATIVE_FAMILIES,
) -> list[dict[str, Any]]:
    """Select median-error fixed-test cases per family, falling back to all splits."""
    selected = []
    for family in families:
        family_rows = [
            row
            for row in ml_case_rows
            if row["family"] == family and row.get("split") == "test"
        ]
        if not family_rows:
            family_rows = [row for row in ml_case_rows if row["family"] == family]
        if not family_rows:
            raise ValueError(f"No ML case rows found for family {family}.")
        ordered = sorted(
            family_rows,
            key=lambda row: float(row["all_cell_velocity_rmse_km_per_s"]),
        )
        selected.append(ordered[len(ordered) // 2])
    return selected


def _plot_case_comparison(case: dict[str, Any], slice_index: int, svg_path: Path) -> None:
    true_slice = _cell_slice(case["true_cells"], case["cell_shape"], slice_index)
    ml_slice = _cell_slice(case["ml_cells"], case["cell_shape"], slice_index)
    reference_slice = _cell_slice(case["reference_ray_cells"], case["cell_shape"], slice_index)
    known_slice = _cell_slice(case["known_ray_cells"], case["cell_shape"], slice_index)
    velocity_slices = (true_slice, ml_slice, reference_slice, known_slice)
    vmin = min(min(row) for grid in velocity_slices for row in grid)
    vmax = max(max(row) for grid in velocity_slices for row in grid)
    error_slices = (
        _subtract_grid(ml_slice, true_slice),
        _subtract_grid(reference_slice, true_slice),
        _subtract_grid(known_slice, true_slice),
    )
    error_max = max(abs(value) for grid in error_slices for row in grid for value in row)
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.6), constrained_layout=True)
    fig.suptitle(f"{case['case_id']} ({case['family']}, {case['split']}), z-cell {slice_index}")
    panels = (
        ("True", true_slice, "velocity"),
        ("PCA-linear ML", ml_slice, "velocity"),
        ("Reference-ray", reference_slice, "velocity"),
        ("Known-ray", known_slice, "velocity"),
        ("", [[0.0]], "blank"),
        ("ML error", error_slices[0], "error"),
        ("Reference error", error_slices[1], "error"),
        ("Known-ray error", error_slices[2], "error"),
    )
    velocity_image = None
    error_image = None
    for ax, (title, grid, kind) in zip(axes.ravel(), panels, strict=True):
        if kind == "blank":
            ax.axis("off")
            continue
        if kind == "velocity":
            image = ax.imshow(grid, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
            velocity_image = image
        else:
            image = ax.imshow(
                grid,
                origin="lower",
                vmin=-error_max,
                vmax=error_max,
                cmap="coolwarm",
            )
            error_image = image
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    if velocity_image is not None:
        fig.colorbar(velocity_image, ax=axes[0, :], orientation="horizontal", fraction=0.05)
    if error_image is not None:
        fig.colorbar(error_image, ax=axes[1, 1:], orientation="horizontal", fraction=0.05)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path)
    fig.savefig(svg_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _phase6_metric_rows(final: dict[str, Any]) -> list[dict[str, object]]:
    ml = final["ml_pca_linear_cell_centered"]
    return [
        {
            "method": "known_ray_classical",
            "case_scope": "all_100_cases",
            "metric_contract": "cell_centered_classical_known_true_rays",
            "all_cell_rmse_km_per_s": final["known_ray_classical"][
                "all_cell_velocity_rmse_km_per_s"
            ]["mean"],
            "covered_or_compatible_rmse_km_per_s": final["known_ray_classical"][
                "covered_cell_velocity_rmse_km_per_s"
            ]["mean"],
            "clipped_cell_percentage_mean": final["clipping_diagnostics"]["known_ray"][
                "clipped_cell_percentage"
            ]["mean"],
            "limitation": "optimistic because ray geometry uses the true synthetic model",
        },
        {
            "method": "reference_ray_classical",
            "case_scope": "all_100_cases",
            "metric_contract": "cell_centered_classical_configured_layered_reference_rays",
            "all_cell_rmse_km_per_s": final["reference_ray_classical"][
                "all_cell_velocity_rmse_km_per_s"
            ]["mean"],
            "covered_or_compatible_rmse_km_per_s": final["reference_ray_classical"][
                "covered_cell_velocity_rmse_km_per_s"
            ]["mean"],
            "clipped_cell_percentage_mean": final["clipping_diagnostics"]["reference_ray"][
                "clipped_cell_percentage"
            ]["mean"],
            "limitation": "fixed-ray regularized inversion with high clipping",
        },
        {
            "method": MAIN_MODEL_NAME,
            "case_scope": "all_100_cases_including_train_validation_test",
            "metric_contract": "node_predictions_converted_to_cell_centered",
            "all_cell_rmse_km_per_s": ml["all_splits"]["all_cell_velocity_rmse_km_per_s"][
                "mean"
            ],
            "covered_or_compatible_rmse_km_per_s": ml["all_splits"][
                "reference_covered_cell_velocity_rmse_km_per_s"
            ]["mean"],
            "clipped_cell_percentage_mean": "",
            "limitation": "includes training cases; use fixed-test and LOO rows for generalization discussion",
        },
        {
            "method": MAIN_MODEL_NAME,
            "case_scope": "fixed_test_only",
            "metric_contract": "node_predictions_converted_to_cell_centered",
            "all_cell_rmse_km_per_s": ml["split_wise_metrics"]["test"][
                "all_cell_velocity_rmse_km_per_s"
            ]["mean"],
            "covered_or_compatible_rmse_km_per_s": ml["split_wise_metrics"]["test"][
                "reference_covered_cell_velocity_rmse_km_per_s"
            ]["mean"],
            "clipped_cell_percentage_mean": "",
            "limitation": "fixed family-stratified test split only",
        },
    ]


def _ml_split_rows(final: dict[str, Any]) -> list[dict[str, object]]:
    split_wise = final["ml_pca_linear_cell_centered"]["split_wise_metrics"]
    return [
        {
            "split": split,
            "case_count": metrics["all_cell_velocity_rmse_km_per_s"]["count"],
            "all_cell_rmse_km_per_s": metrics["all_cell_velocity_rmse_km_per_s"]["mean"],
            "all_cell_mae_km_per_s": metrics["all_cell_velocity_mae_km_per_s"]["mean"],
            "reference_covered_compatible_rmse_km_per_s": metrics[
                "reference_covered_cell_velocity_rmse_km_per_s"
            ]["mean"],
            "original_node_rmse_km_per_s": metrics["node_centered_original_rmse_km_per_s"][
                "mean"
            ],
        }
        for split, metrics in sorted(split_wise.items())
    ]


def _clipping_rows(final: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for method, summary in final["clipping_diagnostics"].items():
        rows.append(
            {
                "method": method,
                "family": "all",
                "case_count": summary["sample_count"],
                "clipped_cell_count_mean": summary["clipped_cell_count"]["mean"],
                "clipped_cell_percentage_mean": summary["clipped_cell_percentage"]["mean"],
                "covered_clipped_cell_count_mean": summary["covered_clipped_cell_count"]["mean"],
                "uncovered_clipped_cell_count_mean": summary["uncovered_clipped_cell_count"][
                    "mean"
                ],
            }
        )
        for family, family_summary in summary["family_wise"].items():
            rows.append(
                {
                    "method": method,
                    "family": family,
                    "case_count": family_summary["sample_count"],
                    "clipped_cell_count_mean": family_summary["clipped_cell_count"]["mean"],
                    "clipped_cell_percentage_mean": family_summary["clipped_cell_percentage"][
                        "mean"
                    ],
                    "covered_clipped_cell_count_mean": family_summary[
                        "covered_clipped_cell_count"
                    ]["mean"],
                    "uncovered_clipped_cell_count_mean": family_summary[
                        "uncovered_clipped_cell_count"
                    ]["mean"],
                }
            )
    return rows


def _loo_rows(loo: dict[str, Any]) -> list[dict[str, object]]:
    record = next(row for row in loo["model_records"] if row["model_name"] == MAIN_MODEL_NAME)
    return [
        {
            "model_name": MAIN_MODEL_NAME,
            "held_out_family": item["held_out_family"],
            "train_case_count": item["train_case_count"],
            "held_out_case_count": item["held_out"]["case_count"],
            "node_centered_mean_rmse_km_per_s": item["held_out"]["mean_rmse"],
            "node_centered_mean_mae_km_per_s": item["held_out"]["mean_mae"],
            "metric_contract": "node_centered_original_loo_not_cell_centered",
        }
        for item in record["held_out_records"]
    ]


def _write_package_summary(
    evidence: dict[str, Any],
    path: Path,
    table_paths: list[Path],
    figure_paths: list[Path],
    repo_root: Path,
) -> Path:
    final = evidence["final_comparison"]
    ml = final["ml_pca_linear_cell_centered"]
    lines = [
        "# Paper Candidate v2 Package Summary",
        "",
        f"Package ID: `{PAPER_CANDIDATE_V2_ID}`",
        "",
        "## Scope",
        "",
        (
            "This package freezes the Phase 6 technical evidence for paper writing. "
            "It preserves `paper_candidate_v1` as the pre-classical-baseline package and adds "
            "known-ray, reference-ray, harmonized ML cell-centered metrics, clipping diagnostics, "
            "and visual comparison figures."
        ),
        "",
        "## Key Phase 6 Metrics",
        "",
        f"- Known-ray all-cell RMSE: `{_format_float(final['known_ray_classical']['all_cell_velocity_rmse_km_per_s']['mean'])}` km/s",
        f"- Known-ray covered-cell RMSE: `{_format_float(final['known_ray_classical']['covered_cell_velocity_rmse_km_per_s']['mean'])}` km/s",
        f"- Reference-ray all-cell RMSE: `{_format_float(final['reference_ray_classical']['all_cell_velocity_rmse_km_per_s']['mean'])}` km/s",
        f"- Reference-ray covered-cell RMSE: `{_format_float(final['reference_ray_classical']['covered_cell_velocity_rmse_km_per_s']['mean'])}` km/s",
        f"- PCA-linear ML all-100-case cell-centered RMSE: `{_format_float(ml['all_splits']['all_cell_velocity_rmse_km_per_s']['mean'])}` km/s",
        f"- PCA-linear ML fixed-test cell-centered RMSE: `{_format_float(ml['split_wise_metrics']['test']['all_cell_velocity_rmse_km_per_s']['mean'])}` km/s",
        "",
        "## Required Caveats",
        "",
        "- The benchmark is synthetic-only and does not validate performance on real data.",
        "- Known-ray inversion is an optimistic diagnostic because it uses true-model ray geometry.",
        "- Reference-ray inversion is a fixed-ray regularized classical baseline, not full nonlinear iterative tomography.",
        "- Reference-ray velocity clipping is high and must be reported as a limitation.",
        "- All-100-case ML cell-centered metrics include training cases; use fixed-test and leave-one-family-out evidence when discussing generalization.",
        "",
        "## Generated Tables",
        "",
    ]
    lines.extend(f"- `{_relative_to_repo(item, repo_root)}`" for item in table_paths)
    lines.extend(["", "## Generated Figures", ""])
    lines.extend(f"- `{_relative_to_repo(item, repo_root)}`" for item in figure_paths)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_experiment_note(
    evidence: dict[str, Any],
    path: Path,
    table_paths: list[Path],
    figure_paths: list[Path],
    figure_metadata_path: Path,
    repo_root: Path,
) -> Path:
    final = evidence["final_comparison"]
    lines = [
        "# Experiment: Paper Candidate v2",
        "",
        f"Experiment ID: `{PAPER_CANDIDATE_V2_ID}`",
        "",
        "## Objective",
        "",
        (
            "Freeze and package the final technical evidence after Phase 6 classical tomography "
            "baselines, including harmonized classical-vs-ML metrics and representative visual "
            "comparison figures."
        ),
        "",
        "## Inputs",
        "",
        f"- Final Phase 6 comparison: `{_relative_to_repo(evidence['paths']['final_comparison_summary'], repo_root)}`",
        f"- ML suite summary: `{_relative_to_repo(evidence['paths']['suite_summary'], repo_root)}`",
        f"- Leave-one-family-out summary: `{_relative_to_repo(evidence['paths']['leave_one_family_out_summary'], repo_root)}`",
        f"- Paper corpus manifest: `{_relative_to_repo(evidence['paths']['corpus_manifest'], repo_root)}`",
        f"- Historical v1 package index: `{_relative_to_repo(evidence['paths']['paper_candidate_v1_index'], repo_root)}`",
        "",
        "## Outputs",
        "",
        f"- Artifact index: `outputs/generated/{PAPER_CANDIDATE_V2_ID}/artifact_index.json`",
        f"- Package summary: `outputs/generated/{PAPER_CANDIDATE_V2_ID}/notes/package_summary.md`",
        f"- Figure metadata: `{_relative_to_repo(figure_metadata_path, repo_root)}`",
        "",
        "## Main Metrics",
        "",
        (
            f"Known-ray classical all-cell RMSE is "
            f"`{_format_float(final['known_ray_classical']['all_cell_velocity_rmse_km_per_s']['mean'])}` km/s, "
            f"and reference-ray classical all-cell RMSE is "
            f"`{_format_float(final['reference_ray_classical']['all_cell_velocity_rmse_km_per_s']['mean'])}` km/s."
        ),
        (
            f"The selected PCA-linear ML model, after node-to-cell conversion, has all-100-case "
            f"cell-centered RMSE `{_format_float(final['ml_pca_linear_cell_centered']['all_splits']['all_cell_velocity_rmse_km_per_s']['mean'])}` km/s "
            f"and fixed-test cell-centered RMSE "
            f"`{_format_float(final['ml_pca_linear_cell_centered']['split_wise_metrics']['test']['all_cell_velocity_rmse_km_per_s']['mean'])}` km/s."
        ),
        "",
        "## Interpretation Boundary",
        "",
        (
            "A cautious paper statement supported by this package is: ML reconstruction "
            "outperformed the implemented fixed-ray regularized tomography baseline under the "
            "configured synthetic benchmark."
        ),
        "",
        "This should not be generalized to real data, full nonlinear tomography, or all tomography workflows.",
        "",
        "## Generated Tables",
        "",
    ]
    lines.extend(f"- `{_relative_to_repo(item, repo_root)}`" for item in table_paths)
    lines.extend(["", "## Generated Figures", ""])
    lines.extend(f"- `{_relative_to_repo(item, repo_root)}`" for item in figure_paths)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_artifact_index(
    evidence: dict[str, Any],
    table_paths: list[Path],
    figure_paths: list[Path],
    figure_metadata_path: Path,
    summary_note: Path,
    experiment_note: Path,
    repo_root: Path,
) -> dict[str, Any]:
    paths = evidence["paths"]
    return {
        "artifact_type": "paper_candidate_v2_artifact_index",
        "package_id": PAPER_CANDIDATE_V2_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "does_not_rerun_training_or_tomography": True,
        "preserves_historical_package": "outputs/generated/paper_candidate_v1/",
        "source_evidence": {
            key: _relative_to_repo(path, repo_root) for key, path in paths.items()
        },
        "generated_tables": [_relative_to_repo(path, repo_root) for path in table_paths],
        "generated_figures": [_relative_to_repo(path, repo_root) for path in figure_paths],
        "figure_metadata_json": _relative_to_repo(figure_metadata_path, repo_root),
        "summary_note": _relative_to_repo(summary_note, repo_root),
        "experiment_note": _relative_to_repo(experiment_note, repo_root),
        "claim_boundary": [
            "Synthetic-only benchmark.",
            "Known-ray inversion is optimistic diagnostic evidence.",
            "Reference-ray inversion is fixed-ray regularized tomography, not full nonlinear tomography.",
            "All-100-case ML cell-centered metrics include training cases; fixed-test and LOO rows are separate.",
            "High reference-ray clipping must be reported.",
        ],
    }


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


def _read_cell_prediction_values(path: Path) -> tuple[float, ...]:
    rows = sorted(_read_csv_rows(path), key=lambda row: int(row["cell_index"]))
    return tuple(float(row["predicted_velocity_km_per_s"]) for row in rows)


def _cell_slice(values: Sequence[float], shape: tuple[int, int, int], iz: int) -> list[list[float]]:
    nx, ny, nz = shape
    if iz < 0 or iz >= nz:
        raise ValueError("Slice index is outside the cell grid.")
    grid = []
    for iy in range(ny):
        row = []
        for ix in range(nx):
            row.append(float(values[ix + nx * (iy + ny * iz)]))
        grid.append(row)
    return grid


def _subtract_grid(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [left_value - right_value for left_value, right_value in zip(left_row, right_row, strict=True)]
        for left_row, right_row in zip(left, right, strict=True)
    ]


def _load_velocity_grid(path: Path) -> CartesianVelocityGrid3D:
    payload = _read_json(path)
    return CartesianVelocityGrid3D(
        grid_id=str(payload["grid_id"]),
        source_velocity_model_id=str(payload["source_velocity_model_id"]),
        x_coordinates_km=tuple(float(value) for value in payload["x_coordinates_km"]),
        y_coordinates_km=tuple(float(value) for value in payload["y_coordinates_km"]),
        z_coordinates_km=tuple(float(value) for value in payload["z_coordinates_km"]),
        p_velocity_km_per_s=tuple(float(value) for value in payload["p_velocity_km_per_s"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _row_by_case(rows: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    return next(row for row in rows if row["case_id"] == case_id)


def _markdown_table(rows: list[dict[str, object]]) -> str:
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row[header]) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _cell(value: object) -> str:
    if isinstance(value, float):
        return _format_float(value)
    return str(value).replace("|", "\\|")


def _format_float(value: float) -> str:
    return f"{float(value):.6f}"


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
