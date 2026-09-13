"""Corpus-scale known-ray tomography diagnostics and aggregation."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from tomobench.tomography.inversion import (
    run_coverage_aware_known_ray_inversion_diagnostic,
    run_known_ray_perturbation_inversion_diagnostic,
)
from tomobench.utils.paths import get_repo_root

DEFAULT_CORPUS_MANIFEST = Path(
    "outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json"
)
DEFAULT_ML_SUITE_SUMMARY = Path(
    "outputs/generated/ml_baselines/paper_pseudo_bending_ml_suite_v1/suite_summary.json"
)
DEFAULT_KNOWN_RAY_CORPUS_OUTPUT_DIR = Path(
    "outputs/generated/classical_tomography/known_ray_corpus_v1"
)
DEFAULT_KNOWN_RAY_PERTURBATION_OUTPUT_DIR = Path(
    "outputs/generated/classical_tomography/known_ray_perturbation_v1"
)
DEFAULT_REFERENCE_RAY_CORPUS_DIR = Path(
    "outputs/generated/classical_tomography/reference_ray_corpus_v1"
)
DEFAULT_LEGACY_KNOWN_RAY_CASE_SUMMARY = Path(
    "outputs/generated/classical_tomography/known_ray_corpus_v1/known_ray_corpus_case_summary.csv"
)


@dataclass(frozen=True)
class KnownRayCorpusInversionOutputs:
    """Artifacts written by the corpus-scale known-ray diagnostic."""

    output_dir: Path
    case_summary_csv: Path
    summary_csv: Path
    summary_json: Path
    comparison_json: Path
    comparison_csv: Path


@dataclass(frozen=True)
class KnownRayPerturbationCorpusInversionOutputs:
    """Artifacts written by the corrected known-ray formulation diagnostic."""

    output_dir: Path
    case_summary_csv: Path
    summary_json: Path
    summary_md: Path


def run_known_ray_corpus_inversion(
    manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    output_dir: Path = DEFAULT_KNOWN_RAY_CORPUS_OUTPUT_DIR,
    ml_suite_summary_path: Path | None = DEFAULT_ML_SUITE_SUMMARY,
    damping_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    smoothing_values: Sequence[float] = (0.0, 0.01, 0.1, 1.0),
    minimum_coverage_km: float = 1.0e-9,
) -> KnownRayCorpusInversionOutputs:
    """Run coverage-aware known-ray inversion for each case in a manifest.

    The geometry matrix is built from pseudo-bending rays traced through the
    true synthetic model. This remains an optimistic diagnostic, not the fair
    reference-model fixed-ray baseline needed for final classical comparison.
    """
    if not damping_values:
        raise ValueError("damping_values must contain at least one value.")
    if not smoothing_values:
        raise ValueError("smoothing_values must contain at least one value.")

    repo_root = get_repo_root()
    resolved_manifest_path = _resolve_path(manifest_path, repo_root, None)
    manifest = _read_json(resolved_manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Corpus manifest must contain at least one item.")

    output_dir = _resolve_path(output_dir, repo_root, None)
    cases_dir = output_dir / "cases"
    case_rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError("Corpus manifest item must be a JSON object.")
        case_id = str(item["case_id"])
        case_output_dir = cases_dir / case_id
        outputs = run_coverage_aware_known_ray_inversion_diagnostic(
            sensitivity_path=_case_sensitivity_path(item, repo_root, resolved_manifest_path),
            sensitivity_metadata_path=_case_sensitivity_metadata_path(
                item,
                repo_root,
                resolved_manifest_path,
            ),
            observations_path=_case_observation_path(item, repo_root, resolved_manifest_path),
            velocity_grid_path=_case_velocity_grid_path(item, repo_root, resolved_manifest_path),
            output_dir=case_output_dir,
            damping_values=damping_values,
            smoothing_values=smoothing_values,
            minimum_coverage_km=minimum_coverage_km,
        )
        case_summary = _read_json(outputs.summary_json)
        case_rows.append(_case_summary_row(item, case_summary, outputs.summary_json))

    output_dir.mkdir(parents=True, exist_ok=True)
    case_summary_csv = output_dir / "known_ray_corpus_case_summary.csv"
    summary_csv = output_dir / "known_ray_corpus_summary.csv"
    summary_json = output_dir / "known_ray_corpus_summary.json"
    comparison_json = output_dir / "known_ray_vs_pca_linear_comparison.json"
    comparison_csv = output_dir / "known_ray_vs_pca_linear_comparison.csv"

    _write_case_summary_csv(case_summary_csv, case_rows)
    corpus_summary = _corpus_summary_payload(
        manifest=manifest,
        manifest_path=resolved_manifest_path,
        output_dir=output_dir,
        case_rows=case_rows,
        damping_values=damping_values,
        smoothing_values=smoothing_values,
        minimum_coverage_km=minimum_coverage_km,
    )
    _write_json(summary_json, corpus_summary)
    _write_summary_csv(summary_csv, corpus_summary)
    comparison_payload = _comparison_payload(
        corpus_summary=corpus_summary,
        ml_suite_summary_path=ml_suite_summary_path,
        repo_root=repo_root,
    )
    _write_json(comparison_json, comparison_payload)
    _write_comparison_csv(comparison_csv, comparison_payload)
    return KnownRayCorpusInversionOutputs(
        output_dir=output_dir,
        case_summary_csv=case_summary_csv,
        summary_csv=summary_csv,
        summary_json=summary_json,
        comparison_json=comparison_json,
        comparison_csv=comparison_csv,
    )


def run_known_ray_perturbation_corpus_inversion(
    manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    output_dir: Path = DEFAULT_KNOWN_RAY_PERTURBATION_OUTPUT_DIR,
    reference_ray_corpus_dir: Path = DEFAULT_REFERENCE_RAY_CORPUS_DIR,
    legacy_case_summary_path: Path = DEFAULT_LEGACY_KNOWN_RAY_CASE_SUMMARY,
    damping: float = 0.01,
    smoothing: float = 1.0,
    minimum_coverage_km: float = 1.0e-9,
) -> KnownRayPerturbationCorpusInversionOutputs:
    """Run the corrected true-geometry perturbation diagnostic across the corpus.

    The fixed parameters mirror the original known-ray diagnostic so the
    formulation change can be isolated.  Reference-ray outputs provide the
    shared layered prior; true-model sensitivity sidecars provide privileged G.
    """
    if damping < 0.0 or smoothing < 0.0:
        raise ValueError("damping and smoothing must be non-negative.")
    if minimum_coverage_km < 0.0:
        raise ValueError("minimum_coverage_km must be non-negative.")

    repo_root = get_repo_root()
    resolved_manifest_path = _resolve_path(manifest_path, repo_root, None)
    manifest = _read_json(resolved_manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Corpus manifest must contain at least one item.")
    resolved_output_dir = _resolve_path(output_dir, repo_root, None)
    resolved_reference_dir = _resolve_path(reference_ray_corpus_dir, repo_root, None)
    resolved_legacy_summary = _resolve_path(legacy_case_summary_path, repo_root, None)
    legacy_rows = _read_csv(resolved_legacy_summary) if resolved_legacy_summary.is_file() else []
    legacy_by_case_id = {str(row["case_id"]): row for row in legacy_rows}

    cases_dir = resolved_output_dir / "cases"
    case_rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError("Corpus manifest item must be a JSON object.")
        case_id = str(item["case_id"])
        reference_summary_path = (
            resolved_reference_dir / "cases" / case_id / "reference_ray_perturbation_summary.json"
        )
        _require_file(reference_summary_path, "reference-ray perturbation summary")
        reference_summary = _read_json(reference_summary_path)
        source_files = reference_summary.get("source_files")
        if not isinstance(source_files, dict) or "reference_velocity_grid" not in source_files:
            raise ValueError(f"Reference summary has no reference velocity grid for {case_id}.")
        case_output_dir = cases_dir / case_id
        case_summary_path = case_output_dir / "known_ray_perturbation_summary.json"
        if case_summary_path.is_file() and _summary_matches_parameters(
            case_summary_path,
            damping=damping,
            smoothing=smoothing,
            minimum_coverage_km=minimum_coverage_km,
        ):
            summary = _read_json(case_summary_path)
        else:
            outputs = run_known_ray_perturbation_inversion_diagnostic(
                sensitivity_path=_case_sensitivity_path(item, repo_root, resolved_manifest_path),
                sensitivity_metadata_path=_case_sensitivity_metadata_path(
                    item,
                    repo_root,
                    resolved_manifest_path,
                ),
                observations_path=_case_observation_path(item, repo_root, resolved_manifest_path),
                velocity_grid_path=_case_velocity_grid_path(item, repo_root, resolved_manifest_path),
                reference_velocity_grid_path=_resolve_existing_path(
                    source_files["reference_velocity_grid"],
                    repo_root,
                    "reference velocity grid",
                ),
                output_dir=case_output_dir,
                damping=damping,
                smoothing=smoothing,
                minimum_coverage_km=minimum_coverage_km,
            )
            case_summary_path = outputs.summary_json
            summary = _read_json(case_summary_path)
        row = _known_ray_perturbation_case_summary_row(item, summary, case_summary_path)
        legacy = legacy_by_case_id.get(case_id)
        if legacy is not None:
            row.update(
                {
                    "legacy_absolute_known_ray_all_cell_rmse_km_per_s": _optional_float(
                        legacy.get("all_cell_velocity_rmse_km_per_s")
                    ),
                    "legacy_absolute_known_ray_covered_cell_rmse_km_per_s": _optional_float(
                        legacy.get("covered_cell_velocity_rmse_km_per_s")
                    ),
                }
            )
        else:
            row.update(
                {
                    "legacy_absolute_known_ray_all_cell_rmse_km_per_s": None,
                    "legacy_absolute_known_ray_covered_cell_rmse_km_per_s": None,
                }
            )
        row["corrected_minus_legacy_all_cell_rmse_km_per_s"] = _difference(
            row["known_ray_perturbation_all_cell_rmse_km_per_s"],
            row["legacy_absolute_known_ray_all_cell_rmse_km_per_s"],
        )
        row["corrected_minus_legacy_covered_cell_rmse_km_per_s"] = _difference(
            row["known_ray_perturbation_covered_cell_rmse_km_per_s"],
            row["legacy_absolute_known_ray_covered_cell_rmse_km_per_s"],
        )
        case_rows.append(row)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    case_summary_csv = resolved_output_dir / "known_ray_perturbation_case_summary.csv"
    summary_json = resolved_output_dir / "known_ray_perturbation_summary.json"
    summary_md = resolved_output_dir / "known_ray_perturbation_summary.md"
    _write_csv(case_summary_csv, case_rows)
    summary_payload = _known_ray_perturbation_corpus_summary(
        manifest=manifest,
        manifest_path=resolved_manifest_path,
        output_dir=resolved_output_dir,
        case_rows=case_rows,
        reference_ray_corpus_dir=resolved_reference_dir,
        legacy_case_summary_path=resolved_legacy_summary,
        damping=damping,
        smoothing=smoothing,
        minimum_coverage_km=minimum_coverage_km,
    )
    _write_json(summary_json, summary_payload)
    _write_known_ray_perturbation_markdown(summary_md, summary_payload, repo_root)
    return KnownRayPerturbationCorpusInversionOutputs(
        output_dir=resolved_output_dir,
        case_summary_csv=case_summary_csv,
        summary_json=summary_json,
        summary_md=summary_md,
    )


def _known_ray_perturbation_case_summary_row(
    item: dict[str, Any],
    summary: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any]:
    all_cell = summary["all_cell_reconstruction_metrics"]
    covered_cell = summary["covered_cell_reconstruction_metrics"]
    prior = summary["reference_prior_metrics"]
    prior_covered = summary["reference_prior_covered_cell_metrics"]
    coverage = summary["coverage_summary"]
    return {
        "case_id": str(item["case_id"]),
        "family": str(item.get("family") or item.get("scenario") or "unknown"),
        "damping": float(summary["damping"]),
        "smoothing": float(summary["smoothing"]),
        "known_ray_perturbation_all_cell_rmse_km_per_s": float(
            all_cell["velocity_rmse_km_per_s"]
        ),
        "known_ray_perturbation_all_cell_mae_km_per_s": float(
            all_cell["velocity_mae_km_per_s"]
        ),
        "known_ray_perturbation_covered_cell_rmse_km_per_s": _optional_float(
            covered_cell.get("velocity_rmse_km_per_s")
        ),
        "known_ray_perturbation_covered_cell_mae_km_per_s": _optional_float(
            covered_cell.get("velocity_mae_km_per_s")
        ),
        "reference_prior_all_cell_rmse_km_per_s": float(prior["velocity_rmse_km_per_s"]),
        "reference_prior_all_cell_mae_km_per_s": float(prior["velocity_mae_km_per_s"]),
        "reference_prior_covered_cell_rmse_km_per_s": _optional_float(
            prior_covered.get("velocity_rmse_km_per_s")
        ),
        "reference_prior_covered_cell_mae_km_per_s": _optional_float(
            prior_covered.get("velocity_mae_km_per_s")
        ),
        "travel_time_rmse_s": float(summary["residual_metrics"]["travel_time_rmse_s"]),
        "travel_time_mae_s": float(summary["residual_metrics"]["travel_time_mae_s"]),
        "covered_cell_count": int(coverage["covered_cell_count"]),
        "uncovered_cell_count": int(coverage["uncovered_cell_count"]),
        "coverage_fraction": float(coverage["coverage_fraction"]),
        "clipped_velocity_cell_count": int(summary["clipped_velocity_cell_count"]),
        "nonphysical_slowness_count": int(summary["nonphysical_slowness_count"]),
        "summary_json": summary_path.as_posix(),
    }


def _summary_matches_parameters(
    path: Path,
    *,
    damping: float,
    smoothing: float,
    minimum_coverage_km: float,
) -> bool:
    try:
        summary = _read_json(path)
        return (
            math.isclose(float(summary["damping"]), damping, rel_tol=0.0, abs_tol=1.0e-15)
            and math.isclose(float(summary["smoothing"]), smoothing, rel_tol=0.0, abs_tol=1.0e-15)
            and math.isclose(
                float(summary["minimum_coverage_threshold_km"]),
                minimum_coverage_km,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _known_ray_perturbation_corpus_summary(
    manifest: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
    case_rows: list[dict[str, Any]],
    reference_ray_corpus_dir: Path,
    legacy_case_summary_path: Path,
    damping: float,
    smoothing: float,
    minimum_coverage_km: float,
) -> dict[str, Any]:
    return {
        "artifact_type": "known_ray_true_geometry_perturbation_corpus_summary",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest_path": manifest_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "reference_ray_corpus_dir": reference_ray_corpus_dir.as_posix(),
        "legacy_absolute_known_ray_case_summary_path": legacy_case_summary_path.as_posix(),
        "input_corpus_id": manifest.get("corpus_id") or manifest.get("batch_id"),
        "diagnostic_type": "known_ray_true_geometry_damped_smoothed_perturbation_corpus",
        "sample_count": len(case_rows),
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in case_rows).items())),
        "damping": float(damping),
        "smoothing": float(smoothing),
        "parameter_selection_scope": "fixed_parameters_for_formulation_diagnosis",
        "minimum_coverage_threshold_km": float(minimum_coverage_km),
        "reference_prior_all_cell_velocity_rmse_km_per_s": _aggregate(
            case_rows, "reference_prior_all_cell_rmse_km_per_s"
        ),
        "known_ray_perturbation_all_cell_velocity_rmse_km_per_s": _aggregate(
            case_rows, "known_ray_perturbation_all_cell_rmse_km_per_s"
        ),
        "reference_prior_covered_cell_velocity_rmse_km_per_s": _aggregate(
            case_rows, "reference_prior_covered_cell_rmse_km_per_s"
        ),
        "known_ray_perturbation_covered_cell_velocity_rmse_km_per_s": _aggregate(
            case_rows, "known_ray_perturbation_covered_cell_rmse_km_per_s"
        ),
        "known_ray_perturbation_minus_prior_all_cell_rmse_km_per_s": _aggregate_difference(
            case_rows,
            "known_ray_perturbation_all_cell_rmse_km_per_s",
            "reference_prior_all_cell_rmse_km_per_s",
        ),
        "known_ray_perturbation_minus_prior_covered_cell_rmse_km_per_s": _aggregate_difference(
            case_rows,
            "known_ray_perturbation_covered_cell_rmse_km_per_s",
            "reference_prior_covered_cell_rmse_km_per_s",
        ),
        "corrected_minus_legacy_absolute_all_cell_rmse_km_per_s": _aggregate(
            case_rows, "corrected_minus_legacy_all_cell_rmse_km_per_s"
        ),
        "corrected_minus_legacy_absolute_covered_cell_rmse_km_per_s": _aggregate(
            case_rows, "corrected_minus_legacy_covered_cell_rmse_km_per_s"
        ),
        "coverage_fraction": _aggregate(case_rows, "coverage_fraction"),
        "clipped_velocity_cell_count": _aggregate(case_rows, "clipped_velocity_cell_count"),
        "case_summary_csv": (output_dir / "known_ray_perturbation_case_summary.csv").as_posix(),
        "interpretation": [
            "The corrected solve estimates a perturbation around the layered reference prior, while retaining true-model ray geometry.",
            "The legacy supplied-ray diagnostic solves absolute slowness with regularization toward zero; its result is retained for diagnosis only.",
            "Both true-geometry diagnostics are privileged and must not be presented as fair held-out classical baselines.",
            "The fixed lambda/alpha pair is used to isolate formulation effects; it is not selected using test reconstruction error.",
        ],
    }


def _write_known_ray_perturbation_markdown(
    path: Path,
    summary: dict[str, Any],
    repo_root: Path,
) -> None:
    lines = [
        "# Corrected Known-Ray Perturbation Diagnostic",
        "",
        "This artifact diagnoses the supplied-ray anomaly by changing the unknown from absolute slowness to a perturbation around the same layered reference prior. True-model ray geometry remains privileged.",
        "",
        f"- Cases: `{summary['sample_count']}`",
        f"- Fixed damping: `{summary['damping']}`",
        f"- Fixed smoothing: `{summary['smoothing']}`",
        f"- Coverage threshold: `{summary['minimum_coverage_threshold_km']}` km",
        "",
        "## Corpus means",
        "",
        f"- Reference-prior all-cell RMSE: `{summary['reference_prior_all_cell_velocity_rmse_km_per_s']['mean']}` km/s",
        f"- Corrected true-geometry all-cell RMSE: `{summary['known_ray_perturbation_all_cell_velocity_rmse_km_per_s']['mean']}` km/s",
        f"- Corrected minus prior all-cell RMSE: `{summary['known_ray_perturbation_minus_prior_all_cell_rmse_km_per_s']['mean']}` km/s",
        f"- Legacy absolute known-ray minus corrected all-cell RMSE: `{summary['corrected_minus_legacy_absolute_all_cell_rmse_km_per_s']['mean']}` km/s",
        "",
        "## Outputs",
        "",
        f"- Case summary: `{_relative_or_absolute(Path(summary['case_summary_csv']), repo_root)}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_summary_row(
    item: dict[str, Any],
    summary: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any]:
    residual = summary["residual_metrics"]
    all_cell = summary["all_cell_reconstruction_metrics"]
    covered_cell = summary["covered_cell_reconstruction_metrics"]
    coverage = summary["coverage_summary"]
    return {
        "case_id": str(item["case_id"]),
        "family": str(item.get("family") or item.get("scenario") or "unknown"),
        "selected_damping": float(summary["selected_damping"]),
        "selected_smoothing": float(summary["selected_smoothing"]),
        "travel_time_rmse_s": float(residual["travel_time_rmse_s"]),
        "travel_time_mae_s": float(residual["travel_time_mae_s"]),
        "all_cell_velocity_rmse_km_per_s": float(all_cell["velocity_rmse_km_per_s"]),
        "all_cell_velocity_mae_km_per_s": float(all_cell["velocity_mae_km_per_s"]),
        "covered_cell_velocity_rmse_km_per_s": _optional_float(
            covered_cell.get("velocity_rmse_km_per_s")
        ),
        "covered_cell_velocity_mae_km_per_s": _optional_float(
            covered_cell.get("velocity_mae_km_per_s")
        ),
        "covered_cell_count": int(coverage["covered_cell_count"]),
        "uncovered_cell_count": int(coverage["uncovered_cell_count"]),
        "coverage_fraction": float(coverage["coverage_fraction"]),
        "min_positive_coverage_km": _optional_float(coverage.get("min_positive_coverage_km")),
        "mean_positive_coverage_km": _optional_float(coverage.get("mean_positive_coverage_km")),
        "max_positive_coverage_km": _optional_float(coverage.get("max_positive_coverage_km")),
        "clipped_velocity_cell_count": int(summary["clipped_velocity_cell_count"]),
        "nonphysical_slowness_count": int(summary["nonphysical_slowness_count"]),
        "summary_json": summary_path.as_posix(),
    }


def _corpus_summary_payload(
    manifest: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
    case_rows: list[dict[str, Any]],
    damping_values: Sequence[float],
    smoothing_values: Sequence[float],
    minimum_coverage_km: float,
) -> dict[str, Any]:
    families = sorted({str(row["family"]) for row in case_rows})
    return {
        "artifact_type": "known_ray_coverage_smoothing_corpus_summary",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest_path": manifest_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "input_corpus_id": manifest.get("corpus_id") or manifest.get("batch_id"),
        "diagnostic_type": "known_ray_true_geometry_damped_smoothed_least_squares_corpus",
        "sample_count": len(case_rows),
        "family_list": families,
        "damping_values": [float(value) for value in damping_values],
        "smoothing_values": [float(value) for value in smoothing_values],
        "minimum_coverage_threshold_km": float(minimum_coverage_km),
        "selected_damping_distribution": _value_distribution(case_rows, "selected_damping"),
        "selected_smoothing_distribution": _value_distribution(case_rows, "selected_smoothing"),
        "selected_lambda_alpha_distribution": _parameter_pair_distribution(case_rows),
        "travel_time_rmse_s": _aggregate(case_rows, "travel_time_rmse_s"),
        "travel_time_mae_s": _aggregate(case_rows, "travel_time_mae_s"),
        "all_cell_velocity_rmse_km_per_s": _aggregate(
            case_rows,
            "all_cell_velocity_rmse_km_per_s",
        ),
        "all_cell_velocity_mae_km_per_s": _aggregate(
            case_rows,
            "all_cell_velocity_mae_km_per_s",
        ),
        "covered_cell_velocity_rmse_km_per_s": _aggregate(
            case_rows,
            "covered_cell_velocity_rmse_km_per_s",
        ),
        "covered_cell_velocity_mae_km_per_s": _aggregate(
            case_rows,
            "covered_cell_velocity_mae_km_per_s",
        ),
        "coverage_fraction": _aggregate(case_rows, "coverage_fraction"),
        "clipped_velocity_cell_count": _aggregate(case_rows, "clipped_velocity_cell_count"),
        "nonphysical_slowness_count": _aggregate(case_rows, "nonphysical_slowness_count"),
        "family_wise_metrics": _family_wise_metrics(case_rows),
        "case_summary_csv": (output_dir / "known_ray_corpus_case_summary.csv").as_posix(),
        "summary_csv": (output_dir / "known_ray_corpus_summary.csv").as_posix(),
        "target_comparison_contract": {
            "known_ray_target": "cell_centered_velocity_from_node_grid_arithmetic_cell_means",
            "sensitivity_representation": "cell_centered_path_lengths_between_adjacent_grid_nodes",
            "metric_domains": [
                "all derived Cartesian cells",
                "covered cells at or above the configured coverage threshold",
            ],
        },
        "assumptions": [
            "This corpus result remains a known-ray diagnostic because each G matrix uses pseudo-bending rays traced through the true synthetic velocity model.",
            "Covered-cell metrics are reported separately from all-cell metrics to avoid interpreting unconstrained cells as reconstruction evidence.",
            "Velocity clipping remains explicit and counted per case.",
            "This artifact does not establish a fair classical-versus-ML comparison; reference-model fixed-ray inversion remains a later Phase 6 step.",
        ],
    }


def _comparison_payload(
    corpus_summary: dict[str, Any],
    ml_suite_summary_path: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    ml_reference = None
    if ml_suite_summary_path is not None:
        resolved_path = _resolve_path(ml_suite_summary_path, repo_root, None)
        if resolved_path.is_file():
            ml_reference = _extract_pca_linear_reference(_read_json(resolved_path), resolved_path)
    return {
        "artifact_type": "known_ray_vs_pca_linear_observation_regressor_comparison_caveated",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "known_ray_corpus_summary": corpus_summary["output_dir"]
        + "/known_ray_corpus_summary.json",
        "known_ray_reference": {
            "sample_count": corpus_summary["sample_count"],
            "travel_time_rmse_s": corpus_summary["travel_time_rmse_s"],
            "all_cell_velocity_rmse_km_per_s": corpus_summary[
                "all_cell_velocity_rmse_km_per_s"
            ],
            "all_cell_velocity_mae_km_per_s": corpus_summary[
                "all_cell_velocity_mae_km_per_s"
            ],
            "covered_cell_velocity_rmse_km_per_s": corpus_summary[
                "covered_cell_velocity_rmse_km_per_s"
            ],
            "covered_cell_velocity_mae_km_per_s": corpus_summary[
                "covered_cell_velocity_mae_km_per_s"
            ],
            "coverage_fraction": corpus_summary["coverage_fraction"],
            "diagnostic_type": corpus_summary["diagnostic_type"],
        },
        "ml_reference": ml_reference,
        "comparison_contract": {
            "known_ray_velocity_metric_domain": "cell-centered derived Cartesian cells",
            "ml_velocity_metric_domain": "node-centered full Cartesian velocity grid",
            "shared_metric_contract": False,
            "interpretation": (
                "The numeric velocity metrics are adjacent diagnostics, not a strict "
                "apples-to-apples ranking, because the known-ray inversion is evaluated "
                "on cell-centered velocities while the selected ML model is evaluated on "
                "node-centered full-grid targets."
            ),
        },
        "scientific_caution": [
            "Known-ray inversion is optimistic because ray geometry comes from the true synthetic model.",
            "The selected ML model reference is included for context only under a different target contract.",
            "No fair classical-versus-ML superiority claim should be made from this artifact.",
        ],
    }


def _extract_pca_linear_reference(
    ml_summary: dict[str, Any],
    path: Path,
) -> dict[str, Any] | None:
    candidates = []
    best_model = ml_summary.get("best_model")
    if isinstance(best_model, dict):
        candidates.append(best_model)
    records = ml_summary.get("fixed_split_model_records")
    if isinstance(records, list):
        candidates.extend(record for record in records if isinstance(record, dict))
    for candidate in candidates:
        if candidate.get("model_name") == "pca_linear_observation_regressor":
            return {
                "source_summary_json": path.as_posix(),
                "model_name": "pca_linear_observation_regressor",
                "experiment_id": candidate.get("experiment_id"),
                "fixed_train_rmse_node_velocity_km_per_s": candidate.get("fixed_train_rmse"),
                "fixed_train_mae_node_velocity_km_per_s": candidate.get("fixed_train_mae"),
                "fixed_validation_rmse_node_velocity_km_per_s": candidate.get(
                    "fixed_validation_rmse"
                ),
                "fixed_validation_mae_node_velocity_km_per_s": candidate.get(
                    "fixed_validation_mae"
                ),
                "fixed_test_rmse_node_velocity_km_per_s": candidate.get("fixed_test_rmse"),
                "fixed_test_mae_node_velocity_km_per_s": candidate.get("fixed_test_mae"),
                "metric_domain": "node_centered_full_grid_velocity_target",
            }
    return None


def _write_case_summary_csv(path: Path, case_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(case_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)


def _write_comparison_csv(path: Path, comparison: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    known_ray = comparison["known_ray_reference"]
    ml_reference = comparison.get("ml_reference") or {}
    rows = [
        {
            "metric": "velocity_rmse_km_per_s",
            "known_ray_scope": "all_cell_mean",
            "known_ray_value": known_ray["all_cell_velocity_rmse_km_per_s"]["mean"],
            "ml_scope": "fixed_test_node_centered",
            "ml_value": ml_reference.get("fixed_test_rmse_node_velocity_km_per_s"),
            "shared_metric_contract": False,
        },
        {
            "metric": "velocity_rmse_km_per_s",
            "known_ray_scope": "covered_cell_mean",
            "known_ray_value": known_ray["covered_cell_velocity_rmse_km_per_s"]["mean"],
            "ml_scope": "fixed_test_node_centered",
            "ml_value": ml_reference.get("fixed_test_rmse_node_velocity_km_per_s"),
            "shared_metric_contract": False,
        },
        {
            "metric": "velocity_mae_km_per_s",
            "known_ray_scope": "all_cell_mean",
            "known_ray_value": known_ray["all_cell_velocity_mae_km_per_s"]["mean"],
            "ml_scope": "fixed_test_node_centered",
            "ml_value": ml_reference.get("fixed_test_mae_node_velocity_km_per_s"),
            "shared_metric_contract": False,
        },
        {
            "metric": "velocity_mae_km_per_s",
            "known_ray_scope": "covered_cell_mean",
            "known_ray_value": known_ray["covered_cell_velocity_mae_km_per_s"]["mean"],
            "ml_scope": "fixed_test_node_centered",
            "ml_value": ml_reference.get("fixed_test_mae_node_velocity_km_per_s"),
            "shared_metric_contract": False,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"section": "corpus", "metric": "sample_count", "value": summary["sample_count"]},
        {"section": "corpus", "metric": "family_list", "value": ";".join(summary["family_list"])},
    ]
    for key in (
        "travel_time_rmse_s",
        "travel_time_mae_s",
        "all_cell_velocity_rmse_km_per_s",
        "all_cell_velocity_mae_km_per_s",
        "covered_cell_velocity_rmse_km_per_s",
        "covered_cell_velocity_mae_km_per_s",
        "coverage_fraction",
        "clipped_velocity_cell_count",
        "nonphysical_slowness_count",
    ):
        rows.extend(_aggregate_rows("corpus", key, summary[key]))
    rows.extend(
        {
            "section": "selected_damping_distribution",
            "metric": key,
            "value": value,
        }
        for key, value in summary["selected_damping_distribution"].items()
    )
    rows.extend(
        {
            "section": "selected_smoothing_distribution",
            "metric": key,
            "value": value,
        }
        for key, value in summary["selected_smoothing_distribution"].items()
    )
    rows.extend(
        {
            "section": "selected_lambda_alpha_distribution",
            "metric": key,
            "value": value,
        }
        for key, value in summary["selected_lambda_alpha_distribution"].items()
    )
    for family, family_summary in summary["family_wise_metrics"].items():
        rows.append(
            {
                "section": f"family:{family}",
                "metric": "sample_count",
                "value": family_summary["sample_count"],
            }
        )
        for key in (
            "travel_time_rmse_s",
            "travel_time_mae_s",
            "all_cell_velocity_rmse_km_per_s",
            "all_cell_velocity_mae_km_per_s",
            "covered_cell_velocity_rmse_km_per_s",
            "covered_cell_velocity_mae_km_per_s",
            "coverage_fraction",
            "clipped_velocity_cell_count",
        ):
            rows.extend(_aggregate_rows(f"family:{family}", key, family_summary[key]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)


def _family_wise_metrics(case_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row["family"])].append(row)
    return {
        family: {
            "sample_count": len(rows),
            "travel_time_rmse_s": _aggregate(rows, "travel_time_rmse_s"),
            "travel_time_mae_s": _aggregate(rows, "travel_time_mae_s"),
            "all_cell_velocity_rmse_km_per_s": _aggregate(
                rows,
                "all_cell_velocity_rmse_km_per_s",
            ),
            "all_cell_velocity_mae_km_per_s": _aggregate(
                rows,
                "all_cell_velocity_mae_km_per_s",
            ),
            "covered_cell_velocity_rmse_km_per_s": _aggregate(
                rows,
                "covered_cell_velocity_rmse_km_per_s",
            ),
            "covered_cell_velocity_mae_km_per_s": _aggregate(
                rows,
                "covered_cell_velocity_mae_km_per_s",
            ),
            "coverage_fraction": _aggregate(rows, "coverage_fraction"),
            "clipped_velocity_cell_count": _aggregate(rows, "clipped_velocity_cell_count"),
        }
        for family, rows in sorted(grouped.items())
    }


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and not math.isnan(float(row[key]))
    ]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _aggregate_difference(
    rows: list[dict[str, Any]],
    left_key: str,
    right_key: str,
) -> dict[str, float | int | None]:
    differences = [
        float(row[left_key]) - float(row[right_key])
        for row in rows
        if row.get(left_key) not in (None, "") and row.get(right_key) not in (None, "")
    ]
    if not differences:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(differences),
        "mean": sum(differences) / len(differences),
        "min": min(differences),
        "max": max(differences),
    }


def _aggregate_rows(
    section: str,
    metric: str,
    aggregate: dict[str, float | int | None],
) -> list[dict[str, float | int | str | None]]:
    return [
        {"section": section, "metric": f"{metric}_{name}", "value": value}
        for name, value in aggregate.items()
    ]


def _value_distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(_float_key(row[key]) for row in rows).items()))


def _parameter_pair_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(
        f"lambda={_float_key(row['selected_damping'])},alpha={_float_key(row['selected_smoothing'])}"
        for row in rows
    )
    return dict(sorted(counter.items()))


def _case_observation_path(item: dict[str, Any], repo_root: Path, manifest_path: Path) -> Path:
    path_value = item.get("observation_csv_path") or item.get("input_dataset_csv_path")
    if not path_value:
        raise ValueError(f"{item.get('case_id')}: missing observation CSV path.")
    return _existing_path(path_value, repo_root, manifest_path, "observation CSV")


def _case_velocity_grid_path(item: dict[str, Any], repo_root: Path, manifest_path: Path) -> Path:
    path_value = item.get("target_velocity_grid_path") or item.get("target_grid_path")
    if not path_value:
        raise ValueError(f"{item.get('case_id')}: missing target velocity grid path.")
    return _existing_path(path_value, repo_root, manifest_path, "target velocity grid")


def _case_sensitivity_path(item: dict[str, Any], repo_root: Path, manifest_path: Path) -> Path:
    path_value = item.get("sensitivity_sidecar_path")
    if path_value:
        return _existing_path(path_value, repo_root, manifest_path, "sensitivity sidecar")
    return _existing_path(
        _inferred_sensitivity_stem(item) + ".ray_cell_sensitivity.jsonl",
        repo_root,
        manifest_path,
        "inferred sensitivity sidecar",
    )


def _case_sensitivity_metadata_path(
    item: dict[str, Any],
    repo_root: Path,
    manifest_path: Path,
) -> Path:
    path_value = item.get("sensitivity_metadata_path")
    if path_value:
        return _existing_path(path_value, repo_root, manifest_path, "sensitivity metadata")
    return _existing_path(
        _inferred_sensitivity_stem(item) + ".ray_cell_sensitivity.metadata.json",
        repo_root,
        manifest_path,
        "inferred sensitivity metadata",
    )


def _inferred_sensitivity_stem(item: dict[str, Any]) -> str:
    observation_path = Path(
        str(item.get("observation_csv_path") or item.get("input_dataset_csv_path") or "")
    )
    if observation_path.name:
        travel_time_stem = observation_path.stem.replace(
            "source_receiver_dataset",
            "travel_times",
            1,
        )
        batch_dir = observation_path.parent.name
        return f"outputs/generated/ml_supervised/{batch_dir}/sensitivity_matrices/{travel_time_stem}"
    case_id = str(item["case_id"])
    return (
        "outputs/generated/ml_supervised/paper_pseudo_bending_observation_pairs_v1/"
        f"sensitivity_matrices/first_vertical_slice_travel_times_{case_id}"
    )


def _existing_path(path_value: Any, repo_root: Path, manifest_path: Path, label: str) -> Path:
    path = _resolve_path(Path(str(path_value)), repo_root, manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _resolve_path(path: Path, repo_root: Path, manifest_path: Path | None) -> Path:
    if path.is_absolute():
        return path
    repo_relative = repo_root / path
    if repo_relative.exists() or manifest_path is None:
        return repo_relative
    return manifest_path.parent / path


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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty corpus case summary.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _optional_float(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _float_key(value: Any) -> str:
    return f"{float(value):g}"


def _difference(left: Any, right: Any) -> float | None:
    if left in (None, "") or right in (None, ""):
        return None
    return float(left) - float(right)


def _resolve_existing_path(path_value: Any, repo_root: Path, label: str) -> Path:
    path = _resolve_path(Path(str(path_value)), repo_root, None)
    _require_file(path, label)
    return path


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
