"""Noise robustness audit for the corrected grouped-split comparison."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tomobench.evaluation.metrics import mean_absolute_error, root_mean_squared_error
from tomobench.models.observation_training import (
    ObservationTrainingSample,
    _fit_phase7_variant_model,
    _phase7_ablation_variants,
    _phase7_variant_context,
    _prepare_paper_pseudo_bending_training_context,
    _samples_for_split,
)
from tomobench.tomography.comparison import node_centered_values_to_cell_centered
from tomobench.tomography.inversion import (
    _assemble_dense_g_matrix,
    _cell_count_from_metadata,
    _cell_shape_from_metadata,
    _load_observation_travel_times,
    _load_velocity_grid,
    _ordered_observation_ids,
    _velocity_bounds_from_grid,
    cell_centered_velocity_target,
    cell_coverage_diagnostics,
    smoothing_normal_matrix,
)
from tomobench.tomography.reference import _solve_reference_candidate
from tomobench.tomography.sensitivity import load_sensitivity_records_jsonl
from tomobench.utils.paths import get_repo_root


DEFAULT_NOISE_MANIFEST = Path(
    "outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json"
)
DEFAULT_NOISE_SPLIT = Path(
    "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/target_group_split_assignments.csv"
)
DEFAULT_NOISE_CLASSICAL_DIR = Path(
    "outputs/generated/classical_tomography/fair_reference_ray_corpus_v1"
)
DEFAULT_NOISE_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/noise_robustness_v1"
)
DEFAULT_NOISE_LEVELS_S = (0.0, 0.025, 0.05, 0.1)
DEFAULT_NOISE_RANDOM_SEED = 909
DEFAULT_REFERENCE_DAMPING = 1.0
DEFAULT_REFERENCE_SMOOTHING = 10.0
ML_NOISE_VARIANT_IDS = (
    "full_input_euclidean_path",
    "travel_time_only",
    "full_input",
)


@dataclass(frozen=True)
class GroupedNoiseRobustnessOutputs:
    """Artifacts written by the grouped-split noise audit."""

    output_dir: Path
    case_metrics_csv: Path
    summary_csv: Path
    summary_json: Path
    summary_md: Path
    metadata_json: Path


def run_grouped_split_noise_robustness(
    *,
    manifest_path: Path = DEFAULT_NOISE_MANIFEST,
    grouped_split_path: Path = DEFAULT_NOISE_SPLIT,
    classical_corpus_dir: Path = DEFAULT_NOISE_CLASSICAL_DIR,
    output_dir: Path = DEFAULT_NOISE_OUTPUT_DIR,
    noise_levels_s: Sequence[float] = DEFAULT_NOISE_LEVELS_S,
    noise_random_seed: int = DEFAULT_NOISE_RANDOM_SEED,
    reference_damping: float = DEFAULT_REFERENCE_DAMPING,
    reference_smoothing: float = DEFAULT_REFERENCE_SMOOTHING,
) -> GroupedNoiseRobustnessOutputs:
    """Evaluate selected ML and reference-ray methods with controlled time noise.

    The ML models are fitted once on the clean grouped training cases.  Noise is
    then added only to the ordered travel-time features for the fixed grouped
    test cases.  The reference-ray solve uses the same perturbations and the
    validation-frozen damping/smoothing pair.  A zero-noise row is included so
    every degradation is paired to the same case-level baseline.
    """
    levels = tuple(sorted({float(level) for level in noise_levels_s}))
    if not levels or any(level < 0.0 for level in levels):
        raise ValueError("noise_levels_s must contain at least one non-negative value.")
    if reference_damping < 0.0 or reference_smoothing < 0.0:
        raise ValueError("Reference regularization values must be non-negative.")

    repo_root = get_repo_root()
    manifest_resolved = _resolve_path(manifest_path, repo_root)
    split_resolved = _resolve_path(grouped_split_path, repo_root)
    classical_resolved = _resolve_path(classical_corpus_dir, repo_root)
    output_resolved = _resolve_path(output_dir, repo_root)
    manifest = _read_json(manifest_resolved)
    split_by_case, target_hash_by_case = _read_grouped_split(split_resolved)
    test_case_ids = sorted(
        case_id for case_id, split in split_by_case.items() if split == "test"
    )
    if not test_case_ids:
        raise ValueError("The grouped split contains no test cases.")

    settings = _load_settings()
    suite_settings = settings.machine_learning.paper_pseudo_bending_ml_suite
    context = _prepare_paper_pseudo_bending_training_context(
        settings=settings,
        pairing_manifest_path=suite_settings.input_manifest_path,
        observation_fields=suite_settings.observation_fields,
        random_seed=suite_settings.fixed_split_seed,
        split_strategy="target_grouped_family_stratified",
    )
    samples_by_case_id = {sample.case_id: sample for sample in context.samples}
    offset_table = build_noise_offset_table(
        sorted(samples_by_case_id),
        observation_count=context.feature_matrix.observation_count,
        noise_levels_s=levels,
        random_seed=noise_random_seed,
    )

    case_rows: list[dict[str, Any]] = []
    ml_variants = {
        str(variant["variant_id"]): variant
        for variant in _phase7_ablation_variants(suite_settings.observation_fields)
        if str(variant["variant_id"]) in ML_NOISE_VARIANT_IDS
    }
    for variant_id in ML_NOISE_VARIANT_IDS:
        variant = ml_variants[variant_id]
        variant_context = _phase7_variant_context(
            context,
            variant_id=variant_id,
            observation_fields=tuple(str(field) for field in variant["observation_fields"]),
            random_seed=707,
        )
        train_samples = _samples_for_split(
            variant_context.samples,
            variant_context.split_by_case_id,
            "train",
        )
        model = _fit_phase7_variant_model(settings, variant, train_samples)
        for level in levels:
            for sample in _samples_for_split(
                variant_context.samples,
                variant_context.split_by_case_id,
                "test",
            ):
                noisy_sample = apply_noise_offsets_to_sample(
                    sample,
                    observation_fields=tuple(str(field) for field in variant["observation_fields"]),
                    offsets=offset_table[(sample.case_id, level)],
                )
                prediction = model.predict(noisy_sample.feature_vector)
                case_rows.append(
                    _ml_case_row(
                        sample=noisy_sample,
                        prediction=prediction,
                        method_id=variant_id,
                        path_length_source=str(variant.get("path_length_source", "not_included")),
                        noise_std_s=level,
                        target_hash=target_hash_by_case[sample.case_id],
                    )
                )

    manifest_items = {
        str(item["case_id"]): item for item in _manifest_items(manifest)
    }
    for level in levels:
        for case_id in test_case_ids:
            item = manifest_items.get(case_id)
            if item is None:
                raise ValueError(f"Grouped split case is absent from manifest: {case_id}")
            case_rows.append(
                _reference_case_row(
                    item=item,
                    classical_corpus_dir=classical_resolved,
                    noise_std_s=level,
                    offsets=offset_table[(case_id, level)],
                    target_hash=target_hash_by_case[case_id],
                    damping=reference_damping,
                    smoothing=reference_smoothing,
                    repo_root=repo_root,
                )
            )
            case_rows.append(
                _reference_prior_case_row(
                    item=item,
                    classical_corpus_dir=classical_resolved,
                    noise_std_s=level,
                    target_hash=target_hash_by_case[case_id],
                    repo_root=repo_root,
                )
            )

    summary_rows = summarize_noise_rows(case_rows)
    output_resolved.mkdir(parents=True, exist_ok=True)
    case_metrics_csv = output_resolved / "noise_case_metrics.csv"
    summary_csv = output_resolved / "noise_summary.csv"
    summary_json = output_resolved / "noise_robustness_summary.json"
    summary_md = output_resolved / "noise_robustness_summary.md"
    metadata_json = output_resolved / "audit_metadata.json"
    _write_csv(case_metrics_csv, case_rows)
    _write_csv(summary_csv, summary_rows)
    summary_payload = _summary_payload(
        manifest=manifest,
        manifest_path=manifest_resolved,
        grouped_split_path=split_resolved,
        classical_corpus_dir=classical_resolved,
        case_rows=case_rows,
        summary_rows=summary_rows,
        noise_levels_s=levels,
        noise_random_seed=noise_random_seed,
        reference_damping=reference_damping,
        reference_smoothing=reference_smoothing,
        target_hash_by_case=target_hash_by_case,
        context=context,
        repo_root=repo_root,
        case_metrics_csv=case_metrics_csv,
        summary_csv=summary_csv,
    )
    _write_json(summary_json, summary_payload)
    _write_markdown(summary_md, summary_payload, repo_root)
    metadata = {
        "artifact_type": "grouped_split_noise_robustness_audit_metadata",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": "tomobench audit-grouped-split-noise-robustness",
        "manifest_path": _relative_or_absolute(manifest_resolved, repo_root),
        "grouped_split_path": _relative_or_absolute(split_resolved, repo_root),
        "classical_corpus_dir": _relative_or_absolute(classical_resolved, repo_root),
        "noise_levels_s": list(levels),
        "noise_random_seed": noise_random_seed,
        "reference_damping": reference_damping,
        "reference_smoothing": reference_smoothing,
        "split_case_counts": _split_counts(split_by_case),
        "split_unique_target_counts": _split_unique_target_counts(
            split_by_case, target_hash_by_case
        ),
        "python_version": _python_version(),
        "numpy_version": np.__version__,
    }
    _write_json(metadata_json, metadata)
    return GroupedNoiseRobustnessOutputs(
        output_dir=output_resolved,
        case_metrics_csv=case_metrics_csv,
        summary_csv=summary_csv,
        summary_json=summary_json,
        summary_md=summary_md,
        metadata_json=metadata_json,
    )


def build_noise_offset_table(
    case_ids: Sequence[str],
    *,
    observation_count: int,
    noise_levels_s: Sequence[float],
    random_seed: int,
) -> dict[tuple[str, float], tuple[float, ...]]:
    """Create one deterministic, method-independent offset per case/observation."""
    if observation_count <= 0:
        raise ValueError("observation_count must be positive.")
    offsets: dict[tuple[str, float], tuple[float, ...]] = {}
    for level in sorted({float(value) for value in noise_levels_s}):
        rng = random.Random(random_seed + int(round(level * 1000.0)))
        for case_id in sorted(str(value) for value in case_ids):
            offsets[(case_id, level)] = tuple(
                rng.gauss(0.0, level) for _ in range(observation_count)
            )
    return offsets


def apply_noise_offsets_to_sample(
    sample: ObservationTrainingSample,
    *,
    observation_fields: tuple[str, ...],
    offsets: tuple[float, ...],
) -> ObservationTrainingSample:
    """Apply supplied offsets only to travel-time fields in one sample."""
    if "travel_time_s" not in observation_fields:
        if any(abs(value) > 0.0 for value in offsets):
            raise ValueError("Noise offsets cannot be applied without travel_time_s.")
        return sample
    field_count = len(observation_fields)
    if field_count == 0 or len(sample.feature_vector) % field_count != 0:
        raise ValueError("Sample feature vector is inconsistent with observation_fields.")
    observation_count = len(sample.feature_vector) // field_count
    if len(offsets) != observation_count:
        raise ValueError("Noise offset count must equal the sample observation count.")
    travel_time_offset = observation_fields.index("travel_time_s")
    feature_vector = list(sample.feature_vector)
    feature_row = dict(sample.feature_row)
    for observation_index, offset in enumerate(offsets):
        index = observation_index * field_count + travel_time_offset
        value = max(1.0e-9, feature_vector[index] + offset)
        feature_vector[index] = value
        feature_row[f"obs_{observation_index:04d}_travel_time_s"] = value
    return ObservationTrainingSample(
        pairing_item_id=sample.pairing_item_id,
        case_id=sample.case_id,
        scenario=sample.scenario,
        feature_row=feature_row,
        feature_vector=tuple(feature_vector),
        target_vector=sample.target_vector,
        target_grid_path=sample.target_grid_path,
        target_spec_path=sample.target_spec_path,
    )


def summarize_noise_rows(case_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate case metrics and paired degradation relative to zero noise."""
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(str(row["method_id"]), float(row["noise_std_s"]))].append(row)
    zero_by_method = {
        method: rows
        for (method, level), rows in grouped.items()
        if level == 0.0
    }
    summary_rows: list[dict[str, Any]] = []
    for (method, level), rows in sorted(grouped.items()):
        zero_rows = {str(row["case_id"]): row for row in zero_by_method.get(method, [])}
        paired_deltas = [
            float(row["cell_rmse_km_per_s"])
            - float(zero_rows[str(row["case_id"])]["cell_rmse_km_per_s"])
            for row in rows
            if str(row["case_id"]) in zero_rows
        ]
        paired_node_deltas = [
            float(row["node_rmse_km_per_s"])
            - float(zero_rows[str(row["case_id"])]["node_rmse_km_per_s"])
            for row in rows
            if str(row["case_id"]) in zero_rows
            and row.get("node_rmse_km_per_s") is not None
            and zero_rows[str(row["case_id"])].get("node_rmse_km_per_s") is not None
        ]
        summary_rows.append(
            {
                "method_id": method,
                "noise_std_s": level,
                "case_count": len(rows),
                "unique_target_count": len({str(row["target_hash"]) for row in rows}),
                "mean_node_rmse_km_per_s": _mean(rows, "node_rmse_km_per_s"),
                "mean_node_mae_km_per_s": _mean(rows, "node_mae_km_per_s"),
                "mean_cell_rmse_km_per_s": _mean(rows, "cell_rmse_km_per_s"),
                "mean_cell_mae_km_per_s": _mean(rows, "cell_mae_km_per_s"),
                "mean_travel_time_rmse_s": _mean_optional(rows, "travel_time_rmse_s"),
                "mean_clipped_velocity_cell_percentage": _mean_optional(
                    rows, "clipped_velocity_cell_percentage"
                ),
                "paired_cell_rmse_delta_from_zero_noise_km_per_s": _mean_values(paired_deltas),
                "paired_node_rmse_delta_from_zero_noise_km_per_s": _mean_values(
                    paired_node_deltas
                ),
                "paired_case_count": len(paired_deltas),
                "noise_repetition_count": 1,
            }
        )
    return summary_rows


def _ml_case_row(
    *,
    sample: ObservationTrainingSample,
    prediction: tuple[float, ...],
    method_id: str,
    path_length_source: str,
    noise_std_s: float,
    target_hash: str,
) -> dict[str, Any]:
    grid = _read_json(sample.target_grid_path)
    nx = len(grid["x_coordinates_km"])
    ny = len(grid["y_coordinates_km"])
    nz = len(grid["z_coordinates_km"])
    target_cells = node_centered_values_to_cell_centered(sample.target_vector, nx, ny, nz)
    prediction_cells = node_centered_values_to_cell_centered(prediction, nx, ny, nz)
    return {
        "method_id": method_id,
        "method_class": "ml",
        "path_length_source": path_length_source,
        "case_id": sample.case_id,
        "family": sample.scenario,
        "split": "test",
        "target_hash": target_hash,
        "noise_std_s": noise_std_s,
        "node_rmse_km_per_s": root_mean_squared_error(sample.target_vector, prediction),
        "node_mae_km_per_s": mean_absolute_error(sample.target_vector, prediction),
        "cell_rmse_km_per_s": root_mean_squared_error(target_cells, prediction_cells),
        "cell_mae_km_per_s": mean_absolute_error(target_cells, prediction_cells),
        "travel_time_rmse_s": None,
        "clipped_velocity_cell_percentage": None,
    }


def _reference_case_row(
    *,
    item: dict[str, Any],
    classical_corpus_dir: Path,
    noise_std_s: float,
    offsets: tuple[float, ...],
    target_hash: str,
    damping: float,
    smoothing: float,
    repo_root: Path,
) -> dict[str, Any]:
    case_id = str(item["case_id"])
    case_dir = classical_corpus_dir / "cases" / case_id
    summary = _read_json(case_dir / "reference_ray_perturbation_summary.json")
    source_files = summary["source_files"]
    sensitivity_metadata_path = _resolve_path(Path(str(source_files["reference_sensitivity_metadata"])), repo_root)
    sensitivity_path = _resolve_path(Path(str(source_files["reference_sensitivity_sidecar"])), repo_root)
    observations_path = _resolve_path(Path(str(source_files["observed_travel_times"])), repo_root)
    target_grid_path = _resolve_path(Path(str(source_files["true_target_velocity_grid"])), repo_root)
    reference_grid_path = _resolve_path(Path(str(source_files["reference_velocity_grid"])), repo_root)
    metadata = _read_json(sensitivity_metadata_path)
    records = load_sensitivity_records_jsonl(sensitivity_path)
    observations = _load_observation_travel_times(observations_path)
    target_grid = _load_velocity_grid(target_grid_path)
    reference_grid = _load_velocity_grid(reference_grid_path)
    observation_ids = _ordered_observation_ids(records, observations)
    if len(observation_ids) != len(offsets):
        raise ValueError(f"{case_id}: noise offsets do not match observation count.")
    observed_times = np.array(
        [observations[observation_id] + offset for observation_id, offset in zip(observation_ids, offsets, strict=True)]
    )
    cell_count = _cell_count_from_metadata(metadata)
    cell_shape = _cell_shape_from_metadata(metadata, target_grid)
    g_matrix = _assemble_dense_g_matrix(records, observation_ids, cell_count)
    coverage = cell_coverage_diagnostics(records, cell_shape)
    covered_mask = np.array(
        [float(row["total_path_length_coverage_km"]) >= 1.0e-9 for row in coverage],
        dtype=bool,
    )
    target_velocity = np.array(cell_centered_velocity_target(target_grid))
    reference_velocity = np.array(cell_centered_velocity_target(reference_grid))
    reference_slowness = 1.0 / reference_velocity
    bounds = _velocity_bounds_from_grid(target_grid)
    result = _solve_reference_candidate(
        g_matrix=g_matrix,
        observed_travel_times=observed_times,
        target_velocity=target_velocity,
        reference_slowness=reference_slowness,
        covered_mask=covered_mask,
        damping=damping,
        smoothing=smoothing,
        smoothing_ltl=smoothing_normal_matrix(cell_shape),
        velocity_bounds=bounds,
    )
    all_metrics = result["all_cell_reconstruction_metrics"]
    covered_metrics = result["covered_cell_reconstruction_metrics"]
    return {
        "method_id": "optimized_fixed_ray",
        "method_class": "reference_ray",
        "path_length_source": "reference_model_fixed_ray_geometry",
        "case_id": case_id,
        "family": str(item.get("family") or item.get("scenario") or "unknown"),
        "split": "test",
        "target_hash": target_hash,
        "noise_std_s": noise_std_s,
        "node_rmse_km_per_s": None,
        "node_mae_km_per_s": None,
        "cell_rmse_km_per_s": all_metrics["velocity_rmse_km_per_s"],
        "cell_mae_km_per_s": all_metrics["velocity_mae_km_per_s"],
        "covered_cell_rmse_km_per_s": covered_metrics["velocity_rmse_km_per_s"],
        "covered_cell_mae_km_per_s": covered_metrics["velocity_mae_km_per_s"],
        "travel_time_rmse_s": result["residual_metrics"]["travel_time_rmse_s"],
        "clipped_velocity_cell_percentage": result["clipped_velocity_cell_percentage"],
    }


def _reference_prior_case_row(
    *,
    item: dict[str, Any],
    classical_corpus_dir: Path,
    noise_std_s: float,
    target_hash: str,
    repo_root: Path,
) -> dict[str, Any]:
    case_id = str(item["case_id"])
    case_dir = classical_corpus_dir / "cases" / case_id
    summary = _read_json(case_dir / "reference_ray_perturbation_summary.json")
    source_files = summary["source_files"]
    target_grid = _load_velocity_grid(
        _resolve_path(Path(str(source_files["true_target_velocity_grid"])), repo_root)
    )
    reference_grid = _load_velocity_grid(
        _resolve_path(Path(str(source_files["reference_velocity_grid"])), repo_root)
    )
    target = np.array(cell_centered_velocity_target(target_grid))
    reference = np.array(cell_centered_velocity_target(reference_grid))
    return {
        "method_id": "reference_prior",
        "method_class": "reference_prior",
        "path_length_source": "configured_layered_reference_model",
        "case_id": case_id,
        "family": str(item.get("family") or item.get("scenario") or "unknown"),
        "split": "test",
        "target_hash": target_hash,
        "noise_std_s": noise_std_s,
        "node_rmse_km_per_s": None,
        "node_mae_km_per_s": None,
        "cell_rmse_km_per_s": root_mean_squared_error(target.tolist(), reference.tolist()),
        "cell_mae_km_per_s": mean_absolute_error(target.tolist(), reference.tolist()),
        "travel_time_rmse_s": None,
        "clipped_velocity_cell_percentage": None,
    }


def _summary_payload(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    grouped_split_path: Path,
    classical_corpus_dir: Path,
    case_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    noise_levels_s: Sequence[float],
    noise_random_seed: int,
    reference_damping: float,
    reference_smoothing: float,
    target_hash_by_case: dict[str, str],
    context: Any,
    repo_root: Path,
    case_metrics_csv: Path,
    summary_csv: Path,
) -> dict[str, Any]:
    return {
        "artifact_type": "grouped_split_noise_robustness_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_corpus_id": manifest.get("corpus_id") or manifest.get("batch_id"),
        "manifest_path": _relative_or_absolute(manifest_path, repo_root),
        "grouped_split_path": _relative_or_absolute(grouped_split_path, repo_root),
        "classical_corpus_dir": _relative_or_absolute(classical_corpus_dir, repo_root),
        "split_strategy": "target_grouped_family_stratified",
        "split_case_counts": _split_counts(context.split_by_case_id),
        "split_unique_target_counts": _split_unique_target_counts(
            context.split_by_case_id, target_hash_by_case
        ),
        "fixed_test_case_count": len({row["case_id"] for row in case_rows if row["split"] == "test"}),
        "fixed_test_unique_target_count": len({
            row["target_hash"] for row in case_rows if row["split"] == "test"
        }),
        "noise_levels_s": list(noise_levels_s),
        "noise_random_seed": noise_random_seed,
        "noise_application": "shared deterministic Gaussian offsets added only to ordered travel_time_s observations; models and reference regularization are not refit per noise level",
        "reference_regularization": {
            "damping": reference_damping,
            "smoothing": reference_smoothing,
            "selection_scope": "validation_only_before_noise_audit",
        },
        "methods": [
            {
                "method_id": "full_input_euclidean_path",
                "method_class": "ml",
                "path_length_source": "euclidean_source_receiver_distance_km",
                "role": "realistic full-input alternative",
            },
            {
                "method_id": "travel_time_only",
                "method_class": "ml",
                "path_length_source": "not_included",
                "role": "travel-time-only control",
            },
            {
                "method_id": "full_input",
                "method_class": "ml",
                "path_length_source": "true_model_ray_path_length_km",
                "role": "privileged synthetic diagnostic",
            },
            {
                "method_id": "optimized_fixed_ray",
                "method_class": "reference_ray",
                "path_length_source": "reference_model_fixed_ray_geometry",
                "role": "validation-frozen classical baseline",
            },
            {
                "method_id": "reference_prior",
                "method_class": "reference_prior",
                "path_length_source": "configured_layered_reference_model",
                "role": "noise-invariant prior comparator",
            },
        ],
        "summary_rows": list(summary_rows),
        "case_metrics_csv": _relative_or_absolute(case_metrics_csv, repo_root),
        "summary_csv": _relative_or_absolute(summary_csv, repo_root),
        "limitations": [
            "Only one deterministic noise realization is evaluated at each level; this is a controlled sensitivity analysis, not a distributional estimate of picking uncertainty.",
            "Gaussian perturbations do not model correlated, phase-dependent, or outlier picking errors.",
            "The reference-ray method keeps its validation-selected regularization fixed and does not retune on noisy test observations.",
            "All results remain synthetic and target-grouped test performance is based on five unique target models.",
        ],
    }


def _read_grouped_split(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    split_by_case: dict[str, str] = {}
    target_hash_by_case: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case_id = str(row["case_id"])
            split_by_case[case_id] = str(row["split"])
            target_hash_by_case[case_id] = str(row["target_hash"])
    if not split_by_case:
        raise ValueError(f"Grouped split CSV has no rows: {path}")
    return split_by_case, target_hash_by_case


def _manifest_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a non-empty items list.")
    return [item for item in items if isinstance(item, dict)]


def _split_counts(split_by_case: dict[str, str]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for split in split_by_case.values():
        counts[split] = counts.get(split, 0) + 1
    return counts


def _split_unique_target_counts(
    split_by_case: dict[str, str], target_hash_by_case: dict[str, str]
) -> dict[str, int]:
    return {
        split: len({target_hash_by_case[case_id] for case_id, value in split_by_case.items() if value == split})
        for split in ("train", "validation", "test")
    }


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _mean_optional(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    return _mean(rows, key)


def _mean_values(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_settings() -> Any:
    from tomobench.config import load_settings

    return load_settings()


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


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV without a schema: {path}")
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any], repo_root: Path) -> None:
    lines = [
        "# Grouped-Split Noise Robustness Audit",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "This audit applies controlled Gaussian perturbations to travel-time observations after the grouped split and model/regularization choices are frozen.",
        "",
        f"- Fixed test cases: `{payload['fixed_test_case_count']}`",
        f"- Fixed test unique targets: `{payload['fixed_test_unique_target_count']}`",
        f"- Noise seed: `{payload['noise_random_seed']}`",
        f"- Reference damping/smoothing: `({payload['reference_regularization']['damping']}, {payload['reference_regularization']['smoothing']})`, selected on validation before this audit",
        "",
        "## Test-set results",
        "",
        "| Method | Noise (s) | Cases | Cell RMSE | Cell MAE | Paired cell RMSE delta | Travel-time RMSE (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| {method_id} | {noise_std_s:.3f} | {case_count} | {rmse:.6f} | {mae:.6f} | {delta} | {time} |".format(
                method_id=row["method_id"],
                noise_std_s=float(row["noise_std_s"]),
                case_count=row["case_count"],
                rmse=float(row["mean_cell_rmse_km_per_s"]),
                mae=float(row["mean_cell_mae_km_per_s"]),
                delta=(
                    f"{float(row['paired_cell_rmse_delta_from_zero_noise_km_per_s']):.6f}"
                    if row["paired_cell_rmse_delta_from_zero_noise_km_per_s"] is not None
                    else "n/a"
                ),
                time=(
                    f"{float(row['mean_travel_time_rmse_s']):.6f}"
                    if row["mean_travel_time_rmse_s"] is not None
                    else "n/a"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "- `full_input_euclidean_path` is the realistic full-input alternative; it does not use true-model ray-path length.",
            "- `full_input` is retained as a privileged synthetic diagnostic because its path-length feature is derived from the true-model ray trace.",
            "- `optimized_fixed_ray` uses the reference-model fixed-ray geometry and the validation-frozen regularization pair.",
            "- The reported degradation is paired case-level change from the zero-noise row. One deterministic realization is used at each noise level, so no across-repetition uncertainty is claimed.",
            "- Gaussian noise is a controlled sensitivity analysis and is not a complete model of real picking uncertainty.",
            "",
            "Source artifacts:",
            "",
            f"- Case metrics: `{_relative_or_absolute(Path(payload['case_metrics_csv']), repo_root)}`",
            f"- Summary CSV: `{_relative_or_absolute(Path(payload['summary_csv']), repo_root)}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _python_version() -> str:
    import sys

    return sys.version.split()[0]
