"""Bounded baseline training workflow for the first supervised ML slice."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.datasets import (
    prepare_expanded_ml_supervised_pairings,
    prepare_first_ml_supervised_pairings,
)
from tomobench.evaluation.metrics import mean_absolute_error, root_mean_squared_error
from tomobench.generation.velocity_grids import fault_plane_signed_offset_km
from tomobench.models.baselines import (
    FeatureDistanceWeightedTargetRegressor,
    MeanTargetRegressor,
    RidgeReducedTargetRegressor,
)
from tomobench.utils.paths import get_repo_root


@dataclass(frozen=True)
class BaselineTrainingOutputs:
    """Artifacts written by the first baseline workflow."""

    feature_matrix_csv: Path
    metrics_json: Path
    experiment_note: Path
    predictions_dir: Path
    model_details_json: Path | None = None


@dataclass(frozen=True)
class ExpandedBaselineSuiteOutputs:
    """Artifacts written by the expanded baseline comparison workflow."""

    mean_target_metrics_json: Path
    feature_distance_metrics_json: Path
    reduced_target_metrics_json: Path
    summary_json: Path
    summary_note: Path


def planned_training_note() -> str:
    """Return the current scope note for training workflows."""
    return (
        "The current bounded ML workflow includes a mean-target baseline and a "
        "feature-distance weighted baseline, plus a reduced-target ridge baseline, "
        "all consuming the supervised pairing manifest under leave-one-scenario-out "
        "evaluation."
    )


def train_first_ml_baseline(
    settings: BenchmarkSettings | None = None,
) -> BaselineTrainingOutputs:
    """Train and evaluate the first interpretable baseline on paired artifacts."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    pairing_manifest_path = _resolve_repo_path(
        loaded_settings.machine_learning.first_baseline.pairing_manifest_path,
        repo_root,
    )
    pairing_manifest_path = _ensure_pairing_manifest(
        loaded_settings,
        pairing_manifest_path,
        repo_root,
    )
    pairing_manifest = _read_json(pairing_manifest_path)
    evaluation_protocol = _evaluation_protocol(pairing_manifest)
    baseline_settings = loaded_settings.machine_learning.first_baseline
    output_dir = _resolve_repo_path(baseline_settings.output_dir, repo_root)
    feature_matrix_csv = output_dir / "feature_matrix.csv"
    metrics_json = output_dir / "metrics.json"
    predictions_dir = output_dir / "predictions"
    experiment_note = (
        repo_root
        / loaded_settings.outputs.experiment_notes_dir
        / f"{baseline_settings.experiment_id}.md"
    )

    pairing_items = _dict_list(pairing_manifest["items"])
    samples = tuple(
        _load_training_sample(item, repo_root, loaded_settings) for item in pairing_items
    )
    _write_feature_matrix_csv(samples, feature_matrix_csv)

    fold_metrics: list[dict[str, Any]] = []
    for held_out_index, held_out_sample in enumerate(samples):
        training_targets = [
            sample.target_vector for index, sample in enumerate(samples) if index != held_out_index
        ]
        model = MeanTargetRegressor.fit(training_targets)
        prediction = model.predict()
        prediction_path = _save_prediction(
            scenario=held_out_sample.scenario,
            prediction=prediction,
            path=predictions_dir / f"{held_out_sample.scenario}.prediction.json",
        )
        fold_metrics.append(
            {
                "scenario": held_out_sample.scenario,
                "pairing_item_id": held_out_sample.pairing_item_id,
                "train_sample_count": len(training_targets),
                "target_vector_length": len(held_out_sample.target_vector),
                "mae": mean_absolute_error(held_out_sample.target_vector, prediction),
                "rmse": root_mean_squared_error(held_out_sample.target_vector, prediction),
                "prediction_path": _relative_to_repo_or_absolute(prediction_path, repo_root),
            }
        )

    metrics_payload = {
        "artifact_type": "ml_baseline_metrics",
        "experiment_id": baseline_settings.experiment_id,
        "model_name": baseline_settings.model_name,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pairing_manifest_path": _relative_to_repo_or_absolute(pairing_manifest_path, repo_root),
        "evaluation_protocol": evaluation_protocol,
        "sample_count": len(samples),
        "feature_columns": list(_feature_fieldnames()),
        "fold_metrics": fold_metrics,
        "aggregate_metrics": {
            "mean_mae": sum(metric["mae"] for metric in fold_metrics) / len(fold_metrics),
            "mean_rmse": sum(metric["rmse"] for metric in fold_metrics) / len(fold_metrics),
        },
        "notes": [
            "This first baseline predicts the element-wise mean target grid from the training scenarios only.",
            "The workflow is intended to validate supervised ML handoff and artifact traceability, not to claim strong inversion skill.",
            "Each held-out fold leaves one of the three current scenarios out for reviewable baseline error reporting.",
        ],
    }
    _save_json(metrics_payload, metrics_json)
    _write_experiment_note(
        path=experiment_note,
        settings=loaded_settings,
        metrics_payload=metrics_payload,
        feature_matrix_csv=feature_matrix_csv,
        metrics_json=metrics_json,
        predictions_dir=predictions_dir,
        repo_root=repo_root,
    )
    return BaselineTrainingOutputs(
        feature_matrix_csv=feature_matrix_csv,
        metrics_json=metrics_json,
        experiment_note=experiment_note,
        predictions_dir=predictions_dir,
        model_details_json=None,
    )


def train_feature_distance_ml_baseline(
    settings: BenchmarkSettings | None = None,
) -> BaselineTrainingOutputs:
    """Train and evaluate the first feature-using baseline on paired artifacts."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    baseline_settings = loaded_settings.machine_learning.feature_baseline
    pairing_manifest_path = _resolve_repo_path(
        baseline_settings.pairing_manifest_path,
        repo_root,
    )
    pairing_manifest_path = _ensure_pairing_manifest(
        loaded_settings,
        pairing_manifest_path,
        repo_root,
    )
    pairing_manifest = _read_json(pairing_manifest_path)
    evaluation_protocol = _evaluation_protocol(pairing_manifest)
    output_dir = _resolve_repo_path(baseline_settings.output_dir, repo_root)
    feature_matrix_csv = output_dir / "feature_matrix.csv"
    metrics_json = output_dir / "metrics.json"
    predictions_dir = output_dir / "predictions"
    experiment_note = (
        repo_root
        / loaded_settings.outputs.experiment_notes_dir
        / f"{baseline_settings.experiment_id}.md"
    )

    pairing_items = _dict_list(pairing_manifest["items"])
    samples = tuple(
        _load_training_sample(item, repo_root, loaded_settings) for item in pairing_items
    )
    _write_feature_matrix_csv(samples, feature_matrix_csv)

    fold_metrics: list[dict[str, Any]] = []
    for held_out_index, held_out_sample in enumerate(samples):
        training_samples = [
            sample for index, sample in enumerate(samples) if index != held_out_index
        ]
        model = FeatureDistanceWeightedTargetRegressor.fit(
            feature_vectors=[sample.feature_vector for sample in training_samples],
            target_vectors=[sample.target_vector for sample in training_samples],
        )
        prediction, distances, weights = model.predict(held_out_sample.feature_vector)
        prediction_path = _save_prediction(
            scenario=held_out_sample.scenario,
            prediction=prediction,
            path=predictions_dir / f"{held_out_sample.scenario}.prediction.json",
        )
        fold_metrics.append(
            {
                "scenario": held_out_sample.scenario,
                "pairing_item_id": held_out_sample.pairing_item_id,
                "train_sample_count": len(training_samples),
                "target_vector_length": len(held_out_sample.target_vector),
                "mae": mean_absolute_error(held_out_sample.target_vector, prediction),
                "rmse": root_mean_squared_error(held_out_sample.target_vector, prediction),
                "prediction_path": _relative_to_repo_or_absolute(prediction_path, repo_root),
                "training_scenarios": [sample.scenario for sample in training_samples],
                "training_feature_distances": list(distances),
                "training_feature_weights": list(weights),
            }
        )

    metrics_payload = {
        "artifact_type": "ml_baseline_metrics",
        "experiment_id": baseline_settings.experiment_id,
        "model_name": baseline_settings.model_name,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pairing_manifest_path": _relative_to_repo_or_absolute(pairing_manifest_path, repo_root),
        "evaluation_protocol": evaluation_protocol,
        "sample_count": len(samples),
        "feature_columns": list(_feature_fieldnames()),
        "fold_metrics": fold_metrics,
        "aggregate_metrics": {
            "mean_mae": sum(metric["mae"] for metric in fold_metrics) / len(fold_metrics),
            "mean_rmse": sum(metric["rmse"] for metric in fold_metrics) / len(fold_metrics),
        },
        "notes": [
            "This baseline predicts the held-out target grid by inverse-distance weighting in standardized feature space.",
            "The workflow remains intentionally small because only three scenario-level paired samples are currently available.",
            "Feature distances and weights are saved per fold so the scenario-level prediction logic remains reviewable.",
        ],
    }
    _save_json(metrics_payload, metrics_json)
    _write_experiment_note(
        path=experiment_note,
        settings=loaded_settings,
        metrics_payload=metrics_payload,
        feature_matrix_csv=feature_matrix_csv,
        metrics_json=metrics_json,
        predictions_dir=predictions_dir,
        repo_root=repo_root,
        title="Feature-Distance Weighted Baseline",
        objective=(
            "Evaluate a simple feature-using baseline that predicts the held-out target "
            "velocity grid from distances in scenario-level observation feature space."
        ),
        limitations=(
            "Only three scenario-level samples are currently available in the first supervised batch.",
            "Distance weighting in feature space is still a very small baseline rather than a strong inversion model.",
            "The current features are summary statistics of the observation dataset and may be too coarse for richer target variation.",
        ),
        next_step=(
            "Compare this feature-using baseline against the mean-target baseline and then decide "
            "whether the next slice should expand feature engineering or move to a simple linear model "
            "on a reduced target representation."
        ),
    )
    return BaselineTrainingOutputs(
        feature_matrix_csv=feature_matrix_csv,
        metrics_json=metrics_json,
        experiment_note=experiment_note,
        predictions_dir=predictions_dir,
        model_details_json=None,
    )


def train_reduced_target_ridge_ml_baseline(
    settings: BenchmarkSettings | None = None,
) -> BaselineTrainingOutputs:
    """Train and evaluate a linear ridge model on a reduced target representation."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    baseline_settings = loaded_settings.machine_learning.reduced_target_baseline
    pairing_manifest_path = _resolve_repo_path(
        baseline_settings.pairing_manifest_path,
        repo_root,
    )
    pairing_manifest_path = _ensure_pairing_manifest(
        loaded_settings,
        pairing_manifest_path,
        repo_root,
    )
    pairing_manifest = _read_json(pairing_manifest_path)
    evaluation_protocol = _evaluation_protocol(pairing_manifest)
    output_dir = _resolve_repo_path(baseline_settings.output_dir, repo_root)
    feature_matrix_csv = output_dir / "feature_matrix.csv"
    reduced_target_csv = output_dir / "reduced_targets.csv"
    metrics_json = output_dir / "metrics.json"
    model_details_json = output_dir / "model_details.json"
    predictions_dir = output_dir / "predictions"
    experiment_note = (
        repo_root
        / loaded_settings.outputs.experiment_notes_dir
        / f"{baseline_settings.experiment_id}.md"
    )

    pairing_items = _dict_list(pairing_manifest["items"])
    samples = tuple(
        _load_training_sample(item, repo_root, loaded_settings) for item in pairing_items
    )
    _write_feature_matrix_csv(samples, feature_matrix_csv)
    _write_reduced_target_csv(samples, reduced_target_csv)

    fold_metrics: list[dict[str, Any]] = []
    fold_models: list[dict[str, Any]] = []
    for held_out_index, held_out_sample in enumerate(samples):
        training_samples = [
            sample for index, sample in enumerate(samples) if index != held_out_index
        ]
        model = RidgeReducedTargetRegressor.fit(
            feature_vectors=[sample.feature_vector for sample in training_samples],
            target_vectors=[sample.reduced_target_vector for sample in training_samples],
            ridge_alpha=baseline_settings.ridge_alpha,
        )
        prediction = model.predict(held_out_sample.feature_vector)
        prediction_path = _save_reduced_target_prediction(
            scenario=held_out_sample.scenario,
            field_names=_numeric_reduced_target_fieldnames(),
            prediction=prediction,
            path=predictions_dir / f"{held_out_sample.scenario}.prediction.json",
        )
        fold_metrics.append(
            {
                "scenario": held_out_sample.scenario,
                "pairing_item_id": held_out_sample.pairing_item_id,
                "train_sample_count": len(training_samples),
                "reduced_target_length": len(held_out_sample.reduced_target_vector),
                "mae": mean_absolute_error(held_out_sample.reduced_target_vector, prediction),
                "rmse": root_mean_squared_error(held_out_sample.reduced_target_vector, prediction),
                "prediction_path": _relative_to_repo_or_absolute(prediction_path, repo_root),
                "training_scenarios": [sample.scenario for sample in training_samples],
            }
        )
        fold_models.append(
            {
                "held_out_scenario": held_out_sample.scenario,
                "feature_means": list(model.feature_means),
                "feature_stds": list(model.feature_stds),
                "coefficients_by_output": {
                    output_name: list(coefficients)
                    for output_name, coefficients in zip(
                        _numeric_reduced_target_fieldnames(),
                        model.coefficients_by_output,
                        strict=True,
                    )
                },
                "intercepts_by_output": {
                    output_name: intercept
                    for output_name, intercept in zip(
                        _numeric_reduced_target_fieldnames(),
                        model.intercepts_by_output,
                        strict=True,
                    )
                },
                "feature_names": list(_numeric_feature_fieldnames()),
            }
        )

    metrics_payload = {
        "artifact_type": "ml_baseline_metrics",
        "experiment_id": baseline_settings.experiment_id,
        "model_name": baseline_settings.model_name,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pairing_manifest_path": _relative_to_repo_or_absolute(pairing_manifest_path, repo_root),
        "evaluation_protocol": evaluation_protocol,
        "sample_count": len(samples),
        "feature_columns": list(_feature_fieldnames()),
        "reduced_target_columns": list(_reduced_target_fieldnames()),
        "fold_metrics": fold_metrics,
        "aggregate_metrics": {
            "mean_mae": sum(metric["mae"] for metric in fold_metrics) / len(fold_metrics),
            "mean_rmse": sum(metric["rmse"] for metric in fold_metrics) / len(fold_metrics),
        },
        "notes": [
            "This baseline fits a ridge-style linear regressor from scenario-level features to a small reduced target summary.",
            "The reduced target contains layer means plus bounded block, fault, salt-dome, and dyke summary terms.",
            "Metrics are reported on the reduced target only and are not directly comparable to full-grid RMSE values from earlier baselines.",
        ],
    }
    _save_json(metrics_payload, metrics_json)
    _save_json(
        {
            "artifact_type": "ml_reduced_target_ridge_model_details",
            "experiment_id": baseline_settings.experiment_id,
            "ridge_alpha": baseline_settings.ridge_alpha,
            "fold_models": fold_models,
        },
        model_details_json,
    )
    _write_experiment_note(
        path=experiment_note,
        settings=loaded_settings,
        metrics_payload=metrics_payload,
        feature_matrix_csv=feature_matrix_csv,
        metrics_json=metrics_json,
        predictions_dir=predictions_dir,
        repo_root=repo_root,
        title="Reduced-Target Ridge Baseline",
        objective=(
            "Evaluate a small linear baseline from scenario-level observation features to a "
            "physically interpretable reduced target summary."
        ),
        limitations=(
            "Only three scenario-level samples are currently available in the first supervised batch.",
            "The model is evaluated on a reduced target summary rather than the full target grid.",
            "Reduced-target metrics are useful for bounded model comparison, but they are not final inversion metrics.",
        ),
        next_step=(
            "Compare the reduced-target ridge metrics with the earlier baselines and then decide "
            "whether to expand the reduced target, add more scenarios, or attempt a full-grid linear model "
            "only after the sample count grows."
        ),
    )
    return BaselineTrainingOutputs(
        feature_matrix_csv=feature_matrix_csv,
        metrics_json=metrics_json,
        experiment_note=experiment_note,
        predictions_dir=predictions_dir,
        model_details_json=model_details_json,
    )


def train_expanded_ml_baseline_suite(
    settings: BenchmarkSettings | None = None,
) -> ExpandedBaselineSuiteOutputs:
    """Retrain bounded baselines on the expanded manifest and write one comparison summary."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    prepare_expanded_ml_supervised_pairings(loaded_settings)
    evaluation_settings = loaded_settings.machine_learning.expanded_baseline_evaluation
    output_dir = _resolve_repo_path(evaluation_settings.output_dir, repo_root)
    summary_json = output_dir / "summary.json"
    summary_note = (
        repo_root
        / loaded_settings.outputs.experiment_notes_dir
        / f"{evaluation_settings.summary_experiment_id}.md"
    )

    expanded_settings = replace(
        loaded_settings,
        machine_learning=replace(
            loaded_settings.machine_learning,
            first_supervised_pairing=replace(
                loaded_settings.machine_learning.first_supervised_pairing,
                manifest_path=evaluation_settings.pairing_manifest_path,
            ),
            first_baseline=replace(
                loaded_settings.machine_learning.first_baseline,
                experiment_id=evaluation_settings.mean_target_experiment_id,
                pairing_manifest_path=evaluation_settings.pairing_manifest_path,
                output_dir=output_dir / evaluation_settings.mean_target_experiment_id,
            ),
            feature_baseline=replace(
                loaded_settings.machine_learning.feature_baseline,
                experiment_id=evaluation_settings.feature_distance_experiment_id,
                pairing_manifest_path=evaluation_settings.pairing_manifest_path,
                output_dir=output_dir / evaluation_settings.feature_distance_experiment_id,
            ),
            reduced_target_baseline=replace(
                loaded_settings.machine_learning.reduced_target_baseline,
                experiment_id=evaluation_settings.reduced_target_experiment_id,
                pairing_manifest_path=evaluation_settings.pairing_manifest_path,
                output_dir=output_dir / evaluation_settings.reduced_target_experiment_id,
            ),
        ),
    )

    mean_outputs = train_first_ml_baseline(expanded_settings)
    feature_outputs = train_feature_distance_ml_baseline(expanded_settings)
    reduced_outputs = train_reduced_target_ridge_ml_baseline(expanded_settings)

    mean_metrics = _read_json(mean_outputs.metrics_json)
    feature_metrics = _read_json(feature_outputs.metrics_json)
    reduced_metrics = _read_json(reduced_outputs.metrics_json)
    preferred_full_grid_model = (
        feature_metrics["model_name"]
        if float(feature_metrics["aggregate_metrics"]["mean_rmse"])
        <= float(mean_metrics["aggregate_metrics"]["mean_rmse"])
        else mean_metrics["model_name"]
    )
    summary_payload = {
        "artifact_type": "ml_expanded_baseline_summary",
        "summary_experiment_id": evaluation_settings.summary_experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pairing_manifest_path": _relative_to_repo_or_absolute(
            _resolve_repo_path(evaluation_settings.pairing_manifest_path, repo_root),
            repo_root,
        ),
        "expanded_case_count": _expanded_case_count(
            _resolve_repo_path(evaluation_settings.pairing_manifest_path, repo_root)
        ),
        "included_families": _expanded_unique_scenarios(
            _resolve_repo_path(evaluation_settings.pairing_manifest_path, repo_root)
        ),
        "full_grid_baselines": {
            "mean_target_regressor": {
                "metrics_json": _relative_to_repo_or_absolute(mean_outputs.metrics_json, repo_root),
                "mean_mae": mean_metrics["aggregate_metrics"]["mean_mae"],
                "mean_rmse": mean_metrics["aggregate_metrics"]["mean_rmse"],
            },
            "feature_distance_weighted_target_regressor": {
                "metrics_json": _relative_to_repo_or_absolute(
                    feature_outputs.metrics_json, repo_root
                ),
                "mean_mae": feature_metrics["aggregate_metrics"]["mean_mae"],
                "mean_rmse": feature_metrics["aggregate_metrics"]["mean_rmse"],
            },
        },
        "reduced_target_baseline": {
            "model_name": reduced_metrics["model_name"],
            "metrics_json": _relative_to_repo_or_absolute(reduced_outputs.metrics_json, repo_root),
            "mean_mae": reduced_metrics["aggregate_metrics"]["mean_mae"],
            "mean_rmse": reduced_metrics["aggregate_metrics"]["mean_rmse"],
            "note": "Reduced-target metrics are not directly comparable to full-grid metrics.",
        },
        "preferred_current_path": {
            "model_name": preferred_full_grid_model,
            "reason": (
                "It remains within the frozen full-grid target contract and performs best among "
                "the directly comparable full-grid baselines on the expanded manifest."
            ),
        },
        "assumptions": [
            "Expanded-case evaluation is leave-one-pairing-item-out because multiple cases now exist within each scenario family.",
            "The current preferred path is selected from directly comparable full-grid baselines only.",
            "The reduced-target ridge baseline remains useful as an interpretable auxiliary check rather than the main path toward the frozen full-grid goal.",
            "Salt-dome and dyke-intrusion cases are represented as bounded synthetic velocity overprints on the same node-centered Cartesian grid rather than as full geological process models.",
        ],
        "limitations": [
            "The expanded dataset still contains only a small number of scenario-level cases relative to the frozen full-grid target dimensionality.",
            "Salt-dome and dyke-intrusion geometries are simplified synthetic families chosen to increase controlled diversity rather than to replicate full field complexity.",
            "The reduced-target baseline remains an auxiliary summary model and its metrics should not be read as full-grid inversion performance.",
        ],
    }
    _save_json(summary_payload, summary_json)
    _write_expanded_summary_note(summary_note, summary_payload, repo_root)
    return ExpandedBaselineSuiteOutputs(
        mean_target_metrics_json=mean_outputs.metrics_json,
        feature_distance_metrics_json=feature_outputs.metrics_json,
        reduced_target_metrics_json=reduced_outputs.metrics_json,
        summary_json=summary_json,
        summary_note=summary_note,
    )


@dataclass(frozen=True)
class TrainingSample:
    """One scenario-level sample assembled from the pairing manifest."""

    pairing_item_id: str
    scenario: str
    feature_row: dict[str, float | str]
    feature_vector: tuple[float, ...]
    target_vector: tuple[float, ...]
    reduced_target_row: dict[str, float | str]
    reduced_target_vector: tuple[float, ...]


def _load_training_sample(
    item: dict[str, Any],
    repo_root: Path,
    settings: BenchmarkSettings,
) -> TrainingSample:
    dataset_csv_path = _resolve_repo_path(Path(str(item["input_dataset_csv_path"])), repo_root)
    target_grid_path = _resolve_repo_path(Path(str(item["target_velocity_grid_path"])), repo_root)
    feature_row = _summarize_observation_dataset(dataset_csv_path)
    target_grid = _read_json(target_grid_path)
    reduced_target_row = _summarize_target_grid(target_grid, settings)
    return TrainingSample(
        pairing_item_id=str(item["pairing_item_id"]),
        scenario=str(item["scenario"]),
        feature_row={
            "pairing_item_id": str(item["pairing_item_id"]),
            "scenario": str(item["scenario"]),
            **feature_row,
        },
        feature_vector=tuple(feature_row[field] for field in _numeric_feature_fieldnames()),
        target_vector=tuple(float(value) for value in target_grid["p_velocity_km_per_s"]),
        reduced_target_row={
            "pairing_item_id": str(item["pairing_item_id"]),
            "scenario": str(item["scenario"]),
            **reduced_target_row,
        },
        reduced_target_vector=tuple(
            reduced_target_row[field] for field in _numeric_reduced_target_fieldnames()
        ),
    )


def _summarize_observation_dataset(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Observation dataset is empty: {path}")
    travel_times = [float(row["travel_time_s"]) for row in rows]
    path_lengths = [float(row["path_length_km"]) for row in rows]
    source_depths = [float(row["source_z_km"]) for row in rows]
    horizontal_offsets = [
        math.sqrt(
            (float(row["source_x_km"]) - float(row["receiver_x_km"])) ** 2
            + (float(row["source_y_km"]) - float(row["receiver_y_km"])) ** 2
        )
        for row in rows
    ]
    return {
        "observation_count": float(len(rows)),
        "travel_time_mean_s": _mean(travel_times),
        "travel_time_min_s": min(travel_times),
        "travel_time_max_s": max(travel_times),
        "path_length_mean_km": _mean(path_lengths),
        "path_length_std_km": _population_std(path_lengths),
        "source_depth_mean_km": _mean(source_depths),
        "source_depth_std_km": _population_std(source_depths),
        "horizontal_offset_mean_km": _mean(horizontal_offsets),
        "horizontal_offset_std_km": _population_std(horizontal_offsets),
    }


def _summarize_target_grid(
    target_grid: dict[str, Any],
    settings: BenchmarkSettings,
) -> dict[str, float]:
    x_coordinates = [float(value) for value in target_grid["x_coordinates_km"]]
    y_coordinates = [float(value) for value in target_grid["y_coordinates_km"]]
    z_coordinates = [float(value) for value in target_grid["z_coordinates_km"]]
    velocities = [float(value) for value in target_grid["p_velocity_km_per_s"]]
    nx = len(x_coordinates)
    ny = len(y_coordinates)
    layer_means: list[float] = []
    depth_boundaries = settings.velocity_model_generation.layered.depth_boundaries_km
    for layer_index in range(settings.velocity_model_generation.layered.layer_count):
        top_depth = depth_boundaries[layer_index]
        bottom_depth = depth_boundaries[layer_index + 1]
        layer_values: list[float] = []
        for iz, z_km in enumerate(z_coordinates):
            in_layer = top_depth <= z_km < bottom_depth or (
                layer_index == settings.velocity_model_generation.layered.layer_count - 1
                and z_km == bottom_depth
            )
            if not in_layer:
                continue
            start = nx * ny * iz
            end = start + (nx * ny)
            layer_values.extend(velocities[start:end])
        layer_means.append(_mean(layer_values))

    block = settings.velocity_model_generation.block_anomaly
    salt_dome = settings.velocity_model_generation.salt_dome
    dyke = settings.velocity_model_generation.dyke_intrusion
    x_half, y_half, z_half = (component / 2.0 for component in block.size_km)
    block_values: list[float] = []
    salt_values: list[float] = []
    dyke_values: list[float] = []
    negative_side_values: list[float] = []
    positive_side_values: list[float] = []
    dyke_strike_rad = math.radians(dyke.strike_deg)
    dyke_along_x = math.cos(dyke_strike_rad)
    dyke_along_y = math.sin(dyke_strike_rad)
    dyke_normal_x = -dyke_along_y
    dyke_normal_y = dyke_along_x
    for iz, z_km in enumerate(z_coordinates):
        for iy, y_km in enumerate(y_coordinates):
            for ix, x_km in enumerate(x_coordinates):
                index = ix + nx * (iy + ny * iz)
                velocity = velocities[index]
                if (
                    fault_plane_signed_offset_km(
                        x_km,
                        y_km,
                        z_km,
                        settings.velocity_model_generation.faulted,
                    )
                    < 0.0
                ):
                    negative_side_values.append(velocity)
                else:
                    positive_side_values.append(velocity)
                if (
                    block.center_km[0] - x_half <= x_km <= block.center_km[0] + x_half
                    and block.center_km[1] - y_half <= y_km <= block.center_km[1] + y_half
                    and block.center_km[2] - z_half <= z_km <= block.center_km[2] + z_half
                ):
                    block_values.append(velocity)
                normalized_salt_radius = (
                    ((x_km - salt_dome.center_km[0]) / salt_dome.radii_km[0]) ** 2
                    + ((y_km - salt_dome.center_km[1]) / salt_dome.radii_km[1]) ** 2
                    + ((z_km - salt_dome.center_km[2]) / salt_dome.radii_km[2]) ** 2
                )
                if normalized_salt_radius <= 1.0:
                    salt_values.append(velocity)
                dyke_dx = x_km - dyke.center_km[0]
                dyke_dy = y_km - dyke.center_km[1]
                dyke_along_distance = abs(dyke_dx * dyke_along_x + dyke_dy * dyke_along_y)
                dyke_normal_distance = abs(dyke_dx * dyke_normal_x + dyke_dy * dyke_normal_y)
                if (
                    dyke.top_depth_km <= z_km <= dyke.bottom_depth_km
                    and dyke_along_distance <= (dyke.length_km / 2.0)
                    and dyke_normal_distance <= (dyke.width_km / 2.0)
                ):
                    dyke_values.append(velocity)

    summary = {
        f"layer_mean_velocity_L{layer_index + 1:02d}_km_per_s": value
        for layer_index, value in enumerate(layer_means)
    }
    summary["block_window_mean_velocity_km_per_s"] = _mean(block_values)
    summary["fault_side_velocity_contrast_km_per_s"] = _mean(positive_side_values) - _mean(
        negative_side_values
    )
    summary["salt_dome_window_mean_velocity_km_per_s"] = _mean(salt_values)
    summary["dyke_window_mean_velocity_km_per_s"] = _mean(dyke_values)
    return summary


def _write_feature_matrix_csv(samples: tuple[TrainingSample, ...], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_feature_fieldnames())
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.feature_row)
    return path


def _write_reduced_target_csv(samples: tuple[TrainingSample, ...], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_reduced_target_fieldnames())
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.reduced_target_row)
    return path


def _save_prediction(scenario: str, prediction: tuple[float, ...], path: Path) -> Path:
    payload = {
        "artifact_type": "ml_baseline_prediction",
        "scenario": scenario,
        "predicted_target_vector_length": len(prediction),
        "predicted_p_velocity_km_per_s": list(prediction),
    }
    return _save_json(payload, path)


def _save_reduced_target_prediction(
    scenario: str,
    field_names: tuple[str, ...],
    prediction: tuple[float, ...],
    path: Path,
) -> Path:
    payload = {
        "artifact_type": "ml_reduced_target_prediction",
        "scenario": scenario,
        "predicted_reduced_target": {
            field_name: value for field_name, value in zip(field_names, prediction, strict=True)
        },
    }
    return _save_json(payload, path)


def _write_experiment_note(
    path: Path,
    settings: BenchmarkSettings,
    metrics_payload: dict[str, Any],
    feature_matrix_csv: Path,
    metrics_json: Path,
    predictions_dir: Path,
    repo_root: Path,
    title: str = "Mean-Target Baseline",
    objective: str = (
        "Validate the new supervised pairing layer by training a minimal interpretable "
        "baseline that predicts the element-wise mean target velocity grid from the "
        "training scenarios."
    ),
    limitations: tuple[str, ...] = (
        "Only three scenario-level samples are currently available in the first supervised batch.",
        "The mean-target regressor does not yet use the feature matrix during prediction.",
        "This workflow validates the data handoff and evaluation plumbing, not final inversion capability.",
    ),
    next_step: str = (
        "Implement the next simple baseline that actually uses the scenario-level feature "
        "matrix, while keeping the same pairing manifest and leave-one-scenario-out evaluation."
    ),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Experiment: First ML {title}",
        "",
        f"Experiment ID: `{metrics_payload['experiment_id']}`",
        "",
        "## Objective",
        "",
        objective,
        "",
        "## Inputs Used",
        "",
        f"- Pairing manifest: `{metrics_payload['pairing_manifest_path']}`",
        f"- Model: `{metrics_payload['model_name']}`",
        f"- Evaluation protocol: `{metrics_payload['evaluation_protocol']}`",
        "",
        "## Outputs Created",
        "",
        f"- Feature matrix: `{_relative_to_repo_or_absolute(feature_matrix_csv, repo_root)}`",
        f"- Metrics JSON: `{_relative_to_repo_or_absolute(metrics_json, repo_root)}`",
        f"- Prediction directory: `{_relative_to_repo_or_absolute(predictions_dir, repo_root)}`",
        "",
        "## Aggregate Metrics",
        "",
        f"- Mean MAE: `{metrics_payload['aggregate_metrics']['mean_mae']:.6f}` km/s",
        f"- Mean RMSE: `{metrics_payload['aggregate_metrics']['mean_rmse']:.6f}` km/s",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in limitations)
    lines.extend(
        [
            "",
            "## Next Recommended Step",
            "",
            next_step,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _population_std(values: list[float]) -> float:
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def _feature_fieldnames() -> tuple[str, ...]:
    return (
        "pairing_item_id",
        "scenario",
        *_numeric_feature_fieldnames(),
    )


def _numeric_feature_fieldnames() -> tuple[str, ...]:
    return (
        "observation_count",
        "travel_time_mean_s",
        "travel_time_min_s",
        "travel_time_max_s",
        "path_length_mean_km",
        "path_length_std_km",
        "source_depth_mean_km",
        "source_depth_std_km",
        "horizontal_offset_mean_km",
        "horizontal_offset_std_km",
    )


def _reduced_target_fieldnames() -> tuple[str, ...]:
    return (
        "pairing_item_id",
        "scenario",
        *_numeric_reduced_target_fieldnames(),
    )


def _numeric_reduced_target_fieldnames() -> tuple[str, ...]:
    return (
        "layer_mean_velocity_L01_km_per_s",
        "layer_mean_velocity_L02_km_per_s",
        "layer_mean_velocity_L03_km_per_s",
        "layer_mean_velocity_L04_km_per_s",
        "layer_mean_velocity_L05_km_per_s",
        "block_window_mean_velocity_km_per_s",
        "fault_side_velocity_contrast_km_per_s",
        "salt_dome_window_mean_velocity_km_per_s",
        "dyke_window_mean_velocity_km_per_s",
    )


def _save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return _dict(json.load(handle))


def _dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object.")
    return dict(value)


def _dict_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("Expected a list of JSON objects.")
    return [_dict(item) for item in value]


def _ensure_pairing_manifest(
    settings: BenchmarkSettings,
    manifest_path: Path,
    repo_root: Path,
) -> Path:
    first_manifest_path = _resolve_repo_path(
        settings.machine_learning.first_supervised_pairing.manifest_path,
        repo_root,
    )
    expanded_manifest_path = _resolve_repo_path(
        settings.machine_learning.expanded_pairing.manifest_path,
        repo_root,
    )
    if manifest_path == expanded_manifest_path:
        prepare_expanded_ml_supervised_pairings(settings)
        return expanded_manifest_path
    if manifest_path == first_manifest_path:
        prepare_first_ml_supervised_pairings(settings)
        return first_manifest_path
    if manifest_path.is_file():
        return manifest_path
    raise ValueError(
        "Pairing manifest does not match a known workflow output and does not exist on disk: "
        f"{manifest_path}"
    )


def _evaluation_protocol(pairing_manifest: dict[str, Any]) -> str:
    item_count = len(_dict_list(pairing_manifest["items"]))
    unique_scenarios = {str(item["scenario"]) for item in _dict_list(pairing_manifest["items"])}
    return (
        "leave_one_scenario_out"
        if len(unique_scenarios) == item_count
        else "leave_one_pairing_item_out"
    )


def _expanded_case_count(path: Path) -> int:
    payload = _read_json(path)
    return int(payload["case_count"])


def _expanded_unique_scenarios(path: Path) -> list[str]:
    payload = _read_json(path)
    return [str(value) for value in payload["unique_scenarios"]]


def _write_expanded_summary_note(path: Path, payload: dict[str, Any], repo_root: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    full_grid = payload["full_grid_baselines"]
    preferred = payload["preferred_current_path"]
    reduced = payload["reduced_target_baseline"]
    lines = [
        "# Experiment: Expanded ML Baseline Suite",
        "",
        f"Experiment ID: `{payload['summary_experiment_id']}`",
        "",
        "## Objective",
        "",
        (
            "Retrain the bounded ML baselines on the expanded paired-sample manifest and make a "
            "concrete recommendation for the current preferred path toward the frozen full-grid target."
        ),
        "",
        "## Inputs Used",
        "",
        f"- Expanded pairing manifest: `{payload['pairing_manifest_path']}`",
        f"- Expanded case count: `{payload['expanded_case_count']}`",
        f"- Included families: `{', '.join(payload['included_families'])}`",
        "",
        "## Full-Grid Comparison",
        "",
        f"- Mean-target baseline RMSE: `{full_grid['mean_target_regressor']['mean_rmse']:.6f}` km/s",
        f"- Feature-distance baseline RMSE: `{full_grid['feature_distance_weighted_target_regressor']['mean_rmse']:.6f}` km/s",
        "",
        "## Reduced-Target Check",
        "",
        f"- Reduced-target ridge RMSE: `{reduced['mean_rmse']:.6f}` km/s",
        f"- Note: {reduced['note']}",
        "",
        "## Current Recommendation",
        "",
        f"- Preferred current path: `{preferred['model_name']}`",
        f"- Reason: {preferred['reason']}",
        "",
        "## Assumptions",
        "",
    ]
    lines.extend(f"- {assumption}" for assumption in payload["assumptions"])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in payload["limitations"])
    lines.extend(
        [
            "",
            "## Next Recommended Step",
            "",
            (
                "Keep the feature-distance weighted full-grid baseline as the current working path for the expanded 13-case manifest, "
                "and treat the reduced-target ridge model as a secondary diagnostic rather than the main inversion path."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_to_repo_or_absolute(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()
