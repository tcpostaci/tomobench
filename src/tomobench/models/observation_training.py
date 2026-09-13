"""Observation-driven supervised baseline training for frozen full-grid inversion targets."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.datasets import (
    assemble_observation_case_feature_matrix,
    prepare_observation_ml_supervised_pairings,
    prepare_paper_pseudo_bending_observation_corpus,
)
from tomobench.datasets.target_identity import target_vector_sha256
from tomobench.evaluation.metrics import mean_absolute_error, root_mean_squared_error
from tomobench.models.baselines import (
    MeanTargetRegressor,
    PCALinearObservationRegressor,
    PCAMLPObservationRegressor,
    PCARandomForestObservationRegressor,
    PCARidgeObservationRegressor,
)
from tomobench.models.splitting import target_grouped_family_split
from tomobench.tomography.comparison import (
    compute_ml_cell_centered_case_metrics,
    node_centered_values_to_cell_centered,
)
from tomobench.utils.paths import get_repo_root


@dataclass(frozen=True)
class ObservationBaselineOutputs:
    """Artifacts written by the observation-driven PCA plus ridge workflow."""

    feature_matrix_csv: Path
    metrics_json: Path
    experiment_note: Path
    predictions_dir: Path
    model_details_json: Path


@dataclass(frozen=True)
class ObservationTuningOutputs:
    """Artifacts written by the repeated-split PCA-plus-ridge tuning workflow."""

    summary_json: Path
    summary_note: Path
    candidate_metrics_json: Path


@dataclass(frozen=True)
class ObservationComparisonOutputs:
    """Artifacts written by the limited observation-driven ML comparison workflow."""

    summary_json: Path
    summary_note: Path
    model_output_dirs: tuple[Path, ...]


@dataclass(frozen=True)
class PaperPseudoBendingMLSuiteOutputs:
    """Artifacts written by the paper pseudo-bending ML suite."""

    summary_json: Path
    repeated_summary_json: Path
    leave_one_family_out_summary_json: Path
    noise_robustness_summary_json: Path
    summary_note: Path
    model_metrics_jsons: tuple[Path, ...]
    output_dir: Path
    target_group_split_csv: Path | None = None


@dataclass(frozen=True)
class PaperPseudoBendingMLStrengthenedOutputs:
    """Artifacts written by the audit-driven strengthened ML workflow."""

    summary_json: Path
    summary_note: Path
    output_dir: Path


@dataclass(frozen=True)
class Phase7ObservationSignalAblationOutputs:
    """Artifacts written by the Phase 7 observation-signal ablation suite."""

    output_dir: Path
    summary_json: Path
    summary_csv: Path
    cell_centered_metrics_csv: Path
    leave_one_family_out_summary_json: Path
    summary_note: Path
    variant_metrics_jsons: tuple[Path, ...]


@dataclass(frozen=True)
class Phase7LearningCurveOutputs:
    """Artifacts written by the Phase 7 PCA-linear learning-curve experiment."""

    output_dir: Path
    summary_json: Path
    summary_csv: Path
    repetition_metrics_csv: Path
    family_wise_csv: Path


@dataclass(frozen=True)
class PaperPseudoBendingFaultedGeneralizationOutputs:
    """Artifacts written by the bounded faulted-generalization workflow."""

    diagnosis_json: Path
    summary_json: Path
    summary_note: Path
    output_dir: Path


ObservationBaselineModel = (
    PCALinearObservationRegressor
    | PCARidgeObservationRegressor
    | PCAMLPObservationRegressor
    | PCARandomForestObservationRegressor
)


@dataclass(frozen=True)
class ObservationModelRunConfig:
    """Small shared contract for one observation-driven model run."""

    experiment_id: str
    model_name: str
    pairing_manifest_path: Path
    output_dir: Path
    observation_fields: tuple[str, ...]
    split_strategy: str
    model_builder: Any
    model_details_builder: Any
    note_title: str
    objective: str
    notes: list[str]
    assumptions: list[str]
    limitations: list[str]
    next_step: str


@dataclass(frozen=True)
class ObservationTrainingContext:
    """Prepared observation-driven training data and split assignments."""

    manifest_path: Path
    pairing_manifest: dict[str, Any]
    feature_matrix: Any
    samples: tuple["ObservationTrainingSample", ...]
    split_by_case_id: dict[str, str]


@dataclass(frozen=True)
class ObservationTrainingSample:
    """One case-level training sample with ordered observation features and a frozen target."""

    pairing_item_id: str
    case_id: str
    scenario: str
    feature_row: dict[str, float | str]
    feature_vector: tuple[float, ...]
    target_vector: tuple[float, ...]
    target_grid_path: Path
    target_spec_path: Path


def train_observation_pca_ridge_ml_baseline(
    settings: BenchmarkSettings | None = None,
) -> ObservationBaselineOutputs:
    """Train the first real observation-driven supervised full-grid baseline."""
    loaded_settings = settings or load_settings()
    return _train_observation_baseline(
        settings=loaded_settings,
        run_config=_observation_model_run_config(
            loaded_settings,
            "pca_ridge_observation_regressor",
        ),
    )


def train_observation_pca_linear_ml_baseline(
    settings: BenchmarkSettings | None = None,
) -> ObservationBaselineOutputs:
    """Train the observation-driven PCA plus ordinary linear full-grid baseline."""
    loaded_settings = settings or load_settings()
    return _train_observation_baseline(
        settings=loaded_settings,
        run_config=_observation_model_run_config(
            loaded_settings,
            "pca_linear_observation_regressor",
        ),
    )


def train_observation_pca_mlp_ml_baseline(
    settings: BenchmarkSettings | None = None,
) -> ObservationBaselineOutputs:
    """Train the first shallow non-linear baseline on the observation-driven corpus."""
    loaded_settings = settings or load_settings()
    return _train_observation_baseline(
        settings=loaded_settings,
        run_config=_observation_model_run_config(
            loaded_settings,
            "pca_mlp_observation_regressor",
        ),
    )


def train_observation_pca_random_forest_ml_baseline(
    settings: BenchmarkSettings | None = None,
) -> ObservationBaselineOutputs:
    """Train the observation-driven PCA plus random-forest full-grid baseline."""
    loaded_settings = settings or load_settings()
    return _train_observation_baseline(
        settings=loaded_settings,
        run_config=_observation_model_run_config(
            loaded_settings,
            "pca_random_forest_observation_regressor",
        ),
    )


def train_observation_ml_comparison(
    settings: BenchmarkSettings | None = None,
) -> ObservationComparisonOutputs:
    """Run the limited observation-driven ML comparison on one shared synthetic split."""
    loaded_settings = settings or load_settings()
    comparison_settings = loaded_settings.machine_learning.observation_comparison
    run_configs = [
        _observation_model_run_config(loaded_settings, model_name)
        for model_name in comparison_settings.selected_models
    ]
    _validate_comparison_model_inputs(run_configs, comparison_settings)
    context = _prepare_observation_training_context(
        settings=loaded_settings,
        pairing_manifest_path=comparison_settings.pairing_manifest_path,
        observation_fields=comparison_settings.observation_fields,
    )
    repo_root = get_repo_root()
    summary_json = _resolve_repo_path(comparison_settings.output_dir, repo_root) / "summary.json"
    summary_note = (
        repo_root
        / loaded_settings.outputs.experiment_notes_dir
        / f"{comparison_settings.experiment_id}.md"
    )
    results = [
        _train_observation_baseline(
            settings=loaded_settings,
            run_config=run_config,
            context=context,
        )
        for run_config in run_configs
    ]
    model_records = []
    for run_config, result in zip(run_configs, results, strict=True):
        metrics_payload = _read_json(result.metrics_json)
        model_records.append(
            {
                "model_name": run_config.model_name,
                "experiment_id": run_config.experiment_id,
                "metrics_json": _relative_to_repo_or_absolute(result.metrics_json, repo_root),
                "model_details_json": _relative_to_repo_or_absolute(
                    result.model_details_json,
                    repo_root,
                ),
                "predictions_dir": _relative_to_repo_or_absolute(result.predictions_dir, repo_root),
                "test_mean_rmse": metrics_payload["split_metrics"]["test"]["mean_rmse"],
                "validation_mean_rmse": metrics_payload["split_metrics"]["validation"]["mean_rmse"],
                "test_mean_mae": metrics_payload["split_metrics"]["test"]["mean_mae"],
                "validation_mean_mae": metrics_payload["split_metrics"]["validation"]["mean_mae"],
            }
        )
    ranked_models = sorted(
        model_records,
        key=lambda item: (
            item["validation_mean_rmse"],
            item["test_mean_rmse"],
            item["validation_mean_mae"],
            item["test_mean_mae"],
            item["model_name"],
        ),
    )
    summary_payload = {
        "artifact_type": "ml_observation_model_comparison_summary",
        "experiment_id": comparison_settings.experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pairing_manifest_path": _relative_to_repo_or_absolute(context.manifest_path, repo_root),
        "split_strategy": comparison_settings.split_strategy,
        "observation_feature_fields": list(comparison_settings.observation_fields),
        "sample_count": len(context.samples),
        "observation_count_per_case": context.feature_matrix.observation_count,
        "feature_count_per_case": len(context.feature_matrix.samples[0].feature_vector),
        "model_count": len(model_records),
        "ranking_metric": "validation_rmse_then_test_rmse",
        "best_model": ranked_models[0],
        "model_ranking": [
            {
                "rank": index + 1,
                **record,
            }
            for index, record in enumerate(ranked_models)
        ],
        "assumptions": [
            "All compared models use the same observation-driven corpus, the same frozen full-grid target contract, and the same family-stratified split assignments.",
            "Linear, ridge, random-forest, and shallow-MLP models remain bounded representatives of broader ML families rather than exhaustive optimized benchmarks.",
            "All four workflows reconstruct the same full-grid target so RMSE and MAE remain directly comparable across the limited suite.",
        ],
        "limitations": [
            "The comparison remains synthetic and bounded to the current five configured geological families plus their controlled parameter and geometry variations.",
            "Three of the four models predict PCA coefficients rather than the full target grid directly, so configured component counts remain part of the modeling contract.",
            "Hyperparameters are intentionally limited and centrally fixed for a publication-friendly representative comparison, not a broad benchmark search.",
        ],
    }
    _save_json(summary_payload, summary_json)
    _write_observation_comparison_note(
        summary_note,
        payload=summary_payload,
        summary_json=summary_json,
        repo_root=repo_root,
    )
    return ObservationComparisonOutputs(
        summary_json=summary_json,
        summary_note=summary_note,
        model_output_dirs=tuple(result.predictions_dir.parent for result in results),
    )


def train_paper_pseudo_bending_ml_suite(
    settings: BenchmarkSettings | None = None,
    *,
    split_strategy: str = "family_stratified",
    output_dir_override: Path | None = None,
    experiment_id_override: str | None = None,
) -> PaperPseudoBendingMLSuiteOutputs:
    """Run paper ML evaluation against the Phase 3 pseudo-bending corpus."""
    loaded_settings = settings or load_settings()
    suite_settings = loaded_settings.machine_learning.paper_pseudo_bending_ml_suite
    repo_root = get_repo_root()
    experiment_id = experiment_id_override or suite_settings.experiment_id
    output_dir = _resolve_repo_path(
        output_dir_override or suite_settings.output_dir,
        repo_root,
    )
    summary_json = output_dir / "suite_summary.json"
    repeated_summary_json = output_dir / "repeated_split_summary.json"
    leave_one_family_out_summary_json = output_dir / "leave_one_family_out_summary.json"
    target_group_split_csv = output_dir / "target_group_split_assignments.csv"
    noise_robustness_summary_json = output_dir / "noise_robustness_summary.json"
    summary_note = (
        repo_root
        / loaded_settings.outputs.experiment_notes_dir
        / f"{experiment_id}.md"
    )
    context = _prepare_paper_pseudo_bending_training_context(
        settings=loaded_settings,
        pairing_manifest_path=suite_settings.input_manifest_path,
        observation_fields=suite_settings.observation_fields,
        random_seed=suite_settings.fixed_split_seed,
        split_strategy=split_strategy,
    )
    _write_target_group_split_assignments(
        context.samples,
        context.split_by_case_id,
        target_group_split_csv,
    )
    fixed_model_records: list[dict[str, Any]] = []
    model_metrics_jsons: list[Path] = []
    repeated_records: list[dict[str, Any]] = []
    loo_records: list[dict[str, Any]] = []
    noise_records: list[dict[str, Any]] = []

    for model_name in suite_settings.selected_models:
        run_config = _paper_model_run_config(
            loaded_settings,
            model_name=model_name,
            experiment_id=f"{experiment_id}_{_model_slug(model_name)}",
            pairing_manifest_path=suite_settings.input_manifest_path,
            output_dir=output_dir / "models" / _model_slug(model_name) / "fixed_split",
            observation_fields=suite_settings.observation_fields,
            split_strategy=split_strategy,
        )
        fixed_output = _train_observation_baseline(
            settings=loaded_settings,
            run_config=run_config,
            context=context,
        )
        model_metrics_jsons.append(fixed_output.metrics_json)
        fixed_metrics = _read_json(fixed_output.metrics_json)
        fixed_model_records.append(
            {
                "model_name": model_name,
                "experiment_id": run_config.experiment_id,
                "metrics_json": _relative_to_repo_or_absolute(fixed_output.metrics_json, repo_root),
                "model_details_json": _relative_to_repo_or_absolute(
                    fixed_output.model_details_json,
                    repo_root,
                ),
                "predictions_dir": _relative_to_repo_or_absolute(
                    fixed_output.predictions_dir,
                    repo_root,
                ),
                "fixed_train_mae": fixed_metrics["split_metrics"]["train"]["mean_mae"],
                "fixed_train_rmse": fixed_metrics["split_metrics"]["train"]["mean_rmse"],
                "fixed_validation_mae": fixed_metrics["split_metrics"]["validation"]["mean_mae"],
                "fixed_validation_rmse": fixed_metrics["split_metrics"]["validation"]["mean_rmse"],
                "fixed_test_mae": fixed_metrics["split_metrics"]["test"]["mean_mae"],
                "fixed_test_rmse": fixed_metrics["split_metrics"]["test"]["mean_rmse"],
            }
        )
        repeated_records.append(
            _evaluate_repeated_family_stratified_splits(
                settings=loaded_settings,
                context=context,
                model_name=model_name,
                split_seeds=suite_settings.repeated_split_seeds,
                split_strategy=split_strategy,
            )
        )
        if suite_settings.leave_one_family_out:
            loo_records.append(
                _evaluate_leave_one_family_out(
                    settings=loaded_settings,
                    context=context,
                    model_name=model_name,
                )
            )
        noise_records.append(
            _evaluate_noise_robustness(
                settings=loaded_settings,
                context=context,
                model_name=model_name,
                noise_levels_s=suite_settings.travel_time_noise_levels_s,
                noise_random_seed=suite_settings.noise_random_seed,
            )
        )

    repeated_payload = {
        "artifact_type": "paper_pseudo_bending_ml_repeated_split_summary",
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_manifest_path": _relative_to_repo_or_absolute(context.manifest_path, repo_root),
        "split_strategy": split_strategy,
        "split_protocol": _repeated_split_protocol_name(split_strategy),
        "split_random_seeds": list(suite_settings.repeated_split_seeds),
        "model_records": repeated_records,
    }
    loo_payload = {
        "artifact_type": "paper_pseudo_bending_ml_leave_one_family_out_summary",
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_manifest_path": _relative_to_repo_or_absolute(context.manifest_path, repo_root),
        "split_protocol": "leave_one_family_out",
        "families": sorted({sample.scenario for sample in context.samples}),
        "model_records": loo_records,
    }
    noise_payload = {
        "artifact_type": "paper_pseudo_bending_ml_noise_robustness_summary",
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_manifest_path": _relative_to_repo_or_absolute(context.manifest_path, repo_root),
        "noise_random_seed": suite_settings.noise_random_seed,
        "travel_time_noise_levels_s": list(suite_settings.travel_time_noise_levels_s),
        "noise_application": "in_memory_gaussian_perturbation_to_travel_time_s_features_only",
        "model_records": noise_records,
    }
    repeated_by_model = {record["model_name"]: record for record in repeated_records}
    ranked_models = sorted(
        fixed_model_records,
        key=lambda item: (
            item["fixed_validation_rmse"],
            repeated_by_model[item["model_name"]]["aggregate_metrics"]["mean_validation_rmse"],
            item["fixed_test_rmse"],
            item["model_name"],
        ),
    )
    summary_payload = {
        "artifact_type": "paper_pseudo_bending_ml_suite_summary",
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_corpus_id": str(context.pairing_manifest.get("corpus_id", "")),
        "input_manifest_path": _relative_to_repo_or_absolute(context.manifest_path, repo_root),
        "simulator": str(context.pairing_manifest.get("simulator_name", "")),
        "sample_count": len(context.samples),
        "families": sorted({sample.scenario for sample in context.samples}),
        "observation_count_per_case": context.feature_matrix.observation_count,
        "observation_feature_fields": list(suite_settings.observation_fields),
        "selected_models": list(suite_settings.selected_models),
        "split_strategy": split_strategy,
        "split_protocols": _effective_split_protocols(suite_settings.split_protocols, split_strategy),
        "fixed_split_seed": suite_settings.fixed_split_seed,
        "repeated_split_seeds": list(suite_settings.repeated_split_seeds),
        "travel_time_noise_levels_s": list(suite_settings.travel_time_noise_levels_s),
        "best_model_selection_metric": suite_settings.best_model_selection_metric,
        "best_model": ranked_models[0],
        "model_ranking": [
            {"rank": index + 1, **record} for index, record in enumerate(ranked_models)
        ],
        "fixed_split_model_records": fixed_model_records,
        "split_case_counts": _split_case_counts(context.split_by_case_id, context.samples),
        "split_unique_target_counts": _split_unique_target_counts(
            context.split_by_case_id,
            context.samples,
        ),
        "target_group_split_assignments_csv": _relative_to_repo_or_absolute(
            target_group_split_csv,
            repo_root,
        ),
        "repeated_split_summary_json": _relative_to_repo_or_absolute(
            repeated_summary_json,
            repo_root,
        ),
        "leave_one_family_out_summary_json": _relative_to_repo_or_absolute(
            leave_one_family_out_summary_json,
            repo_root,
        ),
        "noise_robustness_summary_json": _relative_to_repo_or_absolute(
            noise_robustness_summary_json,
            repo_root,
        ),
        "assumptions": [
            "All metrics are generated from the Phase 3 pseudo-bending observation corpus.",
            "The original corpus is not modified; noise robustness uses derived in-memory feature vectors.",
            "Model selection prioritizes fixed-split validation RMSE, then repeated-split mean validation RMSE.",
        ],
        "limitations": [
            "The suite is synthetic and bounded to the configured five geological families.",
            "The PCA target compression and fixed hyperparameters remain modeling assumptions.",
            "Leave-one-family-out tests extrapolation across configured synthetic families, not real geological deployment.",
        ],
    }
    if split_strategy == "target_grouped_family_stratified":
        summary_payload["assumptions"].extend(
            [
                "All cases with the same canonical SHA-256 target-vector identity are assigned to one split, regardless of acquisition geometry.",
                "The grouped split prioritizes target-group independence and family balance; requested case fractions are secondary because target groups are indivisible.",
            ]
        )
        summary_payload["limitations"].append(
            "The grouped split has only the effective number of unique target models available in the corpus; case count therefore overstates independent experimental units."
        )
    _save_json(repeated_payload, repeated_summary_json)
    _save_json(loo_payload, leave_one_family_out_summary_json)
    _save_json(noise_payload, noise_robustness_summary_json)
    _save_json(summary_payload, summary_json)
    _write_paper_pseudo_bending_ml_suite_note(
        summary_note,
        summary_payload,
        summary_json,
        repeated_summary_json,
        leave_one_family_out_summary_json,
        noise_robustness_summary_json,
        repo_root,
    )
    return PaperPseudoBendingMLSuiteOutputs(
        summary_json=summary_json,
        repeated_summary_json=repeated_summary_json,
        leave_one_family_out_summary_json=leave_one_family_out_summary_json,
        noise_robustness_summary_json=noise_robustness_summary_json,
        summary_note=summary_note,
        model_metrics_jsons=tuple(model_metrics_jsons),
        output_dir=output_dir,
        target_group_split_csv=target_group_split_csv,
    )


def train_paper_pseudo_bending_grouped_ml_suite(
    settings: BenchmarkSettings | None = None,
) -> PaperPseudoBendingMLSuiteOutputs:
    """Run the paper ML suite with target-level leakage prevention enabled."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    grouped_output_dir = (
        _resolve_repo_path(loaded_settings.outputs.generated_base_dir, repo_root)
        / "ml_baselines"
        / "phase9_grouped_split_ml_v1"
    )
    return train_paper_pseudo_bending_ml_suite(
        loaded_settings,
        split_strategy="target_grouped_family_stratified",
        output_dir_override=grouped_output_dir,
        experiment_id_override="phase9_grouped_split_ml_v1",
    )


def train_phase7_observation_signal_ablation_suite(
    settings: BenchmarkSettings | None = None,
    *,
    split_strategy: str = "family_stratified",
    output_dir_override: Path | None = None,
    experiment_id_override: str | None = None,
) -> Phase7ObservationSignalAblationOutputs:
    """Run Phase 7A controls for travel-time signal in the selected observation model."""
    loaded_settings = settings or load_settings()
    suite_settings = loaded_settings.machine_learning.paper_pseudo_bending_ml_suite
    repo_root = get_repo_root()
    experiment_id = experiment_id_override or "phase7_observation_signal_ablation_v1"
    output_dir = _resolve_repo_path(
        output_dir_override
        or Path("outputs/generated/ml_baselines/phase7_observation_signal_ablation_v1"),
        repo_root,
    )
    summary_json = output_dir / "ablation_summary.json"
    summary_csv = output_dir / "ablation_summary.csv"
    cell_centered_metrics_csv = output_dir / "cell_centered_metrics.csv"
    leave_one_family_out_summary_json = output_dir / "leave_one_family_out_summary.json"
    target_group_split_csv = output_dir / "target_group_split_assignments.csv"
    summary_note = repo_root / loaded_settings.outputs.experiment_notes_dir / f"{experiment_id}.md"
    context = _prepare_paper_pseudo_bending_training_context(
        settings=loaded_settings,
        pairing_manifest_path=suite_settings.input_manifest_path,
        observation_fields=suite_settings.observation_fields,
        random_seed=suite_settings.fixed_split_seed,
        split_strategy=split_strategy,
    )
    variants = _phase7_ablation_variants(suite_settings.observation_fields)
    variant_records: list[dict[str, Any]] = []
    repeated_records: list[dict[str, Any]] = []
    loo_records: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    variant_metrics_jsons: list[Path] = []

    for variant in variants:
        variant_context = _phase7_variant_context(
            context,
            variant_id=str(variant["variant_id"]),
            observation_fields=tuple(str(field) for field in variant["observation_fields"]),
            random_seed=707,
        )
        variant_output = _train_phase7_variant(
            settings=loaded_settings,
            context=variant_context,
            variant=variant,
            output_dir=output_dir / "variants" / str(variant["variant_id"]),
            split_strategy=split_strategy,
            experiment_id=experiment_id,
        )
        variant_metrics_jsons.append(variant_output.metrics_json)
        metrics = _read_json(variant_output.metrics_json)
        cell_metric_rows = compute_ml_cell_centered_case_metrics(
            variant_output.metrics_json,
            repo_root,
        )
        for row in cell_metric_rows:
            row["variant_id"] = variant["variant_id"]
            row["model_name"] = variant["model_name"]
        cell_rows.extend(cell_metric_rows)
        variant_records.append(
            _phase7_variant_summary_record(
                variant=variant,
                metrics=metrics,
                cell_metric_rows=cell_metric_rows,
                metrics_json=variant_output.metrics_json,
                feature_matrix_csv=variant_output.feature_matrix_csv,
                repo_root=repo_root,
            )
        )
        repeated_records.append(
            _evaluate_phase7_repeated_splits(
                settings=loaded_settings,
                base_context=context,
                variant=variant,
                split_seeds=suite_settings.repeated_split_seeds,
                random_seed=707,
                split_strategy=split_strategy,
            )
        )
        loo_records.append(
            _evaluate_phase7_leave_one_family_out(
                settings=loaded_settings,
                base_context=context,
                variant=variant,
                random_seed=707,
            )
        )

    summary_rows = _phase7_summary_rows(variant_records, repeated_records, loo_records)
    interpretation = _phase7_interpretation(summary_rows)
    summary_payload = {
        "artifact_type": "phase7_observation_signal_ablation_summary",
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_corpus_id": str(context.pairing_manifest.get("corpus_id", "")),
        "input_manifest_path": _relative_to_repo_or_absolute(context.manifest_path, repo_root),
        "model_family": "pca_linear_observation_regressor_for_model_variants",
        "sample_count": len(context.samples),
        "observation_count_per_case": context.feature_matrix.observation_count,
        "fixed_split_seed": suite_settings.fixed_split_seed,
        "split_strategy": split_strategy,
        "split_case_counts": _split_case_counts(context.split_by_case_id, context.samples),
        "split_unique_target_counts": _split_unique_target_counts(
            context.split_by_case_id,
            context.samples,
        ),
        "repeated_split_seeds": list(suite_settings.repeated_split_seeds),
        "travel_time_shuffle_scope": "within_split_global_travel_time",
        "travel_time_shuffle_seed": 707,
        "variants": variant_records,
        "summary_rows": summary_rows,
        "interpretation": interpretation,
        "cell_centered_metrics_csv": _relative_to_repo_or_absolute(
            cell_centered_metrics_csv,
            repo_root,
        ),
        "leave_one_family_out_summary_json": _relative_to_repo_or_absolute(
            leave_one_family_out_summary_json,
            repo_root,
        ),
        "target_group_split_assignments_csv": _relative_to_repo_or_absolute(
            target_group_split_csv,
            repo_root,
        ),
        "assumptions": [
            "All variants consume the same 100-case paper pseudo-bending observation corpus and frozen node-centered target grids.",
            "Feature ablations preserve the same ordered observation layout and only remove or permute observation fields.",
            "The shuffled travel-time control permutes travel_time_s values within each fixed split across all cases and observation positions, so split-level distributions are preserved without cross-split leakage.",
            "The family-mean target diagnostic uses held-out family labels at prediction time and is therefore not a fair deployable model.",
            "The true-model ray-path length is retained as a privileged synthetic diagnostic; Euclidean-path variants are reported separately as a realistic geometry-only alternative.",
        ],
        "limitations": [
            "The suite remains synthetic-only and does not validate real-data deployment.",
            "The ablation uses the selected PCA-linear family as a focused signal diagnostic rather than a universal ML benchmark.",
            "Leave-one-family-out values measure extrapolation across configured synthetic families only.",
        ],
    }
    if split_strategy == "target_grouped_family_stratified":
        summary_payload["assumptions"].extend(
            [
                "All cases with the same canonical SHA-256 target-vector identity remain in one split, including cases with different acquisition geometries.",
                "The grouped split reports case counts and unique-target counts separately because target groups are the independent experimental units.",
            ]
        )
        summary_payload["limitations"].append(
            "The effective sample size is the number of unique target velocity models, not the number of acquisition cases."
        )
    loo_payload = {
        "artifact_type": "phase7_observation_signal_ablation_leave_one_family_out_summary",
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "split_protocol": "leave_one_family_out",
        "families": sorted({sample.scenario for sample in context.samples}),
        "variant_records": loo_records,
    }
    _save_json(summary_payload, summary_json)
    _save_json(loo_payload, leave_one_family_out_summary_json)
    _write_csv(summary_csv, summary_rows)
    _write_csv(cell_centered_metrics_csv, cell_rows)
    _write_phase7_ablation_note(
        summary_note,
        payload=summary_payload,
        summary_json=summary_json,
        summary_csv=summary_csv,
        cell_centered_metrics_csv=cell_centered_metrics_csv,
        leave_one_family_out_summary_json=leave_one_family_out_summary_json,
        repo_root=repo_root,
    )
    return Phase7ObservationSignalAblationOutputs(
        output_dir=output_dir,
        summary_json=summary_json,
        summary_csv=summary_csv,
        cell_centered_metrics_csv=cell_centered_metrics_csv,
        leave_one_family_out_summary_json=leave_one_family_out_summary_json,
        summary_note=summary_note,
        variant_metrics_jsons=tuple(variant_metrics_jsons),
    )


def train_phase9_grouped_split_ablation_suite(
    settings: BenchmarkSettings | None = None,
) -> Phase7ObservationSignalAblationOutputs:
    """Run the observation-signal controls on the target-grouped split."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    output_dir = (
        _resolve_repo_path(loaded_settings.outputs.generated_base_dir, repo_root)
        / "ml_baselines"
        / "phase9_grouped_split_ablation_v1"
    )
    return train_phase7_observation_signal_ablation_suite(
        loaded_settings,
        split_strategy="target_grouped_family_stratified",
        output_dir_override=output_dir,
        experiment_id_override="phase9_grouped_split_ablation_v1",
    )


def train_phase7_learning_curve(
    settings: BenchmarkSettings | None = None,
    *,
    training_sizes: Sequence[int] = (20, 40, 60, 70),
    repetition_seeds: Sequence[int] = (1701, 1702, 1703),
    output_dir: Path = Path(
        "outputs/generated/ml_baselines/phase7_learning_curve_v1"
    ),
    split_strategy: str = "family_stratified",
    experiment_id: str = "phase7_learning_curve_v1",
    summary_note_path: Path | None = None,
) -> Phase7LearningCurveOutputs:
    """Measure fixed-holdout performance as the PCA-linear training pool grows."""
    loaded_settings = settings or load_settings()
    suite_settings = loaded_settings.machine_learning.paper_pseudo_bending_ml_suite
    repo_root = get_repo_root()
    resolved_output_dir = _resolve_repo_path(output_dir, repo_root)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    context = _prepare_paper_pseudo_bending_training_context(
        settings=loaded_settings,
        pairing_manifest_path=suite_settings.input_manifest_path,
        observation_fields=suite_settings.observation_fields,
        random_seed=suite_settings.fixed_split_seed,
        split_strategy=split_strategy,
    )
    normalized_sizes = tuple(int(size) for size in training_sizes)
    normalized_seeds = tuple(int(seed) for seed in repetition_seeds)
    subset_builder = (
        build_target_grouped_learning_curve_training_subsets
        if split_strategy == "target_grouped_family_stratified"
        else build_learning_curve_training_subsets
    )
    subsets = subset_builder(
        context.samples,
        context.split_by_case_id,
        normalized_sizes,
        normalized_seeds,
    )
    fixed_validation_samples = _samples_for_split(
        context.samples,
        context.split_by_case_id,
        "validation",
    )
    fixed_test_samples = _samples_for_split(context.samples, context.split_by_case_id, "test")
    target_cell_cache: dict[str, tuple[float, ...]] = {}
    repetition_records: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    subset_case_ids: dict[str, list[str]] = {}
    for (training_size, repetition_seed), train_samples in subsets.items():
        model = _fit_suite_model(
            loaded_settings,
            "pca_linear_observation_regressor",
            list(train_samples),
        )
        split_payloads: dict[str, dict[str, Any]] = {}
        for split_name, split_samples in (
            ("train", list(train_samples)),
            ("validation", fixed_validation_samples),
            ("test", fixed_test_samples),
        ):
            split_payload, case_metrics = _evaluate_learning_curve_split(
                split_samples,
                model,
                target_cell_cache,
            )
            split_payloads[split_name] = split_payload
            if split_name == "test":
                family_rows.extend(
                    {
                        "training_size": training_size,
                        "repetition_seed": repetition_seed,
                        "split": split_name,
                        "family": family,
                        **metrics,
                    }
                    for family, metrics in _learning_curve_family_metrics(case_metrics).items()
                )
        repetition_records.append(
            {
                "training_size": training_size,
                "repetition_seed": repetition_seed,
                "train": split_payloads["train"],
                "validation": split_payloads["validation"],
                "test": split_payloads["test"],
            }
        )
        subset_case_ids[f"{training_size}:{repetition_seed}"] = [
            sample.case_id for sample in train_samples
        ]

    summary_rows = aggregate_learning_curve_metrics(repetition_records)
    interpretation = _learning_curve_interpretation(
        summary_rows,
        eligible_training_case_count=sum(
            1
            for sample in context.samples
            if context.split_by_case_id[sample.case_id] == "train"
        ),
        eligible_training_target_count=len(
            {
                target_vector_sha256(sample.target_vector)
                for sample in context.samples
                if context.split_by_case_id[sample.case_id] == "train"
            }
        ),
    )
    summary_json = resolved_output_dir / "learning_curve_summary.json"
    summary_csv = resolved_output_dir / "learning_curve_summary.csv"
    repetition_metrics_csv = resolved_output_dir / "learning_curve_repetition_metrics.csv"
    family_wise_csv = resolved_output_dir / "family_wise_learning_curve.csv"
    split_case_ids = {
        split_name: sorted(
            sample.case_id
            for sample in context.samples
            if context.split_by_case_id[sample.case_id] == split_name
        )
        for split_name in ("train", "validation", "test")
    }
    summary_payload = {
        "artifact_type": "phase7_learning_curve_summary",
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_corpus_id": str(context.pairing_manifest.get("corpus_id", "")),
        "input_manifest_path": _relative_to_repo_or_absolute(
            context.manifest_path,
            repo_root,
        ),
        "model_name": "pca_linear_observation_regressor",
        "observation_feature_fields": list(suite_settings.observation_fields),
        "observation_count_per_case": context.feature_matrix.observation_count,
        "sample_count": len(context.samples),
        "fixed_split_seed": suite_settings.fixed_split_seed,
        "fixed_split_case_counts": {
            split_name: len(case_ids) for split_name, case_ids in split_case_ids.items()
        },
        "fixed_split_unique_target_counts": _split_unique_target_counts(
            context.split_by_case_id,
            context.samples,
        ),
        "fixed_split_case_ids": split_case_ids,
        "split_strategy": split_strategy,
        "training_sizes_requested": list(normalized_sizes),
        "training_sizes_evaluated": sorted({int(row["training_size"]) for row in summary_rows}),
        "repetition_seeds": list(normalized_seeds),
        "training_subset_protocol": (
            "target-group-atomic, family-balanced subsets drawn only from the fixed "
            "target-grouped training pool; fixed validation and test cases are never "
            "eligible for training"
            if split_strategy == "target_grouped_family_stratified"
            else "family-balanced subsets drawn only from the fixed family-stratified "
            "training pool; fixed validation and test cases are never eligible for training"
        ),
        "training_subset_case_ids": subset_case_ids,
        "training_subset_unique_target_counts": {
            f"{training_size}:{repetition_seed}": len(
                {
                    target_vector_sha256(sample.target_vector)
                    for sample in train_samples
                }
            )
            for (training_size, repetition_seed), train_samples in subsets.items()
        },
        "summary_rows": summary_rows,
        "interpretation": interpretation,
        "outputs": {
            "summary_csv": _relative_to_repo_or_absolute(summary_csv, repo_root),
            "repetition_metrics_csv": _relative_to_repo_or_absolute(
                repetition_metrics_csv,
                repo_root,
            ),
            "family_wise_csv": _relative_to_repo_or_absolute(family_wise_csv, repo_root),
        },
        "assumptions": [
            "All repetitions use the same paper pseudo-bending corpus, target grids, ordered observation layout, and PCA-linear model family as the selected grouped-split model.",
            f"The fixed validation and test partitions use the {split_strategy} protocol with seed {suite_settings.fixed_split_seed}.",
            "Training subsets preserve family representation and never include validation or test cases.",
            "Cell-centered metrics convert node-centered predictions and targets by averaging each cell's eight surrounding nodes, matching the primary evaluation contract.",
        ],
        "limitations": [
            (
                "Target-grouped subsets are restricted to case sizes attainable by adding whole "
                "target groups while retaining equal target-group counts per family; this can "
                "produce fewer evaluated sizes than a case-level curve."
                if split_strategy == "target_grouped_family_stratified"
                else "The historical fixed holdout leaves 70 eligible training cases, so 80- and 100-case training sizes cannot be evaluated without changing the holdout protocol."
            ),
            "The experiment is synthetic-only and does not establish real-data validation or universal learning-curve behavior.",
            "The curve tests one selected PCA-linear model and fixed PCA dimensionality; it is not a model or hyperparameter search.",
        ],
    }
    _save_json(summary_payload, summary_json)
    _write_csv(summary_csv, summary_rows)
    _write_csv(repetition_metrics_csv, _learning_curve_repetition_rows(repetition_records))
    _write_csv(family_wise_csv, family_rows)
    summary_note = _resolve_repo_path(
        summary_note_path
        if summary_note_path is not None
        else repo_root / loaded_settings.outputs.experiment_notes_dir / "phase7_learning_curve_v1.md",
        repo_root,
    )
    _write_phase7_learning_curve_note(
        summary_note,
        payload=summary_payload,
        summary_json=summary_json,
        summary_csv=summary_csv,
        repetition_metrics_csv=repetition_metrics_csv,
        family_wise_csv=family_wise_csv,
        repo_root=repo_root,
    )
    return Phase7LearningCurveOutputs(
        output_dir=resolved_output_dir,
        summary_json=summary_json,
        summary_csv=summary_csv,
        repetition_metrics_csv=repetition_metrics_csv,
        family_wise_csv=family_wise_csv,
    )


def train_phase9_grouped_split_learning_curve(
    settings: BenchmarkSettings | None = None,
    *,
    training_sizes: Sequence[int] = (30, 50),
    repetition_seeds: Sequence[int] = (1701, 1702, 1703),
    output_dir: Path = Path(
        "outputs/generated/ml_baselines/phase9_grouped_split_learning_curve_v1"
    ),
) -> Phase7LearningCurveOutputs:
    """Run a target-group-atomic learning curve on the corrected grouped split."""
    return train_phase7_learning_curve(
        settings=settings,
        training_sizes=training_sizes,
        repetition_seeds=repetition_seeds,
        output_dir=output_dir,
        split_strategy="target_grouped_family_stratified",
        experiment_id="phase9_grouped_split_learning_curve_v1",
        summary_note_path=Path(
            "wiki/experiments/phase9_grouped_split_learning_curve_v1.md"
        ),
    )


def build_learning_curve_training_subsets(
    samples: Sequence[ObservationTrainingSample],
    split_by_case_id: Mapping[str, str],
    training_sizes: Sequence[int],
    repetition_seeds: Sequence[int],
) -> dict[tuple[int, int], tuple[ObservationTrainingSample, ...]]:
    """Build deterministic family-balanced subsets from the fixed training split only."""
    if not training_sizes:
        raise ValueError("training_sizes must contain at least one size.")
    if not repetition_seeds:
        raise ValueError("repetition_seeds must contain at least one seed.")
    case_ids = [sample.case_id for sample in samples]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Learning-curve samples must have unique case IDs.")
    training_samples = [
        sample for sample in samples if split_by_case_id.get(sample.case_id) == "train"
    ]
    if not training_samples:
        raise ValueError("The fixed split contains no eligible training cases.")
    grouped: dict[str, list[ObservationTrainingSample]] = {}
    for sample in training_samples:
        grouped.setdefault(sample.scenario, []).append(sample)
    families = sorted(grouped)
    if not families:
        raise ValueError("Learning-curve training samples must contain at least one family.")
    normalized_sizes = tuple(int(size) for size in training_sizes)
    if any(size < len(families) for size in normalized_sizes):
        raise ValueError("Each training size must provide at least one case per family.")
    if any(size % len(families) != 0 for size in normalized_sizes):
        raise ValueError(
            "Family-balanced learning-curve sizes must be divisible by the family count."
        )
    max_size = len(training_samples)
    if any(size > max_size for size in normalized_sizes):
        raise ValueError(
            f"Learning-curve training size cannot exceed the eligible training pool ({max_size})."
        )
    per_family_cap = {family: len(items) for family, items in grouped.items()}
    result: dict[tuple[int, int], tuple[ObservationTrainingSample, ...]] = {}
    for seed in (int(value) for value in repetition_seeds):
        shuffled_by_family: dict[str, list[ObservationTrainingSample]] = {}
        for family in families:
            family_samples = sorted(grouped[family], key=lambda item: item.case_id)
            shuffled = list(family_samples)
            random.Random(seed + sum(ord(char) for char in family)).shuffle(shuffled)
            shuffled_by_family[family] = shuffled
        for size in normalized_sizes:
            per_family = size // len(families)
            if any(per_family > per_family_cap[family] for family in families):
                raise ValueError(
                    f"Training size {size} exceeds at least one family's eligible cases."
                )
            selected = [
                sample
                for family in families
                for sample in shuffled_by_family[family][:per_family]
            ]
            result[(size, seed)] = tuple(sorted(selected, key=lambda item: item.case_id))
    return {
        key: result[key]
        for key in ((int(size), int(seed)) for size in normalized_sizes for seed in repetition_seeds)
    }


def build_target_grouped_learning_curve_training_subsets(
    samples: Sequence[ObservationTrainingSample],
    split_by_case_id: Mapping[str, str],
    training_sizes: Sequence[int],
    repetition_seeds: Sequence[int],
) -> dict[tuple[int, int], tuple[ObservationTrainingSample, ...]]:
    """Build family-balanced learning-curve subsets without splitting target groups."""
    if not training_sizes:
        raise ValueError("training_sizes must contain at least one size.")
    if not repetition_seeds:
        raise ValueError("repetition_seeds must contain at least one seed.")
    training_samples = [
        sample for sample in samples if split_by_case_id.get(sample.case_id) == "train"
    ]
    if not training_samples:
        raise ValueError("The fixed split contains no eligible training cases.")
    groups_by_family: dict[str, dict[str, list[ObservationTrainingSample]]] = {}
    for sample in training_samples:
        family = sample.scenario
        target_hash = target_vector_sha256(sample.target_vector)
        groups_by_family.setdefault(family, {}).setdefault(target_hash, []).append(sample)
    families = sorted(groups_by_family)
    if not families:
        raise ValueError("Learning-curve training samples must contain at least one family.")
    normalized_sizes = tuple(int(size) for size in training_sizes)
    if any(size <= 0 for size in normalized_sizes):
        raise ValueError("training_sizes must contain positive values.")
    result: dict[tuple[int, int], tuple[ObservationTrainingSample, ...]] = {}
    for seed in (int(value) for value in repetition_seeds):
        shuffled_groups: dict[str, list[tuple[str, list[ObservationTrainingSample]]]] = {}
        for family in families:
            family_groups = [
                (target_hash, sorted(group, key=lambda item: item.case_id))
                for target_hash, group in groups_by_family[family].items()
            ]
            random.Random(seed + sum(ord(char) for char in family)).shuffle(family_groups)
            shuffled_groups[family] = family_groups
        max_groups_per_family = max(len(shuffled_groups[family]) for family in families)
        selected_for_size: dict[int, tuple[ObservationTrainingSample, ...]] = {}
        for size in normalized_sizes:
            matches: list[tuple[ObservationTrainingSample, ...]] = []
            for groups_per_family in range(1, max_groups_per_family + 1):
                selected = tuple(
                    sorted(
                        (
                            sample
                            for family in families
                            for _, group in shuffled_groups[family][
                                : min(groups_per_family, len(shuffled_groups[family]))
                            ]
                            for sample in group
                        ),
                        key=lambda item: item.case_id,
                    )
                )
                if len(selected) == size:
                    matches.append(selected)
            if not matches:
                achievable_sizes = sorted(
                    {
                        sum(
                            len(group)
                            for family in families
                            for _, group in shuffled_groups[family][
                                : min(groups_per_family, len(shuffled_groups[family]))
                            ]
                        )
                        for groups_per_family in range(1, max_groups_per_family + 1)
                    }
                )
                raise ValueError(
                    f"Target-grouped learning-curve size {size} is not achievable while "
                    f"adding whole groups equally by family; achievable sizes are "
                    f"{achievable_sizes}."
                )
            selected_for_size[size] = matches[0]
        result.update({(size, seed): selected_for_size[size] for size in normalized_sizes})
    return result


def aggregate_learning_curve_metrics(
    repetition_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int]]:
    """Aggregate mean and population standard deviation by training size."""
    if not repetition_records:
        raise ValueError("repetition_records must contain at least one record.")
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for record in repetition_records:
        training_size = int(record["training_size"])
        grouped.setdefault(training_size, []).append(record)
    metric_names = ("node_rmse", "node_mae", "cell_rmse", "cell_mae")
    rows: list[dict[str, float | int]] = []
    for training_size in sorted(grouped):
        records = grouped[training_size]
        row: dict[str, float | int] = {
            "training_size": training_size,
            "repetition_count": len(records),
        }
        for split_name in ("train", "validation", "test"):
            for metric_name in metric_names:
                values = [
                    float(record[split_name][f"mean_{metric_name}"])
                    for record in records
                    if record[split_name].get(f"mean_{metric_name}") is not None
                ]
                row[f"mean_{split_name}_{metric_name}"] = (
                    sum(values) / len(values) if values else 0.0
                )
                row[f"std_{split_name}_{metric_name}"] = _population_std_for_values(values)
        rows.append(row)
    return rows


def train_paper_pseudo_bending_ml_strengthened(
    settings: BenchmarkSettings | None = None,
) -> PaperPseudoBendingMLStrengthenedOutputs:
    """Run the audit-driven Phase 4.5B strengthened ML experiment."""
    loaded_settings = settings or load_settings()
    strengthened_settings = loaded_settings.machine_learning.paper_pseudo_bending_ml_strengthened
    repo_root = get_repo_root()
    output_dir = _resolve_repo_path(strengthened_settings.output_dir, repo_root)
    summary_json = output_dir / "strengthened_summary.json"
    summary_note = (
        repo_root
        / loaded_settings.outputs.experiment_notes_dir
        / f"{strengthened_settings.experiment_id}.md"
    )
    baseline_suite = _read_json(
        _resolve_repo_path(strengthened_settings.baseline_suite_summary_path, repo_root)
    )
    baseline_audit = _read_json(
        _resolve_repo_path(strengthened_settings.baseline_audit_summary_path, repo_root)
    )
    context = _prepare_paper_pseudo_bending_training_context(
        settings=loaded_settings,
        pairing_manifest_path=strengthened_settings.input_manifest_path,
        observation_fields=strengthened_settings.observation_fields,
        random_seed=strengthened_settings.fixed_split_seed,
    )

    candidate_records = [
        _evaluate_strengthened_candidate(
            settings=loaded_settings,
            context=context,
            model_name=model_name,
            pca_component_count=pca_component_count,
            ridge_alpha=ridge_alpha,
            repeated_split_seeds=strengthened_settings.repeated_split_seeds,
        )
        for model_name, pca_component_count, ridge_alpha in _strengthened_candidates(
            strengthened_settings.candidate_models,
            strengthened_settings.pca_component_counts,
            strengthened_settings.ridge_alphas,
        )
    ]
    ranked_candidates = sorted(
        candidate_records,
        key=lambda record: (
            record["leave_one_family_out"]["worst_held_out_rmse"],
            record["repeated_split"]["aggregate_metrics"]["mean_validation_rmse"],
            record["fixed_split"]["validation"]["mean_rmse"],
            record["model_name"],
            record["pca_component_count"],
            record.get("ridge_alpha") or 0.0,
        ),
    )
    best_candidate = ranked_candidates[0]
    best_model = _fit_strengthened_model(
        settings=loaded_settings,
        model_name=str(best_candidate["model_name"]),
        train_samples=_samples_for_split(context.samples, context.split_by_case_id, "train"),
        pca_component_count=int(best_candidate["pca_component_count"]),
        ridge_alpha=(
            float(best_candidate["ridge_alpha"])
            if best_candidate["ridge_alpha"] is not None
            else None
        ),
    )
    noise_record = _evaluate_strengthened_noise_robustness(
        context=context,
        model=best_model,
        noise_levels_s=strengthened_settings.travel_time_noise_levels_s,
        noise_random_seed=strengthened_settings.noise_random_seed,
    )
    baseline_worst_loo = baseline_audit["leave_one_family_out"]["worst_family"]
    baseline_best = baseline_suite["best_model"]
    metric_deltas = {
        "fixed_validation_rmse_delta": (
            best_candidate["fixed_split"]["validation"]["mean_rmse"]
            - baseline_best["fixed_validation_rmse"]
        ),
        "fixed_test_rmse_delta": (
            best_candidate["fixed_split"]["test"]["mean_rmse"] - baseline_best["fixed_test_rmse"]
        ),
        "repeated_mean_validation_rmse_delta": (
            best_candidate["repeated_split"]["aggregate_metrics"]["mean_validation_rmse"]
            - baseline_audit["stability_indicators"]["repeated_mean_validation_rmse"]
        ),
        "worst_leave_one_family_out_rmse_delta": (
            best_candidate["leave_one_family_out"]["worst_held_out_rmse"]
            - baseline_worst_loo["mean_rmse"]
        ),
    }
    remaining_risks = _strengthened_remaining_risks(best_candidate, baseline_audit, metric_deltas)
    proceed_to_phase_5 = not remaining_risks
    summary_payload = {
        "artifact_type": "paper_pseudo_bending_ml_strengthened_summary",
        "experiment_id": strengthened_settings.experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_manifest_path": _relative_to_repo_or_absolute(context.manifest_path, repo_root),
        "input_corpus_id": str(context.pairing_manifest.get("corpus_id", "")),
        "simulator": str(context.pairing_manifest.get("simulator_name", "")),
        "sample_count": len(context.samples),
        "families": sorted({sample.scenario for sample in context.samples}),
        "audit_driven_rationale": [
            "Phase 4.5 audit flagged weak faulted leave-one-family-out behavior.",
            "Phase 4.5 audit flagged repeated-split validation RMSE higher than fixed validation RMSE.",
            "Linear and ridge models were the only close and stable model families, so strengthening is limited to PCA component and ridge-alpha tuning.",
        ],
        "selected_changes": [
            "Evaluate pca_linear_observation_regressor across configured PCA component counts.",
            "Evaluate pca_ridge_observation_regressor across configured PCA component counts and ridge alphas.",
            "Select by worst leave-one-family-out RMSE, then repeated validation RMSE, then fixed validation RMSE.",
        ],
        "intentionally_not_changed": [
            "No corpus expansion; the audit did not isolate sample count as the sole bottleneck.",
            "No MLP tuning; Phase 4 MLP was substantially weaker than linear/ridge.",
            "No random-forest tuning; Phase 4 random forest was weaker than linear/ridge and did not address the main audit risk.",
            "No feature schema change; the audit did not identify feature scaling as a concrete defect.",
        ],
        "candidate_count": len(candidate_records),
        "candidate_records": candidate_records,
        "best_model": best_candidate,
        "noise_robustness_best_model": noise_record,
        "baseline_reference": {
            "experiment_id": baseline_suite["experiment_id"],
            "best_model": baseline_best,
            "worst_leave_one_family_out": baseline_worst_loo,
            "audit_recommendation": baseline_audit["recommendation"],
        },
        "metric_deltas_vs_phase4": metric_deltas,
        "proceed_to_phase_5": proceed_to_phase_5,
        "recommended_next_phase": "Phase 5"
        if proceed_to_phase_5
        else "another bounded Phase 4B step",
        "remaining_risks": remaining_risks,
    }
    _save_json(summary_payload, summary_json)
    _write_strengthened_note(summary_note, summary_payload, summary_json, repo_root)
    return PaperPseudoBendingMLStrengthenedOutputs(
        summary_json=summary_json,
        summary_note=summary_note,
        output_dir=output_dir,
    )


def train_paper_pseudo_bending_faulted_generalization(
    settings: BenchmarkSettings | None = None,
) -> PaperPseudoBendingFaultedGeneralizationOutputs:
    """Run one bounded faulted-family generalization diagnostic and feature intervention."""
    loaded_settings = settings or load_settings()
    faulted_settings = loaded_settings.machine_learning.paper_pseudo_bending_faulted_generalization
    repo_root = get_repo_root()
    output_dir = _resolve_repo_path(faulted_settings.output_dir, repo_root)
    diagnosis_json = output_dir / "faulted_diagnosis.json"
    summary_json = output_dir / "faulted_generalization_summary.json"
    summary_note = (
        repo_root
        / loaded_settings.outputs.experiment_notes_dir
        / f"{faulted_settings.experiment_id}.md"
    )
    baseline_suite = _read_json(
        _resolve_repo_path(faulted_settings.baseline_suite_summary_path, repo_root)
    )
    baseline_audit = _read_json(
        _resolve_repo_path(faulted_settings.baseline_audit_summary_path, repo_root)
    )
    strengthened_summary = _read_json(
        _resolve_repo_path(faulted_settings.strengthened_summary_path, repo_root)
    )
    base_context = _prepare_paper_pseudo_bending_training_context(
        settings=loaded_settings,
        pairing_manifest_path=faulted_settings.input_manifest_path,
        observation_fields=faulted_settings.observation_fields,
        random_seed=faulted_settings.fixed_split_seed,
    )
    diagnosis_payload = _diagnose_faulted_generalization(
        settings=loaded_settings,
        context=base_context,
        baseline_audit=baseline_audit,
        target_family=faulted_settings.target_family,
        model_name=faulted_settings.model_name,
        pca_component_count=faulted_settings.pca_component_count,
        ridge_alpha=faulted_settings.ridge_alpha,
    )
    _save_json(diagnosis_payload, diagnosis_json)

    enhanced_context = _with_directional_slowness_features(
        base_context,
        observation_fields=faulted_settings.observation_fields,
    )
    base_record = _evaluate_faulted_generalization_candidate(
        settings=loaded_settings,
        context=base_context,
        model_name=faulted_settings.model_name,
        pca_component_count=faulted_settings.pca_component_count,
        ridge_alpha=faulted_settings.ridge_alpha,
        repeated_split_seeds=faulted_settings.repeated_split_seeds,
    )
    enhanced_record = _evaluate_faulted_generalization_candidate(
        settings=loaded_settings,
        context=enhanced_context,
        model_name=faulted_settings.model_name,
        pca_component_count=faulted_settings.pca_component_count,
        ridge_alpha=faulted_settings.ridge_alpha,
        repeated_split_seeds=faulted_settings.repeated_split_seeds,
    )
    comparison = _faulted_generalization_comparison(
        baseline_suite=baseline_suite,
        strengthened_summary=strengthened_summary,
        base_record=base_record,
        enhanced_record=enhanced_record,
        target_family=faulted_settings.target_family,
    )
    recommendation = _faulted_generalization_recommendation(comparison)
    summary_payload = {
        "artifact_type": "paper_pseudo_bending_faulted_generalization_summary",
        "experiment_id": faulted_settings.experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_manifest_path": _relative_to_repo_or_absolute(base_context.manifest_path, repo_root),
        "input_corpus_id": str(base_context.pairing_manifest.get("corpus_id", "")),
        "simulator": str(base_context.pairing_manifest.get("simulator_name", "")),
        "target_family": faulted_settings.target_family,
        "diagnosis_json": _relative_to_repo_or_absolute(diagnosis_json, repo_root),
        "diagnosis_summary": diagnosis_payload["diagnosis_summary"],
        "intervention": {
            "intervention_id": faulted_settings.derived_feature_set,
            "model_name": faulted_settings.model_name,
            "pca_component_count": faulted_settings.pca_component_count,
            "ridge_alpha": faulted_settings.ridge_alpha,
            "rationale": (
                "Faulted held-out behavior is the dominant audit weakness, while the corpus already "
                "contains balanced faulted acquisition profiles. The bounded intervention adds "
                "directional apparent-slowness features that can expose travel-time asymmetry without "
                "altering the original corpus or changing model family."
            ),
            "intentionally_not_changed": [
                "No Phase 3 corpus files were overwritten or expanded.",
                "No MLP, random-forest, or broad hyperparameter search was added.",
                "No family labels or fault parameters were exposed as supervised features.",
            ],
        },
        "base_same_model_no_derived_features": base_record,
        "directional_slowness_feature_model": enhanced_record,
        "before_after_comparison": comparison,
        "recommendation": recommendation,
        "recommendation_policy": faulted_settings.recommendation_policy,
        "limitations": [
            "This is one bounded feature-side intervention, not an exhaustive feature-engineering search.",
            "Leave-one-family-out remains synthetic family extrapolation and is not real-data validation.",
            "If the feature intervention fails, the result should be reported as a limitation rather than tuned indefinitely.",
        ],
    }
    _save_json(summary_payload, summary_json)
    _write_faulted_generalization_note(
        summary_note,
        payload=summary_payload,
        summary_json=summary_json,
        diagnosis_json=diagnosis_json,
        repo_root=repo_root,
    )
    return PaperPseudoBendingFaultedGeneralizationOutputs(
        diagnosis_json=diagnosis_json,
        summary_json=summary_json,
        summary_note=summary_note,
        output_dir=output_dir,
    )


def tune_observation_pca_ridge_ml_baseline(
    settings: BenchmarkSettings | None = None,
) -> ObservationTuningOutputs:
    """Tune the observation-driven PCA-plus-ridge baseline across repeated seeded splits."""
    loaded_settings = settings or load_settings()
    tuning_settings = loaded_settings.machine_learning.observation_tuning
    repo_root = get_repo_root()
    resolved_manifest_path = _resolve_repo_path(tuning_settings.pairing_manifest_path, repo_root)
    resolved_manifest_path = _ensure_observation_pairing_manifest(
        loaded_settings, resolved_manifest_path
    )
    pairing_manifest = _read_json(resolved_manifest_path)
    feature_matrix = assemble_observation_case_feature_matrix(
        resolved_manifest_path,
        repo_root,
        observation_fields=tuning_settings.observation_fields,
    )
    samples = _attach_targets(feature_matrix, pairing_manifest, repo_root)
    output_dir = _resolve_repo_path(tuning_settings.output_dir, repo_root)
    summary_json = output_dir / "summary.json"
    candidate_metrics_json = output_dir / "candidate_metrics.json"
    summary_note = (
        repo_root
        / loaded_settings.outputs.experiment_notes_dir
        / f"{tuning_settings.experiment_id}.md"
    )

    candidate_records: list[dict[str, Any]] = []
    for pca_component_count in tuning_settings.pca_component_counts:
        for ridge_alpha in tuning_settings.ridge_alphas:
            split_records: list[dict[str, Any]] = []
            for split_seed in tuning_settings.split_random_seeds:
                split_by_case_id = _family_stratified_split(
                    samples,
                    train_fraction=loaded_settings.machine_learning.train_fraction,
                    validation_fraction=loaded_settings.machine_learning.validation_fraction,
                    random_seed=split_seed,
                )
                train_samples = [
                    sample for sample in samples if split_by_case_id[sample.case_id] == "train"
                ]
                validation_samples = [
                    sample for sample in samples if split_by_case_id[sample.case_id] == "validation"
                ]
                test_samples = [
                    sample for sample in samples if split_by_case_id[sample.case_id] == "test"
                ]
                model = PCARidgeObservationRegressor.fit(
                    feature_vectors=[sample.feature_vector for sample in train_samples],
                    target_vectors=[sample.target_vector for sample in train_samples],
                    pca_component_count=pca_component_count,
                    ridge_alpha=ridge_alpha,
                    velocity_bounds=loaded_settings.velocity_model_generation.velocity_bounds_km_per_s,
                )
                split_records.append(
                    {
                        "split_seed": split_seed,
                        "train": _evaluate_split_metrics_only(train_samples, model),
                        "validation": _evaluate_split_metrics_only(validation_samples, model),
                        "test": _evaluate_split_metrics_only(test_samples, model),
                    }
                )
            candidate_records.append(
                _summarize_tuning_candidate(
                    pca_component_count=pca_component_count,
                    ridge_alpha=ridge_alpha,
                    split_records=split_records,
                )
            )

    ranked_candidates = sorted(
        candidate_records,
        key=lambda item: (
            item["aggregate_metrics"]["mean_validation_rmse"],
            item["aggregate_metrics"]["mean_test_rmse"],
            item["aggregate_metrics"]["mean_train_rmse"],
            item["pca_component_count"],
            item["ridge_alpha"],
        ),
    )
    best_candidate = ranked_candidates[0]
    summary_payload = {
        "artifact_type": "ml_observation_pca_ridge_tuning_summary",
        "experiment_id": tuning_settings.experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pairing_manifest_path": _relative_to_repo_or_absolute(resolved_manifest_path, repo_root),
        "sample_count": len(samples),
        "observation_count_per_case": feature_matrix.observation_count,
        "feature_count_per_case": len(feature_matrix.samples[0].feature_vector),
        "candidate_count": len(candidate_records),
        "split_strategy": tuning_settings.split_strategy,
        "split_random_seeds": list(tuning_settings.split_random_seeds),
        "best_candidate": best_candidate,
        "candidate_ranking": [
            {
                "rank": index + 1,
                "pca_component_count": candidate["pca_component_count"],
                "ridge_alpha": candidate["ridge_alpha"],
                "mean_validation_rmse": candidate["aggregate_metrics"]["mean_validation_rmse"],
                "mean_test_rmse": candidate["aggregate_metrics"]["mean_test_rmse"],
            }
            for index, candidate in enumerate(ranked_candidates)
        ],
        "assumptions": [
            "The five-family observation corpus and frozen full-grid target contract remain unchanged during tuning.",
            "Each candidate is evaluated across repeated family-stratified splits rather than a single seeded split.",
            "Model selection prioritizes mean validation RMSE and then mean test RMSE across repeated splits.",
        ],
    }
    _save_json(
        {
            "artifact_type": "ml_observation_pca_ridge_tuning_candidates",
            "experiment_id": tuning_settings.experiment_id,
            "candidates": candidate_records,
        },
        candidate_metrics_json,
    )
    _save_json(summary_payload, summary_json)
    _write_observation_tuning_note(
        summary_note,
        summary_payload,
        summary_json,
        candidate_metrics_json,
        repo_root,
    )
    return ObservationTuningOutputs(
        summary_json=summary_json,
        summary_note=summary_note,
        candidate_metrics_json=candidate_metrics_json,
    )


def _observation_model_run_config(
    settings: BenchmarkSettings,
    model_name: str,
) -> ObservationModelRunConfig:
    velocity_bounds = settings.velocity_model_generation.velocity_bounds_km_per_s
    if model_name == "pca_linear_observation_regressor":
        baseline_settings = settings.machine_learning.observation_linear_baseline
        return ObservationModelRunConfig(
            experiment_id=baseline_settings.experiment_id,
            model_name=baseline_settings.model_name,
            pairing_manifest_path=baseline_settings.pairing_manifest_path,
            output_dir=baseline_settings.output_dir,
            observation_fields=baseline_settings.observation_fields,
            split_strategy=baseline_settings.split_strategy,
            model_builder=lambda samples: PCALinearObservationRegressor.fit(
                feature_vectors=[sample.feature_vector for sample in samples],
                target_vectors=[sample.target_vector for sample in samples],
                pca_component_count=baseline_settings.pca_component_count,
                velocity_bounds=velocity_bounds,
            ),
            model_details_builder=lambda model, split_by_case_id, samples: {
                "artifact_type": "ml_observation_baseline_model_details",
                "experiment_id": baseline_settings.experiment_id,
                "trained_component_count": len(model.principal_components),
                "configured_component_count": baseline_settings.pca_component_count,
                "velocity_bounds_km_per_s": list(velocity_bounds),
                "split_case_counts": _split_case_counts(split_by_case_id, samples),
            },
            note_title="Observation-Driven PCA Linear Baseline",
            objective=(
                "Train a plain linear reference model from ordered source-receiver observations "
                "while preserving the same frozen full 3D velocity-grid target."
            ),
            notes=[
                "This baseline keeps the same ordered source-receiver observation inputs as the ridge, random-forest, and shallow-MLP comparison models.",
                "Targets remain the frozen full 3D absolute-velocity grid and are reconstructed from PCA coefficients for direct comparability.",
                "The linear fit is solved as a minimum-norm ordinary least-squares mapping in the induced sample-space feature kernel.",
            ],
            assumptions=[
                "All cases in the observation corpus share the same source-receiver observation count so one fixed-length feature space is valid.",
                "The current observation corpus varies both velocity-model parameters and bounded acquisition geometry while preserving a fixed observation count per case.",
                "This linear model omits ridge regularization so it acts as the plain linear comparison point rather than the preferred stabilized linear baseline.",
            ],
            limitations=[
                "The current corpus is synthetic and still bounded to the five configured geological families.",
                "PCA compression remains part of the target contract, so configured component count still influences reconstruction quality.",
                "Because the model is unregularized, it should be interpreted as a plain comparison reference rather than the most stable linear estimator in this workflow.",
            ],
            next_step=(
                "Compare this plain linear reference against the ridge, random-forest, and shallow-MLP variants on the same fixed synthetic split."
            ),
        )
    if model_name == "pca_ridge_observation_regressor":
        baseline_settings = settings.machine_learning.observation_baseline
        return ObservationModelRunConfig(
            experiment_id=baseline_settings.experiment_id,
            model_name=baseline_settings.model_name,
            pairing_manifest_path=baseline_settings.pairing_manifest_path,
            output_dir=baseline_settings.output_dir,
            observation_fields=baseline_settings.observation_fields,
            split_strategy=baseline_settings.split_strategy,
            model_builder=lambda samples: PCARidgeObservationRegressor.fit(
                feature_vectors=[sample.feature_vector for sample in samples],
                target_vectors=[sample.target_vector for sample in samples],
                pca_component_count=baseline_settings.pca_component_count,
                ridge_alpha=baseline_settings.ridge_alpha,
                velocity_bounds=velocity_bounds,
            ),
            model_details_builder=lambda model, split_by_case_id, samples: {
                "artifact_type": "ml_observation_baseline_model_details",
                "experiment_id": baseline_settings.experiment_id,
                "trained_component_count": len(model.principal_components),
                "configured_component_count": baseline_settings.pca_component_count,
                "ridge_alpha": baseline_settings.ridge_alpha,
                "velocity_bounds_km_per_s": list(velocity_bounds),
                "split_case_counts": _split_case_counts(split_by_case_id, samples),
            },
            note_title="Observation-Driven PCA Ridge Baseline",
            objective=(
                "Train the current stabilized linear inversion baseline directly from ordered "
                "source-receiver observations while preserving the frozen full 3D velocity-grid target."
            ),
            notes=[
                "This baseline uses ordered source-receiver observations rather than scenario-level summary statistics.",
                "Targets remain the frozen full 3D absolute-velocity grid and are compressed only through PCA inside the model.",
                "The current split is family-stratified so each geological family contributes train, validation, and test cases.",
            ],
            assumptions=[
                "All cases in the observation corpus share the same source-receiver observation count so one fixed-length feature space is valid.",
                "The current observation corpus varies both velocity-model parameters and bounded acquisition geometry while preserving a fixed observation count per case.",
                "PCA dimensionality and ridge regularization are fixed from centralized config rather than tuned adaptively.",
            ],
            limitations=[
                "The current corpus now includes bounded acquisition-geometry variation, but it is still limited to a small controlled set of station layouts and random earthquake catalogs.",
                "PCA compresses the target grid for tractable supervised learning, so component count remains a modeling assumption.",
                "The current baseline is linear after the observation-feature map and should be treated as an interpretable starting point rather than a final inversion model.",
            ],
            next_step=(
                "Hold this corpus fixed as the stabilized linear reference, then compare it against the plain linear, random-forest, and shallow-MLP alternatives on the same observation inputs."
            ),
        )
    if model_name == "pca_mlp_observation_regressor":
        baseline_settings = settings.machine_learning.observation_mlp_baseline
        return ObservationModelRunConfig(
            experiment_id=baseline_settings.experiment_id,
            model_name=baseline_settings.model_name,
            pairing_manifest_path=baseline_settings.pairing_manifest_path,
            output_dir=baseline_settings.output_dir,
            observation_fields=baseline_settings.observation_fields,
            split_strategy=baseline_settings.split_strategy,
            model_builder=lambda samples: PCAMLPObservationRegressor.fit(
                feature_vectors=[sample.feature_vector for sample in samples],
                target_vectors=[sample.target_vector for sample in samples],
                pca_component_count=baseline_settings.pca_component_count,
                hidden_width=baseline_settings.hidden_width,
                epoch_count=baseline_settings.epoch_count,
                learning_rate=baseline_settings.learning_rate,
                l2_alpha=baseline_settings.l2_alpha,
                velocity_bounds=velocity_bounds,
                random_seed=settings.random_seeds.machine_learning,
            ),
            model_details_builder=lambda model, split_by_case_id, samples: {
                "artifact_type": "ml_observation_baseline_model_details",
                "experiment_id": baseline_settings.experiment_id,
                "trained_component_count": len(model.principal_components),
                "configured_component_count": baseline_settings.pca_component_count,
                "hidden_width": baseline_settings.hidden_width,
                "epoch_count": baseline_settings.epoch_count,
                "learning_rate": baseline_settings.learning_rate,
                "l2_alpha": baseline_settings.l2_alpha,
                "velocity_bounds_km_per_s": list(velocity_bounds),
                "split_case_counts": _split_case_counts(split_by_case_id, samples),
            },
            note_title="Observation-Driven PCA MLP Baseline",
            objective=(
                "Train the first shallow non-linear inversion baseline from ordered source-receiver "
                "observations while preserving the same frozen full 3D velocity-grid target."
            ),
            notes=[
                "This baseline keeps the same ordered source-receiver observation inputs used by the linear and ridge reference models.",
                "Targets remain the frozen full 3D absolute-velocity grid and are still reconstructed from PCA coefficients for direct comparability.",
                "The shallow MLP adds one bounded non-linear mapping step without changing the supervised dataset contract.",
            ],
            assumptions=[
                "All cases in the observation corpus share the same source-receiver observation count so one fixed-length feature space is valid.",
                "The current observation corpus varies both velocity-model parameters and bounded acquisition geometry while preserving a fixed observation count per case.",
                "The single hidden-layer MLP is intentionally small so it remains tractable and explainable within the current synthetic benchmark workflow.",
            ],
            limitations=[
                "The current corpus remains synthetic and bounded to the five configured geological families.",
                "The MLP still relies on PCA compression of the target grid, so reconstruction quality depends partly on the configured component count.",
                "This implementation uses one small hidden layer and full-batch gradient descent, so it is a deliberate baseline rather than an optimized deep-learning system.",
            ],
            next_step=(
                "Compare this shallow non-linear baseline directly against the linear, ridge, and random-forest alternatives on the fixed observation-driven corpus."
            ),
        )
    if model_name == "pca_random_forest_observation_regressor":
        baseline_settings = settings.machine_learning.observation_random_forest_baseline
        return ObservationModelRunConfig(
            experiment_id=baseline_settings.experiment_id,
            model_name=baseline_settings.model_name,
            pairing_manifest_path=baseline_settings.pairing_manifest_path,
            output_dir=baseline_settings.output_dir,
            observation_fields=baseline_settings.observation_fields,
            split_strategy=baseline_settings.split_strategy,
            model_builder=lambda samples: PCARandomForestObservationRegressor.fit(
                feature_vectors=[sample.feature_vector for sample in samples],
                target_vectors=[sample.target_vector for sample in samples],
                pca_component_count=baseline_settings.pca_component_count,
                tree_count=baseline_settings.tree_count,
                max_depth=baseline_settings.max_depth,
                min_samples_leaf=baseline_settings.min_samples_leaf,
                max_feature_count=baseline_settings.max_feature_count,
                velocity_bounds=velocity_bounds,
                random_seed=settings.random_seeds.machine_learning,
            ),
            model_details_builder=lambda model, split_by_case_id, samples: {
                "artifact_type": "ml_observation_baseline_model_details",
                "experiment_id": baseline_settings.experiment_id,
                "trained_component_count": len(model.principal_components),
                "configured_component_count": baseline_settings.pca_component_count,
                "tree_count": baseline_settings.tree_count,
                "max_depth": baseline_settings.max_depth,
                "min_samples_leaf": baseline_settings.min_samples_leaf,
                "max_feature_count": baseline_settings.max_feature_count,
                "velocity_bounds_km_per_s": list(velocity_bounds),
                "split_case_counts": _split_case_counts(split_by_case_id, samples),
            },
            note_title="Observation-Driven PCA Random-Forest Baseline",
            objective=(
                "Train a bounded tree-ensemble baseline from ordered source-receiver observations "
                "while preserving the same frozen full 3D velocity-grid target."
            ),
            notes=[
                "This baseline keeps the same observation-driven input contract used by the linear, ridge, and shallow-MLP runs.",
                "Targets remain the frozen full 3D absolute-velocity grid and are reconstructed from PCA coefficients for direct comparability.",
                "The random forest introduces a representative non-linear ensemble model without changing the dataset schema or split logic.",
            ],
            assumptions=[
                "All cases in the observation corpus share the same source-receiver observation count so one fixed-length feature space is valid.",
                "The current observation corpus varies both velocity-model parameters and bounded acquisition geometry while preserving a fixed observation count per case.",
                "Tree count, depth, and feature subsampling stay centrally fixed so the current workflow remains a limited representative comparison rather than a large benchmark sweep.",
            ],
            limitations=[
                "The current corpus remains synthetic and bounded to the five configured geological families.",
                "The forest predicts a low-dimensional PCA target representation rather than the full grid directly, so component count remains a modeling assumption.",
                "This small pure-Python forest is intended for controlled comparison only and should not be interpreted as an optimized production ensemble implementation.",
            ],
            next_step=(
                "Compare this bounded ensemble baseline against the plain linear, ridge, and shallow-MLP models on the same fixed synthetic split."
            ),
        )
    raise ValueError(f"Unsupported observation comparison model: {model_name}")


def _validate_comparison_model_inputs(
    run_configs: list[ObservationModelRunConfig],
    comparison_settings: Any,
) -> None:
    if not run_configs:
        raise ValueError("Observation comparison requires at least one configured model.")
    reference_fields = comparison_settings.observation_fields
    for run_config in run_configs:
        if run_config.pairing_manifest_path != comparison_settings.pairing_manifest_path:
            raise ValueError(
                "Observation comparison model pairing manifests must match the configured comparison manifest."
            )
        if run_config.observation_fields != reference_fields:
            raise ValueError(
                "Observation comparison model observation_fields must match the configured comparison fields."
            )
        if run_config.split_strategy != comparison_settings.split_strategy:
            raise ValueError(
                "Observation comparison model split_strategy values must match the configured comparison split."
            )


def _paper_model_run_config(
    settings: BenchmarkSettings,
    *,
    model_name: str,
    experiment_id: str,
    pairing_manifest_path: Path,
    output_dir: Path,
    observation_fields: tuple[str, ...],
    split_strategy: str = "family_stratified",
) -> ObservationModelRunConfig:
    base_config = _observation_model_run_config(settings, model_name)
    return replace(
        base_config,
        experiment_id=experiment_id,
        pairing_manifest_path=pairing_manifest_path,
        output_dir=output_dir,
        observation_fields=observation_fields,
        split_strategy=split_strategy,
        note_title=f"Paper Pseudo-Bending {base_config.note_title}",
        objective=(
            f"{base_config.objective} This Phase 4 run consumes the dedicated "
            "paper_pseudo_bending_observation_pairs_v1 corpus."
        ),
        assumptions=[
            *base_config.assumptions,
            "The input manifest is the Phase 3 pseudo-bending corpus, not the historical observation/eikonal corpus.",
        ],
        limitations=[
            *base_config.limitations,
            "This per-model artifact is one fixed split within the broader paper pseudo-bending ML suite.",
        ],
    )


def _prepare_paper_pseudo_bending_training_context(
    *,
    settings: BenchmarkSettings,
    pairing_manifest_path: Path,
    observation_fields: tuple[str, ...],
    random_seed: int,
    split_strategy: str = "family_stratified",
) -> ObservationTrainingContext:
    repo_root = get_repo_root()
    resolved_manifest_path = _resolve_repo_path(pairing_manifest_path, repo_root)
    resolved_manifest_path = _ensure_paper_pseudo_bending_manifest(
        settings,
        resolved_manifest_path,
    )
    pairing_manifest = _read_json(resolved_manifest_path)
    if str(pairing_manifest.get("simulator_name", "")) != "pseudo_bending_3d":
        raise ValueError("Paper pseudo-bending ML suite requires a pseudo_bending_3d manifest.")
    if "paper_pseudo_bending_observation_pairs_v1" not in str(
        pairing_manifest.get("corpus_id", pairing_manifest.get("batch_id", ""))
    ):
        raise ValueError("Paper pseudo-bending ML suite received the wrong corpus manifest.")
    feature_matrix = assemble_observation_case_feature_matrix(
        resolved_manifest_path,
        repo_root,
        observation_fields=observation_fields,
    )
    samples = _attach_targets(feature_matrix, pairing_manifest, repo_root)
    split_by_case_id = _split_samples_by_strategy(
        samples,
        train_fraction=settings.machine_learning.train_fraction,
        validation_fraction=settings.machine_learning.validation_fraction,
        random_seed=random_seed,
        split_strategy=split_strategy,
    )
    return ObservationTrainingContext(
        manifest_path=resolved_manifest_path,
        pairing_manifest=pairing_manifest,
        feature_matrix=feature_matrix,
        samples=samples,
        split_by_case_id=split_by_case_id,
    )


def _ensure_paper_pseudo_bending_manifest(
    settings: BenchmarkSettings,
    manifest_path: Path,
) -> Path:
    if manifest_path.is_file():
        return manifest_path
    outputs = prepare_paper_pseudo_bending_observation_corpus(settings)
    return outputs.manifest_path


def _evaluate_repeated_family_stratified_splits(
    *,
    settings: BenchmarkSettings,
    context: ObservationTrainingContext,
    model_name: str,
    split_seeds: tuple[int, ...],
    split_strategy: str = "family_stratified",
) -> dict[str, Any]:
    split_records: list[dict[str, Any]] = []
    for split_seed in split_seeds:
        split_by_case_id = _split_samples_by_strategy(
            context.samples,
            train_fraction=settings.machine_learning.train_fraction,
            validation_fraction=settings.machine_learning.validation_fraction,
            random_seed=split_seed,
            split_strategy=split_strategy,
        )
        model = _fit_suite_model(
            settings, model_name, _samples_for_split(context.samples, split_by_case_id, "train")
        )
        split_records.append(
            {
                "split_seed": split_seed,
                "train": _evaluate_split_metrics_only(
                    _samples_for_split(context.samples, split_by_case_id, "train"),
                    model,
                ),
                "validation": _evaluate_split_metrics_only(
                    _samples_for_split(context.samples, split_by_case_id, "validation"),
                    model,
                ),
                "test": _evaluate_split_metrics_only(
                    _samples_for_split(context.samples, split_by_case_id, "test"),
                    model,
                ),
            }
        )
    return {
        "model_name": model_name,
        "split_records": split_records,
        "aggregate_metrics": _aggregate_split_records(split_records),
    }


def _evaluate_leave_one_family_out(
    *,
    settings: BenchmarkSettings,
    context: ObservationTrainingContext,
    model_name: str,
) -> dict[str, Any]:
    families = sorted({sample.scenario for sample in context.samples})
    held_out_records: list[dict[str, Any]] = []
    for family in families:
        train_samples = [sample for sample in context.samples if sample.scenario != family]
        held_out_samples = [sample for sample in context.samples if sample.scenario == family]
        model = _fit_suite_model(settings, model_name, train_samples)
        held_out_records.append(
            {
                "held_out_family": family,
                "train_case_count": len(train_samples),
                "held_out": _evaluate_split_metrics_only(held_out_samples, model),
            }
        )
    return {
        "model_name": model_name,
        "families_evaluated": families,
        "held_out_records": held_out_records,
        "aggregate_metrics": {
            "mean_held_out_mae": _mean_record_metric(
                held_out_records,
                "held_out",
                "mean_mae",
            ),
            "std_held_out_mae": _std_record_metric(
                held_out_records,
                "held_out",
                "mean_mae",
            ),
            "mean_held_out_rmse": _mean_record_metric(
                held_out_records,
                "held_out",
                "mean_rmse",
            ),
            "std_held_out_rmse": _std_record_metric(
                held_out_records,
                "held_out",
                "mean_rmse",
            ),
        },
    }


def _evaluate_noise_robustness(
    *,
    settings: BenchmarkSettings,
    context: ObservationTrainingContext,
    model_name: str,
    noise_levels_s: tuple[float, ...],
    noise_random_seed: int,
) -> dict[str, Any]:
    train_samples = _samples_for_split(context.samples, context.split_by_case_id, "train")
    model = _fit_suite_model(settings, model_name, train_samples)
    level_records: list[dict[str, Any]] = []
    for level in noise_levels_s:
        noisy_samples = _with_travel_time_noise(
            context.samples,
            observation_fields=context.feature_matrix.observation_feature_fields,
            noise_std_s=level,
            random_seed=noise_random_seed + int(round(level * 1000.0)),
        )
        level_records.append(
            {
                "noise_std_s": level,
                "train": _evaluate_split_metrics_only(
                    _samples_for_split(noisy_samples, context.split_by_case_id, "train"),
                    model,
                ),
                "validation": _evaluate_split_metrics_only(
                    _samples_for_split(noisy_samples, context.split_by_case_id, "validation"),
                    model,
                ),
                "test": _evaluate_split_metrics_only(
                    _samples_for_split(noisy_samples, context.split_by_case_id, "test"),
                    model,
                ),
            }
        )
    return {
        "model_name": model_name,
        "noise_level_records": level_records,
    }


def _fit_suite_model(
    settings: BenchmarkSettings,
    model_name: str,
    train_samples: list[ObservationTrainingSample],
) -> ObservationBaselineModel:
    velocity_bounds = settings.velocity_model_generation.velocity_bounds_km_per_s
    if model_name == "pca_linear_observation_regressor":
        baseline_settings = settings.machine_learning.observation_linear_baseline
        return PCALinearObservationRegressor.fit(
            feature_vectors=[sample.feature_vector for sample in train_samples],
            target_vectors=[sample.target_vector for sample in train_samples],
            pca_component_count=baseline_settings.pca_component_count,
            velocity_bounds=velocity_bounds,
        )
    if model_name == "pca_ridge_observation_regressor":
        baseline_settings = settings.machine_learning.observation_baseline
        return PCARidgeObservationRegressor.fit(
            feature_vectors=[sample.feature_vector for sample in train_samples],
            target_vectors=[sample.target_vector for sample in train_samples],
            pca_component_count=baseline_settings.pca_component_count,
            ridge_alpha=baseline_settings.ridge_alpha,
            velocity_bounds=velocity_bounds,
        )
    if model_name == "pca_random_forest_observation_regressor":
        baseline_settings = settings.machine_learning.observation_random_forest_baseline
        return PCARandomForestObservationRegressor.fit(
            feature_vectors=[sample.feature_vector for sample in train_samples],
            target_vectors=[sample.target_vector for sample in train_samples],
            pca_component_count=baseline_settings.pca_component_count,
            tree_count=baseline_settings.tree_count,
            max_depth=baseline_settings.max_depth,
            min_samples_leaf=baseline_settings.min_samples_leaf,
            max_feature_count=baseline_settings.max_feature_count,
            velocity_bounds=velocity_bounds,
            random_seed=settings.random_seeds.machine_learning,
        )
    if model_name == "pca_mlp_observation_regressor":
        baseline_settings = settings.machine_learning.observation_mlp_baseline
        return PCAMLPObservationRegressor.fit(
            feature_vectors=[sample.feature_vector for sample in train_samples],
            target_vectors=[sample.target_vector for sample in train_samples],
            pca_component_count=baseline_settings.pca_component_count,
            hidden_width=baseline_settings.hidden_width,
            epoch_count=baseline_settings.epoch_count,
            learning_rate=baseline_settings.learning_rate,
            l2_alpha=baseline_settings.l2_alpha,
            velocity_bounds=velocity_bounds,
            random_seed=settings.random_seeds.machine_learning,
        )
    raise ValueError(f"Unsupported paper pseudo-bending ML suite model: {model_name}")


def _strengthened_candidates(
    candidate_models: tuple[str, ...],
    pca_component_counts: tuple[int, ...],
    ridge_alphas: tuple[float, ...],
) -> list[tuple[str, int, float | None]]:
    candidates: list[tuple[str, int, float | None]] = []
    for model_name in candidate_models:
        for pca_component_count in pca_component_counts:
            if model_name == "pca_linear_observation_regressor":
                candidates.append((model_name, pca_component_count, None))
            elif model_name == "pca_ridge_observation_regressor":
                for ridge_alpha in ridge_alphas:
                    candidates.append((model_name, pca_component_count, ridge_alpha))
            else:
                raise ValueError(f"Unsupported strengthened model candidate: {model_name}")
    return candidates


def _evaluate_strengthened_candidate(
    *,
    settings: BenchmarkSettings,
    context: ObservationTrainingContext,
    model_name: str,
    pca_component_count: int,
    ridge_alpha: float | None,
    repeated_split_seeds: tuple[int, ...],
) -> dict[str, Any]:
    train_samples = _samples_for_split(context.samples, context.split_by_case_id, "train")
    model = _fit_strengthened_model(
        settings=settings,
        model_name=model_name,
        train_samples=train_samples,
        pca_component_count=pca_component_count,
        ridge_alpha=ridge_alpha,
    )
    fixed_split = {
        split_name: _evaluate_split_metrics_only(
            _samples_for_split(context.samples, context.split_by_case_id, split_name),
            model,
        )
        for split_name in ("train", "validation", "test")
    }
    repeated_split = _evaluate_strengthened_repeated_splits(
        settings=settings,
        context=context,
        model_name=model_name,
        pca_component_count=pca_component_count,
        ridge_alpha=ridge_alpha,
        split_seeds=repeated_split_seeds,
    )
    leave_one_family_out = _evaluate_strengthened_leave_one_family_out(
        settings=settings,
        context=context,
        model_name=model_name,
        pca_component_count=pca_component_count,
        ridge_alpha=ridge_alpha,
    )
    return {
        "model_name": model_name,
        "pca_component_count": pca_component_count,
        "ridge_alpha": ridge_alpha,
        "fixed_split": fixed_split,
        "repeated_split": repeated_split,
        "leave_one_family_out": leave_one_family_out,
    }


def _evaluate_strengthened_repeated_splits(
    *,
    settings: BenchmarkSettings,
    context: ObservationTrainingContext,
    model_name: str,
    pca_component_count: int,
    ridge_alpha: float | None,
    split_seeds: tuple[int, ...],
) -> dict[str, Any]:
    split_records: list[dict[str, Any]] = []
    for split_seed in split_seeds:
        split_by_case_id = _family_stratified_split(
            context.samples,
            train_fraction=settings.machine_learning.train_fraction,
            validation_fraction=settings.machine_learning.validation_fraction,
            random_seed=split_seed,
        )
        model = _fit_strengthened_model(
            settings=settings,
            model_name=model_name,
            train_samples=_samples_for_split(context.samples, split_by_case_id, "train"),
            pca_component_count=pca_component_count,
            ridge_alpha=ridge_alpha,
        )
        split_records.append(
            {
                "split_seed": split_seed,
                "train": _evaluate_split_metrics_only(
                    _samples_for_split(context.samples, split_by_case_id, "train"),
                    model,
                ),
                "validation": _evaluate_split_metrics_only(
                    _samples_for_split(context.samples, split_by_case_id, "validation"),
                    model,
                ),
                "test": _evaluate_split_metrics_only(
                    _samples_for_split(context.samples, split_by_case_id, "test"),
                    model,
                ),
            }
        )
    return {
        "split_records": split_records,
        "aggregate_metrics": _aggregate_split_records(split_records),
    }


def _evaluate_strengthened_leave_one_family_out(
    *,
    settings: BenchmarkSettings,
    context: ObservationTrainingContext,
    model_name: str,
    pca_component_count: int,
    ridge_alpha: float | None,
) -> dict[str, Any]:
    families = sorted({sample.scenario for sample in context.samples})
    held_out_records: list[dict[str, Any]] = []
    for family in families:
        train_samples = [sample for sample in context.samples if sample.scenario != family]
        held_out_samples = [sample for sample in context.samples if sample.scenario == family]
        model = _fit_strengthened_model(
            settings=settings,
            model_name=model_name,
            train_samples=train_samples,
            pca_component_count=pca_component_count,
            ridge_alpha=ridge_alpha,
        )
        held_out_records.append(
            {
                "held_out_family": family,
                "train_case_count": len(train_samples),
                "held_out": _evaluate_split_metrics_only(held_out_samples, model),
            }
        )
    ranked = _rank_strengthened_loo_records(held_out_records)
    return {
        "families_evaluated": families,
        "held_out_records": held_out_records,
        "held_out_family_ranking": ranked,
        "worst_held_out_family": ranked[0]["held_out_family"],
        "worst_held_out_rmse": ranked[0]["mean_rmse"],
        "mean_held_out_rmse": _mean_record_metric(held_out_records, "held_out", "mean_rmse"),
        "std_held_out_rmse": _std_record_metric(held_out_records, "held_out", "mean_rmse"),
    }


def _evaluate_strengthened_noise_robustness(
    *,
    context: ObservationTrainingContext,
    model: ObservationBaselineModel,
    noise_levels_s: tuple[float, ...],
    noise_random_seed: int,
) -> dict[str, Any]:
    level_records: list[dict[str, Any]] = []
    for level in noise_levels_s:
        noisy_samples = _with_travel_time_noise(
            context.samples,
            observation_fields=context.feature_matrix.observation_feature_fields,
            noise_std_s=level,
            random_seed=noise_random_seed + int(round(level * 1000.0)),
        )
        level_records.append(
            {
                "noise_std_s": level,
                "train": _evaluate_split_metrics_only(
                    _samples_for_split(noisy_samples, context.split_by_case_id, "train"),
                    model,
                ),
                "validation": _evaluate_split_metrics_only(
                    _samples_for_split(noisy_samples, context.split_by_case_id, "validation"),
                    model,
                ),
                "test": _evaluate_split_metrics_only(
                    _samples_for_split(noisy_samples, context.split_by_case_id, "test"),
                    model,
                ),
            }
        )
    return {
        "noise_level_records": level_records,
        "test_rmse_delta_0_to_max_noise": (
            level_records[-1]["test"]["mean_rmse"] - level_records[0]["test"]["mean_rmse"]
        ),
    }


def _fit_strengthened_model(
    *,
    settings: BenchmarkSettings,
    model_name: str,
    train_samples: list[ObservationTrainingSample],
    pca_component_count: int,
    ridge_alpha: float | None,
) -> ObservationBaselineModel:
    velocity_bounds = settings.velocity_model_generation.velocity_bounds_km_per_s
    if model_name == "pca_linear_observation_regressor":
        return PCALinearObservationRegressor.fit(
            feature_vectors=[sample.feature_vector for sample in train_samples],
            target_vectors=[sample.target_vector for sample in train_samples],
            pca_component_count=pca_component_count,
            velocity_bounds=velocity_bounds,
        )
    if model_name == "pca_ridge_observation_regressor":
        if ridge_alpha is None:
            raise ValueError("ridge_alpha must be provided for pca_ridge_observation_regressor.")
        return PCARidgeObservationRegressor.fit(
            feature_vectors=[sample.feature_vector for sample in train_samples],
            target_vectors=[sample.target_vector for sample in train_samples],
            pca_component_count=pca_component_count,
            ridge_alpha=ridge_alpha,
            velocity_bounds=velocity_bounds,
        )
    raise ValueError(f"Unsupported strengthened model: {model_name}")


def _rank_strengthened_loo_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _strengthened_remaining_risks(
    best_candidate: dict[str, Any],
    baseline_audit: dict[str, Any],
    metric_deltas: dict[str, float],
) -> list[str]:
    risks: list[str] = []
    worst_loo_rmse = float(best_candidate["leave_one_family_out"]["worst_held_out_rmse"])
    repeated_validation_rmse = float(
        best_candidate["repeated_split"]["aggregate_metrics"]["mean_validation_rmse"]
    )
    fixed_validation_rmse = float(best_candidate["fixed_split"]["validation"]["mean_rmse"])
    if worst_loo_rmse > 0.15:
        risks.append(f"Worst leave-one-family-out RMSE remains high ({worst_loo_rmse:.6f} km/s).")
    if repeated_validation_rmse / fixed_validation_rmse > 1.25:
        risks.append(
            "Repeated-split validation RMSE remains materially higher than fixed validation RMSE."
        )
    if metric_deltas["worst_leave_one_family_out_rmse_delta"] >= 0.0:
        risks.append("Strengthening did not improve the worst leave-one-family-out RMSE.")
    if baseline_audit.get("proceed_to_phase_5") is False and not risks:
        risks.append(
            "Phase 4.5 audit was negative; review strengthened diagnostics before Phase 5."
        )
    return risks


def _diagnose_faulted_generalization(
    *,
    settings: BenchmarkSettings,
    context: ObservationTrainingContext,
    baseline_audit: dict[str, Any],
    target_family: str,
    model_name: str,
    pca_component_count: int,
    ridge_alpha: float | None,
) -> dict[str, Any]:
    family_counts = _count_by_key(
        _dict_list(context.pairing_manifest["items"]),
        "family",
    )
    faulted_items = [
        item
        for item in _dict_list(context.pairing_manifest["items"])
        if str(item.get("family", item.get("scenario", ""))) == target_family
    ]
    faulted_variant_counts = _count_by_key(faulted_items, "parameter_variant_id")
    faulted_geometry_counts = _count_by_key(faulted_items, "geometry_profile_id")
    fault_override_records = _fault_override_records(faulted_items)
    held_out_faulted_diagnostics = _held_out_family_target_error_diagnostics(
        settings=settings,
        context=context,
        target_family=target_family,
        model_name=model_name,
        pca_component_count=pca_component_count,
        ridge_alpha=ridge_alpha,
    )
    worst_family = baseline_audit["leave_one_family_out"]["worst_family"]
    diagnosis_summary = {
        "targeted_weakness": target_family,
        "audit_worst_family": worst_family["held_out_family"],
        "audit_worst_family_rmse": worst_family["mean_rmse"],
        "faulted_case_count": len(faulted_items),
        "faulted_parameter_variant_count": len(faulted_variant_counts),
        "faulted_geometry_profile_count": len(faulted_geometry_counts),
        "primary_diagnosis": (
            "Faulted is the weakest leave-one-family-out family. The corpus has balanced "
            "faulted variants and acquisition profiles, but when faulted is held out the "
            "training families do not contain an equivalent planar velocity-offset target, "
            "so the task is family extrapolation rather than within-family interpolation."
        ),
        "intervention_supported": "directional_slowness_v1",
    }
    return {
        "artifact_type": "paper_pseudo_bending_faulted_generalization_diagnosis",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_manifest_path": _relative_to_repo_or_absolute(
            context.manifest_path, get_repo_root()
        ),
        "simulator": str(context.pairing_manifest.get("simulator_name", "")),
        "diagnosis_summary": diagnosis_summary,
        "family_case_counts": family_counts,
        "faulted_parameter_variant_counts": faulted_variant_counts,
        "faulted_geometry_profile_counts": faulted_geometry_counts,
        "faulted_fault_parameter_ranges": _fault_parameter_ranges(fault_override_records),
        "held_out_faulted_target_error_diagnostics": held_out_faulted_diagnostics,
        "candidate_causes": [
            {
                "cause": "too_few_faulted_parameter_variants",
                "supported": len(faulted_variant_counts) < 4,
                "evidence": f"{len(faulted_variant_counts)} configured faulted variants.",
            },
            {
                "cause": "limited_fault_geometry_scope",
                "supported": True,
                "evidence": (
                    "Configured faulted variants vary x position, side, dip, and velocity offset, "
                    "but keep strike fixed at 90 degrees in the Phase 3 corpus."
                ),
            },
            {
                "cause": "missing_planar_discontinuity_family_when_faulted_is_held_out",
                "supported": True,
                "evidence": (
                    "Leave-one-family-out for faulted trains on layered, block_anomaly, "
                    "salt_dome, and dyke_intrusion targets only."
                ),
            },
            {
                "cause": "feature_representation_may_understate_directional_asymmetry",
                "supported": True,
                "evidence": (
                    "The base feature vector stores ordered observations directly but has no "
                    "explicit directional apparent-slowness aggregates."
                ),
            },
        ],
    }


def _evaluate_faulted_generalization_candidate(
    *,
    settings: BenchmarkSettings,
    context: ObservationTrainingContext,
    model_name: str,
    pca_component_count: int,
    ridge_alpha: float | None,
    repeated_split_seeds: tuple[int, ...],
) -> dict[str, Any]:
    return _evaluate_strengthened_candidate(
        settings=settings,
        context=context,
        model_name=model_name,
        pca_component_count=pca_component_count,
        ridge_alpha=ridge_alpha,
        repeated_split_seeds=repeated_split_seeds,
    )


def _with_directional_slowness_features(
    context: ObservationTrainingContext,
    *,
    observation_fields: tuple[str, ...],
) -> ObservationTrainingContext:
    enhanced_samples: list[ObservationTrainingSample] = []
    for sample in context.samples:
        derived = _directional_slowness_features(sample.feature_vector, observation_fields)
        feature_row = dict(sample.feature_row)
        feature_row.update(derived)
        enhanced_samples.append(
            replace(
                sample,
                feature_row=feature_row,
                feature_vector=tuple(sample.feature_vector) + tuple(derived.values()),
            )
        )
    return replace(context, samples=tuple(enhanced_samples))


def _directional_slowness_features(
    feature_vector: tuple[float, ...],
    observation_fields: tuple[str, ...],
) -> dict[str, float]:
    field_index = {field: index for index, field in enumerate(observation_fields)}
    required_fields = (
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "receiver_x_km",
        "receiver_y_km",
        "receiver_z_km",
        "path_length_km",
        "travel_time_s",
    )
    missing_fields = [field for field in required_fields if field not in field_index]
    if missing_fields:
        raise ValueError(
            "directional_slowness_v1 requires observation fields: " + ", ".join(required_fields)
        )
    field_count = len(observation_fields)
    observation_count = len(feature_vector) // field_count
    slowness_values: list[float] = []
    x_positive: list[float] = []
    x_negative: list[float] = []
    y_positive: list[float] = []
    y_negative: list[float] = []
    center_crossing: list[float] = []
    non_crossing: list[float] = []
    long_paths: list[float] = []
    short_paths: list[float] = []
    x_spans: list[float] = []
    for observation_index in range(observation_count):
        offset = observation_index * field_count
        source_x = feature_vector[offset + field_index["source_x_km"]]
        source_y = feature_vector[offset + field_index["source_y_km"]]
        receiver_x = feature_vector[offset + field_index["receiver_x_km"]]
        receiver_y = feature_vector[offset + field_index["receiver_y_km"]]
        path_length = max(1.0e-9, feature_vector[offset + field_index["path_length_km"]])
        travel_time = feature_vector[offset + field_index["travel_time_s"]]
        slowness = travel_time / path_length
        slowness_values.append(slowness)
        delta_x = receiver_x - source_x
        delta_y = receiver_y - source_y
        x_spans.append(abs(delta_x))
        (x_positive if delta_x >= 0.0 else x_negative).append(slowness)
        (y_positive if delta_y >= 0.0 else y_negative).append(slowness)
        crosses_center = (source_x - 50.0) * (receiver_x - 50.0) <= 0.0
        (center_crossing if crosses_center else non_crossing).append(slowness)
        if path_length >= 55.0:
            long_paths.append(slowness)
        if path_length <= 35.0:
            short_paths.append(slowness)
    return {
        "derived_mean_apparent_slowness_s_per_km": _mean_or_zero(slowness_values),
        "derived_std_apparent_slowness_s_per_km": _population_std_for_values(slowness_values),
        "derived_x_direction_slowness_delta": _mean_or_zero(x_positive) - _mean_or_zero(x_negative),
        "derived_y_direction_slowness_delta": _mean_or_zero(y_positive) - _mean_or_zero(y_negative),
        "derived_center_crossing_slowness_delta": _mean_or_zero(center_crossing)
        - _mean_or_zero(non_crossing),
        "derived_center_crossing_fraction": len(center_crossing) / observation_count,
        "derived_long_minus_short_path_slowness": _mean_or_zero(long_paths)
        - _mean_or_zero(short_paths),
        "derived_mean_abs_x_span_km": _mean_or_zero(x_spans),
        "derived_std_abs_x_span_km": _population_std_for_values(x_spans),
    }


def _held_out_family_target_error_diagnostics(
    *,
    settings: BenchmarkSettings,
    context: ObservationTrainingContext,
    target_family: str,
    model_name: str,
    pca_component_count: int,
    ridge_alpha: float | None,
) -> dict[str, Any]:
    train_samples = [sample for sample in context.samples if sample.scenario != target_family]
    held_out_samples = [sample for sample in context.samples if sample.scenario == target_family]
    model = _fit_strengthened_model(
        settings=settings,
        model_name=model_name,
        train_samples=train_samples,
        pca_component_count=pca_component_count,
        ridge_alpha=ridge_alpha,
    )
    case_records = []
    near_errors: list[float] = []
    far_errors: list[float] = []
    for sample in held_out_samples:
        prediction = model.predict(sample.feature_vector)
        grid_payload = _read_json(sample.target_grid_path)
        near_indices = _near_fault_node_indices(grid_payload)
        near_case_errors = [
            abs(sample.target_vector[index] - prediction[index]) for index in near_indices
        ]
        far_case_errors = [
            abs(sample.target_vector[index] - prediction[index])
            for index in range(len(sample.target_vector))
            if index not in near_indices
        ]
        near_errors.extend(near_case_errors)
        far_errors.extend(far_case_errors)
        case_records.append(
            {
                "case_id": sample.case_id,
                "near_fault_node_count": len(near_case_errors),
                "far_node_count": len(far_case_errors),
                "near_fault_mae": _mean_or_zero(near_case_errors),
                "far_field_mae": _mean_or_zero(far_case_errors),
                "near_minus_far_mae": _mean_or_zero(near_case_errors)
                - _mean_or_zero(far_case_errors),
            }
        )
    return {
        "model_name": model_name,
        "held_out_family": target_family,
        "case_count": len(case_records),
        "near_fault_mae": _mean_or_zero(near_errors),
        "far_field_mae": _mean_or_zero(far_errors),
        "near_minus_far_mae": _mean_or_zero(near_errors) - _mean_or_zero(far_errors),
        "case_records": case_records,
    }


def _near_fault_node_indices(grid_payload: dict[str, Any]) -> set[int]:
    metadata = dict(grid_payload["metadata"])
    faulted = dict(metadata["faulted"])
    x_coordinates = [float(value) for value in grid_payload["x_coordinates_km"]]
    y_coordinates = [float(value) for value in grid_payload["y_coordinates_km"]]
    z_coordinates = [float(value) for value in grid_payload["z_coordinates_km"]]
    nx = len(x_coordinates)
    ny = len(y_coordinates)
    spacing = float(dict(metadata["grid_spacing_km"])["x"])
    near_indices: set[int] = set()
    for iz, z_value in enumerate(z_coordinates):
        for iy, y_value in enumerate(y_coordinates):
            for ix, x_value in enumerate(x_coordinates):
                signed_offset = _fault_signed_offset_from_metadata(
                    x_value,
                    y_value,
                    z_value,
                    faulted,
                )
                if abs(signed_offset) <= spacing:
                    near_indices.add(ix + nx * (iy + ny * iz))
    return near_indices


def _fault_signed_offset_from_metadata(
    x_km: float,
    y_km: float,
    z_km: float,
    faulted: dict[str, Any],
) -> float:
    strike_rad = math.radians(float(faulted["strike_deg"]))
    normal_x = math.cos(strike_rad)
    normal_y = -math.sin(strike_rad)
    dip_deg = float(faulted["dip_deg"])
    horizontal_offset = (
        0.0 if math.isclose(dip_deg, 90.0) else z_km / math.tan(math.radians(dip_deg))
    )
    dip_sign = 1.0 if str(faulted["dip_direction"]) == "positive_normal" else -1.0
    return (
        (x_km - float(faulted["fault_x_km"])) * normal_x
        + (y_km - float(faulted["fault_y_km"])) * normal_y
        - dip_sign * horizontal_offset
    )


def _faulted_generalization_comparison(
    *,
    baseline_suite: dict[str, Any],
    strengthened_summary: dict[str, Any],
    base_record: dict[str, Any],
    enhanced_record: dict[str, Any],
    target_family: str,
) -> dict[str, Any]:
    phase4_loo = _phase4_worst_loo_from_suite(baseline_suite)
    strengthened_loo = strengthened_summary["best_model"]["leave_one_family_out"]
    base_faulted = _held_out_family_metric(base_record, target_family)
    enhanced_faulted = _held_out_family_metric(enhanced_record, target_family)
    return {
        "phase4_best_model": baseline_suite["best_model"]["model_name"],
        "phase4_fixed_validation_rmse": baseline_suite["best_model"]["fixed_validation_rmse"],
        "phase4_fixed_test_rmse": baseline_suite["best_model"]["fixed_test_rmse"],
        "phase4_faulted_loo_rmse": phase4_loo["mean_rmse"],
        "phase4_faulted_loo_mae": phase4_loo["mean_mae"],
        "phase45b_best_model": strengthened_summary["best_model"]["model_name"],
        "phase45b_faulted_loo_rmse": strengthened_loo["worst_held_out_rmse"],
        "base_same_model_faulted_loo_rmse": base_faulted["mean_rmse"],
        "base_same_model_faulted_loo_mae": base_faulted["mean_mae"],
        "directional_features_faulted_loo_rmse": enhanced_faulted["mean_rmse"],
        "directional_features_faulted_loo_mae": enhanced_faulted["mean_mae"],
        "directional_features_worst_family": (
            enhanced_record["leave_one_family_out"]["worst_held_out_family"]
        ),
        "directional_features_worst_family_rmse": (
            enhanced_record["leave_one_family_out"]["worst_held_out_rmse"]
        ),
        "faulted_loo_rmse_delta_vs_phase4": enhanced_faulted["mean_rmse"] - phase4_loo["mean_rmse"],
        "faulted_loo_rmse_delta_vs_phase45b": enhanced_faulted["mean_rmse"]
        - strengthened_loo["worst_held_out_rmse"],
        "faulted_loo_rmse_delta_vs_same_model_no_derived_features": (
            enhanced_faulted["mean_rmse"] - base_faulted["mean_rmse"]
        ),
        "fixed_validation_rmse_delta_vs_phase4": (
            enhanced_record["fixed_split"]["validation"]["mean_rmse"]
            - baseline_suite["best_model"]["fixed_validation_rmse"]
        ),
        "fixed_test_rmse_delta_vs_phase4": (
            enhanced_record["fixed_split"]["test"]["mean_rmse"]
            - baseline_suite["best_model"]["fixed_test_rmse"]
        ),
        "repeated_validation_rmse_delta_vs_phase4": (
            enhanced_record["repeated_split"]["aggregate_metrics"]["mean_validation_rmse"]
            - _phase4_repeated_validation_rmse(baseline_suite)
        ),
    }


def _faulted_generalization_recommendation(comparison: dict[str, Any]) -> dict[str, Any]:
    faulted_delta = float(comparison["faulted_loo_rmse_delta_vs_phase4"])
    fixed_test_delta = float(comparison["fixed_test_rmse_delta_vs_phase4"])
    repeated_delta = float(comparison["repeated_validation_rmse_delta_vs_phase4"])
    if faulted_delta < 0.0 and fixed_test_delta <= 0.01 and repeated_delta <= 0.01:
        return {
            "proceed_to_phase_5": True,
            "recommended_main_model": "directional_slowness_feature_model",
            "recommended_next_phase": "Phase 5",
            "rationale": (
                "The bounded feature intervention improved faulted leave-one-family-out behavior "
                "without material fixed/repeated split degradation."
            ),
        }
    if faulted_delta < 0.0:
        return {
            "proceed_to_phase_5": True,
            "recommended_main_model": "pca_linear_observation_regressor",
            "recommended_next_phase": "Phase 5 with faulted-generalization tradeoff reported",
            "rationale": (
                "Faulted leave-one-family-out improved, but overall split metrics degraded enough "
                "that the simpler Phase 4 model remains the cleaner main model."
            ),
        }
    return {
        "proceed_to_phase_5": True,
        "recommended_main_model": "pca_linear_observation_regressor",
        "recommended_next_phase": "Phase 5 with faulted generalization documented as a limitation",
        "rationale": (
            "The bounded intervention did not improve faulted leave-one-family-out behavior. "
            "Stop tuning and report the family-extrapolation weakness explicitly."
        ),
    }


def _phase4_worst_loo_from_suite(baseline_suite: dict[str, Any]) -> dict[str, Any]:
    loo_summary_path = _resolve_repo_path(
        Path(str(baseline_suite["leave_one_family_out_summary_json"])),
        get_repo_root(),
    )
    loo_summary = _read_json(loo_summary_path)
    best_model = baseline_suite["best_model"]["model_name"]
    record = next(item for item in loo_summary["model_records"] if item["model_name"] == best_model)
    return _held_out_family_metric(record, "faulted")


def _phase4_repeated_validation_rmse(baseline_suite: dict[str, Any]) -> float:
    repeated_summary_path = _resolve_repo_path(
        Path(str(baseline_suite["repeated_split_summary_json"])),
        get_repo_root(),
    )
    repeated_summary = _read_json(repeated_summary_path)
    best_model = baseline_suite["best_model"]["model_name"]
    record = next(
        item for item in repeated_summary["model_records"] if item["model_name"] == best_model
    )
    return float(record["aggregate_metrics"]["mean_validation_rmse"])


def _held_out_family_metric(record: dict[str, Any], family: str) -> dict[str, Any]:
    return (
        next(
            item["held_out"]
            for item in record["leave_one_family_out"]["held_out_records"]
            if item["held_out_family"] == family
        )
        if "leave_one_family_out" in record
        else next(
            item["held_out"]
            for item in record["held_out_records"]
            if item["held_out_family"] == family
        )
    )


def _count_by_key(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _fault_override_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in items:
        overrides = dict(item.get("case_overrides", {}))
        faulted = dict(overrides.get("faulted", {}))
        if not faulted:
            faulted = {
                "fault_x_km": 50.0,
                "fault_y_km": 50.0,
                "strike_deg": 90.0,
                "dip_deg": 70.0,
                "dip_direction": "positive_normal",
                "positive_side": "greater_equal",
                "velocity_offset_km_per_s": 0.4,
            }
        records.append(faulted)
    return records


def _fault_parameter_ranges(records: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = (
        "fault_x_km",
        "fault_y_km",
        "strike_deg",
        "dip_deg",
        "velocity_offset_km_per_s",
    )
    ranges: dict[str, Any] = {}
    for field in numeric_fields:
        values = [float(record[field]) for record in records]
        ranges[field] = {
            "min": min(values),
            "max": max(values),
            "unique_values": sorted({round(value, 6) for value in values}),
        }
    ranges["positive_side_values"] = sorted({str(record["positive_side"]) for record in records})
    ranges["dip_direction_values"] = sorted({str(record["dip_direction"]) for record in records})
    return ranges


def _mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class _MeanTargetObservationAdapter:
    model: MeanTargetRegressor

    def predict(self, _feature_vector: tuple[float, ...]) -> tuple[float, ...]:
        return self.model.predict()


def _phase7_ablation_variants(
    observation_fields: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    geometry_fields = tuple(
        field
        for field in observation_fields
        if field
        in {
            "source_x_km",
            "source_y_km",
            "source_z_km",
            "receiver_x_km",
            "receiver_y_km",
            "receiver_z_km",
        }
    )
    return (
        {
            "variant_id": "full_input",
            "model_name": "pca_linear_observation_regressor",
            "observation_fields": observation_fields,
            "path_length_source": "true_model_ray_path_length_km",
            "description": "Current selected feature set: source/receiver geometry, path_length_km, and travel_time_s.",
        },
        {
            "variant_id": "full_input_euclidean_path",
            "model_name": "pca_linear_observation_regressor",
            "observation_fields": observation_fields,
            "path_length_source": "euclidean_source_receiver_distance_km",
            "description": "Full feature layout with path_length_km replaced by Euclidean source-receiver distance.",
        },
        {
            "variant_id": "no_travel_time",
            "model_name": "pca_linear_observation_regressor",
            "observation_fields": tuple(field for field in observation_fields if field != "travel_time_s"),
            "description": "Geometry and path length retained; travel_time_s removed.",
        },
        {
            "variant_id": "shuffled_travel_time",
            "model_name": "pca_linear_observation_regressor",
            "observation_fields": observation_fields,
            "shuffle_travel_time": True,
            "description": "Full feature layout retained, but travel_time_s is permuted within each fixed split.",
        },
        {
            "variant_id": "mean_target",
            "model_name": "mean_target_regressor",
            "observation_fields": (),
            "description": "Training-set element-wise mean target baseline.",
        },
        {
            "variant_id": "geometry_only",
            "model_name": "pca_linear_observation_regressor",
            "observation_fields": geometry_fields,
            "description": "Source and receiver coordinates only.",
        },
        {
            "variant_id": "path_length_only",
            "model_name": "pca_linear_observation_regressor",
            "observation_fields": ("path_length_km",),
            "path_length_source": "true_model_ray_path_length_km",
            "description": "Path length only.",
        },
        {
            "variant_id": "euclidean_path_only",
            "model_name": "pca_linear_observation_regressor",
            "observation_fields": ("path_length_km",),
            "path_length_source": "euclidean_source_receiver_distance_km",
            "description": "Euclidean source-receiver distance only; no target-model ray path length.",
        },
        {
            "variant_id": "travel_time_only",
            "model_name": "pca_linear_observation_regressor",
            "observation_fields": ("travel_time_s",),
            "description": "Travel time only.",
        },
        {
            "variant_id": "family_mean_target_diagnostic",
            "model_name": "family_mean_target_diagnostic",
            "observation_fields": (),
            "diagnostic_only": True,
            "description": "Diagnostic target prior using family labels at prediction time; not deployable.",
        },
    )


def _phase7_variant_context(
    context: ObservationTrainingContext,
    *,
    variant_id: str,
    observation_fields: tuple[str, ...],
    random_seed: int,
    split_by_case_id: dict[str, str] | None = None,
) -> ObservationTrainingContext:
    split_map = split_by_case_id or context.split_by_case_id
    source_samples = context.samples
    if variant_id in {"full_input_euclidean_path", "euclidean_path_only"}:
        source_samples = _replace_path_length_with_euclidean(
            context.samples,
            source_observation_fields=context.feature_matrix.observation_feature_fields,
        )
    samples = _select_observation_fields(
        source_samples,
        source_observation_fields=context.feature_matrix.observation_feature_fields,
        selected_observation_fields=observation_fields,
    )
    if variant_id == "shuffled_travel_time":
        samples = _shuffle_travel_time_within_splits(
            samples,
            observation_fields=observation_fields,
            split_by_case_id=split_map,
            random_seed=random_seed,
        )
    return replace(
        context,
        samples=samples,
        split_by_case_id=split_map,
    )


def _replace_path_length_with_euclidean(
    samples: tuple[ObservationTrainingSample, ...],
    *,
    source_observation_fields: tuple[str, ...],
) -> tuple[ObservationTrainingSample, ...]:
    required_fields = (
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "receiver_x_km",
        "receiver_y_km",
        "receiver_z_km",
        "path_length_km",
    )
    offsets = {field: source_observation_fields.index(field) for field in required_fields}
    field_count = len(source_observation_fields)
    updated_samples: list[ObservationTrainingSample] = []
    for sample in samples:
        observation_count = len(sample.feature_vector) // field_count
        feature_vector = list(sample.feature_vector)
        feature_row = dict(sample.feature_row)
        for observation_index in range(observation_count):
            base = observation_index * field_count
            source = tuple(
                feature_vector[base + offsets[field]]
                for field in ("source_x_km", "source_y_km", "source_z_km")
            )
            receiver = tuple(
                feature_vector[base + offsets[field]]
                for field in ("receiver_x_km", "receiver_y_km", "receiver_z_km")
            )
            distance = math.sqrt(
                sum((left - right) ** 2 for left, right in zip(source, receiver, strict=True))
            )
            path_index = base + offsets["path_length_km"]
            feature_vector[path_index] = distance
            feature_row[
                f"obs_{observation_index:04d}_path_length_km"
            ] = distance
        updated_samples.append(
            replace(
                sample,
                feature_row=feature_row,
                feature_vector=tuple(feature_vector),
            )
        )
    return tuple(updated_samples)


def _select_observation_fields(
    samples: tuple[ObservationTrainingSample, ...],
    *,
    source_observation_fields: tuple[str, ...],
    selected_observation_fields: tuple[str, ...],
) -> tuple[ObservationTrainingSample, ...]:
    if not samples:
        return samples
    source_offsets = {
        field: index for index, field in enumerate(source_observation_fields)
    }
    missing = [field for field in selected_observation_fields if field not in source_offsets]
    if missing:
        raise ValueError(f"Unknown observation fields for ablation: {missing}")
    source_field_count = len(source_observation_fields)
    selected_offsets = tuple(source_offsets[field] for field in selected_observation_fields)
    ablated_samples: list[ObservationTrainingSample] = []
    for sample in samples:
        if len(sample.feature_vector) % source_field_count != 0:
            raise ValueError("Feature vector length is inconsistent with observation fields.")
        observation_count = len(sample.feature_vector) // source_field_count
        feature_vector: list[float] = []
        feature_row: dict[str, float | str] = {
            "pairing_item_id": sample.pairing_item_id,
            "case_id": sample.case_id,
            "scenario": sample.scenario,
        }
        for observation_index in range(observation_count):
            for field, source_offset in zip(
                selected_observation_fields,
                selected_offsets,
                strict=True,
            ):
                source_index = observation_index * source_field_count + source_offset
                value = float(sample.feature_vector[source_index])
                feature_vector.append(value)
                feature_row[f"obs_{observation_index:04d}_{field}"] = value
        ablated_samples.append(
            replace(
                sample,
                feature_row=feature_row,
                feature_vector=tuple(feature_vector),
            )
        )
    return tuple(ablated_samples)


def _shuffle_travel_time_within_splits(
    samples: tuple[ObservationTrainingSample, ...],
    *,
    observation_fields: tuple[str, ...],
    split_by_case_id: dict[str, str],
    random_seed: int,
) -> tuple[ObservationTrainingSample, ...]:
    if "travel_time_s" not in observation_fields:
        raise ValueError("shuffled_travel_time requires travel_time_s in the feature layout.")
    field_count = len(observation_fields)
    travel_time_offset = observation_fields.index("travel_time_s")
    rng = random.Random(random_seed)
    split_values: dict[str, list[float]] = {}
    for sample in samples:
        split = split_by_case_id[sample.case_id]
        values = split_values.setdefault(split, [])
        observation_count = len(sample.feature_vector) // field_count
        for observation_index in range(observation_count):
            values.append(
                float(sample.feature_vector[observation_index * field_count + travel_time_offset])
            )
    for values in split_values.values():
        rng.shuffle(values)
    split_positions = {split: 0 for split in split_values}
    shuffled_samples: list[ObservationTrainingSample] = []
    for sample in samples:
        split = split_by_case_id[sample.case_id]
        feature_vector = list(sample.feature_vector)
        feature_row = dict(sample.feature_row)
        observation_count = len(feature_vector) // field_count
        for observation_index in range(observation_count):
            position = split_positions[split]
            value = split_values[split][position]
            split_positions[split] = position + 1
            feature_index = observation_index * field_count + travel_time_offset
            feature_vector[feature_index] = value
            feature_row[f"obs_{observation_index:04d}_travel_time_s"] = value
        shuffled_samples.append(
            replace(sample, feature_row=feature_row, feature_vector=tuple(feature_vector))
        )
    return tuple(shuffled_samples)


def _train_phase7_variant(
    *,
    settings: BenchmarkSettings,
    context: ObservationTrainingContext,
    variant: dict[str, Any],
    output_dir: Path,
    split_strategy: str = "family_stratified",
    experiment_id: str = "phase7_observation_signal_ablation_v1",
) -> ObservationBaselineOutputs:
    repo_root = get_repo_root()
    samples = (
        _phase7_family_diagnostic_samples(context.samples)
        if variant["model_name"] == "family_mean_target_diagnostic"
        else context.samples
    )
    feature_matrix_csv = output_dir / "feature_matrix.csv"
    metrics_json = output_dir / "metrics.json"
    model_details_json = output_dir / "model_details.json"
    predictions_dir = output_dir / "predictions"
    _write_feature_matrix_csv(samples, context.split_by_case_id, feature_matrix_csv)
    train_samples = _samples_for_split(samples, context.split_by_case_id, "train")
    model = _fit_phase7_variant_model(settings, variant, train_samples)
    split_metrics = {
        split_name: _evaluate_split(
            split_name,
            _samples_for_split(samples, context.split_by_case_id, split_name),
            model,
            predictions_dir / split_name,
            repo_root,
        )
        for split_name in ("train", "validation", "test")
    }
    metrics_payload = {
        "artifact_type": "phase7_observation_signal_ablation_variant_metrics",
        "experiment_id": f"{experiment_id}_{variant['variant_id']}",
        "variant_id": variant["variant_id"],
        "model_name": variant["model_name"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pairing_manifest_path": _relative_to_repo_or_absolute(context.manifest_path, repo_root),
        "split_strategy": split_strategy,
        "path_length_source": variant.get(
            "path_length_source",
            "not_included",
        ),
        "observation_feature_fields": list(variant["observation_fields"]),
        "observation_count_per_case": context.feature_matrix.observation_count,
        "feature_count_per_case": len(samples[0].feature_vector),
        "sample_count": len(samples),
        "split_case_counts": _split_case_counts(context.split_by_case_id, samples),
        "split_unique_target_counts": _split_unique_target_counts(
            context.split_by_case_id,
            samples,
        ),
        "split_metrics": split_metrics,
        "description": variant["description"],
        "diagnostic_only": bool(variant.get("diagnostic_only", False)),
    }
    _save_json(metrics_payload, metrics_json)
    _save_json(
        {
            "artifact_type": "phase7_observation_signal_ablation_model_details",
            "experiment_id": metrics_payload["experiment_id"],
            "variant_id": variant["variant_id"],
            "model_name": variant["model_name"],
            "observation_feature_fields": list(variant["observation_fields"]),
            "path_length_source": variant.get(
                "path_length_source",
                "not_included",
            ),
            "split_case_counts": _split_case_counts(context.split_by_case_id, samples),
            "split_unique_target_counts": _split_unique_target_counts(
                context.split_by_case_id,
                samples,
            ),
            "split_strategy": split_strategy,
            "diagnostic_only": bool(variant.get("diagnostic_only", False)),
        },
        model_details_json,
    )
    return ObservationBaselineOutputs(
        feature_matrix_csv=feature_matrix_csv,
        metrics_json=metrics_json,
        experiment_note=Path(),
        predictions_dir=predictions_dir,
        model_details_json=model_details_json,
    )


def _fit_phase7_variant_model(
    settings: BenchmarkSettings,
    variant: dict[str, Any],
    train_samples: list[ObservationTrainingSample],
) -> ObservationBaselineModel:
    model_name = str(variant["model_name"])
    if model_name == "mean_target_regressor":
        return _MeanTargetObservationAdapter(
            MeanTargetRegressor.fit([sample.target_vector for sample in train_samples])
        )
    if model_name == "family_mean_target_diagnostic":
        return _FamilyMeanTargetDiagnostic.fit(train_samples)
    return _fit_suite_model(settings, model_name, train_samples)


@dataclass(frozen=True)
class _FamilyMeanTargetDiagnostic:
    mean_by_family_index: dict[int, tuple[float, ...]]
    fallback: tuple[float, ...]

    @classmethod
    def fit(cls, train_samples: list[ObservationTrainingSample]) -> "_FamilyMeanTargetDiagnostic":
        grouped: dict[int, list[tuple[float, ...]]] = {}
        for sample in train_samples:
            if len(sample.feature_vector) != 1:
                raise ValueError("Family-mean diagnostic requires one encoded family feature.")
            grouped.setdefault(int(sample.feature_vector[0]), []).append(sample.target_vector)
        fallback = MeanTargetRegressor.fit([sample.target_vector for sample in train_samples]).predict()
        return cls(
            mean_by_family_index={
                family_index: MeanTargetRegressor.fit(vectors).predict()
                for family_index, vectors in grouped.items()
            },
            fallback=fallback,
        )

    def predict(self, feature_vector: tuple[float, ...]) -> tuple[float, ...]:
        if len(feature_vector) != 1:
            return self.fallback
        return self.mean_by_family_index.get(int(feature_vector[0]), self.fallback)


def _evaluate_phase7_repeated_splits(
    *,
    settings: BenchmarkSettings,
    base_context: ObservationTrainingContext,
    variant: dict[str, Any],
    split_seeds: tuple[int, ...],
    random_seed: int,
    split_strategy: str = "family_stratified",
) -> dict[str, Any]:
    split_records: list[dict[str, Any]] = []
    for split_seed in split_seeds:
        split_by_case_id = _split_samples_by_strategy(
            base_context.samples,
            train_fraction=settings.machine_learning.train_fraction,
            validation_fraction=settings.machine_learning.validation_fraction,
            random_seed=split_seed,
            split_strategy=split_strategy,
        )
        context = _phase7_variant_context(
            base_context,
            variant_id=str(variant["variant_id"]),
            observation_fields=tuple(str(field) for field in variant["observation_fields"]),
            random_seed=random_seed + split_seed,
            split_by_case_id=split_by_case_id,
        )
        samples = _phase7_family_diagnostic_samples(context.samples) if variant["model_name"] == "family_mean_target_diagnostic" else context.samples
        model = _fit_phase7_variant_model(
            settings,
            variant,
            _samples_for_split(samples, split_by_case_id, "train"),
        )
        split_records.append(
            {
                "split_seed": split_seed,
                "train": _evaluate_split_metrics_only(
                    _samples_for_split(samples, split_by_case_id, "train"),
                    model,
                ),
                "validation": _evaluate_split_metrics_only(
                    _samples_for_split(samples, split_by_case_id, "validation"),
                    model,
                ),
                "test": _evaluate_split_metrics_only(
                    _samples_for_split(samples, split_by_case_id, "test"),
                    model,
                ),
            }
        )
    return {
        "variant_id": variant["variant_id"],
        "model_name": variant["model_name"],
        "split_strategy": split_strategy,
        "split_records": split_records,
        "aggregate_metrics": _aggregate_split_records(split_records),
    }


def _evaluate_phase7_leave_one_family_out(
    *,
    settings: BenchmarkSettings,
    base_context: ObservationTrainingContext,
    variant: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    families = sorted({sample.scenario for sample in base_context.samples})
    held_out_records: list[dict[str, Any]] = []
    for family in families:
        split_by_case_id = {
            sample.case_id: "test" if sample.scenario == family else "train"
            for sample in base_context.samples
        }
        context = _phase7_variant_context(
            base_context,
            variant_id=str(variant["variant_id"]),
            observation_fields=tuple(str(field) for field in variant["observation_fields"]),
            random_seed=random_seed,
            split_by_case_id=split_by_case_id,
        )
        samples = _phase7_family_diagnostic_samples(context.samples) if variant["model_name"] == "family_mean_target_diagnostic" else context.samples
        train_samples = _samples_for_split(samples, split_by_case_id, "train")
        held_out_samples = _samples_for_split(samples, split_by_case_id, "test")
        model = _fit_phase7_variant_model(settings, variant, train_samples)
        held_out_records.append(
            {
                "held_out_family": family,
                "train_case_count": len(train_samples),
                "held_out": _evaluate_split_metrics_only(held_out_samples, model),
            }
        )
    return {
        "variant_id": variant["variant_id"],
        "model_name": variant["model_name"],
        "families_evaluated": families,
        "held_out_records": held_out_records,
        "aggregate_metrics": {
            "mean_held_out_mae": _mean_record_metric(held_out_records, "held_out", "mean_mae"),
            "std_held_out_mae": _std_record_metric(held_out_records, "held_out", "mean_mae"),
            "mean_held_out_rmse": _mean_record_metric(held_out_records, "held_out", "mean_rmse"),
            "std_held_out_rmse": _std_record_metric(held_out_records, "held_out", "mean_rmse"),
        },
        "worst_held_out_family": max(
            held_out_records,
            key=lambda record: float(record["held_out"]["mean_rmse"]),
        )["held_out_family"],
        "worst_held_out_rmse": max(
            float(record["held_out"]["mean_rmse"]) for record in held_out_records
        ),
    }


def _phase7_family_diagnostic_samples(
    samples: tuple[ObservationTrainingSample, ...],
) -> tuple[ObservationTrainingSample, ...]:
    families = {family: index for index, family in enumerate(sorted({sample.scenario for sample in samples}))}
    return tuple(
        replace(
            sample,
            feature_row={
                "pairing_item_id": sample.pairing_item_id,
                "case_id": sample.case_id,
                "scenario": sample.scenario,
                "family_label_index": float(families[sample.scenario]),
            },
            feature_vector=(float(families[sample.scenario]),),
        )
        for sample in samples
    )


def _phase7_variant_summary_record(
    *,
    variant: dict[str, Any],
    metrics: dict[str, Any],
    cell_metric_rows: list[dict[str, Any]],
    metrics_json: Path,
    feature_matrix_csv: Path,
    repo_root: Path,
) -> dict[str, Any]:
    cell_by_split = {
        split: _aggregate_phase7_cell_rows(
            [row for row in cell_metric_rows if str(row["split"]) == split]
        )
        for split in ("train", "validation", "test")
    }
    return {
        "variant_id": variant["variant_id"],
        "model_name": variant["model_name"],
        "description": variant["description"],
        "diagnostic_only": bool(variant.get("diagnostic_only", False)),
        "observation_feature_fields": list(variant["observation_fields"]),
        "path_length_source": variant.get("path_length_source", "not_included"),
        "feature_count_per_case": metrics["feature_count_per_case"],
        "fixed_split_node_metrics": {
            split: {
                "mean_mae": metrics["split_metrics"][split]["mean_mae"],
                "mean_rmse": metrics["split_metrics"][split]["mean_rmse"],
            }
            for split in ("train", "validation", "test")
        },
        "fixed_split_cell_metrics": cell_by_split,
        "metrics_json": _relative_to_repo_or_absolute(metrics_json, repo_root),
        "feature_matrix_csv": _relative_to_repo_or_absolute(feature_matrix_csv, repo_root),
    }


def _aggregate_phase7_cell_rows(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {"case_count": 0, "mean_mae": None, "mean_rmse": None}
    return {
        "case_count": len(rows),
        "mean_mae": sum(float(row["all_cell_velocity_mae_km_per_s"]) for row in rows) / len(rows),
        "mean_rmse": sum(float(row["all_cell_velocity_rmse_km_per_s"]) for row in rows) / len(rows),
    }


def _phase7_summary_rows(
    variant_records: list[dict[str, Any]],
    repeated_records: list[dict[str, Any]],
    loo_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    repeated_by_variant = {record["variant_id"]: record for record in repeated_records}
    loo_by_variant = {record["variant_id"]: record for record in loo_records}
    rows: list[dict[str, Any]] = []
    for record in variant_records:
        variant_id = record["variant_id"]
        repeated = repeated_by_variant[variant_id]["aggregate_metrics"]
        loo = loo_by_variant[variant_id]
        rows.append(
            {
                "variant_id": variant_id,
                "model_name": record["model_name"],
                "diagnostic_only": record["diagnostic_only"],
                "feature_count_per_case": record["feature_count_per_case"],
                "path_length_source": record.get("path_length_source", "not_included"),
                "fixed_train_node_rmse": record["fixed_split_node_metrics"]["train"]["mean_rmse"],
                "fixed_validation_node_rmse": record["fixed_split_node_metrics"]["validation"]["mean_rmse"],
                "fixed_test_node_rmse": record["fixed_split_node_metrics"]["test"]["mean_rmse"],
                "fixed_train_cell_rmse": record["fixed_split_cell_metrics"]["train"]["mean_rmse"],
                "fixed_validation_cell_rmse": record["fixed_split_cell_metrics"]["validation"]["mean_rmse"],
                "fixed_test_cell_rmse": record["fixed_split_cell_metrics"]["test"]["mean_rmse"],
                "fixed_test_node_mae": record["fixed_split_node_metrics"]["test"]["mean_mae"],
                "fixed_test_cell_mae": record["fixed_split_cell_metrics"]["test"]["mean_mae"],
                "repeated_mean_validation_rmse": repeated["mean_validation_rmse"],
                "repeated_mean_test_rmse": repeated["mean_test_rmse"],
                "leave_one_family_out_mean_rmse": loo["aggregate_metrics"]["mean_held_out_rmse"],
                "leave_one_family_out_worst_family": loo["worst_held_out_family"],
                "leave_one_family_out_worst_rmse": loo["worst_held_out_rmse"],
            }
        )
    return rows


def _phase7_interpretation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {row["variant_id"]: row for row in rows}
    full = by_variant["full_input"]
    no_time = by_variant["no_travel_time"]
    shuffled = by_variant["shuffled_travel_time"]
    full_test = float(full["fixed_test_cell_rmse"])
    no_time_test = float(no_time["fixed_test_cell_rmse"])
    shuffled_test = float(shuffled["fixed_test_cell_rmse"])
    improves_over_no_time = full_test < no_time_test
    improves_over_shuffled = full_test < shuffled_test
    if improves_over_no_time and improves_over_shuffled:
        statement = (
            "Travel-time observations contributed measurable reconstruction signal beyond "
            "acquisition geometry and target-distribution priors."
        )
    else:
        statement = (
            "The selected model appears strongly influenced by synthetic corpus priors and "
            "acquisition geometry."
        )
    return {
        "primary_metric": "fixed_test_cell_rmse",
        "full_input_fixed_test_cell_rmse": full_test,
        "no_travel_time_fixed_test_cell_rmse": no_time_test,
        "shuffled_travel_time_fixed_test_cell_rmse": shuffled_test,
        "full_beats_no_travel_time": improves_over_no_time,
        "full_beats_shuffled_travel_time": improves_over_shuffled,
        "statement": statement,
    }


def _model_slug(model_name: str) -> str:
    return model_name.replace("_observation_regressor", "").replace("_", "-")


def _samples_for_split(
    samples: tuple[ObservationTrainingSample, ...],
    split_by_case_id: dict[str, str],
    split_name: str,
) -> list[ObservationTrainingSample]:
    return [sample for sample in samples if split_by_case_id[sample.case_id] == split_name]


def _with_travel_time_noise(
    samples: tuple[ObservationTrainingSample, ...],
    *,
    observation_fields: tuple[str, ...],
    noise_std_s: float,
    random_seed: int,
) -> tuple[ObservationTrainingSample, ...]:
    if noise_std_s == 0.0:
        return samples
    rng = random.Random(random_seed)
    noisy_samples: list[ObservationTrainingSample] = []
    travel_time_offsets = [
        index for index, field in enumerate(observation_fields) if field == "travel_time_s"
    ]
    field_count = len(observation_fields)
    for sample in samples:
        feature_vector = list(sample.feature_vector)
        feature_row = dict(sample.feature_row)
        observation_count = len(feature_vector) // field_count
        for observation_index in range(observation_count):
            for field_offset in travel_time_offsets:
                feature_index = observation_index * field_count + field_offset
                noisy_value = max(
                    1.0e-9, feature_vector[feature_index] + rng.gauss(0.0, noise_std_s)
                )
                feature_vector[feature_index] = noisy_value
                feature_row[f"obs_{observation_index:04d}_travel_time_s"] = noisy_value
        noisy_samples.append(
            replace(
                sample,
                feature_row=feature_row,
                feature_vector=tuple(feature_vector),
            )
        )
    return tuple(noisy_samples)


def _prepare_observation_training_context(
    *,
    settings: BenchmarkSettings,
    pairing_manifest_path: Path,
    observation_fields: tuple[str, ...],
) -> ObservationTrainingContext:
    repo_root = get_repo_root()
    resolved_manifest_path = _resolve_repo_path(pairing_manifest_path, repo_root)
    resolved_manifest_path = _ensure_observation_pairing_manifest(settings, resolved_manifest_path)
    pairing_manifest = _read_json(resolved_manifest_path)
    feature_matrix = assemble_observation_case_feature_matrix(
        resolved_manifest_path,
        repo_root,
        observation_fields=observation_fields,
    )
    samples = _attach_targets(feature_matrix, pairing_manifest, repo_root)
    split_by_case_id = _family_stratified_split(
        samples,
        train_fraction=settings.machine_learning.train_fraction,
        validation_fraction=settings.machine_learning.validation_fraction,
        random_seed=settings.random_seeds.machine_learning,
    )
    return ObservationTrainingContext(
        manifest_path=resolved_manifest_path,
        pairing_manifest=pairing_manifest,
        feature_matrix=feature_matrix,
        samples=samples,
        split_by_case_id=split_by_case_id,
    )


def _attach_targets(
    feature_matrix: Any,
    pairing_manifest: dict[str, Any],
    repo_root: Path,
) -> tuple[ObservationTrainingSample, ...]:
    items_by_pairing_id = {
        str(item["pairing_item_id"]): item for item in _dict_list(pairing_manifest["items"])
    }
    samples: list[ObservationTrainingSample] = []
    for sample in feature_matrix.samples:
        manifest_item = items_by_pairing_id[sample.pairing_item_id]
        target_grid_path = _resolve_repo_path(
            Path(str(manifest_item["target_velocity_grid_path"])), repo_root
        )
        target_grid = _read_json(target_grid_path)
        samples.append(
            ObservationTrainingSample(
                pairing_item_id=sample.pairing_item_id,
                case_id=sample.case_id,
                scenario=sample.scenario,
                feature_row=sample.feature_row,
                feature_vector=sample.feature_vector,
                target_vector=tuple(float(value) for value in target_grid["p_velocity_km_per_s"]),
                target_grid_path=target_grid_path,
                target_spec_path=_resolve_repo_path(
                    Path(str(manifest_item["target_spec_path"])), repo_root
                ),
            )
        )
    return tuple(samples)


def _train_observation_baseline(
    *,
    settings: BenchmarkSettings,
    run_config: ObservationModelRunConfig,
    context: ObservationTrainingContext | None = None,
) -> ObservationBaselineOutputs:
    repo_root = get_repo_root()
    training_context = context or _prepare_observation_training_context(
        settings=settings,
        pairing_manifest_path=run_config.pairing_manifest_path,
        observation_fields=run_config.observation_fields,
    )
    resolved_output_dir = _resolve_repo_path(run_config.output_dir, repo_root)
    feature_matrix_csv = resolved_output_dir / "feature_matrix.csv"
    metrics_json = resolved_output_dir / "metrics.json"
    model_details_json = resolved_output_dir / "model_details.json"
    predictions_dir = resolved_output_dir / "predictions"
    experiment_note = (
        repo_root / settings.outputs.experiment_notes_dir / f"{run_config.experiment_id}.md"
    )
    _write_feature_matrix_csv(
        training_context.samples, training_context.split_by_case_id, feature_matrix_csv
    )
    train_samples = [
        sample
        for sample in training_context.samples
        if training_context.split_by_case_id[sample.case_id] == "train"
    ]
    validation_samples = [
        sample
        for sample in training_context.samples
        if training_context.split_by_case_id[sample.case_id] == "validation"
    ]
    test_samples = [
        sample
        for sample in training_context.samples
        if training_context.split_by_case_id[sample.case_id] == "test"
    ]
    model = run_config.model_builder(train_samples)
    split_metrics = {
        split_name: _evaluate_split(
            split_name,
            split_samples,
            model,
            predictions_dir / split_name,
            repo_root,
        )
        for split_name, split_samples in (
            ("train", train_samples),
            ("validation", validation_samples),
            ("test", test_samples),
        )
    }
    metrics_payload = {
        "artifact_type": "ml_observation_baseline_metrics",
        "experiment_id": run_config.experiment_id,
        "model_name": run_config.model_name,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pairing_manifest_path": _relative_to_repo_or_absolute(
            training_context.manifest_path, repo_root
        ),
        "split_strategy": run_config.split_strategy,
        "observation_feature_fields": list(run_config.observation_fields),
        "observation_count_per_case": training_context.feature_matrix.observation_count,
        "feature_count_per_case": len(training_context.feature_matrix.samples[0].feature_vector),
        "sample_count": len(training_context.samples),
        "split_case_counts": _split_case_counts(
            training_context.split_by_case_id,
            training_context.samples,
        ),
        "split_unique_target_counts": _split_unique_target_counts(
            training_context.split_by_case_id,
            training_context.samples,
        ),
        "split_metrics": split_metrics,
        "notes": run_config.notes,
        "assumptions": run_config.assumptions,
    }
    _save_json(metrics_payload, metrics_json)
    model_details = run_config.model_details_builder(
        model,
        training_context.split_by_case_id,
        training_context.samples,
    )
    model_details["experiment_id"] = run_config.experiment_id
    model_details["split_strategy"] = run_config.split_strategy
    model_details["split_unique_target_counts"] = _split_unique_target_counts(
        training_context.split_by_case_id,
        training_context.samples,
    )
    _save_json(model_details, model_details_json)
    _write_experiment_note(
        experiment_note,
        note_title=run_config.note_title,
        objective=run_config.objective,
        metrics_payload=metrics_payload,
        feature_matrix_csv=feature_matrix_csv,
        metrics_json=metrics_json,
        model_details_json=model_details_json,
        predictions_dir=predictions_dir,
        repo_root=repo_root,
        limitations=run_config.limitations,
        next_step=run_config.next_step,
    )
    return ObservationBaselineOutputs(
        feature_matrix_csv=feature_matrix_csv,
        metrics_json=metrics_json,
        experiment_note=experiment_note,
        predictions_dir=predictions_dir,
        model_details_json=model_details_json,
    )


def _ensure_observation_pairing_manifest(
    settings: BenchmarkSettings,
    manifest_path: Path,
) -> Path:
    if manifest_path.is_file():
        return manifest_path
    outputs = prepare_observation_ml_supervised_pairings(settings)
    return outputs.manifest_path


def _family_stratified_split(
    samples: tuple[ObservationTrainingSample, ...],
    *,
    train_fraction: float,
    validation_fraction: float,
    random_seed: int,
) -> dict[str, str]:
    rng = random.Random(random_seed)
    grouped: dict[str, list[ObservationTrainingSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.scenario, []).append(sample)
    split_by_case_id: dict[str, str] = {}
    for scenario, scenario_samples in grouped.items():
        shuffled = list(sorted(scenario_samples, key=lambda item: item.case_id))
        rng.shuffle(shuffled)
        train_count = max(1, int(len(shuffled) * train_fraction))
        validation_count = 1 if len(shuffled) >= 3 and validation_fraction > 0.0 else 0
        if train_count + validation_count >= len(shuffled):
            train_count = max(1, len(shuffled) - validation_count - 1)
        for index, sample in enumerate(shuffled):
            if index < train_count:
                split_by_case_id[sample.case_id] = "train"
            elif index < train_count + validation_count:
                split_by_case_id[sample.case_id] = "validation"
            else:
                split_by_case_id[sample.case_id] = "test"
        if (
            "validation" not in {split_by_case_id[sample.case_id] for sample in shuffled}
            and len(shuffled) >= 3
        ):
            last_train = next(
                sample
                for sample in reversed(shuffled)
                if split_by_case_id[sample.case_id] == "train"
            )
            split_by_case_id[last_train.case_id] = "validation"
        if "test" not in {split_by_case_id[sample.case_id] for sample in shuffled}:
            last_validation = next(
                sample
                for sample in reversed(shuffled)
                if split_by_case_id[sample.case_id] in {"train", "validation"}
            )
            split_by_case_id[last_validation.case_id] = "test"
    return split_by_case_id


def _split_samples_by_strategy(
    samples: tuple[ObservationTrainingSample, ...],
    *,
    train_fraction: float,
    validation_fraction: float,
    random_seed: int,
    split_strategy: str,
) -> dict[str, str]:
    if split_strategy == "family_stratified":
        return _family_stratified_split(
            samples,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            random_seed=random_seed,
        )
    if split_strategy == "target_grouped_family_stratified":
        return target_grouped_family_split(
            samples,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            random_seed=random_seed,
        )
    raise ValueError(f"Unsupported observation split strategy: {split_strategy}")


def _evaluate_split(
    split_name: str,
    samples: list[ObservationTrainingSample],
    model: ObservationBaselineModel,
    predictions_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    case_metrics: list[dict[str, Any]] = []
    for sample in samples:
        prediction = model.predict(sample.feature_vector)
        prediction_path = _save_prediction(
            sample,
            prediction,
            predictions_dir / f"{sample.case_id}.prediction.json",
            repo_root,
        )
        case_metrics.append(
            {
                "case_id": sample.case_id,
                "scenario": sample.scenario,
                "pairing_item_id": sample.pairing_item_id,
                "mae": mean_absolute_error(sample.target_vector, prediction),
                "rmse": root_mean_squared_error(sample.target_vector, prediction),
                "prediction_path": _relative_to_repo_or_absolute(prediction_path, repo_root),
                "target_grid_path": _relative_to_repo_or_absolute(
                    sample.target_grid_path, repo_root
                ),
            }
        )
    return {
        "split_name": split_name,
        "case_count": len(samples),
        "mean_mae": (sum(item["mae"] for item in case_metrics) / len(case_metrics))
        if case_metrics
        else None,
        "mean_rmse": (sum(item["rmse"] for item in case_metrics) / len(case_metrics))
        if case_metrics
        else None,
        "per_family_metrics": _per_family_metrics(case_metrics),
        "case_metrics": case_metrics,
    }


def _evaluate_split_metrics_only(
    samples: list[ObservationTrainingSample],
    model: ObservationBaselineModel,
) -> dict[str, Any]:
    if not samples:
        return {
            "case_count": 0,
            "mean_mae": None,
            "mean_rmse": None,
            "per_family_metrics": {},
        }
    case_metrics = []
    for sample in samples:
        prediction = model.predict(sample.feature_vector)
        case_metrics.append(
            {
                "case_id": sample.case_id,
                "scenario": sample.scenario,
                "mae": mean_absolute_error(sample.target_vector, prediction),
                "rmse": root_mean_squared_error(sample.target_vector, prediction),
            }
        )
    return {
        "case_count": len(samples),
        "mean_mae": sum(item["mae"] for item in case_metrics) / len(case_metrics),
        "mean_rmse": sum(item["rmse"] for item in case_metrics) / len(case_metrics),
        "per_family_metrics": _per_family_metrics(case_metrics),
    }


def _evaluate_learning_curve_split(
    samples: list[ObservationTrainingSample],
    model: ObservationBaselineModel,
    target_cell_cache: dict[str, tuple[float, ...]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_metrics: list[dict[str, Any]] = []
    for sample in samples:
        prediction = model.predict(sample.feature_vector)
        target_cells = _learning_curve_target_cells(sample, target_cell_cache)
        grid_payload = _read_json(sample.target_grid_path)
        nx = len(grid_payload["x_coordinates_km"])
        ny = len(grid_payload["y_coordinates_km"])
        nz = len(grid_payload["z_coordinates_km"])
        predicted_cells = node_centered_values_to_cell_centered(prediction, nx, ny, nz)
        case_metrics.append(
            {
                "case_id": sample.case_id,
                "family": sample.scenario,
                "node_rmse": root_mean_squared_error(sample.target_vector, prediction),
                "node_mae": mean_absolute_error(sample.target_vector, prediction),
                "cell_rmse": root_mean_squared_error(target_cells, predicted_cells),
                "cell_mae": mean_absolute_error(target_cells, predicted_cells),
            }
        )
    metric_names = ("node_rmse", "node_mae", "cell_rmse", "cell_mae")
    payload: dict[str, Any] = {"case_count": len(case_metrics)}
    for metric_name in metric_names:
        values = [float(item[metric_name]) for item in case_metrics]
        payload[f"mean_{metric_name}"] = sum(values) / len(values) if values else None
        payload[f"std_{metric_name}"] = _population_std_for_values(values)
    payload["family_metrics"] = _learning_curve_family_metrics(case_metrics)
    return payload, case_metrics


def _learning_curve_target_cells(
    sample: ObservationTrainingSample,
    target_cell_cache: dict[str, tuple[float, ...]],
) -> tuple[float, ...]:
    cache_key = sample.target_grid_path.resolve().as_posix()
    if cache_key not in target_cell_cache:
        grid_payload = _read_json(sample.target_grid_path)
        nx = len(grid_payload["x_coordinates_km"])
        ny = len(grid_payload["y_coordinates_km"])
        nz = len(grid_payload["z_coordinates_km"])
        target_cell_cache[cache_key] = node_centered_values_to_cell_centered(
            tuple(float(value) for value in grid_payload["p_velocity_km_per_s"]),
            nx,
            ny,
            nz,
        )
    return target_cell_cache[cache_key]


def _learning_curve_family_metrics(
    case_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for metric in case_metrics:
        grouped.setdefault(str(metric["family"]), []).append(metric)
    return {
        family: {
            "case_count": len(items),
            "node_rmse": sum(float(item["node_rmse"]) for item in items) / len(items),
            "node_mae": sum(float(item["node_mae"]) for item in items) / len(items),
            "cell_rmse": sum(float(item["cell_rmse"]) for item in items) / len(items),
            "cell_mae": sum(float(item["cell_mae"]) for item in items) / len(items),
        }
        for family, items in sorted(grouped.items())
    }


def _learning_curve_repetition_rows(
    repetition_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in repetition_records:
        row: dict[str, Any] = {
            "training_size": int(record["training_size"]),
            "repetition_seed": int(record["repetition_seed"]),
        }
        for split_name in ("train", "validation", "test"):
            payload = record[split_name]
            row[f"{split_name}_case_count"] = payload["case_count"]
            for metric_name in ("node_rmse", "node_mae", "cell_rmse", "cell_mae"):
                row[f"{split_name}_mean_{metric_name}"] = payload[
                    f"mean_{metric_name}"
                ]
                row[f"{split_name}_std_{metric_name}"] = payload[f"std_{metric_name}"]
        rows.append(row)
    return rows


def _learning_curve_interpretation(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    eligible_training_case_count: int = 70,
    eligible_training_target_count: int | None = None,
) -> dict[str, Any]:
    ordered = sorted(summary_rows, key=lambda row: int(row["training_size"]))
    if not ordered:
        raise ValueError("Learning-curve interpretation requires summary rows.")
    best = min(ordered, key=lambda row: float(row["mean_test_cell_rmse"]))
    first = ordered[0]
    last = ordered[-1]
    first_to_last_improvement = float(first["mean_test_cell_rmse"]) - float(
        last["mean_test_cell_rmse"]
    )
    last_step_improvement = None
    if len(ordered) >= 2:
        previous = ordered[-2]
        last_step_improvement = float(previous["mean_test_cell_rmse"]) - float(
            last["mean_test_cell_rmse"]
        )
    if best["training_size"] != last["training_size"]:
        trend_statement = (
            "The best observed fixed-test cell-centered RMSE occurs before the largest "
            "evaluated training size, so this curve does not show a simple monotonic gain."
        )
    elif first_to_last_improvement > 0.0:
        trend_statement = (
            "Fixed-test cell-centered RMSE improves from the smallest to the largest "
            "evaluated training size; saturation before the available training-pool limit "
            "is not established."
        )
    else:
        trend_statement = (
            "Fixed-test cell-centered RMSE does not improve from the smallest to the largest "
            "evaluated training size, but this alone does not establish saturation at 100 cases."
        )
    target_caveat = (
        f" The eligible training pool contains {eligible_training_target_count} unique target "
        "groups."
        if eligible_training_target_count is not None
        else ""
    )
    return {
        "primary_metric": "mean_test_cell_rmse",
        "best_observed_training_size": int(best["training_size"]),
        "best_observed_mean_test_cell_rmse": float(best["mean_test_cell_rmse"]),
        "smallest_to_largest_cell_rmse_improvement": first_to_last_improvement,
        "largest_step_cell_rmse_improvement": last_step_improvement,
        "largest_training_size_evaluated": int(last["training_size"]),
        "saturation_at_100_supported": False,
        "statement": trend_statement,
        "caveat": (
            f"The fixed holdout leaves only {eligible_training_case_count} eligible training "
            f"cases; larger training sizes would require changing the holdout protocol."
            + target_caveat
        ),
    }


def _write_phase7_learning_curve_note(
    path: Path,
    *,
    payload: dict[str, Any],
    summary_json: Path,
    summary_csv: Path,
    repetition_metrics_csv: Path,
    family_wise_csv: Path,
    repo_root: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment: Phase 7 ML Learning Curve v1",
        "",
        f"Experiment ID: `{payload['experiment_id']}`",
        "",
        "## Objective",
        "",
        (
            "Test whether the selected `pca_linear_observation_regressor` is already "
            "saturated at the current synthetic corpus scale by measuring fixed-holdout "
            "performance at increasing family-balanced training sizes."
        ),
        "",
        "## Protocol",
        "",
        f"- Input corpus: `{payload['input_corpus_id']}`",
        f"- Fixed split seed: `{payload['fixed_split_seed']}`",
        f"- Fixed split counts: `{payload['fixed_split_case_counts']}`",
        f"- Training sizes evaluated: `{', '.join(str(value) for value in payload['training_sizes_evaluated'])}`",
        f"- Repetition seeds: `{', '.join(str(value) for value in payload['repetition_seeds'])}`",
        f"- Subset protocol: {payload['training_subset_protocol']}.",
        "- Validation and test cases remain fixed across all training sizes and repetitions.",
        "",
        "## Fixed-Test Cell-Centered RMSE",
        "",
        "| Training cases | Mean RMSE (km/s) | Std. dev. (km/s) | Repetitions |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            f"| {row['training_size']} | {float(row['mean_test_cell_rmse']):.6f} | "
            f"{float(row['std_test_cell_rmse']):.6f} | {row['repetition_count']} |"
        )
    interpretation = payload["interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation["statement"],
            "",
            interpretation["caveat"],
            "",
            "This remains a synthetic-only learning-curve diagnostic. It does not validate "
            "real-data performance, establish that 100 cases are sufficient, or claim "
            "universal ML superiority.",
            "",
            "## Outputs",
            "",
            f"- Summary JSON: `{_relative_to_repo_or_absolute(summary_json, repo_root)}`",
            f"- Summary CSV: `{_relative_to_repo_or_absolute(summary_csv, repo_root)}`",
            f"- Repetition metrics: `{_relative_to_repo_or_absolute(repetition_metrics_csv, repo_root)}`",
            f"- Family-wise test metrics: `{_relative_to_repo_or_absolute(family_wise_csv, repo_root)}`",
            "",
            "## Limitations and Next Step",
            "",
            "The fixed holdout constrains the leakage-free training pool to the reported case "
            "and unique-target counts, and target-group atomicity can limit the attainable "
            "training sizes. If the curve is still improving at the largest available size, "
            "expand the synthetic corpus and repeat the experiment with a newly pre-registered holdout.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _per_family_metrics(case_metrics: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for metric in case_metrics:
        grouped.setdefault(str(metric["scenario"]), []).append(metric)
    return {
        family: {
            "case_count": len(items),
            "mean_mae": sum(float(item["mae"]) for item in items) / len(items),
            "mean_rmse": sum(float(item["rmse"]) for item in items) / len(items),
        }
        for family, items in sorted(grouped.items())
    }


def _summarize_tuning_candidate(
    *,
    pca_component_count: int,
    ridge_alpha: float,
    split_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "pca_component_count": pca_component_count,
        "ridge_alpha": ridge_alpha,
        "aggregate_metrics": {
            "mean_train_rmse": _mean_metric(split_records, "train", "mean_rmse"),
            "mean_validation_rmse": _mean_metric(split_records, "validation", "mean_rmse"),
            "mean_test_rmse": _mean_metric(split_records, "test", "mean_rmse"),
            "mean_train_mae": _mean_metric(split_records, "train", "mean_mae"),
            "mean_validation_mae": _mean_metric(split_records, "validation", "mean_mae"),
            "mean_test_mae": _mean_metric(split_records, "test", "mean_mae"),
        },
        "split_records": split_records,
    }


def _mean_metric(
    split_records: list[dict[str, Any]],
    split_name: str,
    metric_name: str,
) -> float:
    values = [float(record[split_name][metric_name]) for record in split_records]
    return sum(values) / len(values)


def _aggregate_split_records(split_records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_train_mae": _mean_metric(split_records, "train", "mean_mae"),
        "std_train_mae": _std_split_metric(split_records, "train", "mean_mae"),
        "mean_train_rmse": _mean_metric(split_records, "train", "mean_rmse"),
        "std_train_rmse": _std_split_metric(split_records, "train", "mean_rmse"),
        "mean_validation_mae": _mean_metric(split_records, "validation", "mean_mae"),
        "std_validation_mae": _std_split_metric(split_records, "validation", "mean_mae"),
        "mean_validation_rmse": _mean_metric(split_records, "validation", "mean_rmse"),
        "std_validation_rmse": _std_split_metric(split_records, "validation", "mean_rmse"),
        "mean_test_mae": _mean_metric(split_records, "test", "mean_mae"),
        "std_test_mae": _std_split_metric(split_records, "test", "mean_mae"),
        "mean_test_rmse": _mean_metric(split_records, "test", "mean_rmse"),
        "std_test_rmse": _std_split_metric(split_records, "test", "mean_rmse"),
    }


def _std_split_metric(
    split_records: list[dict[str, Any]],
    split_name: str,
    metric_name: str,
) -> float:
    values = [float(record[split_name][metric_name]) for record in split_records]
    return _population_std_for_values(values)


def _mean_record_metric(
    records: list[dict[str, Any]],
    metric_group: str,
    metric_name: str,
) -> float:
    values = [float(record[metric_group][metric_name]) for record in records]
    return sum(values) / len(values)


def _std_record_metric(
    records: list[dict[str, Any]],
    metric_group: str,
    metric_name: str,
) -> float:
    values = [float(record[metric_group][metric_name]) for record in records]
    return _population_std_for_values(values)


def _population_std_for_values(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def _split_case_counts(
    split_by_case_id: dict[str, str],
    samples: tuple[ObservationTrainingSample, ...],
) -> dict[str, int]:
    return {
        split_name: len(
            [sample for sample in samples if split_by_case_id[sample.case_id] == split_name]
        )
        for split_name in ("train", "validation", "test")
    }


def _split_unique_target_counts(
    split_by_case_id: dict[str, str],
    samples: tuple[ObservationTrainingSample, ...],
) -> dict[str, int]:
    target_hashes_by_split: dict[str, set[str]] = {
        split_name: set() for split_name in ("train", "validation", "test")
    }
    for sample in samples:
        target_hashes_by_split[split_by_case_id[sample.case_id]].add(
            target_vector_sha256(sample.target_vector)
        )
    return {
        split_name: len(target_hashes_by_split[split_name])
        for split_name in ("train", "validation", "test")
    }


def _repeated_split_protocol_name(split_strategy: str) -> str:
    if split_strategy == "target_grouped_family_stratified":
        return "repeated_seeded_target_grouped_family_stratified"
    return "repeated_seeded_family_stratified"


def _effective_split_protocols(
    configured_protocols: Sequence[str],
    split_strategy: str,
) -> list[str]:
    if split_strategy == "target_grouped_family_stratified":
        return [
            "fixed_target_grouped_family_stratified",
            "repeated_seeded_target_grouped_family_stratified",
            "leave_one_family_out",
        ]
    return list(configured_protocols)


def _save_prediction(
    sample: ObservationTrainingSample,
    prediction: tuple[float, ...],
    path: Path,
    repo_root: Path,
) -> Path:
    payload = {
        "artifact_type": "ml_observation_full_grid_prediction",
        "case_id": sample.case_id,
        "scenario": sample.scenario,
        "pairing_item_id": sample.pairing_item_id,
        "target_grid_path": _relative_to_repo_or_absolute(sample.target_grid_path, repo_root),
        "target_spec_path": _relative_to_repo_or_absolute(sample.target_spec_path, repo_root),
        "predicted_target_vector_length": len(prediction),
        "predicted_p_velocity_km_per_s": list(prediction),
    }
    return _save_json(payload, path)


def _write_target_group_split_assignments(
    samples: tuple[ObservationTrainingSample, ...],
    split_by_case_id: dict[str, str],
    path: Path,
) -> Path:
    """Persist the target identity and split for every case in a suite run."""
    fieldnames = (
        "case_id",
        "pairing_item_id",
        "scenario",
        "split",
        "target_hash",
        "target_vector_length",
        "target_grid_path",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "case_id": sample.case_id,
                    "pairing_item_id": sample.pairing_item_id,
                    "scenario": sample.scenario,
                    "split": split_by_case_id[sample.case_id],
                    "target_hash": target_vector_sha256(sample.target_vector),
                    "target_vector_length": len(sample.target_vector),
                    "target_grid_path": str(sample.target_grid_path),
                }
            )
    return path


def _write_feature_matrix_csv(
    samples: tuple[ObservationTrainingSample, ...],
    split_by_case_id: dict[str, str],
    path: Path,
) -> Path:
    fieldnames = (
        "pairing_item_id",
        "case_id",
        "scenario",
        "split",
        *tuple(
            key
            for key in samples[0].feature_row.keys()
            if key not in {"pairing_item_id", "case_id", "scenario"}
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = dict(sample.feature_row)
            row["split"] = split_by_case_id[sample.case_id]
            writer.writerow(row)
    return path


def _write_phase7_ablation_note(
    path: Path,
    *,
    payload: dict[str, Any],
    summary_json: Path,
    summary_csv: Path,
    cell_centered_metrics_csv: Path,
    leave_one_family_out_summary_json: Path,
    repo_root: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    interpretation = payload["interpretation"]
    rows = payload["summary_rows"]
    by_variant = {row["variant_id"]: row for row in rows}
    lines = [
        "# Experiment: Phase 7 Observation-Signal Ablation v1",
        "",
        f"Experiment ID: `{payload['experiment_id']}`",
        "",
        "## Objective",
        "",
        (
            "Test whether `travel_time_s` contributes measurable inverse-problem information "
            "beyond acquisition geometry, path length, and synthetic target-distribution priors "
            "for the selected PCA-linear observation model."
        ),
        "",
        "## Inputs Used",
        "",
        f"- Input corpus ID: `{payload['input_corpus_id']}`",
        f"- Input manifest: `{payload['input_manifest_path']}`",
        f"- Split strategy: `{payload['split_strategy']}`",
        f"- Fixed split seed: `{payload['fixed_split_seed']}`",
        f"- Repeated split seeds: `{', '.join(str(seed) for seed in payload['repeated_split_seeds'])}`",
        f"- Travel-time shuffle scope: `{payload['travel_time_shuffle_scope']}`",
        f"- Split case counts: `{payload['split_case_counts']}`",
        f"- Split unique-target counts: `{payload['split_unique_target_counts']}`",
        "",
        "## Interpretation",
        "",
        interpretation["statement"],
        "",
        "## Fixed-Test Cell-Centered RMSE",
        "",
    ]
    for variant_id in (
        "full_input",
        "full_input_euclidean_path",
        "no_travel_time",
        "shuffled_travel_time",
        "mean_target",
        "geometry_only",
        "path_length_only",
        "euclidean_path_only",
        "travel_time_only",
        "family_mean_target_diagnostic",
    ):
        row = by_variant[variant_id]
        diagnostic = " diagnostic only" if row["diagnostic_only"] else ""
        lines.append(
            f"- `{variant_id}`{diagnostic}: `{float(row['fixed_test_cell_rmse']):.6f}` km/s"
        )
    lines.extend(
        [
            "",
            "## Repeated and LOO Diagnostics",
            "",
        ]
    )
    for variant_id in (
        "full_input",
        "full_input_euclidean_path",
        "no_travel_time",
        "shuffled_travel_time",
        "mean_target",
        "travel_time_only",
    ):
        row = by_variant[variant_id]
        lines.append(
            f"- `{variant_id}`: repeated validation node RMSE "
            f"`{float(row['repeated_mean_validation_rmse']):.6f}` km/s; worst LOO "
            f"`{row['leave_one_family_out_worst_family']}` RMSE "
            f"`{float(row['leave_one_family_out_worst_rmse']):.6f}` km/s"
        )
    lines.extend(
        [
            "",
            "## Outputs Created",
            "",
            f"- Summary JSON: `{_relative_to_repo_or_absolute(summary_json, repo_root)}`",
            f"- Summary CSV: `{_relative_to_repo_or_absolute(summary_csv, repo_root)}`",
            f"- Cell-centered metrics CSV: `{_relative_to_repo_or_absolute(cell_centered_metrics_csv, repo_root)}`",
            f"- Leave-one-family-out JSON: `{_relative_to_repo_or_absolute(leave_one_family_out_summary_json, repo_root)}`",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in payload["limitations"])
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            (
                "Use this ablation suite as a diagnostic when drafting Phase 7 "
                "Results language; do not treat it as real-data validation or universal ML superiority."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_experiment_note(
    path: Path,
    *,
    note_title: str,
    objective: str,
    metrics_payload: dict[str, Any],
    feature_matrix_csv: Path,
    metrics_json: Path,
    model_details_json: Path,
    predictions_dir: Path,
    repo_root: Path,
    limitations: list[str],
    next_step: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Experiment: {note_title}",
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
        f"- Split strategy: `{metrics_payload['split_strategy']}`",
        f"- Observation feature fields: `{', '.join(metrics_payload['observation_feature_fields'])}`",
        "",
        "## Outputs Created",
        "",
        f"- Feature matrix: `{_relative_to_repo_or_absolute(feature_matrix_csv, repo_root)}`",
        f"- Metrics JSON: `{_relative_to_repo_or_absolute(metrics_json, repo_root)}`",
        f"- Model details JSON: `{_relative_to_repo_or_absolute(model_details_json, repo_root)}`",
        f"- Prediction directory: `{_relative_to_repo_or_absolute(predictions_dir, repo_root)}`",
        "",
        "## Aggregate Metrics",
        "",
    ]
    for split_name in ("train", "validation", "test"):
        split_metrics = metrics_payload["split_metrics"][split_name]
        lines.append(
            f"- {split_name.title()} RMSE: `{split_metrics['mean_rmse']:.6f}` km/s"
            if split_metrics["mean_rmse"] is not None
            else f"- {split_name.title()} RMSE: `n/a`"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in limitations)
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_observation_tuning_note(
    path: Path,
    payload: dict[str, Any],
    summary_json: Path,
    candidate_metrics_json: Path,
    repo_root: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_candidate = payload["best_candidate"]
    lines = [
        "# Experiment: Observation-Driven PCA Ridge Tuning",
        "",
        f"Experiment ID: `{payload['experiment_id']}`",
        "",
        "## Objective",
        "",
        (
            "Tune the current best observation-driven PCA-plus-ridge baseline across repeated "
            "family-stratified splits so the benchmark reference model is selected from stable "
            "validation behavior rather than one seeded split."
        ),
        "",
        "## Inputs Used",
        "",
        f"- Pairing manifest: `{payload['pairing_manifest_path']}`",
        f"- Split strategy: `{payload['split_strategy']}`",
        f"- Split random seeds: `{', '.join(str(value) for value in payload['split_random_seeds'])}`",
        f"- Candidate count: `{payload['candidate_count']}`",
        "",
        "## Best Candidate",
        "",
        f"- PCA component count: `{best_candidate['pca_component_count']}`",
        f"- Ridge alpha: `{best_candidate['ridge_alpha']}`",
        f"- Mean validation RMSE: `{best_candidate['aggregate_metrics']['mean_validation_rmse']:.6f}` km/s",
        f"- Mean test RMSE: `{best_candidate['aggregate_metrics']['mean_test_rmse']:.6f}` km/s",
        "",
        "## Outputs Created",
        "",
        f"- Summary JSON: `{_relative_to_repo_or_absolute(summary_json, repo_root)}`",
        f"- Candidate metrics JSON: `{_relative_to_repo_or_absolute(candidate_metrics_json, repo_root)}`",
        "",
        "## Assumptions",
        "",
    ]
    lines.extend(f"- {assumption}" for assumption in payload["assumptions"])
    lines.extend(
        [
            "",
            "## Next Recommended Step",
            "",
            (
                "Promote the selected PCA-plus-ridge hyperparameters into the main observation baseline configuration, "
                "rerun the baseline once on the default comparison split, and treat that run as the tuned linear reference."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_observation_comparison_note(
    path: Path,
    *,
    payload: dict[str, Any],
    summary_json: Path,
    repo_root: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_model = payload["best_model"]
    lines = [
        "# Experiment: Observation-Driven Limited ML Comparison",
        "",
        f"Experiment ID: `{payload['experiment_id']}`",
        "",
        "## Objective",
        "",
        (
            "Run a limited comparison of representative ML families on the same "
            "observation-driven synthetic inversion task without changing the frozen "
            "full-grid target representation or the existing five-family corpus."
        ),
        "",
        "## Inputs Used",
        "",
        f"- Pairing manifest: `{payload['pairing_manifest_path']}`",
        f"- Split strategy: `{payload['split_strategy']}`",
        f"- Included models: `{', '.join(record['model_name'] for record in payload['model_ranking'])}`",
        "",
        "## Best Current Model",
        "",
        f"- Model name: `{best_model['model_name']}`",
        f"- Experiment ID: `{best_model['experiment_id']}`",
        f"- Validation RMSE: `{best_model['validation_mean_rmse']:.6f}` km/s",
        f"- Test RMSE: `{best_model['test_mean_rmse']:.6f}` km/s",
        "",
        "## Model Ranking",
        "",
    ]
    for record in payload["model_ranking"]:
        lines.append(
            f"- Rank {record['rank']}: `{record['model_name']}` "
            f"(validation RMSE `{record['validation_mean_rmse']:.6f}`, "
            f"test RMSE `{record['test_mean_rmse']:.6f}` km/s)"
        )
    lines.extend(
        [
            "",
            "## Outputs Created",
            "",
            f"- Summary JSON: `{_relative_to_repo_or_absolute(summary_json, repo_root)}`",
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
                "Use the current best-performing model as the benchmark comparison winner for this "
                "bounded synthetic setup, and treat the remaining models as representative "
                "reference points rather than as an exhaustive benchmark frontier."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_paper_pseudo_bending_ml_suite_note(
    path: Path,
    payload: dict[str, Any],
    summary_json: Path,
    repeated_summary_json: Path,
    leave_one_family_out_summary_json: Path,
    noise_robustness_summary_json: Path,
    repo_root: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_model = payload["best_model"]
    lines = [
        "# Experiment: Paper Pseudo-Bending ML Suite v1",
        "",
        f"Experiment ID: `{payload['experiment_id']}`",
        "",
        "## Objective",
        "",
        (
            "Evaluate the selected observation-driven PCA model family on the dedicated "
            "Phase 3 pseudo-bending corpus for paper Results and future SCIE-paper reporting."
        ),
        "",
        "## Inputs Used",
        "",
        f"- Input corpus ID: `{payload['input_corpus_id']}`",
        f"- Input manifest: `{payload['input_manifest_path']}`",
        f"- Simulator recorded in manifest: `{payload['simulator']}`",
        f"- Fixed split seed: `{payload['fixed_split_seed']}`",
        f"- Repeated split seeds: `{', '.join(str(seed) for seed in payload['repeated_split_seeds'])}`",
        f"- Noise levels: `{', '.join(str(level) for level in payload['travel_time_noise_levels_s'])}` seconds",
        "",
        "## Best Current Model",
        "",
        f"- Model name: `{best_model['model_name']}`",
        f"- Selection criterion: `{payload['best_model_selection_metric']}`",
        f"- Fixed validation RMSE: `{best_model['fixed_validation_rmse']:.6f}` km/s",
        f"- Fixed test RMSE: `{best_model['fixed_test_rmse']:.6f}` km/s",
        "",
        "## Model Ranking",
        "",
    ]
    for record in payload["model_ranking"]:
        lines.append(
            f"- Rank {record['rank']}: `{record['model_name']}` "
            f"(validation RMSE `{record['fixed_validation_rmse']:.6f}`, "
            f"test RMSE `{record['fixed_test_rmse']:.6f}` km/s)"
        )
    lines.extend(
        [
            "",
            "## Outputs Created",
            "",
            f"- Suite summary JSON: `{_relative_to_repo_or_absolute(summary_json, repo_root)}`",
            f"- Repeated split summary JSON: `{_relative_to_repo_or_absolute(repeated_summary_json, repo_root)}`",
            f"- Leave-one-family-out summary JSON: `{_relative_to_repo_or_absolute(leave_one_family_out_summary_json, repo_root)}`",
            f"- Noise robustness summary JSON: `{_relative_to_repo_or_absolute(noise_robustness_summary_json, repo_root)}`",
            "",
            "## Assumptions",
            "",
        ]
    )
    lines.extend(f"- {assumption}" for assumption in payload["assumptions"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in payload["limitations"])
    lines.extend(
        [
            "",
            "## Next Recommended Step",
            "",
            (
                "Review the generated metrics and prediction artifacts before converting any "
                "values into paper Results text or paper tables."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_strengthened_note(
    path: Path,
    payload: dict[str, Any],
    summary_json: Path,
    repo_root: Path,
) -> Path:
    best_model = payload["best_model"]
    baseline_best = payload["baseline_reference"]["best_model"]
    baseline_worst = payload["baseline_reference"]["worst_leave_one_family_out"]
    lines = [
        "# Experiment: Paper Pseudo-Bending ML Strengthened v1",
        "",
        f"Experiment ID: `{payload['experiment_id']}`",
        "",
        "## Audit-Driven Rationale",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["audit_driven_rationale"])
    lines.extend(["", "## What Changed", ""])
    lines.extend(f"- {item}" for item in payload["selected_changes"])
    lines.extend(["", "## What Was Intentionally Not Changed", ""])
    lines.extend(f"- {item}" for item in payload["intentionally_not_changed"])
    lines.extend(
        [
            "",
            "## Best Strengthened Candidate",
            "",
            f"- Model: `{best_model['model_name']}`",
            f"- PCA components: `{best_model['pca_component_count']}`",
            f"- Ridge alpha: `{best_model['ridge_alpha']}`",
            f"- Fixed validation RMSE: `{best_model['fixed_split']['validation']['mean_rmse']:.6f}` km/s",
            f"- Fixed test RMSE: `{best_model['fixed_split']['test']['mean_rmse']:.6f}` km/s",
            f"- Repeated mean validation RMSE: `{best_model['repeated_split']['aggregate_metrics']['mean_validation_rmse']:.6f}` km/s",
            f"- Worst leave-one-family-out family: `{best_model['leave_one_family_out']['worst_held_out_family']}`",
            f"- Worst leave-one-family-out RMSE: `{best_model['leave_one_family_out']['worst_held_out_rmse']:.6f}` km/s",
            "",
            "## Before/After Metrics",
            "",
            f"- Phase 4 best model: `{baseline_best['model_name']}`",
            f"- Phase 4 fixed validation RMSE: `{baseline_best['fixed_validation_rmse']:.6f}` km/s",
            f"- Strengthened fixed validation RMSE delta: `{payload['metric_deltas_vs_phase4']['fixed_validation_rmse_delta']:.6f}` km/s",
            f"- Phase 4 fixed test RMSE: `{baseline_best['fixed_test_rmse']:.6f}` km/s",
            f"- Strengthened fixed test RMSE delta: `{payload['metric_deltas_vs_phase4']['fixed_test_rmse_delta']:.6f}` km/s",
            f"- Phase 4 worst LOO family: `{baseline_worst['held_out_family']}` "
            f"RMSE `{baseline_worst['mean_rmse']:.6f}` km/s",
            f"- Strengthened worst LOO RMSE delta: `{payload['metric_deltas_vs_phase4']['worst_leave_one_family_out_rmse_delta']:.6f}` km/s",
            "",
            "## Phase 5 Suitability",
            "",
            f"- Proceed to Phase 5: `{str(payload['proceed_to_phase_5']).lower()}`",
            f"- Recommended next phase: `{payload['recommended_next_phase']}`",
            "",
            "## Remaining Risks",
            "",
        ]
    )
    if payload["remaining_risks"]:
        lines.extend(f"- {risk}" for risk in payload["remaining_risks"])
    else:
        lines.append("- No blocking risks were identified by the configured strengthened criteria.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Strengthened summary JSON: `{_relative_to_repo_or_absolute(summary_json, repo_root)}`",
            "",
            "## Limitations",
            "",
            "This strengthening is a bounded hyperparameter and diagnostic pass. It does not add new "
            "scientific claims, does not expand the corpus, and does not validate against real data.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_faulted_generalization_note(
    path: Path,
    *,
    payload: dict[str, Any],
    summary_json: Path,
    diagnosis_json: Path,
    repo_root: Path,
) -> Path:
    comparison = payload["before_after_comparison"]
    recommendation = payload["recommendation"]
    enhanced = payload["directional_slowness_feature_model"]
    diagnosis = payload["diagnosis_summary"]
    lines = [
        "# Experiment: Paper Pseudo-Bending Faulted Generalization v1",
        "",
        f"Experiment ID: `{payload['experiment_id']}`",
        "",
        "## Objective",
        "",
        (
            "Diagnose the weak `faulted` leave-one-family-out behavior and test exactly one "
            "bounded, scientifically explainable feature-side intervention."
        ),
        "",
        "## Diagnosis Summary",
        "",
        f"- Targeted weakness: `{diagnosis['targeted_weakness']}`",
        f"- Audit worst-family RMSE: `{diagnosis['audit_worst_family_rmse']:.6f}` km/s",
        f"- Faulted case count: `{diagnosis['faulted_case_count']}`",
        f"- Faulted parameter variants: `{diagnosis['faulted_parameter_variant_count']}`",
        f"- Faulted acquisition profiles: `{diagnosis['faulted_geometry_profile_count']}`",
        f"- Primary diagnosis: {diagnosis['primary_diagnosis']}",
        "",
        "## Intervention",
        "",
        f"- Intervention ID: `{payload['intervention']['intervention_id']}`",
        f"- Model: `{payload['intervention']['model_name']}`",
        f"- PCA components: `{payload['intervention']['pca_component_count']}`",
        f"- Rationale: {payload['intervention']['rationale']}",
        "",
        "## What Was Intentionally Not Changed",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["intervention"]["intentionally_not_changed"])
    lines.extend(
        [
            "",
            "## Before/After Metrics",
            "",
            f"- Phase 4 faulted LOO RMSE: `{comparison['phase4_faulted_loo_rmse']:.6f}` km/s",
            f"- Phase 4.5B faulted LOO RMSE: `{comparison['phase45b_faulted_loo_rmse']:.6f}` km/s",
            (
                "- Directional-feature faulted LOO RMSE: "
                f"`{comparison['directional_features_faulted_loo_rmse']:.6f}` km/s"
            ),
            (
                "- Directional-feature faulted LOO MAE: "
                f"`{comparison['directional_features_faulted_loo_mae']:.6f}` km/s"
            ),
            (
                "- Faulted LOO RMSE delta vs Phase 4: "
                f"`{comparison['faulted_loo_rmse_delta_vs_phase4']:.6f}` km/s"
            ),
            (
                "- Faulted LOO RMSE delta vs same model without derived features: "
                f"`{comparison['faulted_loo_rmse_delta_vs_same_model_no_derived_features']:.6f}` km/s"
            ),
            (
                "- Directional-feature fixed validation RMSE: "
                f"`{enhanced['fixed_split']['validation']['mean_rmse']:.6f}` km/s"
            ),
            (
                "- Directional-feature fixed test RMSE: "
                f"`{enhanced['fixed_split']['test']['mean_rmse']:.6f}` km/s"
            ),
            (
                "- Directional-feature repeated mean validation RMSE: "
                f"`{enhanced['repeated_split']['aggregate_metrics']['mean_validation_rmse']:.6f}` km/s"
            ),
            "",
            "## Recommendation",
            "",
            f"- Proceed to Phase 5: `{str(recommendation['proceed_to_phase_5']).lower()}`",
            f"- Recommended main model: `{recommendation['recommended_main_model']}`",
            f"- Recommended next phase: `{recommendation['recommended_next_phase']}`",
            f"- Rationale: {recommendation['rationale']}",
            "",
            "## Outputs",
            "",
            f"- Diagnosis JSON: `{_relative_to_repo_or_absolute(diagnosis_json, repo_root)}`",
            f"- Summary JSON: `{_relative_to_repo_or_absolute(summary_json, repo_root)}`",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


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


def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_to_repo_or_absolute(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()
