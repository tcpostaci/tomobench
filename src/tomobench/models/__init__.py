"""Machine learning model scaffolding."""

from tomobench.models.baselines import PLANNED_BASELINE_MODELS
from tomobench.models.observation_training import (
    train_observation_ml_comparison,
    train_observation_pca_mlp_ml_baseline,
    train_observation_pca_linear_ml_baseline,
    train_observation_pca_random_forest_ml_baseline,
    train_observation_pca_ridge_ml_baseline,
    train_phase9_grouped_split_ablation_suite,
    train_paper_pseudo_bending_grouped_ml_suite,
    train_phase7_observation_signal_ablation_suite,
    tune_observation_pca_ridge_ml_baseline,
)
from tomobench.models.training import (
    planned_training_note,
    train_expanded_ml_baseline_suite,
    train_feature_distance_ml_baseline,
    train_first_ml_baseline,
    train_reduced_target_ridge_ml_baseline,
)

__all__ = [
    "PLANNED_BASELINE_MODELS",
    "planned_training_note",
    "train_observation_ml_comparison",
    "train_observation_pca_mlp_ml_baseline",
    "train_observation_pca_linear_ml_baseline",
    "train_observation_pca_random_forest_ml_baseline",
    "train_observation_pca_ridge_ml_baseline",
    "train_phase9_grouped_split_ablation_suite",
    "train_paper_pseudo_bending_grouped_ml_suite",
    "train_phase7_observation_signal_ablation_suite",
    "tune_observation_pca_ridge_ml_baseline",
    "train_expanded_ml_baseline_suite",
    "train_feature_distance_ml_baseline",
    "train_first_ml_baseline",
    "train_reduced_target_ridge_ml_baseline",
]
