"""Command-line entry points for tomobench workflows."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tomobench.config import load_settings
from tomobench.datasets import (
    prepare_expanded_ml_supervised_pairings,
    prepare_first_ml_supervised_pairings,
    prepare_first_ml_target_batch,
    prepare_observation_ml_supervised_pairings,
    prepare_paper_pseudo_bending_observation_corpus,
    validate_paper_pseudo_bending_observation_corpus,
)
from tomobench.evaluation.eikonal_benchmarks import run_eikonal_benchmarks
from tomobench.evaluation.eikonal_benchmark_visualizations import (
    render_eikonal_benchmark_visualizations,
)
from tomobench.evaluation.pseudo_bending_comparison import run_ray_method_comparison
from tomobench.evaluation.pseudo_bending_paper_benchmark import (
    run_pseudo_bending_paper_benchmark,
)
from tomobench.evaluation.paper_ml_audit import audit_paper_pseudo_bending_ml_suite
from tomobench.evaluation.paper_candidate_package import package_paper_candidate_v1
from tomobench.evaluation.paper_candidate_v2 import package_paper_candidate_v2
from tomobench.evaluation.paper_candidate_v3 import package_paper_candidate_v3
from tomobench.evaluation.paper_candidate_v4 import package_paper_candidate_v4
from tomobench.evaluation.phase8_paired_test_comparison import (
    run_phase8_paired_test_comparison,
)
from tomobench.evaluation.phase8_uncertainty_report import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_PERMUTATION_RESAMPLES,
    DEFAULT_PERMUTATION_SEED,
    run_phase8_uncertainty_report,
)
from tomobench.evaluation.reproducibility_manifest import (
    run_submission_reproducibility_package,
)
from tomobench.evaluation.pca_target_audit import run_pca_target_audit
from tomobench.evaluation.path_length_audit import run_path_length_audit
from tomobench.evaluation.classical_baseline_fairness import (
    run_fair_reference_ray_baseline,
)
from tomobench.evaluation.coverage_aware_evaluation import run_coverage_aware_evaluation
from tomobench.evaluation.forward_solver_failure_diagnostics import (
    run_forward_solver_failure_diagnostics,
)
from tomobench.evaluation.forward_solver_validation import run_forward_solver_validation
from tomobench.evaluation.forward_solver_validation_v2 import (
    run_forward_solver_validation_v2,
)
from tomobench.evaluation.grouped_split_paired_comparison import (
    run_grouped_split_paired_comparison,
    run_grouped_split_uncertainty_report,
)
from tomobench.evaluation.grouped_split_noise_robustness import (
    run_grouped_split_noise_robustness,
)
from tomobench.evaluation.target_level_uncertainty import (
    run_target_level_uncertainty_report,
)
from tomobench.evaluation.structural_recovery_metrics import (
    run_structural_recovery_metrics,
)
from tomobench.evaluation.target_split_audit import run_target_split_audit
from tomobench.generation.stations import generate_grid_stations, save_station_configuration
from tomobench.models.training import (
    train_expanded_ml_baseline_suite,
    train_feature_distance_ml_baseline,
    train_first_ml_baseline,
    train_reduced_target_ridge_ml_baseline,
)
from tomobench.models.observation_training import (
    train_observation_ml_comparison,
    train_observation_pca_mlp_ml_baseline,
    train_observation_pca_linear_ml_baseline,
    train_observation_pca_random_forest_ml_baseline,
    train_observation_pca_ridge_ml_baseline,
    train_phase7_observation_signal_ablation_suite,
    train_phase7_learning_curve,
    train_phase9_grouped_split_learning_curve,
    train_phase9_grouped_split_ablation_suite,
    train_paper_pseudo_bending_grouped_ml_suite,
    train_paper_pseudo_bending_faulted_generalization,
    train_paper_pseudo_bending_ml_strengthened,
    train_paper_pseudo_bending_ml_suite,
    tune_observation_pca_ridge_ml_baseline,
)
from tomobench.tomography.inversion import (
    run_coverage_aware_known_ray_inversion_diagnostic,
    run_known_ray_inversion_diagnostic,
)
from tomobench.tomography.corpus import (
    run_known_ray_corpus_inversion,
    run_known_ray_perturbation_corpus_inversion,
)
from tomobench.tomography.comparison import run_final_phase6_comparison
from tomobench.tomography.reference import (
    run_reference_ray_corpus_inversion,
    run_reference_ray_regularization_sensitivity,
)
from tomobench.utils.paths import get_config_path
from tomobench.utils.paths import get_repo_root
from tomobench.validation import validate_first_vertical_slice
from tomobench.visualization import save_target_grid_structure_figure
from tomobench.workflow import (
    ALL_VELOCITY_SCENARIOS,
    _build_output_paths,
    _normalize_simulation_method,
    _normalize_velocity_scenario,
    compare_first_slice_simulators,
    render_all_velocity_scenarios,
    run_first_vertical_slice,
)


def main() -> None:
    """Run a tomobench command."""
    parser = argparse.ArgumentParser(prog="tomobench")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser(
        "run-first-slice",
        help="Run the first synthetic research vertical slice end to end.",
    )
    run_parser.add_argument(
        "--simulator",
        choices=(
            "straight-line",
            "straight-line-layered-placeholder",
            "layered-ray",
            "layered-ray-tracing",
            "eikonal-3d",
            "pseudo-bending",
        ),
        help="Override simulation.method from config for this run.",
    )
    run_parser.add_argument(
        "--velocity-scenario",
        choices=("layered", "block-anomaly", "faulted", "salt-dome", "dyke-intrusion"),
        help="Override the Cartesian velocity-grid scenario for this run.",
    )
    compare_parser = subparsers.add_parser(
        "compare-simulators",
        help="Compare straight-line, layered ray tracing, and eikonal prototype benchmark travel times.",
    )
    compare_parser.add_argument(
        "--velocity-scenario",
        choices=("layered", "block-anomaly", "faulted", "salt-dome", "dyke-intrusion"),
        help="Override the Cartesian velocity-grid scenario for the comparison run.",
    )
    validate_parser = subparsers.add_parser(
        "validate-first-slice",
        help="Validate generated first-slice artifacts and schema consistency.",
    )
    validate_parser.add_argument(
        "--simulator",
        choices=(
            "straight-line",
            "straight-line-layered-placeholder",
            "layered-ray",
            "layered-ray-tracing",
            "eikonal-3d",
            "pseudo-bending",
        ),
        help="Validate outputs for a specific simulator method.",
    )
    validate_parser.add_argument(
        "--velocity-scenario",
        choices=("layered", "block-anomaly", "faulted", "salt-dome", "dyke-intrusion"),
        help="Validate outputs for a specific Cartesian velocity-grid scenario.",
    )
    subparsers.add_parser(
        "benchmark-eikonal",
        help="Benchmark eikonal runtime and exact-reference accuracy on bounded test cases.",
    )
    subparsers.add_parser(
        "render-eikonal-benchmark-figures",
        help="Render publication-friendly connectivity, path, and travel-time figures for the eikonal benchmark.",
    )
    subparsers.add_parser(
        "compare-ray-methods",
        help="Compare Dijkstra and independent pseudo-bending ray paths on one bounded 3D case.",
    )
    subparsers.add_parser(
        "benchmark-pseudo-bending-paper",
        help="Run the paper pseudo-bending benchmark suite before regenerating the ML corpus.",
    )
    subparsers.add_parser(
        "generate-grid-stations",
        help="Generate and save the configured deterministic grid station layout.",
    )
    subparsers.add_parser(
        "render-all-velocity-scenarios",
        help="Render first-slice artifacts for layered, block-anomaly, faulted, salt-dome, and dyke-intrusion scenarios.",
    )
    subparsers.add_parser(
        "render-target-grid-figure",
        help="Render the frozen ML target-grid structure figure for report use.",
    )
    subparsers.add_parser(
        "prepare-ml-target-batch",
        help="Prepare the first frozen ML target batch for layered, block-anomaly, and faulted scenarios.",
    )
    subparsers.add_parser(
        "prepare-ml-supervised-pairings",
        help="Prepare scenario-matched observation-target pairings for the first supervised ML slice.",
    )
    subparsers.add_parser(
        "prepare-expanded-ml-supervised-pairings",
        help="Prepare additional paired samples within the frozen full-grid target contract, including bounded geological-family variants.",
    )
    subparsers.add_parser(
        "prepare-observation-ml-supervised-pairings",
        help="Prepare the larger five-family paired corpus for observation-driven supervised inversion training.",
    )
    subparsers.add_parser(
        "prepare-paper-pseudo-bending-observation-corpus",
        help="Prepare the paper-facing five-family observation corpus with pseudo_bending_3d.",
    )
    subparsers.add_parser(
        "validate-paper-pseudo-bending-observation-corpus",
        help="Validate the paper-facing pseudo-bending observation corpus manifest and files.",
    )
    subparsers.add_parser(
        "train-first-ml-baseline",
        help="Train and evaluate the first bounded interpretable ML baseline.",
    )
    subparsers.add_parser(
        "train-feature-distance-ml-baseline",
        help="Train and evaluate the next simple feature-using ML baseline.",
    )
    subparsers.add_parser(
        "train-reduced-target-ml-baseline",
        help="Train and evaluate the reduced-target linear ML baseline.",
    )
    subparsers.add_parser(
        "train-expanded-ml-baseline-suite",
        help="Retrain bounded baselines on the expanded manifest and write one summary recommendation.",
    )
    subparsers.add_parser(
        "train-observation-pca-linear-baseline",
        help="Train the observation-driven PCA plus plain linear full-grid baseline.",
    )
    subparsers.add_parser(
        "train-observation-pca-ridge-baseline",
        help="Train the first observation-driven PCA plus ridge full-grid inversion baseline.",
    )
    subparsers.add_parser(
        "train-observation-pca-mlp-baseline",
        help="Train the first shallow non-linear observation-driven PCA plus MLP full-grid baseline.",
    )
    subparsers.add_parser(
        "train-observation-pca-random-forest-baseline",
        help="Train the observation-driven PCA plus random-forest full-grid baseline.",
    )
    subparsers.add_parser(
        "tune-observation-pca-ridge-baseline",
        help="Tune the observation-driven PCA plus ridge baseline across repeated family-stratified splits.",
    )
    subparsers.add_parser(
        "train-observation-ml-comparison",
        help="Run the limited observation-driven ML comparison across linear, ridge, random-forest, and shallow-MLP models.",
    )
    subparsers.add_parser(
        "train-paper-pseudo-bending-ml-suite",
        help="Run the paper ML suite on the Phase 3 pseudo-bending observation corpus.",
    )
    subparsers.add_parser(
        "train-paper-pseudo-bending-grouped-ml-suite",
        help="Run the paper ML suite with target-level grouped splits and preserved old outputs.",
    )
    subparsers.add_parser(
        "audit-paper-pseudo-bending-ml-suite",
        help="Audit generated Phase 4 paper pseudo-bending ML suite results before Phase 5.",
    )
    target_split_audit_parser = subparsers.add_parser(
        "audit-target-split",
        help="Audit exact and tolerance-level velocity-target duplication across the current ML split.",
    )
    target_split_audit_parser.add_argument(
        "--manifest",
        default="outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json",
        help="Target/observation corpus manifest to audit.",
    )
    target_split_audit_parser.add_argument(
        "--current-split",
        default=(
            "outputs/generated/ml_baselines/paper_pseudo_bending_ml_suite_v1/"
            "models/pca-linear/fixed_split/feature_matrix.csv"
        ),
        help="Persisted case-level split artifact used by the current fixed ML evaluation.",
    )
    target_split_audit_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/target_split_audit_v1",
        help="Directory for target registry, hash summaries, leakage records, and audit metadata.",
    )
    target_split_audit_parser.add_argument(
        "--numerical-atol",
        type=float,
        default=1.0e-12,
        help="Primary absolute tolerance for numerical target equivalence.",
    )
    target_split_audit_parser.add_argument(
        "--numerical-rtol",
        type=float,
        default=1.0e-12,
        help="Primary relative tolerance for numerical target equivalence.",
    )
    pca_target_audit_parser = subparsers.add_parser(
        "audit-pca-target-representation",
        help="Audit PCA target variance retention, oracle reconstruction, and model error.",
    )
    pca_target_audit_parser.add_argument(
        "--metrics",
        default=(
            "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/"
            "models/pca-linear/fixed_split/metrics.json"
        ),
        help="Grouped-split PCA-linear metrics JSON used for split assignments and predictions.",
    )
    pca_target_audit_parser.add_argument(
        "--manifest",
        default="outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json",
        help="Corpus manifest containing the target velocity grids.",
    )
    pca_target_audit_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/pca_target_audit_v1",
        help="Directory for PCA variance and reconstruction audit artifacts.",
    )
    pca_target_audit_parser.add_argument(
        "--components",
        type=int,
        default=4,
        help="Configured PCA component count to use for the oracle comparison.",
    )
    pca_target_audit_parser.add_argument(
        "--requested-components",
        default="1,2,3,4,5,10",
        help="Comma-separated component counts for the variance-retention report.",
    )
    path_length_audit_parser = subparsers.add_parser(
        "audit-path-length-feature",
        help="Audit whether stored ray-path length is privileged target-model information.",
    )
    path_length_audit_parser.add_argument(
        "--manifest",
        default="outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json",
        help="Corpus manifest containing source-receiver dataset paths.",
    )
    path_length_audit_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/privileged_path_length_audit_v1",
        help="Directory for path-length provenance and numerical diagnostics.",
    )
    fair_classical_parser = subparsers.add_parser(
        "audit-fair-reference-baseline",
        help="Select reference-ray regularization on validation and freeze it for test evaluation.",
    )
    fair_classical_parser.add_argument(
        "--sensitivity-case-metrics",
        default=(
            "outputs/generated/classical_tomography/reference_ray_regularization_sensitivity_v1/"
            "regularization_sensitivity_case_metrics.csv"
        ),
        help="Existing reference-ray sensitivity case metrics CSV.",
    )
    fair_classical_parser.add_argument(
        "--grouped-split",
        default=(
            "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/"
            "target_group_split_assignments.csv"
        ),
        help="Target-grouped split assignment CSV.",
    )
    fair_classical_parser.add_argument(
        "--ablation-metrics",
        default=(
            "outputs/generated/ml_baselines/phase9_grouped_split_ablation_v1/"
            "cell_centered_metrics.csv"
        ),
        help="Grouped ablation cell metrics used for paired ML controls.",
    )
    fair_classical_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/fair_reference_ray_baseline_v1",
        help="Directory for validation selection and frozen test metrics.",
    )
    coverage_parser = subparsers.add_parser(
        "audit-coverage-aware-evaluation",
        help="Evaluate grouped fixed-test reconstructions over increasing ray-coverage thresholds.",
    )
    coverage_parser.add_argument(
        "--manifest",
        default="outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json",
        help="Corpus manifest containing target grids.",
    )
    coverage_parser.add_argument(
        "--grouped-split",
        default="outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/target_group_split_assignments.csv",
        help="Target-grouped split assignment CSV.",
    )
    coverage_parser.add_argument(
        "--ml-metrics",
        default="outputs/generated/ml_baselines/phase9_grouped_split_ablation_v1/cell_centered_metrics.csv",
        help="Grouped ablation cell-centered ML metrics CSV.",
    )
    coverage_parser.add_argument(
        "--classical-corpus-dir",
        default="outputs/generated/classical_tomography/fair_reference_ray_corpus_v1",
        help="Fair fixed-parameter reference-ray corpus with predictions and coverage.",
    )
    coverage_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/coverage_aware_evaluation_v1",
        help="Directory for coverage-aware case, depth, and distribution tables.",
    )
    coverage_parser.add_argument(
        "--thresholds-km",
        default="0.0,1e-9,0.001,0.1,1.0,5.0,10.0,20.0,50.0",
        help="Comma-separated accumulated path-length thresholds in km.",
    )
    coverage_parser.add_argument(
        "--methods",
        default="reference_prior,optimized_fixed_ray,full_input_euclidean_path,full_input,travel_time_only",
        help="Comma-separated method IDs from the ML ablation and classical outputs.",
    )
    structural_parser = subparsers.add_parser(
        "audit-structural-recovery",
        help="Evaluate family-specific structural recovery metrics on the grouped fixed test set.",
    )
    structural_parser.add_argument(
        "--manifest",
        default="outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json",
        help="Corpus manifest containing target grids and generator metadata.",
    )
    structural_parser.add_argument(
        "--grouped-split",
        default="outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/target_group_split_assignments.csv",
        help="Target-grouped split assignment CSV.",
    )
    structural_parser.add_argument(
        "--ml-metrics",
        default="outputs/generated/ml_baselines/phase9_grouped_split_ablation_v1/cell_centered_metrics.csv",
        help="Grouped ablation cell-centered ML metrics CSV.",
    )
    structural_parser.add_argument(
        "--classical-corpus-dir",
        default="outputs/generated/classical_tomography/fair_reference_ray_corpus_v1",
        help="Fair fixed-parameter reference-ray corpus with predictions.",
    )
    structural_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/structural_recovery_metrics_v1",
        help="Directory for structural case, layer, and summary tables.",
    )
    structural_parser.add_argument(
        "--methods",
        default="reference_prior,optimized_fixed_ray,full_input_euclidean_path,full_input,travel_time_only",
        help="Comma-separated method IDs from the ML ablation and classical outputs.",
    )
    forward_solver_parser = subparsers.add_parser(
        "audit-forward-solver",
        help="Validate analytic pseudo-bending cases and audit corpus solver convergence.",
    )
    forward_solver_parser.add_argument(
        "--manifest",
        default="outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json",
        help="Pseudo-bending corpus manifest whose ray-path sidecars will be audited.",
    )
    forward_solver_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/forward_solver_validation_v1",
        help="Directory for forward-solver validation and convergence artifacts.",
    )
    forward_solver_failure_parser = subparsers.add_parser(
        "audit-forward-solver-failures",
        help="Diagnose historical forward-solver failures and bounded retry strategies.",
    )
    forward_solver_failure_parser.add_argument(
        "--manifest",
        default=(
            "outputs/datasets/ml_supervised/"
            "paper_pseudo_bending_observation_pairs_v1/manifest.json"
        ),
        help="Existing observation-corpus manifest.",
    )
    forward_solver_failure_parser.add_argument(
        "--failure-input",
        default=(
            "outputs/generated/submission_readiness/forward_solver_validation_v1/"
            "corpus_solver_failure_records.csv"
        ),
        help="Historical failure-record CSV to audit.",
    )
    forward_solver_failure_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/forward_solver_validation_v2",
        help="Directory for failure diagnosis outputs.",
    )
    forward_solver_v2_parser = subparsers.add_parser(
        "audit-forward-solver-v2",
        help="Run expanded analytic, heterogeneous, and graph-resolution solver validation.",
    )
    forward_solver_v2_parser.add_argument(
        "--manifest",
        default=(
            "outputs/datasets/ml_supervised/"
            "paper_pseudo_bending_observation_pairs_v1/manifest.json"
        ),
        help="Existing observation-corpus manifest used for heterogeneous validation pairs.",
    )
    forward_solver_v2_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/forward_solver_validation_v2",
        help="Directory for expanded forward-solver validation outputs.",
    )
    forward_solver_v2_parser.add_argument(
        "--heterogeneous-cases-per-family",
        type=int,
        default=5,
        help="Deterministically selected historical cases per geological family.",
    )
    forward_solver_v2_parser.add_argument(
        "--pairs-per-case",
        type=int,
        default=5,
        help="Deterministically selected source-receiver pairs per heterogeneous case.",
    )
    forward_solver_v2_parser.add_argument(
        "--resolution-pairs-per-family",
        type=int,
        default=3,
        help="Pairs per family in the graph resolution study.",
    )
    grouped_noise_parser = subparsers.add_parser(
        "audit-grouped-split-noise-robustness",
        help="Evaluate corrected grouped-test ML and reference-ray methods under travel-time noise.",
    )
    grouped_noise_parser.add_argument(
        "--manifest",
        default="outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json",
        help="Pseudo-bending corpus manifest.",
    )
    grouped_noise_parser.add_argument(
        "--grouped-split",
        default="outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/target_group_split_assignments.csv",
        help="Target-grouped split assignment CSV.",
    )
    grouped_noise_parser.add_argument(
        "--classical-corpus-dir",
        default="outputs/generated/classical_tomography/fair_reference_ray_corpus_v1",
        help="Fair reference-ray corpus containing fixed geometry and reference grids.",
    )
    grouped_noise_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/noise_robustness_v1",
        help="Directory for grouped noise case and summary artifacts.",
    )
    grouped_noise_parser.add_argument(
        "--noise-levels",
        default="0.0,0.025,0.05,0.1",
        help="Comma-separated Gaussian travel-time noise standard deviations in seconds.",
    )
    grouped_noise_parser.add_argument(
        "--noise-seed",
        type=int,
        default=909,
        help="Base seed for deterministic case/observation noise offsets.",
    )
    grouped_noise_parser.add_argument(
        "--reference-damping",
        type=float,
        default=1.0,
        help="Validation-frozen reference-ray damping parameter.",
    )
    grouped_noise_parser.add_argument(
        "--reference-smoothing",
        type=float,
        default=10.0,
        help="Validation-frozen reference-ray smoothing parameter.",
    )
    grouped_comparison_parser = subparsers.add_parser(
        "run-grouped-split-paired-comparison",
        help="Compare corrected grouped-split methods on the same fixed test cases.",
    )
    grouped_comparison_parser.add_argument(
        "--output-dir",
        default=(
            "outputs/generated/submission_readiness/"
            "phase9_grouped_split_paired_comparison_v1"
        ),
        help="Directory for corrected paired comparison artifacts.",
    )
    grouped_comparison_parser.add_argument(
        "--ml-metrics",
        default=(
            "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/"
            "models/pca-linear/fixed_split/metrics.json"
        ),
        help="Grouped-split PCA-linear metrics JSON.",
    )
    grouped_comparison_parser.add_argument(
        "--ablation-metrics",
        default=(
            "outputs/generated/ml_baselines/phase9_grouped_split_ablation_v1/"
            "cell_centered_metrics.csv"
        ),
        help="Grouped-split ablation cell metrics CSV.",
    )
    grouped_comparison_parser.add_argument(
        "--fair-baseline",
        default=(
            "outputs/generated/submission_readiness/fair_reference_ray_baseline_v1/"
            "fair_reference_baseline_case_metrics.csv"
        ),
        help="Validation-selected frozen fixed-ray and reference-prior case metrics CSV.",
    )
    grouped_comparison_parser.add_argument(
        "--known-ray",
        default=(
            "outputs/generated/classical_tomography/known_ray_perturbation_v1/"
            "known_ray_perturbation_case_summary.csv"
        ),
        help="Corrected known-ray perturbation case metrics CSV.",
    )
    grouped_comparison_parser.add_argument(
        "--grouped-split",
        default=(
            "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/"
            "target_group_split_assignments.csv"
        ),
        help="Target-grouped split assignment CSV.",
    )
    grouped_uncertainty_parser = subparsers.add_parser(
        "run-grouped-split-uncertainty-report",
        help="Bootstrap corrected grouped-test metrics and calculate paired method deltas.",
    )
    grouped_uncertainty_parser.add_argument(
        "--input-dir",
        default=(
            "outputs/generated/submission_readiness/"
            "phase9_grouped_split_paired_comparison_v1"
        ),
        help="Corrected grouped paired-comparison directory.",
    )
    grouped_uncertainty_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/phase9_grouped_split_uncertainty_v1",
        help="Directory for grouped bootstrap and paired-delta outputs.",
    )
    grouped_uncertainty_parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Number of case-level bootstrap resamples per method metric.",
    )
    grouped_uncertainty_parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=8_201,
        help="Base seed for deterministic bootstrap intervals.",
    )
    grouped_uncertainty_parser.add_argument(
        "--permutation-resamples",
        type=int,
        default=10_000,
        help="Number of paired sign-flip resamples.",
    )
    grouped_uncertainty_parser.add_argument(
        "--permutation-seed",
        type=int,
        default=8_202,
        help="Seed for paired sign-flip resampling.",
    )
    target_uncertainty_parser = subparsers.add_parser(
        "run-target-level-uncertainty-report",
        help="Aggregate corrected grouped-test uncertainty over unique target models.",
    )
    target_uncertainty_parser.add_argument(
        "--input-dir",
        default=(
            "outputs/generated/submission_readiness/"
            "phase9_grouped_split_paired_comparison_v1"
        ),
        help="Corrected grouped paired-comparison directory.",
    )
    target_uncertainty_parser.add_argument(
        "--grouped-split",
        default=(
            "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/"
            "target_group_split_assignments.csv"
        ),
        help="Frozen target-grouped split assignment CSV.",
    )
    target_uncertainty_parser.add_argument(
        "--paired-deltas-dir",
        default=(
            "outputs/generated/submission_readiness/"
            "phase9_grouped_split_uncertainty_v1"
        ),
        help="Existing case-level uncertainty directory containing paired_case_deltas.csv.",
    )
    target_uncertainty_parser.add_argument(
        "--output-dir",
        default=(
            "outputs/generated/submission_readiness/"
            "phase9_grouped_split_target_level_uncertainty_v1"
        ),
        help="Directory for target-level uncertainty outputs.",
    )
    target_uncertainty_parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="Number of percentile bootstrap resamples over target-level values.",
    )
    target_uncertainty_parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Base seed for target-level bootstrap intervals.",
    )
    target_uncertainty_parser.add_argument(
        "--permutation-resamples",
        type=int,
        default=DEFAULT_PERMUTATION_RESAMPLES,
        help="Number of paired sign-flip resamples over target-level deltas.",
    )
    target_uncertainty_parser.add_argument(
        "--permutation-seed",
        type=int,
        default=DEFAULT_PERMUTATION_SEED,
        help="Base seed for target-level paired sign-flip resampling.",
    )
    subparsers.add_parser(
        "train-paper-pseudo-bending-ml-strengthened",
        help="Run the audit-driven strengthened paper pseudo-bending ML experiment.",
    )
    subparsers.add_parser(
        "train-paper-pseudo-bending-faulted-generalization",
        help="Run the bounded faulted-family generalization diagnosis and feature intervention.",
    )
    subparsers.add_parser(
        "train-phase7-observation-signal-ablation",
        help="Run the Phase 7 observation-signal ablation suite for travel_time_s controls.",
    )
    subparsers.add_parser(
        "train-phase9-grouped-split-ablation",
        help="Run observation-signal ablations on the target-grouped split.",
    )
    learning_curve_parser = subparsers.add_parser(
        "train-phase7-learning-curve",
        help="Run the Phase 7 fixed-holdout PCA-linear ML learning-curve experiment.",
    )
    learning_curve_parser.add_argument(
        "--training-sizes",
        default="20,40,60,70",
        help="Comma-separated family-balanced training sizes from the fixed training pool.",
    )
    learning_curve_parser.add_argument(
        "--repetition-seeds",
        default="1701,1702,1703",
        help="Comma-separated integer seeds for deterministic training subsets.",
    )
    learning_curve_parser.add_argument(
        "--output-dir",
        default="outputs/generated/ml_baselines/phase7_learning_curve_v1",
        help="Directory for Phase 7 learning-curve outputs.",
    )
    grouped_learning_curve_parser = subparsers.add_parser(
        "train-phase9-grouped-split-learning-curve",
        help="Run a target-group-atomic PCA-linear learning curve on the corrected split.",
    )
    grouped_learning_curve_parser.add_argument(
        "--training-sizes",
        default="30,50",
        help="Comma-separated attainable target-group-atomic training sizes.",
    )
    grouped_learning_curve_parser.add_argument(
        "--repetition-seeds",
        default="1701,1702,1703",
        help="Comma-separated integer seeds for deterministic subsets.",
    )
    grouped_learning_curve_parser.add_argument(
        "--output-dir",
        default="outputs/generated/ml_baselines/phase9_grouped_split_learning_curve_v1",
        help="Directory for corrected grouped learning-curve outputs.",
    )
    subparsers.add_parser(
        "package-paper-candidate-v1",
        help="Package existing validated Phase 2-4B evidence for paper drafting.",
    )
    subparsers.add_parser(
        "package-paper-candidate-v2",
        help="Package final Phase 6 classical-vs-ML evidence for paper drafting.",
    )
    subparsers.add_parser(
        "package-paper-candidate-v3",
        help="Package final Phase 6 and Phase 7 evidence for paper drafting.",
    )
    paper_candidate_v4_parser = subparsers.add_parser(
        "package-paper-candidate-v4",
        help="Package final Phase 6-8 evidence for manuscript submission readiness.",
    )
    paper_candidate_v4_parser.add_argument(
        "--output-dir",
        default="outputs/generated/paper_candidate_v4",
        help="Directory for the Phase 8D manuscript evidence package.",
    )
    phase8_parser = subparsers.add_parser(
        "run-phase8-paired-test-comparison",
        help="Compare ML, classical, and sanity baselines on the same fixed ML test cases.",
    )
    phase8_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/phase8_paired_test_comparison_v1",
        help="Directory for Phase 8A paired comparison outputs.",
    )
    phase8_parser.add_argument(
        "--selected-ml-metrics",
        default="outputs/generated/ml_baselines/paper_pseudo_bending_ml_suite_v1/models/pca-linear/fixed_split/metrics.json",
        help="Selected pca-linear fixed-split metrics JSON used to resolve test case IDs.",
    )
    phase8_parser.add_argument(
        "--include-reference-sensitivity-audit",
        action="store_true",
        help="Include the post-hoc Phase 7B best-covered setting as a clearly labeled audit row.",
    )
    phase8_uncertainty_parser = subparsers.add_parser(
        "run-phase8-uncertainty-report",
        help="Add bootstrap uncertainty and paired statistical reporting to Phase 8A.",
    )
    phase8_uncertainty_parser.add_argument(
        "--input-dir",
        default="outputs/generated/submission_readiness/phase8_paired_test_comparison_v1",
        help="Phase 8A paired comparison directory to consume.",
    )
    phase8_uncertainty_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/phase8_uncertainty_report_v1",
        help="Directory for Phase 8B uncertainty outputs.",
    )
    phase8_uncertainty_parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="Number of deterministic bootstrap resamples per summary metric.",
    )
    phase8_uncertainty_parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Base seed for deterministic bootstrap intervals.",
    )
    phase8_uncertainty_parser.add_argument(
        "--permutation-resamples",
        type=int,
        default=DEFAULT_PERMUTATION_RESAMPLES,
        help="Number of deterministic sign-flip resamples per paired metric.",
    )
    phase8_uncertainty_parser.add_argument(
        "--permutation-seed",
        type=int,
        default=DEFAULT_PERMUTATION_SEED,
        help="Base seed for exploratory paired sign-flip p-values.",
    )
    phase8_uncertainty_parser.add_argument(
        "--exclude-travel-time-only-reference",
        action="store_true",
        help="Omit the optional travel_time_only versus reference-ray paired comparison.",
    )
    reproducibility_parser = subparsers.add_parser(
        "package-submission-reproducibility",
        help="Package commands, provenance, artifacts, and availability drafts for submission readiness.",
    )
    reproducibility_parser.add_argument(
        "--output-dir",
        default="outputs/generated/submission_readiness/reproducibility_manifest_v1",
        help="Directory for the Phase 8C reproducibility package.",
    )
    subparsers.add_parser(
        "run-known-ray-inversion-diagnostic",
        help="Run the first Phase 6 known-ray damped least-squares inversion diagnostic on existing first-slice sidecars.",
    )
    subparsers.add_parser(
        "run-coverage-aware-known-ray-inversion",
        help="Run the Phase 6 coverage-aware smoothed known-ray inversion diagnostic on existing first-slice sidecars.",
    )
    known_ray_perturbation_parser = subparsers.add_parser(
        "run-known-ray-perturbation-corpus",
        help="Diagnose the known-ray anomaly with true geometry and perturbations around the reference prior.",
    )
    known_ray_perturbation_parser.add_argument(
        "--manifest",
        default="outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json",
        help="Corpus manifest to consume.",
    )
    known_ray_perturbation_parser.add_argument(
        "--output-dir",
        default="outputs/generated/classical_tomography/known_ray_perturbation_v1",
        help="Directory for corrected known-ray perturbation outputs.",
    )
    known_ray_perturbation_parser.add_argument(
        "--reference-ray-corpus-dir",
        default="outputs/generated/classical_tomography/reference_ray_corpus_v1",
        help="Reference-ray corpus directory containing the shared prior grids.",
    )
    known_ray_perturbation_parser.add_argument(
        "--legacy-case-summary",
        default="outputs/generated/classical_tomography/known_ray_corpus_v1/known_ray_corpus_case_summary.csv",
        help="Legacy absolute-slowness known-ray case summary used for paired diagnosis.",
    )
    known_ray_perturbation_parser.add_argument(
        "--damping",
        type=float,
        default=0.01,
        help="Fixed damping used to isolate the formulation change.",
    )
    known_ray_perturbation_parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Fixed smoothing used to isolate the formulation change.",
    )
    known_ray_perturbation_parser.add_argument(
        "--minimum-coverage-km",
        type=float,
        default=1.0e-9,
        help="Minimum per-cell ray coverage for covered-cell metrics.",
    )
    corpus_known_ray_parser = subparsers.add_parser(
        "run-known-ray-corpus-inversion",
        help="Run the Phase 6 coverage-aware known-ray diagnostic across the paper pseudo-bending corpus.",
    )
    corpus_known_ray_parser.add_argument(
        "--manifest",
        default="outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json",
        help="Corpus manifest to consume.",
    )
    corpus_known_ray_parser.add_argument(
        "--output-dir",
        default="outputs/generated/classical_tomography/known_ray_corpus_v1",
        help="Directory for corpus-level known-ray inversion outputs.",
    )
    corpus_known_ray_parser.add_argument(
        "--ml-suite-summary",
        default="outputs/generated/ml_baselines/paper_pseudo_bending_ml_suite_v1/suite_summary.json",
        help="Existing ML suite summary used for the caveated pca-linear comparison artifact.",
    )
    corpus_known_ray_parser.add_argument(
        "--damping-values",
        default="0.01,0.1,1.0,10.0",
        help="Comma-separated lambda values for damped least squares.",
    )
    corpus_known_ray_parser.add_argument(
        "--smoothing-values",
        default="0.0,0.01,0.1,1.0",
        help="Comma-separated alpha values for Cartesian neighbor smoothing.",
    )
    corpus_known_ray_parser.add_argument(
        "--minimum-coverage-km",
        type=float,
        default=1.0e-9,
        help="Minimum per-cell ray coverage for covered-cell metrics.",
    )
    reference_ray_parser = subparsers.add_parser(
        "run-reference-ray-corpus-inversion",
        help="Run the Phase 6 reference-model fixed-ray perturbation inversion corpus diagnostic.",
    )
    reference_ray_parser.add_argument(
        "--manifest",
        default="outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json",
        help="Corpus manifest to consume.",
    )
    reference_ray_parser.add_argument(
        "--output-dir",
        default="outputs/generated/classical_tomography/reference_ray_corpus_v1",
        help="Directory for reference-ray inversion outputs.",
    )
    reference_ray_parser.add_argument(
        "--known-ray-summary",
        default="outputs/generated/classical_tomography/known_ray_corpus_v1/known_ray_corpus_summary.json",
        help="Existing known-ray corpus summary for the caveated comparison artifact.",
    )
    reference_ray_parser.add_argument(
        "--ml-suite-summary",
        default="outputs/generated/ml_baselines/paper_pseudo_bending_ml_suite_v1/suite_summary.json",
        help="Existing ML suite summary used for the caveated pca-linear comparison artifact.",
    )
    reference_ray_parser.add_argument(
        "--damping-values",
        default="0.01,0.1,1.0,10.0",
        help="Comma-separated lambda values for perturbation inversion.",
    )
    reference_ray_parser.add_argument(
        "--smoothing-values",
        default="0.0,0.01,0.1,1.0",
        help="Comma-separated alpha values for Cartesian neighbor smoothing.",
    )
    reference_ray_parser.add_argument(
        "--minimum-coverage-km",
        type=float,
        default=1.0e-9,
        help="Minimum per-cell ray coverage for covered-cell metrics.",
    )
    final_comparison_parser = subparsers.add_parser(
        "run-final-phase6-comparison",
        help="Run the final Phase 6 classical-vs-ML metric-contract audit.",
    )
    final_comparison_parser.add_argument(
        "--output-dir",
        default="outputs/generated/classical_tomography/final_phase6_comparison_v1",
        help="Directory for final Phase 6 comparison outputs.",
    )
    final_comparison_parser.add_argument(
        "--known-ray-summary",
        default="outputs/generated/classical_tomography/known_ray_corpus_v1/known_ray_corpus_summary.json",
        help="Known-ray corpus summary JSON.",
    )
    final_comparison_parser.add_argument(
        "--known-ray-case-summary",
        default="outputs/generated/classical_tomography/known_ray_corpus_v1/known_ray_corpus_case_summary.csv",
        help="Known-ray corpus case summary CSV.",
    )
    final_comparison_parser.add_argument(
        "--reference-ray-summary",
        default="outputs/generated/classical_tomography/reference_ray_corpus_v1/reference_ray_corpus_summary.json",
        help="Reference-ray corpus summary JSON.",
    )
    final_comparison_parser.add_argument(
        "--reference-ray-case-summary",
        default="outputs/generated/classical_tomography/reference_ray_corpus_v1/reference_ray_corpus_case_summary.csv",
        help="Reference-ray corpus case summary CSV.",
    )
    final_comparison_parser.add_argument(
        "--ml-metrics",
        default="outputs/generated/ml_baselines/paper_pseudo_bending_ml_suite_v1/models/pca-linear/fixed_split/metrics.json",
        help="Selected pca-linear ML metrics JSON with per-case prediction paths.",
    )
    reference_sensitivity_parser = subparsers.add_parser(
        "run-reference-ray-regularization-sensitivity",
        help="Run the Phase 7B reference-ray lambda/alpha and clipping sensitivity audit.",
    )
    reference_sensitivity_parser.add_argument(
        "--manifest",
        default="outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json",
        help="Corpus manifest to consume.",
    )
    reference_sensitivity_parser.add_argument(
        "--reference-case-summary",
        default="outputs/generated/classical_tomography/reference_ray_corpus_v1/reference_ray_corpus_case_summary.csv",
        help="Phase 6 reference-ray case summary CSV with per-case sidecar paths.",
    )
    reference_sensitivity_parser.add_argument(
        "--output-dir",
        default="outputs/generated/classical_tomography/reference_ray_regularization_sensitivity_v1",
        help="Directory for Phase 7B sensitivity outputs.",
    )
    reference_sensitivity_parser.add_argument(
        "--damping-values",
        default="0.001,0.01,0.1,1.0",
        help="Comma-separated lambda values for reference-ray perturbation inversion.",
    )
    reference_sensitivity_parser.add_argument(
        "--smoothing-values",
        default="0.0,0.1,1.0,10.0",
        help="Comma-separated alpha values for Cartesian neighbor smoothing.",
    )
    reference_sensitivity_parser.add_argument(
        "--minimum-coverage-km",
        type=float,
        default=1.0e-9,
        help="Minimum per-cell ray coverage for covered-cell metrics.",
    )
    reference_sensitivity_parser.add_argument(
        "--case-limit",
        type=int,
        help="Optional balanced case limit for a representative subset run.",
    )
    args = parser.parse_args()

    if args.command == "run-first-slice":
        outputs = run_first_vertical_slice(
            simulation_method=args.simulator,
            velocity_scenario=args.velocity_scenario,
        )
        print("First vertical slice completed.")
        print(f"Station configuration: {outputs.station_configuration}")
        print(f"Earthquake configuration: {outputs.earthquake_configuration}")
        print(f"Velocity model: {outputs.velocity_model}")
        print(f"Velocity grid: {outputs.velocity_grid}")
        print(f"Target spec: {outputs.target_spec}")
        print(f"Travel times: {outputs.travel_times_json}")
        print(f"Ray paths: {outputs.ray_paths_jsonl}")
        print(f"Ray-cell sensitivity: {outputs.sensitivity_jsonl}")
        print(f"Ray-cell sensitivity metadata: {outputs.sensitivity_metadata}")
        print(f"Dataset CSV: {outputs.dataset_csv}")
        print(f"Geometry figure: {outputs.geometry_figure}")
        print(f"Subsurface slice figure: {outputs.subsurface_figure}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "compare-simulators":
        comparison_settings = _settings_from_cli_overrides(
            simulator=None,
            velocity_scenario=args.velocity_scenario,
        )
        comparison_path = compare_first_slice_simulators(comparison_settings)
        print("Simulator comparison completed.")
        print(f"Comparison CSV: {comparison_path}")
        return

    if args.command == "validate-first-slice":
        validation_settings = _settings_from_cli_overrides(
            simulator=args.simulator,
            velocity_scenario=args.velocity_scenario,
        )
        result = validate_first_vertical_slice(validation_settings)
        if result.checks:
            print("Validation checks passed:")
            for check in result.checks:
                print(f"- {check}")
        if result.errors:
            print("Validation errors:")
            for error in result.errors:
                print(f"- {error}")
            raise SystemExit(1)
        print("First vertical slice validation completed successfully.")
        return

    if args.command == "benchmark-eikonal":
        outputs = run_eikonal_benchmarks()
        print("Eikonal benchmark completed.")
        print(f"Benchmark CSV: {outputs.csv_path}")
        print(f"Benchmark JSON: {outputs.json_path}")
        print(f"Benchmark note: {outputs.note_path}")
        return

    if args.command == "render-eikonal-benchmark-figures":
        outputs = render_eikonal_benchmark_visualizations()
        print("Eikonal benchmark figures rendered.")
        print(f"Connectivity figure: {outputs.connectivity_figure}")
        print(f"Path overlay figure: {outputs.path_overlay_figure}")
        print(f"Travel-time field figure: {outputs.travel_time_field_figure}")
        print(f"Visualization note: {outputs.note_path}")
        return

    if args.command == "compare-ray-methods":
        outputs = run_ray_method_comparison()
        print("Ray-method comparison completed.")
        print(f"Comparison CSV: {outputs.csv_path}")
        print(f"Comparison JSON: {outputs.json_path}")
        for figure_path in outputs.figure_paths:
            print(f"Path figure: {figure_path}")
        print(f"Experiment note: {outputs.note_path}")
        return

    if args.command == "benchmark-pseudo-bending-paper":
        outputs = run_pseudo_bending_paper_benchmark()
        print("Pseudo-bending paper benchmark completed.")
        print(f"Benchmark CSV: {outputs.csv_path}")
        print(f"Benchmark JSON: {outputs.json_path}")
        print(f"Ray-path directory: {outputs.ray_path_dir}")
        for figure_path in outputs.figure_paths:
            print(f"Figure: {figure_path}")
        print(f"Experiment note: {outputs.note_path}")
        return

    if args.command == "generate-grid-stations":
        settings = load_settings()
        configuration = generate_grid_stations(settings)
        path = (
            get_repo_root()
            / settings.outputs.generated_base_dir
            / "stations"
            / f"{configuration.configuration_id}.json"
        )
        saved_path = save_station_configuration(configuration, path)
        print(f"Grid station configuration saved: {saved_path}")
        return

    if args.command == "render-all-velocity-scenarios":
        outputs_by_scenario = render_all_velocity_scenarios()
        print("Rendered all configured velocity scenarios.")
        for scenario, outputs in zip(ALL_VELOCITY_SCENARIOS, outputs_by_scenario, strict=True):
            print(
                f"{scenario}: velocity_grid={outputs.velocity_grid}, "
                f"subsurface_figure={outputs.subsurface_figure}"
            )
        return

    if args.command == "render-target-grid-figure":
        settings = load_settings()
        output_path = get_repo_root() / "paper" / "figures" / "frozen_target_grid_structure.png"
        saved_path = save_target_grid_structure_figure(settings, output_path)
        print("Frozen target-grid figure rendered.")
        print(f"PNG: {saved_path}")
        print(f"SVG: {saved_path.with_suffix('.svg')}")
        print(f"PDF: {saved_path.with_suffix('.pdf')}")
        return

    if args.command == "prepare-ml-target-batch":
        outputs = prepare_first_ml_target_batch()
        print("First ML target batch prepared.")
        print(f"Source velocity model: {outputs.source_velocity_model_path}")
        print(f"Target batch manifest: {outputs.manifest_path}")
        for artifact in outputs.scenario_artifacts:
            print(
                f"Scenario {artifact.scenario}: grid={artifact.velocity_grid_path}, "
                f"target_spec={artifact.target_spec_path}"
            )
        return

    if args.command == "prepare-ml-supervised-pairings":
        outputs = prepare_first_ml_supervised_pairings()
        print("First ML supervised pairings prepared.")
        print(f"Pairing manifest: {outputs.manifest_path}")
        for pairing in outputs.pairings:
            print(
                f"Scenario {pairing.scenario}: input_csv={pairing.observation_dataset_csv}, "
                f"input_metadata={pairing.observation_dataset_metadata}, "
                f"target_grid={pairing.target_velocity_grid_path}, target_spec={pairing.target_spec_path}"
            )
        return

    if args.command == "prepare-expanded-ml-supervised-pairings":
        outputs = prepare_expanded_ml_supervised_pairings()
        print("Expanded ML supervised pairings prepared.")
        print(f"Pairing manifest: {outputs.manifest_path}")
        print(f"Paired case count: {len(outputs.pairings)}")
        for pairing in outputs.pairings:
            print(
                f"Case {pairing.case_id} ({pairing.scenario}): input_csv={pairing.observation_dataset_csv}, "
                f"target_grid={pairing.target_velocity_grid_path}"
            )
        return

    if args.command == "prepare-observation-ml-supervised-pairings":
        outputs = prepare_observation_ml_supervised_pairings()
        print("Observation-driven ML supervised pairings prepared.")
        print(f"Pairing manifest: {outputs.manifest_path}")
        print(f"Paired case count: {len(outputs.pairings)}")
        for pairing in outputs.pairings:
            print(
                f"Case {pairing.case_id} ({pairing.scenario}): input_csv={pairing.observation_dataset_csv}, "
                f"target_grid={pairing.target_velocity_grid_path}"
            )
        return

    if args.command == "prepare-paper-pseudo-bending-observation-corpus":
        outputs = prepare_paper_pseudo_bending_observation_corpus()
        print("Paper pseudo-bending observation corpus prepared.")
        print(f"Corpus manifest: {outputs.manifest_path}")
        print(f"Paired case count: {len(outputs.pairings)}")
        for pairing in outputs.pairings:
            print(
                f"Case {pairing.case_id} ({pairing.scenario}): input_csv={pairing.observation_dataset_csv}, "
                f"target_grid={pairing.target_velocity_grid_path}"
            )
        return

    if args.command == "validate-paper-pseudo-bending-observation-corpus":
        result = validate_paper_pseudo_bending_observation_corpus()
        if result.checks:
            print("Paper pseudo-bending corpus validation checks passed:")
            for check in result.checks:
                print(f"- {check}")
        if result.errors:
            print("Paper pseudo-bending corpus validation errors:")
            for error in result.errors:
                print(f"- {error}")
            raise SystemExit(1)
        print("Paper pseudo-bending observation corpus validation completed successfully.")
        return

    if args.command == "train-first-ml-baseline":
        outputs = train_first_ml_baseline()
        print("First ML baseline completed.")
        print(f"Feature matrix: {outputs.feature_matrix_csv}")
        print(f"Metrics JSON: {outputs.metrics_json}")
        print(f"Prediction directory: {outputs.predictions_dir}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "train-feature-distance-ml-baseline":
        outputs = train_feature_distance_ml_baseline()
        print("Feature-distance ML baseline completed.")
        print(f"Feature matrix: {outputs.feature_matrix_csv}")
        print(f"Metrics JSON: {outputs.metrics_json}")
        print(f"Prediction directory: {outputs.predictions_dir}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "train-reduced-target-ml-baseline":
        outputs = train_reduced_target_ridge_ml_baseline()
        print("Reduced-target ML baseline completed.")
        print(f"Feature matrix: {outputs.feature_matrix_csv}")
        print(f"Metrics JSON: {outputs.metrics_json}")
        print(f"Prediction directory: {outputs.predictions_dir}")
        if outputs.model_details_json is not None:
            print(f"Model details JSON: {outputs.model_details_json}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "train-expanded-ml-baseline-suite":
        outputs = train_expanded_ml_baseline_suite()
        print("Expanded ML baseline suite completed.")
        print(f"Mean-target metrics: {outputs.mean_target_metrics_json}")
        print(f"Feature-distance metrics: {outputs.feature_distance_metrics_json}")
        print(f"Reduced-target metrics: {outputs.reduced_target_metrics_json}")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary note: {outputs.summary_note}")
        return

    if args.command == "train-observation-pca-linear-baseline":
        outputs = train_observation_pca_linear_ml_baseline()
        print("Observation-driven PCA linear ML baseline completed.")
        print(f"Feature matrix: {outputs.feature_matrix_csv}")
        print(f"Metrics JSON: {outputs.metrics_json}")
        print(f"Model details JSON: {outputs.model_details_json}")
        print(f"Prediction directory: {outputs.predictions_dir}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "train-observation-pca-ridge-baseline":
        outputs = train_observation_pca_ridge_ml_baseline()
        print("Observation-driven PCA ridge ML baseline completed.")
        print(f"Feature matrix: {outputs.feature_matrix_csv}")
        print(f"Metrics JSON: {outputs.metrics_json}")
        print(f"Model details JSON: {outputs.model_details_json}")
        print(f"Prediction directory: {outputs.predictions_dir}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "train-observation-pca-mlp-baseline":
        outputs = train_observation_pca_mlp_ml_baseline()
        print("Observation-driven PCA MLP ML baseline completed.")
        print(f"Feature matrix: {outputs.feature_matrix_csv}")
        print(f"Metrics JSON: {outputs.metrics_json}")
        print(f"Model details JSON: {outputs.model_details_json}")
        print(f"Prediction directory: {outputs.predictions_dir}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "train-observation-pca-random-forest-baseline":
        outputs = train_observation_pca_random_forest_ml_baseline()
        print("Observation-driven PCA random-forest ML baseline completed.")
        print(f"Feature matrix: {outputs.feature_matrix_csv}")
        print(f"Metrics JSON: {outputs.metrics_json}")
        print(f"Model details JSON: {outputs.model_details_json}")
        print(f"Prediction directory: {outputs.predictions_dir}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "tune-observation-pca-ridge-baseline":
        outputs = tune_observation_pca_ridge_ml_baseline()
        print("Observation-driven PCA ridge tuning completed.")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Candidate metrics JSON: {outputs.candidate_metrics_json}")
        print(f"Experiment note: {outputs.summary_note}")
        return

    if args.command == "train-observation-ml-comparison":
        outputs = train_observation_ml_comparison()
        print("Observation-driven limited ML comparison completed.")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary note: {outputs.summary_note}")
        for output_dir in outputs.model_output_dirs:
            print(f"Model output directory: {output_dir}")
        return

    if args.command == "train-paper-pseudo-bending-ml-suite":
        outputs = train_paper_pseudo_bending_ml_suite()
        print("Paper pseudo-bending ML suite completed.")
        print(f"Suite summary JSON: {outputs.summary_json}")
        print(f"Repeated split summary JSON: {outputs.repeated_summary_json}")
        print(f"Leave-one-family-out summary JSON: {outputs.leave_one_family_out_summary_json}")
        print(f"Noise robustness summary JSON: {outputs.noise_robustness_summary_json}")
        print(f"Summary note: {outputs.summary_note}")
        print(f"Output directory: {outputs.output_dir}")
        return

    if args.command == "train-paper-pseudo-bending-grouped-ml-suite":
        outputs = train_paper_pseudo_bending_grouped_ml_suite()
        print("Paper pseudo-bending target-grouped ML suite completed.")
        print(f"Suite summary JSON: {outputs.summary_json}")
        print(f"Repeated split summary JSON: {outputs.repeated_summary_json}")
        print(f"Leave-one-family-out summary JSON: {outputs.leave_one_family_out_summary_json}")
        print(f"Noise robustness summary JSON: {outputs.noise_robustness_summary_json}")
        print(f"Target-group split assignments: {outputs.target_group_split_csv}")
        print(f"Summary note: {outputs.summary_note}")
        print(f"Output directory: {outputs.output_dir}")
        return

    if args.command == "audit-paper-pseudo-bending-ml-suite":
        outputs = audit_paper_pseudo_bending_ml_suite()
        print("Paper pseudo-bending ML suite audit completed.")
        print(f"Audit summary JSON: {outputs.audit_summary_json}")
        print(f"Audit note: {outputs.audit_note}")
        return

    if args.command == "audit-target-split":
        outputs = run_target_split_audit(
            manifest_path=Path(args.manifest),
            current_split_path=Path(args.current_split),
            output_dir=Path(args.output_dir),
            numerical_atol=args.numerical_atol,
            numerical_rtol=args.numerical_rtol,
        )
        print("Target/split audit completed.")
        print(f"Target case registry: {outputs.target_case_registry_csv}")
        print(f"Target hash summary: {outputs.target_hash_summary_csv}")
        print(f"Cross-split leakage: {outputs.cross_split_target_leakage_csv}")
        print(f"Numerical equivalence pairs: {outputs.numerical_equivalence_pairs_csv}")
        print(f"Audit summary: {outputs.audit_summary_md}")
        print(f"Audit metadata: {outputs.audit_metadata_json}")
        return

    if args.command == "audit-pca-target-representation":
        outputs = run_pca_target_audit(
            metrics_path=Path(args.metrics),
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output_dir),
            configured_component_count=args.components,
            requested_component_counts=_parse_int_sequence(
                args.requested_components,
                "requested-components",
            ),
        )
        print("PCA target representation audit completed.")
        print(f"Component variance CSV: {outputs.component_variance_csv}")
        print(f"Reconstruction case metrics CSV: {outputs.reconstruction_case_metrics_csv}")
        print(f"Reconstruction summary CSV: {outputs.reconstruction_summary_csv}")
        print(f"Audit summary: {outputs.audit_summary_md}")
        print(f"Audit metadata: {outputs.audit_metadata_json}")
        return

    if args.command == "audit-path-length-feature":
        outputs = run_path_length_audit(
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output_dir),
        )
        print("Path-length feature audit completed.")
        print(f"Case summary CSV: {outputs.case_summary_csv}")
        print(f"Sample diagnostics CSV: {outputs.sample_diagnostics_csv}")
        print(f"Audit summary: {outputs.audit_summary_md}")
        print(f"Audit metadata: {outputs.audit_metadata_json}")
        return

    if args.command == "audit-fair-reference-baseline":
        outputs = run_fair_reference_ray_baseline(
            sensitivity_case_metrics_path=Path(args.sensitivity_case_metrics),
            grouped_split_path=Path(args.grouped_split),
            ablation_metrics_path=Path(args.ablation_metrics),
            output_dir=Path(args.output_dir),
        )
        print("Fair reference-ray baseline selection completed.")
        print(f"Validation parameter grid: {outputs.parameter_grid_csv}")
        print(f"Frozen case metrics: {outputs.case_metrics_csv}")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary Markdown: {outputs.summary_md}")
        return

    if args.command == "audit-coverage-aware-evaluation":
        outputs = run_coverage_aware_evaluation(
            manifest_path=Path(args.manifest),
            grouped_split_path=Path(args.grouped_split),
            ml_metrics_path=Path(args.ml_metrics),
            classical_corpus_dir=Path(args.classical_corpus_dir),
            output_dir=Path(args.output_dir),
            thresholds_km=_parse_float_sequence(args.thresholds_km, "thresholds-km"),
            method_ids=tuple(
                value.strip() for value in args.methods.split(",") if value.strip()
            ),
        )
        print("Coverage-aware grouped-test evaluation completed.")
        print(f"Case metrics CSV: {outputs.case_metrics_csv}")
        print(f"Depth metrics CSV: {outputs.depth_metrics_csv}")
        print(f"Coverage distribution CSV: {outputs.distribution_csv}")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary Markdown: {outputs.summary_md}")
        return

    if args.command == "audit-structural-recovery":
        outputs = run_structural_recovery_metrics(
            manifest_path=Path(args.manifest),
            grouped_split_path=Path(args.grouped_split),
            ml_metrics_path=Path(args.ml_metrics),
            classical_corpus_dir=Path(args.classical_corpus_dir),
            output_dir=Path(args.output_dir),
            method_ids=tuple(
                value.strip() for value in args.methods.split(",") if value.strip()
            ),
        )
        print("Structure-aware grouped-test evaluation completed.")
        print(f"Case metrics CSV: {outputs.case_metrics_csv}")
        print(f"Layer metrics CSV: {outputs.layer_metrics_csv}")
        print(f"Summary CSV: {outputs.summary_csv}")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary Markdown: {outputs.summary_md}")
        return

    if args.command == "audit-forward-solver":
        outputs = run_forward_solver_validation(
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output_dir),
        )
        print("Forward-solver validation and convergence audit completed.")
        print(f"Validation case metrics CSV: {outputs.validation_case_metrics_csv}")
        print(
            "Corpus convergence summary CSV: "
            f"{outputs.corpus_convergence_case_summary_csv}"
        )
        print(f"Solver failure records CSV: {outputs.corpus_solver_failure_records_csv}")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary Markdown: {outputs.summary_md}")
        print(f"Audit metadata: {outputs.audit_metadata_json}")
        return

    if args.command == "audit-forward-solver-failures":
        outputs = run_forward_solver_failure_diagnostics(
            manifest_path=Path(args.manifest),
            failure_input_path=Path(args.failure_input),
            output_dir=Path(args.output_dir),
        )
        print("Forward-solver failure diagnosis completed.")
        print(f"Observation diagnostics: {outputs.observation_diagnostics_csv}")
        print(f"Failure diagnostics: {outputs.failure_diagnostics_csv}")
        print(f"Failure summary: {outputs.failure_summary_csv}")
        print(f"Strategy results: {outputs.strategy_results_csv}")
        print(f"Strategy summary: {outputs.strategy_summary_csv}")
        print(f"Summary JSON: {outputs.summary_json}")
        return

    if args.command == "audit-forward-solver-v2":
        outputs = run_forward_solver_validation_v2(
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output_dir),
            heterogeneous_cases_per_family=args.heterogeneous_cases_per_family,
            pairs_per_case=args.pairs_per_case,
            resolution_pairs_per_family=args.resolution_pairs_per_family,
        )
        print("Expanded forward-solver v2 validation completed.")
        print(f"Pair metrics: {outputs.validation_pair_metrics_csv}")
        print(f"Validation statistics: {outputs.validation_statistics_csv}")
        print(f"Resolution study: {outputs.resolution_study_csv}")
        print(f"Discrepancy plot: {outputs.discrepancy_plot}")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary Markdown: {outputs.summary_md}")
        print(f"Scientific assessment: {outputs.scientific_assessment_md}")
        print(f"Audit metadata: {outputs.audit_metadata_json}")
        return

    if args.command == "audit-grouped-split-noise-robustness":
        outputs = run_grouped_split_noise_robustness(
            manifest_path=Path(args.manifest),
            grouped_split_path=Path(args.grouped_split),
            classical_corpus_dir=Path(args.classical_corpus_dir),
            output_dir=Path(args.output_dir),
            noise_levels_s=_parse_float_sequence(args.noise_levels, "noise-levels"),
            noise_random_seed=args.noise_seed,
            reference_damping=args.reference_damping,
            reference_smoothing=args.reference_smoothing,
        )
        print("Corrected grouped-split noise robustness audit completed.")
        print(f"Case metrics CSV: {outputs.case_metrics_csv}")
        print(f"Summary CSV: {outputs.summary_csv}")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary Markdown: {outputs.summary_md}")
        print(f"Audit metadata: {outputs.metadata_json}")
        return

    if args.command == "run-grouped-split-paired-comparison":
        outputs = run_grouped_split_paired_comparison(
            output_dir=Path(args.output_dir),
            ml_metrics_path=Path(args.ml_metrics),
            ablation_metrics_path=Path(args.ablation_metrics),
            fair_baseline_path=Path(args.fair_baseline),
            known_ray_path=Path(args.known_ray),
            grouped_split_path=Path(args.grouped_split),
        )
        print("Corrected grouped-split paired comparison completed.")
        print(f"Case metrics CSV: {outputs.paired_test_case_metrics_csv}")
        print(f"Summary CSV: {outputs.paired_test_summary_csv}")
        print(f"Summary JSON: {outputs.paired_test_summary_json}")
        print(f"Fixed test IDs: {outputs.fixed_test_case_ids_json}")
        print(f"Family summary CSV: {outputs.paired_family_summary_csv}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "run-grouped-split-uncertainty-report":
        outputs = run_grouped_split_uncertainty_report(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            permutation_resamples=args.permutation_resamples,
            permutation_seed=args.permutation_seed,
        )
        print("Corrected grouped-split uncertainty report completed.")
        print(f"Method summary CSV: {outputs.method_uncertainty_summary_csv}")
        print(f"Paired delta CSV: {outputs.paired_delta_summary_csv}")
        print(f"Paired case deltas: {outputs.paired_case_deltas_csv}")
        print(f"Bootstrap metadata: {outputs.bootstrap_samples_metadata_json}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "run-target-level-uncertainty-report":
        outputs = run_target_level_uncertainty_report(
            input_dir=Path(args.input_dir),
            paired_deltas_dir=Path(args.paired_deltas_dir),
            grouped_split_path=Path(args.grouped_split),
            output_dir=Path(args.output_dir),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            permutation_resamples=args.permutation_resamples,
            permutation_seed=args.permutation_seed,
        )
        print("Target-level grouped-test uncertainty report completed.")
        print(f"Acquisition case metrics: {outputs.acquisition_case_metrics_csv}")
        print(f"Target-level metrics: {outputs.target_level_metrics_csv}")
        print(f"Target-level paired deltas: {outputs.target_level_paired_deltas_csv}")
        print(f"Summary Markdown: {outputs.summary_md}")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Audit metadata: {outputs.metadata_json}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "train-paper-pseudo-bending-ml-strengthened":
        outputs = train_paper_pseudo_bending_ml_strengthened()
        print("Paper pseudo-bending ML strengthened experiment completed.")
        print(f"Strengthened summary JSON: {outputs.summary_json}")
        print(f"Summary note: {outputs.summary_note}")
        print(f"Output directory: {outputs.output_dir}")
        return

    if args.command == "train-paper-pseudo-bending-faulted-generalization":
        outputs = train_paper_pseudo_bending_faulted_generalization()
        print("Paper pseudo-bending faulted generalization experiment completed.")
        print(f"Diagnosis JSON: {outputs.diagnosis_json}")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary note: {outputs.summary_note}")
        print(f"Output directory: {outputs.output_dir}")
        return

    if args.command == "train-phase7-observation-signal-ablation":
        outputs = train_phase7_observation_signal_ablation_suite()
        print("Phase 7 observation-signal ablation suite completed.")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary CSV: {outputs.summary_csv}")
        print(f"Cell-centered metrics CSV: {outputs.cell_centered_metrics_csv}")
        print(f"Leave-one-family-out summary JSON: {outputs.leave_one_family_out_summary_json}")
        print(f"Summary note: {outputs.summary_note}")
        print(f"Output directory: {outputs.output_dir}")
        return

    if args.command == "train-phase9-grouped-split-ablation":
        outputs = train_phase9_grouped_split_ablation_suite()
        print("Target-grouped observation-signal ablation completed.")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary CSV: {outputs.summary_csv}")
        print(f"Cell-centered metrics CSV: {outputs.cell_centered_metrics_csv}")
        print(f"Leave-one-family-out summary JSON: {outputs.leave_one_family_out_summary_json}")
        print(f"Summary note: {outputs.summary_note}")
        print(f"Output directory: {outputs.output_dir}")
        return

    if args.command == "train-phase7-learning-curve":
        outputs = train_phase7_learning_curve(
            training_sizes=_parse_int_sequence(args.training_sizes, "training-sizes"),
            repetition_seeds=_parse_int_sequence(args.repetition_seeds, "repetition-seeds"),
            output_dir=Path(args.output_dir),
        )
        print("Phase 7 ML learning-curve experiment completed.")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary CSV: {outputs.summary_csv}")
        print(f"Repetition metrics CSV: {outputs.repetition_metrics_csv}")
        print(f"Family-wise CSV: {outputs.family_wise_csv}")
        print(f"Output directory: {outputs.output_dir}")
        return

    if args.command == "train-phase9-grouped-split-learning-curve":
        outputs = train_phase9_grouped_split_learning_curve(
            training_sizes=_parse_int_sequence(args.training_sizes, "training-sizes"),
            repetition_seeds=_parse_int_sequence(
                args.repetition_seeds,
                "repetition-seeds",
            ),
            output_dir=Path(args.output_dir),
        )
        print("Corrected grouped-split learning-curve experiment completed.")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary CSV: {outputs.summary_csv}")
        print(f"Repetition metrics CSV: {outputs.repetition_metrics_csv}")
        print(f"Family-wise CSV: {outputs.family_wise_csv}")
        print(f"Output directory: {outputs.output_dir}")
        return

    if args.command == "package-paper-candidate-v1":
        outputs = package_paper_candidate_v1()
        print("Paper candidate package completed.")
        print(f"Package directory: {outputs.output_dir}")
        print(f"Artifact index JSON: {outputs.artifact_index_json}")
        print(f"Package summary: {outputs.summary_markdown}")
        print(f"Experiment note: {outputs.experiment_note}")
        for table_path in outputs.table_paths:
            print(f"Table: {table_path}")
        for figure_path in outputs.figure_paths:
            print(f"Figure: {figure_path}")
        return

    if args.command == "package-paper-candidate-v2":
        outputs = package_paper_candidate_v2()
        print("Paper candidate v2 package completed.")
        print(f"Package directory: {outputs.output_dir}")
        print(f"Artifact index JSON: {outputs.artifact_index_json}")
        print(f"Package summary: {outputs.summary_markdown}")
        print(f"Experiment note: {outputs.experiment_note}")
        print(f"Figure metadata: {outputs.figure_metadata_json}")
        for table_path in outputs.table_paths:
            print(f"Table: {table_path}")
        for figure_path in outputs.figure_paths:
            print(f"Figure: {figure_path}")
        return

    if args.command == "package-paper-candidate-v3":
        outputs = package_paper_candidate_v3()
        print("Paper candidate v3 package completed.")
        print(f"Package directory: {outputs.output_dir}")
        print(f"Artifact index JSON: {outputs.artifact_index_json}")
        print(f"Package summary: {outputs.summary_markdown}")
        print(f"Experiment note: {outputs.experiment_note}")
        for table_path in outputs.table_paths:
            print(f"Table: {table_path}")
        for figure_path in outputs.figure_references:
            print(f"Referenced figure: {figure_path}")
        return

    if args.command == "package-paper-candidate-v4":
        outputs = package_paper_candidate_v4(output_dir=Path(args.output_dir))
        print("Paper candidate v4 package completed.")
        print(f"Package directory: {outputs.output_dir}")
        print(f"Artifact index JSON: {outputs.artifact_index_json}")
        print(f"Package summary: {outputs.summary_markdown}")
        print(f"Final claims table: {outputs.final_claims_markdown}")
        print(f"Paired method table: {outputs.paired_method_markdown}")
        print(f"Paired family table: {outputs.paired_family_markdown}")
        print(f"Uncertainty method table: {outputs.uncertainty_method_markdown}")
        print(f"Paired delta table: {outputs.paired_delta_markdown}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "run-phase8-paired-test-comparison":
        outputs = run_phase8_paired_test_comparison(
            output_dir=Path(args.output_dir),
            selected_ml_metrics_path=Path(args.selected_ml_metrics),
            include_reference_sensitivity_audit=args.include_reference_sensitivity_audit,
        )
        print("Phase 8A paired same-test-set comparison completed.")
        print(f"Output directory: {outputs.output_dir}")
        print(f"Fixed test IDs: {outputs.fixed_test_case_ids_json}")
        print(f"Case metrics CSV: {outputs.paired_test_case_metrics_csv}")
        print(f"Summary CSV: {outputs.paired_test_summary_csv}")
        print(f"Summary JSON: {outputs.paired_test_summary_json}")
        print(f"Family summary CSV: {outputs.paired_family_summary_csv}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "run-phase8-uncertainty-report":
        outputs = run_phase8_uncertainty_report(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            permutation_resamples=args.permutation_resamples,
            permutation_seed=args.permutation_seed,
            include_travel_time_only_reference=not args.exclude_travel_time_only_reference,
        )
        print("Phase 8B uncertainty and paired statistical report completed.")
        print(f"Output directory: {outputs.output_dir}")
        print(f"Method summary CSV: {outputs.method_uncertainty_summary_csv}")
        print(f"Method summary JSON: {outputs.method_uncertainty_summary_json}")
        print(f"Paired delta CSV: {outputs.paired_delta_summary_csv}")
        print(f"Paired delta JSON: {outputs.paired_delta_summary_json}")
        print(f"Paired case deltas: {outputs.paired_case_deltas_csv}")
        print(f"Bootstrap metadata: {outputs.bootstrap_samples_metadata_json}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "package-submission-reproducibility":
        outputs = run_submission_reproducibility_package(
            output_dir=Path(args.output_dir),
        )
        print("Submission reproducibility package completed.")
        print(f"Output directory: {outputs.output_dir}")
        print(f"Manifest JSON: {outputs.manifest_json}")
        print(f"Manifest Markdown: {outputs.manifest_md}")
        print(f"Commands: {outputs.commands_md}")
        print(f"Artifact map: {outputs.artifact_csv}")
        print(f"Data availability draft: {outputs.data_draft}")
        print(f"Software availability draft: {outputs.software_draft}")
        print(f"Environment summary: {outputs.environment_json}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    if args.command == "run-known-ray-inversion-diagnostic":
        settings = load_settings()
        paths = _build_output_paths(settings, get_repo_root())
        output_dir = (
            get_repo_root()
            / settings.outputs.generated_base_dir
            / "classical_tomography"
            / "known_ray_dls_v1"
            / settings.vertical_slice.experiment_id
        )
        outputs = run_known_ray_inversion_diagnostic(
            sensitivity_path=paths.sensitivity_jsonl,
            sensitivity_metadata_path=paths.sensitivity_metadata,
            observations_path=paths.travel_times_json,
            velocity_grid_path=paths.velocity_grid,
            output_dir=output_dir,
        )
        print("Known-ray inversion diagnostic completed.")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Cell velocity CSV: {outputs.cell_velocity_csv}")
        print(f"Predicted travel times CSV: {outputs.predicted_travel_times_csv}")
        return

    if args.command == "run-coverage-aware-known-ray-inversion":
        settings = load_settings()
        paths = _build_output_paths(settings, get_repo_root())
        output_dir = (
            get_repo_root()
            / settings.outputs.generated_base_dir
            / "classical_tomography"
            / "known_ray_coverage_smoothing_v1"
            / settings.vertical_slice.experiment_id
        )
        outputs = run_coverage_aware_known_ray_inversion_diagnostic(
            sensitivity_path=paths.sensitivity_jsonl,
            sensitivity_metadata_path=paths.sensitivity_metadata,
            observations_path=paths.travel_times_json,
            velocity_grid_path=paths.velocity_grid,
            output_dir=output_dir,
        )
        print("Coverage-aware known-ray inversion diagnostic completed.")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Coverage CSV: {outputs.coverage_csv}")
        print(f"Cell velocity CSV: {outputs.cell_velocity_csv}")
        print(f"Predicted travel times CSV: {outputs.predicted_travel_times_csv}")
        return

    if args.command == "run-known-ray-perturbation-corpus":
        outputs = run_known_ray_perturbation_corpus_inversion(
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output_dir),
            reference_ray_corpus_dir=Path(args.reference_ray_corpus_dir),
            legacy_case_summary_path=Path(args.legacy_case_summary),
            damping=args.damping,
            smoothing=args.smoothing,
            minimum_coverage_km=args.minimum_coverage_km,
        )
        print("Corrected known-ray perturbation corpus diagnostic completed.")
        print(f"Case summary CSV: {outputs.case_summary_csv}")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary Markdown: {outputs.summary_md}")
        return

    if args.command == "run-known-ray-corpus-inversion":
        outputs = run_known_ray_corpus_inversion(
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output_dir),
            ml_suite_summary_path=Path(args.ml_suite_summary)
            if args.ml_suite_summary
            else None,
            damping_values=_parse_float_sequence(args.damping_values, "damping-values"),
            smoothing_values=_parse_float_sequence(
                args.smoothing_values,
                "smoothing-values",
            ),
            minimum_coverage_km=args.minimum_coverage_km,
        )
        print("Known-ray corpus inversion diagnostic completed.")
        print(f"Case summary CSV: {outputs.case_summary_csv}")
        print(f"Corpus summary CSV: {outputs.summary_csv}")
        print(f"Corpus summary JSON: {outputs.summary_json}")
        print(f"Comparison JSON: {outputs.comparison_json}")
        print(f"Comparison CSV: {outputs.comparison_csv}")
        return

    if args.command == "run-reference-ray-corpus-inversion":
        outputs = run_reference_ray_corpus_inversion(
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output_dir),
            known_ray_summary_path=Path(args.known_ray_summary)
            if args.known_ray_summary
            else None,
            ml_suite_summary_path=Path(args.ml_suite_summary)
            if args.ml_suite_summary
            else None,
            damping_values=_parse_float_sequence(args.damping_values, "damping-values"),
            smoothing_values=_parse_float_sequence(
                args.smoothing_values,
                "smoothing-values",
            ),
            minimum_coverage_km=args.minimum_coverage_km,
        )
        print("Reference-ray corpus inversion diagnostic completed.")
        print(f"Case summary CSV: {outputs.case_summary_csv}")
        print(f"Corpus summary CSV: {outputs.summary_csv}")
        print(f"Corpus summary JSON: {outputs.summary_json}")
        print(f"Comparison JSON: {outputs.comparison_json}")
        print(f"Comparison CSV: {outputs.comparison_csv}")
        return

    if args.command == "run-final-phase6-comparison":
        outputs = run_final_phase6_comparison(
            output_dir=Path(args.output_dir),
            known_ray_summary_path=Path(args.known_ray_summary),
            known_ray_case_summary_path=Path(args.known_ray_case_summary),
            reference_ray_summary_path=Path(args.reference_ray_summary),
            reference_ray_case_summary_path=Path(args.reference_ray_case_summary),
            ml_metrics_path=Path(args.ml_metrics),
        )
        print("Final Phase 6 comparison audit completed.")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary CSV: {outputs.summary_csv}")
        print(f"ML cell-centered case metrics CSV: {outputs.ml_case_metrics_csv}")
        print(f"Classical clipping diagnostics CSV: {outputs.clipping_csv}")
        return

    if args.command == "run-reference-ray-regularization-sensitivity":
        outputs = run_reference_ray_regularization_sensitivity(
            manifest_path=Path(args.manifest),
            reference_case_summary_path=Path(args.reference_case_summary),
            output_dir=Path(args.output_dir),
            damping_values=_parse_float_sequence(args.damping_values, "damping-values"),
            smoothing_values=_parse_float_sequence(
                args.smoothing_values,
                "smoothing-values",
            ),
            minimum_coverage_km=args.minimum_coverage_km,
            case_limit=args.case_limit,
        )
        print("Reference-ray regularization sensitivity audit completed.")
        print(f"Summary JSON: {outputs.summary_json}")
        print(f"Summary CSV: {outputs.summary_csv}")
        print(f"Case metrics CSV: {outputs.case_metrics_csv}")
        print(f"Family-wise CSV: {outputs.family_wise_csv}")
        print(f"Tradeoff CSV: {outputs.tradeoff_csv}")
        print(f"Experiment note: {outputs.experiment_note}")
        return

    print(f"tomobench is ready. Central config: {get_config_path()}")
    print("Run the first vertical slice with: tomobench run-first-slice")
    print("Run the layered simulator with: tomobench run-first-slice --simulator layered-ray")
    print("Run the eikonal prototype with: tomobench run-first-slice --simulator eikonal-3d")
    print(
        "Run the pseudo-bending tracer with: tomobench run-first-slice --simulator pseudo-bending"
    )
    print(
        "Run the block anomaly grid with: tomobench run-first-slice --velocity-scenario block-anomaly"
    )
    print("Run the faulted grid with: tomobench run-first-slice --velocity-scenario faulted")
    print("Run the salt-dome grid with: tomobench run-first-slice --velocity-scenario salt-dome")
    print(
        "Run the dyke-intrusion grid with: tomobench run-first-slice --velocity-scenario dyke-intrusion"
    )
    print("Render all five current scenarios with: tomobench render-all-velocity-scenarios")
    print("Render the frozen target-grid figure with: tomobench render-target-grid-figure")
    print("Compare simulators with: tomobench compare-simulators")
    print("Compare Dijkstra and pseudo-bending rays with: tomobench compare-ray-methods")
    print("Run the pseudo-bending paper benchmark with: tomobench benchmark-pseudo-bending-paper")
    print(
        "Run the Phase 6 known-ray inversion diagnostic with: "
        "tomobench run-known-ray-inversion-diagnostic"
    )
    print(
        "Run the Phase 6 coverage-aware known-ray inversion with: "
        "tomobench run-coverage-aware-known-ray-inversion"
    )
    print(
        "Run the Phase 6 known-ray corpus diagnostic with: "
        "tomobench run-known-ray-corpus-inversion"
    )
    print(
        "Run the Phase 6 reference-ray corpus diagnostic with: "
        "tomobench run-reference-ray-corpus-inversion"
    )
    print(
        "Run the final Phase 6 comparison audit with: "
        "tomobench run-final-phase6-comparison"
    )
    print(
        "Run the Phase 7B reference-ray sensitivity audit with: "
        "tomobench run-reference-ray-regularization-sensitivity"
    )
    print("Validate generated artifacts with: tomobench validate-first-slice")
    print("Benchmark the eikonal prototype with: tomobench benchmark-eikonal")
    print("Generate a grid station artifact with: tomobench generate-grid-stations")
    print("Prepare the first ML target batch with: tomobench prepare-ml-target-batch")
    print(
        "Prepare the first supervised ML pairings with: tomobench prepare-ml-supervised-pairings"
    )
    print(
        "Prepare expanded ML supervised pairings with: "
        "tomobench prepare-expanded-ml-supervised-pairings"
    )
    print(
        "Prepare observation-driven ML supervised pairings with: "
        "tomobench prepare-observation-ml-supervised-pairings"
    )
    print(
        "Prepare the paper pseudo-bending observation corpus with: "
        "tomobench prepare-paper-pseudo-bending-observation-corpus"
    )
    print(
        "Validate the paper pseudo-bending observation corpus with: "
        "tomobench validate-paper-pseudo-bending-observation-corpus"
    )
    print("Train the first ML baseline with: tomobench train-first-ml-baseline")
    print(
        "Train the feature-distance ML baseline with: "
        "tomobench train-feature-distance-ml-baseline"
    )
    print("Train the reduced-target ML baseline with: tomobench train-reduced-target-ml-baseline")
    print("Train the expanded ML baseline suite with: tomobench train-expanded-ml-baseline-suite")
    print(
        "Train the observation-driven PCA linear baseline with: "
        "tomobench train-observation-pca-linear-baseline"
    )
    print(
        "Train the observation-driven PCA ridge baseline with: "
        "tomobench train-observation-pca-ridge-baseline"
    )
    print(
        "Train the observation-driven PCA MLP baseline with: "
        "tomobench train-observation-pca-mlp-baseline"
    )
    print(
        "Train the observation-driven PCA random-forest baseline with: "
        "tomobench train-observation-pca-random-forest-baseline"
    )
    print(
        "Tune the observation-driven PCA ridge baseline with: "
        "tomobench tune-observation-pca-ridge-baseline"
    )
    print(
        "Run the observation-driven limited ML comparison with: "
        "tomobench train-observation-ml-comparison"
    )
    print(
        "Run the paper pseudo-bending ML suite with: "
        "tomobench train-paper-pseudo-bending-ml-suite"
    )
    print(
        "Audit the paper pseudo-bending ML suite with: "
        "tomobench audit-paper-pseudo-bending-ml-suite"
    )
    print("Audit target duplication and split overlap with: tomobench audit-target-split")
    print(
        "Run the strengthened paper pseudo-bending ML experiment with: "
        "tomobench train-paper-pseudo-bending-ml-strengthened"
    )
    print(
        "Run the Phase 7 observation-signal ablation with: "
        "tomobench train-phase7-observation-signal-ablation"
    )
    print(
        "Run the Phase 7 ML learning curve with: "
        "tomobench train-phase7-learning-curve"
    )
    print("Package the final paper candidate with: tomobench package-paper-candidate-v1")
    print("Package the Phase 6 paper candidate with: tomobench package-paper-candidate-v2")
    print(
        "Package the Phase 6/7 paper candidate with: "
        "tomobench package-paper-candidate-v3"
    )


def _parse_float_sequence(value: str, label: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise SystemExit(f"Invalid --{label}: {value}") from error
    if not parsed:
        raise SystemExit(f"--{label} must contain at least one numeric value.")
    return parsed


def _parse_int_sequence(value: str, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise SystemExit(f"Invalid --{label}: {value}") from error
    if not parsed:
        raise SystemExit(f"--{label} must contain at least one integer value.")
    return parsed


def _settings_from_cli_overrides(
    simulator: str | None,
    velocity_scenario: str | None,
):
    settings = load_settings()
    if simulator is not None:
        settings = replace(
            settings,
            simulation=replace(
                settings.simulation,
                method=_normalize_simulation_method(simulator),
            ),
        )
    if velocity_scenario is not None:
        settings = replace(
            settings,
            velocity_model_generation=replace(
                settings.velocity_model_generation,
                default_scenario=_normalize_velocity_scenario(velocity_scenario),
            ),
        )
    return settings
