"""Build a submission-readiness reproducibility manifest for the study."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from tomobench.utils.paths import get_repo_root

REPRODUCIBILITY_MANIFEST_ID = "reproducibility_manifest_v1"
DEFAULT_REPRODUCIBILITY_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/reproducibility_manifest_v1"
)
DEFAULT_CONFIG_PATH = Path("config/benchmark_config.yaml")
DEFAULT_PYPROJECT_PATH = Path("code/pyproject.toml")
DEFAULT_LOCKFILE_PATH = Path("code/uv.lock")


@dataclass(frozen=True)
class SubmissionReproducibilityOutputs:
    """Paths written by the Phase 8C reproducibility package."""

    output_dir: Path
    manifest_json: Path
    manifest_md: Path
    commands_md: Path
    artifact_csv: Path
    data_draft: Path
    software_draft: Path
    environment_json: Path
    experiment_note: Path

_ARTIFACT_SPECS: tuple[dict[str, str], ...] = (
    {
        "artifact_id": "central_config",
        "path": "config/benchmark_config.yaml",
        "artifact_type": "config",
        "phase": "configuration",
        "role": "source",
        "description": "Central scientific, dataset, model, split, and random-seed configuration.",
    },
    {
        "artifact_id": "python_project_definition",
        "path": "code/pyproject.toml",
        "artifact_type": "dependency_definition",
        "phase": "environment",
        "role": "source",
        "description": "Python project metadata and direct runtime/development dependencies.",
    },
    {
        "artifact_id": "uv_lockfile",
        "path": "code/uv.lock",
        "artifact_type": "dependency_lockfile",
        "phase": "environment",
        "role": "source",
        "description": "Resolved dependency lockfile used by the uv environment workflow.",
    },
    {
        "artifact_id": "cli_source",
        "path": "code/src/tomobench/cli.py",
        "artifact_type": "source_code",
        "phase": "software",
        "role": "source",
        "description": "CLI definitions for data generation, tomography, ML, audit, and packaging workflows.",
    },
    {
        "artifact_id": "phase8a_source",
        "path": "code/src/tomobench/evaluation/phase8_paired_test_comparison.py",
        "artifact_type": "source_code",
        "phase": "phase8a",
        "role": "source",
        "description": "Exact fixed-test-case resolution and cell-centered paired comparison implementation.",
    },
    {
        "artifact_id": "phase8b_source",
        "path": "code/src/tomobench/evaluation/phase8_uncertainty_report.py",
        "artifact_type": "source_code",
        "phase": "phase8b",
        "role": "source",
        "description": "Deterministic bootstrap, paired delta, and exploratory sign-flip reporting implementation.",
    },
    {
        "artifact_id": "phase8a_tests",
        "path": "code/tests/unit/test_phase8_paired_test_comparison.py",
        "artifact_type": "test_code",
        "phase": "phase8a",
        "role": "verification",
        "description": "Focused tests for fixed IDs, exact joins, metric contract, and Phase 8A aggregation.",
    },
    {
        "artifact_id": "phase8b_tests",
        "path": "code/tests/unit/test_phase8_uncertainty_report.py",
        "artifact_type": "test_code",
        "phase": "phase8b",
        "role": "verification",
        "description": "Focused tests for bootstrap determinism, paired aggregation, and sign-flip behavior.",
    },
    {
        "artifact_id": "paper_candidate_v3_directory",
        "path": "outputs/generated/paper_candidate_v3",
        "artifact_type": "generated_directory",
        "phase": "paper_candidate_v3",
        "role": "evidence",
        "description": "Final Phase 6/7 manuscript evidence package and artifact index.",
    },
    {
        "artifact_id": "paper_candidate_v3_artifact_index",
        "path": "outputs/generated/paper_candidate_v3/artifact_index.json",
        "artifact_type": "generated_manifest",
        "phase": "paper_candidate_v3",
        "role": "evidence",
        "description": "Traceability index for the preserved paper candidate v3 evidence.",
    },
    {
        "artifact_id": "paper_candidate_v3_summary",
        "path": "outputs/generated/paper_candidate_v3/notes/package_summary.md",
        "artifact_type": "generated_documentation",
        "phase": "paper_candidate_v3",
        "role": "documentation",
        "description": "Human-readable paper candidate v3 scope, evidence, and claim boundaries.",
    },
    {
        "artifact_id": "phase6_final_comparison_directory",
        "path": "outputs/generated/classical_tomography/final_phase6_comparison_v1",
        "artifact_type": "generated_directory",
        "phase": "phase6",
        "role": "evidence",
        "description": "Final Phase 6 metric-contract-aware classical-versus-ML comparison artifacts.",
    },
    {
        "artifact_id": "phase6_known_ray_directory",
        "path": "outputs/generated/classical_tomography/known_ray_corpus_v1",
        "artifact_type": "generated_directory",
        "phase": "phase6",
        "role": "evidence",
        "description": "Known-ray corpus inversion diagnostic outputs and per-case metrics.",
    },
    {
        "artifact_id": "phase6_reference_ray_directory",
        "path": "outputs/generated/classical_tomography/reference_ray_corpus_v1",
        "artifact_type": "generated_directory",
        "phase": "phase6",
        "role": "evidence",
        "description": "Reference-model fixed-ray regularized inversion outputs and per-case metrics.",
    },
    {
        "artifact_id": "phase7a_directory",
        "path": "outputs/generated/ml_baselines/phase7_observation_signal_ablation_v1",
        "artifact_type": "generated_directory",
        "phase": "phase7a",
        "role": "evidence",
        "description": "Observation-signal ablation outputs, cell-centered metrics, and controls.",
    },
    {
        "artifact_id": "phase7b_directory",
        "path": "outputs/generated/classical_tomography/reference_ray_regularization_sensitivity_v1",
        "artifact_type": "generated_directory",
        "phase": "phase7b",
        "role": "evidence",
        "description": "Bounded reference-ray damping, smoothing, clipping, and sensitivity audit.",
    },
    {
        "artifact_id": "phase7c_directory",
        "path": "outputs/generated/ml_baselines/phase7_learning_curve_v1",
        "artifact_type": "generated_directory",
        "phase": "phase7c",
        "role": "evidence",
        "description": "Fixed-holdout learning-curve outputs and training-size diagnostics.",
    },
    {
        "artifact_id": "phase8a_directory",
        "path": "outputs/generated/submission_readiness/phase8_paired_test_comparison_v1",
        "artifact_type": "generated_directory",
        "phase": "phase8a",
        "role": "evidence",
        "description": "Paired same-test-set metrics, exact fixed test IDs, and method/family summaries.",
    },
    {
        "artifact_id": "phase8a_summary",
        "path": "outputs/generated/submission_readiness/phase8_paired_test_comparison_v1/paired_test_summary.json",
        "artifact_type": "generated_summary",
        "phase": "phase8a",
        "role": "evidence",
        "description": "Phase 8A methods, ranking, metric contract, and regularization policy.",
    },
    {
        "artifact_id": "phase8a_case_metrics",
        "path": "outputs/generated/submission_readiness/phase8_paired_test_comparison_v1/paired_test_case_metrics.csv",
        "artifact_type": "generated_case_metrics",
        "phase": "phase8a",
        "role": "evidence",
        "description": "175 paired per-case rows across seven methods and 25 fixed test cases.",
    },
    {
        "artifact_id": "phase8b_directory",
        "path": "outputs/generated/submission_readiness/phase8_uncertainty_report_v1",
        "artifact_type": "generated_directory",
        "phase": "phase8b",
        "role": "evidence",
        "description": "Bootstrap uncertainty, paired deltas, sign-flip tests, and reproducibility metadata.",
    },
    {
        "artifact_id": "phase8b_method_summary",
        "path": "outputs/generated/submission_readiness/phase8_uncertainty_report_v1/method_uncertainty_summary.json",
        "artifact_type": "generated_summary",
        "phase": "phase8b",
        "role": "evidence",
        "description": "Per-method mean, standard deviation, median, IQR, and bootstrap CIs.",
    },
    {
        "artifact_id": "phase8b_delta_summary",
        "path": "outputs/generated/submission_readiness/phase8_uncertainty_report_v1/paired_delta_summary.json",
        "artifact_type": "generated_summary",
        "phase": "phase8b",
        "role": "evidence",
        "description": "Paired method deltas, CIs, wins/losses/ties, and exploratory p-values.",
    },
    {
        "artifact_id": "synthetic_observation_corpus",
        "path": "outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1",
        "artifact_type": "generated_dataset_directory",
        "phase": "dataset",
        "role": "input_data",
        "description": "Synthetic five-family pseudo-bending observation/target corpus with traceable sidecars.",
    },
    {
        "artifact_id": "phase8a_experiment_note",
        "path": "wiki/experiments/phase8_paired_test_comparison_v1.md",
        "artifact_type": "documentation",
        "phase": "phase8a",
        "role": "documentation",
        "description": "Phase 8A protocol and scientific claim boundaries.",
    },
    {
        "artifact_id": "phase8b_experiment_note",
        "path": "wiki/experiments/phase8_uncertainty_report_v1.md",
        "artifact_type": "documentation",
        "phase": "phase8b",
        "role": "documentation",
        "description": "Phase 8B uncertainty protocol, results, and interpretation caveats.",
    },
    {
        "artifact_id": "paper_candidate_v3_experiment_note",
        "path": "wiki/experiments/paper_candidate_v3.md",
        "artifact_type": "documentation",
        "phase": "paper_candidate_v3",
        "role": "documentation",
        "description": "Paper candidate v3 package experiment note and claim boundaries.",
    },
    {
        "artifact_id": "task_tracking",
        "path": "tasks/current.md",
        "artifact_type": "project_documentation",
        "phase": "project",
        "role": "documentation",
        "description": "Current implementation and review tracking for the project workspace.",
    },
)


def build_command_plan() -> list[dict[str, Any]]:
    """Return the dependency-ordered commands exposed by the current CLI."""
    commands = [
        (
            "environment_sync",
            "environment",
            "uv sync --group dev",
            "Resolve the locked Python environment from code/uv.lock.",
            ["code/pyproject.toml", "code/uv.lock"],
        ),
        (
            "run_pseudo_bending_first_slice",
            "foundation",
            "uv run tomobench run-first-slice --simulator pseudo-bending",
            "Generate the configured first synthetic pseudo-bending vertical slice and sidecars.",
            ["outputs/generated/first_vertical_slice"],
        ),
        (
            "benchmark_pseudo_bending",
            "benchmark",
            "uv run tomobench benchmark-pseudo-bending-paper",
            "Run the bounded paper-facing pseudo-bending benchmark.",
            ["outputs/benchmarks/pseudo_bending_paper_benchmark_v1"],
        ),
        (
            "prepare_paper_observation_corpus",
            "dataset",
            "uv run tomobench prepare-paper-pseudo-bending-observation-corpus",
            "Generate the five-family synthetic observation/target corpus used by the paper workflow.",
            ["outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1"],
        ),
        (
            "validate_paper_observation_corpus",
            "dataset",
            "uv run tomobench validate-paper-pseudo-bending-observation-corpus",
            "Validate manifest, observations, ray sidecars, target grids, and simulator values.",
            ["outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json"],
        ),
        (
            "train_paper_ml_suite",
            "phase4",
            "uv run tomobench train-paper-pseudo-bending-ml-suite",
            "Train/evaluate the fixed, repeated, leave-one-family-out, and noise-robustness ML suite.",
            ["outputs/generated/ml_baselines/paper_pseudo_bending_ml_suite_v1"],
        ),
        (
            "audit_paper_ml_suite",
            "phase4_5",
            "uv run tomobench audit-paper-pseudo-bending-ml-suite",
            "Audit generated paper ML suite artifacts before strengthened/generalization work.",
            ["outputs/generated/ml_baselines/paper_pseudo_bending_ml_suite_v1/audit_summary.json"],
        ),
        (
            "train_strengthened_ml_suite",
            "phase4_5b",
            "uv run tomobench train-paper-pseudo-bending-ml-strengthened",
            "Run the bounded audit-driven strengthened ML experiment.",
            ["outputs/generated/ml_baselines/paper_pseudo_bending_ml_strengthened_v1"],
        ),
        (
            "train_faulted_generalization",
            "phase4_5b",
            "uv run tomobench train-paper-pseudo-bending-faulted-generalization",
            "Run the bounded faulted-family diagnosis and intervention.",
            ["outputs/generated/ml_baselines/paper_pseudo_bending_faulted_generalization_v1"],
        ),
        (
            "run_known_ray_corpus",
            "phase6",
            "uv run tomobench run-known-ray-corpus-inversion",
            "Run the Phase 6 known-ray optimistic diagnostic across the paper corpus.",
            ["outputs/generated/classical_tomography/known_ray_corpus_v1"],
        ),
        (
            "run_reference_ray_corpus",
            "phase6",
            "uv run tomobench run-reference-ray-corpus-inversion",
            "Run the Phase 6 reference-model fixed-ray regularized baseline.",
            ["outputs/generated/classical_tomography/reference_ray_corpus_v1"],
        ),
        (
            "run_final_phase6_comparison",
            "phase6",
            "uv run tomobench run-final-phase6-comparison",
            "Create the final Phase 6 metric-contract-aware comparison audit.",
            ["outputs/generated/classical_tomography/final_phase6_comparison_v1"],
        ),
        (
            "run_phase7a_ablation",
            "phase7a",
            "uv run tomobench train-phase7-observation-signal-ablation",
            "Run the travel-time observation-signal ablation suite.",
            ["outputs/generated/ml_baselines/phase7_observation_signal_ablation_v1"],
        ),
        (
            "run_phase7b_sensitivity",
            "phase7b",
            "uv run tomobench run-reference-ray-regularization-sensitivity",
            "Run the bounded reference-ray damping/smoothing/clipping sensitivity audit.",
            ["outputs/generated/classical_tomography/reference_ray_regularization_sensitivity_v1"],
        ),
        (
            "run_phase7c_learning_curve",
            "phase7c",
            "uv run tomobench train-phase7-learning-curve --training-sizes 20,40,60,70 --repetition-seeds 1701,1702,1703",
            "Run the fixed-holdout learning curve using the documented training sizes and repetition seeds.",
            ["outputs/generated/ml_baselines/phase7_learning_curve_v1"],
        ),
        (
            "package_paper_candidate_v3",
            "paper_candidate_v3",
            "uv run tomobench package-paper-candidate-v3",
            "Package the preserved Phase 6/7 manuscript evidence without rerunning training or tomography.",
            ["outputs/generated/paper_candidate_v3"],
        ),
        (
            "run_phase8a_paired_comparison",
            "phase8a",
            "uv run tomobench run-phase8-paired-test-comparison",
            "Resolve exact ML fixed-test IDs and build the seven-method paired comparison.",
            ["outputs/generated/submission_readiness/phase8_paired_test_comparison_v1"],
        ),
        (
            "run_phase8b_uncertainty_report",
            "phase8b",
            "uv run tomobench run-phase8-uncertainty-report",
            "Compute deterministic method uncertainty and paired statistical reporting from Phase 8A.",
            ["outputs/generated/submission_readiness/phase8_uncertainty_report_v1"],
        ),
        (
            "package_phase8c_reproducibility",
            "phase8c",
            "uv run tomobench package-submission-reproducibility",
            "Package this manifest, command sequence, artifact map, and availability drafts.",
            ["outputs/generated/submission_readiness/reproducibility_manifest_v1"],
        ),
    ]
    return [
        {
            "order": index,
            "command_id": command_id,
            "phase": phase,
            "working_directory": "code",
            "command": command,
            "purpose": purpose,
            "outputs": outputs,
        }
        for index, (command_id, phase, command, purpose, outputs) in enumerate(commands, start=1)
    ]


def collect_git_metadata(repo_root: Path) -> dict[str, Any]:
    """Collect git metadata without failing when the path is outside a git repository."""
    try:
        inside = _run_git(repo_root, ["rev-parse", "--is-inside-work-tree"])
    except (OSError, subprocess.SubprocessError):
        return {
            "is_git_repository": False,
            "commit_hash": None,
            "branch": None,
            "dirty_worktree": None,
            "status_entry_count": None,
        }
    if inside.strip().lower() != "true":
        return {
            "is_git_repository": False,
            "commit_hash": None,
            "branch": None,
            "dirty_worktree": None,
            "status_entry_count": None,
        }
    try:
        commit_hash = _run_git(repo_root, ["rev-parse", "HEAD"]).strip() or None
        branch = _run_git(repo_root, ["branch", "--show-current"]).strip() or None
        status = _run_git(repo_root, ["status", "--porcelain"])
    except (OSError, subprocess.SubprocessError):
        return {
            "is_git_repository": True,
            "commit_hash": None,
            "branch": None,
            "dirty_worktree": None,
            "status_entry_count": None,
        }
    status_lines = [line for line in status.splitlines() if line.strip()]
    return {
        "is_git_repository": True,
        "commit_hash": commit_hash,
        "branch": branch,
        "dirty_worktree": bool(status_lines),
        "status_entry_count": len(status_lines),
    }


def build_artifact_map(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Inventory the source, configuration, documentation, data, and evidence artifacts."""
    root = repo_root or get_repo_root()
    artifact_map: list[dict[str, Any]] = []
    for spec in _ARTIFACT_SPECS:
        relative_path = Path(spec["path"])
        path = root / relative_path
        exists = path.exists()
        file_count, total_bytes = _path_stats(path) if exists else (0, 0)
        is_file = path.is_file()
        artifact_map.append(
            {
                **spec,
                "exists": exists,
                "status": "present" if exists else "missing",
                "file_count": file_count,
                "total_bytes": total_bytes,
                "hash_algorithm": "sha256" if is_file else "not_computed",
                "sha256": _sha256(path) if is_file else None,
            }
        )
    return artifact_map


def build_reproducibility_manifest(
    repo_root: Path | None = None,
    *,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the factual manifest payload without writing files."""
    root = repo_root or get_repo_root()
    config_path = root / DEFAULT_CONFIG_PATH
    pyproject_path = root / DEFAULT_PYPROJECT_PATH
    lockfile_path = root / DEFAULT_LOCKFILE_PATH
    artifact_map = build_artifact_map(root)
    git_metadata = collect_git_metadata(root)
    environment = collect_environment_summary(pyproject_path, lockfile_path)
    config_info = _config_metadata(config_path)
    commands = build_command_plan()
    main_output_directories = [
        row["path"]
        for row in artifact_map
        if row["artifact_type"] in {"generated_directory", "generated_dataset_directory"}
        and row["phase"] in {"paper_candidate_v3", "phase6", "phase7a", "phase7b", "phase7c", "phase8a", "phase8b"}
    ]
    return {
        "artifact_type": "submission_reproducibility_manifest",
        "package_id": REPRODUCIBILITY_MANIFEST_ID,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "repository": git_metadata,
        "runtime_environment": environment,
        "dependency_sources": {
            "project_definition": _relative_path(DEFAULT_PYPROJECT_PATH),
            "lockfile": _relative_path(DEFAULT_LOCKFILE_PATH) if lockfile_path.is_file() else None,
            "lockfile_present": lockfile_path.is_file(),
        },
        "central_config": config_info,
        "random_seeds": discover_random_seeds(root, config_path),
        "main_output_directories": main_output_directories,
        "artifact_map": artifact_map,
        "command_plan": commands,
        "availability_draft_status": {
            "data": "draft_only_no_public_release_claim",
            "software": "draft_only_no_public_release_claim",
            "repository_url": None,
            "doi": None,
        },
        "known_limitations": _known_limitations(),
        "claim_boundaries": _claim_boundaries(),
        "reproduction_notes": [
            "Commands are listed in dependency order and use the current tomobench CLI names.",
            "Run commands from the repository root with the working directory set to code unless a command is adapted explicitly.",
            "The Phase 8A and Phase 8B outputs are consumed as existing artifacts; Phase 8C does not recompute predictions or tomography.",
            "Directory entries report presence, file count, and byte count; individually listed files also carry SHA-256 hashes.",
        ],
        "package_outputs": [
            "reproducibility_manifest.json",
            "reproducibility_manifest.md",
            "commands_to_regenerate.md",
            "artifact_map.csv",
            "data_availability_draft.md",
            "software_availability_draft.md",
            "environment_summary.json",
        ],
    }


def collect_environment_summary(pyproject_path: Path, lockfile_path: Path) -> dict[str, Any]:
    """Collect interpreter, platform, and installed dependency metadata."""
    dependency_names = ("matplotlib", "numpy", "PyYAML", "pytest", "ruff", "hatchling")
    installed: dict[str, str | None] = {}
    for name in dependency_names:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full_version": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "dependencies": installed,
        "dependency_file_hashes": {
            "code/pyproject.toml": _sha256(pyproject_path) if pyproject_path.is_file() else None,
            "code/uv.lock": _sha256(lockfile_path) if lockfile_path.is_file() else None,
        },
    }


def discover_random_seeds(repo_root: Path, config_path: Path) -> dict[str, Any]:
    """Collect seeds explicitly reported by config and completed experiment summaries."""
    result: dict[str, Any] = {}
    config, config_error = _read_yaml_mapping(config_path)
    if config_error is None:
        result["config.random_seeds"] = _nested(config, "random_seeds")
        result["config.paper_corpus.random_seeds"] = _nested(
            config,
            "machine_learning",
            "paper_pseudo_bending_observation_corpus",
            "random_seeds",
        )
        result["config.paper_ml_suite"] = _selected_fields(
            _nested(config, "machine_learning", "paper_pseudo_bending_ml_suite"),
            ("fixed_split_seed", "repeated_split_seeds", "noise_random_seed"),
        )
    else:
        result["config_parse_error"] = config_error

    summary_sources = {
        "phase7a_ablation_summary": (
            "outputs/generated/ml_baselines/phase7_observation_signal_ablation_v1/ablation_summary.json",
            ("fixed_split_seed", "repeated_split_seeds", "travel_time_shuffle_seed"),
        ),
        "phase7c_learning_curve_summary": (
            "outputs/generated/ml_baselines/phase7_learning_curve_v1/learning_curve_summary.json",
            ("fixed_split_seed", "repetition_seeds"),
        ),
        "phase8b_method_summary": (
            "outputs/generated/submission_readiness/phase8_uncertainty_report_v1/method_uncertainty_summary.json",
            ("bootstrap",),
        ),
        "phase8b_delta_summary": (
            "outputs/generated/submission_readiness/phase8_uncertainty_report_v1/paired_delta_summary.json",
            ("permutation_test",),
        ),
    }
    for source_id, (relative_path, fields) in summary_sources.items():
        payload = _read_json_if_present(repo_root / relative_path)
        if payload:
            selected = _selected_fields(payload, fields)
            if selected:
                result[f"artifact.{source_id}"] = selected
    return {key: value for key, value in result.items() if value not in (None, {}, [])}


def render_data_availability_draft(manifest: Mapping[str, Any]) -> str:
    """Render a conservative manuscript data-availability draft."""
    return "\n".join(
        [
            "# Data Availability Statement Draft",
            "",
            "This study uses synthetic seismic observations, synthetic velocity models, target grids, "
            "and derived ray-path and sensitivity artifacts generated by the tomobench research repository.",
            "",
            "The synthetic datasets and generated artifacts can be made available together with the "
            "corresponding source code and configuration. This draft does not assert that the data have "
            "already been deposited, and no repository URL or DOI is asserted at this stage.",
            "",
            "The reproducibility package records the central configuration, dependency files, available "
            "git metadata, artifact paths, file hashes where applicable, and an ordered command sequence. "
            "These records provide traceability from the manuscript-facing evidence package to the generated "
            "synthetic data and software workflows.",
            "",
            "Before submission, replace this draft with the final repository or archive statement if a "
            "public deposit is completed.",
            "",
        ]
    )


def render_software_availability_draft(manifest: Mapping[str, Any]) -> str:
    """Render a conservative manuscript software-availability draft."""
    return "\n".join(
        [
            "# Software Availability Statement Draft",
            "",
            "The study software is maintained in the tomobench research repository. The Python project "
            "definition, uv lockfile, central configuration, CLI, evaluation modules, and focused tests "
            "are identified in the reproducibility manifest.",
            "",
            "The code used to generate the reported synthetic benchmark artifacts can be made available "
            "with the associated configuration and command sequence. This draft does not claim that the "
            "software has already been publicly released, and no repository URL or DOI is asserted at this stage.",
            "",
            "The manifest records the git commit hash and dirty-worktree status when the packaging environment "
            "can detect them. Dependency provenance is recorded through `code/pyproject.toml` and `code/uv.lock`.",
            "",
            "Before submission, add the final repository or archive reference if a public software deposit is completed.",
            "",
        ]
    )


def run_submission_reproducibility_package(
    *,
    output_dir: Path = DEFAULT_REPRODUCIBILITY_OUTPUT_DIR,
    repo_root: Path | None = None,
    experiment_note_path: Path | None = None,
) -> SubmissionReproducibilityOutputs:
    """Write the Phase 8C reproducibility package and its wiki note."""
    root = repo_root or get_repo_root()
    resolved_output_dir = _resolve_path(output_dir, root)
    manifest = build_reproducibility_manifest(root)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = resolved_output_dir / "reproducibility_manifest.json"
    manifest_md = resolved_output_dir / "reproducibility_manifest.md"
    commands_md = resolved_output_dir / "commands_to_regenerate.md"
    artifact_csv = resolved_output_dir / "artifact_map.csv"
    data_draft = resolved_output_dir / "data_availability_draft.md"
    software_draft = resolved_output_dir / "software_availability_draft.md"
    environment_json = resolved_output_dir / "environment_summary.json"
    _write_json(manifest_json, manifest)
    _write_text(manifest_md, _render_manifest_markdown(manifest))
    _write_text(commands_md, _render_commands_markdown(manifest["command_plan"]))
    _write_artifact_csv(artifact_csv, manifest["artifact_map"])
    _write_text(data_draft, render_data_availability_draft(manifest))
    _write_text(software_draft, render_software_availability_draft(manifest))
    _write_json(environment_json, manifest["runtime_environment"])
    note_path = _resolve_path(
        experiment_note_path or root / "wiki/experiments/reproducibility_manifest_v1.md",
        root,
    )
    _write_text(note_path, _render_wiki_note(manifest, resolved_output_dir, root))
    return SubmissionReproducibilityOutputs(
        output_dir=resolved_output_dir,
        manifest_json=manifest_json,
        manifest_md=manifest_md,
        commands_md=commands_md,
        artifact_csv=artifact_csv,
        data_draft=data_draft,
        software_draft=software_draft,
        environment_json=environment_json,
        experiment_note=note_path,
    )


def _known_limitations() -> list[str]:
    return [
        "The study is synthetic-only and does not provide real-data validation.",
        "The workflow does not demonstrate full nonlinear iterative tomography.",
        "Known-ray inversion is an optimistic diagnostic because it uses true-model ray geometry.",
        "Reference-ray inversion is a fixed-ray regularized baseline, not full nonlinear tomography.",
        "Phase 7C does not establish data saturation because the fixed holdout leaves 70 eligible training cases.",
        "The full-ML versus travel-time-only difference is unresolved on the fixed 25-case comparison.",
        "Phase 8B sign-flip p-values are exploratory only for the small paired synthetic sample.",
    ]


def _claim_boundaries() -> list[str]:
    return [
        "No universal ML superiority claim is made.",
        "Any ML-versus-reference-ray statement is limited to the configured synthetic benchmark and implemented fixed-ray baseline.",
        "Known-ray and reference-ray outputs must retain their diagnostic and fixed-ray labels in manuscript tables.",
        "Generated artifacts are evidence for reproducibility and do not imply public data or software deposition.",
    ]


def _config_metadata(path: Path) -> dict[str, Any]:
    payload, error = _read_yaml_mapping(path)
    result: dict[str, Any] = {
        "path": _relative_path(DEFAULT_CONFIG_PATH),
        "sha256": _sha256(path) if path.is_file() else None,
        "exists": path.is_file(),
    }
    if error is not None:
        result["parse_error"] = error
    elif payload:
        result["project_title"] = _nested(payload, "project", "title")
        result["simulation_method"] = _nested(payload, "simulation", "method")
        result["target_representation"] = _nested(payload, "machine_learning", "target_representation")
    return result


def _read_yaml_mapping(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, f"missing file: {path}"
    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, ValueError, ImportError) as exc:
        return {}, str(exc)
    if not isinstance(payload, dict):
        return {}, "expected a YAML mapping"
    return payload, None


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _selected_fields(payload: Any, fields: Sequence[str]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {field: payload[field] for field in fields if field in payload and payload[field] is not None}


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_git(repo_root: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def _path_stats(path: Path) -> tuple[int, int]:
    if path.is_file():
        return 1, path.stat().st_size
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return len(files), sum(candidate.stat().st_size for candidate in files)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_path(path: Path) -> str:
    return path.as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_artifact_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "artifact_id",
        "path",
        "artifact_type",
        "phase",
        "role",
        "description",
        "exists",
        "status",
        "file_count",
        "total_bytes",
        "hash_algorithm",
        "sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_manifest_markdown(manifest: Mapping[str, Any]) -> str:
    repository = manifest["repository"]
    environment = manifest["runtime_environment"]
    config = manifest["central_config"]
    lines = [
        "# Submission Reproducibility Manifest",
        "",
        f"Package ID: `{manifest['package_id']}`",
        f"Generated at UTC: `{manifest['generated_at_utc']}`",
        "",
        "## Repository and environment",
        "",
        f"- Git repository detected: `{repository['is_git_repository']}`",
        f"- Commit hash: `{repository['commit_hash'] or 'unavailable'}`",
        f"- Branch: `{repository['branch'] or 'unavailable'}`",
        f"- Dirty worktree: `{repository['dirty_worktree'] if repository['dirty_worktree'] is not None else 'unavailable'}`",
        f"- Python: `{environment['python_version']}` ({environment['python_implementation']})",
        f"- Platform: `{environment['platform']}`",
        f"- Central config: `{config['path']}`",
        f"- Config SHA-256: `{config['sha256'] or 'unavailable'}`",
        f"- Dependency lockfile: `{manifest['dependency_sources']['lockfile'] or 'not present'}`",
        "",
        "## Reported random seeds",
        "",
    ]
    for key, value in manifest["random_seeds"].items():
        lines.append(f"- `{key}`: `{json.dumps(value, sort_keys=True)}`")
    lines.extend(["", "## Main output directories", ""])
    for path in manifest["main_output_directories"]:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Artifact inventory",
            "",
            "See `artifact_map.csv` for existence, directory counts, byte counts, and SHA-256 hashes for individually listed files.",
            "",
            "| Artifact | Type | Phase | Status | Path |",
            "|---|---|---|---|---|",
        ]
    )
    for row in manifest["artifact_map"]:
        lines.append(
            f"| `{row['artifact_id']}` | `{row['artifact_type']}` | `{row['phase']}` | "
            f"`{row['status']}` | `{row['path']}` |"
        )
    lines.extend(["", "## Limitations and claim boundaries", ""])
    for limitation in manifest["known_limitations"]:
        lines.append(f"- {limitation}")
    for boundary in manifest["claim_boundaries"]:
        lines.append(f"- {boundary}")
    lines.extend(
        [
            "",
            "## Availability status",
            "",
            "The included data/software statements are drafts only. They do not claim public deposition and contain no invented repository URL, DOI, or citation.",
            "",
            "## Regeneration",
            "",
            "Use `commands_to_regenerate.md` for the dependency-ordered CLI sequence. Phase 8C itself packages existing evidence and does not rerun model predictions or tomography.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_commands_markdown(commands: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Commands to Regenerate Submission Evidence",
        "",
        "Run from the repository root. Each command is executed with `code` as its working directory and uses the current CLI exposed by `code/pyproject.toml`.",
        "The sequence is dependency-ordered; later packaging commands consume earlier generated artifacts.",
        "",
        "| Order | Phase | Command | Main outputs |",
        "|---:|---|---|---|",
    ]
    for item in commands:
        outputs = ", ".join(f"`{output}`" for output in item["outputs"])
        lines.append(
            f"| {item['order']} | `{item['phase']}` | `{item['command']}` | {outputs} |"
        )
    lines.extend(["", "## Notes", "", "- The central scientific configuration is `config/benchmark_config.yaml`.", "- The locked dependency source is `code/uv.lock`.", "- Phase 8A resolves its 25 fixed test IDs from the selected ML metrics artifact; Phase 8B consumes the Phase 8A case table directly.", "- Phase 7B sensitivity does not silently replace the Phase 6 reference-ray baseline in the paired comparison.", "- Commands that regenerate historical artifacts may update their output directories; preserve prior packages if historical comparison is required.", ""])
    return "\n".join(lines)


def _render_wiki_note(manifest: Mapping[str, Any], output_dir: Path, repo_root: Path) -> str:
    repository = manifest["repository"]
    environment = manifest["runtime_environment"]
    phase8b_seeds = {
        key: value
        for key, value in manifest["random_seeds"].items()
        if key.startswith("artifact.phase8b")
    }
    lines = [
        "# Experiment: Submission Reproducibility Manifest v1",
        "",
        f"Experiment ID: `{REPRODUCIBILITY_MANIFEST_ID}`",
        "",
        "## Objective",
        "",
        "Package the Applied Sciences submission-readiness evidence with exact CLI commands, configuration and dependency provenance, available seeds, artifact references, and conservative data/software availability drafts.",
        "",
        "## Package contents",
        "",
        f"- Output directory: `{_repo_relative(output_dir, repo_root)}`",
        f"- Artifact references: `{len(manifest['artifact_map'])}`",
        f"- Regeneration commands: `{len(manifest['command_plan'])}` in dependency order",
        f"- Central config SHA-256: `{manifest['central_config']['sha256'] or 'unavailable'}`",
        f"- Python: `{environment['python_version']}`; lockfile present: `{manifest['dependency_sources']['lockfile_present']}`",
        f"- Git commit: `{repository['commit_hash'] or 'unavailable'}`; dirty-worktree flag: `{repository['dirty_worktree'] if repository['dirty_worktree'] is not None else 'unavailable'}`",
        "",
        "## Relevant seed records",
        "",
    ]
    for key, value in manifest["random_seeds"].items():
        if key.startswith("config.") or key in phase8b_seeds:
            lines.append(f"- `{key}`: `{json.dumps(value, sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Scientific and availability boundaries",
            "",
            "- Synthetic-only study; no real-data validation is claimed.",
            "- The workflow does not demonstrate full nonlinear iterative tomography.",
            "- Known-ray is an optimistic diagnostic; reference-ray is a fixed-ray regularized baseline.",
            "- Phase 7C does not establish saturation at 100 training cases.",
            "- The full-ML versus travel-time-only difference remains unresolved on the fixed 25-case set.",
            "- Data and software statements are drafts; no public repository URL or DOI is asserted.",
            "",
            "## Review files",
            "",
            f"- Manifest: `{_repo_relative(output_dir / 'reproducibility_manifest.json', repo_root)}`",
            f"- Commands: `{_repo_relative(output_dir / 'commands_to_regenerate.md', repo_root)}`",
            f"- Artifact map: `{_repo_relative(output_dir / 'artifact_map.csv', repo_root)}`",
            f"- Data availability draft: `{_repo_relative(output_dir / 'data_availability_draft.md', repo_root)}`",
            f"- Software availability draft: `{_repo_relative(output_dir / 'software_availability_draft.md', repo_root)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
