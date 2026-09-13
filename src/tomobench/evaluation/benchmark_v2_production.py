"""Build and freeze the production 250-target Benchmark v2 corpus.

The production corpus is intentionally separate from the 50-target pilot.  It
reuses the pilot-approved geological distributions and frozen ttcrpy FSM label
contract, but uses a separate deterministic seed namespace, performs exact
target-hash rejection, and writes a target-atomic split before any supervised
model fitting.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from tomobench.config import load_settings
from tomobench.datasets.target_identity import TARGET_HASH_SCHEMA
from tomobench.evaluation.benchmark_v2_pilot import (
    FAMILIES,
    OBSERVATION_COUNT,
    PRODUCTION_ID,
    ProductionConfig,
    PilotTargetDraft,
    _directory_size_bytes,
    _generate_authoritative_observations,
    _generate_target_drafts,
    _git_commit,
    _git_worktree_dirty,
    _grid_shape,
    _package_versions,
    _relative_path,
    _target_quality_rows,
    _target_registry_rows,
    _target_representation_rows,
    _write_csv,
    _write_json,
    nearest_neighbor_diversity_rows,
)
from tomobench.utils.paths import get_repo_root


PRODUCTION_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/benchmark_v2_production_250_v1"
)
PRODUCTION_TARGET_COUNT = 250
PRODUCTION_TARGETS_PER_FAMILY = 50
PRODUCTION_TRAIN_PER_FAMILY = 35
PRODUCTION_VALIDATION_PER_FAMILY = (8, 8, 7, 7, 7)
PRODUCTION_TEST_PER_FAMILY = (7, 7, 8, 8, 8)
SPLIT_NAMES = ("train", "validation", "test")


def run_production(
    *,
    output_dir: Path = PRODUCTION_OUTPUT_DIR,
    base_seed: int = ProductionConfig().base_seed,
) -> Path:
    """Generate, audit, and freeze the 250-target production corpus.

    The function refuses to write into a non-empty directory.  This protects
    the frozen corpus from accidental replacement after downstream analyses
    have begun.
    """

    repo_root = get_repo_root()
    resolved_output = _resolve_path(output_dir, repo_root)
    if resolved_output.exists() and any(resolved_output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing production output directory: {resolved_output}"
        )
    resolved_output.mkdir(parents=True, exist_ok=True)
    config = ProductionConfig(base_seed=base_seed)
    started_at = datetime.now(UTC).isoformat()
    run_started = perf_counter()
    base_settings = load_settings()

    config_payload = config.to_dict()
    config_payload.update(
        {
            "artifact_type": "benchmark_v2_production_configuration",
            "production_id": PRODUCTION_ID,
            "production_target_count": PRODUCTION_TARGET_COUNT,
            "split_contract": {
                "unit": "unique target velocity model",
                "train_unique_targets": PRODUCTION_TRAIN_PER_FAMILY * len(FAMILIES),
                "validation_unique_targets": sum(PRODUCTION_VALIDATION_PER_FAMILY),
                "test_unique_targets": sum(PRODUCTION_TEST_PER_FAMILY),
                "assignment": "sorted (family, target_hash) within family",
            },
            "scope": {
                "ml_fitting": "not performed during corpus construction",
                "test_set": "created and frozen before model selection",
                "legacy_corpus": "not used",
            },
        }
    )
    _write_json(config_payload, resolved_output / "benchmark_v2_production_config.json")

    drafts = _generate_target_drafts(base_settings, config, resolved_output)
    _write_csv(
        resolved_output / "target_case_registry.csv",
        _target_registry_rows(drafts, config),
    )
    (
        observations,
        runtime_rows,
        label_wall_s,
        shared_setup_s,
        shared_setup,
    ) = _generate_authoritative_observations(
        base_settings,
        config,
        resolved_output,
        drafts,
    )
    _write_csv(resolved_output / "travel_time_observations.csv", observations)
    _write_csv(resolved_output / "production_runtime.csv", runtime_rows)
    _write_csv(
        resolved_output / "target_case_registry.csv",
        _target_registry_rows(drafts, config),
    )

    diversity_rows = nearest_neighbor_diversity_rows(drafts)
    _write_csv(resolved_output / "target_diversity_summary.csv", diversity_rows)
    _write_diversity_report(resolved_output / "target_diversity_report.md", drafts, diversity_rows)

    representation_rows = _target_representation_rows(base_settings, drafts, config)
    _write_csv(resolved_output / "target_representation_audit.csv", representation_rows)
    quality_rows = _target_quality_rows(observations, drafts, config)
    _write_csv(resolved_output / "production_travel_time_summary.csv", quality_rows)

    split_rows = build_target_split_manifest(drafts)
    _write_csv(resolved_output / "benchmark_v2_target_split_manifest.csv", split_rows)
    _write_csv(
        resolved_output / "target_case_registry.csv",
        _target_registry_rows_with_split(drafts, config, split_rows),
    )
    production_manifest = build_production_manifest(
        drafts,
        split_rows,
        output_dir=resolved_output,
        config=config,
    )
    _write_json(production_manifest, resolved_output / "benchmark_v2_manifest.json")

    checks = validate_production_integrity(
        drafts=drafts,
        observations=observations,
        representation_rows=representation_rows,
        quality_rows=quality_rows,
        split_rows=split_rows,
        config=config,
    )
    _write_production_quality_report(
        resolved_output / "production_corpus_quality.md",
        drafts=drafts,
        observations=observations,
        quality_rows=quality_rows,
        split_rows=split_rows,
        checks=checks,
        config=config,
    )
    runtime_payload = _write_production_runtime_report(
        resolved_output,
        runtime_rows=runtime_rows,
        label_wall_s=label_wall_s,
        shared_setup_s=shared_setup_s,
        shared_setup=shared_setup,
        run_started=run_started,
        config=config,
    )
    metadata = {
        "artifact_type": "benchmark_v2_production_run_metadata",
        "production_id": PRODUCTION_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": started_at,
        "command": " ".join(sys.argv),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "package_versions": _package_versions(),
        "git_commit": _git_commit(repo_root),
        "git_worktree_dirty": _git_worktree_dirty(repo_root),
        "target_hash_schema": TARGET_HASH_SCHEMA,
        "target_count": len(drafts),
        "observation_count": len(observations),
        "label_generation_wall_s": label_wall_s,
        "end_to_end_wall_s": perf_counter() - run_started,
        "shared_setup": shared_setup,
        "runtime_summary": runtime_payload,
        "integrity_checks": checks,
        "frozen_contract": {
            "forward_solver": "ttcrpy==1.4.2 / 3-D rectilinear FSM / tt_from_rp=False",
            "forward_representation": "node-centered velocity sampled directly from analytic model",
            "forward_spacing_km": [1.25, 1.25, 1.25],
            "target_spacing_km": [2.5, 2.5, 2.5],
            "target_shape": [41, 41, 13],
            "realistic_inputs": [
                "source coordinates",
                "receiver coordinates",
                "Euclidean source-receiver distance",
                "travel time",
            ],
            "true_model_path_length": "excluded from realistic observation representation",
        },
        "scope_exclusions": [
            "No historical leakage-affected corpus was used.",
            "No ML model was fit during corpus construction.",
            "No manuscript file was modified.",
            "No legacy, SPM, DSPM, or PyKonal labels were generated.",
        ],
    }
    _write_json(metadata, resolved_output / "production_metadata.json")
    if not all(checks.values()):
        raise RuntimeError(
            "Production corpus integrity failed: "
            + "; ".join(name for name, passed in checks.items() if not passed)
        )
    return resolved_output


def build_target_split_manifest(
    drafts: Sequence[PilotTargetDraft],
) -> list[dict[str, Any]]:
    """Assign every target once to the frozen 175/37/38 split."""

    if len(drafts) != PRODUCTION_TARGET_COUNT:
        raise ValueError(f"Expected {PRODUCTION_TARGET_COUNT} targets, got {len(drafts)}.")
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for family_index, family in enumerate(FAMILIES):
        family_drafts = sorted(
            (draft for draft in drafts if draft.family == family),
            key=lambda draft: (draft.family, draft.target_hash),
        )
        if len(family_drafts) != PRODUCTION_TARGETS_PER_FAMILY:
            raise ValueError(
                f"Family {family!r} must contain {PRODUCTION_TARGETS_PER_FAMILY} targets; "
                f"got {len(family_drafts)}."
            )
        validation_count = PRODUCTION_VALIDATION_PER_FAMILY[family_index]
        test_count = PRODUCTION_TEST_PER_FAMILY[family_index]
        if PRODUCTION_TRAIN_PER_FAMILY + validation_count + test_count != len(family_drafts):
            raise ValueError(f"Split counts do not sum to 50 for family {family!r}.")
        for local_index, draft in enumerate(family_drafts):
            if draft.target_hash in seen_hashes:
                raise ValueError(f"Duplicate target hash in split manifest: {draft.target_hash}")
            seen_hashes.add(draft.target_hash)
            split = (
                "train"
                if local_index < PRODUCTION_TRAIN_PER_FAMILY
                else "validation"
                if local_index < PRODUCTION_TRAIN_PER_FAMILY + validation_count
                else "test"
            )
            rows.append(
                {
                    "target_id": draft.target_id,
                    "target_hash": draft.target_hash,
                    "target_hash_schema": TARGET_HASH_SCHEMA,
                    "family": draft.family,
                    "split": split,
                    "target_index": draft.target_index,
                    "acquisition_id": draft.acquisition_id,
                    "geology_seed": draft.geology_seed,
                    "station_seed": draft.station_seed,
                    "earthquake_seed": draft.earthquake_seed,
                    "parameter_vector_json": json.dumps(list(draft.parameter_vector)),
                    "parameter_names_json": json.dumps(list(draft.parameter_names)),
                    "physical_parameters_json": json.dumps(
                        draft.physical_parameters,
                        sort_keys=True,
                    ),
                    "target_grid_artifact_path": draft.target_grid_artifact_path,
                    "target_artifact_path": draft.target_artifact_path,
                }
            )
    rows.sort(key=lambda row: (str(row["family"]), str(row["target_hash"])))
    _validate_split_rows(rows)
    return rows


def build_production_manifest(
    drafts: Sequence[PilotTargetDraft],
    split_rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    config: ProductionConfig,
) -> dict[str, Any]:
    """Create the machine-readable case manifest consumed by later analyses."""

    split_by_target = {str(row["target_id"]): str(row["split"]) for row in split_rows}
    if len(split_by_target) != len(split_rows):
        raise ValueError("Split manifest target IDs must be unique.")
    items: list[dict[str, Any]] = []
    for draft in sorted(drafts, key=lambda item: item.target_id):
        observation_path = output_dir / "observations" / f"{draft.target_id}_observations.csv"
        stations_path = output_dir / "acquisitions" / f"{draft.acquisition_id}_stations.json"
        earthquakes_path = output_dir / "acquisitions" / f"{draft.acquisition_id}_earthquakes.json"
        required = (
            observation_path,
            stations_path,
            earthquakes_path,
            Path(_resolve_repo_path(draft.target_grid_artifact_path)),
        )
        if not all(path.is_file() for path in required):
            missing = [str(path) for path in required if not path.is_file()]
            raise FileNotFoundError(f"Missing production artifact(s) for {draft.target_id}: {missing}")
        items.append(
            {
                "case_id": draft.target_id,
                "target_id": draft.target_id,
                "target_hash": draft.target_hash,
                "target_hash_schema": TARGET_HASH_SCHEMA,
                "family": draft.family,
                "scenario": draft.family,
                "split": split_by_target[draft.target_id],
                "acquisition_id": draft.acquisition_id,
                "station_seed": draft.station_seed,
                "earthquake_seed": draft.earthquake_seed,
                "geology_seed": draft.geology_seed,
                "row_count": config.observations_per_target,
                "observation_count": config.observations_per_target,
                "observation_csv_path": _relative_path(observation_path),
                "input_dataset_csv_path": _relative_path(observation_path),
                "target_velocity_grid_path": draft.target_grid_artifact_path,
                "target_grid_path": draft.target_grid_artifact_path,
                "target_artifact_path": draft.target_artifact_path,
                "model_artifact_path": draft.model_artifact_path,
                "physical_parameters": draft.physical_parameters,
                "parameter_vector": list(draft.parameter_vector),
                "parameter_names": list(draft.parameter_names),
                "station_configuration_path": _relative_path(stations_path),
                "earthquake_configuration_path": _relative_path(earthquakes_path),
                "simulation_method": "ttcrpy_fsm",
                "simulator_name": "ttcrpy_fsm",
                "true_model_path_length_included": False,
            }
        )
    return {
        "artifact_type": "benchmark_v2_production_manifest",
        "manifest_version": "benchmark_v2_production_manifest_v1",
        "production_id": PRODUCTION_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "target_count": len(drafts),
        "case_count": len(items),
        "observations_per_case": config.observations_per_target,
        "observation_count": len(items) * config.observations_per_target,
        "target_hash_schema": TARGET_HASH_SCHEMA,
        "target_contract": {
            "shape": [41, 41, 13],
            "vector_length": 41 * 41 * 13,
            "spacing_km": list(config.target_spacing_km),
            "representation": "node-centered absolute P-wave velocity",
        },
        "forward_contract": {
            "package": "ttcrpy==1.4.2",
            "algorithm": "FSM",
            "tt_from_rp": False,
            "spacing_km": list(config.forward_spacing_km),
            "physical_representation": "node-centered velocity",
        },
        "items": items,
    }


def validate_production_integrity(
    *,
    drafts: Sequence[PilotTargetDraft],
    observations: Sequence[Mapping[str, Any]],
    representation_rows: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    config: ProductionConfig,
) -> dict[str, bool]:
    """Return explicit integrity invariants for the frozen production corpus."""

    family_counts = {family: sum(draft.family == family for draft in drafts) for family in FAMILIES}
    hashes = [draft.target_hash for draft in drafts]
    obs_by_target: dict[str, list[Mapping[str, Any]]] = {}
    for row in observations:
        obs_by_target.setdefault(str(row.get("target_id", "")), []).append(row)
    quality_by_target = {str(row["target_id"]): row for row in quality_rows}
    representation_by_target = {str(row["target_id"]): row for row in representation_rows}
    all_finite = all(
        str(row.get("label_status", "")) == "finite"
        and str(row.get("background_status", "")) in {"finite", "same_target_layered_background"}
        and str(row.get("observation_status", "")) == "finite"
        for row in observations
    )
    shape_valid = all(
        _grid_shape(draft.target_grid) == (41, 41, 13)
        and len(draft.target_grid.p_velocity_km_per_s) == 41 * 41 * 13
        for draft in drafts
    )
    represented = all(
        str(row.get("resolution_status")) == "represented"
        for row in representation_rows
        if str(row.get("family")) != "layered"
    )
    dyke_resolution = all(
        bool(row.get("dyke_minimum_width_requirement_met"))
        for row in representation_rows
        if str(row.get("family")) == "dyke_intrusion"
    )
    split_counts = {name: sum(str(row.get("split")) == name for row in split_rows) for name in SPLIT_NAMES}
    split_hash_sets = {
        name: {str(row["target_hash"]) for row in split_rows if str(row.get("split")) == name}
        for name in SPLIT_NAMES
    }
    hash_by_target = {draft.target_id: draft.target_hash for draft in drafts}
    target_hashes_in_rows = {
        str(row.get("target_hash"))
        for target_rows in obs_by_target.values()
        for row in target_rows
    }
    return {
        "exactly_250_targets": len(drafts) == PRODUCTION_TARGET_COUNT,
        "exactly_50_targets_per_family": all(
            family_counts[family] == PRODUCTION_TARGETS_PER_FAMILY for family in FAMILIES
        ),
        "exactly_250_unique_target_hashes": len(set(hashes)) == PRODUCTION_TARGET_COUNT,
        "exactly_96000_observations": len(observations)
        == PRODUCTION_TARGET_COUNT * OBSERVATION_COUNT,
        "every_target_has_384_observations": len(obs_by_target) == len(drafts)
        and all(len(rows) == OBSERVATION_COUNT for rows in obs_by_target.values()),
        "all_fsm_labels_and_backgrounds_finite": all_finite,
        "all_target_arrays_are_41x41x13": shape_valid,
        "all_non_layered_structures_survive_target_sampling": represented,
        "all_dykes_meet_target_resolution_constraint": dyke_resolution,
        "observation_target_hashes_match_registry": target_hashes_in_rows == set(hashes)
        and all(
            all(str(row.get("target_hash")) == hash_by_target[target_id] for row in rows)
            for target_id, rows in obs_by_target.items()
            if target_id in hash_by_target
        ),
        "quality_and_representation_cover_every_target": set(quality_by_target)
        == set(hash_by_target)
        and set(representation_by_target) == set(hash_by_target),
        "split_contains_every_target_once": len(split_rows) == len(drafts)
        and {str(row["target_id"]) for row in split_rows} == set(hash_by_target),
        "split_counts_are_175_37_38": split_counts == {"train": 175, "validation": 37, "test": 38},
        "split_hashes_are_disjoint": not (
            split_hash_sets["train"] & split_hash_sets["validation"]
            or split_hash_sets["train"] & split_hash_sets["test"]
            or split_hash_sets["validation"] & split_hash_sets["test"]
        ),
        "split_family_balance_is_frozen": _split_family_balance_is_valid(split_rows),
        "true_model_path_length_is_not_in_realistic_rows": all(
            str(row.get("true_model_path_length_included", "False")).lower() == "false"
            for row in observations
        ),
        "finite_target_quality_rows": all(
            bool(row.get("all_labels_finite")) and bool(row.get("all_background_labels_finite"))
            for row in quality_rows
        ),
        "config_observation_count_matches": config.observations_per_target == OBSERVATION_COUNT,
    }


def _target_registry_rows_with_split(
    drafts: Sequence[PilotTargetDraft],
    config: ProductionConfig,
    split_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    split_by_target = {str(row["target_id"]): str(row["split"]) for row in split_rows}
    rows = _target_registry_rows(drafts, config)
    for row in rows:
        row["split"] = split_by_target[str(row["target_id"])]
    return rows


def _validate_split_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != PRODUCTION_TARGET_COUNT:
        raise ValueError("Frozen split must contain exactly 250 rows.")
    if len({str(row["target_id"]) for row in rows}) != len(rows):
        raise ValueError("Frozen split contains duplicate target IDs.")
    if len({str(row["target_hash"]) for row in rows}) != len(rows):
        raise ValueError("Frozen split contains duplicate target hashes.")
    counts = {name: sum(str(row["split"]) == name for row in rows) for name in SPLIT_NAMES}
    if counts != {"train": 175, "validation": 37, "test": 38}:
        raise ValueError(f"Unexpected frozen split counts: {counts}")
    if not _split_family_balance_is_valid(rows):
        raise ValueError("Frozen split family balance is not the prepared 35/(8|7)/(7|8) design.")


def _split_family_balance_is_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    expected = {
        family: {
            "train": PRODUCTION_TRAIN_PER_FAMILY,
            "validation": PRODUCTION_VALIDATION_PER_FAMILY[index],
            "test": PRODUCTION_TEST_PER_FAMILY[index],
        }
        for index, family in enumerate(FAMILIES)
    }
    actual = {
        family: {name: 0 for name in SPLIT_NAMES}
        for family in FAMILIES
    }
    for row in rows:
        family = str(row.get("family"))
        split = str(row.get("split"))
        if family not in actual or split not in SPLIT_NAMES:
            return False
        actual[family][split] += 1
    return actual == expected


def _write_diversity_report(
    path: Path,
    drafts: Sequence[PilotTargetDraft],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Benchmark v2 production target diversity report",
        "",
        "Exact duplicate target hashes are rejected during generation. Near-neighbor models are retained when they arise from valid parameter draws and are reported descriptively rather than removed by an arbitrary threshold.",
        "",
        f"- Unique geological targets: `{len(drafts)}`",
        f"- Unique SHA-256 target hashes: `{len({draft.target_hash for draft in drafts})}`",
        "- Target identity schema: `target_velocity_vector_sha256_v1`",
        "",
        "## Within-family nearest-neighbor summaries",
        "",
        "| Family | Targets | Node RMSE median (km/s) | Cell RMSE median (km/s) | Node RMSE minimum (km/s) | Cell RMSE minimum (km/s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        family_rows = [row for row in rows if str(row["family"]) == family]
        node = np.asarray([float(row["nearest_node_rmse_km_per_s"]) for row in family_rows])
        cell = np.asarray([float(row["nearest_cell_rmse_km_per_s"]) for row in family_rows])
        lines.append(
            f"| {family} | {len(family_rows)} | {np.median(node):.6g} | {np.median(cell):.6g} | {np.min(node):.6g} | {np.min(cell):.6g} |"
        )
    lines.extend(
        [
            "",
            "The CSV additionally records target means, target standard deviations, nearest-neighbor maximum absolute differences, and normalized physical-parameter distances.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_production_quality_report(
    path: Path,
    *,
    drafts: Sequence[PilotTargetDraft],
    observations: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    checks: Mapping[str, bool],
    config: ProductionConfig,
) -> None:
    lines = [
        "# Benchmark v2 production corpus quality",
        "",
        "This is the integrity report for the frozen 250-target corpus. It is not a new forward-solver gate. Labels use the adopted finite-grid ttcrpy FSM approximation.",
        "",
        "## Frozen corpus contract",
        "",
        "- 250 unique geological targets; 50 per family",
        "- One acquisition realization per target",
        "- 16 stations × 24 earthquakes = 384 observations per target",
        "- Forward labels: `ttcrpy==1.4.2`, 3-D rectilinear FSM, `tt_from_rp=False`, node-centered velocity, 1.25 km spacing",
        "- Target grid: node-centered 41 × 41 × 13 values at 2.5 km spacing",
        "- Realistic input rows exclude true-model ray-path length",
        "",
        "## Counts",
        "",
        f"- Unique targets: `{len(drafts)}`",
        f"- Acquisition cases: `{len(drafts)}`",
        f"- Observations: `{len(observations)}`",
        f"- Split counts: `{ {name: sum(str(row['split']) == name for row in split_rows) for name in SPLIT_NAMES} }`",
        "",
        "## Integrity checks",
        "",
        "| Check | Result |",
        "|---|:---:|",
    ]
    for name, passed in checks.items():
        lines.append(f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Travel-time sensitivity summary",
            "",
            "`delta_t_geology = t_target - t_layered_background`. The sensitivity classes are descriptive and do not reject weakly observed targets.",
            "",
            "| Family | Targets | Informative | Weakly observed | Essentially unobserved | Median target max |delta t| (s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for family in FAMILIES:
        family_rows = [row for row in quality_rows if str(row["family"]) == family]
        signals = np.asarray(
            [float(row["delta_t_geology_abs_max_s"]) for row in family_rows],
            dtype=float,
        )
        lines.append(
            f"| {family} | {len(family_rows)} | "
            f"{sum(row['sensitivity_class'] == 'informative' for row in family_rows)} | "
            f"{sum(row['sensitivity_class'] == 'weakly_observed' for row in family_rows)} | "
            f"{sum(row['sensitivity_class'] == 'essentially_unobserved' for row in family_rows)} | "
            f"{np.median(signals):.6g} |"
        )
    lines.extend(
        [
            "",
            "All targets, including weakly observed structures, remain in the corpus so that acquisition sensitivity is represented rather than hidden.",
            "",
            "## Split freeze",
            "",
            "The split is target-atomic and was written before PCA fitting or model selection. No target hash may appear in more than one split. The 38 test targets are frozen for the definitive evaluation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_production_runtime_report(
    output_dir: Path,
    *,
    runtime_rows: Sequence[Mapping[str, Any]],
    label_wall_s: float,
    shared_setup_s: float,
    shared_setup: Mapping[str, Any],
    run_started: float,
    config: ProductionConfig,
) -> dict[str, Any]:
    target_times = np.asarray([float(row["target_elapsed_s"]) for row in runtime_rows], dtype=float)
    output_size = _directory_size_bytes(output_dir)
    payload = {
        "target_count": len(runtime_rows),
        "label_generation_wall_s": label_wall_s,
        "end_to_end_wall_s": perf_counter() - run_started,
        "shared_setup_s": shared_setup_s,
        "per_target_mean_s": float(np.mean(target_times)) if target_times.size else None,
        "per_target_median_s": float(np.median(target_times)) if target_times.size else None,
        "per_target_min_s": float(np.min(target_times)) if target_times.size else None,
        "per_target_max_s": float(np.max(target_times)) if target_times.size else None,
        "projected_250_target_runtime_h": label_wall_s / 3600.0,
        "output_size_bytes": output_size,
        "solver_setup": dict(shared_setup),
        "forward_grid_spacing_km": list(config.forward_spacing_km),
        "observations_per_target": config.observations_per_target,
    }
    lines = [
        "# Benchmark v2 production runtime and storage report",
        "",
        "Runtime is measured from the actual 250-target production process. Working-set values in `production_runtime.csv` are process snapshots, not operating-system peak-memory guarantees.",
        "",
        f"- Targets: `{len(runtime_rows)}`",
        f"- Label-generation wall time: `{label_wall_s:.3f}` s ({label_wall_s / 3600.0:.3f} h)",
        f"- End-to-end elapsed time at report write: `{payload['end_to_end_wall_s']:.3f}` s",
        f"- Mean target elapsed time: `{payload['per_target_mean_s']:.3f}` s",
        f"- Median target elapsed time: `{payload['per_target_median_s']:.3f}` s",
        f"- Target elapsed range: `{payload['per_target_min_s']:.3f}`–`{payload['per_target_max_s']:.3f}` s",
        f"- Output size at report write: `{output_size}` bytes",
        "",
        "## Frozen numerical configuration",
        "",
        "- `ttcrpy==1.4.2 / 3-D rectilinear / FSM / tt_from_rp=False`",
        "- Node-centered velocity sampled directly at 1.25 km forward nodes",
        "- No ray paths required for authoritative label generation",
        "",
    ]
    (output_dir / "production_runtime_storage_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return get_repo_root() / path


__all__ = [
    "PRODUCTION_ID",
    "PRODUCTION_OUTPUT_DIR",
    "ProductionConfig",
    "build_production_manifest",
    "build_target_split_manifest",
    "run_production",
    "validate_production_integrity",
]
