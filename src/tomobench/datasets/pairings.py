"""Observation-target pairing workflow for the first supervised ML slice."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.datasets.target_batches import prepare_first_ml_target_batch
from tomobench.utils.paths import get_repo_root
from tomobench.workflow import run_first_vertical_slice


@dataclass(frozen=True)
class MLSupervisedPairingArtifact:
    """One scenario-matched observation-target pairing entry."""

    scenario: str
    observation_dataset_csv: Path
    observation_dataset_metadata: Path
    target_velocity_grid_path: Path
    target_spec_path: Path


@dataclass(frozen=True)
class MLSupervisedPairingOutputs:
    """Paths produced by the first supervised pairing workflow."""

    manifest_path: Path
    pairings: tuple[MLSupervisedPairingArtifact, ...]


def prepare_first_ml_supervised_pairings(
    settings: BenchmarkSettings | None = None,
) -> MLSupervisedPairingOutputs:
    """Prepare scenario-matched observation and target references for later training."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    pairing_settings = loaded_settings.machine_learning.first_supervised_pairing
    manifest_path = _resolve_repo_path(pairing_settings.manifest_path, repo_root)

    target_batch_outputs = prepare_first_ml_target_batch(loaded_settings)
    target_batch_manifest = _read_json(target_batch_outputs.manifest_path)
    target_items_by_scenario = {
        str(item["scenario"]): item for item in _dict_list(target_batch_manifest["items"])
    }
    target_contract = _dict(target_batch_manifest["target_contract"])

    pairing_items: list[dict[str, Any]] = []
    pairing_artifacts: list[MLSupervisedPairingArtifact] = []
    for scenario in pairing_settings.scenarios:
        slice_outputs = run_first_vertical_slice(
            settings=loaded_settings,
            simulation_method=pairing_settings.simulation_method,
            velocity_scenario=scenario,
        )
        target_item = target_items_by_scenario[scenario]
        observation_target_spec = _read_json(slice_outputs.target_spec)
        _assert_target_contract_consistency(
            scenario=scenario,
            observation_target_spec=observation_target_spec,
            target_item=target_item,
            target_contract=target_contract,
        )
        observation_metadata = _read_json(slice_outputs.dataset_metadata)
        pairing_items.append(
            {
                "pairing_item_id": f"{pairing_settings.pairing_id}_{scenario}",
                "scenario": scenario,
                "input_record_level": observation_metadata["record_level"],
                "input_dataset_id": observation_metadata["dataset_id"],
                "input_dataset_csv_path": _relative_to_repo_or_absolute(
                    slice_outputs.dataset_csv,
                    repo_root,
                ),
                "input_dataset_metadata_path": _relative_to_repo_or_absolute(
                    slice_outputs.dataset_metadata,
                    repo_root,
                ),
                "input_observation_count": observation_metadata["row_count"],
                "input_simulation_method": observation_metadata["simulation_method"],
                "input_target_spec_path": _relative_to_repo_or_absolute(
                    slice_outputs.target_spec,
                    repo_root,
                ),
                "input_velocity_grid_path": str(
                    observation_metadata["source_files"]["velocity_grid"]
                ),
                "input_velocity_model_path": str(
                    observation_metadata["source_files"]["velocity_model"]
                ),
                "target_batch_id": target_batch_manifest["batch_id"],
                "target_grid_id": target_item["grid_id"],
                "target_velocity_grid_path": str(target_item["velocity_grid_path"]),
                "target_spec_path": str(target_item["target_spec_path"]),
                "target_vector_length": target_item["target_vector_length"],
                "grid_shape": target_item["grid_shape"],
                "flattening_order": target_item["flattening_order"],
                "value_semantics": target_item["value_semantics"],
            }
        )
        pairing_artifacts.append(
            MLSupervisedPairingArtifact(
                scenario=scenario,
                observation_dataset_csv=slice_outputs.dataset_csv,
                observation_dataset_metadata=slice_outputs.dataset_metadata,
                target_velocity_grid_path=_resolve_repo_path(
                    Path(str(target_item["velocity_grid_path"])),
                    repo_root,
                ),
                target_spec_path=_resolve_repo_path(
                    Path(str(target_item["target_spec_path"])),
                    repo_root,
                ),
            )
        )

    payload = {
        "artifact_type": "ml_supervised_pairing_manifest",
        "pairing_id": pairing_settings.pairing_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scenario_batch": list(pairing_settings.scenarios),
        "pairing_count": len(pairing_items),
        "input_simulation_method": pairing_settings.simulation_method,
        "target_batch_manifest_path": _relative_to_repo_or_absolute(
            target_batch_outputs.manifest_path,
            repo_root,
        ),
        "target_contract": target_contract,
        "items": pairing_items,
        "notes": [
            "Each pairing item links one scenario-matched observation dataset to one frozen target-grid artifact.",
            "This manifest is intended for later supervised dataset assembly and does not yet implement training or split orchestration.",
            "The current observation-side simulator for this pairing layer is configured centrally and kept explicit for review.",
        ],
    }
    _save_json(payload, manifest_path)
    return MLSupervisedPairingOutputs(
        manifest_path=manifest_path,
        pairings=tuple(pairing_artifacts),
    )


def _assert_target_contract_consistency(
    scenario: str,
    observation_target_spec: dict[str, Any],
    target_item: dict[str, Any],
    target_contract: dict[str, Any],
) -> None:
    if observation_target_spec["current_scenario"] != scenario:
        raise ValueError(f"Observation target spec scenario mismatch for {scenario}.")
    if target_item["scenario"] != scenario:
        raise ValueError(f"Target batch scenario mismatch for {scenario}.")
    comparisons = {
        "target_representation": observation_target_spec["target_representation"],
        "representation": observation_target_spec["representation"],
        "value_field": observation_target_spec["value_field"],
        "value_semantics": observation_target_spec["value_semantics"],
        "value_location": observation_target_spec["value_location"],
        "flattening_order": observation_target_spec["flattening_order"],
        "grid_shape": observation_target_spec["grid_shape"],
        "target_vector_length": observation_target_spec["target_vector_length"],
        "coordinate_system": observation_target_spec["coordinate_system"],
        "grid_spacing_km": observation_target_spec["grid_spacing_km"],
    }
    if comparisons != target_contract:
        raise ValueError(
            f"Frozen target contract drift detected while pairing scenario {scenario}."
        )
    if (
        observation_target_spec["current_scenario"]
        not in observation_target_spec["allowed_scenarios"]
    ):
        raise ValueError(f"Scenario {scenario} is not listed in the observation target spec.")


def _save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _dict(payload)


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
