"""Expanded paired-sample workflow within the frozen three-scenario ML family."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.utils.paths import get_repo_root
from tomobench.workflow import run_first_vertical_slice


@dataclass(frozen=True)
class MLExpandedPairingArtifact:
    """One case-specific paired observation-target artifact set."""

    case_id: str
    scenario: str
    observation_dataset_csv: Path
    observation_dataset_metadata: Path
    target_velocity_grid_path: Path
    target_spec_path: Path


@dataclass(frozen=True)
class MLExpandedPairingOutputs:
    """Artifacts written by the expanded paired-sample workflow."""

    manifest_path: Path
    pairings: tuple[MLExpandedPairingArtifact, ...]


def prepare_expanded_ml_supervised_pairings(
    settings: BenchmarkSettings | None = None,
) -> MLExpandedPairingOutputs:
    """Prepare additional paired samples within the frozen three-scenario family."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    expanded_settings = loaded_settings.machine_learning.expanded_pairing
    manifest_path = _resolve_repo_path(expanded_settings.manifest_path, repo_root)
    dataset_dir = manifest_path.parent
    generated_dir = (
        repo_root
        / loaded_settings.outputs.generated_base_dir
        / "ml_supervised"
        / expanded_settings.batch_id
    )
    figures_dir = (
        repo_root
        / loaded_settings.outputs.figures_dir
        / "ml_supervised"
        / expanded_settings.batch_id
    )
    notes_dir = (
        repo_root
        / loaded_settings.outputs.experiment_notes_dir
        / "ml_supervised"
        / expanded_settings.batch_id
    )

    pairing_items: list[dict[str, Any]] = []
    pairing_artifacts: list[MLExpandedPairingArtifact] = []
    target_contract: dict[str, Any] | None = None
    for case in expanded_settings.cases:
        case_settings = _settings_for_case(
            loaded_settings,
            case_id=case.case_id,
            scenario=case.scenario,
            generated_dir=generated_dir,
            dataset_dir=dataset_dir,
            figures_dir=figures_dir,
            notes_dir=notes_dir,
            block_anomaly_override=_block_override(case),
            faulted_override=_fault_override(case),
            salt_dome_override=_salt_override(case),
            dyke_intrusion_override=_dyke_override(case),
        )
        outputs = run_first_vertical_slice(
            settings=case_settings,
            simulation_method=expanded_settings.simulation_method,
            velocity_scenario=case.scenario,
        )
        observation_metadata = _read_json(outputs.dataset_metadata)
        target_spec = _read_json(outputs.target_spec)
        current_contract = _extract_target_contract(target_spec)
        target_contract = _merge_target_contract(target_contract, current_contract)
        pairing_items.append(
            {
                "pairing_item_id": f"{expanded_settings.batch_id}_{case.case_id}",
                "case_id": case.case_id,
                "scenario": case.scenario,
                "input_record_level": observation_metadata["record_level"],
                "input_dataset_id": observation_metadata["dataset_id"],
                "input_dataset_csv_path": _relative_to_repo_or_absolute(
                    outputs.dataset_csv, repo_root
                ),
                "input_dataset_metadata_path": _relative_to_repo_or_absolute(
                    outputs.dataset_metadata,
                    repo_root,
                ),
                "input_observation_count": observation_metadata["row_count"],
                "input_simulation_method": observation_metadata["simulation_method"],
                "target_velocity_grid_path": str(
                    observation_metadata["source_files"]["velocity_grid"]
                ),
                "target_spec_path": _relative_to_repo_or_absolute(outputs.target_spec, repo_root),
                "target_vector_length": target_spec["target_vector_length"],
                "grid_shape": target_spec["grid_shape"],
                "flattening_order": target_spec["flattening_order"],
                "value_semantics": target_spec["value_semantics"],
                "case_overrides": _case_overrides_payload(case),
            }
        )
        pairing_artifacts.append(
            MLExpandedPairingArtifact(
                case_id=case.case_id,
                scenario=case.scenario,
                observation_dataset_csv=outputs.dataset_csv,
                observation_dataset_metadata=outputs.dataset_metadata,
                target_velocity_grid_path=outputs.velocity_grid,
                target_spec_path=outputs.target_spec,
            )
        )

    payload = {
        "artifact_type": "ml_expanded_supervised_pairing_manifest",
        "batch_id": expanded_settings.batch_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "case_count": len(pairing_items),
        "scenario_batch": [case.scenario for case in expanded_settings.cases],
        "unique_scenarios": sorted({case.scenario for case in expanded_settings.cases}),
        "input_simulation_method": expanded_settings.simulation_method,
        "target_contract": target_contract,
        "items": pairing_items,
        "notes": [
            "This manifest increases paired sample count while preserving the frozen target-grid representation contract.",
            "Each case uses the same target-grid contract while allowing bounded geological-family and scenario-parameter variation.",
            "The observation-side simulator remains explicit so later ML comparisons stay traceable.",
        ],
    }
    _save_json(payload, manifest_path)
    return MLExpandedPairingOutputs(
        manifest_path=manifest_path,
        pairings=tuple(pairing_artifacts),
    )


def _settings_for_case(
    settings: BenchmarkSettings,
    case_id: str,
    scenario: str,
    generated_dir: Path,
    dataset_dir: Path,
    figures_dir: Path,
    notes_dir: Path,
    block_anomaly_override: dict[str, object] | None,
    faulted_override: dict[str, object] | None,
    salt_dome_override: dict[str, object] | None,
    dyke_intrusion_override: dict[str, object] | None,
) -> BenchmarkSettings:
    velocity_model_generation = replace(
        settings.velocity_model_generation,
        default_scenario=scenario,
    )
    if block_anomaly_override is not None:
        velocity_model_generation = replace(
            velocity_model_generation,
            block_anomaly=replace(
                velocity_model_generation.block_anomaly,
                center_km=block_anomaly_override["center_km"],
                size_km=block_anomaly_override["size_km"],
                velocity_delta_km_per_s=block_anomaly_override["velocity_delta_km_per_s"],
            ),
        )
    if faulted_override is not None:
        velocity_model_generation = replace(
            velocity_model_generation,
            faulted=replace(
                velocity_model_generation.faulted,
                fault_x_km=faulted_override["fault_x_km"],
                fault_y_km=faulted_override["fault_y_km"],
                strike_deg=faulted_override["strike_deg"],
                dip_deg=faulted_override["dip_deg"],
                dip_direction=faulted_override["dip_direction"],
                positive_side=faulted_override["positive_side"],
                velocity_offset_km_per_s=faulted_override["velocity_offset_km_per_s"],
            ),
        )
    if salt_dome_override is not None:
        velocity_model_generation = replace(
            velocity_model_generation,
            salt_dome=replace(
                velocity_model_generation.salt_dome,
                center_km=salt_dome_override["center_km"],
                radii_km=salt_dome_override["radii_km"],
                body_velocity_km_per_s=salt_dome_override["body_velocity_km_per_s"],
            ),
        )
    if dyke_intrusion_override is not None:
        velocity_model_generation = replace(
            velocity_model_generation,
            dyke_intrusion=replace(
                velocity_model_generation.dyke_intrusion,
                center_km=dyke_intrusion_override["center_km"],
                strike_deg=dyke_intrusion_override["strike_deg"],
                length_km=dyke_intrusion_override["length_km"],
                width_km=dyke_intrusion_override["width_km"],
                top_depth_km=dyke_intrusion_override["top_depth_km"],
                bottom_depth_km=dyke_intrusion_override["bottom_depth_km"],
                body_velocity_km_per_s=dyke_intrusion_override["body_velocity_km_per_s"],
            ),
        )
    return replace(
        settings,
        dataset_export=replace(settings.dataset_export, base_dir=dataset_dir),
        outputs=replace(
            settings.outputs,
            generated_base_dir=generated_dir,
            figures_dir=figures_dir,
            experiment_notes_dir=notes_dir,
        ),
        velocity_model_generation=velocity_model_generation,
        vertical_slice=replace(
            settings.vertical_slice,
            experiment_id=f"{settings.machine_learning.expanded_pairing.batch_id}_{case_id}",
            velocity_model_id=f"{settings.vertical_slice.velocity_model_id}_{case_id}",
            simulation_id=f"{settings.vertical_slice.simulation_id}_{case_id}",
            dataset_id=f"{settings.vertical_slice.dataset_id}_{case_id}",
        ),
    )


def _block_override(case: object) -> dict[str, object] | None:
    if getattr(case, "scenario") != "block_anomaly":
        return None
    if getattr(case, "block_anomaly_center_km") is None:
        return None
    return {
        "center_km": getattr(case, "block_anomaly_center_km"),
        "size_km": getattr(case, "block_anomaly_size_km"),
        "velocity_delta_km_per_s": getattr(case, "block_anomaly_velocity_delta_km_per_s"),
    }


def _fault_override(case: object) -> dict[str, object] | None:
    if getattr(case, "scenario") != "faulted":
        return None
    if getattr(case, "fault_x_km") is None:
        return None
    return {
        "fault_x_km": getattr(case, "fault_x_km"),
        "fault_y_km": getattr(case, "fault_y_km"),
        "strike_deg": getattr(case, "fault_strike_deg"),
        "dip_deg": getattr(case, "fault_dip_deg"),
        "dip_direction": getattr(case, "fault_dip_direction"),
        "positive_side": getattr(case, "fault_positive_side"),
        "velocity_offset_km_per_s": getattr(case, "fault_velocity_offset_km_per_s"),
    }


def _case_overrides_payload(case: object) -> dict[str, object]:
    payload: dict[str, object] = {}
    block_override = _block_override(case)
    fault_override = _fault_override(case)
    salt_override = _salt_override(case)
    dyke_override = _dyke_override(case)
    if block_override is not None:
        payload["block_anomaly"] = block_override
    if fault_override is not None:
        payload["faulted"] = fault_override
    if salt_override is not None:
        payload["salt_dome"] = salt_override
    if dyke_override is not None:
        payload["dyke_intrusion"] = dyke_override
    return payload


def _salt_override(case: object) -> dict[str, object] | None:
    if getattr(case, "scenario") != "salt_dome":
        return None
    if getattr(case, "salt_dome_center_km") is None:
        return None
    return {
        "center_km": getattr(case, "salt_dome_center_km"),
        "radii_km": getattr(case, "salt_dome_radii_km"),
        "body_velocity_km_per_s": (
            getattr(case, "salt_dome_body_velocity_km_per_s")
            if getattr(case, "salt_dome_body_velocity_km_per_s", None) is not None
            else getattr(case, "salt_dome_velocity_delta_km_per_s")
        ),
    }


def _dyke_override(case: object) -> dict[str, object] | None:
    if getattr(case, "scenario") != "dyke_intrusion":
        return None
    if getattr(case, "dyke_intrusion_center_km") is None:
        return None
    return {
        "center_km": getattr(case, "dyke_intrusion_center_km"),
        "strike_deg": getattr(case, "dyke_intrusion_strike_deg"),
        "length_km": getattr(case, "dyke_intrusion_length_km"),
        "width_km": getattr(case, "dyke_intrusion_width_km"),
        "top_depth_km": getattr(case, "dyke_intrusion_top_depth_km"),
        "bottom_depth_km": getattr(case, "dyke_intrusion_bottom_depth_km"),
        "body_velocity_km_per_s": (
            getattr(case, "dyke_intrusion_body_velocity_km_per_s")
            if getattr(case, "dyke_intrusion_body_velocity_km_per_s", None) is not None
            else getattr(case, "dyke_intrusion_velocity_delta_km_per_s")
        ),
    }


def _extract_target_contract(target_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_representation": target_spec["target_representation"],
        "representation": target_spec["representation"],
        "value_field": target_spec["value_field"],
        "value_semantics": target_spec["value_semantics"],
        "value_location": target_spec["value_location"],
        "flattening_order": target_spec["flattening_order"],
        "grid_shape": target_spec["grid_shape"],
        "target_vector_length": target_spec["target_vector_length"],
        "coordinate_system": target_spec["coordinate_system"],
        "grid_spacing_km": target_spec["grid_spacing_km"],
    }


def _merge_target_contract(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if current is None:
        return dict(candidate)
    if current != candidate:
        raise ValueError("Expanded paired samples drifted from the frozen target contract.")
    return current


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Expected a JSON object.")
    return dict(payload)


def _save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_to_repo_or_absolute(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()
