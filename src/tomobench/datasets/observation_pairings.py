"""Observation-driven paired-case workflow for the five-family supervised baseline."""

from __future__ import annotations

import json
import csv
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tomobench.config.settings import (
    MachineLearningObservationPairingSettings,
    BenchmarkSettings,
    load_settings,
)
from tomobench.datasets.builder import dataset_columns
from tomobench.simulation.travel_times import PSEUDO_BENDING_3D_METHOD
from tomobench.utils.paths import get_repo_root
from tomobench.workflow import run_first_vertical_slice


@dataclass(frozen=True)
class MLObservationPairingArtifact:
    """One case-specific paired observation-target artifact set."""

    case_id: str
    scenario: str
    observation_dataset_csv: Path
    observation_dataset_metadata: Path
    ray_path_sidecar_path: Path | None
    sensitivity_sidecar_path: Path | None
    sensitivity_metadata_path: Path | None
    target_velocity_grid_path: Path
    target_spec_path: Path


@dataclass(frozen=True)
class MLObservationPairingOutputs:
    """Artifacts written by the observation-driven paired-sample workflow."""

    manifest_path: Path
    pairings: tuple[MLObservationPairingArtifact, ...]


@dataclass(frozen=True)
class CorpusValidationResult:
    """Validation checks and errors for a generated observation corpus."""

    manifest_path: Path
    checks: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def prepare_observation_ml_supervised_pairings(
    settings: BenchmarkSettings | None = None,
) -> MLObservationPairingOutputs:
    """Prepare the larger five-family paired corpus for observation-level ML."""
    loaded_settings = settings or load_settings()
    return _prepare_observation_pairings(
        loaded_settings,
        pairing_settings=loaded_settings.machine_learning.observation_pairing,
        artifact_type="ml_observation_supervised_pairing_manifest",
        notes=(
            "This manifest is the first observation-driven supervised corpus for the frozen full-grid target contract.",
            "Cases vary only within the existing five geological families so the target representation remains unchanged.",
            "The observation-side simulator remains explicit and fixed across the corpus for traceable ML training.",
            "Acquisition geometry variation is introduced through bounded station-layout and earthquake-catalog overrides while keeping observation count fixed.",
        ),
    )


def prepare_paper_pseudo_bending_observation_corpus(
    settings: BenchmarkSettings | None = None,
) -> MLObservationPairingOutputs:
    """Prepare the paper-facing pseudo-bending observation corpus."""
    loaded_settings = settings or load_settings()
    pairing_settings = loaded_settings.machine_learning.paper_pseudo_bending_observation_corpus
    if pairing_settings.simulation_method != PSEUDO_BENDING_3D_METHOD:
        raise ValueError("The paper observation corpus must use pseudo_bending_3d.")
    outputs = _prepare_observation_pairings(
        loaded_settings,
        pairing_settings=pairing_settings,
        artifact_type="paper_pseudo_bending_observation_corpus_manifest",
        notes=(
            "This is the paper-facing ML observation corpus after pseudo_bending_3d became the default simulator.",
            "Historical eikonal/prototype observation corpora are intentionally not overwritten.",
            "Cases inherit the established five-family parameter variants and acquisition-geometry profiles from the earlier observation corpus design.",
            "Every item records direct links to generated geometry, velocity, target, observation, metadata, simulator, config, and seed provenance.",
        ),
    )
    validation = validate_paper_pseudo_bending_observation_corpus(
        outputs.manifest_path,
        loaded_settings,
    )
    if validation.errors:
        raise ValueError(
            "Generated paper pseudo-bending observation corpus failed validation: "
            + "; ".join(validation.errors)
        )
    return outputs


def _prepare_observation_pairings(
    loaded_settings: BenchmarkSettings,
    *,
    pairing_settings: MachineLearningObservationPairingSettings,
    artifact_type: str,
    notes: tuple[str, ...],
) -> MLObservationPairingOutputs:
    """Prepare observation-target pairings for a selected observation-corpus config."""
    repo_root = get_repo_root()
    manifest_path = _resolve_repo_path(pairing_settings.manifest_path, repo_root)
    dataset_dir = manifest_path.parent
    generated_dir = (
        repo_root
        / loaded_settings.outputs.generated_base_dir
        / "ml_supervised"
        / pairing_settings.batch_id
    )
    figures_dir = (
        repo_root
        / loaded_settings.outputs.figures_dir
        / "ml_supervised"
        / pairing_settings.batch_id
    )
    notes_dir = (
        repo_root
        / loaded_settings.outputs.experiment_notes_dir
        / "ml_supervised"
        / pairing_settings.batch_id
    )

    pairing_items: list[dict[str, Any]] = []
    pairing_artifacts: list[MLObservationPairingArtifact] = []
    target_contract: dict[str, Any] | None = None
    for case in pairing_settings.cases:
        case_settings = _settings_for_case(
            loaded_settings,
            pairing_settings,
            case_id=case.case_id,
            scenario=case.scenario,
            generated_dir=generated_dir,
            dataset_dir=dataset_dir,
            figures_dir=figures_dir,
            notes_dir=notes_dir,
            station_generator_mode=case.station_generator_mode,
            station_seed=case.station_seed,
            earthquake_seed=case.earthquake_seed,
            layered_velocities_override=case.layered_velocities_km_per_s,
            block_anomaly_override=_block_override(case),
            faulted_override=_fault_override(case),
            salt_dome_override=_salt_override(case),
            dyke_intrusion_override=_dyke_override(case),
        )
        outputs = run_first_vertical_slice(
            settings=case_settings,
            simulation_method=pairing_settings.simulation_method,
            velocity_scenario=case.scenario,
        )
        observation_metadata = _read_json(outputs.dataset_metadata)
        target_spec = _read_json(outputs.target_spec)
        travel_time_metadata = _read_json(outputs.travel_times_json)
        current_contract = _extract_target_contract(target_spec)
        target_contract = _merge_target_contract(target_contract, current_contract)
        source_files = observation_metadata["source_files"]
        ray_path_sidecar_path = source_files.get("ray_paths")
        sensitivity_sidecar_path = source_files.get("ray_cell_sensitivity")
        sensitivity_metadata_path = source_files.get("ray_cell_sensitivity_metadata")
        pairing_items.append(
            {
                "pairing_item_id": f"{pairing_settings.batch_id}_{case.case_id}",
                "corpus_id": pairing_settings.batch_id,
                "batch_id": pairing_settings.batch_id,
                "case_id": case.case_id,
                "scenario": case.scenario,
                "family": case.scenario,
                "parameter_variant_id": case.parameter_variant_id or case.case_id,
                "geometry_profile_id": case.geometry_profile_id,
                "input_record_level": observation_metadata["record_level"],
                "input_dataset_id": observation_metadata["dataset_id"],
                "schema_version": observation_metadata["schema_version"],
                "simulator_name": pairing_settings.simulation_method,
                "input_dataset_csv_path": _relative_to_repo_or_absolute(
                    outputs.dataset_csv, repo_root
                ),
                "input_dataset_metadata_path": _relative_to_repo_or_absolute(
                    outputs.dataset_metadata,
                    repo_root,
                ),
                "observation_csv_path": _relative_to_repo_or_absolute(
                    outputs.dataset_csv, repo_root
                ),
                "observation_metadata_path": _relative_to_repo_or_absolute(
                    outputs.dataset_metadata,
                    repo_root,
                ),
                "ray_path_sidecar_path": ray_path_sidecar_path,
                "sensitivity_sidecar_path": sensitivity_sidecar_path,
                "sensitivity_metadata_path": sensitivity_metadata_path,
                "input_observation_count": observation_metadata["row_count"],
                "row_count": observation_metadata["row_count"],
                "input_simulation_method": observation_metadata["simulation_method"],
                "simulation_method": observation_metadata["simulation_method"],
                "station_configuration_path": source_files["station_configuration"],
                "earthquake_configuration_path": source_files["earthquake_configuration"],
                "velocity_model_path": source_files["velocity_model"],
                "velocity_model_grid_path": source_files["velocity_grid"],
                "target_velocity_grid_path": source_files["velocity_grid"],
                "target_spec_path": _relative_to_repo_or_absolute(outputs.target_spec, repo_root),
                "target_grid_path": source_files["velocity_grid"],
                "target_vector_length": target_spec["target_vector_length"],
                "grid_shape": target_spec["grid_shape"],
                "flattening_order": target_spec["flattening_order"],
                "value_semantics": target_spec["value_semantics"],
                "feature_fields": list(_observation_feature_fields()),
                "target_representation": observation_metadata["target_representation"],
                "station_generation_mode": case_settings.station_generation.default_mode,
                "station_seed": case_settings.random_seeds.stations,
                "earthquake_seed": case_settings.random_seeds.earthquakes,
                "velocity_model_seed": case_settings.random_seeds.velocity_models,
                "station_count": case_settings.vertical_slice.station_count,
                "earthquake_count": case_settings.vertical_slice.earthquake_count,
                "seeds_used": {
                    "global": case_settings.random_seeds.global_seed,
                    "stations": case_settings.random_seeds.stations,
                    "earthquakes": case_settings.random_seeds.earthquakes,
                    "velocity_models": case_settings.random_seeds.velocity_models,
                    "machine_learning": case_settings.random_seeds.machine_learning,
                },
                "pseudo_bending_solver": travel_time_metadata.get("pseudo_bending_solver"),
                "simulation_config_snapshot": {
                    "phase": case_settings.simulation.phase,
                    "method": case_settings.simulation.method,
                    "first_arrival_only": case_settings.simulation.first_arrival_only,
                    "allow_curved_rays": case_settings.simulation.allow_curved_rays,
                    "pseudo_bending_solver": travel_time_metadata.get("pseudo_bending_solver"),
                    "config_path": "config/benchmark_config.yaml",
                },
                "case_overrides": _case_overrides_payload(case),
            }
        )
        pairing_artifacts.append(
            MLObservationPairingArtifact(
                case_id=case.case_id,
                scenario=case.scenario,
                observation_dataset_csv=outputs.dataset_csv,
                observation_dataset_metadata=outputs.dataset_metadata,
                ray_path_sidecar_path=(
                    _resolve_manifest_path(
                        str(ray_path_sidecar_path),
                        repo_root,
                        manifest_path,
                    )
                    if ray_path_sidecar_path
                    else None
                ),
                sensitivity_sidecar_path=(
                    _resolve_manifest_path(
                        str(sensitivity_sidecar_path),
                        repo_root,
                        manifest_path,
                    )
                    if sensitivity_sidecar_path
                    else None
                ),
                sensitivity_metadata_path=(
                    _resolve_manifest_path(
                        str(sensitivity_metadata_path),
                        repo_root,
                        manifest_path,
                    )
                    if sensitivity_metadata_path
                    else None
                ),
                target_velocity_grid_path=outputs.velocity_grid,
                target_spec_path=outputs.target_spec,
            )
        )

    payload = {
        "artifact_type": artifact_type,
        "corpus_id": pairing_settings.batch_id,
        "batch_id": pairing_settings.batch_id,
        "schema_version": loaded_settings.dataset_export.schema_version,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "case_count": len(pairing_items),
        "scenario_batch": [case.scenario for case in pairing_settings.cases],
        "unique_scenarios": sorted({case.scenario for case in pairing_settings.cases}),
        "simulator_name": pairing_settings.simulation_method,
        "input_simulation_method": pairing_settings.simulation_method,
        "pseudo_bending_solver": _pseudo_bending_solver_payload(loaded_settings),
        "config_snapshot_reference": "config/benchmark_config.yaml",
        "station_count": pairing_settings.station_count,
        "earthquake_count": pairing_settings.earthquake_count,
        "grid_rows": pairing_settings.grid_rows,
        "grid_cols": pairing_settings.grid_cols,
        "feature_fields": list(_observation_feature_fields()),
        "target_representation": loaded_settings.machine_learning.target_representation,
        "target_contract": target_contract,
        "items": pairing_items,
        "notes": list(notes),
    }
    _save_json(payload, manifest_path)
    return MLObservationPairingOutputs(
        manifest_path=manifest_path,
        pairings=tuple(pairing_artifacts),
    )


def _settings_for_case(
    settings: BenchmarkSettings,
    pairing_settings: MachineLearningObservationPairingSettings,
    case_id: str,
    scenario: str,
    generated_dir: Path,
    dataset_dir: Path,
    figures_dir: Path,
    notes_dir: Path,
    station_generator_mode: str | None,
    station_seed: int | None,
    earthquake_seed: int | None,
    layered_velocities_override: tuple[float, ...] | None,
    block_anomaly_override: dict[str, object] | None,
    faulted_override: dict[str, object] | None,
    salt_dome_override: dict[str, object] | None,
    dyke_intrusion_override: dict[str, object] | None,
) -> BenchmarkSettings:
    velocity_model_generation = replace(
        settings.velocity_model_generation,
        default_scenario=scenario,
    )
    if layered_velocities_override is not None:
        velocity_model_generation = replace(
            velocity_model_generation,
            layered=replace(
                velocity_model_generation.layered,
                velocities_km_per_s=layered_velocities_override,
            ),
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
    station_generation = replace(
        settings.station_generation,
        default_mode=station_generator_mode or settings.station_generation.default_mode,
        station_count=pairing_settings.station_count,
        grid_rows=pairing_settings.grid_rows,
        grid_cols=pairing_settings.grid_cols,
    )
    random_seeds = replace(
        settings.random_seeds,
        stations=station_seed or settings.random_seeds.stations,
        earthquakes=earthquake_seed or settings.random_seeds.earthquakes,
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
        station_generation=station_generation,
        velocity_model_generation=velocity_model_generation,
        random_seeds=random_seeds,
        vertical_slice=replace(
            settings.vertical_slice,
            experiment_id=f"{pairing_settings.batch_id}_{case_id}",
            station_configuration_id=f"{settings.vertical_slice.station_configuration_id}_{case_id}",
            earthquake_configuration_id=f"{settings.vertical_slice.earthquake_configuration_id}_{case_id}",
            velocity_model_id=f"{settings.vertical_slice.velocity_model_id}_{case_id}",
            simulation_id=f"{settings.vertical_slice.simulation_id}_{case_id}",
            dataset_id=f"{settings.vertical_slice.dataset_id}_{case_id}",
            station_count=pairing_settings.station_count,
            earthquake_count=pairing_settings.earthquake_count,
            case_id=case_id,
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


def validate_paper_pseudo_bending_observation_corpus(
    manifest_path: Path | None = None,
    settings: BenchmarkSettings | None = None,
) -> CorpusValidationResult:
    """Validate the paper pseudo-bending observation corpus manifest and files."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    pairing_settings = loaded_settings.machine_learning.paper_pseudo_bending_observation_corpus
    resolved_manifest_path = _resolve_repo_path(
        manifest_path or pairing_settings.manifest_path,
        repo_root,
    )
    checks: list[str] = []
    errors: list[str] = []
    if not resolved_manifest_path.is_file():
        return CorpusValidationResult(
            manifest_path=resolved_manifest_path,
            checks=tuple(checks),
            errors=(f"Manifest does not exist: {resolved_manifest_path}",),
        )

    manifest = _read_json(resolved_manifest_path)
    if manifest.get("corpus_id") != pairing_settings.batch_id:
        errors.append("Manifest corpus_id does not match configured paper corpus batch_id.")
    if manifest.get("batch_id") != pairing_settings.batch_id:
        errors.append("Manifest batch_id does not match configured paper corpus batch_id.")
    if manifest.get("simulator_name") != PSEUDO_BENDING_3D_METHOD:
        errors.append("Manifest simulator_name is not pseudo_bending_3d.")
    if manifest.get("input_simulation_method") != PSEUDO_BENDING_3D_METHOD:
        errors.append("Manifest input_simulation_method is not pseudo_bending_3d.")

    items = manifest.get("items")
    if not isinstance(items, list):
        return CorpusValidationResult(
            manifest_path=resolved_manifest_path,
            checks=tuple(checks),
            errors=tuple(errors + ["Manifest items must be a list."]),
        )
    expected_case_ids = {case.case_id for case in pairing_settings.cases}
    observed_case_ids = {
        str(item.get("case_id")) for item in items if isinstance(item, dict) and item.get("case_id")
    }
    if observed_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - observed_case_ids)
        extra = sorted(observed_case_ids - expected_case_ids)
        errors.append(f"Case set mismatch. Missing={missing}; extra={extra}.")
    else:
        checks.append("All configured scenario/variant/profile cases are present.")

    observed_families = {
        str(item.get("scenario"))
        for item in items
        if isinstance(item, dict) and item.get("scenario")
    }
    expected_families = {"layered", "block_anomaly", "faulted", "salt_dome", "dyke_intrusion"}
    if observed_families != expected_families:
        errors.append(
            f"Manifest families must be {sorted(expected_families)}; got {sorted(observed_families)}."
        )
    else:
        checks.append("All five geological families are present.")

    expected_row_count = pairing_settings.station_count * pairing_settings.earthquake_count
    required_columns = set(dataset_columns())
    required_path_fields = {
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "receiver_x_km",
        "receiver_y_km",
        "receiver_z_km",
        "travel_time_s",
        "path_length_km",
        "simulation_method",
    }
    target_contract = manifest.get("target_contract")
    target_grid_shape: dict[str, Any] | None = None
    for item in items:
        if not isinstance(item, dict):
            errors.append("Manifest item is not a JSON object.")
            continue
        case_id = str(item.get("case_id"))
        if item.get("row_count") != expected_row_count:
            errors.append(f"{case_id}: row_count is not {expected_row_count}.")
        if item.get("simulation_method") != PSEUDO_BENDING_3D_METHOD:
            errors.append(f"{case_id}: manifest simulation_method is not pseudo_bending_3d.")
        if item.get("simulator_name") != PSEUDO_BENDING_3D_METHOD:
            errors.append(f"{case_id}: manifest simulator_name is not pseudo_bending_3d.")

        csv_path = _resolve_manifest_path(
            str(item.get("observation_csv_path") or item.get("input_dataset_csv_path") or ""),
            repo_root,
            resolved_manifest_path,
        )
        ray_path_sidecar_path = _resolve_manifest_path(
            str(item.get("ray_path_sidecar_path") or ""),
            repo_root,
            resolved_manifest_path,
        )
        sensitivity_sidecar_path = _resolve_manifest_path(
            str(item.get("sensitivity_sidecar_path") or ""),
            repo_root,
            resolved_manifest_path,
        )
        sensitivity_metadata_path = _resolve_manifest_path(
            str(item.get("sensitivity_metadata_path") or ""),
            repo_root,
            resolved_manifest_path,
        )
        metadata_path = _resolve_manifest_path(
            str(
                item.get("observation_metadata_path")
                or item.get("input_dataset_metadata_path")
                or ""
            ),
            repo_root,
            resolved_manifest_path,
        )
        target_spec_path = _resolve_manifest_path(
            str(item.get("target_spec_path") or ""),
            repo_root,
            resolved_manifest_path,
        )
        for label, path in (
            ("observation CSV", csv_path),
            ("ray-path sidecar", ray_path_sidecar_path),
            ("ray-cell sensitivity sidecar", sensitivity_sidecar_path),
            ("ray-cell sensitivity metadata", sensitivity_metadata_path),
            ("observation metadata", metadata_path),
            ("target spec", target_spec_path),
            (
                "station configuration",
                _resolve_manifest_path(
                    str(item.get("station_configuration_path") or ""),
                    repo_root,
                    resolved_manifest_path,
                ),
            ),
            (
                "earthquake configuration",
                _resolve_manifest_path(
                    str(item.get("earthquake_configuration_path") or ""),
                    repo_root,
                    resolved_manifest_path,
                ),
            ),
            (
                "velocity model/grid",
                _resolve_manifest_path(
                    str(item.get("target_velocity_grid_path") or ""),
                    repo_root,
                    resolved_manifest_path,
                ),
            ),
        ):
            if not path.is_file():
                errors.append(f"{case_id}: missing {label}: {path}")

        if csv_path.is_file():
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or ())
                rows = list(reader)
            csv_observation_ids = {row.get("observation_id") for row in rows}
            if len(rows) != expected_row_count:
                errors.append(f"{case_id}: CSV row count is not {expected_row_count}.")
            if not required_columns.issubset(fieldnames):
                errors.append(f"{case_id}: CSV is missing required dataset columns.")
            if not required_path_fields.issubset(fieldnames):
                errors.append(f"{case_id}: CSV is missing required path/travel-time fields.")
            for row in rows:
                if row.get("simulation_method") != PSEUDO_BENDING_3D_METHOD:
                    errors.append(f"{case_id}: CSV contains non-pseudo-bending simulator values.")
                    break
                try:
                    if float(row["travel_time_s"]) <= 0.0:
                        errors.append(f"{case_id}: CSV contains non-positive travel times.")
                        break
                except (KeyError, ValueError):
                    errors.append(f"{case_id}: CSV contains invalid travel_time_s values.")
                    break
            else:
                checks.append(f"{case_id}: observation CSV schema and travel times are valid.")

            if ray_path_sidecar_path.is_file():
                ray_records = _read_jsonl(ray_path_sidecar_path)
                ray_observation_ids = {
                    str(record.get("observation_id")) for record in ray_records
                }
                if ray_observation_ids != csv_observation_ids:
                    errors.append(
                        f"{case_id}: ray-path sidecar observation IDs do not match the CSV."
                    )
                elif len(ray_records) != expected_row_count:
                    errors.append(
                        f"{case_id}: ray-path sidecar row count is not {expected_row_count}."
                    )
                elif not all(
                    record.get("simulation_method") == PSEUDO_BENDING_3D_METHOD
                    for record in ray_records
                ):
                    errors.append(
                        f"{case_id}: ray-path sidecar contains non-pseudo-bending simulator values."
                    )
                elif not all(
                    isinstance(record.get("ray_path"), list)
                    and len(record.get("ray_path", [])) >= 2
                    for record in ray_records
                ):
                    errors.append(f"{case_id}: ray-path sidecar contains invalid ray paths.")
                else:
                    checks.append(
                        f"{case_id}: ray-path sidecar matches observation IDs and method."
                    )

            if sensitivity_sidecar_path.is_file() and ray_path_sidecar_path.is_file():
                sensitivity_records = _read_jsonl(sensitivity_sidecar_path)
                sensitivity_observation_ids = {
                    str(record.get("observation_id")) for record in sensitivity_records
                }
                if sensitivity_observation_ids != csv_observation_ids:
                    errors.append(
                        f"{case_id}: sensitivity sidecar observation IDs do not match the CSV."
                    )
                elif not _sensitivity_lengths_match_ray_paths(
                    sensitivity_records,
                    _read_jsonl(ray_path_sidecar_path),
                ):
                    errors.append(
                        f"{case_id}: sensitivity path lengths do not conserve ray path lengths."
                    )
                elif not all(
                    record.get("simulation_method") == PSEUDO_BENDING_3D_METHOD
                    for record in sensitivity_records
                ):
                    errors.append(
                        f"{case_id}: sensitivity sidecar contains non-pseudo-bending simulator values."
                    )
                else:
                    checks.append(
                        f"{case_id}: sensitivity sidecar matches observations and ray lengths."
                    )

            if sensitivity_metadata_path.is_file():
                sensitivity_metadata = _read_json(sensitivity_metadata_path)
                validation_payload = sensitivity_metadata.get("validation")
                if not isinstance(validation_payload, dict) or not validation_payload.get(
                    "path_length_conservation_passed"
                ):
                    errors.append(
                        f"{case_id}: sensitivity metadata does not record passing path-length validation."
                    )

        if metadata_path.is_file():
            metadata = _read_json(metadata_path)
            if metadata.get("simulation_method") != PSEUDO_BENDING_3D_METHOD:
                errors.append(f"{case_id}: metadata simulation_method is not pseudo_bending_3d.")
            if metadata.get("row_count") != expected_row_count:
                errors.append(f"{case_id}: metadata row_count is not {expected_row_count}.")
            source_files = metadata.get("source_files")
            if not isinstance(source_files, dict) or not source_files.get("ray_paths"):
                errors.append(f"{case_id}: metadata does not link the ray-path sidecar.")
            if not isinstance(source_files, dict) or not source_files.get("ray_cell_sensitivity"):
                errors.append(f"{case_id}: metadata does not link the sensitivity sidecar.")

        if target_spec_path.is_file():
            target_spec = _read_json(target_spec_path)
            current_shape = target_spec.get("grid_shape")
            if target_grid_shape is None and isinstance(current_shape, dict):
                target_grid_shape = current_shape
            elif current_shape != target_grid_shape:
                errors.append(f"{case_id}: target grid shape differs from the first case.")
            if target_contract is not None:
                current_contract = _extract_target_contract(target_spec)
                if current_contract != target_contract:
                    errors.append(f"{case_id}: target spec differs from manifest target contract.")

    if not errors:
        checks.append("Paper pseudo-bending observation corpus validation completed successfully.")
    return CorpusValidationResult(
        manifest_path=resolved_manifest_path,
        checks=tuple(checks),
        errors=tuple(errors),
    )


def _case_overrides_payload(case: object) -> dict[str, object]:
    payload: dict[str, object] = {}
    if (
        getattr(case, "station_generator_mode", None) is not None
        or getattr(case, "station_seed", None) is not None
    ):
        payload["station_generation"] = {
            "mode": getattr(case, "station_generator_mode", None) or "random",
            "station_seed": getattr(case, "station_seed", None),
        }
    if getattr(case, "earthquake_seed", None) is not None:
        payload["earthquake_generation"] = {
            "mode": "random",
            "earthquake_seed": getattr(case, "earthquake_seed", None),
        }
    layered_velocities = getattr(case, "layered_velocities_km_per_s", None)
    block_override = _block_override(case)
    fault_override = _fault_override(case)
    salt_override = _salt_override(case)
    dyke_override = _dyke_override(case)
    if layered_velocities is not None:
        payload["layered"] = {
            "velocities_km_per_s": list(layered_velocities),
        }
    if block_override is not None:
        payload["block_anomaly"] = block_override
    if fault_override is not None:
        payload["faulted"] = fault_override
    if salt_override is not None:
        payload["salt_dome"] = salt_override
    if dyke_override is not None:
        payload["dyke_intrusion"] = dyke_override
    return payload


def _observation_feature_fields() -> tuple[str, ...]:
    return (
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "receiver_x_km",
        "receiver_y_km",
        "receiver_z_km",
        "path_length_km",
        "travel_time_s",
    )


def _pseudo_bending_solver_payload(settings: BenchmarkSettings) -> dict[str, object]:
    return {
        "initial_ray_point_count": settings.pseudo_bending_solver.initial_ray_point_count,
        "max_ray_point_count": settings.pseudo_bending_solver.max_ray_point_count,
        "max_iterations_per_level": settings.pseudo_bending_solver.max_iterations_per_level,
        "convergence_tolerance_s": settings.pseudo_bending_solver.convergence_tolerance_s,
        "perturbation_step_km": settings.pseudo_bending_solver.perturbation_step_km,
        "min_perturbation_step_km": settings.pseudo_bending_solver.min_perturbation_step_km,
        "max_point_move_km": settings.pseudo_bending_solver.max_point_move_km,
        "finite_difference_step_km": settings.pseudo_bending_solver.finite_difference_step_km,
        "velocity_interpolation": settings.pseudo_bending_solver.velocity_interpolation,
        "boundary_handling": settings.pseudo_bending_solver.boundary_handling,
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
        raise ValueError("Observation paired samples drifted from the frozen target contract.")
    return current


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Expected a JSON object.")
    return dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("Expected JSON object lines.")
            records.append(dict(payload))
    return records


def _sensitivity_lengths_match_ray_paths(
    sensitivity_records: list[dict[str, Any]],
    ray_records: list[dict[str, Any]],
    tolerance_km: float = 1.0e-6,
) -> bool:
    sensitivity_lengths: dict[str, float] = {}
    for record in sensitivity_records:
        observation_id = str(record.get("observation_id"))
        sensitivity_lengths[observation_id] = sensitivity_lengths.get(observation_id, 0.0) + float(
            record.get("path_length_km", 0.0)
        )

    for record in ray_records:
        observation_id = str(record.get("observation_id"))
        expected = float(record.get("path_length_km", 0.0))
        if abs(sensitivity_lengths.get(observation_id, -math.inf) - expected) > tolerance_km:
            return False
    return set(sensitivity_lengths) == {str(record.get("observation_id")) for record in ray_records}


def _save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _resolve_manifest_path(path_value: str, repo_root: Path, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_relative = repo_root / path
    if repo_relative.exists():
        return repo_relative
    return manifest_path.parent / path


def _relative_to_repo_or_absolute(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()
