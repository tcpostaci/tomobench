"""Preparation workflow for the first bounded ML target batch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.datasets.targets import (
    build_velocity_grid_target_spec,
    save_velocity_grid_target_spec,
)
from tomobench.generation.velocity_grids import (
    build_cartesian_velocity_grid,
    save_velocity_grid,
)
from tomobench.generation.velocity_models import (
    generate_layered_velocity_model,
    save_velocity_model,
)
from tomobench.utils.paths import get_repo_root


@dataclass(frozen=True)
class MLTargetBatchScenarioArtifact:
    """One prepared scenario artifact entry for later ML supervision."""

    scenario: str
    velocity_grid_path: Path
    target_spec_path: Path


@dataclass(frozen=True)
class MLTargetBatchOutputs:
    """Paths produced by the first ML target-batch workflow."""

    source_velocity_model_path: Path
    manifest_path: Path
    scenario_artifacts: tuple[MLTargetBatchScenarioArtifact, ...]


def prepare_first_ml_target_batch(
    settings: BenchmarkSettings | None = None,
) -> MLTargetBatchOutputs:
    """Generate the first frozen-batch collection of ML target-grid artifacts."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    batch_id = loaded_settings.machine_learning.first_target_batch.batch_id
    generated_batch_dir = (
        _resolve_repo_path(loaded_settings.outputs.generated_base_dir, repo_root)
        / "ml_targets"
        / batch_id
    )
    manifest_path = _resolve_repo_path(
        loaded_settings.machine_learning.first_target_batch.manifest_path,
        repo_root,
    )
    target_specs_dir = manifest_path.parent / "target_specs"

    velocity_model = generate_layered_velocity_model(loaded_settings)
    source_velocity_model_path = save_velocity_model(
        velocity_model,
        generated_batch_dir / "velocity_models" / f"{velocity_model.model_id}.json",
    )

    scenario_artifacts: list[MLTargetBatchScenarioArtifact] = []
    manifest_items: list[dict[str, Any]] = []
    batch_contract: dict[str, Any] | None = None
    for scenario in loaded_settings.machine_learning.first_target_batch.scenarios:
        velocity_grid = build_cartesian_velocity_grid(
            loaded_settings,
            velocity_model,
            scenario=scenario,
        )
        velocity_grid_path = save_velocity_grid(
            velocity_grid,
            generated_batch_dir / "velocity_grids" / f"{velocity_grid.grid_id}.json",
        )
        target_spec = build_velocity_grid_target_spec(velocity_grid, loaded_settings)
        target_spec["target_batch_id"] = batch_id
        target_spec["target_batch_scenarios"] = list(
            loaded_settings.machine_learning.first_target_batch.scenarios
        )
        target_spec_path = save_velocity_grid_target_spec(
            target_spec,
            target_specs_dir / f"{batch_id}_{scenario}.target_spec.json",
        )
        contract = _extract_target_contract(target_spec)
        batch_contract = _merge_batch_contract(batch_contract, contract)
        manifest_items.append(
            {
                "scenario": scenario,
                "grid_id": velocity_grid.grid_id,
                "source_velocity_model_id": velocity_grid.source_velocity_model_id,
                "source_velocity_model_path": _relative_to_repo_or_absolute(
                    source_velocity_model_path,
                    repo_root,
                ),
                "velocity_grid_path": _relative_to_repo_or_absolute(velocity_grid_path, repo_root),
                "target_spec_path": _relative_to_repo_or_absolute(target_spec_path, repo_root),
                "grid_shape": target_spec["grid_shape"],
                "target_vector_length": target_spec["target_vector_length"],
                "coordinate_system": target_spec["coordinate_system"],
                "grid_spacing_km": target_spec["grid_spacing_km"],
                "target_representation": target_spec["target_representation"],
                "value_field": target_spec["value_field"],
                "value_semantics": target_spec["value_semantics"],
                "value_location": target_spec["value_location"],
                "flattening_order": target_spec["flattening_order"],
            }
        )
        scenario_artifacts.append(
            MLTargetBatchScenarioArtifact(
                scenario=scenario,
                velocity_grid_path=velocity_grid_path,
                target_spec_path=target_spec_path,
            )
        )

    if batch_contract is None:
        raise ValueError("The first ML target batch did not produce any scenario artifacts.")

    manifest_payload = {
        "artifact_type": "ml_target_batch_manifest",
        "batch_id": batch_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scenario_batch": list(loaded_settings.machine_learning.first_target_batch.scenarios),
        "artifact_count": len(manifest_items),
        "source_velocity_model": {
            "model_id": velocity_model.model_id,
            "path": _relative_to_repo_or_absolute(source_velocity_model_path, repo_root),
        },
        "target_contract": batch_contract,
        "items": manifest_items,
        "notes": [
            "This manifest groups only the current first ML scenario batch.",
            "The target container remains the frozen node-centered absolute-velocity Cartesian grid.",
            "Later geometries may be added in future slices only if they map into the same frozen target contract.",
        ],
    }
    _save_json(manifest_payload, manifest_path)
    return MLTargetBatchOutputs(
        source_velocity_model_path=source_velocity_model_path,
        manifest_path=manifest_path,
        scenario_artifacts=tuple(scenario_artifacts),
    )


def _extract_target_contract(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_representation": spec["target_representation"],
        "representation": spec["representation"],
        "value_field": spec["value_field"],
        "value_semantics": spec["value_semantics"],
        "value_location": spec["value_location"],
        "flattening_order": spec["flattening_order"],
        "grid_shape": spec["grid_shape"],
        "target_vector_length": spec["target_vector_length"],
        "coordinate_system": spec["coordinate_system"],
        "grid_spacing_km": spec["grid_spacing_km"],
    }


def _merge_batch_contract(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if current is None:
        return dict(candidate)
    if current != candidate:
        raise ValueError("Target-grid contract drift detected within the first ML target batch.")
    return current


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
