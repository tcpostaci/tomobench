"""Generate and audit the 50-unique-target Benchmark v2 pilot.

This module deliberately owns a new pilot contract rather than extending the
historical case-level benchmark.  Geological targets are sampled first and
identified by a canonical target hash.  Each target receives one independent
acquisition realization.  Travel-time labels are generated only by the frozen
ttcrpy FSM configuration; paths and true path lengths are not part of the
realistic observation table.

The pilot is a corpus-design checkpoint.  It does not fit ML models, define a
final split, or modify manuscript text.
"""

from __future__ import annotations

import csv
import ctypes
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tomobench.config.settings import (
    BlockAnomalySettings,
    DykeIntrusionSettings,
    FaultedGridSettings,
    LayeredVelocitySettings,
    SaltDomeSettings,
    BenchmarkSettings,
    VerticalSliceSettings,
    load_settings,
)
from tomobench.datasets.target_identity import TARGET_HASH_SCHEMA, target_vector_sha256
from tomobench.domain.schemas import CartesianVelocityGrid3D, LayeredVelocityModel, dataclass_to_dict
from tomobench.generation.earthquakes import generate_random_earthquakes
from tomobench.generation.stations import generate_random_stations
from tomobench.generation.velocity_grids import (
    build_cartesian_forward_grid,
    build_cartesian_velocity_grid,
    save_velocity_grid,
)
from tomobench.generation.velocity_models import generate_layered_velocity_model
from tomobench.simulation.ttcrpy_forward import (
    TtcrpyGridConfiguration,
    TtcrpyRectilinearForwardSolver,
)
from tomobench.utils.paths import get_repo_root


PILOT_ID = "benchmark_v2_pilot_50_v1"
PRODUCTION_ID = "benchmark_v2_production_250_v1"
FAMILIES = ("layered", "block_anomaly", "faulted", "salt_dome", "dyke_intrusion")
TARGETS_PER_FAMILY = 10
TARGET_SPACING_KM = (2.5, 2.5, 2.5)
FORWARD_SPACING_KM = (1.25, 1.25, 1.25)
STATION_COUNT = 16
EARTHQUAKE_COUNT = 24
OBSERVATION_COUNT = STATION_COUNT * EARTHQUAKE_COUNT
MAX_FORWARD_NODES = 5_000_000
BASE_SEED = 20260829
SIGNAL_EPSILON_S = 1.0e-6
CONTEXT_NOISE_SCALE_S = 0.025


PILOT_DISTRIBUTIONS: dict[str, Any] = {
    "layered": {
        "layer_count": 5,
        "minimum_layer_thickness_km": 4.0,
        "remaining_thickness_km": 10.0,
        "initial_velocity_km_per_s": [3.8, 4.2],
        "velocity_increment_km_per_s": [0.45, 0.75],
    },
    "block_anomaly": {
        "center_x_km": [25.0, 75.0],
        "center_y_km": [25.0, 75.0],
        "center_z_km": [8.0, 22.0],
        "size_x_km": [10.0, 20.0],
        "size_y_km": [10.0, 20.0],
        "size_z_km": [7.0, 14.0],
        "velocity_delta_absolute_km_per_s": [0.35, 0.60],
    },
    "faulted": {
        "fault_x_km": [25.0, 75.0],
        "fault_y_km": [25.0, 75.0],
        "strike_deg": [0.0, 180.0],
        "dip_deg": [65.0, 85.0],
        "velocity_offset_absolute_km_per_s": [0.25, 0.50],
        "dip_direction": ["positive_normal", "negative_normal"],
        "positive_side": ["greater_equal", "less_equal"],
    },
    "salt_dome": {
        "center_x_km": [25.0, 75.0],
        "center_y_km": [25.0, 75.0],
        "center_z_km": [8.0, 14.0],
        "radius_x_km": [7.5, 15.0],
        "radius_y_km": [7.5, 15.0],
        "radius_z_km": [4.5, 7.0],
        "body_velocity_km_per_s": [5.8, 7.6],
    },
    "dyke_intrusion": {
        "center_x_km": [25.0, 75.0],
        "center_y_km": [25.0, 75.0],
        "strike_deg": [0.0, 180.0],
        "length_km": [30.0, 55.0],
        "width_km": [7.5, 15.0],
        "top_depth_km": [4.0, 8.0],
        "bottom_depth_km": [20.0, 28.0],
        "body_velocity_km_per_s": [6.2, 7.8],
    },
}


@dataclass(frozen=True)
class PilotConfig:
    """Frozen, serializable configuration for one pilot generation."""

    pilot_id: str = PILOT_ID
    families: tuple[str, ...] = FAMILIES
    targets_per_family: int = TARGETS_PER_FAMILY
    target_spacing_km: tuple[float, float, float] = TARGET_SPACING_KM
    forward_spacing_km: tuple[float, float, float] = FORWARD_SPACING_KM
    station_count: int = STATION_COUNT
    earthquake_count: int = EARTHQUAKE_COUNT
    observations_per_target: int = OBSERVATION_COUNT
    base_seed: int = BASE_SEED
    max_forward_nodes: int = MAX_FORWARD_NODES
    signal_epsilon_s: float = SIGNAL_EPSILON_S
    context_noise_scale_s: float = CONTEXT_NOISE_SCALE_S
    solver: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the frozen configuration as JSON-compatible data."""
        payload = asdict(self)
        payload["families"] = list(self.families)
        payload["target_spacing_km"] = list(self.target_spacing_km)
        payload["forward_spacing_km"] = list(self.forward_spacing_km)
        payload["solver"] = self.solver or {
            "package": "ttcrpy==1.4.2",
            "geometry": "3-D rectilinear grid",
            "algorithm": "Fast-Sweeping Method",
            "method_argument": "FSM",
            "tt_from_rp": False,
            "physical_representation": "node-centered velocity",
            "source_receiver_endpoint_policy": "coordinates passed without silent snapping",
            "ray_paths": "diagnostic only; not generated for pilot labels",
        }
        payload["target_grid"] = {
            "spacing_km": list(self.target_spacing_km),
            "shape": [41, 41, 13],
            "target_vector_length": 41 * 41 * 13,
            "representation": "node-centered absolute P-wave velocity",
            "flattening_order": "x_fastest_index_ix_plus_nx_times_iy_plus_ny_times_iz",
        }
        payload["forward_grid"] = {
            "spacing_km": list(self.forward_spacing_km),
            "expected_shape": [81, 81, 25],
            "expected_node_count": 81 * 81 * 25,
            "sampling": "analytic geological model evaluated directly at forward nodes",
        }
        payload["acquisition"] = {
            "realizations_per_target": 1,
            "station_distribution": "random surface stations, same distribution for every family",
            "earthquake_distribution": "random hypocenters, same distribution for every family",
            "station_count": self.station_count,
            "earthquake_count": self.earthquake_count,
            "observations_per_target": self.observations_per_target,
            "geometry_seeds_independent_of_geology": True,
        }
        payload["distributions"] = PILOT_DISTRIBUTIONS
        payload["scope"] = {
            "target_count": len(self.families) * self.targets_per_family,
            "ml_fitting": "not performed in pilot",
            "split_definition": "deferred until production corpus; target-atomic by construction",
            "legacy_solver": "historical diagnostic only",
            "privileged_true_model_path_length": "excluded from realistic observation representation",
        }
        return payload

    @property
    def target_count(self) -> int:
        """Return the number of unique target models requested by this run."""
        return len(self.families) * self.targets_per_family


@dataclass(frozen=True)
class ProductionConfig(PilotConfig):
    """Frozen configuration for the 250-target production corpus.

    The production run reuses the pilot-validated distributions and numerical
    contract, but uses a separate deterministic seed namespace so the pilot is
    not silently duplicated into the final benchmark.
    """

    pilot_id: str = PRODUCTION_ID
    targets_per_family: int = 50
    base_seed: int = BASE_SEED + 10_000_000


@dataclass(frozen=True)
class SampledFamilyParameters:
    """Typed settings and normalized descriptors for one sampled target."""

    velocity_settings: Any
    physical_parameters: dict[str, Any]
    parameter_vector: tuple[float, ...]
    parameter_names: tuple[str, ...]


@dataclass
class PilotTargetDraft:
    """In-memory target record retained while artifacts are generated."""

    target_id: str
    target_index: int
    family: str
    settings: BenchmarkSettings
    model: LayeredVelocityModel
    target_grid: CartesianVelocityGrid3D
    target_hash: str
    physical_parameters: dict[str, Any]
    parameter_vector: tuple[float, ...]
    parameter_names: tuple[str, ...]
    geology_seed: int
    station_seed: int
    earthquake_seed: int
    acquisition_id: str
    target_grid_build_s: float
    model_build_s: float
    target_artifact_path: str = ""
    target_grid_artifact_path: str = ""
    model_artifact_path: str = ""
    forward_artifact_path: str = ""
    label_finite_count: int | None = None
    background_finite_count: int | None = None


def canonical_target_hash(values: Sequence[float]) -> str:
    """Return the project-wide canonical SHA-256 target identity."""
    return target_vector_sha256(values)


def cell_centered_values_from_node_vector(
    values: Sequence[float],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Average eight node values into x-fastest cell-centered values."""
    nx, ny, nz = shape
    flat = np.asarray(values, dtype=np.float64)
    if flat.size != nx * ny * nz:
        raise ValueError("Node vector length does not match the supplied shape.")
    node = flat.reshape((nz, ny, nx)).transpose(2, 1, 0)
    cells = (
        node[:-1, :-1, :-1]
        + node[1:, :-1, :-1]
        + node[:-1, 1:, :-1]
        + node[1:, 1:, :-1]
        + node[:-1, :-1, 1:]
        + node[1:, :-1, 1:]
        + node[:-1, 1:, 1:]
        + node[1:, 1:, 1:]
    ) / 8.0
    return cells.transpose(2, 1, 0).reshape(-1)


def nearest_neighbor_diversity_rows(targets: Sequence[PilotTargetDraft]) -> list[dict[str, Any]]:
    """Calculate descriptive nearest-neighbor target diversity within family."""
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_targets = [target for target in targets if target.family == family]
        if not family_targets:
            continue
        target_vectors = [
            np.asarray(target.target_grid.p_velocity_km_per_s, dtype=np.float64)
            for target in family_targets
        ]
        if len(family_targets) == 1:
            target = family_targets[0]
            rows.append(
                {
                    "target_id": target.target_id,
                    "family": family,
                    "target_hash": target.target_hash,
                    "target_mean_velocity_km_per_s": float(np.mean(target_vectors[0])),
                    "target_std_velocity_km_per_s": float(np.std(target_vectors[0])),
                    "nearest_node_target_id": None,
                    "nearest_node_rmse_km_per_s": None,
                    "nearest_node_max_abs_difference_km_per_s": None,
                    "nearest_cell_target_id": None,
                    "nearest_cell_rmse_km_per_s": None,
                    "nearest_cell_max_abs_difference_km_per_s": None,
                    "nearest_parameter_target_id": None,
                    "nearest_normalized_parameter_distance": None,
                }
            )
            continue
        cell_vectors = [
            cell_centered_values_from_node_vector(values, _grid_shape(target.target_grid))
            for values, target in zip(target_vectors, family_targets, strict=True)
        ]
        for index, target in enumerate(family_targets):
            node_distances = np.asarray(
                [_rmse(target_vectors[index], other) for other in target_vectors], dtype=np.float64
            )
            cell_distances = np.asarray(
                [_rmse(cell_vectors[index], other) for other in cell_vectors], dtype=np.float64
            )
            node_distances[index] = np.inf
            cell_distances[index] = np.inf
            node_neighbor = int(np.argmin(node_distances))
            cell_neighbor = int(np.argmin(cell_distances))
            node_difference = np.abs(target_vectors[index] - target_vectors[node_neighbor])
            cell_difference = np.abs(cell_vectors[index] - cell_vectors[cell_neighbor])
            parameter_distances = np.asarray(
                [
                    _normalized_parameter_distance(
                        target.parameter_vector,
                        other.parameter_vector,
                    )
                    for other in family_targets
                ],
                dtype=np.float64,
            )
            parameter_distances[index] = np.inf
            parameter_neighbor = int(np.argmin(parameter_distances))
            rows.append(
                {
                    "target_id": target.target_id,
                    "family": family,
                    "target_hash": target.target_hash,
                    "target_mean_velocity_km_per_s": float(np.mean(target_vectors[index])),
                    "target_std_velocity_km_per_s": float(np.std(target_vectors[index])),
                    "nearest_node_target_id": family_targets[node_neighbor].target_id,
                    "nearest_node_rmse_km_per_s": float(node_distances[node_neighbor]),
                    "nearest_node_max_abs_difference_km_per_s": float(np.max(node_difference)),
                    "nearest_cell_target_id": family_targets[cell_neighbor].target_id,
                    "nearest_cell_rmse_km_per_s": float(cell_distances[cell_neighbor]),
                    "nearest_cell_max_abs_difference_km_per_s": float(np.max(cell_difference)),
                    "nearest_parameter_target_id": family_targets[parameter_neighbor].target_id,
                    "nearest_normalized_parameter_distance": float(
                        parameter_distances[parameter_neighbor]
                    ),
                }
            )
    return rows


def _sample_family_parameters(
    base_settings: BenchmarkSettings,
    family: str,
    rng: np.random.Generator,
) -> SampledFamilyParameters:
    """Sample one bounded, physically constrained family configuration."""
    if family == "layered":
        widths = 4.0 + rng.dirichlet(np.full(5, 4.0)) * 10.0
        widths = np.round(widths, 3)
        widths[-1] = round(30.0 - float(np.sum(widths[:-1])), 3)
        boundaries = np.concatenate(([0.0], np.cumsum(widths)))
        boundaries[-1] = 30.0
        initial = float(rng.uniform(3.8, 4.2))
        increments = rng.uniform(0.45, 0.75, size=4)
        velocities = np.round(np.concatenate(([initial], initial + np.cumsum(increments))), 3)
        layered = LayeredVelocitySettings(
            layer_count=5,
            depth_boundaries_km=tuple(float(value) for value in boundaries),
            velocities_km_per_s=tuple(float(value) for value in velocities),
        )
        return SampledFamilyParameters(
            velocity_settings=replace(
                base_settings.velocity_model_generation,
                layered=layered,
            ),
            physical_parameters={
                "layered": {
                    "depth_boundaries_km": [float(value) for value in boundaries],
                    "layer_thicknesses_km": [float(value) for value in widths],
                    "velocities_km_per_s": [float(value) for value in velocities],
                }
            },
            parameter_vector=tuple(float(value) for value in np.r_[widths / 30.0, velocities / 8.0]),
            parameter_names=tuple(
                [f"layer_{index + 1}_thickness_fraction" for index in range(5)]
                + [f"layer_{index + 1}_velocity_scaled" for index in range(5)]
            ),
        )

    if family == "block_anomaly":
        center = tuple(float(rng.uniform(low, high)) for low, high in ((25, 75), (25, 75), (8, 22)))
        size = tuple(float(rng.uniform(low, high)) for low, high in ((10, 20), (10, 20), (7, 14)))
        delta = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.35, 0.60))
        anomaly = BlockAnomalySettings(
            center_km=center,
            size_km=size,
            velocity_delta_km_per_s=round(delta, 3),
        )
        return SampledFamilyParameters(
            velocity_settings=replace(
                base_settings.velocity_model_generation,
                block_anomaly=anomaly,
            ),
            physical_parameters={
                "block_anomaly": {
                    "center_km": list(center),
                    "size_km": list(size),
                    "velocity_delta_km_per_s": round(delta, 3),
                }
            },
            parameter_vector=tuple(
                [center[0] / 100.0, center[1] / 100.0, center[2] / 30.0]
                + [size[0] / 30.0, size[1] / 30.0, size[2] / 30.0, (delta + 1.0) / 2.0]
            ),
            parameter_names=(
                "center_x_fraction",
                "center_y_fraction",
                "center_z_fraction",
                "size_x_fraction",
                "size_y_fraction",
                "size_z_fraction",
                "signed_delta_scaled",
            ),
        )

    if family == "faulted":
        fault_x = float(rng.uniform(25.0, 75.0))
        fault_y = float(rng.uniform(25.0, 75.0))
        strike = float(rng.uniform(0.0, 180.0))
        dip = float(rng.uniform(65.0, 85.0))
        dip_direction = str(rng.choice(("positive_normal", "negative_normal")))
        positive_side = str(rng.choice(("greater_equal", "less_equal")))
        offset = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.25, 0.50))
        fault = FaultedGridSettings(
            fault_x_km=round(fault_x, 3),
            fault_y_km=round(fault_y, 3),
            strike_deg=round(strike, 3),
            dip_deg=round(dip, 3),
            dip_direction=dip_direction,
            positive_side=positive_side,
            velocity_offset_km_per_s=round(offset, 3),
        )
        return SampledFamilyParameters(
            velocity_settings=replace(
                base_settings.velocity_model_generation,
                faulted=fault,
            ),
            physical_parameters={
                "faulted": {
                    "fault_x_km": round(fault_x, 3),
                    "fault_y_km": round(fault_y, 3),
                    "strike_deg": round(strike, 3),
                    "dip_deg": round(dip, 3),
                    "dip_direction": dip_direction,
                    "positive_side": positive_side,
                    "velocity_offset_km_per_s": round(offset, 3),
                }
            },
            parameter_vector=(
                fault_x / 100.0,
                fault_y / 100.0,
                strike / 180.0,
                dip / 90.0,
                float(dip_direction == "negative_normal"),
                float(positive_side == "less_equal"),
                (offset + 1.0) / 2.0,
            ),
            parameter_names=(
                "fault_x_fraction",
                "fault_y_fraction",
                "strike_fraction",
                "dip_fraction",
                "negative_dip_direction_indicator",
                "less_equal_side_indicator",
                "signed_offset_scaled",
            ),
        )

    if family == "salt_dome":
        center = tuple(float(rng.uniform(low, high)) for low, high in ((25, 75), (25, 75), (8, 14)))
        radii = tuple(float(rng.uniform(low, high)) for low, high in ((7.5, 15), (7.5, 15), (4.5, 7)))
        center = (round(center[0], 3), round(center[1], 3), round(center[2], 3))
        radii = (round(radii[0], 3), round(radii[1], 3), round(radii[2], 3))
        body_velocity = round(float(rng.uniform(5.8, 7.6)), 3)
        salt = SaltDomeSettings(
            center_km=center,
            radii_km=radii,
            body_velocity_km_per_s=body_velocity,
        )
        return SampledFamilyParameters(
            velocity_settings=replace(
                base_settings.velocity_model_generation,
                salt_dome=salt,
            ),
            physical_parameters={
                "salt_dome": {
                    "center_km": list(center),
                    "radii_km": list(radii),
                    "diameters_km": [2.0 * value for value in radii],
                    "body_velocity_km_per_s": body_velocity,
                }
            },
            parameter_vector=(
                center[0] / 100.0,
                center[1] / 100.0,
                center[2] / 30.0,
                radii[0] / 30.0,
                radii[1] / 30.0,
                radii[2] / 15.0,
                (body_velocity - 3.0) / 5.0,
            ),
            parameter_names=(
                "center_x_fraction",
                "center_y_fraction",
                "center_z_fraction",
                "radius_x_fraction",
                "radius_y_fraction",
                "radius_z_fraction",
                "body_velocity_scaled",
            ),
        )

    if family == "dyke_intrusion":
        center_x = float(rng.uniform(25.0, 75.0))
        center_y = float(rng.uniform(25.0, 75.0))
        strike = float(rng.uniform(0.0, 180.0))
        length = float(rng.uniform(30.0, 55.0))
        width = float(rng.uniform(7.5, 15.0))
        top = float(rng.uniform(4.0, 8.0))
        bottom = float(rng.uniform(max(top + 14.0, 20.0), min(top + 22.0, 28.0)))
        body_velocity = float(rng.uniform(6.2, 7.8))
        center = (round(center_x, 3), round(center_y, 3), round((top + bottom) / 2.0, 3))
        dyke = DykeIntrusionSettings(
            center_km=center,
            strike_deg=round(strike, 3),
            length_km=round(length, 3),
            width_km=round(width, 3),
            top_depth_km=round(top, 3),
            bottom_depth_km=round(bottom, 3),
            body_velocity_km_per_s=round(body_velocity, 3),
        )
        return SampledFamilyParameters(
            velocity_settings=replace(
                base_settings.velocity_model_generation,
                dyke_intrusion=dyke,
            ),
            physical_parameters={
                "dyke_intrusion": {
                    "center_km": list(center),
                    "strike_deg": round(strike, 3),
                    "length_km": round(length, 3),
                    "width_km": round(width, 3),
                    "top_depth_km": round(top, 3),
                    "bottom_depth_km": round(bottom, 3),
                    "body_velocity_km_per_s": round(body_velocity, 3),
                }
            },
            parameter_vector=(
                center_x / 100.0,
                center_y / 100.0,
                strike / 180.0,
                length / 100.0,
                width / 30.0,
                top / 30.0,
                bottom / 30.0,
                (body_velocity - 3.0) / 5.0,
            ),
            parameter_names=(
                "center_x_fraction",
                "center_y_fraction",
                "strike_fraction",
                "length_fraction",
                "width_fraction",
                "top_depth_fraction",
                "bottom_depth_fraction",
                "body_velocity_scaled",
            ),
        )

    raise ValueError(f"Unsupported pilot geological family: {family}")


def _settings_for_target(
    base_settings: BenchmarkSettings,
    target_id: str,
    family_settings: Any,
    geology_seed: int,
    station_seed: int,
    earthquake_seed: int,
    run_id: str = PILOT_ID,
) -> BenchmarkSettings:
    """Create typed settings with target-specific model and acquisition IDs."""
    station_generation = replace(
        base_settings.station_generation,
        default_mode="random",
        station_count=STATION_COUNT,
    )
    earthquake_generation = replace(
        base_settings.earthquake_generation,
        default_mode="random",
        event_count=EARTHQUAKE_COUNT,
    )
    random_seeds = replace(
        base_settings.random_seeds,
        stations=station_seed,
        earthquakes=earthquake_seed,
        velocity_models=geology_seed,
    )
    vertical_slice = VerticalSliceSettings(
        experiment_id=f"{run_id}_{target_id}",
        scenario_id=target_id,
        station_configuration_id=f"{target_id}_acquisition_01_stations",
        earthquake_configuration_id=f"{target_id}_acquisition_01_earthquakes",
        velocity_model_id=target_id,
        simulation_id=f"{target_id}_ttcrpy_fsm",
        dataset_id=f"{target_id}_observations",
        station_count=STATION_COUNT,
        earthquake_count=EARTHQUAKE_COUNT,
        case_id=target_id,
    )
    return replace(
        base_settings,
        station_generation=station_generation,
        earthquake_generation=earthquake_generation,
        velocity_model_generation=family_settings,
        random_seeds=random_seeds,
        vertical_slice=vertical_slice,
    )


def _generate_target_drafts(
    base_settings: BenchmarkSettings,
    config: PilotConfig,
    output_dir: Path,
) -> list[PilotTargetDraft]:
    """Sample, hash, and persist all unique target-grid artifacts."""
    target_dir = output_dir / "targets"
    model_dir = output_dir / "velocity_models"
    target_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    drafts: list[PilotTargetDraft] = []
    seen_hashes: set[str] = set()
    target_index = 0
    for family_index, family in enumerate(config.families):
        for family_target_index in range(config.targets_per_family):
            target_index += 1
            target_id = f"{family}_target_{family_target_index + 1:02d}"
            attempt = 0
            while True:
                geology_seed = (
                    config.base_seed
                    + 1_000_000
                    + target_index * 1_000
                    + family_index * 100
                    + attempt
                )
                station_seed = config.base_seed + 2_000_000 + target_index
                earthquake_seed = config.base_seed + 3_000_000 + target_index
                rng = np.random.default_rng(geology_seed)
                sampled = _sample_family_parameters(base_settings, family, rng)
                target_settings = _settings_for_target(
                    base_settings,
                    target_id,
                    sampled.velocity_settings,
                    geology_seed,
                    station_seed,
                    earthquake_seed,
                    config.pilot_id,
                )
                model_started = perf_counter()
                model = generate_layered_velocity_model(target_settings)
                model_build_s = perf_counter() - model_started
                target_grid_started = perf_counter()
                target_grid = build_cartesian_velocity_grid(
                    target_settings,
                    model,
                    scenario=family,
                    spacing_by_axis_km=config.target_spacing_km,
                    max_grid_nodes=config.max_forward_nodes,
                    grid_role="ml_target_grid",
                )
                target_grid_build_s = perf_counter() - target_grid_started
                target_hash = canonical_target_hash(target_grid.p_velocity_km_per_s)
                if target_hash not in seen_hashes:
                    break
                attempt += 1
                if attempt > 1_000:
                    raise RuntimeError(
                        f"Unable to find a unique sampled target for {target_id} after 1000 attempts."
                    )
            seen_hashes.add(target_hash)
            physical_parameters = {
                "family": family,
                "analytic_model_seed": geology_seed,
                "common_background_layered_model": _layered_parameter_payload(
                    base_settings.velocity_model_generation.layered
                ),
                "sampled_family_parameters": sampled.physical_parameters,
                "sampling_attempt": attempt,
            }
            target_artifact = target_dir / f"{target_id}.npz"
            np.savez_compressed(
                target_artifact,
                x_km=np.asarray(target_grid.x_coordinates_km, dtype=np.float64),
                y_km=np.asarray(target_grid.y_coordinates_km, dtype=np.float64),
                z_km=np.asarray(target_grid.z_coordinates_km, dtype=np.float64),
                p_velocity_km_per_s=np.asarray(target_grid.p_velocity_km_per_s, dtype=np.float64),
            )
            target_grid_artifact = output_dir / "target_grids" / f"{target_id}.json"
            save_velocity_grid(target_grid, target_grid_artifact)
            model_artifact = model_dir / f"{target_id}.json"
            _write_json(
                {
                    "artifact_type": f"{config.pilot_id}_layered_model_definition",
                    "target_id": target_id,
                    "family": family,
                    "target_hash": target_hash,
                    "physical_parameters": physical_parameters,
                    "model": dataclass_to_dict(model),
                    "target_grid_metadata": target_grid.metadata,
                    "target_grid_shape": list(_grid_shape(target_grid)),
                    "target_hash_schema": TARGET_HASH_SCHEMA,
                },
                model_artifact,
            )
            drafts.append(
                PilotTargetDraft(
                    target_id=target_id,
                    target_index=target_index,
                    family=family,
                    settings=target_settings,
                    model=model,
                    target_grid=target_grid,
                    target_hash=target_hash,
                    physical_parameters=physical_parameters,
                    parameter_vector=sampled.parameter_vector,
                    parameter_names=sampled.parameter_names,
                    geology_seed=geology_seed,
                    station_seed=station_seed,
                    earthquake_seed=earthquake_seed,
                    acquisition_id=f"{target_id}_acq_01",
                    target_grid_build_s=target_grid_build_s,
                    model_build_s=model_build_s,
                    target_artifact_path=_relative_path(target_artifact),
                    target_grid_artifact_path=_relative_path(target_grid_artifact),
                    model_artifact_path=_relative_path(model_artifact),
                )
            )
    if len(drafts) != config.target_count:
        raise RuntimeError("The generated target count does not match the frozen pilot configuration.")
    return drafts


def _generate_authoritative_observations(
    base_settings: BenchmarkSettings,
    config: PilotConfig,
    output_dir: Path,
    drafts: list[PilotTargetDraft],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, float, dict[str, Any]]:
    """Generate authoritative FSM labels and return observations/runtime rows."""
    acquisition_dir = output_dir / "acquisitions"
    forward_dir = output_dir / "forward_grids"
    observation_dir = output_dir / "observations"
    for directory in (acquisition_dir, forward_dir, observation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    background_settings = _background_settings(base_settings, config.pilot_id)
    background_model = generate_layered_velocity_model(background_settings)
    shared_grid_started = perf_counter()
    background_grid = build_cartesian_forward_grid(
        background_settings,
        background_model,
        scenario="layered",
        spacing_by_axis_km=config.forward_spacing_km,
        max_grid_nodes=config.max_forward_nodes,
    )
    shared_background_grid_build_s = perf_counter() - shared_grid_started
    solver_config = TtcrpyGridConfiguration(
        method="FSM",
        cell_slowness=False,
        tt_from_rp=False,
        maxit=100,
        n_threads=1,
    )
    shared_solver_started = perf_counter()
    background_solver = TtcrpyRectilinearForwardSolver.from_cartesian_grid(
        background_grid,
        solver_config,
    )
    shared_background_solver_build_s = perf_counter() - shared_solver_started
    background_artifact = forward_dir / "common_layered_background.npz"
    np.savez_compressed(
        background_artifact,
        x_km=np.asarray(background_grid.x_coordinates_km, dtype=np.float64),
        y_km=np.asarray(background_grid.y_coordinates_km, dtype=np.float64),
        z_km=np.asarray(background_grid.z_coordinates_km, dtype=np.float64),
        p_velocity_km_per_s=np.asarray(background_grid.p_velocity_km_per_s, dtype=np.float64),
    )
    _write_json(
        {
            "artifact_type": f"{config.pilot_id}_common_layered_background_forward_grid",
            "grid_metadata": background_grid.metadata,
            "grid_shape": list(_grid_shape(background_grid)),
            "solver_configuration": solver_config.constructor_kwargs(),
            "label_policy": "diagnostic background only; not an ML target",
        },
        forward_dir / "common_layered_background.json",
    )

    all_observation_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    label_generation_started = perf_counter()
    for draft in drafts:
        target_started = perf_counter()
        memory_before = _working_set_mb()
        forward_grid_started = perf_counter()
        forward_grid = build_cartesian_forward_grid(
            draft.settings,
            draft.model,
            scenario=draft.family,
            spacing_by_axis_km=config.forward_spacing_km,
            max_grid_nodes=config.max_forward_nodes,
        )
        forward_grid_build_s = perf_counter() - forward_grid_started
        forward_artifact = forward_dir / f"{draft.target_id}.npz"
        np.savez_compressed(
            forward_artifact,
            x_km=np.asarray(forward_grid.x_coordinates_km, dtype=np.float64),
            y_km=np.asarray(forward_grid.y_coordinates_km, dtype=np.float64),
            z_km=np.asarray(forward_grid.z_coordinates_km, dtype=np.float64),
            p_velocity_km_per_s=np.asarray(forward_grid.p_velocity_km_per_s, dtype=np.float64),
        )
        draft.forward_artifact_path = _relative_path(forward_artifact)
        memory_after_grid = _working_set_mb()
        solver_started = perf_counter()
        target_solver = TtcrpyRectilinearForwardSolver.from_cartesian_grid(
            forward_grid,
            solver_config,
        )
        solver_construction_s = perf_counter() - solver_started
        memory_after_solver = _working_set_mb()

        stations = generate_random_stations(draft.settings)
        earthquakes = generate_random_earthquakes(draft.settings)
        _write_json(dataclass_to_dict(stations), acquisition_dir / f"{draft.acquisition_id}_stations.json")
        _write_json(
            dataclass_to_dict(earthquakes),
            acquisition_dir / f"{draft.acquisition_id}_earthquakes.json",
        )
        target_points = [station.location for station in stations.stations]
        target_times_by_event, target_runtime_s, target_errors = _batched_source_times(
            target_solver,
            earthquakes.earthquakes,
            target_points,
        )
        background_times_by_event: list[list[float | None]] = []
        background_runtime_s = 0.0
        background_errors: list[str | None] = []
        if draft.family == "layered":
            background_times_by_event = [
                list(event_times) for event_times in target_times_by_event
            ]
            background_errors = [None for _ in earthquakes.earthquakes]
        else:
            background_times_by_event, background_runtime_s, background_errors = (
                _batched_source_times(
                    background_solver,
                    earthquakes.earthquakes,
                    target_points,
                )
            )

        target_rows: list[dict[str, Any]] = []
        finite_label_count = 0
        finite_background_count = 0
        for event_index, earthquake in enumerate(earthquakes.earthquakes):
            source = earthquake.hypocenter
            for station_index, station in enumerate(stations.stations):
                receiver = station.location
                target_time = target_times_by_event[event_index][station_index]
                background_time = background_times_by_event[event_index][station_index]
                label_status = "finite" if _finite_or_none(target_time) else "nonfinite_error"
                background_status = (
                    "same_target_layered_background"
                    if draft.family == "layered" and _finite_or_none(background_time)
                    else "finite"
                    if _finite_or_none(background_time)
                    else "nonfinite_error"
                )
                if _finite_or_none(target_time):
                    finite_label_count += 1
                if _finite_or_none(background_time):
                    finite_background_count += 1
                if _finite_or_none(target_time) and _finite_or_none(background_time):
                    delta_time = float(target_time) - float(background_time)
                    observation_status = "finite"
                else:
                    delta_time = None
                    observation_status = "invalid_label_or_background"
                euclidean_distance = math.dist(
                    (source.x_km, source.y_km, source.z_km),
                    (receiver.x_km, receiver.y_km, receiver.z_km),
                )
                target_error = target_errors[event_index]
                background_error = background_errors[event_index]
                target_rows.append(
                    {
                        "observation_id": (
                            f"{draft.target_id}_{draft.acquisition_id}_"
                            f"{earthquake.earthquake_id}_{station.station_id}"
                        ),
                        "target_id": draft.target_id,
                        "target_hash": draft.target_hash,
                        "family": draft.family,
                        "acquisition_id": draft.acquisition_id,
                        "earthquake_id": earthquake.earthquake_id,
                        "station_id": station.station_id,
                        "source_x_km": source.x_km,
                        "source_y_km": source.y_km,
                        "source_z_km": source.z_km,
                        "receiver_x_km": receiver.x_km,
                        "receiver_y_km": receiver.y_km,
                        "receiver_z_km": receiver.z_km,
                        "euclidean_distance_km": euclidean_distance,
                        "travel_time_s": target_time,
                        "background_travel_time_s": background_time,
                        "delta_t_geology_s": delta_time,
                        "label_status": label_status,
                        "background_status": background_status,
                        "observation_status": observation_status,
                        "target_solver_error": target_error,
                        "background_solver_error": background_error,
                        "solver_package": "ttcrpy==1.4.2",
                        "solver_geometry": "3-D rectilinear grid",
                        "solver_method": "FSM",
                        "tt_from_rp": False,
                        "physical_representation": "node-centered velocity",
                        "forward_grid_spacing_x_km": config.forward_spacing_km[0],
                        "forward_grid_spacing_y_km": config.forward_spacing_km[1],
                        "forward_grid_spacing_z_km": config.forward_spacing_km[2],
                        "source_receiver_endpoint_policy": "no silent snapping",
                        "true_model_path_length_included": False,
                    }
                )
        observation_file = observation_dir / f"{draft.target_id}_observations.csv"
        _write_csv(observation_file, target_rows)
        all_observation_rows.extend(target_rows)
        draft.label_finite_count = finite_label_count
        draft.background_finite_count = finite_background_count
        memory_after_observations = _working_set_mb()
        observed_memory_values = [
            value
            for value in (memory_before, memory_after_grid, memory_after_solver, memory_after_observations)
            if value is not None
        ]
        runtime_rows.append(
            {
                "target_id": draft.target_id,
                "family": draft.family,
                "target_grid_build_s": draft.target_grid_build_s,
                "model_build_s": draft.model_build_s,
                "forward_grid_build_s": forward_grid_build_s,
                "solver_construction_s": solver_construction_s,
                "target_travel_time_s": target_runtime_s,
                "background_travel_time_s": background_runtime_s,
                "target_solver_error_count": sum(error is not None for error in target_errors),
                "background_solver_error_count": sum(
                    error is not None for error in background_errors
                ),
                "observation_count": len(target_rows),
                "finite_label_count": finite_label_count,
                "finite_background_count": finite_background_count,
                "memory_before_mb": memory_before,
                "memory_after_grid_mb": memory_after_grid,
                "memory_after_solver_mb": memory_after_solver,
                "memory_after_observations_mb": memory_after_observations,
                "peak_observed_working_set_mb": max(observed_memory_values)
                if observed_memory_values
                else None,
                "target_elapsed_s": perf_counter() - target_started,
                "target_output_size_bytes": _directory_size_bytes(output_dir),
            }
        )
    label_generation_wall_s = perf_counter() - label_generation_started
    shared_setup = {
        "background_grid_build_s": shared_background_grid_build_s,
        "background_solver_build_s": shared_background_solver_build_s,
        "background_grid_artifact": _relative_path(background_artifact),
        "solver_configuration": solver_config.constructor_kwargs(),
    }
    return (
        all_observation_rows,
        runtime_rows,
        label_generation_wall_s,
        shared_background_grid_build_s + shared_background_solver_build_s,
        shared_setup,
    )


def run_pilot(
    *,
    output_dir: Path | None = None,
    targets_per_family: int = TARGETS_PER_FAMILY,
    base_seed: int = BASE_SEED,
) -> Path:
    """Generate the pilot and all diagnostic artifacts."""
    if targets_per_family < 1:
        raise ValueError("targets_per_family must be positive.")
    repo_root = get_repo_root()
    resolved_output = _resolve_output_dir(output_dir or Path(
        "outputs/generated/submission_readiness/benchmark_v2_pilot_50_v1"
    ), repo_root)
    if resolved_output.exists() and any(resolved_output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing pilot output directory: {resolved_output}"
        )
    resolved_output.mkdir(parents=True, exist_ok=True)
    config = PilotConfig(targets_per_family=targets_per_family, base_seed=base_seed)
    started_utc = datetime.now(UTC).isoformat()
    run_started = perf_counter()
    _write_json(config.to_dict(), resolved_output / "pilot_config.json")
    base_settings = load_settings()
    drafts = _generate_target_drafts(base_settings, config, resolved_output)
    _write_csv(resolved_output / "target_case_registry.csv", _target_registry_rows(drafts, config))
    observations, runtime_rows, label_wall_s, shared_setup_s, shared_setup = (
        _generate_authoritative_observations(
            base_settings,
            config,
            resolved_output,
            drafts,
        )
    )
    _write_csv(resolved_output / "travel_time_observations.csv", observations)
    _write_csv(resolved_output / "pilot_runtime.csv", runtime_rows)
    _write_csv(resolved_output / "target_case_registry.csv", _target_registry_rows(drafts, config))
    diversity_rows = nearest_neighbor_diversity_rows(drafts)
    _write_csv(resolved_output / "target_diversity_summary.csv", diversity_rows)
    _write_diversity_report(resolved_output / "target_diversity_report.md", drafts, diversity_rows)
    representation_rows = _target_representation_rows(base_settings, drafts, config)
    _write_csv(resolved_output / "target_representation_audit.csv", representation_rows)
    quality_rows = _target_quality_rows(observations, drafts, config)
    _write_csv(resolved_output / "pilot_travel_time_summary.csv", quality_rows)
    _write_quality_report(resolved_output / "pilot_quality_summary.md", observations, quality_rows, config)
    pca_payload = _write_pca_spectrum(resolved_output, drafts)
    runtime_payload = _write_runtime_report(
        resolved_output,
        runtime_rows,
        label_wall_s,
        shared_setup_s,
        shared_setup,
        run_started,
        config,
    )
    gate_ready = _write_pilot_gate(
        resolved_output,
        drafts,
        observations,
        runtime_rows,
        representation_rows,
        quality_rows,
        config,
        runtime_payload,
        pca_payload,
    )
    if gate_ready:
        _write_production_design(resolved_output, drafts, config)
    metadata = {
        "artifact_type": "benchmark_v2_pilot_run_metadata",
        "pilot_id": config.pilot_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": started_utc,
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
        "gate_ready": gate_ready,
        "scope_exclusions": [
            "No Benchmark v2 production targets were generated.",
            "No ML model was fit or evaluated.",
            "No manuscript file was modified.",
            "No legacy solver, SPM, DSPM, or PyKonal label generation was used.",
        ],
    }
    _write_json(metadata, resolved_output / "pilot_metadata.json")
    return resolved_output


def _target_registry_rows(
    drafts: Sequence[PilotTargetDraft],
    config: PilotConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for draft in drafts:
        rows.append(
            {
                "target_id": draft.target_id,
                "target_index": draft.target_index,
                "family": draft.family,
                "target_hash": draft.target_hash,
                "target_hash_schema": TARGET_HASH_SCHEMA,
                "target_shape": "x".join(str(value) for value in _grid_shape(draft.target_grid)),
                "target_vector_length": len(draft.target_grid.p_velocity_km_per_s),
                "target_artifact_path": draft.target_artifact_path,
                "target_grid_artifact_path": draft.target_grid_artifact_path,
                "model_artifact_path": draft.model_artifact_path,
                "forward_artifact_path": draft.forward_artifact_path,
                "geology_seed": draft.geology_seed,
                "station_seed": draft.station_seed,
                "earthquake_seed": draft.earthquake_seed,
                "acquisition_id": draft.acquisition_id,
                "station_configuration_id": draft.settings.vertical_slice.station_configuration_id,
                "earthquake_configuration_id": draft.settings.vertical_slice.earthquake_configuration_id,
                "station_count": config.station_count,
                "earthquake_count": config.earthquake_count,
                "observation_count": config.observations_per_target,
                "target_spacing_km": json.dumps(list(config.target_spacing_km)),
                "forward_spacing_km": json.dumps(list(config.forward_spacing_km)),
                "parameter_vector_json": json.dumps(list(draft.parameter_vector)),
                "parameter_names_json": json.dumps(list(draft.parameter_names)),
                "physical_parameters_json": json.dumps(
                    draft.physical_parameters,
                    sort_keys=True,
                ),
                "finite_label_count": draft.label_finite_count,
                "finite_background_count": draft.background_finite_count,
            }
        )
    return rows


def _target_representation_rows(
    base_settings: BenchmarkSettings,
    drafts: Sequence[PilotTargetDraft],
    config: PilotConfig,
) -> list[dict[str, Any]]:
    base_slice = replace(
        base_settings.vertical_slice,
        velocity_model_id="benchmark_v2_pilot_common_background",
    )
    background_settings = replace(base_settings, vertical_slice=base_slice)
    background_model = generate_layered_velocity_model(background_settings)
    background_grid = build_cartesian_velocity_grid(
        background_settings,
        background_model,
        scenario="layered",
        spacing_by_axis_km=config.target_spacing_km,
        max_grid_nodes=config.max_forward_nodes,
        grid_role="ml_target_background_reference",
    )
    background_values = np.asarray(background_grid.p_velocity_km_per_s, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for draft in drafts:
        values = np.asarray(draft.target_grid.p_velocity_km_per_s, dtype=np.float64)
        difference = np.abs(values - background_values)
        changed = difference > 1.0e-12
        metadata = draft.target_grid.metadata.get(draft.family, {})
        family_parameters = draft.physical_parameters["sampled_family_parameters"].get(
            draft.family,
            {},
        )
        nominal_x, nominal_y, nominal_z = _nominal_structure_dimensions(draft.family, family_parameters)
        intended_nodes = metadata.get("affected_node_count")
        if intended_nodes is None:
            intended_nodes = int(np.count_nonzero(changed))
        changed_values = difference[changed]
        contrast_mean = float(np.mean(changed_values)) if changed_values.size else 0.0
        contrast_max = float(np.max(changed_values)) if changed_values.size else 0.0
        min_lateral_intervals = (
            min(nominal_x, nominal_y) / config.target_spacing_km[0]
            if nominal_x is not None and nominal_y is not None
            else None
        )
        vertical_intervals = (
            nominal_z / config.target_spacing_km[2] if nominal_z is not None else None
        )
        resolution_status = "represented" if np.count_nonzero(changed) > 0 else "structure_disappeared"
        rows.append(
            {
                "target_id": draft.target_id,
                "family": draft.family,
                "target_hash": draft.target_hash,
                "target_shape": "x".join(str(value) for value in _grid_shape(draft.target_grid)),
                "target_node_count": int(values.size),
                "changed_node_count_vs_common_layered_background": int(np.count_nonzero(changed)),
                "changed_node_fraction_vs_common_layered_background": float(np.mean(changed)),
                "analytic_intended_structure_node_count": int(intended_nodes),
                "nominal_x_dimension_km": nominal_x,
                "nominal_y_dimension_km": nominal_y,
                "nominal_z_dimension_km": nominal_z,
                "minimum_lateral_target_intervals": min_lateral_intervals,
                "vertical_target_intervals": vertical_intervals,
                "mean_abs_velocity_contrast_km_per_s": contrast_mean,
                "max_abs_velocity_contrast_km_per_s": contrast_max,
                "resolution_status": resolution_status,
                "dyke_minimum_width_requirement_met": (
                    draft.family != "dyke_intrusion"
                    or nominal_x is None
                    or nominal_x / config.target_spacing_km[0] >= 3.0
                ),
                "target_grid_sampling": "analytic model sampled directly at 2.5 km nodes",
            }
        )
    return rows


def _target_quality_rows(
    observations: Sequence[Mapping[str, Any]],
    drafts: Sequence[PilotTargetDraft],
    config: PilotConfig,
) -> list[dict[str, Any]]:
    by_target: dict[str, list[Mapping[str, Any]]] = {draft.target_id: [] for draft in drafts}
    for row in observations:
        by_target[str(row["target_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for draft in drafts:
        target_rows = by_target[draft.target_id]
        finite_times = np.asarray(
            [float(row["travel_time_s"]) for row in target_rows if _finite_or_none(row["travel_time_s"])],
            dtype=np.float64,
        )
        finite_distances = np.asarray(
            [
                float(row["euclidean_distance_km"])
                for row in target_rows
                if _finite_or_none(row["travel_time_s"])
            ],
            dtype=np.float64,
        )
        finite_deltas = np.asarray(
            [
                float(row["delta_t_geology_s"])
                for row in target_rows
                if _finite_or_none(row["delta_t_geology_s"])
            ],
            dtype=np.float64,
        )
        abs_deltas = np.abs(finite_deltas)
        all_labels_finite = len(finite_times) == config.observations_per_target
        all_background_finite = sum(
            _finite_or_none(row["background_travel_time_s"]) for row in target_rows
        ) == config.observations_per_target
        if draft.family == "layered":
            sensitivity_class = "not_applicable_layered_family"
        elif not len(abs_deltas) or float(np.max(abs_deltas, initial=0.0)) < config.signal_epsilon_s:
            sensitivity_class = "essentially_unobserved"
        elif float(np.max(abs_deltas)) < config.context_noise_scale_s:
            sensitivity_class = "weakly_observed"
        else:
            sensitivity_class = "informative"
        correlation = _pearson_or_none(finite_distances, finite_times)
        rows.append(
            {
                "target_id": draft.target_id,
                "family": draft.family,
                "target_hash": draft.target_hash,
                "observation_count": len(target_rows),
                "finite_label_count": int(len(finite_times)),
                "finite_background_count": int(
                    sum(_finite_or_none(row["background_travel_time_s"]) for row in target_rows)
                ),
                "all_labels_finite": all_labels_finite,
                "all_background_labels_finite": all_background_finite,
                "travel_time_min_s": float(np.min(finite_times)) if finite_times.size else None,
                "travel_time_max_s": float(np.max(finite_times)) if finite_times.size else None,
                "travel_time_mean_s": float(np.mean(finite_times)) if finite_times.size else None,
                "travel_time_std_s": float(np.std(finite_times)) if finite_times.size else None,
                "mean_euclidean_distance_km": (
                    float(np.mean(finite_distances)) if finite_distances.size else None
                ),
                "distance_travel_time_pearson_r": correlation,
                "delta_t_geology_min_s": float(np.min(finite_deltas)) if finite_deltas.size else None,
                "delta_t_geology_max_s": float(np.max(finite_deltas)) if finite_deltas.size else None,
                "delta_t_geology_mean_s": float(np.mean(finite_deltas)) if finite_deltas.size else None,
                "delta_t_geology_std_s": float(np.std(finite_deltas)) if finite_deltas.size else None,
                "delta_t_geology_abs_max_s": float(np.max(abs_deltas)) if abs_deltas.size else None,
                "fraction_abs_signal_at_or_above_context_noise": (
                    float(np.mean(abs_deltas >= config.context_noise_scale_s))
                    if abs_deltas.size
                    else None
                ),
                "sensitivity_class": sensitivity_class,
                "context_noise_scale_s": config.context_noise_scale_s,
            }
        )
    return rows


def _write_pca_spectrum(output_dir: Path, drafts: Sequence[PilotTargetDraft]) -> dict[str, Any]:
    """Write a descriptive PCA spectrum without fitting an inversion model."""
    matrix = np.asarray(
        [draft.target_grid.p_velocity_km_per_s for draft in drafts],
        dtype=np.float64,
    )
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    tolerance = (
        np.finfo(np.float64).eps
        * max(centered.shape)
        * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    eigenvalues = singular_values[:rank] ** 2
    total = float(np.sum(eigenvalues))
    cumulative = np.cumsum(eigenvalues / total) if total > 0.0 else np.zeros(rank)
    rows = []
    for index, (singular, eigenvalue, cumulative_value) in enumerate(
        zip(singular_values[:rank], eigenvalues, cumulative, strict=True),
        start=1,
    ):
        rows.append(
            {
                "component": index,
                "singular_value": float(singular),
                "eigenvalue": float(eigenvalue),
                "explained_variance_ratio": float(eigenvalue / total) if total else 0.0,
                "cumulative_explained_variance_ratio": float(cumulative_value),
            }
        )
    _write_csv(output_dir / "pca_spectrum.csv", rows)
    thresholds = {
        str(threshold): _first_component_at_least(cumulative, threshold)
        for threshold in (0.90, 0.95, 0.99)
    }
    report_lines = [
        "# Benchmark v2 pilot PCA spectrum",
        "",
        "This is a descriptive target-space audit only. No train/validation/test split and no ML model were fitted.",
        "",
        f"- Unique target vectors: `{len(drafts)}`",
        f"- Target vector length: `{matrix.shape[1]}`",
        f"- Centered target-matrix numerical rank: `{rank}`",
        "- Historical comparison: four components explained approximately `93.85%` with only nine historical training targets.",
        "",
        "## Components reaching cumulative variance thresholds",
        "",
        "| Threshold | First component count |",
        "|---:|---:|",
    ]
    for threshold, component in thresholds.items():
        report_lines.append(f"| {float(threshold):.0%} | {component if component is not None else 'not reached'} |")
    report_lines.extend(
        [
            "",
            "The spectrum is used to check whether the parameterized catalogue has collapsed into a very small subspace. It is not a component-selection result for the production benchmark.",
            "",
        ]
    )
    (output_dir / "pca_spectrum_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "target_count": len(drafts),
        "target_vector_length": int(matrix.shape[1]),
        "rank": rank,
        "threshold_component_counts": thresholds,
        "total_centered_sum_of_squares": total,
    }


def _write_runtime_report(
    output_dir: Path,
    runtime_rows: Sequence[Mapping[str, Any]],
    label_wall_s: float,
    shared_setup_s: float,
    shared_setup: Mapping[str, Any],
    run_started: float,
    config: PilotConfig,
) -> dict[str, Any]:
    """Write measured runtime/storage results and linear production projections."""
    target_elapsed = np.asarray(
        [float(row["target_elapsed_s"]) for row in runtime_rows],
        dtype=np.float64,
    )
    per_target_mean = float(np.mean(target_elapsed)) if target_elapsed.size else 0.0
    per_target_median = float(np.median(target_elapsed)) if target_elapsed.size else 0.0
    projected_mean_250_h = (shared_setup_s + per_target_mean * 250.0) / 3600.0
    projected_median_250_h = (shared_setup_s + per_target_median * 250.0) / 3600.0
    projected_min_250_h = (
        (shared_setup_s + float(np.min(target_elapsed)) * 250.0) / 3600.0
        if target_elapsed.size
        else None
    )
    projected_max_250_h = (
        (shared_setup_s + float(np.max(target_elapsed)) * 250.0) / 3600.0
        if target_elapsed.size
        else None
    )
    output_size = _directory_size_bytes(output_dir)
    lines = [
        "# Benchmark v2 pilot runtime and storage report",
        "",
        "Runtime is measured from the actual pilot generation process. The working-set values are observed process snapshots, not operating-system peak-memory guarantees.",
        "",
        f"- Pilot targets: `{len(runtime_rows)}`",
        f"- Labels per target: `{config.observations_per_target}`",
        f"- Label-generation wall time: `{label_wall_s:.3f}` s",
        f"- End-to-end elapsed time at report creation: `{perf_counter() - run_started:.3f}` s",
        f"- Shared background setup: `{shared_setup_s:.3f}` s",
        f"- Current pilot output size: `{output_size}` bytes",
        f"- Mean per-target measured elapsed time: `{per_target_mean:.3f}` s",
        f"- Median per-target measured elapsed time: `{per_target_median:.3f}` s",
        "",
        "## Production projection",
        "",
        "The projection assumes one acquisition realization per target and linear scaling from the measured pilot target times. It excludes later production-only orchestration overhead.",
        "",
        f"- 250 targets at the measured mean: `{projected_mean_250_h:.3f}` h",
        f"- 250 targets at the measured median: `{projected_median_250_h:.3f}` h",
        f"- 250-target range using observed target times: `{projected_min_250_h:.3f}`–`{projected_max_250_h:.3f}` h",
        "",
        "## Frozen setup",
        "",
        f"- Background grid build: `{shared_setup.get('background_grid_build_s')}` s",
        f"- Background solver construction: `{shared_setup.get('background_solver_build_s')}` s",
        "- Label solver: `ttcrpy==1.4.2 / 3-D rectilinear / FSM / tt_from_rp=False`",
        "- Label representation: `node-centered velocity sampled directly at 1.25 km forward nodes`",
        "",
    ]
    (output_dir / "pilot_runtime_storage_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "pilot_target_count": len(runtime_rows),
        "label_generation_wall_s": label_wall_s,
        "shared_setup_s": shared_setup_s,
        "per_target_mean_s": per_target_mean,
        "per_target_median_s": per_target_median,
        "projected_250_targets_mean_h": projected_mean_250_h,
        "projected_250_targets_median_h": projected_median_250_h,
        "projected_250_targets_min_h": projected_min_250_h,
        "projected_250_targets_max_h": projected_max_250_h,
        "output_size_bytes": output_size,
    }


def _write_pilot_gate(
    output_dir: Path,
    drafts: Sequence[PilotTargetDraft],
    observations: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    representation_rows: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    config: PilotConfig,
    runtime_payload: Mapping[str, Any],
    pca_payload: Mapping[str, Any],
) -> bool:
    """Evaluate corpus-design invariants and write the only pilot gate."""
    hash_counts: dict[str, int] = {}
    for draft in drafts:
        hash_counts[draft.target_hash] = hash_counts.get(draft.target_hash, 0) + 1
    family_counts = {family: sum(draft.family == family for draft in drafts) for family in FAMILIES}
    all_labels_finite = all(bool(row["all_labels_finite"]) for row in quality_rows)
    all_background_finite = all(bool(row["all_background_labels_finite"]) for row in quality_rows)
    ids_unique = len({draft.target_id for draft in drafts}) == len(drafts)
    acquisitions_unique = len({draft.acquisition_id for draft in drafts}) == len(drafts)
    seeds_unique = (
        len({draft.station_seed for draft in drafts}) == len(drafts)
        and len({draft.earthquake_seed for draft in drafts}) == len(drafts)
    )
    target_grid_valid = all(
        _grid_shape(draft.target_grid) == (41, 41, 13)
        and len(draft.target_grid.p_velocity_km_per_s) == 41 * 41 * 13
        for draft in drafts
    )
    no_disappeared_structures = all(
        row["resolution_status"] == "represented"
        for row in representation_rows
        if row["family"] != "layered"
    )
    dyke_resolution_valid = all(
        bool(row["dyke_minimum_width_requirement_met"])
        for row in representation_rows
        if row["family"] == "dyke_intrusion"
    )
    runtime_valid = (
        len(runtime_rows) == len(drafts)
        and all(float(row["target_elapsed_s"]) >= 0.0 for row in runtime_rows)
    )
    exact_target_count = len(drafts) == 50 and config.targets_per_family == 10
    checks = [
        ("requested 50-target pilot completed", exact_target_count),
        ("ten target models exist for every family", all(value == 10 for value in family_counts.values())),
        ("target IDs are unique", ids_unique),
        ("canonical target hashes are unique", len(hash_counts) == len(drafts)),
        ("all authoritative travel-time labels are finite", all_labels_finite),
        ("all layered-background diagnostics are finite", all_background_finite),
        ("all target arrays have the frozen 41x41x13 shape", target_grid_valid),
        ("all non-layered structures survive target sampling", no_disappeared_structures),
        ("all pilot dykes meet the three-interval minimum-width design", dyke_resolution_valid),
        ("one acquisition realization exists per target", acquisitions_unique),
        ("acquisition seeds are target-unique and independent fields", seeds_unique),
        ("all 384-observation target records are present", len(observations) == len(drafts) * OBSERVATION_COUNT),
        ("runtime was measured for every target", runtime_valid),
    ]
    ready = all(result for _, result in checks)
    weak_count = sum(row["sensitivity_class"] == "weakly_observed" for row in quality_rows)
    unobserved_count = sum(row["sensitivity_class"] == "essentially_unobserved" for row in quality_rows)
    lines = [
        "# Benchmark v2 pilot gate",
        "",
        "This is the corpus-design gate for the frozen Benchmark v2 pilot. It is not a new forward-solver scientific gate. The adopted travel-time labels are explicitly documented finite-grid FSM approximations.",
        "",
        "## Frozen configuration",
        "",
        "- Package: `ttcrpy==1.4.2`",
        "- Algorithm: `3-D rectilinear Fast-Sweeping Method (FSM)`",
        "- Label mode: `tt_from_rp=False`",
        "- Physical representation: `node-centered velocity sampled directly from the analytic model`",
        "- Forward grid: `1.25 x 1.25 x 1.25 km`",
        "- ML target grid: `2.5 x 2.5 x 2.5 km`, shape `41 x 41 x 13`",
        "- Realistic observation inputs persisted: source/receiver coordinates, Euclidean distance, travel time",
        "- True-model path length: excluded from realistic observations; no ray paths were required for labels",
        "",
        "## Corpus",
        "",
        f"- Unique target models: `{len(drafts)}`",
        f"- Acquisition cases: `{len(drafts)}`",
        f"- Source-receiver observations: `{len(observations)}`",
        f"- Unique target hashes: `{len(hash_counts)}`",
        f"- Exact duplicate target hashes: `{sum(count > 1 for count in hash_counts.values())}`",
        f"- Family counts: `{family_counts}`",
        f"- Weakly observed non-layered targets at the contextual 0.025 s scale: `{weak_count}`",
        f"- Essentially unobserved non-layered targets: `{unobserved_count}`",
        "",
        "Weak sensitivity is reported as a corpus property and is not, by itself, a design failure.",
        "",
        "## Invariant checks",
        "",
        "| Check | Result |",
        "|---|:---:|",
    ]
    for label, result in checks:
        lines.append(f"| {label} | {'PASS' if result else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Target-space diagnostic",
            "",
            f"- Centered target PCA rank: `{pca_payload.get('rank')}`",
            f"- First components reaching 90/95/99% variance: `{pca_payload.get('threshold_component_counts')}`",
            "- This pilot PCA spectrum is descriptive and does not select a production model.",
            "",
            "## Runtime",
            "",
            f"- Measured label-generation wall time: `{runtime_payload.get('label_generation_wall_s'):.3f}` s",
            f"- Projected 250-target one-acquisition runtime at the pilot mean: `{runtime_payload.get('projected_250_targets_mean_h'):.3f}` h",
            "",
            "## Verdict",
            "",
            "READY FOR PRODUCTION BENCHMARK V2" if ready else "PILOT DESIGN FAILURE",
            "",
        ]
    )
    (output_dir / "pilot_gate.md").write_text("\n".join(lines), encoding="utf-8")
    return ready


def _write_production_design(
    output_dir: Path,
    drafts: Sequence[PilotTargetDraft],
    config: PilotConfig,
) -> None:
    """Prepare, but do not execute, the production target-atomic split design."""
    del drafts
    lines = [
        "# Production Benchmark v2 design (prepared, not generated)",
        "",
        "The 50-target pilot passed its corpus-design gate. This file defines the next production construction but does not generate production targets, observations, or ML results.",
        "",
        "## Corpus target",
        "",
        "- Five families: layered, block anomaly, faulted, salt dome, and dyke intrusion",
        "- 50 unique target models per family",
        "- 250 unique geological targets total",
        "- Initially one acquisition realization per target",
        "- 16 stations, 24 earthquakes, and 384 observations per acquisition",
        "",
        "## Frozen label contract",
        "",
        "- `ttcrpy==1.4.2`, 3-D rectilinear FSM, `tt_from_rp=False`",
        "- Node-centered analytic sampling on the `1.25 x 1.25 x 1.25 km` forward grid",
        "- `2.5 x 2.5 x 2.5 km` node-centered ML target grid",
        "- True-model path length remains diagnostic only",
        "",
        "## Deterministic target-atomic split",
        "",
        "Targets will be sorted by `(family, target_hash)` after generation and assigned within each family. For each family, 35 targets are training targets. The first two family blocks receive 8 validation and 7 test targets; the remaining three receive 7 validation and 8 test targets.",
        "",
        "| Split | Unique targets | Cases with one acquisition |",
        "|---|---:|---:|",
        "| Training | 175 | 175 |",
        "| Validation | 37 | 37 |",
        "| Test | 38 | 38 |",
        "| Total | 250 | 250 |",
        "",
        "The split manifest will contain target IDs, hashes, family, physical parameters, seeds, and split. It will be frozen before any PCA, model, or baseline selection. Any later acquisition replicas will inherit the target split and will not be counted as independent geological units.",
        "",
        f"Pilot contract source: `{config.pilot_id}`.",
        "",
    ]
    (output_dir / "production_benchmark_v2_design.md").write_text("\n".join(lines), encoding="utf-8")


def _write_diversity_report(path: Path, drafts: Sequence[PilotTargetDraft], rows: Sequence[Mapping[str, Any]]) -> None:
    duplicate_hashes = _duplicate_values([draft.target_hash for draft in drafts])
    lines = [
        "# Benchmark v2 pilot target diversity report",
        "",
        "The report is descriptive. Exact target hashes are rejected during generation; near-similar targets are retained when they arise from valid parameter draws and are shown rather than removed by an arbitrary threshold.",
        "",
        f"- Target count: `{len(drafts)}`",
        f"- Unique SHA-256 target hashes: `{len({draft.target_hash for draft in drafts})}`",
        f"- Duplicate hashes: `{duplicate_hashes if duplicate_hashes else 'none'}`",
        "",
        "## Within-family nearest-neighbor summaries",
        "",
        "| Family | Targets | Node RMSE median (km/s) | Cell RMSE median (km/s) | Node RMSE minimum (km/s) | Cell RMSE minimum (km/s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        node = np.asarray([row["nearest_node_rmse_km_per_s"] for row in family_rows], dtype=float)
        cell = np.asarray([row["nearest_cell_rmse_km_per_s"] for row in family_rows], dtype=float)
        lines.append(
            f"| {family} | {len(family_rows)} | {np.median(node):.6g} | {np.median(cell):.6g} | {np.min(node):.6g} | {np.min(cell):.6g} |"
        )
    lines.extend(
        [
            "",
            "The paired CSV also records target means, target standard deviations, nearest-neighbor maximum absolute differences, and normalized parameter distances. These values are used to identify collapsed or unusually similar draws before production scaling; they are not a post-hoc target-selection criterion.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_quality_report(
    path: Path,
    observations: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    config: PilotConfig,
) -> None:
    family_summary: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_rows = [row for row in quality_rows if row["family"] == family]
        signals = np.asarray(
            [
                row["delta_t_geology_abs_max_s"]
                for row in family_rows
                if row["delta_t_geology_abs_max_s"] is not None
            ],
            dtype=float,
        )
        times = np.asarray(
            [
                row["travel_time_mean_s"]
                for row in family_rows
                if row["travel_time_mean_s"] is not None
            ],
            dtype=float,
        )
        family_summary.append(
            {
                "family": family,
                "target_count": len(family_rows),
                "target_mean_time_median_s": float(np.median(times)) if times.size else None,
                "target_max_abs_signal_median_s": float(np.median(signals)) if signals.size else None,
                "target_max_abs_signal_p95_s": float(np.percentile(signals, 95)) if signals.size else None,
                "informative_targets": sum(row["sensitivity_class"] == "informative" for row in family_rows),
                "weak_targets": sum(row["sensitivity_class"] == "weakly_observed" for row in family_rows),
                "essentially_unobserved_targets": sum(
                    row["sensitivity_class"] == "essentially_unobserved" for row in family_rows
                ),
            }
        )
    lines = [
        "# Benchmark v2 pilot travel-time quality summary",
        "",
        "These are corpus-quality diagnostics, not a solver-validation gate. The authoritative labels use the frozen finite-grid FSM approximation documented in the pilot configuration.",
        "",
        f"- Total observations: `{len(observations)}`",
        f"- Expected observations: `{len(quality_rows) * config.observations_per_target}`",
        f"- All labels finite: `{all(bool(row['all_labels_finite']) for row in quality_rows)}`",
        f"- All background diagnostics finite: `{all(bool(row['all_background_labels_finite']) for row in quality_rows)}`",
        "- `delta_t_geology = t_target - t_layered_background`.",
        f"- `essentially_unobserved` means max absolute signal < `{config.signal_epsilon_s}` s.",
        f"- `weakly_observed` means max absolute signal < `{config.context_noise_scale_s}` s after excluding essentially unobserved targets.",
        "- The 0.025 s value is a contextual later-noise scale, not a rejection threshold for the pilot.",
        "",
        "## Family summary",
        "",
        "| Family | Targets | Median mean travel time (s) | Median target max |signal| (s) | P95 target max |signal| (s) | Informative | Weak | Essentially unobserved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in family_summary:
        lines.append(
            f"| {row['family']} | {row['target_count']} | {row['target_mean_time_median_s']} | {row['target_max_abs_signal_median_s']} | {row['target_max_abs_signal_p95_s']} | {row['informative_targets']} | {row['weak_targets']} | {row['essentially_unobserved_targets']} |"
        )
    lines.extend(
        [
            "",
            "Every target remains in the pilot registry, including weakly observed structures. The sensitivity class is intended to expose acquisition/model interactions before production scaling.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _batched_source_times(
    solver: TtcrpyRectilinearForwardSolver,
    earthquakes: Sequence[Any],
    receivers: Sequence[Any],
) -> tuple[list[list[float | None]], float, list[str | None]]:
    times: list[list[float | None]] = []
    errors: list[str | None] = []
    runtime_s = 0.0
    for earthquake in earthquakes:
        try:
            result = solver.raytrace(
                [earthquake.hypocenter],
                receivers,
                aggregate_src=True,
                return_rays=False,
            )
            runtime_s += result.runtime_s
            values = [float(value) for value in result.travel_times_s]
            times.append(values)
            errors.append(None)
        except Exception as exc:  # pragma: no cover - exercised by solver environment failures
            times.append([None for _ in receivers])
            errors.append(f"{type(exc).__name__}: {exc}")
    return times, runtime_s, errors


def _background_settings(base_settings: BenchmarkSettings, run_id: str = PILOT_ID) -> BenchmarkSettings:
    return replace(
        base_settings,
        vertical_slice=replace(
            base_settings.vertical_slice,
            experiment_id=f"{run_id}_common_background",
            scenario_id="common_layered_background",
            velocity_model_id="common_layered_background",
        ),
    )


def _layered_parameter_payload(layered: LayeredVelocitySettings) -> dict[str, Any]:
    return {
        "depth_boundaries_km": list(layered.depth_boundaries_km),
        "layer_thicknesses_km": [
            right - left
            for left, right in zip(
                layered.depth_boundaries_km[:-1],
                layered.depth_boundaries_km[1:],
                strict=True,
            )
        ],
        "velocities_km_per_s": list(layered.velocities_km_per_s),
    }


def _nominal_structure_dimensions(
    family: str,
    parameters: Mapping[str, Any],
) -> tuple[float | None, float | None, float | None]:
    if family == "block_anomaly":
        size = parameters.get("size_km", (None, None, None))
        return float(size[0]), float(size[1]), float(size[2])
    if family == "salt_dome":
        radii = parameters.get("radii_km", (None, None, None))
        return 2.0 * float(radii[0]), 2.0 * float(radii[1]), 2.0 * float(radii[2])
    if family == "dyke_intrusion":
        return float(parameters["width_km"]), float(parameters["length_km"]), float(
            parameters["bottom_depth_km"] - parameters["top_depth_km"]
        )
    if family == "layered":
        thicknesses = parameters.get("layer_thicknesses_km", ())
        minimum = min(float(value) for value in thicknesses) if thicknesses else None
        return None, None, minimum
    return None, None, None


def _grid_shape(grid: CartesianVelocityGrid3D) -> tuple[int, int, int]:
    return (
        len(grid.x_coordinates_km),
        len(grid.y_coordinates_km),
        len(grid.z_coordinates_km),
    )


def _normalized_parameter_distance(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Parameter vectors compared within a family must have equal length.")
    return float(np.linalg.norm(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)))


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean((first - second) ** 2)))


def _first_component_at_least(values: np.ndarray, threshold: float) -> int | None:
    indexes = np.flatnonzero(values >= threshold)
    return int(indexes[0] + 1) if indexes.size else None


def _finite_or_none(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _pearson_or_none(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 2 or second.size != first.size:
        return None
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _directory_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _relative_path(path: Path) -> str:
    repo_root = get_repo_root()
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_output_dir(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _duplicate_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {value: count for value, count in counts.items() if count > 1}


def _working_set_mb() -> float | None:
    if sys.platform != "win32":
        return None

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_memory_info.restype = ctypes.c_int
        succeeded = get_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
    except Exception:
        return None
    if not succeeded:
        return None
    return float(counters.WorkingSetSize) / (1024.0 * 1024.0)


def _package_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for package in ("ttcrpy", "numpy", "PyYAML", "tomobench"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_worktree_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    return result.returncode != 0
