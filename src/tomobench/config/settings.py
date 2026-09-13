"""Typed loading and validation for the central benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tomobench.utils.paths import get_config_path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    yaml = None


def load_benchmark_config(path: Path | None = None) -> dict[str, Any]:
    """Load the central YAML configuration file for the project."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load config/benchmark_config.yaml. "
            "Install project dependencies from code/pyproject.toml first."
        )

    config_path = path or get_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass(frozen=True)
class StudyAreaSettings:
    """Validated Cartesian study-area settings in kilometers."""

    coordinate_system: str
    x_range_km: tuple[float, float]
    y_range_km: tuple[float, float]
    z_range_km: tuple[float, float]
    surface_elevation_km: float
    margin_km: float

    @property
    def station_x_bounds_km(self) -> tuple[float, float]:
        """Return lateral station-generation bounds after applying the margin."""
        return _apply_margin(self.x_range_km, self.margin_km)

    @property
    def station_y_bounds_km(self) -> tuple[float, float]:
        """Return lateral station-generation bounds after applying the margin."""
        return _apply_margin(self.y_range_km, self.margin_km)


@dataclass(frozen=True)
class StationGenerationSettings:
    """Settings for synthetic station generation."""

    default_mode: str
    station_count: int
    grid_rows: int
    grid_cols: int
    station_elevation_km: float


@dataclass(frozen=True)
class EarthquakeGenerationSettings:
    """Settings for synthetic earthquake generation."""

    default_mode: str
    event_count: int
    depth_range_km: tuple[float, float]


@dataclass(frozen=True)
class LayeredVelocitySettings:
    """Settings for the initial 1D layered P-wave velocity model."""

    layer_count: int
    depth_boundaries_km: tuple[float, ...]
    velocities_km_per_s: tuple[float, ...]


@dataclass(frozen=True)
class BlockAnomalySettings:
    """Settings for a small laterally heterogeneous block anomaly."""

    center_km: tuple[float, float, float]
    size_km: tuple[float, float, float]
    velocity_delta_km_per_s: float


@dataclass(frozen=True)
class FaultedGridSettings:
    """Settings for a simple planar faulted velocity-grid overprint."""

    fault_x_km: float
    fault_y_km: float
    strike_deg: float
    dip_deg: float
    dip_direction: str
    positive_side: str
    velocity_offset_km_per_s: float


@dataclass(frozen=True)
class SaltDomeSettings:
    """Settings for a bounded ellipsoidal salt-dome style velocity overprint."""

    center_km: tuple[float, float, float]
    radii_km: tuple[float, float, float]
    body_velocity_km_per_s: float


@dataclass(frozen=True)
class DykeIntrusionSettings:
    """Settings for a bounded dyke-like vertical slab velocity overprint."""

    center_km: tuple[float, float, float]
    strike_deg: float
    length_km: float
    width_km: float
    top_depth_km: float
    bottom_depth_km: float
    body_velocity_km_per_s: float


@dataclass(frozen=True)
class VelocityModelGenerationSettings:
    """Settings for synthetic velocity model generation."""

    default_scenario: str
    supported_scenarios: tuple[str, ...]
    velocity_bounds_km_per_s: tuple[float, float]
    layered: LayeredVelocitySettings
    block_anomaly: BlockAnomalySettings
    faulted: FaultedGridSettings
    salt_dome: SaltDomeSettings
    dyke_intrusion: DykeIntrusionSettings


@dataclass(frozen=True)
class SimulationSettings:
    """Settings for travel-time simulation."""

    phase: str
    method: str
    supported_methods: tuple[str, ...]
    allow_curved_rays: bool
    first_arrival_only: bool
    numerical_tolerance: float
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class EikonalSolverSettings:
    """Settings for the graph-based 3D eikonal prototype solver."""

    x_grid_spacing_km: float
    y_grid_spacing_km: float
    z_grid_spacing_km: float
    connectivity: int
    interpolation: str
    max_grid_nodes: int
    benchmark_connectivities: tuple[int, ...]
    benchmark_grid_spacings_km: tuple[float, ...]
    runtime_repeat_count: int

    @property
    def grid_spacing_km(self) -> float:
        """Return the legacy isotropic spacing value used by older reports."""
        if (
            self.x_grid_spacing_km == self.y_grid_spacing_km
            and self.x_grid_spacing_km == self.z_grid_spacing_km
        ):
            return self.x_grid_spacing_km
        return self.x_grid_spacing_km

    @property
    def grid_spacing_by_axis_km(self) -> dict[str, float]:
        """Return explicit per-axis grid spacing in kilometers."""
        return {
            "x": self.x_grid_spacing_km,
            "y": self.y_grid_spacing_km,
            "z": self.z_grid_spacing_km,
        }

    @property
    def grid_spacing_label(self) -> str:
        """Return a compact label for isotropic or anisotropic spacing."""
        if (
            self.x_grid_spacing_km == self.y_grid_spacing_km
            and self.x_grid_spacing_km == self.z_grid_spacing_km
        ):
            return f"{self.x_grid_spacing_km:g} km"
        return (
            f"x={self.x_grid_spacing_km:g} km, "
            f"y={self.y_grid_spacing_km:g} km, "
            f"z={self.z_grid_spacing_km:g} km"
        )


@dataclass(frozen=True)
class PseudoBendingSolverSettings:
    """Settings for the Cartesian-grid 3D pseudo-bending ray tracer."""

    initial_ray_point_count: int
    max_ray_point_count: int
    max_iterations_per_level: int
    convergence_tolerance_s: float
    perturbation_step_km: float
    min_perturbation_step_km: float
    max_point_move_km: float
    finite_difference_step_km: float
    velocity_interpolation: str
    boundary_handling: str


@dataclass(frozen=True)
class TargetGridSettings:
    """Frozen ML target-grid representation settings."""

    representation: str
    value_field: str
    value_semantics: str
    value_location: str
    flattening_order: str
    scenario_batch: tuple[str, ...]


@dataclass(frozen=True)
class MachineLearningTargetBatchSettings:
    """Settings for the first bounded ML target-batch preparation workflow."""

    batch_id: str
    scenarios: tuple[str, ...]
    manifest_path: Path


@dataclass(frozen=True)
class MachineLearningSupervisedPairingSettings:
    """Settings for the first observation-target pairing manifest."""

    pairing_id: str
    scenarios: tuple[str, ...]
    simulation_method: str
    manifest_path: Path


@dataclass(frozen=True)
class MachineLearningExpandedScenarioCaseSettings:
    """One additional paired-sample case within the frozen target-grid contract."""

    case_id: str
    scenario: str
    parameter_variant_id: str | None
    geometry_profile_id: str | None
    station_generator_mode: str | None
    station_seed: int | None
    earthquake_seed: int | None
    layered_velocities_km_per_s: tuple[float, ...] | None
    block_anomaly_center_km: tuple[float, float, float] | None
    block_anomaly_size_km: tuple[float, float, float] | None
    block_anomaly_velocity_delta_km_per_s: float | None
    fault_x_km: float | None
    fault_y_km: float | None
    fault_strike_deg: float | None
    fault_dip_deg: float | None
    fault_dip_direction: str | None
    fault_positive_side: str | None
    fault_velocity_offset_km_per_s: float | None
    salt_dome_center_km: tuple[float, float, float] | None
    salt_dome_radii_km: tuple[float, float, float] | None
    salt_dome_velocity_delta_km_per_s: float | None
    salt_dome_body_velocity_km_per_s: float | None
    dyke_intrusion_center_km: tuple[float, float, float] | None
    dyke_intrusion_strike_deg: float | None
    dyke_intrusion_length_km: float | None
    dyke_intrusion_width_km: float | None
    dyke_intrusion_top_depth_km: float | None
    dyke_intrusion_bottom_depth_km: float | None
    dyke_intrusion_velocity_delta_km_per_s: float | None
    dyke_intrusion_body_velocity_km_per_s: float | None


@dataclass(frozen=True)
class MachineLearningExpandedPairingSettings:
    """Settings for expanding paired samples within the frozen target-grid contract."""

    batch_id: str
    simulation_method: str
    manifest_path: Path
    cases: tuple["MachineLearningExpandedScenarioCaseSettings", ...]


@dataclass(frozen=True)
class MachineLearningObservationPairingSettings:
    """Settings for the larger observation-driven supervised pairing corpus."""

    batch_id: str
    simulation_method: str
    manifest_path: Path
    station_count: int
    earthquake_count: int
    grid_rows: int
    grid_cols: int
    cases: tuple["MachineLearningExpandedScenarioCaseSettings", ...]


@dataclass(frozen=True)
class MachineLearningObservationGeometryProfileSettings:
    """One reusable acquisition-geometry profile for observation-corpus generation."""

    profile_id: str
    station_generator_mode: str
    station_seed: int | None
    earthquake_seed: int | None


@dataclass(frozen=True)
class MachineLearningObservationFamilyTemplateSettings:
    """One scalable family template for observation-corpus case generation."""

    scenario: str
    geometry_profile_ids: tuple[str, ...]
    parameter_variants: tuple["MachineLearningExpandedScenarioCaseSettings", ...]


@dataclass(frozen=True)
class MachineLearningBaselineSettings:
    """Settings for the first bounded interpretable baseline workflow."""

    experiment_id: str
    model_name: str
    pairing_manifest_path: Path
    output_dir: Path


@dataclass(frozen=True)
class MachineLearningFeatureBaselineSettings:
    """Settings for the next simple feature-using baseline workflow."""

    experiment_id: str
    model_name: str
    pairing_manifest_path: Path
    output_dir: Path


@dataclass(frozen=True)
class MachineLearningReducedTargetBaselineSettings:
    """Settings for the first reduced-target linear baseline workflow."""

    experiment_id: str
    model_name: str
    pairing_manifest_path: Path
    output_dir: Path
    ridge_alpha: float


@dataclass(frozen=True)
class MachineLearningObservationBaselineSettings:
    """Settings for the observation-driven PCA plus ridge full-grid baseline."""

    experiment_id: str
    model_name: str
    pairing_manifest_path: Path
    output_dir: Path
    pca_component_count: int
    ridge_alpha: float
    observation_fields: tuple[str, ...]
    split_strategy: str


@dataclass(frozen=True)
class MachineLearningObservationLinearBaselineSettings:
    """Settings for the observation-driven PCA plus linear full-grid baseline."""

    experiment_id: str
    model_name: str
    pairing_manifest_path: Path
    output_dir: Path
    pca_component_count: int
    observation_fields: tuple[str, ...]
    split_strategy: str


@dataclass(frozen=True)
class MachineLearningObservationMLPBaselineSettings:
    """Settings for the shallow observation-driven PCA plus MLP full-grid baseline."""

    experiment_id: str
    model_name: str
    pairing_manifest_path: Path
    output_dir: Path
    pca_component_count: int
    hidden_width: int
    epoch_count: int
    learning_rate: float
    l2_alpha: float
    observation_fields: tuple[str, ...]
    split_strategy: str


@dataclass(frozen=True)
class MachineLearningObservationRandomForestBaselineSettings:
    """Settings for the observation-driven PCA plus random-forest full-grid baseline."""

    experiment_id: str
    model_name: str
    pairing_manifest_path: Path
    output_dir: Path
    pca_component_count: int
    tree_count: int
    max_depth: int
    min_samples_leaf: int
    max_feature_count: int
    observation_fields: tuple[str, ...]
    split_strategy: str


@dataclass(frozen=True)
class MachineLearningObservationTuningSettings:
    """Settings for repeated-split tuning of the PCA-plus-ridge observation baseline."""

    experiment_id: str
    pairing_manifest_path: Path
    output_dir: Path
    pca_component_counts: tuple[int, ...]
    ridge_alphas: tuple[float, ...]
    split_random_seeds: tuple[int, ...]
    observation_fields: tuple[str, ...]
    split_strategy: str


@dataclass(frozen=True)
class MachineLearningObservationComparisonSettings:
    """Settings for the limited observation-driven ML comparison workflow."""

    experiment_id: str
    pairing_manifest_path: Path
    output_dir: Path
    selected_models: tuple[str, ...]
    observation_fields: tuple[str, ...]
    split_strategy: str


@dataclass(frozen=True)
class MachineLearningPaperPseudoBendingMLSuiteSettings:
    """Settings for paper ML evaluation on the pseudo-bending corpus."""

    experiment_id: str
    input_manifest_path: Path
    output_dir: Path
    selected_models: tuple[str, ...]
    split_protocols: tuple[str, ...]
    fixed_split_seed: int
    repeated_split_seeds: tuple[int, ...]
    leave_one_family_out: bool
    travel_time_noise_levels_s: tuple[float, ...]
    noise_random_seed: int
    observation_fields: tuple[str, ...]
    best_model_selection_metric: str


@dataclass(frozen=True)
class MachineLearningPaperPseudoBendingMLStrengthenedSettings:
    """Settings for audit-driven Phase 4.5B ML strengthening."""

    experiment_id: str
    input_manifest_path: Path
    baseline_suite_summary_path: Path
    baseline_audit_summary_path: Path
    output_dir: Path
    candidate_models: tuple[str, ...]
    pca_component_counts: tuple[int, ...]
    ridge_alphas: tuple[float, ...]
    fixed_split_seed: int
    repeated_split_seeds: tuple[int, ...]
    travel_time_noise_levels_s: tuple[float, ...]
    noise_random_seed: int
    observation_fields: tuple[str, ...]
    best_model_selection_metric: str


@dataclass(frozen=True)
class MachineLearningPaperPseudoBendingFaultedGeneralizationSettings:
    """Settings for bounded faulted-family generalization diagnosis."""

    experiment_id: str
    input_manifest_path: Path
    baseline_suite_summary_path: Path
    baseline_audit_summary_path: Path
    strengthened_summary_path: Path
    output_dir: Path
    target_family: str
    model_name: str
    pca_component_count: int
    ridge_alpha: float | None
    fixed_split_seed: int
    repeated_split_seeds: tuple[int, ...]
    observation_fields: tuple[str, ...]
    derived_feature_set: str
    recommendation_policy: str


@dataclass(frozen=True)
class MachineLearningExpandedBaselineEvaluationSettings:
    """Settings for retraining baseline workflows on the expanded pairing manifest."""

    pairing_manifest_path: Path
    output_dir: Path
    mean_target_experiment_id: str
    feature_distance_experiment_id: str
    reduced_target_experiment_id: str
    summary_experiment_id: str


@dataclass(frozen=True)
class MachineLearningSettings:
    """Settings for later ML dataset and training work."""

    target_representation: str
    target_grid: TargetGridSettings
    first_target_batch: "MachineLearningTargetBatchSettings"
    first_supervised_pairing: "MachineLearningSupervisedPairingSettings"
    expanded_pairing: "MachineLearningExpandedPairingSettings"
    observation_pairing: "MachineLearningObservationPairingSettings"
    paper_pseudo_bending_observation_corpus: "MachineLearningObservationPairingSettings"
    first_baseline: "MachineLearningBaselineSettings"
    feature_baseline: "MachineLearningFeatureBaselineSettings"
    reduced_target_baseline: "MachineLearningReducedTargetBaselineSettings"
    observation_baseline: "MachineLearningObservationBaselineSettings"
    observation_linear_baseline: "MachineLearningObservationLinearBaselineSettings"
    observation_mlp_baseline: "MachineLearningObservationMLPBaselineSettings"
    observation_random_forest_baseline: "MachineLearningObservationRandomForestBaselineSettings"
    observation_tuning: "MachineLearningObservationTuningSettings"
    observation_comparison: "MachineLearningObservationComparisonSettings"
    paper_pseudo_bending_ml_suite: "MachineLearningPaperPseudoBendingMLSuiteSettings"
    paper_pseudo_bending_ml_strengthened: "MachineLearningPaperPseudoBendingMLStrengthenedSettings"
    paper_pseudo_bending_faulted_generalization: (
        "MachineLearningPaperPseudoBendingFaultedGeneralizationSettings"
    )
    expanded_baseline_evaluation: "MachineLearningExpandedBaselineEvaluationSettings"
    baseline_models: tuple[str, ...]
    advanced_models: tuple[str, ...]
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    metrics: tuple[str, ...]


SUPPORTED_SIMULATION_METHODS = (
    "straight_line_layered_placeholder",
    "layered_ray_tracing",
    "eikonal_3d_first_arrival_prototype",
    "pseudo_bending_3d",
)


@dataclass(frozen=True)
class DatasetExportSettings:
    """Settings for source-receiver dataset export."""

    base_dir: Path
    schema_version: str
    export_formats: tuple[str, ...]
    include_metadata_columns: bool


@dataclass(frozen=True)
class RandomSeedSettings:
    """Explicit random seeds used by reproducible generators."""

    global_seed: int
    stations: int
    earthquakes: int
    velocity_models: int
    machine_learning: int


@dataclass(frozen=True)
class OutputSettings:
    """Repository-relative output locations."""

    generated_base_dir: Path
    figures_dir: Path
    benchmarks_dir: Path
    experiment_notes_dir: Path


@dataclass(frozen=True)
class VerticalSliceSettings:
    """Stable identifiers and small counts for the first end-to-end slice."""

    experiment_id: str
    scenario_id: str
    station_configuration_id: str
    earthquake_configuration_id: str
    velocity_model_id: str
    simulation_id: str
    dataset_id: str
    station_count: int
    earthquake_count: int
    case_id: str | None = None


@dataclass(frozen=True)
class BenchmarkSettings:
    """Validated typed access to the central benchmark configuration."""

    raw: dict[str, Any]
    study_area: StudyAreaSettings
    station_generation: StationGenerationSettings
    earthquake_generation: EarthquakeGenerationSettings
    velocity_model_generation: VelocityModelGenerationSettings
    simulation: SimulationSettings
    eikonal_solver: EikonalSolverSettings
    pseudo_bending_solver: PseudoBendingSolverSettings
    machine_learning: MachineLearningSettings
    dataset_export: DatasetExportSettings
    random_seeds: RandomSeedSettings
    outputs: OutputSettings
    vertical_slice: VerticalSliceSettings


def load_settings(path: Path | None = None) -> BenchmarkSettings:
    """Load and validate ``config/benchmark_config.yaml`` as typed settings."""
    raw = load_benchmark_config(path)
    study_area = _parse_study_area(_mapping(raw, "study_area"))
    station_generation = _parse_station_generation(_mapping(raw, "station_generation"))
    earthquake_generation = _parse_earthquake_generation(_mapping(raw, "earthquake_generation"))
    velocity_model_generation = _parse_velocity_model_generation(
        _mapping(raw, "velocity_model_generation"),
        study_area,
    )
    simulation = _parse_simulation(_mapping(raw, "simulation"))
    eikonal_solver = _parse_eikonal_solver(_mapping(raw, "eikonal_solver"))
    pseudo_bending_solver = _parse_pseudo_bending_solver(_mapping(raw, "pseudo_bending_solver"))
    machine_learning = _parse_machine_learning(
        _mapping(raw, "machine_learning"),
        velocity_model_generation,
    )
    dataset_export = _parse_dataset_export(_mapping(raw, "dataset_export"))
    random_seeds = _parse_random_seeds(_mapping(raw, "random_seeds"))
    outputs = _parse_outputs(_mapping(raw, "outputs"))
    vertical_slice = _parse_vertical_slice(_mapping(raw, "vertical_slice"))

    _validate_generation_bounds(
        study_area,
        earthquake_generation,
        velocity_model_generation=velocity_model_generation,
    )
    _validate_ml_target_batch(
        machine_learning=machine_learning,
        velocity_model_generation=velocity_model_generation,
    )

    return BenchmarkSettings(
        raw=raw,
        study_area=study_area,
        station_generation=station_generation,
        earthquake_generation=earthquake_generation,
        velocity_model_generation=velocity_model_generation,
        simulation=simulation,
        eikonal_solver=eikonal_solver,
        pseudo_bending_solver=pseudo_bending_solver,
        machine_learning=machine_learning,
        dataset_export=dataset_export,
        random_seeds=random_seeds,
        outputs=outputs,
        vertical_slice=vertical_slice,
    )


def _parse_study_area(data: dict[str, Any]) -> StudyAreaSettings:
    return StudyAreaSettings(
        coordinate_system=str(data["coordinate_system"]),
        x_range_km=_float_pair(data["x_range_km"], "study_area.x_range_km"),
        y_range_km=_float_pair(data["y_range_km"], "study_area.y_range_km"),
        z_range_km=_float_pair(data["z_range_km"], "study_area.z_range_km"),
        surface_elevation_km=float(data["surface_elevation_km"]),
        margin_km=float(data["margin_km"]),
    )


def _parse_station_generation(data: dict[str, Any]) -> StationGenerationSettings:
    return StationGenerationSettings(
        default_mode=str(data["default_mode"]),
        station_count=_positive_int(data["station_count"], "station_generation.station_count"),
        grid_rows=_positive_int(data["grid_rows"], "station_generation.grid_rows"),
        grid_cols=_positive_int(data["grid_cols"], "station_generation.grid_cols"),
        station_elevation_km=float(data["station_elevation_km"]),
    )


def _parse_earthquake_generation(data: dict[str, Any]) -> EarthquakeGenerationSettings:
    return EarthquakeGenerationSettings(
        default_mode=str(data["default_mode"]),
        event_count=_positive_int(data["event_count"], "earthquake_generation.event_count"),
        depth_range_km=_float_pair(data["depth_range_km"], "earthquake_generation.depth_range_km"),
    )


def _parse_velocity_model_generation(
    data: dict[str, Any],
    study_area: StudyAreaSettings,
) -> VelocityModelGenerationSettings:
    layered_data = _mapping(data, "layered")
    layered = LayeredVelocitySettings(
        layer_count=_positive_int(layered_data["layer_count"], "velocity_model.layer_count"),
        depth_boundaries_km=tuple(float(value) for value in layered_data["depth_boundaries_km"]),
        velocities_km_per_s=tuple(float(value) for value in layered_data["velocities_km_per_s"]),
    )
    block_anomaly_data = _mapping(data, "block_anomaly")
    block_anomaly = BlockAnomalySettings(
        center_km=_float_triple(block_anomaly_data["center_km"], "block_anomaly.center_km"),
        size_km=_positive_float_triple(block_anomaly_data["size_km"], "block_anomaly.size_km"),
        velocity_delta_km_per_s=float(block_anomaly_data["velocity_delta_km_per_s"]),
    )
    faulted_data = _mapping(data, "faulted")
    faulted = FaultedGridSettings(
        fault_x_km=float(faulted_data["fault_x_km"]),
        fault_y_km=float(
            faulted_data.get(
                "fault_y_km",
                (study_area.y_range_km[0] + study_area.y_range_km[1]) / 2.0,
            )
        ),
        strike_deg=float(faulted_data.get("strike_deg", 90.0)),
        dip_deg=float(faulted_data.get("dip_deg", 90.0)),
        dip_direction=str(faulted_data.get("dip_direction", "positive_normal")),
        positive_side=str(faulted_data.get("positive_side", "greater_equal")),
        velocity_offset_km_per_s=float(faulted_data["velocity_offset_km_per_s"]),
    )
    salt_dome_data = _mapping(data, "salt_dome")
    salt_dome = SaltDomeSettings(
        center_km=_float_triple(salt_dome_data["center_km"], "salt_dome.center_km"),
        radii_km=_positive_float_triple(salt_dome_data["radii_km"], "salt_dome.radii_km"),
        body_velocity_km_per_s=float(
            salt_dome_data.get(
                "body_velocity_km_per_s",
                salt_dome_data.get("velocity_delta_km_per_s", 0.0),
            )
        ),
    )
    dyke_intrusion_data = _mapping(data, "dyke_intrusion")
    dyke_intrusion = DykeIntrusionSettings(
        center_km=_float_triple(dyke_intrusion_data["center_km"], "dyke_intrusion.center_km"),
        strike_deg=float(dyke_intrusion_data["strike_deg"]),
        length_km=float(dyke_intrusion_data["length_km"]),
        width_km=float(dyke_intrusion_data["width_km"]),
        top_depth_km=float(dyke_intrusion_data["top_depth_km"]),
        bottom_depth_km=float(dyke_intrusion_data["bottom_depth_km"]),
        body_velocity_km_per_s=float(
            dyke_intrusion_data.get(
                "body_velocity_km_per_s",
                dyke_intrusion_data.get("velocity_delta_km_per_s", 0.0),
            )
        ),
    )
    if len(layered.depth_boundaries_km) != layered.layer_count + 1:
        raise ValueError("Layer depth boundaries must contain layer_count + 1 values.")
    if len(layered.velocities_km_per_s) != layered.layer_count:
        raise ValueError("Layer velocities must contain exactly layer_count values.")
    if sorted(layered.depth_boundaries_km) != list(layered.depth_boundaries_km):
        raise ValueError("Layer depth boundaries must be sorted from shallow to deep.")

    velocity_bounds = _float_pair(
        data["velocity_bounds_km_per_s"], "velocity_model_generation.velocity_bounds_km_per_s"
    )
    for velocity in layered.velocities_km_per_s:
        if not velocity_bounds[0] <= velocity <= velocity_bounds[1]:
            raise ValueError("Layer velocity is outside configured velocity bounds.")
    supported_scenarios = tuple(str(item) for item in data.get("supported_scenarios", ()))
    if not supported_scenarios:
        supported_scenarios = ("layered", "block_anomaly")
    if str(data["default_scenario"]) not in supported_scenarios:
        raise ValueError("velocity_model_generation.default_scenario must be supported.")
    for layer_velocity in layered.velocities_km_per_s:
        anomaly_velocity = layer_velocity + block_anomaly.velocity_delta_km_per_s
        if not velocity_bounds[0] <= anomaly_velocity <= velocity_bounds[1]:
            raise ValueError(
                "Block-anomaly velocity would fall outside configured velocity bounds."
            )
    if faulted.positive_side not in ("greater_equal", "less_equal"):
        raise ValueError("faulted.positive_side must be 'greater_equal' or 'less_equal'.")
    if faulted.dip_direction not in ("positive_normal", "negative_normal"):
        raise ValueError("faulted.dip_direction must be 'positive_normal' or 'negative_normal'.")
    if not 0.0 < faulted.dip_deg <= 90.0:
        raise ValueError("faulted.dip_deg must be within (0, 90] degrees.")
    if not study_area.x_range_km[0] <= faulted.fault_x_km <= study_area.x_range_km[1]:
        raise ValueError("faulted.fault_x_km must lie within the study-area x range.")
    if not study_area.y_range_km[0] <= faulted.fault_y_km <= study_area.y_range_km[1]:
        raise ValueError("faulted.fault_y_km must lie within the study-area y range.")
    for layer_velocity in layered.velocities_km_per_s:
        faulted_velocity = layer_velocity + faulted.velocity_offset_km_per_s
        if not velocity_bounds[0] <= faulted_velocity <= velocity_bounds[1]:
            raise ValueError("Faulted-grid velocity would fall outside configured velocity bounds.")
    if not velocity_bounds[0] <= salt_dome.body_velocity_km_per_s <= velocity_bounds[1]:
        raise ValueError("Salt-dome body velocity is outside configured velocity bounds.")
    if not velocity_bounds[0] <= dyke_intrusion.body_velocity_km_per_s <= velocity_bounds[1]:
        raise ValueError("Dyke-intrusion body velocity is outside configured velocity bounds.")
    salt_top = salt_dome.center_km[2] - salt_dome.radii_km[2]
    salt_bottom = salt_dome.center_km[2] + salt_dome.radii_km[2]
    if salt_top < study_area.z_range_km[0] or salt_bottom > study_area.z_range_km[1]:
        raise ValueError("Salt-dome bounds must stay inside the study-area depth range.")
    if dyke_intrusion.length_km <= 0.0:
        raise ValueError("dyke_intrusion.length_km must be positive.")
    if dyke_intrusion.width_km <= 0.0:
        raise ValueError("dyke_intrusion.width_km must be positive.")
    if not 0.0 <= dyke_intrusion.top_depth_km < dyke_intrusion.bottom_depth_km:
        raise ValueError(
            "dyke_intrusion top and bottom depths must be non-negative and strictly increasing."
        )

    return VelocityModelGenerationSettings(
        default_scenario=str(data["default_scenario"]),
        supported_scenarios=supported_scenarios,
        velocity_bounds_km_per_s=velocity_bounds,
        layered=layered,
        block_anomaly=block_anomaly,
        faulted=faulted,
        salt_dome=salt_dome,
        dyke_intrusion=dyke_intrusion,
    )


def _parse_simulation(data: dict[str, Any]) -> SimulationSettings:
    method = str(data["method"])
    supported_methods = tuple(str(item) for item in data.get("supported_methods", ()))
    if not supported_methods:
        supported_methods = SUPPORTED_SIMULATION_METHODS
    unsupported_configured = sorted(set(supported_methods) - set(SUPPORTED_SIMULATION_METHODS))
    if unsupported_configured:
        raise ValueError(
            "simulation.supported_methods contains unsupported values: "
            f"{', '.join(unsupported_configured)}."
        )
    if method not in supported_methods:
        raise ValueError(
            f"Unsupported simulation.method '{method}'. "
            f"Supported methods: {', '.join(supported_methods)}."
        )
    return SimulationSettings(
        phase=str(data["phase"]),
        method=method,
        supported_methods=supported_methods,
        allow_curved_rays=bool(data["allow_curved_rays"]),
        first_arrival_only=bool(data["first_arrival_only"]),
        numerical_tolerance=float(data["numerical_tolerance"]),
        assumptions=tuple(str(item) for item in data.get("assumptions", ())),
    )


def _parse_eikonal_solver(data: dict[str, Any]) -> EikonalSolverSettings:
    default_spacing_km = float(data.get("grid_spacing_km", 0.0))
    x_grid_spacing_km = float(data.get("x_grid_spacing_km", default_spacing_km))
    y_grid_spacing_km = float(data.get("y_grid_spacing_km", default_spacing_km))
    z_grid_spacing_km = float(data.get("z_grid_spacing_km", default_spacing_km))
    for label, spacing_km in (
        ("eikonal_solver.x_grid_spacing_km", x_grid_spacing_km),
        ("eikonal_solver.y_grid_spacing_km", y_grid_spacing_km),
        ("eikonal_solver.z_grid_spacing_km", z_grid_spacing_km),
    ):
        if spacing_km <= 0.0:
            raise ValueError(f"{label} must be positive.")
    connectivity = int(data["connectivity"])
    if connectivity not in (6, 18, 26):
        raise ValueError("eikonal_solver.connectivity must be one of 6, 18, or 26.")
    interpolation = str(data["interpolation"])
    if interpolation != "nearest":
        raise ValueError("Only nearest interpolation is supported by the prototype solver.")
    return EikonalSolverSettings(
        x_grid_spacing_km=x_grid_spacing_km,
        y_grid_spacing_km=y_grid_spacing_km,
        z_grid_spacing_km=z_grid_spacing_km,
        connectivity=connectivity,
        interpolation=interpolation,
        max_grid_nodes=_positive_int(data["max_grid_nodes"], "eikonal_solver.max_grid_nodes"),
        benchmark_connectivities=tuple(
            _connectivity_list(
                data.get("benchmark_connectivities", (connectivity,)),
                "eikonal_solver.benchmark_connectivities",
            )
        ),
        benchmark_grid_spacings_km=tuple(
            _positive_float_list(
                data.get("benchmark_grid_spacings_km", (x_grid_spacing_km,)),
                "eikonal_solver.benchmark_grid_spacings_km",
            )
        ),
        runtime_repeat_count=_positive_int(
            data.get("runtime_repeat_count", 1),
            "eikonal_solver.runtime_repeat_count",
        ),
    )


def _parse_pseudo_bending_solver(data: dict[str, Any]) -> PseudoBendingSolverSettings:
    initial_ray_point_count = _positive_int(
        data["initial_ray_point_count"],
        "pseudo_bending_solver.initial_ray_point_count",
    )
    max_ray_point_count = _positive_int(
        data["max_ray_point_count"],
        "pseudo_bending_solver.max_ray_point_count",
    )
    if initial_ray_point_count < 3:
        raise ValueError("pseudo_bending_solver.initial_ray_point_count must be at least 3.")
    if max_ray_point_count < initial_ray_point_count:
        raise ValueError(
            "pseudo_bending_solver.max_ray_point_count must be greater than or equal to "
            "initial_ray_point_count."
        )
    velocity_interpolation = str(data["velocity_interpolation"])
    if velocity_interpolation != "piecewise_constant_interfaces":
        raise ValueError(
            "Only piecewise_constant_interfaces interpolation is supported by pseudo-bending."
        )
    boundary_handling = str(data["boundary_handling"])
    if boundary_handling != "clamp_to_model_domain":
        raise ValueError(
            "Only clamp_to_model_domain boundary handling is supported by pseudo-bending."
        )
    return PseudoBendingSolverSettings(
        initial_ray_point_count=initial_ray_point_count,
        max_ray_point_count=max_ray_point_count,
        max_iterations_per_level=_positive_int(
            data["max_iterations_per_level"],
            "pseudo_bending_solver.max_iterations_per_level",
        ),
        convergence_tolerance_s=_positive_float(
            data["convergence_tolerance_s"],
            "pseudo_bending_solver.convergence_tolerance_s",
        ),
        perturbation_step_km=_positive_float(
            data["perturbation_step_km"],
            "pseudo_bending_solver.perturbation_step_km",
        ),
        min_perturbation_step_km=_positive_float(
            data["min_perturbation_step_km"],
            "pseudo_bending_solver.min_perturbation_step_km",
        ),
        max_point_move_km=_positive_float(
            data["max_point_move_km"],
            "pseudo_bending_solver.max_point_move_km",
        ),
        finite_difference_step_km=_positive_float(
            data["finite_difference_step_km"],
            "pseudo_bending_solver.finite_difference_step_km",
        ),
        velocity_interpolation=velocity_interpolation,
        boundary_handling=boundary_handling,
    )


def _parse_dataset_export(data: dict[str, Any]) -> DatasetExportSettings:
    return DatasetExportSettings(
        base_dir=Path(str(data["base_dir"])),
        schema_version=str(data["schema_version"]),
        export_formats=tuple(str(item) for item in data["export_formats"]),
        include_metadata_columns=bool(data["include_metadata_columns"]),
    )


def _parse_machine_learning(
    data: dict[str, Any],
    velocity_model_generation: VelocityModelGenerationSettings,
) -> MachineLearningSettings:
    target_grid_data = _mapping(data, "target_grid")
    target_grid = TargetGridSettings(
        representation=str(target_grid_data["representation"]),
        value_field=str(target_grid_data["value_field"]),
        value_semantics=str(target_grid_data["value_semantics"]),
        value_location=str(target_grid_data["value_location"]),
        flattening_order=str(target_grid_data["flattening_order"]),
        scenario_batch=tuple(str(item) for item in target_grid_data["scenario_batch"]),
    )
    if target_grid.representation != "fixed_shape_cartesian_velocity_grid_3d":
        raise ValueError(
            "machine_learning.target_grid.representation must be fixed_shape_cartesian_velocity_grid_3d."
        )
    if target_grid.value_field != "p_velocity_km_per_s":
        raise ValueError("machine_learning.target_grid.value_field must be p_velocity_km_per_s.")
    if target_grid.value_semantics not in ("absolute_velocity",):
        raise ValueError(
            "Only absolute_velocity target semantics are supported in the frozen target spec."
        )
    if target_grid.value_location != "node_centered":
        raise ValueError("machine_learning.target_grid.value_location must be node_centered.")
    if target_grid.flattening_order != "x_fastest_index_ix_plus_nx_times_iy_plus_ny_times_iz":
        raise ValueError(
            "machine_learning.target_grid.flattening_order must match the current grid storage order."
        )
    first_target_batch_data = _mapping(data, "first_target_batch")
    first_target_batch = MachineLearningTargetBatchSettings(
        batch_id=str(first_target_batch_data["batch_id"]),
        scenarios=tuple(str(item) for item in first_target_batch_data["scenarios"]),
        manifest_path=Path(str(first_target_batch_data["manifest_path"])),
    )
    if not first_target_batch.scenarios:
        raise ValueError("machine_learning.first_target_batch.scenarios must not be empty.")
    if len(set(first_target_batch.scenarios)) != len(first_target_batch.scenarios):
        raise ValueError(
            "machine_learning.first_target_batch.scenarios must not contain duplicates."
        )
    if first_target_batch.scenarios != ("layered", "block_anomaly", "faulted"):
        raise ValueError(
            "machine_learning.first_target_batch.scenarios must currently be "
            "('layered', 'block_anomaly', 'faulted')."
        )
    if tuple(target_grid.scenario_batch) != first_target_batch.scenarios:
        raise ValueError(
            "machine_learning.first_target_batch.scenarios must match "
            "machine_learning.target_grid.scenario_batch."
        )
    first_supervised_pairing_data = _mapping(data, "first_supervised_pairing")
    first_supervised_pairing = MachineLearningSupervisedPairingSettings(
        pairing_id=str(first_supervised_pairing_data["pairing_id"]),
        scenarios=tuple(str(item) for item in first_supervised_pairing_data["scenarios"]),
        simulation_method=str(first_supervised_pairing_data["simulation_method"]),
        manifest_path=Path(str(first_supervised_pairing_data["manifest_path"])),
    )
    if first_supervised_pairing.scenarios != first_target_batch.scenarios:
        raise ValueError(
            "machine_learning.first_supervised_pairing.scenarios must match "
            "machine_learning.first_target_batch.scenarios."
        )
    if first_supervised_pairing.simulation_method not in SUPPORTED_SIMULATION_METHODS:
        raise ValueError(
            "machine_learning.first_supervised_pairing.simulation_method must be one of: "
            f"{', '.join(SUPPORTED_SIMULATION_METHODS)}."
        )
    expanded_pairing_data = _mapping(data, "expanded_pairing")
    expanded_pairing = MachineLearningExpandedPairingSettings(
        batch_id=str(expanded_pairing_data["batch_id"]),
        simulation_method=str(expanded_pairing_data["simulation_method"]),
        manifest_path=Path(str(expanded_pairing_data["manifest_path"])),
        cases=_parse_machine_learning_case_settings(
            expanded_pairing_data,
            "machine_learning.expanded_pairing",
        ),
    )
    _validate_machine_learning_case_collection(
        expanded_pairing.cases,
        expanded_pairing.simulation_method,
        "machine_learning.expanded_pairing",
        allowed_scenarios=tuple(target_grid.scenario_batch) + ("salt_dome", "dyke_intrusion"),
        allowed_scenarios_message=(
            "machine_learning.expanded_pairing.cases may only use scenarios from the frozen "
            "target-grid contract plus the approved salt_dome and dyke_intrusion families."
        ),
    )
    observation_pairing_data = _mapping(data, "observation_pairing")
    observation_pairing = MachineLearningObservationPairingSettings(
        batch_id=str(observation_pairing_data["batch_id"]),
        simulation_method=str(observation_pairing_data["simulation_method"]),
        manifest_path=Path(str(observation_pairing_data["manifest_path"])),
        station_count=_positive_int(
            observation_pairing_data["station_count"],
            "machine_learning.observation_pairing.station_count",
        ),
        earthquake_count=_positive_int(
            observation_pairing_data["earthquake_count"],
            "machine_learning.observation_pairing.earthquake_count",
        ),
        grid_rows=_positive_int(
            observation_pairing_data["grid_rows"],
            "machine_learning.observation_pairing.grid_rows",
        ),
        grid_cols=_positive_int(
            observation_pairing_data["grid_cols"],
            "machine_learning.observation_pairing.grid_cols",
        ),
        cases=_parse_observation_pairing_cases(
            observation_pairing_data,
            "machine_learning.observation_pairing",
        ),
    )
    _validate_machine_learning_case_collection(
        observation_pairing.cases,
        observation_pairing.simulation_method,
        "machine_learning.observation_pairing",
        allowed_scenarios=velocity_model_generation.supported_scenarios,
        allowed_scenarios_message=(
            "machine_learning.observation_pairing.cases may only use the existing five "
            "supported geological families."
        ),
    )
    if (
        observation_pairing.grid_rows * observation_pairing.grid_cols
        != observation_pairing.station_count
    ):
        raise ValueError(
            "machine_learning.observation_pairing grid_rows * grid_cols must equal station_count."
        )
    paper_corpus_data = _mapping(data, "paper_pseudo_bending_observation_corpus")
    if paper_corpus_data.get("inherit_templates_from") == "observation_pairing":
        paper_corpus_template_data = dict(observation_pairing_data)
        paper_corpus_template_data.update(paper_corpus_data)
        paper_corpus_data = paper_corpus_template_data
    paper_pseudo_bending_observation_corpus = MachineLearningObservationPairingSettings(
        batch_id=str(paper_corpus_data["batch_id"]),
        simulation_method=str(paper_corpus_data["simulation_method"]),
        manifest_path=Path(str(paper_corpus_data["manifest_path"])),
        station_count=_positive_int(
            paper_corpus_data["station_count"],
            "machine_learning.paper_pseudo_bending_observation_corpus.station_count",
        ),
        earthquake_count=_positive_int(
            paper_corpus_data["earthquake_count"],
            "machine_learning.paper_pseudo_bending_observation_corpus.earthquake_count",
        ),
        grid_rows=_positive_int(
            paper_corpus_data["grid_rows"],
            "machine_learning.paper_pseudo_bending_observation_corpus.grid_rows",
        ),
        grid_cols=_positive_int(
            paper_corpus_data["grid_cols"],
            "machine_learning.paper_pseudo_bending_observation_corpus.grid_cols",
        ),
        cases=_parse_observation_pairing_cases(
            paper_corpus_data,
            "machine_learning.paper_pseudo_bending_observation_corpus",
        ),
    )
    _validate_machine_learning_case_collection(
        paper_pseudo_bending_observation_corpus.cases,
        paper_pseudo_bending_observation_corpus.simulation_method,
        "machine_learning.paper_pseudo_bending_observation_corpus",
        allowed_scenarios=velocity_model_generation.supported_scenarios,
        allowed_scenarios_message=(
            "machine_learning.paper_pseudo_bending_observation_corpus.cases may only use the "
            "existing five supported geological families."
        ),
    )
    if paper_pseudo_bending_observation_corpus.simulation_method != "pseudo_bending_3d":
        raise ValueError(
            "machine_learning.paper_pseudo_bending_observation_corpus.simulation_method must "
            "be pseudo_bending_3d."
        )
    if (
        paper_pseudo_bending_observation_corpus.grid_rows
        * paper_pseudo_bending_observation_corpus.grid_cols
        != paper_pseudo_bending_observation_corpus.station_count
    ):
        raise ValueError(
            "machine_learning.paper_pseudo_bending_observation_corpus grid_rows * grid_cols "
            "must equal station_count."
        )
    if paper_pseudo_bending_observation_corpus.batch_id == observation_pairing.batch_id:
        raise ValueError(
            "machine_learning.paper_pseudo_bending_observation_corpus.batch_id must be distinct "
            "from machine_learning.observation_pairing.batch_id."
        )
    if paper_pseudo_bending_observation_corpus.manifest_path == observation_pairing.manifest_path:
        raise ValueError(
            "machine_learning.paper_pseudo_bending_observation_corpus.manifest_path must be "
            "distinct from machine_learning.observation_pairing.manifest_path."
        )
    for case_collection_label, case_collection in (
        ("machine_learning.expanded_pairing", expanded_pairing.cases),
        ("machine_learning.observation_pairing", observation_pairing.cases),
        (
            "machine_learning.paper_pseudo_bending_observation_corpus",
            paper_pseudo_bending_observation_corpus.cases,
        ),
    ):
        for case in case_collection:
            if (
                case.layered_velocities_km_per_s is not None
                and len(case.layered_velocities_km_per_s)
                != velocity_model_generation.layered.layer_count
            ):
                raise ValueError(
                    f"{case_collection_label}.layered_velocities_km_per_s must provide exactly "
                    f"{velocity_model_generation.layered.layer_count} values."
                )
    first_baseline_data = _mapping(data, "first_baseline")
    first_baseline = MachineLearningBaselineSettings(
        experiment_id=str(first_baseline_data["experiment_id"]),
        model_name=str(first_baseline_data["model_name"]),
        pairing_manifest_path=Path(str(first_baseline_data["pairing_manifest_path"])),
        output_dir=Path(str(first_baseline_data["output_dir"])),
    )
    if first_baseline.model_name != "mean_target_regressor":
        raise ValueError(
            "machine_learning.first_baseline.model_name must currently be 'mean_target_regressor'."
        )
    if first_baseline.pairing_manifest_path != first_supervised_pairing.manifest_path:
        raise ValueError(
            "machine_learning.first_baseline.pairing_manifest_path must match "
            "machine_learning.first_supervised_pairing.manifest_path."
        )
    feature_baseline_data = _mapping(data, "feature_baseline")
    feature_baseline = MachineLearningFeatureBaselineSettings(
        experiment_id=str(feature_baseline_data["experiment_id"]),
        model_name=str(feature_baseline_data["model_name"]),
        pairing_manifest_path=Path(str(feature_baseline_data["pairing_manifest_path"])),
        output_dir=Path(str(feature_baseline_data["output_dir"])),
    )
    if feature_baseline.model_name != "feature_distance_weighted_target_regressor":
        raise ValueError(
            "machine_learning.feature_baseline.model_name must currently be "
            "'feature_distance_weighted_target_regressor'."
        )
    if feature_baseline.pairing_manifest_path != first_supervised_pairing.manifest_path:
        raise ValueError(
            "machine_learning.feature_baseline.pairing_manifest_path must match "
            "machine_learning.first_supervised_pairing.manifest_path."
        )
    reduced_target_baseline_data = _mapping(data, "reduced_target_baseline")
    reduced_target_baseline = MachineLearningReducedTargetBaselineSettings(
        experiment_id=str(reduced_target_baseline_data["experiment_id"]),
        model_name=str(reduced_target_baseline_data["model_name"]),
        pairing_manifest_path=Path(str(reduced_target_baseline_data["pairing_manifest_path"])),
        output_dir=Path(str(reduced_target_baseline_data["output_dir"])),
        ridge_alpha=float(reduced_target_baseline_data["ridge_alpha"]),
    )
    if reduced_target_baseline.model_name != "ridge_reduced_target_regressor":
        raise ValueError(
            "machine_learning.reduced_target_baseline.model_name must currently be "
            "'ridge_reduced_target_regressor'."
        )
    if reduced_target_baseline.pairing_manifest_path != first_supervised_pairing.manifest_path:
        raise ValueError(
            "machine_learning.reduced_target_baseline.pairing_manifest_path must match "
            "machine_learning.first_supervised_pairing.manifest_path."
        )
    if reduced_target_baseline.ridge_alpha <= 0.0:
        raise ValueError("machine_learning.reduced_target_baseline.ridge_alpha must be positive.")
    observation_baseline_data = _mapping(data, "observation_baseline")
    observation_baseline = MachineLearningObservationBaselineSettings(
        experiment_id=str(observation_baseline_data["experiment_id"]),
        model_name=str(observation_baseline_data["model_name"]),
        pairing_manifest_path=Path(str(observation_baseline_data["pairing_manifest_path"])),
        output_dir=Path(str(observation_baseline_data["output_dir"])),
        pca_component_count=_positive_int(
            observation_baseline_data["pca_component_count"],
            "machine_learning.observation_baseline.pca_component_count",
        ),
        ridge_alpha=float(observation_baseline_data["ridge_alpha"]),
        observation_fields=tuple(
            str(item) for item in observation_baseline_data["observation_fields"]
        ),
        split_strategy=str(observation_baseline_data["split_strategy"]),
    )
    if observation_baseline.model_name != "pca_ridge_observation_regressor":
        raise ValueError(
            "machine_learning.observation_baseline.model_name must currently be "
            "'pca_ridge_observation_regressor'."
        )
    if observation_baseline.pairing_manifest_path != observation_pairing.manifest_path:
        raise ValueError(
            "machine_learning.observation_baseline.pairing_manifest_path must match "
            "machine_learning.observation_pairing.manifest_path."
        )
    if observation_baseline.ridge_alpha <= 0.0:
        raise ValueError("machine_learning.observation_baseline.ridge_alpha must be positive.")
    if observation_baseline.split_strategy != "family_stratified":
        raise ValueError(
            "machine_learning.observation_baseline.split_strategy must currently be "
            "'family_stratified'."
        )
    if not observation_baseline.observation_fields:
        raise ValueError(
            "machine_learning.observation_baseline.observation_fields must not be empty."
        )
    observation_linear_baseline_data = _mapping(data, "observation_linear_baseline")
    observation_linear_baseline = MachineLearningObservationLinearBaselineSettings(
        experiment_id=str(observation_linear_baseline_data["experiment_id"]),
        model_name=str(observation_linear_baseline_data["model_name"]),
        pairing_manifest_path=Path(str(observation_linear_baseline_data["pairing_manifest_path"])),
        output_dir=Path(str(observation_linear_baseline_data["output_dir"])),
        pca_component_count=_positive_int(
            observation_linear_baseline_data["pca_component_count"],
            "machine_learning.observation_linear_baseline.pca_component_count",
        ),
        observation_fields=tuple(
            str(item) for item in observation_linear_baseline_data["observation_fields"]
        ),
        split_strategy=str(observation_linear_baseline_data["split_strategy"]),
    )
    if observation_linear_baseline.model_name != "pca_linear_observation_regressor":
        raise ValueError(
            "machine_learning.observation_linear_baseline.model_name must currently be "
            "'pca_linear_observation_regressor'."
        )
    _validate_observation_workflow_reference(
        observation_linear_baseline.pairing_manifest_path,
        observation_pairing.manifest_path,
        "machine_learning.observation_linear_baseline.pairing_manifest_path",
    )
    _validate_observation_split_strategy(
        observation_linear_baseline.split_strategy,
        "machine_learning.observation_linear_baseline.split_strategy",
    )
    _validate_non_empty_observation_fields(
        observation_linear_baseline.observation_fields,
        "machine_learning.observation_linear_baseline.observation_fields",
    )
    observation_mlp_baseline_data = _mapping(data, "observation_mlp_baseline")
    observation_mlp_baseline = MachineLearningObservationMLPBaselineSettings(
        experiment_id=str(observation_mlp_baseline_data["experiment_id"]),
        model_name=str(observation_mlp_baseline_data["model_name"]),
        pairing_manifest_path=Path(str(observation_mlp_baseline_data["pairing_manifest_path"])),
        output_dir=Path(str(observation_mlp_baseline_data["output_dir"])),
        pca_component_count=_positive_int(
            observation_mlp_baseline_data["pca_component_count"],
            "machine_learning.observation_mlp_baseline.pca_component_count",
        ),
        hidden_width=_positive_int(
            observation_mlp_baseline_data["hidden_width"],
            "machine_learning.observation_mlp_baseline.hidden_width",
        ),
        epoch_count=_positive_int(
            observation_mlp_baseline_data["epoch_count"],
            "machine_learning.observation_mlp_baseline.epoch_count",
        ),
        learning_rate=float(observation_mlp_baseline_data["learning_rate"]),
        l2_alpha=float(observation_mlp_baseline_data["l2_alpha"]),
        observation_fields=tuple(
            str(item) for item in observation_mlp_baseline_data["observation_fields"]
        ),
        split_strategy=str(observation_mlp_baseline_data["split_strategy"]),
    )
    if observation_mlp_baseline.model_name != "pca_mlp_observation_regressor":
        raise ValueError(
            "machine_learning.observation_mlp_baseline.model_name must currently be "
            "'pca_mlp_observation_regressor'."
        )
    _validate_observation_workflow_reference(
        observation_mlp_baseline.pairing_manifest_path,
        observation_pairing.manifest_path,
        "machine_learning.observation_mlp_baseline.pairing_manifest_path",
    )
    if observation_mlp_baseline.learning_rate <= 0.0:
        raise ValueError(
            "machine_learning.observation_mlp_baseline.learning_rate must be positive."
        )
    if observation_mlp_baseline.l2_alpha < 0.0:
        raise ValueError("machine_learning.observation_mlp_baseline.l2_alpha must be non-negative.")
    _validate_observation_split_strategy(
        observation_mlp_baseline.split_strategy,
        "machine_learning.observation_mlp_baseline.split_strategy",
    )
    _validate_non_empty_observation_fields(
        observation_mlp_baseline.observation_fields,
        "machine_learning.observation_mlp_baseline.observation_fields",
    )
    observation_random_forest_baseline_data = _mapping(data, "observation_random_forest_baseline")
    observation_random_forest_baseline = MachineLearningObservationRandomForestBaselineSettings(
        experiment_id=str(observation_random_forest_baseline_data["experiment_id"]),
        model_name=str(observation_random_forest_baseline_data["model_name"]),
        pairing_manifest_path=Path(
            str(observation_random_forest_baseline_data["pairing_manifest_path"])
        ),
        output_dir=Path(str(observation_random_forest_baseline_data["output_dir"])),
        pca_component_count=_positive_int(
            observation_random_forest_baseline_data["pca_component_count"],
            "machine_learning.observation_random_forest_baseline.pca_component_count",
        ),
        tree_count=_positive_int(
            observation_random_forest_baseline_data["tree_count"],
            "machine_learning.observation_random_forest_baseline.tree_count",
        ),
        max_depth=_positive_int(
            observation_random_forest_baseline_data["max_depth"],
            "machine_learning.observation_random_forest_baseline.max_depth",
        ),
        min_samples_leaf=_positive_int(
            observation_random_forest_baseline_data["min_samples_leaf"],
            "machine_learning.observation_random_forest_baseline.min_samples_leaf",
        ),
        max_feature_count=_positive_int(
            observation_random_forest_baseline_data["max_feature_count"],
            "machine_learning.observation_random_forest_baseline.max_feature_count",
        ),
        observation_fields=tuple(
            str(item) for item in observation_random_forest_baseline_data["observation_fields"]
        ),
        split_strategy=str(observation_random_forest_baseline_data["split_strategy"]),
    )
    if observation_random_forest_baseline.model_name != "pca_random_forest_observation_regressor":
        raise ValueError(
            "machine_learning.observation_random_forest_baseline.model_name must currently be "
            "'pca_random_forest_observation_regressor'."
        )
    _validate_observation_workflow_reference(
        observation_random_forest_baseline.pairing_manifest_path,
        observation_pairing.manifest_path,
        "machine_learning.observation_random_forest_baseline.pairing_manifest_path",
    )
    _validate_observation_split_strategy(
        observation_random_forest_baseline.split_strategy,
        "machine_learning.observation_random_forest_baseline.split_strategy",
    )
    _validate_non_empty_observation_fields(
        observation_random_forest_baseline.observation_fields,
        "machine_learning.observation_random_forest_baseline.observation_fields",
    )
    observation_tuning_data = _mapping(data, "observation_tuning")
    observation_tuning = MachineLearningObservationTuningSettings(
        experiment_id=str(observation_tuning_data["experiment_id"]),
        pairing_manifest_path=Path(str(observation_tuning_data["pairing_manifest_path"])),
        output_dir=Path(str(observation_tuning_data["output_dir"])),
        pca_component_counts=tuple(
            _positive_int(
                item,
                "machine_learning.observation_tuning.pca_component_counts entry",
            )
            for item in observation_tuning_data["pca_component_counts"]
        ),
        ridge_alphas=tuple(float(item) for item in observation_tuning_data["ridge_alphas"]),
        split_random_seeds=tuple(
            _positive_int(
                item,
                "machine_learning.observation_tuning.split_random_seeds entry",
            )
            for item in observation_tuning_data["split_random_seeds"]
        ),
        observation_fields=tuple(
            str(item) for item in observation_tuning_data["observation_fields"]
        ),
        split_strategy=str(observation_tuning_data["split_strategy"]),
    )
    _validate_observation_workflow_reference(
        observation_tuning.pairing_manifest_path,
        observation_pairing.manifest_path,
        "machine_learning.observation_tuning.pairing_manifest_path",
    )
    if not observation_tuning.pca_component_counts:
        raise ValueError(
            "machine_learning.observation_tuning.pca_component_counts must not be empty."
        )
    if not observation_tuning.ridge_alphas:
        raise ValueError("machine_learning.observation_tuning.ridge_alphas must not be empty.")
    if any(alpha <= 0.0 for alpha in observation_tuning.ridge_alphas):
        raise ValueError(
            "machine_learning.observation_tuning.ridge_alphas entries must be positive."
        )
    if not observation_tuning.split_random_seeds:
        raise ValueError(
            "machine_learning.observation_tuning.split_random_seeds must not be empty."
        )
    _validate_observation_split_strategy(
        observation_tuning.split_strategy,
        "machine_learning.observation_tuning.split_strategy",
    )
    _validate_non_empty_observation_fields(
        observation_tuning.observation_fields,
        "machine_learning.observation_tuning.observation_fields",
    )
    observation_comparison_data = _mapping(data, "observation_comparison")
    observation_comparison = MachineLearningObservationComparisonSettings(
        experiment_id=str(observation_comparison_data["experiment_id"]),
        pairing_manifest_path=Path(str(observation_comparison_data["pairing_manifest_path"])),
        output_dir=Path(str(observation_comparison_data["output_dir"])),
        selected_models=tuple(str(item) for item in observation_comparison_data["selected_models"]),
        observation_fields=tuple(
            str(item) for item in observation_comparison_data["observation_fields"]
        ),
        split_strategy=str(observation_comparison_data["split_strategy"]),
    )
    _validate_observation_workflow_reference(
        observation_comparison.pairing_manifest_path,
        observation_pairing.manifest_path,
        "machine_learning.observation_comparison.pairing_manifest_path",
    )
    _validate_observation_split_strategy(
        observation_comparison.split_strategy,
        "machine_learning.observation_comparison.split_strategy",
    )
    _validate_non_empty_observation_fields(
        observation_comparison.observation_fields,
        "machine_learning.observation_comparison.observation_fields",
    )
    _validate_observation_comparison_models(
        observation_comparison.selected_models,
        (
            observation_linear_baseline.model_name,
            observation_baseline.model_name,
            observation_random_forest_baseline.model_name,
            observation_mlp_baseline.model_name,
        ),
        "machine_learning.observation_comparison.selected_models",
    )
    paper_suite_data = _mapping(data, "paper_pseudo_bending_ml_suite")
    paper_pseudo_bending_ml_suite = MachineLearningPaperPseudoBendingMLSuiteSettings(
        experiment_id=str(paper_suite_data["experiment_id"]),
        input_manifest_path=Path(str(paper_suite_data["input_manifest_path"])),
        output_dir=Path(str(paper_suite_data["output_dir"])),
        selected_models=tuple(str(item) for item in paper_suite_data["selected_models"]),
        split_protocols=tuple(str(item) for item in paper_suite_data["split_protocols"]),
        fixed_split_seed=_positive_int(
            paper_suite_data["fixed_split_seed"],
            "machine_learning.paper_pseudo_bending_ml_suite.fixed_split_seed",
        ),
        repeated_split_seeds=tuple(
            _positive_int(
                item,
                "machine_learning.paper_pseudo_bending_ml_suite.repeated_split_seeds entry",
            )
            for item in paper_suite_data["repeated_split_seeds"]
        ),
        leave_one_family_out=bool(paper_suite_data["leave_one_family_out"]),
        travel_time_noise_levels_s=tuple(
            float(item) for item in paper_suite_data["travel_time_noise_levels_s"]
        ),
        noise_random_seed=_positive_int(
            paper_suite_data["noise_random_seed"],
            "machine_learning.paper_pseudo_bending_ml_suite.noise_random_seed",
        ),
        observation_fields=tuple(str(item) for item in paper_suite_data["observation_fields"]),
        best_model_selection_metric=str(paper_suite_data["best_model_selection_metric"]),
    )
    if (
        paper_pseudo_bending_ml_suite.input_manifest_path
        != paper_pseudo_bending_observation_corpus.manifest_path
    ):
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_suite.input_manifest_path must match "
            "machine_learning.paper_pseudo_bending_observation_corpus.manifest_path."
        )
    if paper_pseudo_bending_ml_suite.input_manifest_path == observation_pairing.manifest_path:
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_suite.input_manifest_path must not use "
            "the historical observation_pairing manifest."
        )
    _validate_observation_comparison_models(
        paper_pseudo_bending_ml_suite.selected_models,
        (
            observation_linear_baseline.model_name,
            observation_baseline.model_name,
            observation_random_forest_baseline.model_name,
            observation_mlp_baseline.model_name,
        ),
        "machine_learning.paper_pseudo_bending_ml_suite.selected_models",
    )
    _validate_required_split_protocols(
        paper_pseudo_bending_ml_suite.split_protocols,
        "machine_learning.paper_pseudo_bending_ml_suite.split_protocols",
    )
    if not paper_pseudo_bending_ml_suite.repeated_split_seeds:
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_suite.repeated_split_seeds must not be empty."
        )
    if any(level < 0.0 for level in paper_pseudo_bending_ml_suite.travel_time_noise_levels_s):
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_suite.travel_time_noise_levels_s entries "
            "must be non-negative."
        )
    _validate_non_empty_observation_fields(
        paper_pseudo_bending_ml_suite.observation_fields,
        "machine_learning.paper_pseudo_bending_ml_suite.observation_fields",
    )
    if "travel_time_s" not in paper_pseudo_bending_ml_suite.observation_fields:
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_suite.observation_fields must include "
            "travel_time_s for noise robustness evaluation."
        )
    if (
        paper_pseudo_bending_ml_suite.best_model_selection_metric
        != "fixed_validation_rmse_then_repeated_validation_rmse"
    ):
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_suite.best_model_selection_metric must be "
            "'fixed_validation_rmse_then_repeated_validation_rmse'."
        )
    strengthened_data = _mapping(data, "paper_pseudo_bending_ml_strengthened")
    paper_pseudo_bending_ml_strengthened = MachineLearningPaperPseudoBendingMLStrengthenedSettings(
        experiment_id=str(strengthened_data["experiment_id"]),
        input_manifest_path=Path(str(strengthened_data["input_manifest_path"])),
        baseline_suite_summary_path=Path(str(strengthened_data["baseline_suite_summary_path"])),
        baseline_audit_summary_path=Path(str(strengthened_data["baseline_audit_summary_path"])),
        output_dir=Path(str(strengthened_data["output_dir"])),
        candidate_models=tuple(str(item) for item in strengthened_data["candidate_models"]),
        pca_component_counts=tuple(
            _positive_int(
                item,
                "machine_learning.paper_pseudo_bending_ml_strengthened.pca_component_counts entry",
            )
            for item in strengthened_data["pca_component_counts"]
        ),
        ridge_alphas=tuple(float(item) for item in strengthened_data["ridge_alphas"]),
        fixed_split_seed=_positive_int(
            strengthened_data["fixed_split_seed"],
            "machine_learning.paper_pseudo_bending_ml_strengthened.fixed_split_seed",
        ),
        repeated_split_seeds=tuple(
            _positive_int(
                item,
                "machine_learning.paper_pseudo_bending_ml_strengthened.repeated_split_seeds entry",
            )
            for item in strengthened_data["repeated_split_seeds"]
        ),
        travel_time_noise_levels_s=tuple(
            float(item) for item in strengthened_data["travel_time_noise_levels_s"]
        ),
        noise_random_seed=_positive_int(
            strengthened_data["noise_random_seed"],
            "machine_learning.paper_pseudo_bending_ml_strengthened.noise_random_seed",
        ),
        observation_fields=tuple(str(item) for item in strengthened_data["observation_fields"]),
        best_model_selection_metric=str(strengthened_data["best_model_selection_metric"]),
    )
    if (
        paper_pseudo_bending_ml_strengthened.input_manifest_path
        != paper_pseudo_bending_observation_corpus.manifest_path
    ):
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_strengthened.input_manifest_path must "
            "match machine_learning.paper_pseudo_bending_observation_corpus.manifest_path."
        )
    if (
        paper_pseudo_bending_ml_strengthened.input_manifest_path
        == observation_pairing.manifest_path
    ):
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_strengthened.input_manifest_path must not "
            "use the historical observation_pairing manifest."
        )
    _validate_observation_comparison_models(
        paper_pseudo_bending_ml_strengthened.candidate_models,
        (
            observation_linear_baseline.model_name,
            observation_baseline.model_name,
        ),
        "machine_learning.paper_pseudo_bending_ml_strengthened.candidate_models",
    )
    if not paper_pseudo_bending_ml_strengthened.pca_component_counts:
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_strengthened.pca_component_counts must not be empty."
        )
    if not paper_pseudo_bending_ml_strengthened.ridge_alphas:
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_strengthened.ridge_alphas must not be empty."
        )
    if any(alpha <= 0.0 for alpha in paper_pseudo_bending_ml_strengthened.ridge_alphas):
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_strengthened.ridge_alphas entries must be positive."
        )
    if not paper_pseudo_bending_ml_strengthened.repeated_split_seeds:
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_strengthened.repeated_split_seeds must not be empty."
        )
    if any(
        level < 0.0 for level in paper_pseudo_bending_ml_strengthened.travel_time_noise_levels_s
    ):
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_strengthened.travel_time_noise_levels_s "
            "entries must be non-negative."
        )
    _validate_non_empty_observation_fields(
        paper_pseudo_bending_ml_strengthened.observation_fields,
        "machine_learning.paper_pseudo_bending_ml_strengthened.observation_fields",
    )
    if "travel_time_s" not in paper_pseudo_bending_ml_strengthened.observation_fields:
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_strengthened.observation_fields must "
            "include travel_time_s for noise robustness evaluation."
        )
    if (
        paper_pseudo_bending_ml_strengthened.best_model_selection_metric
        != "worst_leave_one_family_out_rmse_then_repeated_validation_rmse"
    ):
        raise ValueError(
            "machine_learning.paper_pseudo_bending_ml_strengthened.best_model_selection_metric "
            "must be 'worst_leave_one_family_out_rmse_then_repeated_validation_rmse'."
        )
    faulted_generalization_data = _mapping(
        data,
        "paper_pseudo_bending_faulted_generalization",
    )
    ridge_alpha_value = faulted_generalization_data.get("ridge_alpha")
    paper_pseudo_bending_faulted_generalization = MachineLearningPaperPseudoBendingFaultedGeneralizationSettings(
        experiment_id=str(faulted_generalization_data["experiment_id"]),
        input_manifest_path=Path(str(faulted_generalization_data["input_manifest_path"])),
        baseline_suite_summary_path=Path(
            str(faulted_generalization_data["baseline_suite_summary_path"])
        ),
        baseline_audit_summary_path=Path(
            str(faulted_generalization_data["baseline_audit_summary_path"])
        ),
        strengthened_summary_path=Path(
            str(faulted_generalization_data["strengthened_summary_path"])
        ),
        output_dir=Path(str(faulted_generalization_data["output_dir"])),
        target_family=str(faulted_generalization_data["target_family"]),
        model_name=str(faulted_generalization_data["model_name"]),
        pca_component_count=_positive_int(
            faulted_generalization_data["pca_component_count"],
            "machine_learning.paper_pseudo_bending_faulted_generalization.pca_component_count",
        ),
        ridge_alpha=None if ridge_alpha_value is None else float(ridge_alpha_value),
        fixed_split_seed=_positive_int(
            faulted_generalization_data["fixed_split_seed"],
            "machine_learning.paper_pseudo_bending_faulted_generalization.fixed_split_seed",
        ),
        repeated_split_seeds=tuple(
            _positive_int(
                item,
                "machine_learning.paper_pseudo_bending_faulted_generalization.repeated_split_seeds entry",
            )
            for item in faulted_generalization_data["repeated_split_seeds"]
        ),
        observation_fields=tuple(
            str(item) for item in faulted_generalization_data["observation_fields"]
        ),
        derived_feature_set=str(faulted_generalization_data["derived_feature_set"]),
        recommendation_policy=str(faulted_generalization_data["recommendation_policy"]),
    )
    if (
        paper_pseudo_bending_faulted_generalization.input_manifest_path
        != paper_pseudo_bending_observation_corpus.manifest_path
    ):
        raise ValueError(
            "machine_learning.paper_pseudo_bending_faulted_generalization.input_manifest_path "
            "must match machine_learning.paper_pseudo_bending_observation_corpus.manifest_path."
        )
    if (
        paper_pseudo_bending_faulted_generalization.input_manifest_path
        == observation_pairing.manifest_path
    ):
        raise ValueError(
            "machine_learning.paper_pseudo_bending_faulted_generalization.input_manifest_path "
            "must not use the historical observation_pairing manifest."
        )
    if paper_pseudo_bending_faulted_generalization.target_family != "faulted":
        raise ValueError(
            "machine_learning.paper_pseudo_bending_faulted_generalization.target_family "
            "must be 'faulted'."
        )
    _validate_observation_comparison_models(
        (paper_pseudo_bending_faulted_generalization.model_name,),
        (
            observation_linear_baseline.model_name,
            observation_baseline.model_name,
        ),
        "machine_learning.paper_pseudo_bending_faulted_generalization.model_name",
    )
    if paper_pseudo_bending_faulted_generalization.model_name == observation_baseline.model_name:
        if paper_pseudo_bending_faulted_generalization.ridge_alpha is None:
            raise ValueError(
                "machine_learning.paper_pseudo_bending_faulted_generalization.ridge_alpha "
                "must be set for the ridge model."
            )
        if paper_pseudo_bending_faulted_generalization.ridge_alpha <= 0.0:
            raise ValueError(
                "machine_learning.paper_pseudo_bending_faulted_generalization.ridge_alpha "
                "must be positive."
            )
    if not paper_pseudo_bending_faulted_generalization.repeated_split_seeds:
        raise ValueError(
            "machine_learning.paper_pseudo_bending_faulted_generalization.repeated_split_seeds "
            "must not be empty."
        )
    _validate_non_empty_observation_fields(
        paper_pseudo_bending_faulted_generalization.observation_fields,
        "machine_learning.paper_pseudo_bending_faulted_generalization.observation_fields",
    )
    if paper_pseudo_bending_faulted_generalization.derived_feature_set != "directional_slowness_v1":
        raise ValueError(
            "machine_learning.paper_pseudo_bending_faulted_generalization.derived_feature_set "
            "must be 'directional_slowness_v1'."
        )
    if (
        paper_pseudo_bending_faulted_generalization.recommendation_policy
        != "stop_after_one_bounded_intervention"
    ):
        raise ValueError(
            "machine_learning.paper_pseudo_bending_faulted_generalization.recommendation_policy "
            "must be 'stop_after_one_bounded_intervention'."
        )
    expanded_baseline_evaluation_data = _mapping(data, "expanded_baseline_evaluation")
    expanded_baseline_evaluation = MachineLearningExpandedBaselineEvaluationSettings(
        pairing_manifest_path=Path(str(expanded_baseline_evaluation_data["pairing_manifest_path"])),
        output_dir=Path(str(expanded_baseline_evaluation_data["output_dir"])),
        mean_target_experiment_id=str(
            expanded_baseline_evaluation_data["mean_target_experiment_id"]
        ),
        feature_distance_experiment_id=str(
            expanded_baseline_evaluation_data["feature_distance_experiment_id"]
        ),
        reduced_target_experiment_id=str(
            expanded_baseline_evaluation_data["reduced_target_experiment_id"]
        ),
        summary_experiment_id=str(expanded_baseline_evaluation_data["summary_experiment_id"]),
    )
    if expanded_baseline_evaluation.pairing_manifest_path != expanded_pairing.manifest_path:
        raise ValueError(
            "machine_learning.expanded_baseline_evaluation.pairing_manifest_path must match "
            "machine_learning.expanded_pairing.manifest_path."
        )

    train_fraction = float(data["train_fraction"])
    validation_fraction = float(data["validation_fraction"])
    test_fraction = float(data["test_fraction"])
    total_fraction = train_fraction + validation_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1.0e-9:
        raise ValueError("machine_learning split fractions must sum to 1.0.")

    return MachineLearningSettings(
        target_representation=str(data["target_representation"]),
        target_grid=target_grid,
        first_target_batch=first_target_batch,
        first_supervised_pairing=first_supervised_pairing,
        expanded_pairing=expanded_pairing,
        observation_pairing=observation_pairing,
        paper_pseudo_bending_observation_corpus=paper_pseudo_bending_observation_corpus,
        first_baseline=first_baseline,
        feature_baseline=feature_baseline,
        reduced_target_baseline=reduced_target_baseline,
        observation_baseline=observation_baseline,
        observation_linear_baseline=observation_linear_baseline,
        observation_mlp_baseline=observation_mlp_baseline,
        observation_random_forest_baseline=observation_random_forest_baseline,
        observation_tuning=observation_tuning,
        observation_comparison=observation_comparison,
        paper_pseudo_bending_ml_suite=paper_pseudo_bending_ml_suite,
        paper_pseudo_bending_ml_strengthened=paper_pseudo_bending_ml_strengthened,
        paper_pseudo_bending_faulted_generalization=(paper_pseudo_bending_faulted_generalization),
        expanded_baseline_evaluation=expanded_baseline_evaluation,
        baseline_models=tuple(str(item) for item in data["baseline_models"]),
        advanced_models=tuple(str(item) for item in data["advanced_models"]),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        metrics=tuple(str(item) for item in data["metrics"]),
    )


def _validate_observation_workflow_reference(
    actual_path: Path,
    expected_path: Path,
    label: str,
) -> None:
    if actual_path != expected_path:
        raise ValueError(f"{label} must match machine_learning.observation_pairing.manifest_path.")


def _validate_observation_split_strategy(split_strategy: str, label: str) -> None:
    if split_strategy != "family_stratified":
        raise ValueError(f"{label} must currently be 'family_stratified'.")


def _validate_non_empty_observation_fields(
    observation_fields: tuple[str, ...],
    label: str,
) -> None:
    if not observation_fields:
        raise ValueError(f"{label} must not be empty.")


def _validate_observation_comparison_models(
    selected_models: tuple[str, ...],
    supported_models: tuple[str, ...],
    label: str = "machine_learning.observation_comparison.selected_models",
) -> None:
    if not selected_models:
        raise ValueError(f"{label} must not be empty.")
    unsupported = sorted(set(selected_models) - set(supported_models))
    if unsupported:
        raise ValueError(f"{label} contains unsupported models: {', '.join(unsupported)}.")


def _validate_required_split_protocols(
    split_protocols: tuple[str, ...],
    label: str,
) -> None:
    required = {
        "fixed_family_stratified",
        "repeated_seeded_family_stratified",
        "leave_one_family_out",
    }
    if not split_protocols:
        raise ValueError(f"{label} must not be empty.")
    unsupported = sorted(set(split_protocols) - required)
    if unsupported:
        raise ValueError(f"{label} contains unsupported protocols: {', '.join(unsupported)}.")
    missing = sorted(required - set(split_protocols))
    if missing:
        raise ValueError(f"{label} is missing required protocols: {', '.join(missing)}.")


def _parse_random_seeds(data: dict[str, Any]) -> RandomSeedSettings:
    return RandomSeedSettings(
        global_seed=int(data["global"]),
        stations=int(data["stations"]),
        earthquakes=int(data["earthquakes"]),
        velocity_models=int(data["velocity_models"]),
        machine_learning=int(data["machine_learning"]),
    )


def _parse_outputs(data: dict[str, Any]) -> OutputSettings:
    return OutputSettings(
        generated_base_dir=Path(str(data["generated_base_dir"])),
        figures_dir=Path(str(data["figures_dir"])),
        benchmarks_dir=Path(str(data["benchmarks_dir"])),
        experiment_notes_dir=Path(str(data["experiment_notes_dir"])),
    )


def _parse_machine_learning_case_settings(
    data: dict[str, Any],
    label: str,
) -> tuple[MachineLearningExpandedScenarioCaseSettings, ...]:
    cases: list[MachineLearningExpandedScenarioCaseSettings] = []
    for case_data in data["cases"]:
        if not isinstance(case_data, dict):
            raise TypeError(f"{label}.cases must contain mappings.")
        cases.append(_parse_machine_learning_case_mapping(case_data, label=label))
    return tuple(cases)


def _parse_machine_learning_case_mapping(
    case_data: dict[str, Any],
    *,
    label: str,
    case_id_key: str = "case_id",
    case_id_value: str | None = None,
    scenario_value: str | None = None,
) -> MachineLearningExpandedScenarioCaseSettings:
    return MachineLearningExpandedScenarioCaseSettings(
        case_id=case_id_value or str(case_data[case_id_key]),
        scenario=scenario_value or str(case_data["scenario"]),
        parameter_variant_id=(
            str(case_data["parameter_variant_id"])
            if "parameter_variant_id" in case_data
            else str(case_data[case_id_key])
        ),
        geometry_profile_id=(
            str(case_data["geometry_profile_id"]) if "geometry_profile_id" in case_data else None
        ),
        station_generator_mode=(
            str(case_data["station_generator_mode"])
            if "station_generator_mode" in case_data
            else None
        ),
        station_seed=(int(case_data["station_seed"]) if "station_seed" in case_data else None),
        earthquake_seed=(
            int(case_data["earthquake_seed"]) if "earthquake_seed" in case_data else None
        ),
        layered_velocities_km_per_s=(
            _positive_float_list(
                case_data["layered_velocities_km_per_s"],
                f"{label}.layered_velocities_km_per_s",
            )
            if "layered_velocities_km_per_s" in case_data
            else None
        ),
        block_anomaly_center_km=(
            _float_triple(
                case_data["block_anomaly_center_km"],
                f"{label}.block_anomaly_center_km",
            )
            if "block_anomaly_center_km" in case_data
            else None
        ),
        block_anomaly_size_km=(
            _positive_float_triple(
                case_data["block_anomaly_size_km"],
                f"{label}.block_anomaly_size_km",
            )
            if "block_anomaly_size_km" in case_data
            else None
        ),
        block_anomaly_velocity_delta_km_per_s=(
            float(case_data["block_anomaly_velocity_delta_km_per_s"])
            if "block_anomaly_velocity_delta_km_per_s" in case_data
            else None
        ),
        fault_x_km=float(case_data["fault_x_km"]) if "fault_x_km" in case_data else None,
        fault_y_km=float(case_data["fault_y_km"]) if "fault_y_km" in case_data else None,
        fault_strike_deg=(
            float(case_data["fault_strike_deg"]) if "fault_strike_deg" in case_data else None
        ),
        fault_dip_deg=(float(case_data["fault_dip_deg"]) if "fault_dip_deg" in case_data else None),
        fault_dip_direction=(
            str(case_data["fault_dip_direction"]) if "fault_dip_direction" in case_data else None
        ),
        fault_positive_side=(
            str(case_data["fault_positive_side"]) if "fault_positive_side" in case_data else None
        ),
        fault_velocity_offset_km_per_s=(
            float(case_data["fault_velocity_offset_km_per_s"])
            if "fault_velocity_offset_km_per_s" in case_data
            else None
        ),
        salt_dome_center_km=(
            _float_triple(
                case_data["salt_dome_center_km"],
                f"{label}.salt_dome_center_km",
            )
            if "salt_dome_center_km" in case_data
            else None
        ),
        salt_dome_radii_km=(
            _positive_float_triple(
                case_data["salt_dome_radii_km"],
                f"{label}.salt_dome_radii_km",
            )
            if "salt_dome_radii_km" in case_data
            else None
        ),
        salt_dome_velocity_delta_km_per_s=(
            float(case_data["salt_dome_velocity_delta_km_per_s"])
            if "salt_dome_velocity_delta_km_per_s" in case_data
            else None
        ),
        salt_dome_body_velocity_km_per_s=(
            float(case_data["salt_dome_body_velocity_km_per_s"])
            if "salt_dome_body_velocity_km_per_s" in case_data
            else None
        ),
        dyke_intrusion_center_km=(
            _float_triple(
                case_data["dyke_intrusion_center_km"],
                f"{label}.dyke_intrusion_center_km",
            )
            if "dyke_intrusion_center_km" in case_data
            else None
        ),
        dyke_intrusion_strike_deg=(
            float(case_data["dyke_intrusion_strike_deg"])
            if "dyke_intrusion_strike_deg" in case_data
            else None
        ),
        dyke_intrusion_length_km=(
            float(case_data["dyke_intrusion_length_km"])
            if "dyke_intrusion_length_km" in case_data
            else None
        ),
        dyke_intrusion_width_km=(
            float(case_data["dyke_intrusion_width_km"])
            if "dyke_intrusion_width_km" in case_data
            else None
        ),
        dyke_intrusion_top_depth_km=(
            float(case_data["dyke_intrusion_top_depth_km"])
            if "dyke_intrusion_top_depth_km" in case_data
            else None
        ),
        dyke_intrusion_bottom_depth_km=(
            float(case_data["dyke_intrusion_bottom_depth_km"])
            if "dyke_intrusion_bottom_depth_km" in case_data
            else None
        ),
        dyke_intrusion_velocity_delta_km_per_s=(
            float(case_data["dyke_intrusion_velocity_delta_km_per_s"])
            if "dyke_intrusion_velocity_delta_km_per_s" in case_data
            else None
        ),
        dyke_intrusion_body_velocity_km_per_s=(
            float(case_data["dyke_intrusion_body_velocity_km_per_s"])
            if "dyke_intrusion_body_velocity_km_per_s" in case_data
            else None
        ),
    )


def _validate_machine_learning_case_collection(
    cases: tuple[MachineLearningExpandedScenarioCaseSettings, ...],
    simulation_method: str,
    label: str,
    *,
    allowed_scenarios: tuple[str, ...],
    allowed_scenarios_message: str,
) -> None:
    if not cases:
        raise ValueError(f"{label}.cases must not be empty.")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError(f"{label}.cases must use unique case_id values.")
    if simulation_method not in SUPPORTED_SIMULATION_METHODS:
        raise ValueError(
            f"{label}.simulation_method must be one of: {', '.join(SUPPORTED_SIMULATION_METHODS)}."
        )
    for case in cases:
        if case.scenario not in allowed_scenarios:
            raise ValueError(allowed_scenarios_message)
        if case.station_generator_mode is not None and case.station_generator_mode not in (
            "random",
            "grid",
        ):
            raise ValueError("station_generator_mode must be 'random' or 'grid'.")
        if case.station_seed is not None and case.station_seed <= 0:
            raise ValueError("station_seed must be positive when provided.")
        if case.earthquake_seed is not None and case.earthquake_seed <= 0:
            raise ValueError("earthquake_seed must be positive when provided.")
        configured_layered = case.layered_velocities_km_per_s is not None
        if case.scenario != "layered" and configured_layered:
            raise ValueError("Layered overrides are only allowed for layered cases.")
        if (
            case.scenario == "layered"
            and configured_layered
            and len(case.layered_velocities_km_per_s) == 0
        ):
            raise ValueError("Layered overrides must contain one or more velocities.")
        if case.scenario != "block_anomaly" and (
            case.block_anomaly_center_km is not None
            or case.block_anomaly_size_km is not None
            or case.block_anomaly_velocity_delta_km_per_s is not None
        ):
            raise ValueError("Block-anomaly overrides are only allowed for block_anomaly cases.")
        if case.scenario == "block_anomaly":
            configured_block_fields = [
                case.block_anomaly_center_km is not None,
                case.block_anomaly_size_km is not None,
                case.block_anomaly_velocity_delta_km_per_s is not None,
            ]
            if any(configured_block_fields) and not all(configured_block_fields):
                raise ValueError(
                    "Block-anomaly cases must either omit overrides entirely or provide all "
                    "block-anomaly override fields."
                )
        if case.scenario != "faulted" and (
            case.fault_x_km is not None
            or case.fault_y_km is not None
            or case.fault_strike_deg is not None
            or case.fault_dip_deg is not None
            or case.fault_dip_direction is not None
            or case.fault_positive_side is not None
            or case.fault_velocity_offset_km_per_s is not None
        ):
            raise ValueError("Fault overrides are only allowed for faulted cases.")
        if case.scenario == "faulted":
            configured_fault_fields = [
                case.fault_x_km is not None,
                case.fault_y_km is not None,
                case.fault_strike_deg is not None,
                case.fault_dip_deg is not None,
                case.fault_dip_direction is not None,
                case.fault_positive_side is not None,
                case.fault_velocity_offset_km_per_s is not None,
            ]
            if any(configured_fault_fields) and not all(configured_fault_fields):
                raise ValueError(
                    "Faulted cases must either omit overrides entirely or provide all fault "
                    "override fields."
                )
        if case.scenario != "salt_dome" and (
            case.salt_dome_center_km is not None
            or case.salt_dome_radii_km is not None
            or case.salt_dome_velocity_delta_km_per_s is not None
            or case.salt_dome_body_velocity_km_per_s is not None
        ):
            raise ValueError("Salt-dome overrides are only allowed for salt_dome cases.")
        if case.scenario == "salt_dome":
            salt_velocity_configured = (
                case.salt_dome_velocity_delta_km_per_s is not None
                or case.salt_dome_body_velocity_km_per_s is not None
            )
            configured_salt_fields = [
                case.salt_dome_center_km is not None,
                case.salt_dome_radii_km is not None,
                salt_velocity_configured,
            ]
            if (
                case.salt_dome_velocity_delta_km_per_s is not None
                and case.salt_dome_body_velocity_km_per_s is not None
            ):
                raise ValueError(
                    "Salt-dome cases must not provide both velocity_delta and body_velocity overrides."
                )
            if any(configured_salt_fields) and not all(configured_salt_fields):
                raise ValueError(
                    "Salt-dome cases must either omit overrides entirely or provide all salt-dome "
                    "override fields."
                )
        if case.scenario != "dyke_intrusion" and (
            case.dyke_intrusion_center_km is not None
            or case.dyke_intrusion_strike_deg is not None
            or case.dyke_intrusion_length_km is not None
            or case.dyke_intrusion_width_km is not None
            or case.dyke_intrusion_top_depth_km is not None
            or case.dyke_intrusion_bottom_depth_km is not None
            or case.dyke_intrusion_velocity_delta_km_per_s is not None
            or case.dyke_intrusion_body_velocity_km_per_s is not None
        ):
            raise ValueError("Dyke-intrusion overrides are only allowed for dyke_intrusion cases.")
        if case.scenario == "dyke_intrusion":
            dyke_velocity_configured = (
                case.dyke_intrusion_velocity_delta_km_per_s is not None
                or case.dyke_intrusion_body_velocity_km_per_s is not None
            )
            configured_dyke_fields = [
                case.dyke_intrusion_center_km is not None,
                case.dyke_intrusion_strike_deg is not None,
                case.dyke_intrusion_length_km is not None,
                case.dyke_intrusion_width_km is not None,
                case.dyke_intrusion_top_depth_km is not None,
                case.dyke_intrusion_bottom_depth_km is not None,
                dyke_velocity_configured,
            ]
            if (
                case.dyke_intrusion_velocity_delta_km_per_s is not None
                and case.dyke_intrusion_body_velocity_km_per_s is not None
            ):
                raise ValueError(
                    "Dyke-intrusion cases must not provide both velocity_delta and body_velocity overrides."
                )
            if any(configured_dyke_fields) and not all(configured_dyke_fields):
                raise ValueError(
                    "Dyke-intrusion cases must either omit overrides entirely or provide all "
                    "dyke-intrusion override fields."
                )
            if all(configured_dyke_fields):
                if case.dyke_intrusion_length_km <= 0.0 or case.dyke_intrusion_width_km <= 0.0:
                    raise ValueError("Dyke-intrusion override length and width must be positive.")
                if case.dyke_intrusion_top_depth_km < 0.0 or (
                    case.dyke_intrusion_top_depth_km >= case.dyke_intrusion_bottom_depth_km
                ):
                    raise ValueError(
                        "Dyke-intrusion override depths must be non-negative and strictly increasing."
                    )


def _parse_observation_pairing_cases(
    data: dict[str, Any],
    label: str,
) -> tuple[MachineLearningExpandedScenarioCaseSettings, ...]:
    if "cases" in data:
        return _parse_machine_learning_case_settings(data, label)
    geometry_profiles = _parse_observation_geometry_profiles(data, label)
    families = _parse_observation_family_templates(data, label)
    geometry_by_id = {profile.profile_id: profile for profile in geometry_profiles}
    cases: list[MachineLearningExpandedScenarioCaseSettings] = []
    for family in families:
        for geometry_profile_id in family.geometry_profile_ids:
            if geometry_profile_id not in geometry_by_id:
                raise ValueError(
                    f"{label}.families references unknown geometry profile '{geometry_profile_id}'."
                )
            profile = geometry_by_id[geometry_profile_id]
            for variant in family.parameter_variants:
                cases.append(
                    replace(
                        variant,
                        case_id=f"{family.scenario}_{variant.case_id}_{profile.profile_id}",
                        scenario=family.scenario,
                        parameter_variant_id=variant.case_id,
                        geometry_profile_id=profile.profile_id,
                        station_generator_mode=profile.station_generator_mode,
                        station_seed=profile.station_seed,
                        earthquake_seed=profile.earthquake_seed,
                    )
                )
    return tuple(cases)


def _parse_observation_geometry_profiles(
    data: dict[str, Any],
    label: str,
) -> tuple[MachineLearningObservationGeometryProfileSettings, ...]:
    raw_profiles = data.get("geometry_profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError(f"{label}.geometry_profiles must contain one or more profiles.")
    profiles: list[MachineLearningObservationGeometryProfileSettings] = []
    for profile_data in raw_profiles:
        if not isinstance(profile_data, dict):
            raise TypeError(f"{label}.geometry_profiles must contain mappings.")
        profiles.append(
            MachineLearningObservationGeometryProfileSettings(
                profile_id=str(profile_data["profile_id"]),
                station_generator_mode=str(profile_data["station_generator_mode"]),
                station_seed=(
                    int(profile_data["station_seed"]) if "station_seed" in profile_data else None
                ),
                earthquake_seed=(
                    int(profile_data["earthquake_seed"])
                    if "earthquake_seed" in profile_data
                    else None
                ),
            )
        )
    if len({profile.profile_id for profile in profiles}) != len(profiles):
        raise ValueError(f"{label}.geometry_profiles must use unique profile_id values.")
    return tuple(profiles)


def _parse_observation_family_templates(
    data: dict[str, Any],
    label: str,
) -> tuple[MachineLearningObservationFamilyTemplateSettings, ...]:
    raw_families = data.get("families")
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError(f"{label}.families must contain one or more family templates.")
    families: list[MachineLearningObservationFamilyTemplateSettings] = []
    for family_data in raw_families:
        if not isinstance(family_data, dict):
            raise TypeError(f"{label}.families must contain mappings.")
        scenario = str(family_data["scenario"])
        geometry_profile_ids = tuple(str(item) for item in family_data["geometry_profile_ids"])
        if not geometry_profile_ids:
            raise ValueError(f"{label}.families.geometry_profile_ids must not be empty.")
        raw_variants = family_data.get("parameter_variants")
        if not isinstance(raw_variants, list) or not raw_variants:
            raise ValueError(f"{label}.families.parameter_variants must not be empty.")
        parameter_variants: list[MachineLearningExpandedScenarioCaseSettings] = []
        for variant_data in raw_variants:
            if not isinstance(variant_data, dict):
                raise TypeError(f"{label}.families.parameter_variants must contain mappings.")
            parameter_variants.append(
                _parse_machine_learning_case_mapping(
                    variant_data,
                    label=f"{label}.{scenario}",
                    case_id_key="variant_id",
                    scenario_value=scenario,
                )
            )
        if len({variant.case_id for variant in parameter_variants}) != len(parameter_variants):
            raise ValueError(
                f"{label}.families.parameter_variants must use unique variant_id values within a family."
            )
        families.append(
            MachineLearningObservationFamilyTemplateSettings(
                scenario=scenario,
                geometry_profile_ids=geometry_profile_ids,
                parameter_variants=tuple(parameter_variants),
            )
        )
    if len({family.scenario for family in families}) != len(families):
        raise ValueError(f"{label}.families must use each scenario at most once.")
    return tuple(families)


def _parse_vertical_slice(data: dict[str, Any]) -> VerticalSliceSettings:
    return VerticalSliceSettings(
        experiment_id=str(data["experiment_id"]),
        scenario_id=str(data["scenario_id"]),
        station_configuration_id=str(data["station_configuration_id"]),
        earthquake_configuration_id=str(data["earthquake_configuration_id"]),
        velocity_model_id=str(data["velocity_model_id"]),
        simulation_id=str(data["simulation_id"]),
        dataset_id=str(data["dataset_id"]),
        station_count=_positive_int(data["station_count"], "vertical_slice.station_count"),
        earthquake_count=_positive_int(data["earthquake_count"], "vertical_slice.earthquake_count"),
        case_id=str(data["case_id"]) if data.get("case_id") is not None else None,
    )


def _validate_generation_bounds(
    study_area: StudyAreaSettings,
    earthquake_generation: EarthquakeGenerationSettings,
    velocity_model_generation: VelocityModelGenerationSettings | None = None,
) -> None:
    _apply_margin(study_area.x_range_km, study_area.margin_km)
    _apply_margin(study_area.y_range_km, study_area.margin_km)
    z_bounds = _apply_margin(study_area.z_range_km, study_area.margin_km)
    depth_min = max(z_bounds[0], earthquake_generation.depth_range_km[0])
    depth_max = min(z_bounds[1], earthquake_generation.depth_range_km[1])
    if depth_min >= depth_max:
        raise ValueError("Earthquake depth bounds are empty after applying margin and depth range.")
    if velocity_model_generation is None:
        return
    salt = velocity_model_generation.salt_dome
    if not (
        study_area.x_range_km[0] <= salt.center_km[0] <= study_area.x_range_km[1]
        and study_area.y_range_km[0] <= salt.center_km[1] <= study_area.y_range_km[1]
        and study_area.z_range_km[0] <= salt.center_km[2] <= study_area.z_range_km[1]
    ):
        raise ValueError("salt_dome.center_km must stay inside the configured study area.")
    dyke = velocity_model_generation.dyke_intrusion
    if not (
        study_area.x_range_km[0] <= dyke.center_km[0] <= study_area.x_range_km[1]
        and study_area.y_range_km[0] <= dyke.center_km[1] <= study_area.y_range_km[1]
        and study_area.z_range_km[0] <= dyke.center_km[2] <= study_area.z_range_km[1]
    ):
        raise ValueError("dyke_intrusion.center_km must stay inside the configured study area.")
    if (
        not study_area.z_range_km[0]
        <= dyke.top_depth_km
        < dyke.bottom_depth_km
        <= study_area.z_range_km[1]
    ):
        raise ValueError(
            "dyke_intrusion top and bottom depths must stay inside the configured study-area depth range."
        )


def _validate_ml_target_batch(
    machine_learning: MachineLearningSettings,
    velocity_model_generation: VelocityModelGenerationSettings,
) -> None:
    unsupported = sorted(
        set(machine_learning.first_target_batch.scenarios)
        - set(velocity_model_generation.supported_scenarios)
    )
    if unsupported:
        raise ValueError(
            "machine_learning.first_target_batch.scenarios contains scenarios not supported by "
            "velocity_model_generation.supported_scenarios: "
            f"{', '.join(unsupported)}."
        )
    unsupported_pairing = sorted(
        set(machine_learning.first_supervised_pairing.scenarios)
        - set(velocity_model_generation.supported_scenarios)
    )
    if unsupported_pairing:
        raise ValueError(
            "machine_learning.first_supervised_pairing.scenarios contains scenarios not "
            "supported by velocity_model_generation.supported_scenarios: "
            f"{', '.join(unsupported_pairing)}."
        )
    unsupported_expanded = sorted(
        {case.scenario for case in machine_learning.expanded_pairing.cases}
        - set(velocity_model_generation.supported_scenarios)
    )
    if unsupported_expanded:
        raise ValueError(
            "machine_learning.expanded_pairing.cases contains scenarios not supported by "
            "velocity_model_generation.supported_scenarios: "
            f"{', '.join(unsupported_expanded)}."
        )
    unsupported_observation = sorted(
        {case.scenario for case in machine_learning.observation_pairing.cases}
        - set(velocity_model_generation.supported_scenarios)
    )
    if unsupported_observation:
        raise ValueError(
            "machine_learning.observation_pairing.cases contains scenarios not supported by "
            "velocity_model_generation.supported_scenarios: "
            f"{', '.join(unsupported_observation)}."
        )
    unsupported_paper_observation = sorted(
        {case.scenario for case in machine_learning.paper_pseudo_bending_observation_corpus.cases}
        - set(velocity_model_generation.supported_scenarios)
    )
    if unsupported_paper_observation:
        raise ValueError(
            "machine_learning.paper_pseudo_bending_observation_corpus.cases contains scenarios "
            "not supported by velocity_model_generation.supported_scenarios: "
            f"{', '.join(unsupported_paper_observation)}."
        )


def _apply_margin(bounds: tuple[float, float], margin: float) -> tuple[float, float]:
    low, high = bounds
    if low >= high:
        raise ValueError(f"Invalid bounds: {bounds}")
    if margin < 0:
        raise ValueError("Margins must be non-negative.")
    narrowed = (low + margin, high - margin)
    if narrowed[0] >= narrowed[1]:
        raise ValueError(f"Margin {margin} km leaves no usable interval for bounds {bounds}.")
    return narrowed


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"Expected '{key}' to be a mapping.")
    return value


def _float_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise TypeError(f"{label} must contain exactly two numeric values.")
    low, high = float(value[0]), float(value[1])
    if low >= high:
        raise ValueError(f"{label} must be increasing.")
    return low, high


def _float_triple(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise TypeError(f"{label} must contain exactly three numeric values.")
    return float(value[0]), float(value[1]), float(value[2])


def _positive_float_triple(value: Any, label: str) -> tuple[float, float, float]:
    parsed = _float_triple(value, label)
    if any(component <= 0.0 for component in parsed):
        raise ValueError(f"{label} must contain only positive values.")
    return parsed


def _positive_float_list(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise TypeError(f"{label} must contain one or more numeric values.")
    parsed = tuple(float(item) for item in value)
    if any(item <= 0.0 for item in parsed):
        raise ValueError(f"{label} values must be positive.")
    return parsed


def _positive_float(value: Any, label: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return parsed


def _connectivity_list(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise TypeError(f"{label} must contain one or more connectivity values.")
    parsed = tuple(int(item) for item in value)
    if any(item not in (6, 18, 26) for item in parsed):
        raise ValueError(f"{label} values must be drawn from 6, 18, or 26.")
    return parsed


def _positive_int(value: Any, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive.")
    return parsed
