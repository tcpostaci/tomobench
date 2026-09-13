"""End-to-end workflow for the first synthetic research vertical slice."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.datasets.builder import build_source_receiver_dataset
from tomobench.datasets.io import export_dataset_csv, export_dataset_metadata
from tomobench.datasets.targets import (
    build_velocity_grid_target_spec,
    save_velocity_grid_target_spec,
)
from tomobench.generation.earthquakes import (
    generate_earthquake_configuration,
    save_earthquake_configuration,
)
from tomobench.generation.stations import (
    generate_station_configuration,
    save_station_configuration,
)
from tomobench.generation.velocity_models import (
    generate_layered_velocity_model,
    save_velocity_model,
)
from tomobench.generation.velocity_grids import (
    build_cartesian_velocity_grid,
    save_velocity_grid,
)
from tomobench.simulation.travel_times import (
    EIKONAL_3D_FIRST_ARRIVAL_METHOD,
    Eikonal3DFirstArrivalTravelTimeSimulator,
    LAYERED_RAY_TRACING_METHOD,
    PSEUDO_BENDING_3D_METHOD,
    STRAIGHT_LINE_LAYERED_METHOD,
    LayeredRayTracingTravelTimeSimulator,
    PseudoBending3DTravelTimeSimulator,
    StraightLineLayeredTravelTimeSimulator,
    build_travel_time_simulator,
    save_travel_time_ray_paths_jsonl,
    save_travel_time_csv,
    save_travel_time_observations,
)
from tomobench.tomography.sensitivity import save_ray_path_sensitivity_sidecar
from tomobench.utils.paths import get_repo_root
from tomobench.visualization.plots import (
    save_geometry_svg,
    save_velocity_profile_svg,
)
from tomobench.visualization.velocity_grid_section import save_velocity_grid_section_figure


@dataclass(frozen=True)
class VerticalSliceOutputs:
    """Paths produced by the first vertical-slice workflow."""

    station_configuration: Path
    earthquake_configuration: Path
    velocity_model: Path
    velocity_grid: Path
    target_spec: Path
    travel_times_json: Path
    travel_times_csv: Path
    ray_paths_jsonl: Path
    sensitivity_jsonl: Path
    sensitivity_metadata: Path
    dataset_csv: Path
    dataset_metadata: Path
    geometry_figure: Path
    velocity_figure: Path
    subsurface_figure: Path
    experiment_note: Path


COMPARISON_CSV_NAME = "first_vertical_slice_travel_time_method_comparison.csv"
ALL_VELOCITY_SCENARIOS = (
    "layered",
    "block_anomaly",
    "faulted",
    "salt_dome",
    "dyke_intrusion",
)


def run_first_vertical_slice(
    settings: BenchmarkSettings | None = None,
    simulation_method: str | None = None,
    velocity_scenario: str | None = None,
) -> VerticalSliceOutputs:
    """Run the complete first synthetic research vertical slice."""
    loaded_settings = _settings_with_velocity_scenario(
        _settings_with_simulation_method(settings or load_settings(), simulation_method),
        velocity_scenario,
    )
    repo_root = get_repo_root()
    paths = _build_output_paths(loaded_settings, repo_root)

    stations = generate_station_configuration(loaded_settings)
    earthquakes = generate_earthquake_configuration(loaded_settings)
    velocity_model = generate_layered_velocity_model(loaded_settings)

    station_path = save_station_configuration(stations, paths.station_configuration)
    earthquake_path = save_earthquake_configuration(earthquakes, paths.earthquake_configuration)
    velocity_model_path = save_velocity_model(velocity_model, paths.velocity_model)
    velocity_grid = build_cartesian_velocity_grid(loaded_settings, velocity_model)
    velocity_grid_path = save_velocity_grid(velocity_grid, paths.velocity_grid)
    target_spec = build_velocity_grid_target_spec(velocity_grid, loaded_settings)
    target_spec_path = save_velocity_grid_target_spec(target_spec, paths.target_spec)

    simulator = build_travel_time_simulator(loaded_settings)
    ray_paths = ()
    if isinstance(simulator, PseudoBending3DTravelTimeSimulator):
        observations, ray_paths = simulator.simulate_with_ray_paths(
            stations,
            earthquakes,
            velocity_model,
        )
    else:
        observations = simulator.simulate(stations, earthquakes, velocity_model)
    travel_json_path = save_travel_time_observations(
        observations,
        paths.travel_times_json,
        loaded_settings,
    )
    travel_csv_path = save_travel_time_csv(observations, paths.travel_times_csv)
    ray_paths_path = (
        save_travel_time_ray_paths_jsonl(ray_paths, paths.ray_paths_jsonl, loaded_settings)
        if ray_paths
        else paths.ray_paths_jsonl
    )
    sensitivity_path = paths.sensitivity_jsonl
    sensitivity_metadata_path = paths.sensitivity_metadata
    if ray_paths:
        sensitivity_path, sensitivity_metadata_path, sensitivity_validation = (
            save_ray_path_sensitivity_sidecar(
                ray_paths,
                velocity_grid,
                paths.sensitivity_jsonl,
                paths.sensitivity_metadata,
                loaded_settings,
                source_ray_path_sidecar=ray_paths_path,
            )
        )
        if not sensitivity_validation.passed:
            raise ValueError(
                "Generated ray-cell sensitivity sidecar failed path-length validation: "
                + "; ".join(sensitivity_validation.errors)
            )

    rows = build_source_receiver_dataset(observations, loaded_settings)
    dataset_csv_path = export_dataset_csv(rows, paths.dataset_csv)
    source_files = {
        "station_configuration": _relative_to_repo(station_path, repo_root),
        "earthquake_configuration": _relative_to_repo(earthquake_path, repo_root),
        "velocity_model": _relative_to_repo(velocity_model_path, repo_root),
        "velocity_grid": _relative_to_repo(velocity_grid_path, repo_root),
        "target_spec": _relative_to_repo(target_spec_path, repo_root),
        "travel_times": _relative_to_repo(travel_json_path, repo_root),
    }
    if ray_paths:
        source_files["ray_paths"] = _relative_to_repo(ray_paths_path, repo_root)
        source_files["ray_cell_sensitivity"] = _relative_to_repo(sensitivity_path, repo_root)
        source_files["ray_cell_sensitivity_metadata"] = _relative_to_repo(
            sensitivity_metadata_path,
            repo_root,
        )
    dataset_metadata_path = export_dataset_metadata(
        rows,
        paths.dataset_metadata,
        loaded_settings,
        source_files=source_files,
    )

    geometry_path = save_geometry_svg(stations, earthquakes, loaded_settings, paths.geometry_figure)
    velocity_figure_path = save_velocity_profile_svg(velocity_model, paths.velocity_figure)
    subsurface_figure_path = save_velocity_grid_section_figure(
        velocity_grid, paths.subsurface_figure
    )
    experiment_note_path = _write_experiment_note(
        loaded_settings,
        paths.experiment_note,
        outputs=VerticalSliceOutputs(
            station_configuration=station_path,
            earthquake_configuration=earthquake_path,
            velocity_model=velocity_model_path,
            velocity_grid=velocity_grid_path,
            target_spec=target_spec_path,
            travel_times_json=travel_json_path,
            travel_times_csv=travel_csv_path,
            ray_paths_jsonl=ray_paths_path,
            sensitivity_jsonl=sensitivity_path,
            sensitivity_metadata=sensitivity_metadata_path,
            dataset_csv=dataset_csv_path,
            dataset_metadata=dataset_metadata_path,
            geometry_figure=geometry_path,
            velocity_figure=velocity_figure_path,
            subsurface_figure=subsurface_figure_path,
            experiment_note=paths.experiment_note,
        ),
        observation_count=len(observations),
        repo_root=repo_root,
    )

    return VerticalSliceOutputs(
        station_configuration=station_path,
        earthquake_configuration=earthquake_path,
        velocity_model=velocity_model_path,
        velocity_grid=velocity_grid_path,
        target_spec=target_spec_path,
        travel_times_json=travel_json_path,
        travel_times_csv=travel_csv_path,
        ray_paths_jsonl=ray_paths_path,
        sensitivity_jsonl=sensitivity_path,
        sensitivity_metadata=sensitivity_metadata_path,
        dataset_csv=dataset_csv_path,
        dataset_metadata=dataset_metadata_path,
        geometry_figure=geometry_path,
        velocity_figure=velocity_figure_path,
        subsurface_figure=subsurface_figure_path,
        experiment_note=experiment_note_path,
    )


def compare_first_slice_simulators(settings: BenchmarkSettings | None = None) -> Path:
    """Compare all first-slice simulator methods on identical inputs."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    comparison_path = (
        repo_root
        / loaded_settings.dataset_export.base_dir
        / _comparison_csv_name(loaded_settings.velocity_model_generation.default_scenario)
    )

    stations = generate_station_configuration(loaded_settings)
    earthquakes = generate_earthquake_configuration(loaded_settings)
    velocity_model = generate_layered_velocity_model(loaded_settings)

    straight_settings = _settings_with_simulation_method(
        loaded_settings, STRAIGHT_LINE_LAYERED_METHOD
    )
    layered_settings = _settings_with_simulation_method(loaded_settings, LAYERED_RAY_TRACING_METHOD)
    eikonal_settings = _settings_with_simulation_method(
        loaded_settings, EIKONAL_3D_FIRST_ARRIVAL_METHOD
    )

    straight_observations = StraightLineLayeredTravelTimeSimulator(straight_settings).simulate(
        stations, earthquakes, velocity_model
    )
    layered_observations = LayeredRayTracingTravelTimeSimulator(layered_settings).simulate(
        stations, earthquakes, velocity_model
    )
    velocity_grid = build_cartesian_velocity_grid(eikonal_settings, velocity_model)
    eikonal_observations = Eikonal3DFirstArrivalTravelTimeSimulator(
        eikonal_settings, velocity_grid=velocity_grid
    ).simulate(stations, earthquakes, velocity_model)

    if not (len(straight_observations) == len(layered_observations) == len(eikonal_observations)):
        raise ValueError("Simulator outputs have different observation counts.")

    layered_by_id = {
        observation.observation_id: observation for observation in layered_observations
    }
    eikonal_by_id = {
        observation.observation_id: observation for observation in eikonal_observations
    }
    rows: list[dict[str, object]] = []
    for straight in straight_observations:
        layered = layered_by_id[straight.observation_id]
        eikonal = eikonal_by_id[straight.observation_id]
        difference = layered.travel_time_s - straight.travel_time_s
        relative_difference = (
            (difference / straight.travel_time_s) * 100.0 if straight.travel_time_s != 0.0 else 0.0
        )
        rows.append(
            {
                "observation_id": straight.observation_id,
                "earthquake_id": straight.earthquake_id,
                "station_id": straight.station_id,
                "straight_line_travel_time_s": straight.travel_time_s,
                "layered_ray_tracing_travel_time_s": layered.travel_time_s,
                "eikonal_3d_travel_time_s": eikonal.travel_time_s,
                "difference_s": difference,
                "relative_difference_percent": relative_difference,
                "path_length_km": straight.path_length_km,
                "eikonal_minus_layered_ray_s": eikonal.travel_time_s - layered.travel_time_s,
                "eikonal_minus_straight_line_s": eikonal.travel_time_s - straight.travel_time_s,
                "horizontal_offset_km": _horizontal_offset_km(straight.source, straight.receiver),
                "straight_line_simulation_method": straight.simulation_method,
                "layered_ray_tracing_simulation_method": layered.simulation_method,
                "eikonal_3d_simulation_method": eikonal.simulation_method,
            }
        )

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = tuple(rows[0].keys()) if rows else comparison_columns()
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return comparison_path


def render_all_velocity_scenarios(
    settings: BenchmarkSettings | None = None,
) -> tuple[VerticalSliceOutputs, ...]:
    """Run the first-slice workflow for every supported current scenario family."""
    loaded_settings = settings or load_settings()
    return tuple(
        run_first_vertical_slice(
            settings=loaded_settings,
            velocity_scenario=scenario,
        )
        for scenario in ALL_VELOCITY_SCENARIOS
    )


def comparison_columns() -> tuple[str, ...]:
    """Return the simulator comparison CSV schema."""
    return (
        "observation_id",
        "earthquake_id",
        "station_id",
        "straight_line_travel_time_s",
        "layered_ray_tracing_travel_time_s",
        "eikonal_3d_travel_time_s",
        "difference_s",
        "relative_difference_percent",
        "path_length_km",
        "eikonal_minus_layered_ray_s",
        "eikonal_minus_straight_line_s",
        "horizontal_offset_km",
        "straight_line_simulation_method",
        "layered_ray_tracing_simulation_method",
        "eikonal_3d_simulation_method",
    )


def _build_output_paths(settings: BenchmarkSettings, repo_root: Path) -> VerticalSliceOutputs:
    generated_base = repo_root / settings.outputs.generated_base_dir
    figures_dir = repo_root / settings.outputs.figures_dir
    dataset_dir = repo_root / settings.dataset_export.base_dir
    experiment_dir = repo_root / settings.outputs.experiment_notes_dir

    experiment_id = settings.vertical_slice.experiment_id
    artifact_suffix = _artifact_suffix(settings.velocity_model_generation.default_scenario)
    return VerticalSliceOutputs(
        station_configuration=(
            generated_base / "stations" / f"{settings.vertical_slice.station_configuration_id}.json"
        ),
        earthquake_configuration=(
            generated_base
            / "earthquakes"
            / f"{settings.vertical_slice.earthquake_configuration_id}.json"
        ),
        velocity_model=(
            generated_base / "velocity_models" / f"{settings.vertical_slice.velocity_model_id}.json"
        ),
        velocity_grid=(
            generated_base
            / "velocity_grids"
            / f"{settings.vertical_slice.velocity_model_id}{artifact_suffix}_cartesian_3d_grid.json"
        ),
        target_spec=(
            dataset_dir / f"{settings.vertical_slice.dataset_id}{artifact_suffix}.target_spec.json"
        ),
        travel_times_json=(
            generated_base
            / "travel_times"
            / f"{settings.vertical_slice.simulation_id}{artifact_suffix}.json"
        ),
        travel_times_csv=(
            generated_base
            / "travel_times"
            / f"{settings.vertical_slice.simulation_id}{artifact_suffix}.csv"
        ),
        ray_paths_jsonl=(
            generated_base
            / "ray_paths"
            / f"{settings.vertical_slice.simulation_id}{artifact_suffix}.ray_paths.jsonl"
        ),
        sensitivity_jsonl=(
            generated_base
            / "sensitivity_matrices"
            / f"{settings.vertical_slice.simulation_id}{artifact_suffix}.ray_cell_sensitivity.jsonl"
        ),
        sensitivity_metadata=(
            generated_base
            / "sensitivity_matrices"
            / f"{settings.vertical_slice.simulation_id}{artifact_suffix}.ray_cell_sensitivity.metadata.json"
        ),
        dataset_csv=dataset_dir / f"{settings.vertical_slice.dataset_id}{artifact_suffix}.csv",
        dataset_metadata=dataset_dir
        / f"{settings.vertical_slice.dataset_id}{artifact_suffix}.metadata.json",
        geometry_figure=figures_dir / f"{experiment_id}{artifact_suffix}_geometry.svg",
        velocity_figure=figures_dir / f"{experiment_id}{artifact_suffix}_velocity_profile.svg",
        subsurface_figure=figures_dir / f"{experiment_id}{artifact_suffix}_subsurface_section.png",
        experiment_note=experiment_dir / f"{experiment_id}{artifact_suffix}.md",
    )


def _write_experiment_note(
    settings: BenchmarkSettings,
    path: Path,
    outputs: VerticalSliceOutputs,
    observation_count: int,
    repo_root: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment: First Synthetic Vertical Slice",
        "",
        f"Experiment ID: `{settings.vertical_slice.experiment_id}`",
        "",
        "## Objective",
        "",
        (
            "Demonstrate a small, reproducible end-to-end synthetic tomography workflow "
            "from configuration loading through geometry generation, a simple layered "
            "travel-time calculation, dataset export, and sanity-check visualization."
        ),
        "",
        "## Inputs Used",
        "",
        "- Central configuration: `config/benchmark_config.yaml`",
        f"- Station seed: `{settings.random_seeds.stations}`",
        f"- Earthquake seed: `{settings.random_seeds.earthquakes}`",
        f"- Velocity model seed recorded for provenance: `{settings.random_seeds.velocity_models}`",
        "",
        "## Configuration Summary",
        "",
        f"- Study area: x={settings.study_area.x_range_km} km, "
        f"y={settings.study_area.y_range_km} km, z={settings.study_area.z_range_km} km",
        f"- Margin: `{settings.study_area.margin_km}` km",
        f"- Stations: `{settings.vertical_slice.station_count}` `{settings.station_generation.default_mode}` surface stations",
        f"- Earthquakes: `{settings.vertical_slice.earthquake_count}` `{settings.earthquake_generation.default_mode}` hypocenters",
        (
            f"- Velocity model: `{settings.velocity_model_generation.layered.layer_count}` "
            "horizontal constant-velocity layers"
        ),
        f"- Velocity-grid scenario: `{settings.velocity_model_generation.default_scenario}`",
        f"- Travel-time method: `{settings.simulation.method}`",
        f"- Dataset rows: `{observation_count}` source-receiver observations",
        "",
        "## Outputs Created",
        "",
        f"- Station configuration: `{_relative_to_repo(outputs.station_configuration, repo_root)}`",
        f"- Earthquake configuration: `{_relative_to_repo(outputs.earthquake_configuration, repo_root)}`",
        f"- Velocity model: `{_relative_to_repo(outputs.velocity_model, repo_root)}`",
        f"- Velocity grid: `{_relative_to_repo(outputs.velocity_grid, repo_root)}`",
        f"- Target spec: `{_relative_to_repo(outputs.target_spec, repo_root)}`",
        f"- Simulation JSON: `{_relative_to_repo(outputs.travel_times_json, repo_root)}`",
        f"- Simulation CSV: `{_relative_to_repo(outputs.travel_times_csv, repo_root)}`",
        f"- Ray-path sidecar JSONL: `{_relative_to_repo(outputs.ray_paths_jsonl, repo_root)}`",
        f"- Ray-cell sensitivity JSONL: `{_relative_to_repo(outputs.sensitivity_jsonl, repo_root)}`",
        f"- Ray-cell sensitivity metadata: `{_relative_to_repo(outputs.sensitivity_metadata, repo_root)}`",
        f"- Dataset CSV: `{_relative_to_repo(outputs.dataset_csv, repo_root)}`",
        f"- Dataset metadata: `{_relative_to_repo(outputs.dataset_metadata, repo_root)}`",
        f"- Geometry figure: `{_relative_to_repo(outputs.geometry_figure, repo_root)}`",
        f"- Velocity profile figure: `{_relative_to_repo(outputs.velocity_figure, repo_root)}`",
        f"- Subsurface section figure: `{_relative_to_repo(outputs.subsurface_figure, repo_root)}`",
        "",
        "## Assumptions",
        "",
    ]
    lines.extend(f"- {assumption}" for assumption in settings.simulation.assumptions)
    lines.extend(
        [
            "- Station and earthquake coordinates use kilometers in a Cartesian coordinate system.",
            "- Positive `z_km` denotes depth below the surface.",
            "- This note documents a software and data-flow check, not a scientific result.",
            "",
            "## Known Limitations",
            "",
            "- The selected simulator is still an approximation, not full 3D tomography-grade ray tracing.",
            "- The 3D eikonal prototype maps off-grid sources and receivers to nearest grid nodes.",
            "- The layered ray-parameter method models only the direct branch and does not include head waves.",
            "- The selected velocity-grid scenario may still be a simplified synthetic representation rather than a geologically complete 3D model.",
            "- Source and receiver placement is random rather than geologically constrained.",
            "- No machine-learning inversion or classical tomography comparison is included.",
            "- Pseudo-bending ray paths are persisted as sidecar artifacts to support the Phase 6 classical tomography baseline, but no inversion has been evaluated yet.",
            "- Ray-cell sensitivity sidecars represent geometric path lengths through cells derived from adjacent node intervals; they are G-matrix inputs, not tomography results.",
            "",
            "## Next Recommended Step",
            "",
            (
                "Review the layered ray-parameter comparison output, then decide whether "
                "the next simulator slice should add head-wave handling or move toward a "
                "documented 3D ray-tracing or eikonal-solver backend."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _settings_with_simulation_method(
    settings: BenchmarkSettings,
    simulation_method: str | None,
) -> BenchmarkSettings:
    if simulation_method is None:
        return settings
    normalized_method = _normalize_simulation_method(simulation_method)
    return replace(
        settings,
        simulation=replace(settings.simulation, method=normalized_method),
    )


def _settings_with_velocity_scenario(
    settings: BenchmarkSettings,
    velocity_scenario: str | None,
) -> BenchmarkSettings:
    if velocity_scenario is None:
        return settings
    normalized_scenario = _normalize_velocity_scenario(velocity_scenario)
    return replace(
        settings,
        velocity_model_generation=replace(
            settings.velocity_model_generation,
            default_scenario=normalized_scenario,
        ),
    )


def _normalize_simulation_method(simulation_method: str) -> str:
    normalized = simulation_method.strip().replace("-", "_")
    aliases = {
        "straight_line": STRAIGHT_LINE_LAYERED_METHOD,
        "straight_line_layered": STRAIGHT_LINE_LAYERED_METHOD,
        "straight_line_layered_placeholder": STRAIGHT_LINE_LAYERED_METHOD,
        "layered_ray": LAYERED_RAY_TRACING_METHOD,
        "layered_ray_tracing": LAYERED_RAY_TRACING_METHOD,
        "eikonal_3d": EIKONAL_3D_FIRST_ARRIVAL_METHOD,
        "eikonal_3d_first_arrival": EIKONAL_3D_FIRST_ARRIVAL_METHOD,
        "eikonal_3d_first_arrival_prototype": EIKONAL_3D_FIRST_ARRIVAL_METHOD,
        "pseudo_bending": PSEUDO_BENDING_3D_METHOD,
        "pseudo_bending_3d": PSEUDO_BENDING_3D_METHOD,
        "pseudo_bending_ray_tracing": PSEUDO_BENDING_3D_METHOD,
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported simulator. Use one of: straight-line, "
            "straight-line-layered-placeholder, layered-ray, layered-ray-tracing, "
            "eikonal-3d, pseudo-bending."
        )
    return aliases[normalized]


def _normalize_velocity_scenario(velocity_scenario: str) -> str:
    normalized = velocity_scenario.strip().lower().replace("-", "_")
    aliases = {
        "layered": "layered",
        "block_anomaly": "block_anomaly",
        "faulted": "faulted",
        "salt_dome": "salt_dome",
        "dyke_intrusion": "dyke_intrusion",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported velocity scenario. Use one of: layered, block-anomaly, faulted, salt-dome, dyke-intrusion."
        )
    return aliases[normalized]


def _artifact_suffix(velocity_scenario: str) -> str:
    normalized = _normalize_velocity_scenario(velocity_scenario)
    return "" if normalized == "layered" else f"_{normalized}"


def _comparison_csv_name(velocity_scenario: str) -> str:
    return f"first_vertical_slice_travel_time_method_comparison{_artifact_suffix(velocity_scenario)}.csv"


def _horizontal_offset_km(source: object, receiver: object) -> float:
    return ((source.x_km - receiver.x_km) ** 2 + (source.y_km - receiver.y_km) ** 2) ** 0.5
