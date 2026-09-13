"""Validation checks for generated first-slice artifacts."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.datasets.builder import dataset_columns
from tomobench.simulation.travel_times import PSEUDO_BENDING_3D_METHOD
from tomobench.tomography.sensitivity import validate_sensitivity_sidecar_files
from tomobench.utils.paths import get_repo_root
from tomobench.workflow import _build_output_paths


_MANUSCRIPT_RECORD_PATH = re.compile(
    r"(?<![A-Za-z0-9_./-])records/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*"
)


def referenced_record_paths(text: str) -> list[str]:
    """Return unique POSIX record paths cited by a manuscript-facing document."""

    return sorted(set(_MANUSCRIPT_RECORD_PATH.findall(text)))


def missing_referenced_record_paths(
    documents: list[Path] | tuple[Path, ...],
    reproducibility_root: Path,
) -> list[str]:
    """Return cited ``records/...`` paths absent from a reproducibility package."""

    missing: set[str] = set()
    for document in documents:
        for relative in referenced_record_paths(document.read_text(encoding="utf-8")):
            if not (reproducibility_root / relative).is_file():
                missing.add(relative)
    return sorted(missing)


@dataclass(frozen=True)
class ValidationResult:
    """Summary of validation checks for the first synthetic slice."""

    passed: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...]


def validate_first_vertical_slice(settings: BenchmarkSettings | None = None) -> ValidationResult:
    """Validate generated artifacts for the first synthetic vertical slice."""
    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    paths = _build_output_paths(loaded_settings, repo_root)

    checks: list[str] = []
    errors: list[str] = []

    required_files = {
        "station configuration": paths.station_configuration,
        "earthquake configuration": paths.earthquake_configuration,
        "velocity model": paths.velocity_model,
        "velocity grid": paths.velocity_grid,
        "target spec": paths.target_spec,
        "travel-time JSON": paths.travel_times_json,
        "travel-time CSV": paths.travel_times_csv,
        "dataset CSV": paths.dataset_csv,
        "dataset metadata": paths.dataset_metadata,
        "geometry figure": paths.geometry_figure,
        "velocity figure": paths.velocity_figure,
        "subsurface section figure": paths.subsurface_figure,
        "experiment note": paths.experiment_note,
    }
    if loaded_settings.simulation.method == PSEUDO_BENDING_3D_METHOD:
        required_files["ray-path sidecar JSONL"] = paths.ray_paths_jsonl
        required_files["ray-cell sensitivity JSONL"] = paths.sensitivity_jsonl
        required_files["ray-cell sensitivity metadata"] = paths.sensitivity_metadata
    for label, path in required_files.items():
        if path.is_file():
            checks.append(f"{label} exists: {path}")
        else:
            errors.append(f"Missing {label}: {path}")

    if errors:
        return ValidationResult(passed=False, checks=tuple(checks), errors=tuple(errors))

    station_payload = _read_json(paths.station_configuration)
    earthquake_payload = _read_json(paths.earthquake_configuration)
    travel_payload = _read_json(paths.travel_times_json)
    metadata_payload = _read_json(paths.dataset_metadata)
    dataset_rows = _read_csv(paths.dataset_csv)
    travel_method = str(travel_payload.get("method", ""))

    _validate_station_payload(station_payload, loaded_settings, checks, errors)
    _validate_earthquake_payload(earthquake_payload, loaded_settings, checks, errors)
    _validate_travel_payload(travel_payload, loaded_settings, checks, errors)
    _validate_dataset_rows(
        dataset_rows, metadata_payload, loaded_settings, travel_method, checks, errors
    )
    if loaded_settings.simulation.method == PSEUDO_BENDING_3D_METHOD:
        _validate_ray_path_sidecar(paths.ray_paths_jsonl, dataset_rows, metadata_payload, checks, errors)
        _validate_sensitivity_sidecar(
            paths.sensitivity_jsonl,
            paths.ray_paths_jsonl,
            metadata_payload,
            checks,
            errors,
        )

    return ValidationResult(passed=not errors, checks=tuple(checks), errors=tuple(errors))


def _validate_station_payload(
    payload: dict[str, object],
    settings: BenchmarkSettings,
    checks: list[str],
    errors: list[str],
) -> None:
    if payload.get("configuration_id") != settings.vertical_slice.station_configuration_id:
        errors.append("Station configuration ID does not match config.")
    stations = _list(payload.get("stations"))
    if len(stations) == settings.vertical_slice.station_count:
        checks.append("Station count matches config.")
    else:
        errors.append(
            f"Station count mismatch: expected {settings.vertical_slice.station_count}, "
            f"found {len(stations)}."
        )
    x_min, x_max = settings.study_area.station_x_bounds_km
    y_min, y_max = settings.study_area.station_y_bounds_km
    z_km = settings.station_generation.station_elevation_km
    for station in stations:
        location = _dict(station.get("location"))
        if not (
            x_min <= float(location["x_km"]) <= x_max
            and y_min <= float(location["y_km"]) <= y_max
            and float(location["z_km"]) == z_km
        ):
            errors.append(f"Station outside configured receiver bounds: {station}")
            return
    checks.append("All stations respect configured lateral bounds and surface elevation.")


def _validate_earthquake_payload(
    payload: dict[str, object],
    settings: BenchmarkSettings,
    checks: list[str],
    errors: list[str],
) -> None:
    if payload.get("configuration_id") != settings.vertical_slice.earthquake_configuration_id:
        errors.append("Earthquake configuration ID does not match config.")
    earthquakes = _list(payload.get("earthquakes"))
    if len(earthquakes) == settings.vertical_slice.earthquake_count:
        checks.append("Earthquake count matches config.")
    else:
        errors.append(
            f"Earthquake count mismatch: expected {settings.vertical_slice.earthquake_count}, "
            f"found {len(earthquakes)}."
        )
    bounds = _dict(_dict(payload.get("metadata")).get("bounds_used_km"))
    x_min, x_max = _float_bounds(bounds["x"])
    y_min, y_max = _float_bounds(bounds["y"])
    z_min, z_max = _float_bounds(bounds["z"])
    for earthquake in earthquakes:
        hypocenter = _dict(earthquake.get("hypocenter"))
        if not (
            x_min <= float(hypocenter["x_km"]) <= x_max
            and y_min <= float(hypocenter["y_km"]) <= y_max
            and z_min <= float(hypocenter["z_km"]) <= z_max
        ):
            errors.append(f"Earthquake outside configured source bounds: {earthquake}")
            return
    checks.append("All earthquakes respect saved source bounds.")


def _validate_travel_payload(
    payload: dict[str, object],
    settings: BenchmarkSettings,
    checks: list[str],
    errors: list[str],
) -> None:
    expected_count = (
        settings.vertical_slice.station_count * settings.vertical_slice.earthquake_count
    )
    if payload.get("simulation_id") != settings.vertical_slice.simulation_id:
        errors.append("Simulation ID does not match config.")
    observations = _list(payload.get("observations"))
    if len(observations) == expected_count:
        checks.append("Travel-time observation count matches station x earthquake count.")
    else:
        errors.append(
            f"Travel-time count mismatch: expected {expected_count}, found {len(observations)}."
        )
    if all(float(observation["travel_time_s"]) > 0.0 for observation in observations):
        checks.append("All travel times are positive.")
    else:
        errors.append("One or more travel times are not positive.")
    generated_method = str(payload.get("method", ""))
    if generated_method not in settings.simulation.supported_methods:
        errors.append(
            f"Generated simulator method is not configured as supported: {generated_method}"
        )
    elif all(observation["simulation_method"] == generated_method for observation in observations):
        checks.append("All travel-time observations record the generated simulator method.")
    else:
        errors.append("One or more travel-time observations have unexpected simulator metadata.")


def _validate_dataset_rows(
    rows: list[dict[str, str]],
    metadata: dict[str, object],
    settings: BenchmarkSettings,
    travel_method: str,
    checks: list[str],
    errors: list[str],
) -> None:
    expected_count = (
        settings.vertical_slice.station_count * settings.vertical_slice.earthquake_count
    )
    if len(rows) == expected_count:
        checks.append("Dataset row count matches station x earthquake count.")
    else:
        errors.append(f"Dataset row count mismatch: expected {expected_count}, found {len(rows)}.")
    if rows and tuple(rows[0].keys()) == dataset_columns():
        checks.append("Dataset CSV columns match schema version 0.1.0.")
    else:
        errors.append("Dataset CSV columns do not match the configured schema.")
    if int(metadata.get("row_count", -1)) == len(rows):
        checks.append("Dataset metadata row count matches CSV row count.")
    else:
        errors.append("Dataset metadata row count does not match CSV row count.")
    if all(row["dataset_id"] == settings.vertical_slice.dataset_id for row in rows):
        checks.append("Dataset IDs match config.")
    else:
        errors.append("One or more dataset row IDs do not match config.")
    if all(row["simulation_method"] == travel_method for row in rows):
        checks.append("Dataset rows record the generated simulator method.")
    else:
        errors.append("One or more dataset rows have unexpected simulator metadata.")


def _validate_ray_path_sidecar(
    path: Path,
    dataset_rows: list[dict[str, str]],
    metadata: dict[str, object],
    checks: list[str],
    errors: list[str],
) -> None:
    records = _read_jsonl(path)
    dataset_ids = {row["observation_id"] for row in dataset_rows}
    ray_path_ids = {str(record.get("observation_id")) for record in records}
    if ray_path_ids == dataset_ids:
        checks.append("Ray-path sidecar observation IDs match the dataset CSV.")
    else:
        errors.append("Ray-path sidecar observation IDs do not match the dataset CSV.")
    source_files = metadata.get("source_files")
    if isinstance(source_files, dict) and source_files.get("ray_paths"):
        checks.append("Dataset metadata links the ray-path sidecar artifact.")
    else:
        errors.append("Dataset metadata does not link the ray-path sidecar artifact.")
    for record in records:
        if record.get("simulation_method") != PSEUDO_BENDING_3D_METHOD:
            errors.append("Ray-path sidecar contains a non-pseudo-bending method.")
            return
        ray_path = record.get("ray_path")
        if not isinstance(ray_path, list) or len(ray_path) < 2:
            errors.append("Ray-path sidecar contains an invalid or empty ray_path.")
            return
    checks.append("Ray-path sidecar records contain pseudo-bending ray geometries.")


def _validate_sensitivity_sidecar(
    sensitivity_path: Path,
    ray_path_sidecar_path: Path,
    metadata: dict[str, object],
    checks: list[str],
    errors: list[str],
) -> None:
    validation = validate_sensitivity_sidecar_files(sensitivity_path, ray_path_sidecar_path)
    if validation.passed:
        checks.append("Ray-cell sensitivity rows conserve per-observation ray path lengths.")
    else:
        errors.extend(validation.errors)
    source_files = metadata.get("source_files")
    if isinstance(source_files, dict) and source_files.get("ray_cell_sensitivity"):
        checks.append("Dataset metadata links the ray-cell sensitivity sidecar artifact.")
    else:
        errors.append("Dataset metadata does not link the ray-cell sensitivity sidecar artifact.")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError(f"Expected JSON object line in {path}.")
                records.append(payload)
    return records


def _dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Expected a dictionary in generated artifact.")
    return value


def _list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("Expected a list in generated artifact.")
    return [_dict(item) for item in value]


def _float_bounds(value: object) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError("Expected two numeric bounds.")
    return float(value[0]), float(value[1])
