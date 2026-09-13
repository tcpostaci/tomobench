"""Observation-level case feature assembly for supervised inversion baselines."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OBSERVATION_FEATURE_FIELDS = (
    "source_x_km",
    "source_y_km",
    "source_z_km",
    "receiver_x_km",
    "receiver_y_km",
    "receiver_z_km",
    "path_length_km",
    "travel_time_s",
)


@dataclass(frozen=True)
class ObservationCaseFeatureSample:
    """One fixed-length case feature vector assembled from ordered observations."""

    pairing_item_id: str
    case_id: str
    scenario: str
    observation_count: int
    feature_row: dict[str, float | str]
    feature_vector: tuple[float, ...]


@dataclass(frozen=True)
class ObservationCaseFeatureMatrix:
    """A case-level feature matrix assembled from an observation pairing manifest."""

    fieldnames: tuple[str, ...]
    observation_feature_fields: tuple[str, ...]
    observation_count: int
    samples: tuple[ObservationCaseFeatureSample, ...]


def assemble_observation_case_feature_matrix(
    manifest_path: Path,
    repo_root: Path,
    observation_fields: tuple[str, ...] = DEFAULT_OBSERVATION_FEATURE_FIELDS,
) -> ObservationCaseFeatureMatrix:
    """Flatten ordered source-receiver observations into one fixed-length vector per case."""
    manifest_payload = _read_json(manifest_path)
    items = _dict_list(manifest_payload["items"])
    samples: list[ObservationCaseFeatureSample] = []
    expected_observation_count: int | None = None
    fieldnames: tuple[str, ...] | None = None
    for item in items:
        dataset_csv_path = _resolve_repo_path(Path(str(item["input_dataset_csv_path"])), repo_root)
        rows = _ordered_observation_rows(dataset_csv_path)
        observation_count = len(rows)
        if expected_observation_count is None:
            expected_observation_count = observation_count
            fieldnames = _feature_fieldnames(observation_fields, observation_count)
        elif observation_count != expected_observation_count:
            raise ValueError(
                "Observation pairing manifest does not define a fixed-length feature space because "
                "cases contain different numbers of source-receiver observations."
            )
        feature_row = {
            "pairing_item_id": str(item["pairing_item_id"]),
            "case_id": str(item.get("case_id", item["pairing_item_id"])),
            "scenario": str(item["scenario"]),
        }
        numeric_feature_values: list[float] = []
        for observation_index, row in enumerate(rows):
            for field in observation_fields:
                column_name = _observation_feature_column_name(observation_index, field)
                value = float(row[field])
                feature_row[column_name] = value
                numeric_feature_values.append(value)
        samples.append(
            ObservationCaseFeatureSample(
                pairing_item_id=str(item["pairing_item_id"]),
                case_id=str(item.get("case_id", item["pairing_item_id"])),
                scenario=str(item["scenario"]),
                observation_count=observation_count,
                feature_row=feature_row,
                feature_vector=tuple(numeric_feature_values),
            )
        )
    if expected_observation_count is None or fieldnames is None:
        raise ValueError(f"Observation pairing manifest contains no items: {manifest_path}")
    return ObservationCaseFeatureMatrix(
        fieldnames=fieldnames,
        observation_feature_fields=observation_fields,
        observation_count=expected_observation_count,
        samples=tuple(samples),
    )


def write_observation_case_feature_matrix_csv(
    matrix: ObservationCaseFeatureMatrix,
    path: Path,
) -> Path:
    """Persist the case-level observation feature matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=matrix.fieldnames)
        writer.writeheader()
        for sample in matrix.samples:
            writer.writerow(sample.feature_row)
    return path


def _ordered_observation_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Observation dataset is empty: {path}")
    return sorted(
        rows,
        key=lambda row: (
            row["earthquake_id"],
            row["station_id"],
            row["observation_id"],
        ),
    )


def _feature_fieldnames(
    observation_fields: tuple[str, ...],
    observation_count: int,
) -> tuple[str, ...]:
    names: list[str] = ["pairing_item_id", "case_id", "scenario"]
    for observation_index in range(observation_count):
        for field in observation_fields:
            names.append(_observation_feature_column_name(observation_index, field))
    return tuple(names)


def _observation_feature_column_name(observation_index: int, field: str) -> str:
    return f"obs_{observation_index:04d}_{field}"


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Expected a JSON object.")
    return dict(payload)


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("Expected a list of JSON objects.")
    return [dict(item) for item in value]


def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path
