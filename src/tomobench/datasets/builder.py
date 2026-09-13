"""Source-receiver dataset assembly for synthetic travel-time observations."""

from __future__ import annotations

from typing import Any

from tomobench.config.settings import BenchmarkSettings
from tomobench.domain.schemas import TravelTimeObservation

DATASET_RECORD_LEVEL = "source_receiver_pair"


def dataset_columns() -> tuple[str, ...]:
    """Return the source-receiver dataset schema for the first vertical slice."""
    return (
        "observation_id",
        "earthquake_id",
        "station_id",
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "receiver_x_km",
        "receiver_y_km",
        "receiver_z_km",
        "travel_time_s",
        "path_length_km",
        "phase",
        "simulation_method",
        "station_configuration_id",
        "earthquake_configuration_id",
        "velocity_model_id",
        "simulation_id",
        "dataset_id",
        "schema_version",
    )


def build_source_receiver_dataset(
    observations: tuple[TravelTimeObservation, ...],
    settings: BenchmarkSettings,
) -> list[dict[str, Any]]:
    """Build one dataset row per earthquake-station travel-time observation."""
    rows: list[dict[str, Any]] = []
    for observation in observations:
        rows.append(
            {
                "observation_id": observation.observation_id,
                "earthquake_id": observation.earthquake_id,
                "station_id": observation.station_id,
                "source_x_km": observation.source.x_km,
                "source_y_km": observation.source.y_km,
                "source_z_km": observation.source.z_km,
                "receiver_x_km": observation.receiver.x_km,
                "receiver_y_km": observation.receiver.y_km,
                "receiver_z_km": observation.receiver.z_km,
                "travel_time_s": observation.travel_time_s,
                "path_length_km": observation.path_length_km,
                "phase": observation.phase,
                "simulation_method": observation.simulation_method,
                "station_configuration_id": observation.station_configuration_id,
                "earthquake_configuration_id": observation.earthquake_configuration_id,
                "velocity_model_id": observation.velocity_model_id,
                "simulation_id": observation.simulation_id,
                "dataset_id": settings.vertical_slice.dataset_id,
                "schema_version": settings.dataset_export.schema_version,
            }
        )
    return rows
