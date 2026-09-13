"""Synthetic station configuration generation."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from tomobench.config.settings import BenchmarkSettings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import Station, StationConfiguration, dataclass_to_dict

SUPPORTED_STATION_MODES: Final[tuple[str, ...]] = ("random", "grid")


def generate_station_configuration(settings: BenchmarkSettings) -> StationConfiguration:
    """Dispatch to the configured station generator."""
    if settings.station_generation.default_mode == "random":
        return generate_random_stations(settings)
    if settings.station_generation.default_mode == "grid":
        return generate_grid_stations(
            settings,
            configuration_id=settings.vertical_slice.station_configuration_id,
        )
    raise ValueError(
        "Unsupported station_generation.default_mode. "
        f"Supported modes: {', '.join(SUPPORTED_STATION_MODES)}."
    )


def generate_random_stations(settings: BenchmarkSettings) -> StationConfiguration:
    """Generate a reproducible random surface station configuration."""
    rng = random.Random(settings.random_seeds.stations)
    x_min, x_max = settings.study_area.station_x_bounds_km
    y_min, y_max = settings.study_area.station_y_bounds_km
    z_km = settings.station_generation.station_elevation_km

    stations = tuple(
        Station(
            station_id=f"STA{i + 1:03d}",
            location=Point3D(
                x_km=rng.uniform(x_min, x_max),
                y_km=rng.uniform(y_min, y_max),
                z_km=z_km,
            ),
        )
        for i in range(settings.vertical_slice.station_count)
    )

    metadata = {
        "generator_type": "random_surface_stations",
        "seed": settings.random_seeds.stations,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "station_count": len(stations),
        "coordinate_system": settings.study_area.coordinate_system,
        "bounds_used_km": {
            "x": [x_min, x_max],
            "y": [y_min, y_max],
            "z": [z_km, z_km],
        },
        "margin_km": settings.study_area.margin_km,
        "assumptions": [
            "Stations are generated independently from a uniform lateral distribution.",
            "All stations are placed at the configured surface elevation.",
        ],
    }
    return StationConfiguration(
        configuration_id=settings.vertical_slice.station_configuration_id,
        stations=stations,
        metadata=metadata,
    )


def generate_grid_stations(
    settings: BenchmarkSettings,
    configuration_id: str = "grid_station_configuration",
) -> StationConfiguration:
    """Generate a deterministic grid of surface stations inside configured margins."""
    x_min, x_max = settings.study_area.station_x_bounds_km
    y_min, y_max = settings.study_area.station_y_bounds_km
    z_km = settings.station_generation.station_elevation_km
    rows = settings.station_generation.grid_rows
    cols = settings.station_generation.grid_cols

    x_values = _evenly_spaced(x_min, x_max, cols)
    y_values = _evenly_spaced(y_min, y_max, rows)
    stations = tuple(
        Station(
            station_id=f"GST{index + 1:03d}",
            location=Point3D(x_km=x_km, y_km=y_km, z_km=z_km),
        )
        for index, (y_km, x_km) in enumerate(
            (y_value, x_value) for y_value in y_values for x_value in x_values
        )
    )

    metadata = {
        "generator_type": "grid_surface_stations",
        "seed": None,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "station_count": len(stations),
        "grid_rows": rows,
        "grid_cols": cols,
        "coordinate_system": settings.study_area.coordinate_system,
        "bounds_used_km": {
            "x": [x_min, x_max],
            "y": [y_min, y_max],
            "z": [z_km, z_km],
        },
        "margin_km": settings.study_area.margin_km,
        "assumptions": [
            "Stations are placed at evenly spaced lateral grid intersections.",
            "All stations are placed at the configured surface elevation.",
        ],
    }
    return StationConfiguration(
        configuration_id=configuration_id,
        stations=stations,
        metadata=metadata,
    )


def save_station_configuration(configuration: StationConfiguration, path: Path) -> Path:
    """Save a station configuration as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclass_to_dict(configuration)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _evenly_spaced(start: float, stop: float, count: int) -> tuple[float, ...]:
    if count == 1:
        return ((start + stop) / 2.0,)
    step = (stop - start) / (count - 1)
    return tuple(start + step * index for index in range(count))
