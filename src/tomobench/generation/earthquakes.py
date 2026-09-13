"""Synthetic earthquake hypocenter generation."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from tomobench.config.settings import BenchmarkSettings
from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import Earthquake, EarthquakeConfiguration, dataclass_to_dict

SUPPORTED_EARTHQUAKE_MODES: Final[tuple[str, ...]] = (
    "random",
    "clustered",
    "fault_based",
)


def generate_earthquake_configuration(settings: BenchmarkSettings) -> EarthquakeConfiguration:
    """Dispatch to the configured earthquake generator."""
    if settings.earthquake_generation.default_mode == "random":
        return generate_random_earthquakes(settings)
    raise ValueError(
        "Unsupported earthquake_generation.default_mode for the current workflow. "
        "Only 'random' is implemented for artifact generation so far."
    )


def generate_random_earthquakes(settings: BenchmarkSettings) -> EarthquakeConfiguration:
    """Generate reproducible random hypocenters inside the configured 3D domain."""
    rng = random.Random(settings.random_seeds.earthquakes)
    margin = settings.study_area.margin_km
    x_min, x_max = settings.study_area.station_x_bounds_km
    y_min, y_max = settings.study_area.station_y_bounds_km
    z_min = max(
        settings.study_area.z_range_km[0] + margin, settings.earthquake_generation.depth_range_km[0]
    )
    z_max = min(
        settings.study_area.z_range_km[1] - margin, settings.earthquake_generation.depth_range_km[1]
    )

    earthquakes = tuple(
        Earthquake(
            earthquake_id=f"EQ{i + 1:04d}",
            hypocenter=Point3D(
                x_km=rng.uniform(x_min, x_max),
                y_km=rng.uniform(y_min, y_max),
                z_km=rng.uniform(z_min, z_max),
            ),
        )
        for i in range(settings.vertical_slice.earthquake_count)
    )

    metadata = {
        "generator_type": "random_earthquakes",
        "seed": settings.random_seeds.earthquakes,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "event_count": len(earthquakes),
        "coordinate_system": settings.study_area.coordinate_system,
        "bounds_used_km": {
            "x": [x_min, x_max],
            "y": [y_min, y_max],
            "z": [z_min, z_max],
        },
        "margin_km": margin,
        "configured_depth_range_km": list(settings.earthquake_generation.depth_range_km),
        "assumptions": [
            "Hypocenters are generated independently from uniform x, y, and z distributions.",
            "Positive z is interpreted as depth below the surface.",
        ],
    }
    return EarthquakeConfiguration(
        configuration_id=settings.vertical_slice.earthquake_configuration_id,
        earthquakes=earthquakes,
        metadata=metadata,
    )


def save_earthquake_configuration(configuration: EarthquakeConfiguration, path: Path) -> Path:
    """Save an earthquake configuration as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclass_to_dict(configuration)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
