"""Synthetic data generation interfaces."""

from tomobench.generation.earthquakes import (
    SUPPORTED_EARTHQUAKE_MODES,
    generate_earthquake_configuration,
)
from tomobench.generation.stations import (
    SUPPORTED_STATION_MODES,
    generate_station_configuration,
    generate_grid_stations,
    generate_random_stations,
)
from tomobench.generation.velocity_models import SUPPORTED_VELOCITY_SCENARIOS
from tomobench.generation.velocity_grids import (
    SUPPORTED_CARTESIAN_GRID_SCENARIOS,
    build_cartesian_velocity_grid,
)

__all__ = [
    "SUPPORTED_CARTESIAN_GRID_SCENARIOS",
    "SUPPORTED_EARTHQUAKE_MODES",
    "SUPPORTED_STATION_MODES",
    "SUPPORTED_VELOCITY_SCENARIOS",
    "build_cartesian_velocity_grid",
    "generate_earthquake_configuration",
    "generate_station_configuration",
    "generate_grid_stations",
    "generate_random_stations",
]
