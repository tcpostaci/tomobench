"""Lightweight schemas for traceable generated artifacts."""

from dataclasses import asdict, dataclass
from typing import Any

from tomobench.domain.geometry import Point3D


@dataclass(frozen=True)
class ScenarioReference:
    """References that link derived outputs back to their source configurations."""

    station_configuration_id: str
    earthquake_configuration_id: str
    velocity_model_id: str


@dataclass(frozen=True)
class Station:
    """Synthetic seismic station at a receiver location."""

    station_id: str
    location: Point3D


@dataclass(frozen=True)
class Earthquake:
    """Synthetic earthquake hypocenter."""

    earthquake_id: str
    hypocenter: Point3D


@dataclass(frozen=True)
class VelocityLayer:
    """One depth interval in a 1D layered P-wave velocity model."""

    layer_id: str
    top_depth_km: float
    bottom_depth_km: float
    p_velocity_km_per_s: float


@dataclass(frozen=True)
class StationConfiguration:
    """A saved group of generated stations with provenance metadata."""

    configuration_id: str
    stations: tuple[Station, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EarthquakeConfiguration:
    """A saved group of generated earthquakes with provenance metadata."""

    configuration_id: str
    earthquakes: tuple[Earthquake, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LayeredVelocityModel:
    """A 1D layered P-wave velocity model used by the first vertical slice."""

    model_id: str
    layers: tuple[VelocityLayer, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CartesianVelocityGrid3D:
    """Node-centered 3D Cartesian P-wave velocity grid.

    Velocities are stored at grid nodes, not cells. The flattened velocity tuple
    uses x-fastest order: ``index = ix + nx * (iy + ny * iz)``.
    """

    grid_id: str
    source_velocity_model_id: str
    x_coordinates_km: tuple[float, ...]
    y_coordinates_km: tuple[float, ...]
    z_coordinates_km: tuple[float, ...]
    p_velocity_km_per_s: tuple[float, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CartesianCellVelocityField3D:
    """Cell-centered 3D Cartesian P-wave velocity field.

    The coordinate axes contain cell boundaries/nodes.  Velocity values are
    sampled at cell centers and stored in x-fastest order with shape
    ``(nx - 1, ny - 1, nz - 1)``.  This representation is kept separate from
    :class:`CartesianVelocityGrid3D` so a cell field cannot be mistaken for a
    node-centered ML target.
    """

    field_id: str
    source_velocity_model_id: str
    x_coordinates_km: tuple[float, ...]
    y_coordinates_km: tuple[float, ...]
    z_coordinates_km: tuple[float, ...]
    p_velocity_km_per_s: tuple[float, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TravelTimeObservation:
    """One source-receiver P-wave travel-time observation."""

    observation_id: str
    earthquake_id: str
    station_id: str
    source: Point3D
    receiver: Point3D
    travel_time_s: float
    path_length_km: float
    phase: str
    simulation_method: str
    station_configuration_id: str
    earthquake_configuration_id: str
    velocity_model_id: str
    simulation_id: str


@dataclass(frozen=True)
class TravelTimeRayPath:
    """Traceable ray path sidecar for one source-receiver travel-time observation."""

    observation_id: str
    earthquake_id: str
    station_id: str
    source: Point3D
    receiver: Point3D
    ray_path: tuple[Point3D, ...]
    travel_time_s: float
    path_length_km: float
    phase: str
    simulation_method: str
    simulator_solver_type: str
    station_configuration_id: str
    earthquake_configuration_id: str
    velocity_model_id: str
    simulation_id: str
    case_id: str | None
    converged: bool
    iteration_count: int
    termination_reason: str = "unknown"


def dataclass_to_dict(value: object) -> dict[str, Any]:
    """Convert a nested dataclass object to a JSON-serializable dictionary."""
    return asdict(value)
