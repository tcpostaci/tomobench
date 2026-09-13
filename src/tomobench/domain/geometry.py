"""Geometric primitives used across stations, events, and velocity models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Point3D:
    """Simple Cartesian point expressed in kilometers."""

    x_km: float
    y_km: float
    z_km: float


@dataclass(frozen=True)
class CartesianBounds3D:
    """Axis-aligned 3D study-area bounds expressed in kilometers."""

    x_min_km: float
    x_max_km: float
    y_min_km: float
    y_max_km: float
    z_min_km: float
    z_max_km: float
