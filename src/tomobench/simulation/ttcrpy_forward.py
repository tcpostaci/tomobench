"""Optional ttcrpy adapter for authoritative rectilinear-grid travel times.

The project keeps this adapter separate from the legacy graph and interface-aware
optimizers.  It is deliberately strict about grid shape and endpoint handling:
points are passed to ttcrpy in the project Cartesian-kilometre convention and are
never silently clamped or snapped by this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Sequence

import numpy as np

from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import CartesianCellVelocityField3D, CartesianVelocityGrid3D

TtcrpyMethod = Literal["FSM", "SPM", "DSPM"]


@dataclass(frozen=True)
class TtcrpyGridConfiguration:
    """Numerical controls forwarded to ``ttcrpy.rgrid.Grid3d``."""

    method: TtcrpyMethod = "FSM"
    n_threads: int = 1
    cell_slowness: bool = False
    tt_from_rp: bool = False
    interp_vel: bool = False
    eps: float = 1.0e-5
    maxit: int = 50
    weno: bool = True
    nsnx: int = 5
    nsny: int = 5
    nsnz: int = 5
    n_secondary: int = 2
    n_tertiary: int = 2
    radius_factor_tertiary: float = 3.0
    translate_grid: bool = False

    def __post_init__(self) -> None:
        if self.method not in ("FSM", "SPM", "DSPM"):
            raise ValueError("ttcrpy method must be FSM, SPM, or DSPM.")
        if self.n_threads < 1:
            raise ValueError("ttcrpy n_threads must be positive.")
        if self.maxit < 1:
            raise ValueError("ttcrpy maxit must be positive.")
        if self.eps <= 0.0:
            raise ValueError("ttcrpy eps must be positive.")
        for name in ("nsnx", "nsny", "nsnz", "n_secondary", "n_tertiary"):
            if getattr(self, name) < 0:
                raise ValueError(f"ttcrpy {name} must be non-negative.")
        if self.radius_factor_tertiary <= 0.0:
            raise ValueError("ttcrpy radius_factor_tertiary must be positive.")

    def constructor_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by the ttcrpy 3D constructor."""
        return {
            "n_threads": self.n_threads,
            "cell_slowness": int(self.cell_slowness),
            "method": self.method,
            "tt_from_rp": int(self.tt_from_rp),
            "interp_vel": int(self.interp_vel),
            "eps": self.eps,
            "maxit": self.maxit,
            "weno": int(self.weno),
            "nsnx": self.nsnx,
            "nsny": self.nsny,
            "nsnz": self.nsnz,
            "n_secondary": self.n_secondary,
            "n_tertiary": self.n_tertiary,
            "radius_factor_tertiary": self.radius_factor_tertiary,
            "translate_grid": int(self.translate_grid),
        }


@dataclass(frozen=True)
class TtcrpyRaytraceResult:
    """Travel times and optional paths returned by one batched call."""

    travel_times_s: tuple[float, ...]
    ray_paths: tuple[tuple[Point3D, ...], ...] | None
    path_lengths_km: tuple[float, ...] | None
    source_endpoint_errors_km: tuple[float, ...] | None
    receiver_endpoint_errors_km: tuple[float, ...] | None
    runtime_s: float


def require_ttcrpy() -> Any:
    """Import ttcrpy or raise an actionable optional-dependency error."""
    try:
        from ttcrpy import rgrid
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "The authoritative ttcrpy adapter requires the optional solver environment. "
            "Install the pinned authoritative-forward dependency group first."
        ) from exc
    return rgrid


class TtcrpyRectilinearForwardSolver:
    """Strict adapter around ``ttcrpy.rgrid.Grid3d``."""

    def __init__(
        self,
        x_coordinates_km: Sequence[float],
        y_coordinates_km: Sequence[float],
        z_coordinates_km: Sequence[float],
        p_velocity_km_per_s: np.ndarray,
        configuration: TtcrpyGridConfiguration | None = None,
        *,
        values_kind: Literal["velocity", "slowness"] = "velocity",
    ) -> None:
        self.configuration = configuration or TtcrpyGridConfiguration()
        if values_kind not in ("velocity", "slowness"):
            raise ValueError("values_kind must be 'velocity' or 'slowness'.")
        self.values_kind = values_kind
        self.x_coordinates_km = _validate_axis(x_coordinates_km, "x")
        self.y_coordinates_km = _validate_axis(y_coordinates_km, "y")
        self.z_coordinates_km = _validate_axis(z_coordinates_km, "z")
        self.field_values = _validate_field_shape(
            p_velocity_km_per_s,
            (len(self.x_coordinates_km), len(self.y_coordinates_km), len(self.z_coordinates_km)),
            self.configuration.cell_slowness,
            self.values_kind,
        )
        rgrid = require_ttcrpy()
        self._grid = rgrid.Grid3d(
            np.asarray(self.x_coordinates_km, dtype=np.float64),
            np.asarray(self.y_coordinates_km, dtype=np.float64),
            np.asarray(self.z_coordinates_km, dtype=np.float64),
            dtype=np.float64,
            **self.configuration.constructor_kwargs(),
        )
        if self.values_kind == "velocity":
            self._grid.set_velocity(np.asarray(self.field_values, dtype=np.float64))
        else:
            self._grid.set_slowness(np.asarray(self.field_values, dtype=np.float64))

    @classmethod
    def from_cartesian_grid(
        cls,
        grid: CartesianVelocityGrid3D,
        configuration: TtcrpyGridConfiguration | None = None,
    ) -> "TtcrpyRectilinearForwardSolver":
        """Construct an adapter from the repository's x-fastest grid schema."""
        shape = (
            len(grid.x_coordinates_km),
            len(grid.y_coordinates_km),
            len(grid.z_coordinates_km),
        )
        flat = np.asarray(grid.p_velocity_km_per_s, dtype=np.float64)
        if flat.size != np.prod(shape):
            raise ValueError("Cartesian velocity grid flattened size does not match coordinates.")
        values = flat.reshape((shape[2], shape[1], shape[0])).transpose(2, 1, 0)
        return cls(
            grid.x_coordinates_km,
            grid.y_coordinates_km,
            grid.z_coordinates_km,
            values,
            configuration,
        )

    @classmethod
    def from_cell_velocity_field(
        cls,
        field: CartesianCellVelocityField3D,
        configuration: TtcrpyGridConfiguration | None = None,
    ) -> "TtcrpyRectilinearForwardSolver":
        """Construct a cell-slowness solver from direct cell-center velocity samples."""
        selected = configuration or TtcrpyGridConfiguration(cell_slowness=True)
        if not selected.cell_slowness:
            raise ValueError("Cell velocity fields require configuration cell_slowness=True.")
        shape = (
            len(field.x_coordinates_km) - 1,
            len(field.y_coordinates_km) - 1,
            len(field.z_coordinates_km) - 1,
        )
        flat_velocity = np.asarray(field.p_velocity_km_per_s, dtype=np.float64)
        if flat_velocity.size != np.prod(shape):
            raise ValueError("Cell velocity field flattened size does not match boundary axes.")
        velocity = flat_velocity.reshape((shape[2], shape[1], shape[0])).transpose(2, 1, 0)
        slowness = 1.0 / velocity
        return cls(
            field.x_coordinates_km,
            field.y_coordinates_km,
            field.z_coordinates_km,
            slowness,
            selected,
            values_kind="slowness",
        )

    @property
    def grid(self) -> Any:
        """Return the underlying ttcrpy grid for advanced diagnostics."""
        return self._grid

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Return the velocity-array shape expected by the adapter."""
        return tuple(int(value) for value in self.field_values.shape)

    @property
    def value_location(self) -> Literal["node", "cell"]:
        """Return whether the configured physical values are node- or cell-attributed."""
        return "cell" if self.configuration.cell_slowness else "node"

    @property
    def domain_bounds_km(self) -> dict[str, tuple[float, float]]:
        """Return the closed Cartesian domain represented by the grid."""
        return {
            "x": (self.x_coordinates_km[0], self.x_coordinates_km[-1]),
            "y": (self.y_coordinates_km[0], self.y_coordinates_km[-1]),
            "z": (self.z_coordinates_km[0], self.z_coordinates_km[-1]),
        }

    def raytrace(
        self,
        sources: Sequence[Point3D] | np.ndarray,
        receivers: Sequence[Point3D] | np.ndarray,
        *,
        return_rays: bool = False,
        aggregate_src: bool = False,
    ) -> TtcrpyRaytraceResult:
        """Compute travel times without endpoint snapping or silent clamping.

        A source and receiver row correspond to one pair unless ``aggregate_src``
        is true, in which case a single source may be paired with many receivers.
        The returned path orientation is normalized to source-to-receiver even
        though ttcrpy methods may return opposite orientations.
        """
        source_array = _points_array(sources, "sources")
        receiver_array = _points_array(receivers, "receivers")
        if aggregate_src:
            if len(source_array) != 1:
                raise ValueError("aggregate_src requires exactly one source row.")
            if len(receiver_array) < 1:
                raise ValueError("At least one receiver is required.")
        elif len(source_array) != len(receiver_array):
            raise ValueError("sources and receivers must have the same row count.")
        _validate_points_in_domain(source_array, self.domain_bounds_km, "sources")
        _validate_points_in_domain(receiver_array, self.domain_bounds_km, "receivers")

        start = perf_counter()
        raw = self._grid.raytrace(
            source_array,
            receiver_array,
            aggregate_src=aggregate_src,
            return_rays=return_rays,
        )
        runtime_s = perf_counter() - start
        if return_rays:
            if not isinstance(raw, tuple) or len(raw) < 2:
                raise RuntimeError("ttcrpy did not return ray paths when return_rays=True.")
            travel_times = np.asarray(raw[0], dtype=np.float64).reshape(-1)
            raw_rays = raw[1]
        else:
            travel_times = np.asarray(raw, dtype=np.float64).reshape(-1)
            raw_rays = None

        expected_count = len(receiver_array) if aggregate_src else len(source_array)
        if len(travel_times) != expected_count:
            raise RuntimeError(
                f"ttcrpy returned {len(travel_times)} travel times for {expected_count} pairs."
            )
        if not np.all(np.isfinite(travel_times)):
            raise RuntimeError("ttcrpy returned a non-finite travel time.")

        if raw_rays is None:
            return TtcrpyRaytraceResult(
                travel_times_s=tuple(float(value) for value in travel_times),
                ray_paths=None,
                path_lengths_km=None,
                source_endpoint_errors_km=None,
                receiver_endpoint_errors_km=None,
                runtime_s=runtime_s,
            )

        source_rows = np.repeat(source_array, len(receiver_array), axis=0) if aggregate_src else source_array
        ray_paths: list[tuple[Point3D, ...]] = []
        path_lengths: list[float] = []
        source_errors: list[float] = []
        receiver_errors: list[float] = []
        for source_row, receiver_row, raw_path in zip(source_rows, receiver_array, raw_rays):
            path = _normalize_ray_path(raw_path, source_row, receiver_row)
            ray_paths.append(tuple(Point3D(*map(float, point)) for point in path))
            path_lengths.append(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()))
            source_errors.append(float(np.linalg.norm(path[0] - source_row)))
            receiver_errors.append(float(np.linalg.norm(path[-1] - receiver_row)))
        return TtcrpyRaytraceResult(
            travel_times_s=tuple(float(value) for value in travel_times),
            ray_paths=tuple(ray_paths),
            path_lengths_km=tuple(path_lengths),
            source_endpoint_errors_km=tuple(source_errors),
            receiver_endpoint_errors_km=tuple(receiver_errors),
            runtime_s=runtime_s,
        )


def _validate_axis(values: Sequence[float], name: str) -> tuple[float, ...]:
    axis = tuple(float(value) for value in values)
    if len(axis) < 2:
        raise ValueError(f"{name} coordinate axis must have at least two nodes.")
    if not np.all(np.isfinite(axis)) or any(right <= left for left, right in zip(axis, axis[1:])):
        raise ValueError(f"{name} coordinate axis must be finite and strictly increasing.")
    return axis


def _validate_field_shape(
    velocity: np.ndarray,
    node_shape: tuple[int, int, int],
    cell_slowness: bool,
    values_kind: Literal["velocity", "slowness"],
) -> np.ndarray:
    values = np.asarray(velocity, dtype=np.float64)
    expected = tuple(size - 1 for size in node_shape) if cell_slowness else node_shape
    if values.shape != expected:
        label = "cell" if cell_slowness else "node"
        value_label = "slowness" if values_kind == "slowness" else "velocity"
        raise ValueError(
            f"{label}-centered {value_label} must have shape {expected}, got {values.shape}."
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        value_label = "slowness" if values_kind == "slowness" else "velocity"
        raise ValueError(f"{value_label.capitalize()} values must be finite and strictly positive.")
    return np.ascontiguousarray(values)


def _points_array(points: Sequence[Point3D] | np.ndarray, name: str) -> np.ndarray:
    if isinstance(points, np.ndarray):
        array = np.asarray(points, dtype=np.float64)
    else:
        array = np.asarray(
            [[point.x_km, point.y_km, point.z_km] for point in points],
            dtype=np.float64,
        )
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty array with shape (n, 3).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite coordinates.")
    return np.ascontiguousarray(array)


def _validate_points_in_domain(
    points: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    name: str,
) -> None:
    lower = np.array([bounds[axis][0] for axis in ("x", "y", "z")])
    upper = np.array([bounds[axis][1] for axis in ("x", "y", "z")])
    if np.any(points < lower) or np.any(points > upper):
        raise ValueError(f"{name} contains a point outside the rectilinear-grid domain.")


def _normalize_ray_path(
    raw_path: Any,
    source: np.ndarray,
    receiver: np.ndarray,
) -> np.ndarray:
    path = np.asarray(raw_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 3:
        raise RuntimeError("ttcrpy returned an invalid ray-path array.")
    forward_cost = np.linalg.norm(path[0] - source) + np.linalg.norm(path[-1] - receiver)
    reverse_cost = np.linalg.norm(path[-1] - source) + np.linalg.norm(path[0] - receiver)
    if reverse_cost < forward_cost:
        path = path[::-1].copy()
    return path


def velocity_array_from_cartesian_grid(grid: CartesianVelocityGrid3D) -> np.ndarray:
    """Convert the repository x-fastest flat grid to ``(nx, ny, nz)`` order."""
    shape = (
        len(grid.x_coordinates_km),
        len(grid.y_coordinates_km),
        len(grid.z_coordinates_km),
    )
    flat = np.asarray(grid.p_velocity_km_per_s, dtype=np.float64)
    if flat.size != np.prod(shape):
        raise ValueError("Cartesian grid velocity count does not match coordinate axes.")
    return flat.reshape((shape[2], shape[1], shape[0])).transpose(2, 1, 0)
