"""Publication-ready visualization of the frozen ML target-grid structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch

from tomobench.config.settings import BenchmarkSettings

SURFACE_FRONT_COLOR = "#dbeafe"
SURFACE_SIDE_COLOR = "#bfdbfe"
SURFACE_TOP_COLOR = "#e0f2fe"
SURFACE_EDGE_COLOR = "#334155"
NODE_COLOR = "#0f172a"
SLICE_COLOR = "#d97706"
GRID_LINE_COLOR = "#cbd5e1"
CARD_BACKGROUND = "#f8fafc"
CARD_BORDER = "#cbd5e1"
TEXT_MUTED = "#475569"


@dataclass(frozen=True)
class TargetGridGeometry:
    """Derived geometry for the frozen ML target container."""

    x_coordinates_km: tuple[float, ...]
    y_coordinates_km: tuple[float, ...]
    z_coordinates_km: tuple[float, ...]
    x_spacing_km: float
    y_spacing_km: float
    z_spacing_km: float
    spacing_label: str
    flattening_order: str
    value_location: str
    value_semantics: str

    @property
    def nx(self) -> int:
        return len(self.x_coordinates_km)

    @property
    def ny(self) -> int:
        return len(self.y_coordinates_km)

    @property
    def nz(self) -> int:
        return len(self.z_coordinates_km)

    @property
    def target_vector_length(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def x_range_km(self) -> tuple[float, float]:
        return self.x_coordinates_km[0], self.x_coordinates_km[-1]

    @property
    def y_range_km(self) -> tuple[float, float]:
        return self.y_coordinates_km[0], self.y_coordinates_km[-1]

    @property
    def z_range_km(self) -> tuple[float, float]:
        return self.z_coordinates_km[0], self.z_coordinates_km[-1]

    @property
    def mid_y_km(self) -> float:
        return self.y_coordinates_km[len(self.y_coordinates_km) // 2]

    @property
    def mid_z_km(self) -> float:
        return self.z_coordinates_km[len(self.z_coordinates_km) // 2]


def build_target_grid_geometry(settings: BenchmarkSettings) -> TargetGridGeometry:
    """Derive the frozen target-grid geometry from the central configuration."""
    return TargetGridGeometry(
        x_coordinates_km=_regular_axis_coordinates(
            settings.study_area.x_range_km,
            settings.eikonal_solver.x_grid_spacing_km,
            "x",
        ),
        y_coordinates_km=_regular_axis_coordinates(
            settings.study_area.y_range_km,
            settings.eikonal_solver.y_grid_spacing_km,
            "y",
        ),
        z_coordinates_km=_regular_axis_coordinates(
            settings.study_area.z_range_km,
            settings.eikonal_solver.z_grid_spacing_km,
            "z",
        ),
        x_spacing_km=settings.eikonal_solver.x_grid_spacing_km,
        y_spacing_km=settings.eikonal_solver.y_grid_spacing_km,
        z_spacing_km=settings.eikonal_solver.z_grid_spacing_km,
        spacing_label=settings.eikonal_solver.grid_spacing_label,
        flattening_order=settings.machine_learning.target_grid.flattening_order,
        value_location=settings.machine_learning.target_grid.value_location,
        value_semantics=settings.machine_learning.target_grid.value_semantics,
    )


def save_target_grid_structure_figure(settings: BenchmarkSettings, path: Path) -> Path:
    """Render the frozen target-grid structure as PNG, SVG, and PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry = build_target_grid_geometry(settings)
    figure = _build_target_grid_structure_figure(geometry)
    try:
        for suffix in (".png", ".svg", ".pdf"):
            output_path = path.with_suffix(suffix)
            save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
            if suffix == ".png":
                save_kwargs["dpi"] = 320
            figure.savefig(output_path, **save_kwargs)
    finally:
        plt.close(figure)
    return path.with_suffix(".png")


def _build_target_grid_structure_figure(geometry: TargetGridGeometry):
    figure = plt.figure(figsize=(16.0, 10.4), constrained_layout=False)
    grid_spec = figure.add_gridspec(
        3,
        2,
        width_ratios=[1.45, 1.0],
        height_ratios=[1.0, 0.92, 0.62],
        wspace=0.16,
        hspace=0.34,
    )

    overview_axis = figure.add_subplot(grid_spec[0:2, 0], projection="3d")
    plan_axis = figure.add_subplot(grid_spec[0, 1])
    section_axis = figure.add_subplot(grid_spec[1, 1])
    summary_axis = figure.add_subplot(grid_spec[2, :])

    _draw_3d_overview_panel(overview_axis, geometry)
    _draw_xy_slice_panel(plan_axis, geometry)
    _draw_xz_slice_panel(section_axis, geometry)
    _draw_summary_band(summary_axis, geometry)

    figure.suptitle(
        "Frozen ML Target Grid: Node-Centered 3D Cartesian Velocity Representation",
        fontsize=20,
        fontweight="bold",
        x=0.49,
        y=0.975,
    )
    figure.text(
        0.06,
        0.935,
        f"Configured spacing ({geometry.spacing_label}) over the 0-100 km x 0-100 km x 0-30 km "
        f"study area produces a fixed {geometry.nx} x {geometry.ny} x {geometry.nz} inversion "
        f"container with {geometry.target_vector_length:,} node values.",
        fontsize=11,
        color=TEXT_MUTED,
    )
    return figure


def _draw_3d_overview_panel(axis, geometry: TargetGridGeometry) -> None:
    axis.set_title("A. 3D Target Container", loc="left", pad=12, fontsize=14, fontweight="bold")

    x_min, x_max = geometry.x_range_km
    y_min, y_max = geometry.y_range_km
    z_min, z_max = geometry.z_range_km
    y_mid = geometry.mid_y_km
    z_mid = geometry.mid_z_km

    _draw_surface(
        axis, x_min, x_max, y_min, y_max, z_value=z_min, color=SURFACE_TOP_COLOR, alpha=0.42
    )
    _draw_surface(
        axis,
        x_min,
        x_max,
        y_min,
        y_min,
        z_min=z_min,
        z_max=z_max,
        color=SURFACE_FRONT_COLOR,
        alpha=0.62,
    )
    _draw_surface(
        axis,
        x_max,
        x_max,
        y_min,
        y_max,
        z_min=z_min,
        z_max=z_max,
        color=SURFACE_SIDE_COLOR,
        alpha=0.54,
    )

    _draw_surface(axis, x_min, x_max, y_min, y_max, z_value=z_mid, color=SLICE_COLOR, alpha=0.10)
    _draw_surface(
        axis, x_min, x_max, y_mid, y_mid, z_min=z_min, z_max=z_max, color=SLICE_COLOR, alpha=0.10
    )

    _draw_surface_nodes(axis, geometry)
    _draw_cube_edges(axis, geometry)

    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_zlim(z_max, z_min)
    axis.set_box_aspect((100.0, 100.0, 30.0))
    axis.view_init(elev=24, azim=-58)
    axis.set_proj_type("persp")
    axis.set_xlabel("x (km)", labelpad=10)
    axis.set_ylabel("y (km)", labelpad=10)
    axis.set_zlabel("depth z (km)", labelpad=8)
    axis.set_xticks((0, 25, 50, 75, 100))
    axis.set_yticks((0, 25, 50, 75, 100))
    axis.set_zticks((0, 10, 20, 30))
    axis.grid(False)
    axis.xaxis.set_pane_color(to_rgba("#ffffff", alpha=0.0))
    axis.yaxis.set_pane_color(to_rgba("#ffffff", alpha=0.0))
    axis.zaxis.set_pane_color(to_rgba("#ffffff", alpha=0.0))
    axis.xaxis.line.set_color("#94a3b8")
    axis.yaxis.line.set_color("#94a3b8")
    axis.zaxis.line.set_color("#94a3b8")

    axis.text2D(
        0.03,
        0.96,
        "Visible faces show the node-centered lattice.\n"
        "Orange planes mark the representative slices shown in panels B and C.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#1e293b",
        bbox={
            "facecolor": "white",
            "edgecolor": CARD_BORDER,
            "boxstyle": "round,pad=0.4",
        },
    )
    axis.text2D(
        0.03,
        0.08,
        "100 km x 100 km x 30 km domain\n41 x 41 x 13 node-centered target grid",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=TEXT_MUTED,
    )


def _draw_xy_slice_panel(axis, geometry: TargetGridGeometry) -> None:
    axis.set_title(
        f"B. Mid-Depth x-y Slice (z = {geometry.mid_z_km:.1f} km)",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    axis.set_facecolor("#fbfdff")
    _draw_regular_lattice(
        axis,
        geometry.x_coordinates_km,
        geometry.y_coordinates_km,
        "x (km)",
        "y (km)",
    )
    axis.set_aspect("equal")
    _add_panel_badge(axis, f"{geometry.nx} x {geometry.ny} = {geometry.nx * geometry.ny:,} nodes")


def _draw_xz_slice_panel(axis, geometry: TargetGridGeometry) -> None:
    axis.set_title(
        f"C. Central x-z Slice (y = {geometry.mid_y_km:.1f} km)",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    axis.set_facecolor("#fbfdff")
    _draw_regular_lattice(
        axis,
        geometry.x_coordinates_km,
        geometry.z_coordinates_km,
        "x (km)",
        "depth z (km)",
    )
    axis.invert_yaxis()
    _add_panel_badge(axis, f"{geometry.nx} x {geometry.nz} = {geometry.nx * geometry.nz:,} nodes")


def _draw_summary_band(axis, geometry: TargetGridGeometry) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    x_min, x_max = geometry.x_range_km
    y_min, y_max = geometry.y_range_km
    z_min, z_max = geometry.z_range_km

    _draw_summary_card(
        axis,
        x0=0.02,
        width=0.29,
        title="Grid Contract",
        lines=(
            "Node values: absolute P-wave velocity",
            f"Value location: {geometry.value_location.replace('_', '-')}",
            f"Grid spacing: {geometry.spacing_label}",
            "Boundary nodes retained on every axis",
        ),
    )
    _draw_summary_card(
        axis,
        x0=0.355,
        width=0.29,
        title="Dimension Derivation",
        lines=(
            f"nx = ({x_max:.0f} - {x_min:.0f}) / {geometry.x_spacing_km:.1f} + 1 = {geometry.nx}",
            f"ny = ({y_max:.0f} - {y_min:.0f}) / {geometry.y_spacing_km:.1f} + 1 = {geometry.ny}",
            f"nz = ({z_max:.0f} - {z_min:.0f}) / {geometry.z_spacing_km:.1f} + 1 = {geometry.nz}",
        ),
        monospace=True,
    )
    _draw_summary_card(
        axis,
        x0=0.69,
        width=0.29,
        title="Target Vector",
        lines=(
            f"Target shape: {geometry.nx} x {geometry.ny} x {geometry.nz}",
            f"Vector length: {geometry.target_vector_length:,}",
            f"Flattening order: {_pretty_flattening_order(geometry.flattening_order)}",
        ),
    )


def _draw_summary_card(
    axis,
    *,
    x0: float,
    width: float,
    title: str,
    lines: tuple[str, ...],
    monospace: bool = False,
) -> None:
    card = FancyBboxPatch(
        (x0, 0.08),
        width,
        0.82,
        boxstyle="round,pad=0.012,rounding_size=12",
        linewidth=1.1,
        edgecolor=CARD_BORDER,
        facecolor=CARD_BACKGROUND,
        transform=axis.transAxes,
    )
    axis.add_patch(card)
    axis.text(
        x0 + 0.02,
        0.82,
        title,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
    )
    axis.plot(
        [x0 + 0.02, x0 + width - 0.02],
        [0.74, 0.74],
        transform=axis.transAxes,
        color="#e2e8f0",
        linewidth=1.0,
    )
    axis.text(
        x0 + 0.02,
        0.68,
        "\n".join(lines),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.7,
        family="DejaVu Sans Mono" if monospace else "DejaVu Sans",
        color="#1e293b",
        linespacing=1.45,
    )


def _draw_regular_lattice(
    axis,
    x_coordinates_km: tuple[float, ...],
    y_coordinates_km: tuple[float, ...],
    x_label: str,
    y_label: str,
) -> None:
    x_values = np.asarray(x_coordinates_km, dtype=float)
    y_values = np.asarray(y_coordinates_km, dtype=float)

    for x_value in x_values:
        axis.plot(
            [x_value, x_value],
            [y_values[0], y_values[-1]],
            color=GRID_LINE_COLOR,
            linewidth=0.55,
            alpha=0.9,
            zorder=1,
        )
    for y_value in y_values:
        axis.plot(
            [x_values[0], x_values[-1]],
            [y_value, y_value],
            color=GRID_LINE_COLOR,
            linewidth=0.55,
            alpha=0.9,
            zorder=1,
        )

    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    axis.scatter(
        xx.ravel(),
        yy.ravel(),
        s=12,
        color=NODE_COLOR,
        alpha=0.88,
        linewidths=0.0,
        zorder=2,
    )

    axis.set_xlim(x_values[0], x_values[-1])
    axis.set_ylim(y_values[0], y_values[-1])
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.xaxis.labelpad = 2
    axis.yaxis.labelpad = 2
    axis.set_xticks((0, 20, 40, 60, 80, 100))
    if y_values[-1] <= 35.0:
        axis.set_yticks((0, 5, 10, 15, 20, 25, 30))
    else:
        axis.set_yticks((0, 20, 40, 60, 80, 100))
    for spine in axis.spines.values():
        spine.set_color("#94a3b8")
        spine.set_linewidth(1.0)


def _add_panel_badge(axis, label: str) -> None:
    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#1e293b",
        bbox={
            "facecolor": "white",
            "edgecolor": CARD_BORDER,
            "boxstyle": "round,pad=0.28",
        },
    )


def _draw_surface(
    axis,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    z_value: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    color: str,
    alpha: float,
) -> None:
    if z_value is not None:
        x_grid, y_grid = np.meshgrid((x_min, x_max), (y_min, y_max), indexing="xy")
        z_grid = np.full_like(x_grid, fill_value=z_value, dtype=float)
    else:
        if z_min is None or z_max is None:
            raise ValueError("z_min and z_max are required for vertical surfaces.")
        if np.isclose(x_min, x_max):
            y_grid, z_grid = np.meshgrid((y_min, y_max), (z_min, z_max), indexing="xy")
            x_grid = np.full_like(y_grid, fill_value=x_min, dtype=float)
        else:
            x_grid, z_grid = np.meshgrid((x_min, x_max), (z_min, z_max), indexing="xy")
            y_grid = np.full_like(x_grid, fill_value=y_min, dtype=float)

    axis.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color=color,
        alpha=alpha,
        linewidth=0.0,
        antialiased=True,
        shade=False,
    )


def _draw_surface_nodes(axis, geometry: TargetGridGeometry) -> None:
    x_values = np.asarray(geometry.x_coordinates_km, dtype=float)
    y_values = np.asarray(geometry.y_coordinates_km, dtype=float)
    z_values = np.asarray(geometry.z_coordinates_km, dtype=float)
    x_max = float(x_values[-1])
    y_min = float(y_values[0])
    z_min = float(z_values[0])

    xx_top, yy_top = np.meshgrid(x_values, y_values, indexing="xy")
    axis.scatter(
        xx_top.ravel(),
        yy_top.ravel(),
        np.full(xx_top.size, z_min),
        s=5,
        c=NODE_COLOR,
        alpha=0.55,
        depthshade=False,
    )

    xx_front, zz_front = np.meshgrid(x_values, z_values, indexing="xy")
    axis.scatter(
        xx_front.ravel(),
        np.full(xx_front.size, y_min),
        zz_front.ravel(),
        s=8,
        c=NODE_COLOR,
        alpha=0.8,
        depthshade=False,
    )

    yy_side, zz_side = np.meshgrid(y_values, z_values, indexing="xy")
    axis.scatter(
        np.full(yy_side.size, x_max),
        yy_side.ravel(),
        zz_side.ravel(),
        s=8,
        c=NODE_COLOR,
        alpha=0.72,
        depthshade=False,
    )


def _draw_cube_edges(axis, geometry: TargetGridGeometry) -> None:
    x_min, x_max = geometry.x_range_km
    y_min, y_max = geometry.y_range_km
    z_min, z_max = geometry.z_range_km
    corners = [
        (x_min, y_min, z_min),
        (x_max, y_min, z_min),
        (x_max, y_max, z_min),
        (x_min, y_max, z_min),
        (x_min, y_min, z_max),
        (x_max, y_min, z_max),
        (x_max, y_max, z_max),
        (x_min, y_max, z_max),
    ]
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for start_index, end_index in edges:
        start = corners[start_index]
        end = corners[end_index]
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=SURFACE_EDGE_COLOR,
            linewidth=1.3,
            alpha=0.9,
        )


def _regular_axis_coordinates(
    bounds_km: tuple[float, float],
    spacing_km: float,
    axis_label: str,
) -> tuple[float, ...]:
    low_km, high_km = bounds_km
    span_km = high_km - low_km
    step_count = span_km / spacing_km
    rounded_step_count = round(step_count)
    if not np.isclose(step_count, rounded_step_count, atol=1.0e-9):
        raise ValueError(
            f"{axis_label}-axis span {span_km} km is not divisible by spacing {spacing_km} km."
        )
    coordinates = np.linspace(low_km, high_km, int(rounded_step_count) + 1)
    return tuple(float(value) for value in coordinates)


def _pretty_flattening_order(flattening_order: str) -> str:
    if flattening_order == "x_fastest_index_ix_plus_nx_times_iy_plus_ny_times_iz":
        return "ix + nx * (iy + ny * iz)"
    return flattening_order
