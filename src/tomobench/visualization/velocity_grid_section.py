"""Matplotlib-based 2D geological sections for synthetic velocity grids."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse, Polygon, Rectangle

from tomobench.config.settings import FaultedGridSettings
from tomobench.domain.schemas import CartesianVelocityGrid3D
from tomobench.generation.velocity_grids import (
    fault_dip_direction_sign,
    fault_plane_horizontal_normal,
)

ANOMALY_EDGE_COLOR = "#c2410c"
ANOMALY_FACE_COLOR = "#fb923c"
SLICE_LINE_COLOR = "#0f172a"


def save_velocity_grid_section_figure(grid: CartesianVelocityGrid3D, path: Path) -> Path:
    """Save a front-looking x-z geological section and an SVG sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)

    values = _grid_values_3d(grid)
    x_coordinates = np.asarray(grid.x_coordinates_km, dtype=float)
    y_coordinates = np.asarray(grid.y_coordinates_km, dtype=float)
    z_coordinates = np.asarray(grid.z_coordinates_km, dtype=float)
    y_focus = select_section_y(grid)
    iy = _nearest_index(y_coordinates, y_focus)
    y_value = float(y_coordinates[iy])
    section = values[:, iy, :]

    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = colormaps["viridis"]

    png_figure = plt.figure(figsize=(10.8, 7.4), constrained_layout=False)
    main_axis = png_figure.add_axes([0.08, 0.12, 0.68, 0.76])
    inset_axis = png_figure.add_axes([0.80, 0.58, 0.16, 0.22])

    image = main_axis.imshow(
        section,
        extent=[
            float(x_coordinates[0]),
            float(x_coordinates[-1]),
            float(z_coordinates[-1]),
            float(z_coordinates[0]),
        ],
        origin="upper",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    _style_main_axis(main_axis, grid, y_value)
    _draw_layer_horizons(main_axis, grid)
    _draw_section_overlay(main_axis, grid, y_value)
    _draw_plan_inset(inset_axis, grid, y_value)
    _finalize_section_figure(png_figure, main_axis, image, grid, y_value)

    png_figure.savefig(path, dpi=220, bbox_inches="tight")
    if path.suffix.lower() != ".svg":
        svg_figure = plt.figure(figsize=(9.2, 7.4), constrained_layout=False)
        svg_main_axis = svg_figure.add_axes([0.10, 0.12, 0.74, 0.76])
        svg_image = svg_main_axis.imshow(
            section,
            extent=[
                float(x_coordinates[0]),
                float(x_coordinates[-1]),
                float(z_coordinates[-1]),
                float(z_coordinates[0]),
            ],
            origin="upper",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        _style_main_axis(svg_main_axis, grid, y_value)
        _draw_layer_horizons(svg_main_axis, grid)
        _draw_section_overlay(svg_main_axis, grid, y_value)
        _finalize_section_figure(svg_figure, svg_main_axis, svg_image, grid, y_value)
        svg_figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(svg_figure)
    plt.close(png_figure)
    return path


def _finalize_section_figure(
    figure, main_axis, image, grid: CartesianVelocityGrid3D, y_value: float
) -> None:
    scenario_name = _pretty_scenario_name(str(grid.metadata.get("grid_scenario", "unknown")))
    figure.suptitle(
        f"{scenario_name} front-looking section at y={y_value:.1f} km",
        fontsize=18,
        y=0.96,
    )
    figure.text(
        0.08,
        0.055,
        _scenario_description(grid, y_value),
        fontsize=10.5,
        color="#334155",
    )
    colorbar = figure.colorbar(image, ax=main_axis, fraction=0.036, pad=0.03)
    colorbar.set_label("P-wave velocity (km/s)")


def select_section_y(grid: CartesianVelocityGrid3D) -> float:
    """Return the most informative y coordinate for the front-looking section."""
    y_coordinates = np.asarray(grid.y_coordinates_km, dtype=float)
    default_y = float(y_coordinates[len(y_coordinates) // 2])
    scenario = str(grid.metadata.get("grid_scenario", ""))
    if scenario == "faulted":
        return float(_scenario_mapping(grid, "faulted")["fault_y_km"])
    if scenario in {"block_anomaly", "salt_dome", "dyke_intrusion"}:
        return float(_float_triplet(_scenario_mapping(grid, scenario)["center_km"])[1])
    return default_y


def _style_main_axis(axis, grid: CartesianVelocityGrid3D, y_value: float) -> None:
    axis.set_facecolor("#f8fafc")
    axis.set_xlabel("x (km)")
    axis.set_ylabel("depth z (km)")
    axis.set_xlim(float(grid.x_coordinates_km[0]), float(grid.x_coordinates_km[-1]))
    axis.set_ylim(float(grid.z_coordinates_km[-1]), float(grid.z_coordinates_km[0]))
    axis.grid(color="#cbd5e1", linestyle="--", linewidth=0.7, alpha=0.65)
    axis.text(
        0.015,
        0.98,
        f"Section plane: y={y_value:.1f} km",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#0f172a",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.25"},
    )


def _draw_layer_horizons(axis, grid: CartesianVelocityGrid3D) -> None:
    layered = _scenario_mapping(grid, "layered_model")
    for depth_km in layered.get("depth_boundaries_km", [])[1:-1]:
        axis.axhline(
            float(depth_km),
            color="white",
            linewidth=1.2,
            alpha=0.9,
        )
        axis.axhline(
            float(depth_km),
            color="#64748b",
            linewidth=0.7,
            alpha=0.65,
            linestyle="--",
        )


def _draw_section_overlay(axis, grid: CartesianVelocityGrid3D, y_value: float) -> None:
    scenario = str(grid.metadata.get("grid_scenario", ""))
    if scenario == "block_anomaly":
        block = _scenario_mapping(grid, "block_anomaly")
        center_x, center_y, center_z = _float_triplet(block["center_km"])
        size_x, size_y, size_z = _float_triplet(block["size_km"])
        if abs(y_value - center_y) <= size_y / 2.0:
            axis.add_patch(
                Rectangle(
                    (center_x - size_x / 2.0, center_z - size_z / 2.0),
                    size_x,
                    size_z,
                    linewidth=1.8,
                    edgecolor=ANOMALY_EDGE_COLOR,
                    facecolor="none",
                )
            )
        return
    if scenario == "faulted":
        faulted = _fault_settings_from_metadata(_scenario_mapping(grid, "faulted"))
        curve = _fault_section_curve(
            faulted,
            y_value,
            (float(grid.x_coordinates_km[0]), float(grid.x_coordinates_km[-1])),
            (float(grid.z_coordinates_km[0]), float(grid.z_coordinates_km[-1])),
        )
        if curve is not None:
            x_values, z_values = curve
            axis.plot(
                x_values,
                z_values,
                color=ANOMALY_EDGE_COLOR,
                linewidth=2.0,
                linestyle="--",
            )
        return
    if scenario == "salt_dome":
        salt_dome = _scenario_mapping(grid, "salt_dome")
        center_x, center_y, center_z = _float_triplet(salt_dome["center_km"])
        radius_x, radius_y, radius_z = _float_triplet(salt_dome["radii_km"])
        y_offset = (y_value - center_y) / radius_y
        if abs(y_offset) <= 1.0:
            scale = math.sqrt(max(0.0, 1.0 - y_offset**2))
            axis.add_patch(
                Ellipse(
                    (center_x, center_z),
                    width=2.0 * radius_x * scale,
                    height=2.0 * radius_z * scale,
                    linewidth=2.0,
                    edgecolor=ANOMALY_EDGE_COLOR,
                    facecolor="none",
                )
            )
        return
    if scenario == "dyke_intrusion":
        dyke = _scenario_mapping(grid, "dyke_intrusion")
        x_bounds = _dyke_section_x_bounds(dyke, y_value)
        if x_bounds is not None:
            x_min, x_max = x_bounds
            axis.add_patch(
                Rectangle(
                    (x_min, float(dyke["top_depth_km"])),
                    x_max - x_min,
                    float(dyke["bottom_depth_km"]) - float(dyke["top_depth_km"]),
                    linewidth=1.8,
                    edgecolor=ANOMALY_EDGE_COLOR,
                    facecolor="none",
                )
            )


def _draw_plan_inset(axis, grid: CartesianVelocityGrid3D, y_value: float) -> None:
    axis.set_facecolor("white")
    x_min, x_max = float(grid.x_coordinates_km[0]), float(grid.x_coordinates_km[-1])
    y_min, y_max = float(grid.y_coordinates_km[0]), float(grid.y_coordinates_km[-1])
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_title("Plan View", fontsize=10, pad=6)
    axis.set_xlabel("x", fontsize=9)
    axis.set_ylabel("y", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.grid(color="#e2e8f0", linestyle="--", linewidth=0.6)

    scenario = str(grid.metadata.get("grid_scenario", ""))
    if scenario == "block_anomaly":
        block = _scenario_mapping(grid, "block_anomaly")
        center_x, center_y, _ = _float_triplet(block["center_km"])
        size_x, size_y, _ = _float_triplet(block["size_km"])
        axis.add_patch(
            Rectangle(
                (center_x - size_x / 2.0, center_y - size_y / 2.0),
                size_x,
                size_y,
                edgecolor=ANOMALY_EDGE_COLOR,
                facecolor=ANOMALY_FACE_COLOR,
                alpha=0.25,
                linewidth=1.5,
            )
        )
    if scenario == "faulted":
        segment = _fault_trace_segment(
            _fault_settings_from_metadata(_scenario_mapping(grid, "faulted")),
            (x_min, x_max),
            (y_min, y_max),
        )
        if segment is not None:
            (x_start, y_start), (x_end, y_end) = segment
            axis.plot(
                [x_start, x_end],
                [y_start, y_end],
                color=ANOMALY_EDGE_COLOR,
                linewidth=1.8,
            )
    if scenario == "salt_dome":
        salt_dome = _scenario_mapping(grid, "salt_dome")
        center_x, center_y, _ = _float_triplet(salt_dome["center_km"])
        radius_x, radius_y, _ = _float_triplet(salt_dome["radii_km"])
        axis.add_patch(
            Ellipse(
                (center_x, center_y),
                width=2.0 * radius_x,
                height=2.0 * radius_y,
                edgecolor=ANOMALY_EDGE_COLOR,
                facecolor=ANOMALY_FACE_COLOR,
                alpha=0.22,
                linewidth=1.5,
            )
        )
    if scenario == "dyke_intrusion":
        dyke = _scenario_mapping(grid, "dyke_intrusion")
        axis.add_patch(
            Polygon(
                _dyke_plan_polygon(dyke),
                closed=True,
                edgecolor=ANOMALY_EDGE_COLOR,
                facecolor=ANOMALY_FACE_COLOR,
                alpha=0.22,
                linewidth=1.5,
            )
        )

    axis.plot([x_min, x_max], [y_value, y_value], color=SLICE_LINE_COLOR, linewidth=2.0)
    axis.text(
        x_min + 2.0,
        y_value + 2.0,
        f"y={y_value:.1f} km",
        fontsize=8.5,
        color=SLICE_LINE_COLOR,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.2"},
    )


def _grid_values_3d(grid: CartesianVelocityGrid3D) -> np.ndarray:
    shape = (
        len(grid.z_coordinates_km),
        len(grid.y_coordinates_km),
        len(grid.x_coordinates_km),
    )
    return np.asarray(grid.p_velocity_km_per_s, dtype=float).reshape(shape)


def _fault_settings_from_metadata(faulted: dict[str, object]) -> FaultedGridSettings:
    return FaultedGridSettings(
        fault_x_km=float(faulted["fault_x_km"]),
        fault_y_km=float(faulted["fault_y_km"]),
        strike_deg=float(faulted["strike_deg"]),
        dip_deg=float(faulted["dip_deg"]),
        dip_direction=str(faulted["dip_direction"]),
        positive_side=str(faulted["positive_side"]),
        velocity_offset_km_per_s=float(faulted["velocity_offset_km_per_s"]),
    )


def _fault_section_curve(
    faulted: FaultedGridSettings,
    y_value: float,
    x_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> tuple[list[float], list[float]] | None:
    normal_x, normal_y = fault_plane_horizontal_normal(faulted)
    if math.isclose(normal_x, 0.0, rel_tol=0.0, abs_tol=1.0e-9):
        return None
    horizontal_shift_sign = fault_dip_direction_sign(faulted)
    x_values: list[float] = []
    z_values: list[float] = []
    for z_value in np.linspace(z_bounds[0], z_bounds[1], 200):
        horizontal_offset_km = 0.0
        if not math.isclose(faulted.dip_deg, 90.0, rel_tol=0.0, abs_tol=1.0e-9):
            horizontal_offset_km = z_value / math.tan(math.radians(faulted.dip_deg))
        x_value = (
            faulted.fault_x_km
            + (
                horizontal_shift_sign * horizontal_offset_km
                - normal_y * (y_value - faulted.fault_y_km)
            )
            / normal_x
        )
        if x_bounds[0] <= x_value <= x_bounds[1]:
            x_values.append(float(x_value))
            z_values.append(float(z_value))
    if len(x_values) < 2:
        return None
    return x_values, z_values


def _fault_trace_segment(
    faulted: FaultedGridSettings,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    strike_rad = math.radians(faulted.strike_deg)
    along_x = math.cos(strike_rad)
    along_y = math.sin(strike_rad)
    candidates: list[tuple[float, float, float]] = []

    if not math.isclose(along_x, 0.0, rel_tol=0.0, abs_tol=1.0e-9):
        for x_value in x_bounds:
            parameter = (x_value - faulted.fault_x_km) / along_x
            y_value = faulted.fault_y_km + parameter * along_y
            if y_bounds[0] <= y_value <= y_bounds[1]:
                candidates.append((parameter, float(x_value), float(y_value)))
    if not math.isclose(along_y, 0.0, rel_tol=0.0, abs_tol=1.0e-9):
        for y_value in y_bounds:
            parameter = (y_value - faulted.fault_y_km) / along_y
            x_value = faulted.fault_x_km + parameter * along_x
            if x_bounds[0] <= x_value <= x_bounds[1]:
                candidates.append((parameter, float(x_value), float(y_value)))

    unique_candidates: list[tuple[float, float, float]] = []
    for candidate in candidates:
        if any(
            math.isclose(candidate[1], existing[1], rel_tol=0.0, abs_tol=1.0e-9)
            and math.isclose(candidate[2], existing[2], rel_tol=0.0, abs_tol=1.0e-9)
            for existing in unique_candidates
        ):
            continue
        unique_candidates.append(candidate)

    if len(unique_candidates) < 2:
        return None
    unique_candidates.sort(key=lambda item: item[0])
    start = unique_candidates[0]
    end = unique_candidates[-1]
    return (start[1], start[2]), (end[1], end[2])


def _dyke_plan_polygon(dyke: dict[str, object]) -> list[tuple[float, float]]:
    center_x, center_y, _ = _float_triplet(dyke["center_km"])
    strike_rad = math.radians(float(dyke["strike_deg"]))
    length = float(dyke["length_km"])
    width = float(dyke["width_km"])
    along = np.array([math.cos(strike_rad), math.sin(strike_rad)])
    normal = np.array([-along[1], along[0]])
    center = np.array([center_x, center_y])
    corners = [
        center - along * (length / 2.0) - normal * (width / 2.0),
        center + along * (length / 2.0) - normal * (width / 2.0),
        center + along * (length / 2.0) + normal * (width / 2.0),
        center - along * (length / 2.0) + normal * (width / 2.0),
    ]
    return [(float(point[0]), float(point[1])) for point in corners]


def _dyke_section_x_bounds(dyke: dict[str, object], y_value: float) -> tuple[float, float] | None:
    center_x, center_y, _ = _float_triplet(dyke["center_km"])
    strike_rad = math.radians(float(dyke["strike_deg"]))
    length = float(dyke["length_km"])
    width = float(dyke["width_km"])
    along_x = math.cos(strike_rad)
    along_y = math.sin(strike_rad)
    normal_x = -along_y
    normal_y = along_x
    dy = y_value - center_y
    x_samples = np.linspace(center_x - length, center_x + length, 400)
    inside = []
    for x_value in x_samples:
        dx = x_value - center_x
        along_distance = abs(dx * along_x + dy * along_y)
        normal_distance = abs(dx * normal_x + dy * normal_y)
        if along_distance <= length / 2.0 and normal_distance <= width / 2.0:
            inside.append(float(x_value))
    if not inside:
        return None
    return min(inside), max(inside)


def _nearest_index(coordinates: np.ndarray, focus_value: float) -> int:
    return int(np.abs(coordinates - focus_value).argmin())


def _scenario_description(grid: CartesianVelocityGrid3D, y_value: float) -> str:
    scenario = str(grid.metadata.get("grid_scenario", ""))
    if scenario == "layered":
        return (
            "Front-looking x-z section through the layered reference model, with a plan-view inset "
            "showing the chosen y position."
        )
    if scenario == "block_anomaly":
        return (
            f"Front-looking x-z section at y={y_value:.1f} km through the positive-velocity block anomaly; "
            "the inset shows the block footprint and section line."
        )
    if scenario == "faulted":
        faulted = _scenario_mapping(grid, "faulted")
        return (
            f"Front-looking x-z section at y={y_value:.1f} km through the faulted model; "
            f"the dashed line marks the dipping fault plane (strike {float(faulted['strike_deg']):.0f}°, "
            f"dip {float(faulted['dip_deg']):.0f}°)."
        )
    if scenario == "salt_dome":
        return (
            f"Front-looking x-z section at y={y_value:.1f} km through the salt dome; "
            "the ellipse outlines the dome cross-section and the inset shows its plan-view footprint."
        )
    if scenario == "dyke_intrusion":
        return (
            f"Front-looking x-z section at y={y_value:.1f} km through the dyke-like intrusion; "
            "the inset shows the rotated dyke footprint and where the section cuts it."
        )
    return f"Front-looking x-z section at y={y_value:.1f} km through the configured velocity grid."


def _pretty_scenario_name(scenario: str) -> str:
    return scenario.replace("_", " ").title()


def _scenario_mapping(grid: CartesianVelocityGrid3D, key: str) -> dict[str, object]:
    value = grid.metadata.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Expected '{key}' metadata mapping in velocity grid.")
    return value


def _float_triplet(values: object) -> tuple[float, float, float]:
    if not isinstance(values, list) or len(values) != 3:
        raise TypeError("Expected a list of three numeric values.")
    return float(values[0]), float(values[1]), float(values[2])
