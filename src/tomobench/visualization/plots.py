"""Simple dependency-free SVG visualizations for the first vertical slice."""

from __future__ import annotations

import math
from pathlib import Path

from tomobench.config.settings import BenchmarkSettings
from tomobench.domain.schemas import (
    CartesianVelocityGrid3D,
    EarthquakeConfiguration,
    LayeredVelocityModel,
    StationConfiguration,
)


def plot_types() -> tuple[str, ...]:
    """Return the initial plot categories supported by the project."""
    return (
        "study_area_overview",
        "station_layout",
        "earthquake_distribution",
        "velocity_slice",
        "velocity_grid_orthogonal_slices",
        "prediction_vs_reference",
    )


def save_geometry_svg(
    stations: StationConfiguration,
    earthquakes: EarthquakeConfiguration,
    settings: BenchmarkSettings,
    path: Path,
) -> Path:
    """Save an x-y geometry sanity-check figure as SVG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 820, 700
    left, top, plot_size = 90, 60, 560
    x_min, x_max = settings.study_area.x_range_km
    y_min, y_max = settings.study_area.y_range_km

    def sx(x_km: float) -> float:
        return left + ((x_km - x_min) / (x_max - x_min)) * plot_size

    def sy(y_km: float) -> float:
        return top + plot_size - ((y_km - y_min) / (y_max - y_min)) * plot_size

    station_marks = "\n".join(
        (
            f'<path d="M {sx(station.location.x_km):.2f} {sy(station.location.y_km) - 6:.2f} '
            f"L {sx(station.location.x_km) - 6:.2f} {sy(station.location.y_km) + 6:.2f} "
            f'L {sx(station.location.x_km) + 6:.2f} {sy(station.location.y_km) + 6:.2f} Z" '
            'fill="#2563eb" stroke="#1e3a8a" stroke-width="1">'
            f"<title>{station.station_id}</title></path>"
        )
        for station in stations.stations
    )
    earthquake_marks = "\n".join(
        (
            f'<circle cx="{sx(event.hypocenter.x_km):.2f}" '
            f'cy="{sy(event.hypocenter.y_km):.2f}" r="4" '
            'fill="#dc2626" fill-opacity="0.62" stroke="#7f1d1d" stroke-width="0.8">'
            f"<title>{event.earthquake_id}, depth {event.hypocenter.z_km:.2f} km</title></circle>"
        )
        for event in earthquakes.earthquakes
    )
    margin = settings.study_area.margin_km
    margin_rect = (
        f'<rect x="{sx(x_min + margin):.2f}" y="{sy(y_max - margin):.2f}" '
        f'width="{sx(x_max - margin) - sx(x_min + margin):.2f}" '
        f'height="{sy(y_min + margin) - sy(y_max - margin):.2f}" '
        'fill="none" stroke="#64748b" stroke-dasharray="5 5" stroke-width="1.4" />'
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{left}" y="32" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">First vertical slice geometry</text>
  <text x="{left}" y="52" font-family="Arial, sans-serif" font-size="12" fill="#374151">Cartesian x-y view; station triangles at z=0 km, earthquake circles colored uniformly with depth shown in tooltips.</text>
  <rect x="{left}" y="{top}" width="{plot_size}" height="{plot_size}" fill="#f8fafc" stroke="#111827" stroke-width="1.5" />
  {margin_rect}
  {earthquake_marks}
  {station_marks}
  <line x1="{left}" y1="{top + plot_size}" x2="{left + plot_size}" y2="{top + plot_size}" stroke="#111827" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_size}" stroke="#111827" />
  <text x="{left + plot_size / 2 - 30}" y="{top + plot_size + 42}" font-family="Arial, sans-serif" font-size="14" fill="#111827">x (km)</text>
  <text x="26" y="{top + plot_size / 2}" transform="rotate(-90 26 {top + plot_size / 2})" font-family="Arial, sans-serif" font-size="14" fill="#111827">y (km)</text>
  <text x="{left}" y="{top + plot_size + 20}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{x_min:g}</text>
  <text x="{left + plot_size - 24}" y="{top + plot_size + 20}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{x_max:g}</text>
  <text x="{left - 35}" y="{top + plot_size + 4}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{y_min:g}</text>
  <text x="{left - 42}" y="{top + 4}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{y_max:g}</text>
  <path d="M 690 118 L 678 142 L 702 142 Z" fill="#2563eb" stroke="#1e3a8a" />
  <text x="716" y="137" font-family="Arial, sans-serif" font-size="13" fill="#111827">Stations ({len(stations.stations)})</text>
  <circle cx="690" cy="172" r="5" fill="#dc2626" fill-opacity="0.62" stroke="#7f1d1d" />
  <text x="716" y="177" font-family="Arial, sans-serif" font-size="13" fill="#111827">Earthquakes ({len(earthquakes.earthquakes)})</text>
  <line x1="678" y1="210" x2="702" y2="210" stroke="#64748b" stroke-dasharray="5 5" stroke-width="1.4" />
  <text x="716" y="215" font-family="Arial, sans-serif" font-size="13" fill="#111827">Configured margin</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def save_velocity_profile_svg(model: LayeredVelocityModel, path: Path) -> Path:
    """Save a simple velocity-depth profile SVG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 520, 520
    left, top, plot_width, plot_height = 90, 45, 330, 390
    min_velocity = min(layer.p_velocity_km_per_s for layer in model.layers)
    max_velocity = max(layer.p_velocity_km_per_s for layer in model.layers)
    min_depth = min(layer.top_depth_km for layer in model.layers)
    max_depth = max(layer.bottom_depth_km for layer in model.layers)

    def sx(velocity: float) -> float:
        if max_velocity == min_velocity:
            return left + plot_width / 2
        return left + ((velocity - min_velocity) / (max_velocity - min_velocity)) * plot_width

    def sy(depth: float) -> float:
        return top + ((depth - min_depth) / (max_depth - min_depth)) * plot_height

    blocks = "\n".join(
        (
            f'<rect x="{left}" y="{sy(layer.top_depth_km):.2f}" width="{plot_width}" '
            f'height="{sy(layer.bottom_depth_km) - sy(layer.top_depth_km):.2f}" '
            f'fill="{_layer_color(index)}" fill-opacity="0.24" stroke="#cbd5e1" />'
            f'<text x="{left + plot_width + 16}" y="{(sy(layer.top_depth_km) + sy(layer.bottom_depth_km)) / 2:.2f}" '
            'font-family="Arial, sans-serif" font-size="12" fill="#111827">'
            f"{layer.layer_id}: {layer.p_velocity_km_per_s:g} km/s</text>"
        )
        for index, layer in enumerate(model.layers)
    )
    profile_points = []
    for layer in model.layers:
        profile_points.append(f"{sx(layer.p_velocity_km_per_s):.2f},{sy(layer.top_depth_km):.2f}")
        profile_points.append(
            f"{sx(layer.p_velocity_km_per_s):.2f},{sy(layer.bottom_depth_km):.2f}"
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{left}" y="28" font-family="Arial, sans-serif" font-size="19" font-weight="700" fill="#111827">Layered P-wave velocity profile</text>
  <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#f8fafc" stroke="#111827" />
  {blocks}
  <polyline points="{" ".join(profile_points)}" fill="none" stroke="#0f766e" stroke-width="3" />
  <text x="{left + plot_width / 2 - 60}" y="{top + plot_height + 42}" font-family="Arial, sans-serif" font-size="14" fill="#111827">P velocity (km/s)</text>
  <text x="26" y="{top + plot_height / 2}" transform="rotate(-90 26 {top + plot_height / 2})" font-family="Arial, sans-serif" font-size="14" fill="#111827">Depth (km)</text>
  <text x="{left}" y="{top + plot_height + 20}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{min_velocity:g}</text>
  <text x="{left + plot_width - 24}" y="{top + plot_height + 20}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{max_velocity:g}</text>
  <text x="{left - 42}" y="{top + 4}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{min_depth:g}</text>
  <text x="{left - 42}" y="{top + plot_height + 4}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{max_depth:g}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def save_velocity_grid_slices_svg(grid: CartesianVelocityGrid3D, path: Path) -> Path:
    """Save three orthogonal velocity-grid slices as one SVG figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 920
    panel_width, panel_height = 300.0, 230.0
    panel_left = (70.0, 440.0, 810.0)
    panel_top = (90.0, 90.0, 90.0)
    x_mid_index = len(grid.x_coordinates_km) // 2
    y_mid_index = len(grid.y_coordinates_km) // 2
    z_mid_index = len(grid.z_coordinates_km) // 2
    velocities = tuple(float(value) for value in grid.p_velocity_km_per_s)
    min_velocity = min(velocities)
    max_velocity = max(velocities)
    colorbar_left = 1060.0
    colorbar_top = 150.0
    colorbar_height = 420.0

    xy_cells = _render_slice_cells(
        grid=grid,
        slice_kind="xy",
        fixed_index=z_mid_index,
        left=panel_left[0],
        top=panel_top[0],
        width=panel_width,
        height=panel_height,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
    )
    xz_cells = _render_slice_cells(
        grid=grid,
        slice_kind="xz",
        fixed_index=y_mid_index,
        left=panel_left[1],
        top=panel_top[1],
        width=panel_width,
        height=panel_height,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
    )
    yz_cells = _render_slice_cells(
        grid=grid,
        slice_kind="yz",
        fixed_index=x_mid_index,
        left=panel_left[2],
        top=panel_top[2],
        width=panel_width,
        height=panel_height,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
    )
    colorbar_cells = _render_colorbar(
        left=colorbar_left,
        top=colorbar_top,
        width=26.0,
        height=colorbar_height,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
    )
    scenario_label = str(grid.metadata.get("grid_scenario", "unknown"))
    summary_lines = _grid_summary_lines(grid)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="70" y="38" font-family="Arial, sans-serif" font-size="26" font-weight="700" fill="#111827">3D subsurface velocity-grid slices</text>
  <text x="70" y="66" font-family="Arial, sans-serif" font-size="15" fill="#374151">Scenario: {scenario_label}. Three orthogonal slices from the node-centered Cartesian P-wave velocity grid.</text>
  {_panel_frame(panel_left[0], panel_top[0], panel_width, panel_height, f"Plan view (x-y) at z={grid.z_coordinates_km[z_mid_index]:.1f} km")}
  {_panel_frame(panel_left[1], panel_top[1], panel_width, panel_height, f"Cross-section (x-z) at y={grid.y_coordinates_km[y_mid_index]:.1f} km")}
  {_panel_frame(panel_left[2], panel_top[2], panel_width, panel_height, f"Cross-section (y-z) at x={grid.x_coordinates_km[x_mid_index]:.1f} km")}
  {xy_cells}
  {xz_cells}
  {yz_cells}
  {_panel_axes_text(panel_left[0], panel_top[0], panel_width, panel_height, "x (km)", "y (km)", f"{grid.x_coordinates_km[0]:g}", f"{grid.x_coordinates_km[-1]:g}", f"{grid.y_coordinates_km[0]:g}", f"{grid.y_coordinates_km[-1]:g}")}
  {_panel_axes_text(panel_left[1], panel_top[1], panel_width, panel_height, "x (km)", "depth z (km)", f"{grid.x_coordinates_km[0]:g}", f"{grid.x_coordinates_km[-1]:g}", f"{grid.z_coordinates_km[0]:g}", f"{grid.z_coordinates_km[-1]:g}")}
  {_panel_axes_text(panel_left[2], panel_top[2], panel_width, panel_height, "y (km)", "depth z (km)", f"{grid.y_coordinates_km[0]:g}", f"{grid.y_coordinates_km[-1]:g}", f"{grid.z_coordinates_km[0]:g}", f"{grid.z_coordinates_km[-1]:g}")}
  <text x="{colorbar_left - 6:.2f}" y="{colorbar_top - 16:.2f}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#111827">P velocity</text>
  {colorbar_cells}
  <text x="{colorbar_left + 36:.2f}" y="{colorbar_top + 8:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{max_velocity:.2f} km/s</text>
  <text x="{colorbar_left + 36:.2f}" y="{colorbar_top + colorbar_height + 4:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{min_velocity:.2f} km/s</text>
  <text x="70" y="390" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#111827">Grid summary</text>
  {"".join(summary_lines)}
  <text x="70" y="846" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Color differences reflect absolute node-centered velocity values on the frozen ML target grid, not perturbation percentages.</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def _panel_frame(left: float, top: float, width: float, height: float, title: str) -> str:
    return (
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" '
        'fill="#f8fafc" stroke="#0f172a" stroke-width="1.3" />'
        f'<text x="{left:.2f}" y="{top - 16:.2f}" font-family="Arial, sans-serif" '
        f'font-size="15" font-weight="700" fill="#111827">{title}</text>'
    )


def _panel_axes_text(
    left: float,
    top: float,
    width: float,
    height: float,
    x_label: str,
    y_label: str,
    x_min: str,
    x_max: str,
    y_min: str,
    y_max: str,
) -> str:
    return (
        f'<text x="{left + width / 2 - 24:.2f}" y="{top + height + 34:.2f}" '
        f'font-family="Arial, sans-serif" font-size="13" fill="#111827">{x_label}</text>'
        f'<text x="{left - 46:.2f}" y="{top + height / 2:.2f}" transform="rotate(-90 {left - 46:.2f} {top + height / 2:.2f})" '
        f'font-family="Arial, sans-serif" font-size="13" fill="#111827">{y_label}</text>'
        f'<text x="{left:.2f}" y="{top + height + 16:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{x_min}</text>'
        f'<text x="{left + width - 24:.2f}" y="{top + height + 16:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{x_max}</text>'
        f'<text x="{left - 34:.2f}" y="{top + height + 4:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{y_max}</text>'
        f'<text x="{left - 28:.2f}" y="{top + 4:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{y_min}</text>'
    )


def _render_colorbar(
    left: float,
    top: float,
    width: float,
    height: float,
    min_velocity: float,
    max_velocity: float,
) -> str:
    steps = 40
    bar_height = height / steps
    cells = []
    for index in range(steps):
        position = 1.0 - (index / max(steps - 1, 1))
        velocity = min_velocity + position * (max_velocity - min_velocity)
        cells.append(
            f'<rect x="{left:.2f}" y="{top + index * bar_height:.2f}" width="{width:.2f}" '
            f'height="{bar_height + 0.4:.2f}" fill="{_velocity_color(velocity, min_velocity, max_velocity)}" stroke="none" />'
        )
    cells.append(
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" fill="none" stroke="#0f172a" stroke-width="1.0" />'
    )
    return "".join(cells)


def _render_slice_cells(
    grid: CartesianVelocityGrid3D,
    slice_kind: str,
    fixed_index: int,
    left: float,
    top: float,
    width: float,
    height: float,
    min_velocity: float,
    max_velocity: float,
) -> str:
    if slice_kind == "xy":
        x_count = len(grid.x_coordinates_km)
        y_count = len(grid.y_coordinates_km)
        cell_width = width / x_count
        cell_height = height / y_count
        cells = []
        for iy in range(y_count):
            for ix in range(x_count):
                velocity = _grid_velocity_at(grid, ix=ix, iy=iy, iz=fixed_index)
                x = left + ix * cell_width
                y = top + height - (iy + 1) * cell_height
                cells.append(
                    _cell_rect(x, y, cell_width, cell_height, velocity, min_velocity, max_velocity)
                )
        return "".join(cells)
    if slice_kind == "xz":
        x_count = len(grid.x_coordinates_km)
        z_count = len(grid.z_coordinates_km)
        cell_width = width / x_count
        cell_height = height / z_count
        cells = []
        for iz in range(z_count):
            for ix in range(x_count):
                velocity = _grid_velocity_at(grid, ix=ix, iy=fixed_index, iz=iz)
                x = left + ix * cell_width
                y = top + iz * cell_height
                cells.append(
                    _cell_rect(x, y, cell_width, cell_height, velocity, min_velocity, max_velocity)
                )
        return "".join(cells)
    if slice_kind == "yz":
        y_count = len(grid.y_coordinates_km)
        z_count = len(grid.z_coordinates_km)
        cell_width = width / y_count
        cell_height = height / z_count
        cells = []
        for iz in range(z_count):
            for iy in range(y_count):
                velocity = _grid_velocity_at(grid, ix=fixed_index, iy=iy, iz=iz)
                x = left + iy * cell_width
                y = top + iz * cell_height
                cells.append(
                    _cell_rect(x, y, cell_width, cell_height, velocity, min_velocity, max_velocity)
                )
        return "".join(cells)
    raise ValueError(f"Unsupported slice kind: {slice_kind}")


def _cell_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    velocity: float,
    min_velocity: float,
    max_velocity: float,
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width + 0.25:.2f}" height="{height + 0.25:.2f}" '
        f'fill="{_velocity_color(velocity, min_velocity, max_velocity)}" stroke="none" />'
    )


def _velocity_color(velocity: float, min_velocity: float, max_velocity: float) -> str:
    if math.isclose(min_velocity, max_velocity, rel_tol=0.0, abs_tol=1.0e-12):
        normalized = 0.5
    else:
        normalized = max(0.0, min(1.0, (velocity - min_velocity) / (max_velocity - min_velocity)))
    # Blue -> cyan -> yellow -> red
    if normalized < 0.33:
        local = normalized / 0.33
        red = int(38 + local * (80 - 38))
        green = int(88 + local * (196 - 88))
        blue = int(220 + local * (212 - 220))
    elif normalized < 0.66:
        local = (normalized - 0.33) / 0.33
        red = int(80 + local * (245 - 80))
        green = int(196 + local * (215 - 196))
        blue = int(212 - local * (80))
    else:
        local = (normalized - 0.66) / 0.34
        red = int(245)
        green = int(215 - local * (150))
        blue = int(132 - local * (90))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _grid_velocity_at(grid: CartesianVelocityGrid3D, ix: int, iy: int, iz: int) -> float:
    nx = len(grid.x_coordinates_km)
    ny = len(grid.y_coordinates_km)
    index = ix + nx * (iy + ny * iz)
    return float(grid.p_velocity_km_per_s[index])


def _grid_summary_lines(grid: CartesianVelocityGrid3D) -> list[str]:
    metadata = grid.metadata
    lines = [
        f"Grid ID: {grid.grid_id}",
        f"Source velocity model: {grid.source_velocity_model_id}",
        f"Scenario: {metadata.get('grid_scenario', 'unknown')}",
        f"Grid shape: nx={len(grid.x_coordinates_km)}, ny={len(grid.y_coordinates_km)}, nz={len(grid.z_coordinates_km)}",
        f"Grid spacing: {metadata.get('grid_spacing_km', 'unknown')} km",
    ]
    scenario_key = str(metadata.get("grid_scenario", ""))
    if scenario_key in metadata and isinstance(metadata[scenario_key], dict):
        scenario_metadata = metadata[scenario_key]
        for key, value in scenario_metadata.items():
            if key == "bounds_km":
                continue
            lines.append(f"{key}: {value}")
    rendered: list[str] = []
    for index, line in enumerate(lines):
        rendered.append(
            f'<text x="70" y="{420 + index * 24:.2f}" font-family="Arial, sans-serif" font-size="13" fill="#374151">{line}</text>'
        )
    return rendered


def _layer_color(index: int) -> str:
    colors = ("#93c5fd", "#86efac", "#fde68a", "#fca5a5", "#c4b5fd")
    return colors[index % len(colors)]
