"""SVG figures for explaining the eikonal benchmark workflow."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import CartesianVelocityGrid3D
from tomobench.simulation.travel_times import _neighbor_offsets
from tomobench.visualization.plots import (
    _panel_axes_text,
    _panel_frame,
    _render_colorbar,
    _velocity_color,
)


def save_eikonal_connectivity_svg(path: Path) -> Path:
    """Save a conceptual 6/18/26-neighbor connectivity figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1240, 520
    panel_width, panel_height = 300.0, 260.0
    panel_left = (70.0, 450.0, 830.0)
    panel_top = 150.0
    connectivities = (6, 18, 26)
    legend = _connectivity_legend()

    panels: list[str] = []
    for left, connectivity in zip(panel_left, connectivities, strict=True):
        panels.append(
            _render_connectivity_panel(
                left=left,
                top=panel_top,
                width=panel_width,
                height=panel_height,
                connectivity=connectivity,
            )
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="70" y="42" font-family="Arial, sans-serif" font-size="26" font-weight="700" fill="#111827">3D graph connectivity used by the eikonal prototype</text>
  <text x="70" y="68" font-family="Arial, sans-serif" font-size="15" fill="#374151">Each panel shows the neighbors reachable from one central grid node when Dijkstra propagation is limited to face, face-plus-edge, or full face-edge-corner connections.</text>
  <text x="70" y="94" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">The benchmark CSV compares these choices because higher connectivity gives the shortest-path solver more directional freedom on the Cartesian grid.</text>
  {"".join(panels)}
  {legend}
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def save_eikonal_path_overlay_svg(
    grid: CartesianVelocityGrid3D,
    source: Point3D,
    receiver: Point3D,
    path_points: tuple[Point3D, ...],
    path_travel_time_s: float,
    straight_travel_time_s: float,
    connectivity: int,
    grid_spacing_km: float,
    benchmark_case_name: str,
    path: Path,
) -> Path:
    """Save a Dijkstra-path overlay figure on orthogonal grid projections."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1240, 980
    panel_width, panel_height = 300.0, 250.0
    panel_left = (70.0, 440.0, 810.0)
    panel_top = (150.0, 150.0, 150.0)
    velocity_value = float(grid.p_velocity_km_per_s[0])

    xy_panel = _render_projection_panel(
        grid=grid,
        left=panel_left[0],
        top=panel_top[0],
        width=panel_width,
        height=panel_height,
        axis_pair=("x_km", "y_km"),
        title="Plan view (x-y projection)",
        source=source,
        receiver=receiver,
        path_points=path_points,
    )
    xz_panel = _render_projection_panel(
        grid=grid,
        left=panel_left[1],
        top=panel_top[1],
        width=panel_width,
        height=panel_height,
        axis_pair=("x_km", "z_km"),
        title="Cross-section (x-z projection)",
        source=source,
        receiver=receiver,
        path_points=path_points,
    )
    yz_panel = _render_projection_panel(
        grid=grid,
        left=panel_left[2],
        top=panel_top[2],
        width=panel_width,
        height=panel_height,
        axis_pair=("y_km", "z_km"),
        title="Cross-section (y-z projection)",
        source=source,
        receiver=receiver,
        path_points=path_points,
    )
    summary_lines = _summary_lines(
        (
            f"Benchmark case: {benchmark_case_name}",
            f"Grid scenario: {grid.metadata.get('grid_scenario', 'unknown')}",
            f"Connectivity: {connectivity} neighbors",
            f"Grid spacing: {grid_spacing_km:g} km",
            f"Source: ({source.x_km:g}, {source.y_km:g}, {source.z_km:g}) km",
            f"Receiver: ({receiver.x_km:g}, {receiver.y_km:g}, {receiver.z_km:g}) km",
            f"Dijkstra path nodes: {len(path_points)}",
            f"Dijkstra first-arrival time: {path_travel_time_s:.3f} s",
            f"Straight-ray time in the homogeneous control: {straight_travel_time_s:.3f} s",
            f"Uniform benchmark velocity: {velocity_value:.2f} km/s",
        ),
        start_y=470.0,
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="70" y="42" font-family="Arial, sans-serif" font-size="26" font-weight="700" fill="#111827">Dijkstra first-arrival path on orthogonal grid projections</text>
  <text x="70" y="68" font-family="Arial, sans-serif" font-size="15" fill="#374151">The dashed black line is the straight source-receiver ray. The orange polyline is the node-to-node shortest travel-time path recovered from the same Dijkstra graph used by the solver.</text>
  <text x="70" y="94" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">This figure uses the homogeneous control geometry from the benchmark sweep so the path shape can be interpreted without laterally varying velocity overprints.</text>
  {xy_panel}
  {xz_panel}
  {yz_panel}
  {_projection_axes_text(panel_left[0], panel_top[0], panel_width, panel_height, "x (km)", "y (km)", grid.x_coordinates_km[0], grid.x_coordinates_km[-1], grid.y_coordinates_km[0], grid.y_coordinates_km[-1])}
  {_projection_axes_text(panel_left[1], panel_top[1], panel_width, panel_height, "x (km)", "depth z (km)", grid.x_coordinates_km[0], grid.x_coordinates_km[-1], grid.z_coordinates_km[0], grid.z_coordinates_km[-1])}
  {_projection_axes_text(panel_left[2], panel_top[2], panel_width, panel_height, "y (km)", "depth z (km)", grid.y_coordinates_km[0], grid.y_coordinates_km[-1], grid.z_coordinates_km[0], grid.z_coordinates_km[-1])}
  {_path_overlay_legend()}
  {"".join(summary_lines)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def save_eikonal_travel_time_slices_svg(
    grid: CartesianVelocityGrid3D,
    travel_times_by_node: list[float],
    source: Point3D,
    receiver: Point3D,
    path_points: tuple[Point3D, ...],
    connectivity: int,
    grid_spacing_km: float,
    benchmark_case_name: str,
    path: Path,
) -> Path:
    """Save orthogonal travel-time field slices from one benchmark source."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1240, 980
    panel_width, panel_height = 300.0, 230.0
    panel_left = (70.0, 440.0, 810.0)
    panel_top = (150.0, 150.0, 150.0)
    min_time = min(value for value in travel_times_by_node if math.isfinite(value))
    max_time = max(value for value in travel_times_by_node if math.isfinite(value))

    block_metadata = grid.metadata.get("block_anomaly", {})
    anomaly_center = block_metadata.get("center_km", [50.0, 50.0, 15.0])
    z_index = _nearest_axis_index(grid.z_coordinates_km, float(anomaly_center[2]))
    y_index = _nearest_axis_index(grid.y_coordinates_km, source.y_km)
    x_index = _nearest_axis_index(grid.x_coordinates_km, source.x_km)

    xy_panel = _render_scalar_slice_panel(
        grid=grid,
        values=travel_times_by_node,
        slice_kind="xy",
        fixed_index=z_index,
        left=panel_left[0],
        top=panel_top[0],
        width=panel_width,
        height=panel_height,
        title=f"Travel-time slice (x-y) at z={grid.z_coordinates_km[z_index]:.1f} km",
        min_value=min_time,
        max_value=max_time,
        source=source,
        receiver=receiver,
        path_points=path_points,
    )
    xz_panel = _render_scalar_slice_panel(
        grid=grid,
        values=travel_times_by_node,
        slice_kind="xz",
        fixed_index=y_index,
        left=panel_left[1],
        top=panel_top[1],
        width=panel_width,
        height=panel_height,
        title=f"Travel-time slice (x-z) at y={grid.y_coordinates_km[y_index]:.1f} km",
        min_value=min_time,
        max_value=max_time,
        source=source,
        receiver=receiver,
        path_points=path_points,
    )
    yz_panel = _render_scalar_slice_panel(
        grid=grid,
        values=travel_times_by_node,
        slice_kind="yz",
        fixed_index=x_index,
        left=panel_left[2],
        top=panel_top[2],
        width=panel_width,
        height=panel_height,
        title=f"Travel-time slice (y-z) at x={grid.x_coordinates_km[x_index]:.1f} km",
        min_value=min_time,
        max_value=max_time,
        source=source,
        receiver=receiver,
        path_points=path_points,
    )
    summary_lines = _summary_lines(
        (
            f"Benchmark case: {benchmark_case_name}",
            f"Grid scenario: {grid.metadata.get('grid_scenario', 'unknown')}",
            f"Connectivity: {connectivity} neighbors",
            f"Grid spacing: {grid_spacing_km:g} km",
            f"Source node: ({source.x_km:g}, {source.y_km:g}, {source.z_km:g}) km",
            f"Receiver node: ({receiver.x_km:g}, {receiver.y_km:g}, {receiver.z_km:g}) km",
            f"Travel-time range on the grid: {min_time:.2f} to {max_time:.2f} s",
            f"Block-anomaly center: ({anomaly_center[0]}, {anomaly_center[1]}, {anomaly_center[2]}) km",
        ),
        start_y=450.0,
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="70" y="42" font-family="Arial, sans-serif" font-size="26" font-weight="700" fill="#111827">First-arrival travel-time field on the block-anomaly grid</text>
  <text x="70" y="68" font-family="Arial, sans-serif" font-size="15" fill="#374151">Each cell is colored by the minimum travel time from the source node after Dijkstra propagation on the 3D Cartesian grid. The dashed black line is the straight source-receiver ray, and the orange polyline is the recovered shortest travel-time path.</text>
  <text x="70" y="94" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">This illustrative off-axis source-receiver pair bends toward the faster block anomaly, making the graph-based detour visible on the travel-time slices.</text>
  {xy_panel}
  {xz_panel}
  {yz_panel}
  {_panel_axes_text(panel_left[0], panel_top[0], panel_width, panel_height, "x (km)", "y (km)", f"{grid.x_coordinates_km[0]:g}", f"{grid.x_coordinates_km[-1]:g}", f"{grid.y_coordinates_km[0]:g}", f"{grid.y_coordinates_km[-1]:g}")}
  {_panel_axes_text(panel_left[1], panel_top[1], panel_width, panel_height, "x (km)", "depth z (km)", f"{grid.x_coordinates_km[0]:g}", f"{grid.x_coordinates_km[-1]:g}", f"{grid.z_coordinates_km[0]:g}", f"{grid.z_coordinates_km[-1]:g}")}
  {_panel_axes_text(panel_left[2], panel_top[2], panel_width, panel_height, "y (km)", "depth z (km)", f"{grid.y_coordinates_km[0]:g}", f"{grid.y_coordinates_km[-1]:g}", f"{grid.z_coordinates_km[0]:g}", f"{grid.z_coordinates_km[-1]:g}")}
  <text x="1086" y="134" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#111827">First-arrival time</text>
  {_render_colorbar(1086.0, 150.0, 26.0, 420.0, min_time, max_time)}
  <text x="1122" y="160" font-family="Arial, sans-serif" font-size="12" fill="#374151">{max_time:.2f} s</text>
  <text x="1122" y="574" font-family="Arial, sans-serif" font-size="12" fill="#374151">{min_time:.2f} s</text>
  {_travel_time_legend()}
  {"".join(summary_lines)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def _render_connectivity_panel(
    left: float,
    top: float,
    width: float,
    height: float,
    connectivity: int,
) -> str:
    active = set(_neighbor_offsets(connectivity))
    wireframe: list[str] = []
    active_edges: list[str] = []
    nodes: list[str] = []
    center_x = left + width / 2.0 - 22.0
    center_y = top + height / 2.0 + 12.0

    for dy in (-1, 0, 1):
        corners = (
            _cube_neighbor_projection(center_x, center_y, -1, dy, -1),
            _cube_neighbor_projection(center_x, center_y, 1, dy, -1),
            _cube_neighbor_projection(center_x, center_y, 1, dy, 1),
            _cube_neighbor_projection(center_x, center_y, -1, dy, 1),
        )
        wireframe.append(
            f'<polygon points="{" ".join(f"{px:.2f},{py:.2f}" for px, py in corners)}" fill="none" stroke="#cbd5e1" stroke-width="1.4" />'
        )

    for dx in (-1, 1):
        for dz in (-1, 1):
            start = _cube_neighbor_projection(center_x, center_y, dx, -1, dz)
            end = _cube_neighbor_projection(center_x, center_y, dx, 1, dz)
            wireframe.append(
                f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" y2="{end[1]:.2f}" stroke="#cbd5e1" stroke-width="1.4" />'
            )

    for dx, dy, dz in _neighbor_offsets(26):
        px, py = _cube_neighbor_projection(center_x, center_y, dx, dy, dz)
        manhattan = abs(dx) + abs(dy) + abs(dz)
        node_fill = "#ffffff"
        node_stroke = "#94a3b8"
        if (dx, dy, dz) in active:
            node_fill = _connectivity_color(manhattan)
            node_stroke = "#0f172a"
            active_edges.append(
                f'<line x1="{center_x:.2f}" y1="{center_y:.2f}" x2="{px:.2f}" y2="{py:.2f}" stroke="{node_fill}" stroke-width="2.4" stroke-linecap="round" opacity="0.78" />'
            )
        nodes.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6.6" fill="{node_fill}" stroke="{node_stroke}" stroke-width="1.1" />'
        )

    label = f"{connectivity}-neighbor connectivity"
    description = {
        6: "Faces only",
        18: "Faces plus edges",
        26: "Faces, edges, and corners",
    }[connectivity]
    return (
        _panel_frame(left, top, width, height, label)
        + f'<text x="{left:.2f}" y="{top + height + 28:.2f}" font-family="Arial, sans-serif" font-size="13" fill="#374151">{description}; {len(active)} active of 26 possible neighbors.</text>'
        + "".join(wireframe)
        + "".join(active_edges)
        + f'<circle cx="{center_x:.2f}" cy="{center_y:.2f}" r="9.2" fill="#111827" stroke="#111827" stroke-width="1.2" />'
        + "".join(nodes)
    )


def _render_projection_panel(
    grid: CartesianVelocityGrid3D,
    left: float,
    top: float,
    width: float,
    height: float,
    axis_pair: tuple[str, str],
    title: str,
    source: Point3D,
    receiver: Point3D,
    path_points: tuple[Point3D, ...],
) -> str:
    attr_x, attr_y = axis_pair
    x_values = _axis_values_for_attribute(grid, attr_x)
    y_values = _axis_values_for_attribute(grid, attr_y)
    map_x = _linear_mapper(left, left + width, x_values[0], x_values[-1])
    map_y = _axis_mapper(attr_y, top, height, y_values[0], y_values[-1])
    path_polyline = _polyline_points(path_points, attr_x, attr_y, map_x, map_y)
    straight_polyline = _polyline_points((source, receiver), attr_x, attr_y, map_x, map_y)

    background = f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" fill="#eef2ff" stroke="none" />'
    nodes = _render_projection_nodes(x_values, y_values, attr_x, attr_y, map_x, map_y)
    source_marker = _render_point_marker(source, attr_x, attr_y, map_x, map_y, "#111827", 7.0)
    receiver_marker = _render_triangle_marker(receiver, attr_x, attr_y, map_x, map_y, "#0f766e")
    path_markers = _render_path_markers(path_points, attr_x, attr_y, map_x, map_y)
    return (
        _panel_frame(left, top, width, height, title)
        + background
        + nodes
        + f'<polyline points="{straight_polyline}" fill="none" stroke="#111827" stroke-width="2.3" stroke-dasharray="9 7" stroke-linecap="round" stroke-linejoin="round" opacity="0.8" />'
        + f'<polyline points="{path_polyline}" fill="none" stroke="#f97316" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round" />'
        + path_markers
        + source_marker
        + receiver_marker
    )


def _render_scalar_slice_panel(
    grid: CartesianVelocityGrid3D,
    values: list[float],
    slice_kind: str,
    fixed_index: int,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    min_value: float,
    max_value: float,
    source: Point3D,
    receiver: Point3D,
    path_points: tuple[Point3D, ...],
) -> str:
    panel = [_panel_frame(left, top, width, height, title)]
    if slice_kind == "xy":
        x_count = len(grid.x_coordinates_km)
        y_count = len(grid.y_coordinates_km)
        cell_width = width / x_count
        cell_height = height / y_count
        for iy in range(y_count):
            for ix in range(x_count):
                value = _scalar_value_at(grid, values, ix=ix, iy=iy, iz=fixed_index)
                x = left + ix * cell_width
                y = top + height - (iy + 1) * cell_height
                panel.append(
                    _scalar_cell_rect(x, y, cell_width, cell_height, value, min_value, max_value)
                )
    elif slice_kind == "xz":
        x_count = len(grid.x_coordinates_km)
        z_count = len(grid.z_coordinates_km)
        cell_width = width / x_count
        cell_height = height / z_count
        for iz in range(z_count):
            for ix in range(x_count):
                value = _scalar_value_at(grid, values, ix=ix, iy=fixed_index, iz=iz)
                x = left + ix * cell_width
                y = top + iz * cell_height
                panel.append(
                    _scalar_cell_rect(x, y, cell_width, cell_height, value, min_value, max_value)
                )
    elif slice_kind == "yz":
        y_count = len(grid.y_coordinates_km)
        z_count = len(grid.z_coordinates_km)
        cell_width = width / y_count
        cell_height = height / z_count
        for iz in range(z_count):
            for iy in range(y_count):
                value = _scalar_value_at(grid, values, ix=fixed_index, iy=iy, iz=iz)
                x = left + iy * cell_width
                y = top + iz * cell_height
                panel.append(
                    _scalar_cell_rect(x, y, cell_width, cell_height, value, min_value, max_value)
                )
    else:
        raise ValueError(f"Unsupported slice kind: {slice_kind}")

    if slice_kind == "xy":
        panel.append(
            _overlay_projection_on_slice(
                left=left,
                top=top,
                width=width,
                height=height,
                source=source,
                receiver=receiver,
                path_points=path_points,
                axis_pair=("x_km", "y_km"),
                x_range=(grid.x_coordinates_km[0], grid.x_coordinates_km[-1]),
                y_range=(grid.y_coordinates_km[0], grid.y_coordinates_km[-1]),
            )
        )
    if slice_kind == "xz":
        panel.append(
            _overlay_projection_on_slice(
                left=left,
                top=top,
                width=width,
                height=height,
                source=source,
                receiver=receiver,
                path_points=path_points,
                axis_pair=("x_km", "z_km"),
                x_range=(grid.x_coordinates_km[0], grid.x_coordinates_km[-1]),
                y_range=(grid.z_coordinates_km[0], grid.z_coordinates_km[-1]),
            )
        )
    if slice_kind == "yz":
        panel.append(
            _overlay_projection_on_slice(
                left=left,
                top=top,
                width=width,
                height=height,
                source=source,
                receiver=receiver,
                path_points=path_points,
                axis_pair=("y_km", "z_km"),
                x_range=(grid.y_coordinates_km[0], grid.y_coordinates_km[-1]),
                y_range=(grid.z_coordinates_km[0], grid.z_coordinates_km[-1]),
            )
        )
    return "".join(panel)


def _overlay_projection_on_slice(
    left: float,
    top: float,
    width: float,
    height: float,
    source: Point3D,
    receiver: Point3D,
    path_points: tuple[Point3D, ...],
    axis_pair: tuple[str, str],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> str:
    attr_x, attr_y = axis_pair
    map_x = _linear_mapper(left, left + width, x_range[0], x_range[1])
    map_y = _axis_mapper(attr_y, top, height, y_range[0], y_range[1])
    return (
        f'<polyline points="{_polyline_points((source, receiver), attr_x, attr_y, map_x, map_y)}" fill="none" stroke="#111827" stroke-width="2.2" stroke-dasharray="9 7" stroke-linecap="round" stroke-linejoin="round" opacity="0.78" />'
        + f'<polyline points="{_polyline_points(path_points, attr_x, attr_y, map_x, map_y)}" fill="none" stroke="#f97316" stroke-width="3.0" stroke-linecap="round" stroke-linejoin="round" opacity="0.96" />'
        + _render_point_marker(source, attr_x, attr_y, map_x, map_y, "#ffffff", 6.0, "#111827")
        + _render_triangle_marker(receiver, attr_x, attr_y, map_x, map_y, "#ffffff", "#111827")
    )


def _path_overlay_legend() -> str:
    return (
        '<line x1="70" y1="436" x2="118" y2="436" stroke="#111827" stroke-width="2.2" stroke-dasharray="9 7" />'
        '<text x="128" y="441" font-family="Arial, sans-serif" font-size="13" fill="#111827">Straight-ray reference</text>'
        '<line x1="290" y1="436" x2="338" y2="436" stroke="#f97316" stroke-width="3.4" />'
        '<text x="348" y="441" font-family="Arial, sans-serif" font-size="13" fill="#111827">Recovered Dijkstra node path</text>'
        '<circle cx="590" cy="436" r="5.5" fill="#111827" />'
        '<text x="602" y="441" font-family="Arial, sans-serif" font-size="13" fill="#111827">Source node</text>'
        '<path d="M 770 430 L 763 443 L 777 443 Z" fill="#0f766e" stroke="#0f766e" />'
        '<text x="786" y="441" font-family="Arial, sans-serif" font-size="13" fill="#111827">Receiver node</text>'
    )


def _travel_time_legend() -> str:
    return (
        '<line x1="70" y1="434" x2="118" y2="434" stroke="#111827" stroke-width="2.2" stroke-dasharray="9 7" />'
        '<text x="128" y="439" font-family="Arial, sans-serif" font-size="13" fill="#111827">Straight-ray reference</text>'
        '<line x1="330" y1="434" x2="378" y2="434" stroke="#f97316" stroke-width="3.0" />'
        '<text x="388" y="439" font-family="Arial, sans-serif" font-size="13" fill="#111827">Recovered Dijkstra path</text>'
        '<circle cx="650" cy="434" r="5.5" fill="#ffffff" stroke="#111827" stroke-width="1.5" />'
        '<text x="662" y="439" font-family="Arial, sans-serif" font-size="13" fill="#111827">Source</text>'
        '<path d="M 772 428 L 765 441 L 779 441 Z" fill="#ffffff" stroke="#111827" stroke-width="1.4" />'
        '<text x="788" y="439" font-family="Arial, sans-serif" font-size="13" fill="#111827">Receiver</text>'
    )


def _connectivity_legend() -> str:
    return (
        '<circle cx="118" cy="472" r="8" fill="#3b82f6" stroke="#0f172a" stroke-width="1.0" />'
        '<text x="134" y="477" font-family="Arial, sans-serif" font-size="13" fill="#111827">Face neighbor</text>'
        '<circle cx="324" cy="472" r="8" fill="#f59e0b" stroke="#0f172a" stroke-width="1.0" />'
        '<text x="340" y="477" font-family="Arial, sans-serif" font-size="13" fill="#111827">Edge neighbor</text>'
        '<circle cx="522" cy="472" r="8" fill="#ef4444" stroke="#0f172a" stroke-width="1.0" />'
        '<text x="538" y="477" font-family="Arial, sans-serif" font-size="13" fill="#111827">Corner neighbor</text>'
        '<circle cx="740" cy="472" r="8.5" fill="#111827" stroke="#111827" stroke-width="1.0" />'
        '<text x="758" y="477" font-family="Arial, sans-serif" font-size="13" fill="#111827">Central node</text>'
    )


def _projection_axes_text(
    left: float,
    top: float,
    width: float,
    height: float,
    x_label: str,
    y_label: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    y_top = f"{y_max:g}" if "depth" not in y_label else f"{y_min:g}"
    y_bottom = f"{y_min:g}" if "depth" not in y_label else f"{y_max:g}"
    return (
        f'<text x="{left + width / 2 - 22:.2f}" y="{top + height + 34:.2f}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{x_label}</text>'
        f'<text x="{left - 46:.2f}" y="{top + height / 2:.2f}" transform="rotate(-90 {left - 46:.2f} {top + height / 2:.2f})" font-family="Arial, sans-serif" font-size="13" fill="#111827">{y_label}</text>'
        f'<text x="{left:.2f}" y="{top + height + 16:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{x_min:g}</text>'
        f'<text x="{left + width - 24:.2f}" y="{top + height + 16:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{x_max:g}</text>'
        f'<text x="{left - 34:.2f}" y="{top + height + 4:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{y_bottom}</text>'
        f'<text x="{left - 28:.2f}" y="{top + 4:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{y_top}</text>'
    )


def _render_projection_nodes(
    x_values: tuple[float, ...],
    y_values: tuple[float, ...],
    attr_x: str,
    attr_y: str,
    map_x: Callable[[float], float],
    map_y: Callable[[float], float],
) -> str:
    circles: list[str] = []
    x_step = max(1, math.ceil(len(x_values) / 21))
    y_step = max(1, math.ceil(len(y_values) / 21))
    for y_value in y_values[::y_step]:
        for x_value in x_values[::x_step]:
            point = Point3D(
                x_km=x_value if attr_x == "x_km" else (y_value if attr_y == "x_km" else 0.0),
                y_km=x_value if attr_x == "y_km" else (y_value if attr_y == "y_km" else 0.0),
                z_km=x_value if attr_x == "z_km" else (y_value if attr_y == "z_km" else 0.0),
            )
            circles.append(
                f'<circle cx="{map_x(_coordinate(point, attr_x)):.2f}" cy="{map_y(_coordinate(point, attr_y)):.2f}" r="1.35" fill="#94a3b8" opacity="0.45" />'
            )
    return "".join(circles)


def _render_path_markers(
    path_points: tuple[Point3D, ...],
    attr_x: str,
    attr_y: str,
    map_x: Callable[[float], float],
    map_y: Callable[[float], float],
) -> str:
    return "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="#fb923c" stroke="#9a3412" stroke-width="0.5" />'
        for x, y in _projected_pixels(path_points, attr_x, attr_y, map_x, map_y)
    )


def _render_point_marker(
    point: Point3D,
    attr_x: str,
    attr_y: str,
    map_x: Callable[[float], float],
    map_y: Callable[[float], float],
    fill: str,
    radius: float,
    stroke: str | None = None,
) -> str:
    marker_stroke = stroke or fill
    return f'<circle cx="{map_x(_coordinate(point, attr_x)):.2f}" cy="{map_y(_coordinate(point, attr_y)):.2f}" r="{radius:.2f}" fill="{fill}" stroke="{marker_stroke}" stroke-width="1.6" />'


def _render_triangle_marker(
    point: Point3D,
    attr_x: str,
    attr_y: str,
    map_x: Callable[[float], float],
    map_y: Callable[[float], float],
    fill: str,
    stroke: str | None = None,
) -> str:
    x = map_x(_coordinate(point, attr_x))
    y = map_y(_coordinate(point, attr_y))
    marker_stroke = stroke or fill
    return f'<path d="M {x:.2f} {y - 8:.2f} L {x - 7:.2f} {y + 6:.2f} L {x + 7:.2f} {y + 6:.2f} Z" fill="{fill}" stroke="{marker_stroke}" stroke-width="1.5" />'


def _polyline_points(
    points: tuple[Point3D, ...],
    attr_x: str,
    attr_y: str,
    map_x: Callable[[float], float],
    map_y: Callable[[float], float],
) -> str:
    return " ".join(
        f"{x:.2f},{y:.2f}" for x, y in _projected_pixels(points, attr_x, attr_y, map_x, map_y)
    )


def _summary_lines(lines: tuple[str, ...], start_y: float) -> list[str]:
    return [
        f'<text x="70" y="{start_y + index * 24:.2f}" font-family="Arial, sans-serif" font-size="13" fill="#374151">{line}</text>'
        for index, line in enumerate(lines)
    ]


def _scalar_cell_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    value: float,
    min_value: float,
    max_value: float,
) -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{width + 0.25:.2f}" height="{height + 0.25:.2f}" fill="{_velocity_color(value, min_value, max_value)}" stroke="none" />'


def _scalar_value_at(
    grid: CartesianVelocityGrid3D,
    values: list[float],
    ix: int,
    iy: int,
    iz: int,
) -> float:
    nx = len(grid.x_coordinates_km)
    ny = len(grid.y_coordinates_km)
    return float(values[ix + nx * (iy + ny * iz)])


def _axis_values_for_attribute(grid: CartesianVelocityGrid3D, attribute: str) -> tuple[float, ...]:
    if attribute == "x_km":
        return grid.x_coordinates_km
    if attribute == "y_km":
        return grid.y_coordinates_km
    if attribute == "z_km":
        return grid.z_coordinates_km
    raise ValueError(f"Unsupported point attribute: {attribute}")


def _coordinate(point: Point3D, attribute: str) -> float:
    if attribute == "x_km":
        return point.x_km
    if attribute == "y_km":
        return point.y_km
    if attribute == "z_km":
        return point.z_km
    raise ValueError(f"Unsupported point attribute: {attribute}")


def _linear_mapper(
    start_pixel: float,
    end_pixel: float,
    min_value: float,
    max_value: float,
) -> Callable[[float], float]:
    span = max(max_value - min_value, 1.0e-12)
    scale = (end_pixel - start_pixel) / span
    return lambda value: start_pixel + (value - min_value) * scale


def _axis_mapper(
    attribute: str,
    top: float,
    height: float,
    min_value: float,
    max_value: float,
) -> Callable[[float], float]:
    if attribute == "z_km":
        return lambda value: (
            top + ((value - min_value) / max(max_value - min_value, 1.0e-12)) * height
        )
    return lambda value: (
        top + height - ((value - min_value) / max(max_value - min_value, 1.0e-12)) * height
    )


def _projected_pixels(
    points: tuple[Point3D, ...],
    attr_x: str,
    attr_y: str,
    map_x: Callable[[float], float],
    map_y: Callable[[float], float],
) -> tuple[tuple[float, float], ...]:
    projected: list[tuple[float, float]] = []
    for point in points:
        pixel = (
            round(map_x(_coordinate(point, attr_x)), 2),
            round(map_y(_coordinate(point, attr_y)), 2),
        )
        if projected and pixel == projected[-1]:
            continue
        projected.append(pixel)
    return tuple(projected)


def _cube_neighbor_projection(
    center_x: float,
    center_y: float,
    dx: int,
    dy: int,
    dz: int,
) -> tuple[float, float]:
    scale = 54.0
    depth_offset_x = 30.0
    depth_offset_y = 20.0
    return (
        center_x + scale * dx + depth_offset_x * dy,
        center_y - scale * dz + depth_offset_y * dy,
    )


def _connectivity_color(manhattan: int) -> str:
    if manhattan == 1:
        return "#3b82f6"
    if manhattan == 2:
        return "#f59e0b"
    return "#ef4444"


def _nearest_axis_index(coordinates: tuple[float, ...], value: float) -> int:
    return min(range(len(coordinates)), key=lambda index: abs(coordinates[index] - value))
