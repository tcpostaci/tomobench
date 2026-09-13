"""SVG figures for comparing Dijkstra and pseudo-bending ray paths."""

from __future__ import annotations

from pathlib import Path

from tomobench.domain.geometry import Point3D
from tomobench.domain.schemas import CartesianVelocityGrid3D


def save_ray_path_comparison_svg(
    grid: CartesianVelocityGrid3D,
    source: Point3D,
    receiver: Point3D,
    dijkstra_path: tuple[Point3D, ...],
    pseudo_bending_path: tuple[Point3D, ...],
    dijkstra_time_s: float,
    pseudo_bending_time_s: float,
    path: Path,
    *,
    title: str = "Dijkstra vs pseudo-bending ray paths",
    reference_label: str = "Dijkstra path",
    pseudo_label: str = "Pseudo-bending path",
) -> Path:
    """Save a two-panel x-y and x-z ray-path comparison figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 620
    panel_width, panel_height = 360.0, 360.0
    xy_left, xz_left, panel_top = 80.0, 530.0, 100.0

    x_min, x_max = grid.x_coordinates_km[0], grid.x_coordinates_km[-1]
    y_min, y_max = grid.y_coordinates_km[0], grid.y_coordinates_km[-1]
    z_min, z_max = grid.z_coordinates_km[0], grid.z_coordinates_km[-1]

    def sx(left: float, x_km: float) -> float:
        return left + (x_km - x_min) / (x_max - x_min) * panel_width

    def sy_xy(y_km: float) -> float:
        return panel_top + panel_height - (y_km - y_min) / (y_max - y_min) * panel_height

    def sy_xz(z_km: float) -> float:
        return panel_top + (z_km - z_min) / (z_max - z_min) * panel_height

    dijkstra_xy = _polyline_points(
        tuple((sx(xy_left, point.x_km), sy_xy(point.y_km)) for point in dijkstra_path)
    )
    pseudo_xy = _polyline_points(
        tuple((sx(xy_left, point.x_km), sy_xy(point.y_km)) for point in pseudo_bending_path)
    )
    straight_xy = _polyline_points(
        (
            (sx(xy_left, source.x_km), sy_xy(source.y_km)),
            (sx(xy_left, receiver.x_km), sy_xy(receiver.y_km)),
        )
    )
    dijkstra_xz = _polyline_points(
        tuple((sx(xz_left, point.x_km), sy_xz(point.z_km)) for point in dijkstra_path)
    )
    pseudo_xz = _polyline_points(
        tuple((sx(xz_left, point.x_km), sy_xz(point.z_km)) for point in pseudo_bending_path)
    )
    straight_xz = _polyline_points(
        (
            (sx(xz_left, source.x_km), sy_xz(source.z_km)),
            (sx(xz_left, receiver.x_km), sy_xz(receiver.z_km)),
        )
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="80" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">{title}</text>
  <text x="80" y="64" font-family="Arial, sans-serif" font-size="13" fill="#374151">Scenario: {grid.metadata.get("grid_scenario", "unknown")}; grid spacing: {grid.metadata.get("grid_spacing_km", "unknown")} km</text>
  {_panel(xy_left, panel_top, panel_width, panel_height, "Plan view (x-y)")}
  {_panel(xz_left, panel_top, panel_width, panel_height, "Cross-section (x-z)")}
  <polyline points="{straight_xy}" fill="none" stroke="#64748b" stroke-width="2" stroke-dasharray="5 5" />
  <polyline points="{dijkstra_xy}" fill="none" stroke="#ea580c" stroke-width="3" stroke-linejoin="round" />
  <polyline points="{pseudo_xy}" fill="none" stroke="#0f766e" stroke-width="3" stroke-linejoin="round" />
  <polyline points="{straight_xz}" fill="none" stroke="#64748b" stroke-width="2" stroke-dasharray="5 5" />
  <polyline points="{dijkstra_xz}" fill="none" stroke="#ea580c" stroke-width="3" stroke-linejoin="round" />
  <polyline points="{pseudo_xz}" fill="none" stroke="#0f766e" stroke-width="3" stroke-linejoin="round" />
  {_endpoint_marks(source, receiver, xy_left, xz_left, panel_top, panel_width, panel_height, x_min, x_max, y_min, y_max, z_min, z_max)}
  {_axis_labels(xy_left, panel_top, panel_width, panel_height, "x (km)", "y (km)", x_min, x_max, y_min, y_max, invert_y=True)}
  {_axis_labels(xz_left, panel_top, panel_width, panel_height, "x (km)", "depth z (km)", x_min, x_max, z_min, z_max, invert_y=False)}
  <line x1="104" y1="520" x2="154" y2="520" stroke="#64748b" stroke-width="2" stroke-dasharray="5 5" />
  <text x="166" y="525" font-family="Arial, sans-serif" font-size="13" fill="#111827">Straight initial ray</text>
  <line x1="330" y1="520" x2="380" y2="520" stroke="#ea580c" stroke-width="3" />
  <text x="392" y="525" font-family="Arial, sans-serif" font-size="13" fill="#111827">{reference_label} ({dijkstra_time_s:.4f} s)</text>
  <line x1="590" y1="520" x2="640" y2="520" stroke="#0f766e" stroke-width="3" />
  <text x="652" y="525" font-family="Arial, sans-serif" font-size="13" fill="#111827">{pseudo_label} ({pseudo_bending_time_s:.4f} s)</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def _panel(left: float, top: float, width: float, height: float, title: str) -> str:
    return (
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" '
        'fill="#f8fafc" stroke="#0f172a" stroke-width="1.2" />'
        f'<text x="{left:.2f}" y="{top - 16:.2f}" font-family="Arial, sans-serif" '
        f'font-size="15" font-weight="700" fill="#111827">{title}</text>'
    )


def _polyline_points(points: tuple[tuple[float, float], ...]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _endpoint_marks(
    source: Point3D,
    receiver: Point3D,
    xy_left: float,
    xz_left: float,
    top: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> str:
    def sx(left: float, x_km: float) -> float:
        return left + (x_km - x_min) / (x_max - x_min) * width

    def sy_xy(y_km: float) -> float:
        return top + height - (y_km - y_min) / (y_max - y_min) * height

    def sy_xz(z_km: float) -> float:
        return top + (z_km - z_min) / (z_max - z_min) * height

    marks = []
    for left, y_func, source_label, receiver_label in (
        (xy_left, sy_xy, "S", "R"),
        (xz_left, sy_xz, "S", "R"),
    ):
        for point, label, color in (
            (source, source_label, "#dc2626"),
            (receiver, receiver_label, "#2563eb"),
        ):
            marks.append(
                f'<circle cx="{sx(left, point.x_km):.2f}" cy="{y_func(point.y_km if left == xy_left else point.z_km):.2f}" '
                f'r="6" fill="{color}" stroke="#111827" stroke-width="1" />'
            )
            marks.append(
                f'<text x="{sx(left, point.x_km) + 9:.2f}" '
                f'y="{y_func(point.y_km if left == xy_left else point.z_km) + 4:.2f}" '
                'font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#111827">'
                f"{label}</text>"
            )
    return "".join(marks)


def _axis_labels(
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
    *,
    invert_y: bool,
) -> str:
    bottom_value = y_min if invert_y else y_max
    top_value = y_max if invert_y else y_min
    return (
        f'<text x="{left + width / 2 - 22:.2f}" y="{top + height + 34:.2f}" '
        f'font-family="Arial, sans-serif" font-size="13" fill="#111827">{x_label}</text>'
        f'<text x="{left - 46:.2f}" y="{top + height / 2:.2f}" transform="rotate(-90 {left - 46:.2f} {top + height / 2:.2f})" '
        f'font-family="Arial, sans-serif" font-size="13" fill="#111827">{y_label}</text>'
        f'<text x="{left:.2f}" y="{top + height + 16:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{x_min:g}</text>'
        f'<text x="{left + width - 26:.2f}" y="{top + height + 16:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{x_max:g}</text>'
        f'<text x="{left - 36:.2f}" y="{top + height + 4:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{bottom_value:g}</text>'
        f'<text x="{left - 30:.2f}" y="{top + 4:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{top_value:g}</text>'
    )
