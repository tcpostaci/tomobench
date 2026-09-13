"""Visualization scaffolding for geometry and experiment outputs."""

from tomobench.visualization.plots import (
    plot_types,
    save_geometry_svg,
    save_velocity_grid_slices_svg,
    save_velocity_profile_svg,
)
from tomobench.visualization.target_grid_structure import (
    build_target_grid_geometry,
    save_target_grid_structure_figure,
)
from tomobench.visualization.velocity_grid_section import save_velocity_grid_section_figure

__all__ = [
    "build_target_grid_geometry",
    "plot_types",
    "save_geometry_svg",
    "save_target_grid_structure_figure",
    "save_velocity_grid_section_figure",
    "save_velocity_grid_slices_svg",
    "save_velocity_profile_svg",
]
