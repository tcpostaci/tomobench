"""Synthetic P-wave velocity model generation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from tomobench.config.settings import BenchmarkSettings
from tomobench.domain.schemas import LayeredVelocityModel, VelocityLayer, dataclass_to_dict

SUPPORTED_VELOCITY_SCENARIOS: Final[tuple[str, ...]] = (
    "layered",
    "block_anomaly",
    "faulted",
    "salt_dome",
    "dyke_intrusion",
)


def generate_layered_velocity_model(settings: BenchmarkSettings) -> LayeredVelocityModel:
    """Generate the first simple 1D layered P-wave velocity model."""
    layered = settings.velocity_model_generation.layered
    layers = tuple(
        VelocityLayer(
            layer_id=f"LAYER{i + 1:02d}",
            top_depth_km=layered.depth_boundaries_km[i],
            bottom_depth_km=layered.depth_boundaries_km[i + 1],
            p_velocity_km_per_s=layered.velocities_km_per_s[i],
        )
        for i in range(layered.layer_count)
    )
    metadata = {
        "generator_type": "layered_1d_p_velocity_model",
        "seed": settings.random_seeds.velocity_models,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scenario": "layered",
        "layer_count": len(layers),
        "velocity_bounds_km_per_s": list(
            settings.velocity_model_generation.velocity_bounds_km_per_s
        ),
        "coordinate_system": settings.study_area.coordinate_system,
        "assumptions": [
            "Velocity varies only with depth in this first slice.",
            "Each layer has a constant isotropic P-wave velocity.",
            "This representation is intended as an interchangeable placeholder for later 3D models.",
        ],
    }
    return LayeredVelocityModel(
        model_id=settings.vertical_slice.velocity_model_id,
        layers=layers,
        metadata=metadata,
    )


def velocity_at_depth(model: LayeredVelocityModel, depth_km: float) -> float:
    """Return the layer velocity at the requested positive-downward depth."""
    for layer in model.layers:
        if layer.top_depth_km <= depth_km < layer.bottom_depth_km:
            return layer.p_velocity_km_per_s
    deepest = model.layers[-1]
    if depth_km == deepest.bottom_depth_km:
        return deepest.p_velocity_km_per_s
    raise ValueError(f"Depth {depth_km} km is outside the layered velocity model.")


def save_velocity_model(model: LayeredVelocityModel, path: Path) -> Path:
    """Save a layered velocity model as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclass_to_dict(model)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
