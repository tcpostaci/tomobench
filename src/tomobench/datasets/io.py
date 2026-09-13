"""Dataset export and persistence helpers."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tomobench.config.settings import BenchmarkSettings
from tomobench.datasets.builder import dataset_columns

SUPPORTED_EXPORT_FORMATS = ("csv", "parquet")


def export_dataset_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Export source-receiver dataset rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset_columns())
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_dataset_metadata(
    rows: list[dict[str, Any]],
    path: Path,
    settings: BenchmarkSettings,
    source_files: dict[str, str],
) -> Path:
    """Save metadata for a generated dataset export."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": settings.vertical_slice.dataset_id,
        "schema_version": settings.dataset_export.schema_version,
        "record_level": "source_receiver_pair",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "row_count": len(rows),
        "columns": list(dataset_columns()),
        "simulation_method": settings.simulation.method,
        "target_representation": settings.machine_learning.target_representation,
        "target_value_semantics": settings.machine_learning.target_grid.value_semantics,
        "target_flattening_order": settings.machine_learning.target_grid.flattening_order,
        "source_files": source_files,
        "export_formats": ["csv"],
        "parquet_status": (
            "Not written in this first slice; CSV is the mandatory reviewable export."
        ),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
