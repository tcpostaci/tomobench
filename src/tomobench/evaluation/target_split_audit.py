"""Audit target identity and split independence for case-level ML corpora.

The paper-facing corpus varies acquisition geometry around a smaller set of
geological targets.  This module makes that relationship explicit by hashing
the stored target vectors and comparing those identities with the persisted
case split used by the current ML suite.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tomobench.config import BenchmarkSettings, load_settings
from tomobench.datasets.target_identity import TARGET_HASH_SCHEMA, target_vector_sha256
from tomobench.utils.paths import get_repo_root


AUDIT_VERSION = "target_split_audit_v1"
DEFAULT_NUMERICAL_ATOL = 1.0e-12
DEFAULT_NUMERICAL_RTOL = 1.0e-12
DEFAULT_SECONDARY_NUMERICAL_ATOL = 1.0e-9
DEFAULT_SECONDARY_NUMERICAL_RTOL = 1.0e-9


@dataclass(frozen=True)
class TargetSplitAuditOutputs:
    """Paths written by the target/split audit."""

    output_dir: Path
    target_case_registry_csv: Path
    target_hash_summary_csv: Path
    cross_split_target_leakage_csv: Path
    numerical_equivalence_pairs_csv: Path
    audit_summary_md: Path
    audit_metadata_json: Path


def run_target_split_audit(
    *,
    manifest_path: Path | None = None,
    current_split_path: Path | None = None,
    output_dir: Path | None = None,
    settings: BenchmarkSettings | None = None,
    numerical_atol: float = DEFAULT_NUMERICAL_ATOL,
    numerical_rtol: float = DEFAULT_NUMERICAL_RTOL,
    secondary_numerical_atol: float = DEFAULT_SECONDARY_NUMERICAL_ATOL,
    secondary_numerical_rtol: float = DEFAULT_SECONDARY_NUMERICAL_RTOL,
) -> TargetSplitAuditOutputs:
    """Audit target duplication, numerical equivalence, and current split overlap.

    The current split is read from the persisted fixed-split feature matrix so
    the audit describes the generated evaluation artifact rather than
    reconstructing a possibly different split from source code.  Exact target
    identity is determined by SHA-256 over a canonical little-endian float64
    representation.  A pairwise ``numpy.allclose`` comparison supplements the
    exact hash check for arrays that differ only within the configured
    floating-point tolerances.
    """
    if any(value < 0.0 for value in (
        numerical_atol,
        numerical_rtol,
        secondary_numerical_atol,
        secondary_numerical_rtol,
    )):
        raise ValueError("Numerical tolerances must be non-negative.")

    loaded_settings = settings or load_settings()
    repo_root = get_repo_root()
    resolved_manifest_path = _resolve_input_path(
        manifest_path
        or loaded_settings.machine_learning.paper_pseudo_bending_observation_corpus.manifest_path,
        repo_root,
    )
    resolved_split_path = _resolve_input_path(
        current_split_path
        or Path(
            "outputs/generated/ml_baselines/"
            "paper_pseudo_bending_ml_suite_v1/models/pca-linear/fixed_split/feature_matrix.csv"
        ),
        repo_root,
    )
    resolved_output_dir = _resolve_input_path(
        output_dir or Path("outputs/generated/submission_readiness/target_split_audit_v1"),
        repo_root,
    )
    _validate_file(resolved_manifest_path, "manifest")
    _validate_file(resolved_split_path, "current split")

    manifest = _read_json_object(resolved_manifest_path)
    items = _manifest_items(manifest)
    split_by_case_id = _read_split_assignments(resolved_split_path)
    manifest_case_ids = {str(item.get("case_id", "")) for item in items}
    missing_splits = sorted(case_id for case_id in manifest_case_ids if case_id not in split_by_case_id)
    if missing_splits:
        raise ValueError(
            "Current split is missing manifest case IDs: " + ", ".join(missing_splits)
        )
    extra_splits = sorted(case_id for case_id in split_by_case_id if case_id not in manifest_case_ids)
    if extra_splits:
        raise ValueError(
            "Current split contains case IDs absent from the manifest: " + ", ".join(extra_splits)
        )

    records: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    for item in items:
        case_id = str(item.get("case_id", ""))
        if not case_id:
            raise ValueError("Every manifest item must define a non-empty case_id.")
        target_path = _resolve_manifest_path(
            item.get("target_velocity_grid_path")
            or item.get("velocity_model_grid_path")
            or item.get("target_grid_path"),
            repo_root,
            resolved_manifest_path,
        )
        _validate_file(target_path, f"target grid for {case_id}")
        target_payload = _read_json_object(target_path)
        target_array = _target_array(target_payload, case_id)
        target_shape = _target_shape(item, target_payload, target_array.size, case_id)
        target_hash = target_vector_sha256(target_array)
        metadata = _object_or_empty(target_payload.get("metadata"))
        model_definition_hash = _model_definition_sha256(
            item=item,
            target_payload=target_payload,
            target_shape=target_shape,
        )
        records.append(
            {
                "case_id": case_id,
                "family": str(item.get("family") or item.get("scenario") or ""),
                "scenario": str(item.get("scenario") or item.get("family") or ""),
                "parameter_variant_id": str(item.get("parameter_variant_id") or ""),
                "geometry_profile_id": str(item.get("geometry_profile_id") or ""),
                "station_generator_mode": str(item.get("station_generation_mode") or ""),
                "station_seed": _optional_int(item.get("station_seed")),
                "earthquake_seed": _optional_int(item.get("earthquake_seed")),
                "velocity_model_seed": _optional_int(item.get("velocity_model_seed")),
                "velocity_model_id": str(
                    target_payload.get("source_velocity_model_id")
                    or item.get("velocity_model_id")
                    or ""
                ),
                "velocity_model_path": _relative_or_absolute(
                    _resolve_manifest_path(item.get("velocity_model_path"), repo_root, resolved_manifest_path),
                    repo_root,
                ),
                "target_grid_id": str(target_payload.get("grid_id") or ""),
                "target_grid_path": _relative_or_absolute(target_path, repo_root),
                "target_shape": _shape_text(target_shape),
                "target_vector_length": int(target_array.size),
                "target_dtype": "<f8",
                "target_hash": target_hash,
                "model_definition_hash": model_definition_hash,
                "case_overrides_json": _canonical_json(item.get("case_overrides") or {}),
                "split": split_by_case_id[case_id],
                "_metadata": metadata,
            }
        )
        arrays.append(target_array)

    exact_groups = _group_indices(records, key="target_hash")
    numerical_pairs = _numerical_equivalence_pairs(
        records,
        arrays,
        numerical_atol=numerical_atol,
        numerical_rtol=numerical_rtol,
        secondary_numerical_atol=secondary_numerical_atol,
        secondary_numerical_rtol=secondary_numerical_rtol,
    )
    numerical_groups = _record_aware_numerical_groups(records, numerical_pairs)
    exact_group_id_by_index = _deterministic_group_ids(exact_groups, records, prefix="exact_target")
    numerical_group_id_by_index = _deterministic_group_ids(
        numerical_groups,
        records,
        prefix="target_group",
    )

    _add_group_fields(
        records,
        exact_groups=exact_groups,
        numerical_groups=numerical_groups,
        exact_group_id_by_index=exact_group_id_by_index,
        numerical_group_id_by_index=numerical_group_id_by_index,
    )

    output_dir_path = resolved_output_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir_path / "target_case_registry.csv"
    hash_summary_path = output_dir_path / "target_hash_summary.csv"
    leakage_path = output_dir_path / "cross_split_target_leakage.csv"
    numerical_pairs_path = output_dir_path / "numerical_equivalence_pairs.csv"
    summary_path = output_dir_path / "audit_summary.md"
    metadata_path = output_dir_path / "audit_metadata.json"

    _write_registry(records, registry_path)
    _write_hash_summary(records, exact_groups, hash_summary_path)
    _write_numerical_pairs(numerical_pairs, numerical_pairs_path)
    _write_cross_split_leakage(records, exact_groups, numerical_groups, leakage_path)

    metadata_payload = _metadata_payload(
        settings=loaded_settings,
        manifest=manifest,
        manifest_path=resolved_manifest_path,
        current_split_path=resolved_split_path,
        output_dir=output_dir_path,
        repo_root=repo_root,
        numerical_atol=numerical_atol,
        numerical_rtol=numerical_rtol,
        secondary_numerical_atol=secondary_numerical_atol,
        secondary_numerical_rtol=secondary_numerical_rtol,
    )
    _write_json(metadata_payload, metadata_path)
    _write_summary(
        records=records,
        exact_groups=exact_groups,
        numerical_groups=numerical_groups,
        numerical_pairs=numerical_pairs,
        manifest=manifest,
        manifest_path=resolved_manifest_path,
        current_split_path=resolved_split_path,
        summary_path=summary_path,
        metadata=metadata_payload,
    )
    return TargetSplitAuditOutputs(
        output_dir=output_dir_path,
        target_case_registry_csv=registry_path,
        target_hash_summary_csv=hash_summary_path,
        cross_split_target_leakage_csv=leakage_path,
        numerical_equivalence_pairs_csv=numerical_pairs_path,
        audit_summary_md=summary_path,
        audit_metadata_json=metadata_path,
    )


def _resolve_input_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _resolve_manifest_path(value: object, repo_root: Path, manifest_path: Path) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Manifest item does not define a required path: {manifest_path}")
    path = Path(str(value))
    if path.is_absolute():
        return path
    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return repo_candidate
    return manifest_path.parent / path


def _validate_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label.capitalize()} file does not exist: {path}")


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return dict(payload)


def _manifest_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = manifest.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Manifest must contain a non-empty items list.")
    items: list[dict[str, Any]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise TypeError("Manifest items must be JSON objects.")
        items.append(dict(raw_item))
    return items


def _read_split_assignments(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Current split file is empty: {path}")
    assignments: dict[str, str] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        split = str(row.get("split") or "")
        if not case_id or not split:
            raise ValueError("Current split file must contain non-empty case_id and split columns.")
        if case_id in assignments:
            raise ValueError(f"Current split file contains duplicate case_id: {case_id}")
        assignments[case_id] = split
    return assignments


def _target_array(target_payload: dict[str, Any], case_id: str) -> np.ndarray:
    values = target_payload.get("p_velocity_km_per_s")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{case_id}: target grid does not contain a non-empty velocity vector.")
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1:
        raise ValueError(f"{case_id}: target velocity values must be a one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{case_id}: target velocity vector contains non-finite values.")
    return np.ascontiguousarray(array, dtype="<f8")


def _target_shape(
    item: dict[str, Any],
    target_payload: dict[str, Any],
    vector_length: int,
    case_id: str,
) -> tuple[int, ...]:
    metadata = _object_or_empty(target_payload.get("metadata"))
    raw_shape = metadata.get("shape") or item.get("grid_shape")
    if not isinstance(raw_shape, dict):
        return (vector_length,)
    axis_names = ("nx", "ny", "nz")
    try:
        shape = tuple(int(raw_shape[name]) for name in axis_names)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{case_id}: target grid shape is not a valid nx/ny/nz mapping.") from error
    if any(axis <= 0 for axis in shape) or int(np.prod(shape)) != vector_length:
        raise ValueError(
            f"{case_id}: target shape {_shape_text(shape)} does not match vector length {vector_length}."
        )
    return shape


def _model_definition_sha256(
    *,
    item: dict[str, Any],
    target_payload: dict[str, Any],
    target_shape: tuple[int, ...],
) -> str:
    metadata = _object_or_empty(target_payload.get("metadata"))
    definition = {
        "family": item.get("family") or item.get("scenario"),
        "scenario": item.get("scenario") or item.get("family"),
        "grid_shape": list(target_shape),
        "grid_spacing_km": metadata.get("grid_spacing_km"),
        "coordinates_km": {
            "x": target_payload.get("x_coordinates_km"),
            "y": target_payload.get("y_coordinates_km"),
            "z": target_payload.get("z_coordinates_km"),
        },
        "layered_model": metadata.get("layered_model"),
        "scenario_definition": metadata.get(
            str(item.get("scenario") or item.get("family") or "")
        ),
    }
    encoded = _canonical_json(definition).encode("utf-8")
    return hashlib.sha256(b"velocity_model_definition_sha256_v1\0" + encoded).hexdigest()


def _group_indices(records: list[dict[str, Any]], *, key: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record[key])].append(index)
    return dict(groups)


def _numerical_equivalence_pairs(
    records: list[dict[str, Any]],
    arrays: list[np.ndarray],
    *,
    numerical_atol: float,
    numerical_rtol: float,
    secondary_numerical_atol: float,
    secondary_numerical_rtol: float,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_index in range(len(arrays)):
        for right_index in range(left_index + 1, len(arrays)):
            left = arrays[left_index]
            right = arrays[right_index]
            if left.shape != right.shape:
                continue
            difference = np.abs(left - right)
            max_abs = float(np.max(difference))
            denominator = np.maximum(
                np.maximum(np.abs(left), np.abs(right)),
                np.finfo(np.float64).tiny,
            )
            max_relative = float(np.max(difference / denominator))
            primary = bool(
                np.allclose(
                    left,
                    right,
                    atol=numerical_atol,
                    rtol=numerical_rtol,
                )
            )
            secondary = bool(
                np.allclose(
                    left,
                    right,
                    atol=secondary_numerical_atol,
                    rtol=secondary_numerical_rtol,
                )
            )
            if primary or secondary:
                pairs.append(
                    {
                        "case_id_a": records[left_index]["case_id"],
                        "case_id_b": records[right_index]["case_id"],
                        "target_hash_a": records[left_index]["target_hash"],
                        "target_hash_b": records[right_index]["target_hash"],
                        "exact_hash_match": records[left_index]["target_hash"]
                        == records[right_index]["target_hash"],
                        "primary_allclose": primary,
                        "secondary_allclose": secondary,
                        "max_abs_difference": max_abs,
                        "max_relative_difference": max_relative,
                    }
                )
    return pairs


def _record_aware_numerical_groups(
    records: list[dict[str, Any]],
    numerical_pairs: list[dict[str, Any]],
) -> dict[str, list[int]]:
    index_by_case_id = {str(record["case_id"]): index for index, record in enumerate(records)}
    parent = list(range(len(records)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for pair in numerical_pairs:
        if not pair["primary_allclose"]:
            continue
        union(index_by_case_id[str(pair["case_id_a"])], index_by_case_id[str(pair["case_id_b"])])
    groups: dict[str, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[str(find(index))].append(index)
    return dict(groups)


def _deterministic_group_ids(
    groups: dict[str, list[int]],
    records: list[dict[str, Any]],
    *,
    prefix: str,
) -> dict[int, str]:
    ordered_groups = sorted(
        groups.values(),
        key=lambda indices: min(str(records[index]["case_id"]) for index in indices),
    )
    result: dict[int, str] = {}
    for group_number, indices in enumerate(ordered_groups, start=1):
        group_id = f"{prefix}_{group_number:03d}"
        for index in indices:
            result[index] = group_id
    return result


def _add_group_fields(
    records: list[dict[str, Any]],
    *,
    exact_groups: dict[str, list[int]],
    numerical_groups: dict[str, list[int]],
    exact_group_id_by_index: dict[int, str],
    numerical_group_id_by_index: dict[int, str],
) -> None:
    exact_group_by_index = {
        index: indices for indices in exact_groups.values() for index in indices
    }
    numerical_group_by_index = {
        index: indices for indices in numerical_groups.values() for index in indices
    }
    for index, record in enumerate(records):
        exact_indices = exact_group_by_index[index]
        numeric_indices = numerical_group_by_index[index]
        exact_splits = sorted({str(records[item]["split"]) for item in exact_indices})
        numeric_splits = sorted({str(records[item]["split"]) for item in numeric_indices})
        record.update(
            {
                "exact_group_id": exact_group_id_by_index[index],
                "exact_group_size": len(exact_indices),
                "exact_group_split_names": exact_splits,
                "exact_group_split_count": len(exact_splits),
                "exact_group_cross_split": len(exact_splits) > 1,
                "numerical_group_id": numerical_group_id_by_index[index],
                "numerical_group_size": len(numeric_indices),
                "numerical_group_split_names": numeric_splits,
                "numerical_group_split_count": len(numeric_splits),
                "numerical_group_cross_split": len(numeric_splits) > 1,
            }
        )


def _write_registry(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "case_id",
        "family",
        "scenario",
        "parameter_variant_id",
        "geometry_profile_id",
        "station_generator_mode",
        "station_seed",
        "earthquake_seed",
        "velocity_model_seed",
        "velocity_model_id",
        "velocity_model_path",
        "target_grid_id",
        "target_grid_path",
        "target_shape",
        "target_vector_length",
        "target_dtype",
        "target_hash",
        "model_definition_hash",
        "case_overrides_json",
        "split",
        "exact_group_id",
        "exact_group_size",
        "exact_group_split_names",
        "exact_group_split_count",
        "exact_group_cross_split",
        "numerical_group_id",
        "numerical_group_size",
        "numerical_group_split_names",
        "numerical_group_split_count",
        "numerical_group_cross_split",
    ]
    rows = []
    for record in sorted(records, key=lambda item: str(item["case_id"])):
        row = {name: record[name] for name in fieldnames}
        row["exact_group_split_names"] = ";".join(record["exact_group_split_names"])
        row["numerical_group_split_names"] = ";".join(record["numerical_group_split_names"])
        rows.append(row)
    _write_csv(rows, fieldnames, path)


def _write_hash_summary(
    records: list[dict[str, Any]],
    exact_groups: dict[str, list[int]],
    path: Path,
) -> None:
    fieldnames = [
        "target_hash",
        "target_group_id",
        "target_count",
        "target_shape",
        "target_vector_length",
        "families",
        "parameter_variant_ids",
        "geometry_profile_ids",
        "case_ids",
        "split_names",
        "split_count",
        "cross_split",
        "model_definition_hashes",
        "velocity_model_ids",
        "max_within_hash_abs_difference",
        "max_within_hash_relative_difference",
    ]
    rows: list[dict[str, Any]] = []
    for target_hash, indices in sorted(exact_groups.items()):
        first = records[indices[0]]
        splits = sorted({str(records[index]["split"]) for index in indices})
        rows.append(
            {
                "target_hash": target_hash,
                "target_group_id": first["numerical_group_id"],
                "target_count": len(indices),
                "target_shape": first["target_shape"],
                "target_vector_length": first["target_vector_length"],
                "families": ";".join(sorted({str(records[index]["family"]) for index in indices})),
                "parameter_variant_ids": ";".join(
                    sorted({str(records[index]["parameter_variant_id"]) for index in indices})
                ),
                "geometry_profile_ids": ";".join(
                    sorted({str(records[index]["geometry_profile_id"]) for index in indices})
                ),
                "case_ids": ";".join(sorted(str(records[index]["case_id"]) for index in indices)),
                "split_names": ";".join(splits),
                "split_count": len(splits),
                "cross_split": len(splits) > 1,
                "model_definition_hashes": ";".join(
                    sorted({str(records[index]["model_definition_hash"]) for index in indices})
                ),
                "velocity_model_ids": ";".join(
                    sorted({str(records[index]["velocity_model_id"]) for index in indices})
                ),
                "max_within_hash_abs_difference": 0.0,
                "max_within_hash_relative_difference": 0.0,
            }
        )
    _write_csv(rows, fieldnames, path)


def _write_cross_split_leakage(
    records: list[dict[str, Any]],
    exact_groups: dict[str, list[int]],
    numerical_groups: dict[str, list[int]],
    path: Path,
) -> None:
    del exact_groups, numerical_groups
    fieldnames = [
        "target_hash",
        "target_group_id",
        "equivalence_basis",
        "case_id",
        "family",
        "parameter_variant_id",
        "geometry_profile_id",
        "station_generator_mode",
        "station_seed",
        "earthquake_seed",
        "split",
        "split_names",
        "split_count",
        "target_group_case_count",
        "target_shape",
        "target_vector_length",
        "target_grid_path",
        "velocity_model_id",
        "velocity_model_path",
    ]
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (str(item["target_hash"]), str(item["case_id"]))):
        exact_cross = bool(record["exact_group_cross_split"])
        numeric_cross = bool(record["numerical_group_cross_split"])
        if not exact_cross and not numeric_cross:
            continue
        if exact_cross and numeric_cross:
            basis = "exact_sha256_and_primary_allclose"
        elif exact_cross:
            basis = "exact_sha256"
        else:
            basis = "primary_allclose"
        rows.append(
            {
                "target_hash": record["target_hash"],
                "target_group_id": record["numerical_group_id"],
                "equivalence_basis": basis,
                "case_id": record["case_id"],
                "family": record["family"],
                "parameter_variant_id": record["parameter_variant_id"],
                "geometry_profile_id": record["geometry_profile_id"],
                "station_generator_mode": record["station_generator_mode"],
                "station_seed": record["station_seed"],
                "earthquake_seed": record["earthquake_seed"],
                "split": record["split"],
                "split_names": ";".join(record["numerical_group_split_names"]),
                "split_count": record["numerical_group_split_count"],
                "target_group_case_count": record["numerical_group_size"],
                "target_shape": record["target_shape"],
                "target_vector_length": record["target_vector_length"],
                "target_grid_path": record["target_grid_path"],
                "velocity_model_id": record["velocity_model_id"],
                "velocity_model_path": record["velocity_model_path"],
            }
        )
    _write_csv(rows, fieldnames, path)


def _write_numerical_pairs(pairs: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "case_id_a",
        "case_id_b",
        "target_hash_a",
        "target_hash_b",
        "exact_hash_match",
        "primary_allclose",
        "secondary_allclose",
        "max_abs_difference",
        "max_relative_difference",
    ]
    _write_csv(sorted(pairs, key=lambda item: (item["case_id_a"], item["case_id_b"])), fieldnames, path)


def _write_csv(rows: list[dict[str, Any]], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metadata_payload(
    *,
    settings: BenchmarkSettings,
    manifest: dict[str, Any],
    manifest_path: Path,
    current_split_path: Path,
    output_dir: Path,
    repo_root: Path,
    numerical_atol: float,
    numerical_rtol: float,
    secondary_numerical_atol: float,
    secondary_numerical_rtol: float,
) -> dict[str, Any]:
    return {
        "artifact_type": "target_split_audit_metadata",
        "audit_version": AUDIT_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": "uv run tomobench audit-target-split",
        "input_manifest_path": _relative_or_absolute(manifest_path, repo_root),
        "current_split_path": _relative_or_absolute(current_split_path, repo_root),
        "config_snapshot_path": "config/benchmark_config.yaml",
        "output_directory": _relative_or_absolute(output_dir, repo_root),
        "input_sha256": {
            "manifest": _file_sha256(manifest_path),
            "current_split": _file_sha256(current_split_path),
            "config": _file_sha256(repo_root / "config" / "benchmark_config.yaml"),
        },
        "corpus": {
            "corpus_id": manifest.get("corpus_id") or manifest.get("batch_id"),
            "declared_case_count": manifest.get("case_count"),
            "observed_case_count": len(manifest.get("items", [])),
            "simulator_name": manifest.get("simulator_name"),
        },
        "current_split": {
            "source_artifact": "persisted fixed-split pca-linear feature matrix",
            "strategy": "family_stratified",
            "seed": settings.random_seeds.machine_learning,
            "configured_fractions": {
                "train": settings.machine_learning.train_fraction,
                "validation": settings.machine_learning.validation_fraction,
                "test": settings.machine_learning.test_fraction,
            },
        },
        "hashing": {
            "algorithm": "SHA-256",
            "schema": TARGET_HASH_SCHEMA,
            "canonical_representation": (
                "ASCII schema prefix, NUL separator, dtype=<f8, NUL separator, "
                "ASCII vector length, NUL separator, then contiguous little-endian float64 bytes"
            ),
            "array_field": "p_velocity_km_per_s",
        },
        "numerical_equivalence": {
            "primary_atol": numerical_atol,
            "primary_rtol": numerical_rtol,
            "secondary_atol": secondary_numerical_atol,
            "secondary_rtol": secondary_numerical_rtol,
            "comparison": "numpy.allclose with pairwise max absolute and symmetric max relative differences",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }


def _write_summary(
    *,
    records: list[dict[str, Any]],
    exact_groups: dict[str, list[int]],
    numerical_groups: dict[str, list[int]],
    numerical_pairs: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_path: Path,
    current_split_path: Path,
    summary_path: Path,
    metadata: dict[str, Any],
) -> None:
    exact_cross_groups = {
        key: indices
        for key, indices in exact_groups.items()
        if len({str(records[index]["split"]) for index in indices}) > 1
    }
    numerical_cross_groups = {
        key: indices
        for key, indices in numerical_groups.items()
        if len({str(records[index]["split"]) for index in indices}) > 1
    }
    affected_exact = {
        index
        for indices in exact_cross_groups.values()
        for index in indices
    }
    affected_numerical = {
        index
        for indices in numerical_cross_groups.values()
        for index in indices
    }
    split_counts = Counter(str(record["split"]) for record in records)
    split_unique_hashes = {
        split: len({str(record["target_hash"]) for record in records if record["split"] == split})
        for split in ("train", "validation", "test")
    }
    hash_multiplicities = Counter(len(indices) for indices in exact_groups.values())
    family_rows = []
    for family in sorted({str(record["family"]) for record in records}):
        family_records = [record for record in records if record["family"] == family]
        family_hashes = {str(record["target_hash"]) for record in family_records}
        family_exact_cross = {
            target_hash
            for target_hash in family_hashes
            if target_hash in exact_cross_groups
        }
        family_affected = sum(1 for record in family_records if record["exact_group_cross_split"])
        family_rows.append(
            "| {family} | {cases} | {hashes} | {affected} | {cross} | {splits} |".format(
                family=family,
                cases=len(family_records),
                hashes=len(family_hashes),
                affected=family_affected,
                cross=len(family_exact_cross),
                splits="; ".join(
                    f"{split}={sum(1 for record in family_records if record['split'] == split)}"
                    for split in ("train", "validation", "test")
                ),
            )
        )
    leakage_rows = []
    for target_hash, indices in sorted(exact_cross_groups.items()):
        leakage_rows.append(
            "| `{hash}` | {count} | {families} | {variants} | {splits} | {cases} |".format(
                hash=target_hash,
                count=len(indices),
                families="; ".join(sorted({str(records[index]['family']) for index in indices})),
                variants="; ".join(
                    sorted({str(records[index]['parameter_variant_id']) for index in indices})
                ),
                splits="; ".join(sorted({str(records[index]['split']) for index in indices})),
                cases="; ".join(sorted(str(records[index]["case_id"]) for index in indices)),
            )
        )
    multiplicity_text = ", ".join(
        f"{count} hash group(s) occur {multiplicity} time(s)"
        for multiplicity, count in sorted(hash_multiplicities.items())
    )
    lines = [
        "# Target/split audit",
        "",
        f"- Audit version: `{AUDIT_VERSION}`",
        f"- Manifest: `{_relative_or_absolute(manifest_path, get_repo_root())}`",
        f"- Current split artifact: `{_relative_or_absolute(current_split_path, get_repo_root())}`",
        f"- Manifest corpus ID: `{manifest.get('corpus_id') or manifest.get('batch_id')}`",
        "",
        "## Verdict",
        "",
    ]
    if exact_cross_groups or numerical_cross_groups:
        lines.extend(
            [
                "**CRITICAL: target-level split leakage is confirmed.** At least one exact or numerically equivalent target appears in more than one current ML split. The existing case-level headline evaluation must not be treated as an independent-target test until a grouped split is defined and the dependent analyses are regenerated.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No exact or tolerance-level target overlap crosses the current train/validation/test boundaries in this audit.",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary counts",
            "",
            f"- Total manifest cases: **{len(records)}** (declared: `{manifest.get('case_count')}`).",
            f"- Unique exact SHA-256 target hashes: **{len(exact_groups)}**.",
            f"- Unique primary tolerance groups: **{len(numerical_groups)}**.",
            f"- Target hashes appearing once: **{sum(1 for indices in exact_groups.values() if len(indices) == 1)}**.",
            f"- Target hashes appearing multiple times: **{sum(1 for indices in exact_groups.values() if len(indices) > 1)}**.",
            f"- Multiplicity distribution: {multiplicity_text or 'none'}.",
            f"- Exact target hashes crossing split boundaries: **{len(exact_cross_groups)}**.",
            f"- Primary tolerance groups crossing split boundaries: **{len(numerical_cross_groups)}**.",
            f"- Cases affected by exact target overlap: **{len(affected_exact)}**.",
            f"- Cases affected by exact or primary tolerance overlap: **{len(affected_exact | affected_numerical)}**.",
            "",
            "## Current case split",
            "",
            f"- Case counts: train={split_counts.get('train', 0)}, validation={split_counts.get('validation', 0)}, test={split_counts.get('test', 0)}.",
            f"- Distinct exact target hashes represented in each split: train={split_unique_hashes['train']}, validation={split_unique_hashes['validation']}, test={split_unique_hashes['test']}. These are representations, not independent target assignments, because groups cross boundaries.",
            "",
            "## Interpretation of identity and acquisition geometry",
            "",
            "- The corpus manifest identifies a case by geological family/parameter variant crossed with an acquisition geometry profile. The geometry profile changes station and earthquake generation metadata; it is not part of the stored target velocity vector.",
            "- This audit treats the stored `p_velocity_km_per_s` vector as the definitive target identity for leakage detection. It also records the manifest parameter variant, geometry profile, station seed, earthquake seed, velocity-model ID, and target-grid metadata for traceability.",
            "- The nominal five-family × four-variant × five-profile construction does not imply 100 independent velocity targets. The stored vectors resolve to the counts above; the nominal salt-dome reference and shallow cases are among the duplicate target groups when their discretized arrays are identical.",
            "",
            "## Family counts",
            "",
            "| Family | Cases | Unique exact targets | Affected cases | Crossing exact targets | Case split counts |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            *family_rows,
            "",
            "## Exact target groups crossing splits",
            "",
        ]
    )
    if leakage_rows:
        lines.extend(
            [
                "| Target hash | Cases in group | Families | Parameter variants | Splits | Case IDs |",
                "| --- | ---: | --- | --- | --- | --- |",
                *leakage_rows,
                "",
            ]
        )
    else:
        lines.extend(["None.", ""])
    lines.extend(
        [
            "## Numerical-tolerance check",
            "",
            f"- Primary comparison: `numpy.allclose(atol={metadata['numerical_equivalence']['primary_atol']}, rtol={metadata['numerical_equivalence']['primary_rtol']})`.",
            f"- Secondary sensitivity comparison: `numpy.allclose(atol={metadata['numerical_equivalence']['secondary_atol']}, rtol={metadata['numerical_equivalence']['secondary_rtol']})`.",
            f"- Pair records written: **{len(numerical_pairs)}**. See `numerical_equivalence_pairs.csv` for maximum absolute and relative differences.",
            "",
            "## Reproducibility",
            "",
            "- Metadata: `audit_metadata.json` (input file hashes, canonical hash schema, tolerances, command, and environment).",
            "- Case registry: `target_case_registry.csv`.",
            "- Exact hash summary: `target_hash_summary.csv`.",
            "- Cross-split leakage records: `cross_split_target_leakage.csv`.",
            "",
            "## Required next step",
            "",
            "Because target groups cross the current split, define a deterministic grouped split keyed by the underlying target identity (at minimum the exact target hash, with the tolerance result considered) before rerunning or interpreting the primary ML evaluation. Preserve the current artifacts as the pre-audit comparison and do not overwrite them.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _object_or_empty(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(axis) for axis in shape)


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)
