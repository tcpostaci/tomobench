"""Structure-aware reconstruction metrics for the synthetic target families."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tomobench.evaluation.metrics import mean_absolute_error, root_mean_squared_error
from tomobench.tomography.comparison import node_centered_values_to_cell_centered
from tomobench.utils.paths import get_repo_root


DEFAULT_STRUCTURAL_MANIFEST = Path(
    "outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json"
)
DEFAULT_STRUCTURAL_SPLIT = Path(
    "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/target_group_split_assignments.csv"
)
DEFAULT_STRUCTURAL_ML_METRICS = Path(
    "outputs/generated/ml_baselines/phase9_grouped_split_ablation_v1/cell_centered_metrics.csv"
)
DEFAULT_STRUCTURAL_CLASSICAL_DIR = Path(
    "outputs/generated/classical_tomography/fair_reference_ray_corpus_v1"
)
DEFAULT_STRUCTURAL_OUTPUT_DIR = Path(
    "outputs/generated/submission_readiness/structural_recovery_metrics_v1"
)
DEFAULT_STRUCTURAL_METHODS = (
    "reference_prior",
    "optimized_fixed_ray",
    "full_input_euclidean_path",
    "full_input",
    "travel_time_only",
)


class StructuralRecoveryOutputs:
    """Paths written by the structure-aware metric audit."""

    def __init__(
        self,
        output_dir: Path,
        case_metrics_csv: Path,
        layer_metrics_csv: Path,
        summary_csv: Path,
        summary_json: Path,
        summary_md: Path,
        metadata_json: Path,
    ) -> None:
        self.output_dir = output_dir
        self.case_metrics_csv = case_metrics_csv
        self.layer_metrics_csv = layer_metrics_csv
        self.summary_csv = summary_csv
        self.summary_json = summary_json
        self.summary_md = summary_md
        self.metadata_json = metadata_json


def run_structural_recovery_metrics(
    *,
    manifest_path: Path = DEFAULT_STRUCTURAL_MANIFEST,
    grouped_split_path: Path = DEFAULT_STRUCTURAL_SPLIT,
    ml_metrics_path: Path = DEFAULT_STRUCTURAL_ML_METRICS,
    classical_corpus_dir: Path = DEFAULT_STRUCTURAL_CLASSICAL_DIR,
    output_dir: Path = DEFAULT_STRUCTURAL_OUTPUT_DIR,
    method_ids: Sequence[str] = DEFAULT_STRUCTURAL_METHODS,
) -> StructuralRecoveryOutputs:
    """Evaluate objectively defined synthetic structures on the fixed test cases.

    Structure masks are derived from the saved generator metadata and the
    target-versus-layered-background cell difference.  No predicted mask or
    test-set threshold is used to define the true structure.
    """
    methods = tuple(dict.fromkeys(str(value) for value in method_ids))
    if not methods:
        raise ValueError("method_ids must contain at least one method.")
    repo_root = get_repo_root()
    manifest_resolved = _resolve_path(Path(manifest_path), repo_root)
    split_resolved = _resolve_path(Path(grouped_split_path), repo_root)
    ml_metrics_resolved = _resolve_path(Path(ml_metrics_path), repo_root)
    classical_resolved = _resolve_path(Path(classical_corpus_dir), repo_root)
    output_resolved = _resolve_path(Path(output_dir), repo_root)
    manifest = _read_json(manifest_resolved)
    split_by_case, target_hash_by_case = _read_grouped_split(split_resolved)
    ml_by_variant_case = _index_ml_rows(_read_csv(ml_metrics_resolved), methods)
    test_items = [
        item
        for item in _manifest_items(manifest)
        if split_by_case.get(str(item["case_id"])) == "test"
    ]
    if not test_items:
        raise ValueError("The grouped split contains no test cases.")

    case_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for item in sorted(test_items, key=lambda value: str(value["case_id"])):
        case_id = str(item["case_id"])
        family = str(item.get("family") or item.get("scenario") or "unknown")
        target_grid_path = _resolve_manifest_path(
            item.get("target_velocity_grid_path") or item.get("target_grid_path"),
            repo_root,
        )
        target_grid = _read_json(target_grid_path)
        nx = len(target_grid["x_coordinates_km"])
        ny = len(target_grid["y_coordinates_km"])
        nz = len(target_grid["z_coordinates_km"])
        target_cells = np.asarray(
            node_centered_values_to_cell_centered(
                target_grid["p_velocity_km_per_s"], nx, ny, nz
            ),
            dtype=float,
        )
        base_cells = _base_layered_cell_values(target_grid)
        structure_mask = np.abs(target_cells - base_cells) > 1.0e-12
        masks = _family_masks(target_grid, family, structure_mask)
        methods_by_case = _load_case_predictions(
            case_id=case_id,
            target_grid=target_grid,
            expected_cell_count=target_cells.size,
            methods=methods,
            ml_by_variant_case=ml_by_variant_case,
            classical_dir=classical_resolved,
            repo_root=repo_root,
        )

        for method in methods:
            prediction = methods_by_case[method]
            for scope, mask_map in masks.items():
                for mask_label, mask in mask_map.items():
                    metrics = _masked_metrics(target_cells, prediction, mask)
                    case_rows.append(
                        {
                            "case_id": case_id,
                            "family": family,
                            "target_hash": target_hash_by_case[case_id],
                            "method_id": method,
                            "metric_scope": scope,
                            "mask_label": mask_label,
                            "cell_count": int(np.count_nonzero(mask)),
                            "all_cell_count": int(target_cells.size),
                            "rmse_km_per_s": metrics["rmse"],
                            "mae_km_per_s": metrics["mae"],
                            "bias_km_per_s": metrics["bias"],
                            "true_mean_km_per_s": metrics["true_mean"],
                            "predicted_mean_km_per_s": metrics["predicted_mean"],
                            "true_contrast_km_per_s": None,
                            "predicted_contrast_km_per_s": None,
                            "contrast_error_km_per_s": None,
                        }
                    )
            if family in {"block_anomaly", "salt_dome", "dyke_intrusion"}:
                contrast = _contrast_metrics(
                    target_cells,
                    prediction,
                    masks["structure"]["structure_body"],
                    masks["structure"]["background"],
                )
                case_rows.append(
                    {
                        "case_id": case_id,
                        "family": family,
                        "target_hash": target_hash_by_case[case_id],
                        "method_id": method,
                        "metric_scope": "structure_contrast",
                        "mask_label": "body_minus_background",
                        "cell_count": int(
                            np.count_nonzero(masks["structure"]["structure_body"])
                        ),
                        "all_cell_count": int(target_cells.size),
                        "rmse_km_per_s": None,
                        "mae_km_per_s": None,
                        "bias_km_per_s": None,
                        "true_mean_km_per_s": None,
                        "predicted_mean_km_per_s": None,
                        "true_contrast_km_per_s": contrast["true_contrast"],
                        "predicted_contrast_km_per_s": contrast["predicted_contrast"],
                        "contrast_error_km_per_s": contrast["contrast_error"],
                    }
                )
            if family == "faulted":
                contrast = _contrast_metrics(
                    target_cells,
                    prediction,
                    masks["fault"]["positive_side"],
                    masks["fault"]["negative_side"],
                )
                case_rows.append(
                    {
                        "case_id": case_id,
                        "family": family,
                        "target_hash": target_hash_by_case[case_id],
                        "method_id": method,
                        "metric_scope": "fault_contrast",
                        "mask_label": "positive_minus_negative_side",
                        "cell_count": int(np.count_nonzero(masks["fault"]["positive_side"])),
                        "all_cell_count": int(target_cells.size),
                        "rmse_km_per_s": None,
                        "mae_km_per_s": None,
                        "bias_km_per_s": None,
                        "true_mean_km_per_s": None,
                        "predicted_mean_km_per_s": None,
                        "true_contrast_km_per_s": contrast["true_contrast"],
                        "predicted_contrast_km_per_s": contrast["predicted_contrast"],
                        "contrast_error_km_per_s": contrast["contrast_error"],
                    }
                )
            if family == "layered":
                layer_rows.extend(
                    _layer_metric_rows(
                        case_id=case_id,
                        family=family,
                        target_hash=target_hash_by_case[case_id],
                        methods_by_case={method: prediction},
                        target_cells=target_cells,
                        z_coordinates=target_grid["z_coordinates_km"],
                        layered_metadata=target_grid["metadata"]["layered_model"],
                        nx=nx,
                        ny=ny,
                        nz=nz,
                    )
                )

    output_resolved.mkdir(parents=True, exist_ok=True)
    case_metrics_csv = output_resolved / "structural_case_metrics.csv"
    layer_metrics_csv = output_resolved / "layer_metrics.csv"
    summary_csv = output_resolved / "structural_summary.csv"
    summary_json = output_resolved / "structural_recovery_summary.json"
    summary_md = output_resolved / "structural_recovery_summary.md"
    metadata_json = output_resolved / "audit_metadata.json"
    _write_csv(case_metrics_csv, case_rows)
    _write_csv(layer_metrics_csv, layer_rows)
    summary_rows = _summary_rows(case_rows)
    _write_csv(summary_csv, summary_rows)
    summary = {
        "artifact_type": "structure_aware_grouped_test_metrics_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_files": {
            "manifest": _relative_or_absolute(manifest_resolved, repo_root),
            "grouped_split": _relative_or_absolute(split_resolved, repo_root),
            "ml_metrics": _relative_or_absolute(ml_metrics_resolved, repo_root),
            "fair_classical_corpus": _relative_or_absolute(classical_resolved, repo_root),
        },
        "test_case_count": len(test_items),
        "test_unique_target_count": len({target_hash_by_case[str(item["case_id"])] for item in test_items}),
        "methods": list(methods),
        "mask_definition": "cell-centered target differs from the saved layered background by more than 1e-12 km/s; fault sides use the saved fault-plane parameters; layered metrics use saved layer boundaries",
        "structure_families": sorted({str(item.get("family") or item.get("scenario")) for item in test_items}),
        "summary_rows": summary_rows,
        "limitations": [
            "The structure masks are objective masks of the synthetic generator, not masks available to an inversion algorithm.",
            "No predicted localization-overlap score is reported because defining a predicted body threshold would introduce an additional arbitrary threshold; contrast and body/background errors are reported instead.",
            "Layer-boundary recovery error is not reported because the current predictions do not include a boundary-identification procedure.",
            "The full_input method remains a privileged true-model path-length diagnostic; full_input_euclidean_path is the realistic path-length alternative.",
        ],
    }
    _write_json(summary_json, summary)
    _write_markdown(summary_md, summary, case_metrics_csv, layer_metrics_csv, repo_root)
    _write_json(
        metadata_json,
        {
            "artifact_type": "structure_aware_metrics_metadata_v1",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "command": "tomobench audit-structural-recovery",
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "methods": list(methods),
            "source_file_sha256": {
                key: _sha256_file(path)
                for key, path in (
                    ("manifest", manifest_resolved),
                    ("grouped_split", split_resolved),
                    ("ml_metrics", ml_metrics_resolved),
                )
            },
        },
    )
    return StructuralRecoveryOutputs(
        output_dir=output_resolved,
        case_metrics_csv=case_metrics_csv,
        layer_metrics_csv=layer_metrics_csv,
        summary_csv=summary_csv,
        summary_json=summary_json,
        summary_md=summary_md,
        metadata_json=metadata_json,
    )


def _family_masks(
    grid: dict[str, Any],
    family: str,
    structure_mask: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    if family == "layered":
        return {"layer": _layer_masks(grid)}
    if family == "faulted":
        positive, negative = _fault_side_masks(grid)
        return {
            "fault": {"positive_side": positive, "negative_side": negative},
            "structure": {
                "structure_body": structure_mask,
                "background": ~structure_mask,
            },
        }
    return {
        "structure": {
            "structure_body": structure_mask,
            "background": ~structure_mask,
        }
    }


def _layer_masks(grid: dict[str, Any]) -> dict[str, np.ndarray]:
    z_coordinates = [float(value) for value in grid["z_coordinates_km"]]
    layered = grid["metadata"]["layered_model"]
    boundaries = [float(value) for value in layered["depth_boundaries_km"]]
    nx_cells = len(grid["x_coordinates_km"]) - 1
    ny_cells = len(grid["y_coordinates_km"]) - 1
    nz_cells = len(z_coordinates) - 1
    depth_centers = [
        0.5 * (z_coordinates[iz] + z_coordinates[iz + 1]) for iz in range(nz_cells)
    ]
    masks: dict[str, np.ndarray] = {}
    for layer_index in range(len(boundaries) - 1):
        selected_depths = {
            iz
            for iz, depth in enumerate(depth_centers)
            if boundaries[layer_index] <= depth < boundaries[layer_index + 1]
        }
        mask = np.array(
            [
                iz in selected_depths
                for iz in range(nz_cells)
                for _iy in range(ny_cells)
                for _ix in range(nx_cells)
            ],
            dtype=bool,
        )
        masks[f"layer_{layer_index + 1:02d}"] = mask
    return masks


def _fault_side_masks(grid: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    parameters = grid["metadata"]["faulted"]
    strike_rad = np.deg2rad(float(parameters["strike_deg"]))
    normal_x = float(np.sin(strike_rad))
    normal_y = float(-np.cos(strike_rad))
    dip_deg = float(parameters["dip_deg"])
    dip_sign = 1.0 if parameters["dip_direction"] == "positive_normal" else -1.0
    x_coordinates = [float(value) for value in grid["x_coordinates_km"]]
    y_coordinates = [float(value) for value in grid["y_coordinates_km"]]
    z_coordinates = [float(value) for value in grid["z_coordinates_km"]]
    values: list[bool] = []
    for iz in range(len(z_coordinates) - 1):
        z = 0.5 * (z_coordinates[iz] + z_coordinates[iz + 1])
        horizontal_offset = z / np.tan(np.deg2rad(dip_deg)) if not np.isclose(dip_deg, 90.0) else 0.0
        for iy in range(len(y_coordinates) - 1):
            y = 0.5 * (y_coordinates[iy] + y_coordinates[iy + 1])
            for ix in range(len(x_coordinates) - 1):
                x = 0.5 * (x_coordinates[ix] + x_coordinates[ix + 1])
                signed_offset = (
                    (x - float(parameters["fault_x_km"])) * normal_x
                    + (y - float(parameters["fault_y_km"])) * normal_y
                    - dip_sign * horizontal_offset
                )
                values.append(
                    signed_offset >= 0.0
                    if parameters["positive_side"] == "greater_equal"
                    else signed_offset <= 0.0
                )
    positive = np.array(values, dtype=bool)
    return positive, ~positive


def _base_layered_cell_values(grid: dict[str, Any]) -> np.ndarray:
    metadata = grid["metadata"]["layered_model"]
    boundaries = [float(value) for value in metadata["depth_boundaries_km"]]
    velocities = [float(value) for value in metadata["velocities_km_per_s"]]
    x = [float(value) for value in grid["x_coordinates_km"]]
    y = [float(value) for value in grid["y_coordinates_km"]]
    z = [float(value) for value in grid["z_coordinates_km"]]
    nodes = []
    for z_value in z:
        velocity = _layer_velocity(z_value, boundaries, velocities)
        nodes.extend([velocity] * (len(x) * len(y)))
    return np.asarray(
        node_centered_values_to_cell_centered(nodes, len(x), len(y), len(z)),
        dtype=float,
    )


def _layer_velocity(depth: float, boundaries: Sequence[float], velocities: Sequence[float]) -> float:
    for index, velocity in enumerate(velocities):
        if boundaries[index] <= depth < boundaries[index + 1]:
            return float(velocity)
    if np.isclose(depth, boundaries[-1]):
        return float(velocities[-1])
    raise ValueError(f"Depth outside layered model: {depth}")


def _load_case_predictions(
    *,
    case_id: str,
    target_grid: dict[str, Any],
    expected_cell_count: int,
    methods: Sequence[str],
    ml_by_variant_case: dict[tuple[str, str], dict[str, Any]],
    classical_dir: Path,
    repo_root: Path,
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    if "reference_prior" in methods:
        summary = _read_json(
            classical_dir / "cases" / case_id / "reference_ray_perturbation_summary.json"
        )
        reference_grid = _read_json(
            _resolve_path(Path(str(summary["source_files"]["reference_velocity_grid"])), repo_root)
        )
        predictions["reference_prior"] = _node_to_cells(reference_grid)
    if "optimized_fixed_ray" in methods:
        predictions["optimized_fixed_ray"] = _read_classical_prediction(
            classical_dir / "cases" / case_id / "reference_ray_cell_velocity_predictions.csv",
            expected_cell_count,
        )
    for method in methods:
        if method in {"reference_prior", "optimized_fixed_ray"}:
            continue
        row = ml_by_variant_case.get((method, case_id))
        if row is None:
            raise ValueError(f"Missing ML row for method={method}, case={case_id}.")
        prediction = _read_json(_resolve_path(Path(str(row["prediction_path"])), repo_root))
        nodes = prediction["predicted_p_velocity_km_per_s"]
        predictions[method] = np.asarray(
            node_centered_values_to_cell_centered(
                nodes,
                len(target_grid["x_coordinates_km"]),
                len(target_grid["y_coordinates_km"]),
                len(target_grid["z_coordinates_km"]),
            ),
            dtype=float,
        )
    for method, prediction in predictions.items():
        if prediction.size != expected_cell_count:
            raise ValueError(f"{method} prediction length mismatch for case={case_id}.")
    return predictions


def _node_to_cells(grid: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        node_centered_values_to_cell_centered(
            grid["p_velocity_km_per_s"],
            len(grid["x_coordinates_km"]),
            len(grid["y_coordinates_km"]),
            len(grid["z_coordinates_km"]),
        ),
        dtype=float,
    )


def _layer_metric_rows(
    *,
    case_id: str,
    family: str,
    target_hash: str,
    methods_by_case: dict[str, np.ndarray],
    target_cells: np.ndarray,
    z_coordinates: Sequence[float],
    layered_metadata: dict[str, Any],
    nx: int,
    ny: int,
    nz: int,
) -> list[dict[str, Any]]:
    boundaries = [float(value) for value in layered_metadata["depth_boundaries_km"]]
    layer_rows: list[dict[str, Any]] = []
    layer_indices = [
        next(
            index
            for index in range(len(boundaries) - 1)
            if boundaries[index] <= 0.5 * (z_coordinates[iz] + z_coordinates[iz + 1]) < boundaries[index + 1]
        )
        for iz in range(nz - 1)
    ]
    for method, prediction in methods_by_case.items():
        for layer_index in range(len(boundaries) - 1):
            mask = np.array(
                [
                    layer_indices[iz] == layer_index
                    for iz in range(nz - 1)
                    for _iy in range(ny - 1)
                    for _ix in range(nx - 1)
                ],
                dtype=bool,
            )
            metrics = _masked_metrics(target_cells, prediction, mask)
            layer_rows.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "target_hash": target_hash,
                    "method_id": method,
                    "layer_index": layer_index + 1,
                    "top_depth_km": boundaries[layer_index],
                    "bottom_depth_km": boundaries[layer_index + 1],
                    "cell_count": int(np.count_nonzero(mask)),
                    "rmse_km_per_s": metrics["rmse"],
                    "mae_km_per_s": metrics["mae"],
                    "bias_km_per_s": metrics["bias"],
                }
            )
    return layer_rows


def _masked_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    if target.shape != prediction.shape or target.shape != mask.shape:
        raise ValueError("Target, prediction, and mask shapes must match.")
    target_values = target[mask]
    prediction_values = prediction[mask]
    if target_values.size == 0:
        return {
            "rmse": None,
            "mae": None,
            "bias": None,
            "true_mean": None,
            "predicted_mean": None,
        }
    return {
        "rmse": root_mean_squared_error(target_values.tolist(), prediction_values.tolist()),
        "mae": mean_absolute_error(target_values.tolist(), prediction_values.tolist()),
        "bias": float(np.mean(prediction_values - target_values)),
        "true_mean": float(np.mean(target_values)),
        "predicted_mean": float(np.mean(prediction_values)),
    }


def _contrast_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    body_mask: np.ndarray,
    background_mask: np.ndarray,
) -> dict[str, float | None]:
    if not np.any(body_mask) or not np.any(background_mask):
        return {"true_contrast": None, "predicted_contrast": None, "contrast_error": None}
    true_contrast = float(np.mean(target[body_mask]) - np.mean(target[background_mask]))
    predicted_contrast = float(
        np.mean(prediction[body_mask]) - np.mean(prediction[background_mask])
    )
    return {
        "true_contrast": true_contrast,
        "predicted_contrast": predicted_contrast,
        "contrast_error": predicted_contrast - true_contrast,
    }


def _summary_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[
            (
                str(row["family"]),
                str(row["method_id"]),
                str(row["metric_scope"]),
                str(row["mask_label"]),
            )
        ].append(row)
    rows: list[dict[str, Any]] = []
    for (family, method, scope, mask_label), group in sorted(grouped.items()):
        rows.append(
            {
                "family": family,
                "method_id": method,
                "metric_scope": scope,
                "mask_label": mask_label,
                "case_count": len(group),
                "unique_target_count": len({str(row["target_hash"]) for row in group}),
                "mean_cell_count": _mean(group, "cell_count"),
                "mean_rmse_km_per_s": _mean(group, "rmse_km_per_s"),
                "mean_mae_km_per_s": _mean(group, "mae_km_per_s"),
                "mean_bias_km_per_s": _mean(group, "bias_km_per_s"),
                "mean_true_contrast_km_per_s": _mean(group, "true_contrast_km_per_s"),
                "mean_predicted_contrast_km_per_s": _mean(
                    group, "predicted_contrast_km_per_s"
                ),
                "mean_contrast_error_km_per_s": _mean(group, "contrast_error_km_per_s"),
            }
        )
    return rows


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return sum(values) / len(values) if values else None


def _index_ml_rows(rows: list[dict[str, Any]], methods: Sequence[str]) -> dict[tuple[str, str], dict[str, Any]]:
    method_set = set(methods)
    return {
        (str(row.get("variant_id", "")), str(row["case_id"])): row
        for row in rows
        if str(row.get("split", "")) == "test" and str(row.get("variant_id", "")) in method_set
    }


def _read_classical_prediction(path: Path, expected_cell_count: int) -> np.ndarray:
    rows = _read_csv(path)
    values = np.asarray([float(row["predicted_velocity_km_per_s"]) for row in rows], dtype=float)
    if values.size != expected_cell_count:
        raise ValueError(f"Classical prediction length mismatch: {path}")
    return values


def _manifest_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    items = manifest.get("items")
    if not isinstance(items, list) or not items or not all(isinstance(item, dict) for item in items):
        raise ValueError("Manifest must contain a non-empty list of object items.")
    return [item for item in items if isinstance(item, dict)]


def _read_grouped_split(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    split_by_case: dict[str, str] = {}
    target_hash_by_case: dict[str, str] = {}
    target_splits: dict[str, set[str]] = defaultdict(set)
    for row in _read_csv(path):
        case_id = str(row["case_id"])
        target_hash = str(row["target_hash"])
        split_by_case[case_id] = str(row["split"])
        target_hash_by_case[case_id] = target_hash
        target_splits[target_hash].add(str(row["split"]))
    if any(len(splits) > 1 for splits in target_splits.values()):
        raise ValueError("Target groups cross grouped-split boundaries.")
    return split_by_case, target_hash_by_case


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, summary: dict[str, Any], case_path: Path, layer_path: Path, repo_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Structure-Aware Grouped-Test Metrics",
                "",
                "Masks are derived from saved synthetic generator metadata and are used only for evaluation.",
                "",
                f"- Test cases: `{summary['test_case_count']}`",
                f"- Unique test targets: `{summary['test_unique_target_count']}`",
                f"- Case metrics: `{_relative_or_absolute(case_path, repo_root)}`",
                f"- Layer metrics: `{_relative_or_absolute(layer_path, repo_root)}`",
                "",
                "Localization-overlap and boundary-depth metrics are intentionally omitted where their definition would require an arbitrary predicted-structure threshold or a boundary detector.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _resolve_manifest_path(path_value: Any, repo_root: Path) -> Path:
    if not path_value:
        raise ValueError("Missing target grid path in manifest.")
    path = _resolve_path(Path(str(path_value)), repo_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["StructuralRecoveryOutputs", "run_structural_recovery_metrics"]
