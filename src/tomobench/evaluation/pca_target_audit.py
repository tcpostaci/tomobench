"""Audit the information loss and prior strength of the PCA target contract."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tomobench.datasets.target_identity import target_vector_sha256
from tomobench.evaluation.metrics import mean_absolute_error, root_mean_squared_error
from tomobench.tomography.comparison import node_centered_values_to_cell_centered
from tomobench.utils.paths import get_repo_root


PCA_TARGET_AUDIT_VERSION = "pca_target_audit_v1"
DEFAULT_PCA_METRICS_PATH = Path(
    "outputs/generated/ml_baselines/phase9_grouped_split_ml_v1/"
    "models/pca-linear/fixed_split/metrics.json"
)
DEFAULT_PCA_MANIFEST_PATH = Path(
    "outputs/datasets/ml_supervised/paper_pseudo_bending_observation_pairs_v1/manifest.json"
)
DEFAULT_PCA_OUTPUT_DIR = Path("outputs/generated/submission_readiness/pca_target_audit_v1")
DEFAULT_REQUESTED_COMPONENT_COUNTS = (1, 2, 3, 4, 5, 10)


@dataclass(frozen=True)
class PCATargetAuditOutputs:
    """Paths written by the PCA target audit."""

    output_dir: Path
    component_variance_csv: Path
    reconstruction_case_metrics_csv: Path
    reconstruction_summary_csv: Path
    audit_summary_md: Path
    audit_metadata_json: Path


def run_pca_target_audit(
    *,
    metrics_path: Path = DEFAULT_PCA_METRICS_PATH,
    manifest_path: Path = DEFAULT_PCA_MANIFEST_PATH,
    output_dir: Path = DEFAULT_PCA_OUTPUT_DIR,
    configured_component_count: int = 4,
    requested_component_counts: Sequence[int] = DEFAULT_REQUESTED_COMPONENT_COUNTS,
    settings: Any | None = None,
) -> PCATargetAuditOutputs:
    """Quantify PCA variance retention, oracle error, and model reconstruction error.

    The PCA basis is fitted only to the target vectors belonging to the
    training split recorded by ``metrics_path``.  The supplied PCA-linear
    predictions are read from that same metrics artifact, so this audit does
    not retrain or select a model using validation/test targets.
    """
    del settings  # Reserved for a future typed settings dependency.
    if configured_component_count <= 0:
        raise ValueError("configured_component_count must be positive.")
    normalized_requested = tuple(
        sorted({int(value) for value in requested_component_counts if int(value) > 0})
    )
    if not normalized_requested:
        raise ValueError("requested_component_counts must contain a positive value.")

    repo_root = get_repo_root()
    resolved_metrics_path = _resolve_path(Path(metrics_path), repo_root)
    resolved_manifest_path = _resolve_path(Path(manifest_path), repo_root)
    resolved_output_dir = _resolve_path(Path(output_dir), repo_root)
    _require_file(resolved_metrics_path, "PCA metrics")
    _require_file(resolved_manifest_path, "corpus manifest")

    metrics = _read_json(resolved_metrics_path)
    manifest = _read_json(resolved_manifest_path)
    split_by_case_id, predicted_by_case_id = _read_split_and_predictions(
        metrics,
        repo_root,
    )
    targets = _load_manifest_targets(manifest, repo_root)
    missing_case_ids = sorted(set(targets) - set(split_by_case_id))
    extra_case_ids = sorted(set(split_by_case_id) - set(targets))
    if missing_case_ids or extra_case_ids:
        raise ValueError(
            "PCA metrics and manifest case IDs do not match. "
            f"Missing split IDs: {missing_case_ids}; extra split IDs: {extra_case_ids}."
        )

    training_case_ids = [
        case_id for case_id in targets if split_by_case_id[case_id] == "train"
    ]
    if len(training_case_ids) < 2:
        raise ValueError("PCA target audit requires at least two training cases.")
    target_length = len(targets[training_case_ids[0]]["target"])
    if any(len(targets[case_id]["target"]) != target_length for case_id in targets):
        raise ValueError("All target vectors must have the same length.")

    training_matrix = np.asarray(
        [targets[case_id]["target"] for case_id in training_case_ids],
        dtype=np.float64,
    )
    target_mean = np.mean(training_matrix, axis=0)
    components, eigenvalues = _fit_pca_basis(training_matrix)
    rank = len(components)
    total_variance = float(np.sum(eigenvalues))
    explained_variance_rows = _explained_variance_rows(
        eigenvalues,
        normalized_requested,
    )

    effective_oracle_components = min(configured_component_count, rank)
    if effective_oracle_components <= 0:
        raise ValueError("Training targets contain no non-zero PCA variation.")
    oracle_predictions = _reconstruct_with_pca(
        np.asarray([record["target"] for record in targets.values()], dtype=np.float64),
        target_mean,
        components,
        effective_oracle_components,
    )
    oracle_by_case_id = {
        case_id: tuple(float(value) for value in prediction)
        for case_id, prediction in zip(targets, oracle_predictions, strict=True)
    }
    mean_prediction = tuple(float(value) for value in target_mean)
    comparison_variants = {
        "training_target_mean": {
            case_id: mean_prediction for case_id in targets
        },
        f"oracle_pca_{effective_oracle_components}": oracle_by_case_id,
        "predicted_pca_linear": predicted_by_case_id,
    }

    case_rows: list[dict[str, Any]] = []
    for variant_id, predictions in comparison_variants.items():
        for case_id, target_record in targets.items():
            if case_id not in predictions:
                raise ValueError(f"Missing {variant_id} prediction for case {case_id}.")
            case_rows.append(
                _case_metric_row(
                    case_id=case_id,
                    target_record=target_record,
                    target=target_record["target"],
                    prediction=predictions[case_id],
                    split=split_by_case_id[case_id],
                    variant_id=variant_id,
                )
            )
    summary_rows = _aggregate_case_rows(case_rows, targets)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    component_variance_csv = resolved_output_dir / "pca_component_variance.csv"
    reconstruction_case_metrics_csv = resolved_output_dir / "pca_reconstruction_case_metrics.csv"
    reconstruction_summary_csv = resolved_output_dir / "pca_reconstruction_summary.csv"
    audit_summary_md = resolved_output_dir / "audit_summary.md"
    audit_metadata_json = resolved_output_dir / "audit_metadata.json"
    _write_csv(component_variance_csv, explained_variance_rows)
    _write_csv(reconstruction_case_metrics_csv, case_rows)
    _write_csv(reconstruction_summary_csv, summary_rows)

    summary_payload = {
        "artifact_type": PCA_TARGET_AUDIT_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "metrics_path": _relative_or_absolute(resolved_metrics_path, repo_root),
        "manifest_path": _relative_or_absolute(resolved_manifest_path, repo_root),
        "split_strategy": str(metrics.get("split_strategy", "")),
        "target_vector_length": target_length,
        "training_case_count": len(training_case_ids),
        "training_unique_target_count": len(
            {target_vector_sha256(targets[case_id]["target"]) for case_id in training_case_ids}
        ),
        "total_case_count": len(targets),
        "total_unique_target_count": len(
            {target_vector_sha256(record["target"]) for record in targets.values()}
        ),
        "nonzero_pca_rank": rank,
        "configured_component_count": configured_component_count,
        "effective_oracle_component_count": effective_oracle_components,
        "requested_component_counts": list(normalized_requested),
        "unavailable_requested_component_counts": [
            value for value in normalized_requested if value > rank
        ],
        "total_centered_sum_of_squares": total_variance,
        "explained_variance_rows": explained_variance_rows,
        "split_case_counts": _split_counts(split_by_case_id),
        "split_unique_target_counts": _split_unique_counts(split_by_case_id, targets),
        "comparison_summary": summary_rows,
        "metric_contract": {
            "node_centered": "RMSE/MAE over the persisted node-centered target vector",
            "cell_centered": "RMSE/MAE after averaging each eight-node cell neighborhood",
            "oracle": "True target projected onto a training-only PCA basis and reconstructed with true coefficients",
            "predicted_pca_linear": "Persisted grouped-split PCA-linear prediction; no test fitting occurs in this audit",
        },
        "interpretation_caution": [
            "Oracle PCA error is a representation ceiling, not a deployable model result.",
            "The training-target mean baseline is fitted only on training cases.",
            "Case-level aggregates retain acquisition-replicate performance; unique-target counts identify the independent target units.",
        ],
    }
    _write_json(summary_payload, resolved_output_dir / "pca_target_audit_summary.json")
    metadata_payload = {
        "artifact_type": f"{PCA_TARGET_AUDIT_VERSION}_metadata",
        "command": "uv run tomobench audit-pca-target-representation",
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "metrics_sha256": _file_sha256(resolved_metrics_path),
        "manifest_sha256": _file_sha256(resolved_manifest_path),
        "canonical_target_hash_schema": "target_velocity_vector_sha256_v1",
        "configured_component_count": configured_component_count,
        "requested_component_counts": list(normalized_requested),
    }
    _write_json(metadata_payload, audit_metadata_json)
    _write_markdown_summary(
        audit_summary_md,
        summary_payload,
        component_variance_csv,
        reconstruction_summary_csv,
        repo_root,
    )
    return PCATargetAuditOutputs(
        output_dir=resolved_output_dir,
        component_variance_csv=component_variance_csv,
        reconstruction_case_metrics_csv=reconstruction_case_metrics_csv,
        reconstruction_summary_csv=reconstruction_summary_csv,
        audit_summary_md=audit_summary_md,
        audit_metadata_json=audit_metadata_json,
    )


def _fit_pca_basis(target_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = target_matrix - np.mean(target_matrix, axis=0)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values**2
    tolerance = max(float(eigenvalues[0]) if len(eigenvalues) else 0.0, 1.0) * 1.0e-12
    nonzero = eigenvalues > tolerance
    return right_vectors[nonzero], eigenvalues[nonzero]


def _explained_variance_rows(
    eigenvalues: np.ndarray,
    requested_component_counts: Sequence[int],
) -> list[dict[str, Any]]:
    total = float(np.sum(eigenvalues))
    requested = set(requested_component_counts)
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for index, eigenvalue in enumerate(eigenvalues, start=1):
        ratio = float(eigenvalue / total) if total > 0.0 else 0.0
        cumulative += ratio
        rows.append(
            {
                "component": index,
                "eigenvalue": float(eigenvalue),
                "explained_variance_ratio": ratio,
                "cumulative_explained_variance_ratio": cumulative,
                "requested_component_count": index in requested,
            }
        )
    return rows


def _reconstruct_with_pca(
    targets: np.ndarray,
    target_mean: np.ndarray,
    components: np.ndarray,
    component_count: int,
) -> np.ndarray:
    centered = targets - target_mean
    coefficients = centered @ components[:component_count].T
    return target_mean + coefficients @ components[:component_count]


def _case_metric_row(
    *,
    case_id: str,
    target_record: dict[str, Any],
    target: Sequence[float],
    prediction: Sequence[float],
    split: str,
    variant_id: str,
) -> dict[str, Any]:
    target_cells = node_centered_values_to_cell_centered(
        target,
        *target_record["shape"],
    )
    prediction_cells = node_centered_values_to_cell_centered(
        prediction,
        *target_record["shape"],
    )
    return {
        "case_id": case_id,
        "family": target_record["family"],
        "split": split,
        "variant_id": variant_id,
        "target_hash": target_vector_sha256(target),
        "node_rmse_km_per_s": root_mean_squared_error(target, prediction),
        "node_mae_km_per_s": mean_absolute_error(target, prediction),
        "cell_rmse_km_per_s": root_mean_squared_error(target_cells, prediction_cells),
        "cell_mae_km_per_s": mean_absolute_error(target_cells, prediction_cells),
    }


def _aggregate_case_rows(
    case_rows: list[dict[str, Any]],
    targets: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    variants = sorted({str(row["variant_id"]) for row in case_rows})
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for split in ("train", "validation", "test"):
            selected = [
                row
                for row in case_rows
                if row["variant_id"] == variant and row["split"] == split
            ]
            if not selected:
                continue
            rows.append(
                {
                    "variant_id": variant,
                    "split": split,
                    "case_count": len(selected),
                    "unique_target_count": len(
                        {
                            target_vector_sha256(targets[str(row["case_id"])]["target"])
                            for row in selected
                        }
                    ),
                    "mean_node_rmse_km_per_s": _mean(selected, "node_rmse_km_per_s"),
                    "mean_node_mae_km_per_s": _mean(selected, "node_mae_km_per_s"),
                    "mean_cell_rmse_km_per_s": _mean(selected, "cell_rmse_km_per_s"),
                    "mean_cell_mae_km_per_s": _mean(selected, "cell_mae_km_per_s"),
                }
            )
    return rows


def _load_manifest_targets(
    manifest: dict[str, Any],
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("Corpus manifest must contain an items list.")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Corpus manifest items must be objects.")
        case_id = str(item.get("case_id", ""))
        target_path = _resolve_path(
            Path(str(item.get("target_velocity_grid_path", ""))),
            repo_root,
        )
        _require_file(target_path, f"target grid for {case_id}")
        target_payload = _read_json(target_path)
        target = tuple(float(value) for value in target_payload["p_velocity_km_per_s"])
        shape = (
            len(target_payload["x_coordinates_km"]),
            len(target_payload["y_coordinates_km"]),
            len(target_payload["z_coordinates_km"]),
        )
        if len(target) != shape[0] * shape[1] * shape[2]:
            raise ValueError(f"Target shape does not match vector length for {case_id}.")
        targets[case_id] = {
            "family": str(item.get("scenario") or item.get("family") or ""),
            "target": target,
            "shape": shape,
            "target_grid_path": _relative_or_absolute(target_path, repo_root),
        }
    return targets


def _read_split_and_predictions(
    metrics: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, str], dict[str, tuple[float, ...]]]:
    split_metrics = metrics.get("split_metrics")
    if not isinstance(split_metrics, dict):
        raise ValueError("PCA metrics must contain split_metrics.")
    split_by_case_id: dict[str, str] = {}
    predictions: dict[str, tuple[float, ...]] = {}
    for split_name, split_payload in split_metrics.items():
        if not isinstance(split_payload, dict):
            continue
        case_metrics = split_payload.get("case_metrics")
        if not isinstance(case_metrics, list):
            continue
        for record in case_metrics:
            if not isinstance(record, dict):
                continue
            case_id = str(record.get("case_id", ""))
            if case_id in split_by_case_id:
                raise ValueError(f"Case appears in more than one split: {case_id}")
            split_by_case_id[case_id] = str(split_name)
            prediction_path = _resolve_path(Path(str(record["prediction_path"])), repo_root)
            prediction_payload = _read_json(prediction_path)
            predictions[case_id] = tuple(
                float(value) for value in prediction_payload["predicted_p_velocity_km_per_s"]
            )
    return split_by_case_id, predictions


def _split_counts(split_by_case_id: dict[str, str]) -> dict[str, int]:
    return {
        split_name: sum(split == split_name for split in split_by_case_id.values())
        for split_name in ("train", "validation", "test")
    }


def _split_unique_counts(
    split_by_case_id: dict[str, str],
    targets: dict[str, dict[str, Any]],
) -> dict[str, int]:
    return {
        split_name: len(
            {
                target_vector_sha256(targets[case_id]["target"])
                for case_id, split in split_by_case_id.items()
                if split == split_name
            }
        )
        for split_name in ("train", "validation", "test")
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _write_markdown_summary(
    path: Path,
    payload: dict[str, Any],
    variance_csv: Path,
    summary_csv: Path,
    repo_root: Path,
) -> None:
    lines = [
        "# PCA Target Representation Audit",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Scope",
        "",
        f"- Metrics: `{payload['metrics_path']}`",
        f"- Split strategy: `{payload['split_strategy']}`",
        f"- Training cases / unique targets: `{payload['training_case_count']}` / `{payload['training_unique_target_count']}`",
        f"- Total cases / unique targets: `{payload['total_case_count']}` / `{payload['total_unique_target_count']}`",
        f"- PCA rank available from training targets: `{payload['nonzero_pca_rank']}`",
        f"- Configured components: `{payload['configured_component_count']}`",
        "",
        "## Interpretation",
        "",
        "The oracle rows use the true target coefficients in a PCA basis fitted only on training targets. They therefore quantify representation error, not deployable inversion performance.",
        "",
        "## Comparison",
        "",
    ]
    for row in payload["comparison_summary"]:
        lines.append(
            f"- `{row['variant_id']}` / `{row['split']}`: cell RMSE `{float(row['mean_cell_rmse_km_per_s']):.6f}` km/s; node RMSE `{float(row['mean_node_rmse_km_per_s']):.6f}` km/s; cases `{row['case_count']}`; unique targets `{row['unique_target_count']}`."
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Component variance CSV: `{_relative_or_absolute(variance_csv, repo_root)}`",
            f"- Reconstruction summary CSV: `{_relative_or_absolute(summary_csv, repo_root)}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["PCATargetAuditOutputs", "run_pca_target_audit"]
