"""Render Figures 1, 3 and 4 and Supplementary Figures S1 and S2 from frozen records.

This script reads the frozen production corpus, the persisted predictions, and the
stored metric and diagnostic records. It does not fit models, select
configurations, resample, or re-run any forward model, so it regenerates no
scientific result.

Figures 2 and 5 are not rendered here: their published files are unchanged, and
regenerating them would need the PCA-spectrum, timing-noise and coverage inputs
produced by the full analysis run rather than the persisted records used below.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from tomobench.evaluation.benchmark_v2_final_analysis import load_production_cases  # noqa: E402
from tomobench.evaluation.final_robustness_checks import (  # noqa: E402
    plot_fsm_fixed_ray_jacobian_scatter,
)
from tomobench.evaluation.final_evidence_pass import (  # noqa: E402
    _plot_primary_comparison,
    _plot_structural_fields,
    _plot_supplementary_structure_slices,
    _plot_workflow,
    _read_csv_local,
)


DEFAULT_PRODUCTION = REPO_ROOT / "outputs/generated/submission_readiness/benchmark_v2_production_250_v1"
DEFAULT_EVIDENCE = REPO_ROOT / "outputs/generated/submission_readiness/final_evidence_pass_v2/evidence"
DEFAULT_JACOBIAN = (
    REPO_ROOT
    / "outputs/generated/submission_readiness/applied_sciences_final_minor_revision_closure"
    / "records/secondary_diagnostics/fsm_fixed_ray_jacobian"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/generated/submission_readiness/final_evidence_pass_v2/submission/figures"

EXPECTED_JACOBIAN_ENTRIES = 120


def _load_node_native(production_dir: Path, evidence_dir: Path):
    """Rebuild the plotting inputs from the frozen corpus and persisted predictions."""

    cases = load_production_cases(production_dir)
    test = [case for case in cases if case.split == "test"]
    rows = _read_csv_local(evidence_dir / "node_native" / "case_metrics.csv")
    prediction_path = evidence_dir / "node_native" / "realistic_full_test_predictions.npz"
    with np.load(prediction_path) as artifact:
        target_ids = [str(value) for value in artifact["target_ids"].tolist()]
        predictions = np.asarray(artifact["predicted_node_velocity_km_per_s"], dtype=float)
    expected_ids = [case.target_id for case in test]
    if target_ids != expected_ids:
        raise ValueError("Persisted prediction ordering does not match the frozen test target order.")
    if predictions.shape[0] != len(test):
        raise ValueError("Persisted prediction count does not match the frozen test set.")
    node_native = {
        "cases": test,
        "test": test,
        "metric_rows": rows,
        "predictions": {"realistic_full": predictions},
    }
    return node_native, test


def _render_supplementary_jacobian(jacobian_dir: Path, output_dir: Path) -> Path:
    """Render Supplementary Figure S2 from the stored diagnostic entries."""

    entries_path = jacobian_dir / "fsm_fixed_ray_jacobian_entries.csv"
    with entries_path.open(encoding="utf-8", newline="") as handle:
        entries = list(csv.DictReader(handle))
    if len(entries) != EXPECTED_JACOBIAN_ENTRIES:
        raise ValueError(
            f"Expected {EXPECTED_JACOBIAN_ENTRIES} stored jacobian entries, found {len(entries)}."
        )
    produced = plot_fsm_fixed_ray_jacobian_scatter(entries, output_dir)
    target = output_dir / "figure_S2_fsm_reference_ray_sensitivity.png"
    produced.replace(target)
    (output_dir / "fsm_fixed_ray_jacobian_scatter.svg").unlink(missing_ok=True)
    return target


def render(production_dir: Path, evidence_dir: Path, jacobian_dir: Path, output_dir: Path) -> None:
    """Render the five figure assets from persisted records."""

    node_native, test = _load_node_native(production_dir, evidence_dir)
    reference_ray = {
        "test_rows": _read_csv_local(evidence_dir / "reference_ray" / "test_metrics.csv")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_workflow(output_dir)
    _plot_primary_comparison(node_native, reference_ray, output_dir)
    _plot_structural_fields(node_native, test, output_dir)
    _plot_supplementary_structure_slices(node_native, output_dir)
    _render_supplementary_jacobian(jacobian_dir, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--jacobian-dir", type=Path, default=DEFAULT_JACOBIAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(
        args.production_dir.resolve(),
        args.evidence_dir.resolve(),
        args.jacobian_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
