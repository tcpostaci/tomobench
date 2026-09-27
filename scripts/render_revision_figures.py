"""Render Figure 1 and Supplementary Figures S1, S4 and S5 at print width.

    python code/scripts/render_revision_figures.py --package PKG            # render and install
    python code/scripts/render_revision_figures.py --package PKG --check    # re-render, compare bytes

Reviewer 1 of applsci-4595434 rated the figures "must be improved". The MDPI
documents place every figure 5.46 in wide, and these four were drawn 9.5 to 15.5 in
wide at 9 pt, so their labels printed at about 3 pt (S1, S4, S5) and 5 pt
(Figure 1). The plotting functions in tomobench.evaluation.final_evidence_pass now
draw them close to the placed width. Data, target selection and slice rules are
unchanged: the two selection records the functions write are compared byte for
byte with the deposited ones under records/figures/, and the run fails if either
differs. Supplementary Figures S2, S3 and S6 already printed at 78-91% scale and
are not re-rendered.

Inputs are the frozen production corpus (for S1 and S4, which draw analytic target
fields) and the package's own records. The corpus manifest stores the corpus's
original repository path; it now lives under _archive/, so --corpus-dir is
substituted for that stored prefix when paths are resolved.

Run with code/.venv, the recorded Matplotlib 3.10.9 environment. The frozen
submitted package is refused: it is the reviewers' baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

import tomobench.evaluation.benchmark_v2_final_analysis as bfa  # noqa: E402
import tomobench.evaluation.final_evidence_pass as fep  # noqa: E402

FROZEN_PACKAGE_NAME = "applied_sciences_final_submission"
STORED_CORPUS_PREFIX = "outputs/generated/submission_readiness/benchmark_v2_production_250_v1"
DEFAULT_CORPUS = REPO_ROOT / "outputs/generated/submission_readiness/_archive/benchmark_v2_production_250_v1"

# rendered file -> installed copies, relative to the package
INSTALL = {
    "figure_1_workflow.png": ("figures/figure_1_workflow.png", "records/figures/figure_1_workflow.png"),
    "figure_S1_structure_intersecting_slices.png": (
        "supplementary_figures/figure_S1_structure_intersecting_slices.png",
        "records/figures/figure_S1_structure_intersecting_slices.png"),
    "figure_4_structural_fields.png": (
        "supplementary_figures/figure_S4_structural_fields.png",
        "records/figures/figure_4_structural_fields.png"),
    "figure_5_coverage_noise.png": (
        "supplementary_figures/figure_S5_coverage_noise.png",
        "records/figures/figure_5_coverage_noise.png"),
}
SELECTIONS = ("figure_4_selection.json", "figure_S1_selection.json")


def _rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def _use_corpus(corpus: Path) -> None:
    original = bfa._resolve_repo_path

    def resolve(path_value: str) -> Path:
        value = path_value.replace("\\", "/")
        if value.startswith(STORED_CORPUS_PREFIX):
            return corpus / value[len(STORED_CORPUS_PREFIX):].lstrip("/")
        return original(path_value)

    bfa._resolve_repo_path = resolve


def render(package: Path, corpus: Path, out: Path) -> None:
    records = package / "records"
    _use_corpus(corpus)
    cases = bfa.load_production_cases(corpus)
    test = [case for case in cases if case.split == "test"]
    with np.load(records / "node_native/realistic_full_test_predictions.npz") as artifact:
        target_ids = [str(value) for value in artifact["target_ids"].tolist()]
        predictions = np.asarray(artifact["predicted_node_velocity_km_per_s"], dtype=float)
    if target_ids != [case.target_id for case in test]:
        raise ValueError("Persisted prediction ordering does not match the frozen test target order.")
    node_native = {"cases": test, "test": test, "metric_rows": _rows(records / "node_native/case_metrics.csv"),
                   "predictions": {"realistic_full": predictions}}
    out.mkdir(parents=True, exist_ok=True)
    fep._plot_workflow(out)
    fep._plot_structural_fields(node_native, test, out)
    fep._plot_supplementary_structure_slices(node_native, out)
    fep._plot_noise_coverage({"summary_rows": _rows(records / "noise/summary.csv")},
                             {"coverage_summary": _rows(records / "coverage_structure/coverage_summary.csv")}, out)
    for name in SELECTIONS:
        if (out / name).read_bytes() != (records / "figures" / name).read_bytes():
            raise SystemExit("selection record %s differs from records/figures/%s: the re-render would "
                             "show different targets or slices" % (name, name))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--check", action="store_true", help="re-render and compare with the installed files")
    args = ap.parse_args()
    package = args.package.resolve()
    if package.name == FROZEN_PACKAGE_NAME:
        raise SystemExit("refusing the frozen submitted package")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        render(package, args.corpus_dir.resolve(), out)
        bad = 0
        for name, targets in INSTALL.items():
            digest = _sha(out / name)
            for target in targets:
                dest = package / target
                if args.check:
                    same = dest.is_file() and _sha(dest) == digest
                    bad += not same
                    print("%s %s %s" % ("OK " if same else "BAD", target, digest[:16]))
                else:
                    shutil.copyfile(out / name, dest)
                    print("installed %s %s" % (target, digest[:16]))
    print("selection records identical to records/figures/: %s" % ", ".join(SELECTIONS))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
