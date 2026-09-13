# tomobench

Reference implementation of the discrete synthetic benchmark for **PCA-ridge** and
**Reference-ray** 3-D velocity reconstruction from first-arrival travel times.

This repository holds the code. The data, the evaluation records and the frozen configuration
live in the reproducibility archive, which is the authoritative artifact: every value published
in the article re-derives from the records deposited there, and the article's `records/...`
identifiers are paths within it.

| | |
|---|---|
| Article | *A Discrete Synthetic Benchmark for PCA-Ridge and Reference-Ray 3-D Velocity Reconstruction from First-Arrival Travel Times*, *Applied Sciences* — DOI assigned on publication |
| Reproducibility archive | [10.5281/zenodo.22733701](https://doi.org/10.5281/zenodo.22733701) |
| Licence | CC BY 4.0, matching the article |

## What the benchmark is

250 synthetic geological targets in five parameterised families, 384 first-arrival observations
each, a 175 / 37 / 38 target-atomic split. Travel times are computed with a finite-grid
Fast-Sweeping calculation on a 1.25 km forward grid; reconstruction is evaluated by direct
analytic velocity at 2.5 km cell centers. A full-input PCA-ridge estimator is compared against a
reference-model fixed-ray regularized path-operator baseline under one declared contract, with
target-level paired uncertainty.

The contribution is the *contract*, not any individual component: PCA, ridge regression,
finite-grid Fast-Sweeping and supervised learning for tomography are all established.

## Install

```bash
git clone https://github.com/tcpostaci/tomobench.git
cd tomobench
pip install -e .
```

Python 3.11 or newer. `pyproject.toml` and `uv.lock` pin the environment; label generation
requires `ttcrpy==1.4.2`. The exact versions used for the published run are listed in the
archive's `reproducibility/dependency_versions.txt`.

## Reproduce

These are the commands recorded in the archive's `reproducibility/regeneration_instructions.md`,
which is authoritative if the two ever disagree.

```bash
python scripts/run_benchmark_v2_production.py --output-dir <fresh-output-dir>
python scripts/run_final_evidence_pass.py
python scripts/render_minor_revision_figures.py
```

Two further entry points reproduce the sensitivities the Supplementary Material reports
separately:

```bash
python scripts/run_matched_prior_reference_ray.py
python scripts/run_observation_ordering_sensitivity.py
```

Sampling and permutation draws depend on the NumPy generator implementation, so the recorded
seeds reproduce the stored draws only under the pinned NumPy version.

## Layout

```
src/tomobench/     the package: generation, simulation, tomography, evaluation, datasets
scripts/           the six entry points the article and the archive name
configs/           the frozen production configuration
docs/              implementation-level method contracts
```

The archive preserves the original monorepo layout, so paths cited there carry a `code/`
prefix. The mapping is mechanical:

| Cited in the archive | Here |
|---|---|
| `code/src/tomobench/…` | `src/tomobench/…` |
| `code/scripts/…` | `scripts/…` |
| `config/…` | `configs/…` |
| `paper/method_contracts/…` | `docs/method_contracts/…` |

## Relationship to the deposited snapshot

The archive carries a source snapshot under `reproducibility/source/` that mirrors this
repository — the same 83 modules and the same six entry points, in the monorepo layout. It is
the copy pinned by checksum alongside the data, so the archive is self-contained and its
regeneration commands can be run from it directly.

Use whichever is convenient. They are the same code; the archive's copy is the one covered by
`reproducibility/SHA256SUMS.txt`.

## Citing

Cite the article and, if you use the data, the archive. `CITATION.cff` carries the
machine-readable form. Please cite the article rather than this repository alone.

## Licence

CC BY 4.0 — the full legal code is in `LICENSE`. It is the licence MDPI applies to the published
article and the one the archive carries, so text, data and code are under a single licence.
