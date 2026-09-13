"""Run the matched-prior Reference-ray comparator and deposit its records.

The published Reference-ray baseline is supplied with the exact anomaly-removed
generator background for four of the five families, while PCA-ridge learns its
prior from the training corpus. The 2026-09-03 blind review called that a
confounded comparison. This runs the same comparator with its background
estimated from training targets only, changing nothing else, and deposits the
per-target record.

Contract:
  * The layer BOUNDARIES are retained from the configured model; only the five
    layer VELOCITIES are estimated, as the mean over all training-target nodes
    in each layer. This isolates privileged values from privileged structural
    form. It does not remove the latter, and the write-up says so.
  * Damping and smoothing are held at the published 0.001 / 10.0 rather than
    reselected, so the background is the only difference between the two runs.
    No test-set selection occurs in either.
  * Only the 175 training targets contribute to the background. Held-out
    targets contribute nothing.
  * Residual formulation is the published one, t_obs - t_FSM(s_0), with the FSM
    reference times recomputed on the forward grid under the new background.

Two controls run first and the script aborts if either fails:
  1. Rays re-traced through the configured background must reproduce the
     deposited G_ref geometry.
  2. An end-to-end run with the configured background must reproduce the
     published reference_ray/test_metrics.csv.

Environment. This needs the recorded analysis environment, not the document
tooling: CPython 3.12 with ttcrpy 1.4.2 and numpy 1.26.4, per
reproducibility/environment_metadata.txt. ttcrpy additionally requires scipy and
vtk. The run takes roughly 40 minutes, almost all of it FSM solves.

    python code/scripts/run_matched_prior_reference_ray.py [--package DIR]

A note on one non-obvious contract: the reference velocity grid must carry
metadata["layered_model"]. With velocity_interpolation
"piecewise_constant_interfaces" the pseudo-bending tracer reads the interfaces
from it; omit the key and the tracer has no layers, returns a straight
source-receiver ray, and G_ref is silently wrong while every velocity value
remains correct.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/src"))
DEFAULT_PACKAGE = (REPO_ROOT / "outputs/generated/submission_readiness"
                   / "applied_sciences_final_submission")

from tomobench.config import load_settings  # noqa: E402
from tomobench.domain.geometry import Point3D  # noqa: E402
from tomobench.domain.schemas import CartesianVelocityGrid3D  # noqa: E402
from tomobench.evaluation.benchmark_v2_final_analysis import (  # noqa: E402
    RayGeometry, cell_centered_values_from_node_vector, ray_path_cell_lengths,
    solve_fixed_ray_case,
)
from tomobench.evaluation.robustness_checks import _load_geometry  # noqa: E402
from tomobench.simulation.ray_tracing import trace_pseudo_bending_ray  # noqa: E402
from tomobench.simulation.ttcrpy_forward import (  # noqa: E402
    TtcrpyGridConfiguration, TtcrpyRectilinearForwardSolver,
)

DAMPING, SMOOTHING = 0.001, 10.0
VELOCITY_BOUNDS = (3.0, 8.0)
CELL_SHAPE = (40, 40, 12)
BOOTSTRAP_RESAMPLES = 10000
OUT_REL = "records/reference_ray_training_prior"

FSM_CONFIG = TtcrpyGridConfiguration(
    method="FSM", n_threads=1, cell_slowness=False, tt_from_rp=False, interp_vel=False,
    eps=1.0e-5, maxit=100, weno=True, nsnx=5, nsny=5, nsnz=5,
    n_secondary=2, n_tertiary=2, radius_factor_tertiary=3.0, translate_grid=False,
)


def stable_seed(label):
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8],
                          "little", signed=False) % (2 ** 32 - 1)


def bootstrap_ci(values, label, resamples=BOOTSTRAP_RESAMPLES):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed(label))
    m = rng.choice(a, size=(resamples, a.size), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def family_ci(values, families, label, resamples=BOOTSTRAP_RESAMPLES):
    grouped = defaultdict(list)
    for f, v in zip(families, values):
        grouped[f].append(float(v))
    rng = np.random.default_rng(stable_seed(label))
    arrays = [np.asarray(grouped[k], dtype=float) for k in sorted(grouped)]
    draws = [rng.choice(a, size=resamples, replace=True) for a in arrays]
    m = np.mean(np.vstack(draws), axis=0)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = ap.parse_args(argv)
    pkg = args.package.resolve()
    settings = load_settings()
    out_dir = pkg / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    print("package : %s" % pkg)

    man = list(csv.DictReader(
        (pkg / "records/benchmark_v2_target_split_manifest.csv").open(
            encoding="utf-8", newline="")))
    split = {r["target_id"]: r["split"] for r in man}
    family = {r["target_id"]: r["family"] for r in man}
    phys = json.loads(man[0]["physical_parameters_json"])["common_background_layered_model"]
    boundaries = phys["depth_boundaries_km"]
    configured = phys["velocities_km_per_s"]

    def layer_of(depth):
        for i in range(len(configured)):
            if boundaries[i] <= depth < boundaries[i + 1]:
                return i
        return len(configured) - 1

    with np.load(pkg / "records/corpus/forward_grids/faulted_target_07.npz") as d:
        fwd = (d["x_km"].astype(float), d["y_km"].astype(float), d["z_km"].astype(float))
    with np.load(pkg / "records/corpus/targets/faulted_target_07.npz") as d:
        tgt = (d["x_km"].astype(float), d["y_km"].astype(float), d["z_km"].astype(float))
    node_shape = (len(tgt[0]), len(tgt[1]), len(tgt[2]))

    def node_grid(vels, coords, grid_id):
        x, y, z = coords
        vals = []
        for depth in z:
            vals.extend([float(vels[layer_of(depth)])] * (len(x) * len(y)))
        return CartesianVelocityGrid3D(
            grid_id=grid_id, source_velocity_model_id=grid_id,
            x_coordinates_km=tuple(x.tolist()), y_coordinates_km=tuple(y.tolist()),
            z_coordinates_km=tuple(z.tolist()), p_velocity_km_per_s=tuple(vals),
            metadata={
                "artifact_type": "layered_reference_cartesian_p_velocity_grid_3d",
                "coordinate_system": "cartesian_km",
                "velocity_representation": "node_centered",
                "grid_shape": {"nx": len(x), "ny": len(y), "nz": len(z)},
                # Load-bearing: the tracer reads its interfaces from here.
                "layered_model": {"depth_boundaries_km": list(boundaries),
                                  "velocities_km_per_s": [float(v) for v in vels]},
            })

    # ------------------------------------------------ training-only background
    train = sorted(t for t in split if split[t] == "train")
    layer_index = np.array([layer_of(z) for z in tgt[2]])
    sums = np.zeros(len(configured))
    counts = np.zeros(len(configured))
    for t in train:
        with np.load(pkg / ("records/corpus/targets/%s.npz" % t)) as d:
            g = d["p_velocity_km_per_s"].astype(float).reshape(
                node_shape[2], node_shape[1], node_shape[0])
        for li in range(len(configured)):
            sel = layer_index == li
            if sel.any():
                sums[li] += g[sel].sum()
                counts[li] += g[sel].size
    training = (sums / counts).tolist()
    print("training-estimated layer velocities (%d training targets): %s"
          % (len(train), ["%.4f" % v for v in training]))

    def observations(tid):
        rows = list(csv.DictReader(
            (pkg / ("records/corpus/observations/%s_observations.csv" % tid)).open(
                encoding="utf-8", newline="")))
        return np.array([[float(r["source_x_km"]), float(r["source_y_km"]),
                          float(r["source_z_km"]), float(r["receiver_x_km"]),
                          float(r["receiver_y_km"]), float(r["receiver_z_km"]),
                          0.0, float(r["travel_time_s"])] for r in rows], dtype=float)

    class Shim:
        def __init__(self, obs):
            self.observations = obs

    def trace_geometry(grid, obs):
        row_ptr, cells, lengths, conv = [0], [], [], 0
        for row in obs:
            ray = trace_pseudo_bending_ray(
                source=Point3D(x_km=row[0], y_km=row[1], z_km=row[2]),
                receiver=Point3D(x_km=row[3], y_km=row[4], z_km=row[5]),
                velocity_grid=grid, settings=settings.pseudo_bending_solver)
            conv += int(ray.converged)
            for ci, ln in sorted(ray_path_cell_lengths(tuple(ray.ray_path), grid).items()):
                cells.append(int(ci))
                lengths.append(float(ln))
            row_ptr.append(len(cells))
        coverage = np.zeros(int(np.prod([v - 1 for v in node_shape])))
        if cells:
            np.add.at(coverage, np.asarray(cells, dtype=np.int64), np.asarray(lengths))
        return RayGeometry(row_ptr=np.asarray(row_ptr, dtype=np.int64),
                           cell_indices=np.asarray(cells, dtype=np.int64),
                           path_lengths_km=np.asarray(lengths), coverage_km=coverage,
                           converged_count=conv, observation_count=len(obs), runtime_s=0.0)

    def fsm_times(solver, obs):
        n = obs.shape[0]
        out = np.empty(n)
        for s in range(0, n, 16):
            e = min(s + 16, n)
            out[s:e] = np.asarray(
                solver.raytrace(obs[s:s + 1, 0:3], obs[s:e, 3:6],
                                aggregate_src=True).travel_times_s, dtype=float)
        return out

    test_ids = sorted(t for t in split if split[t] == "test")
    published = {r["target_id"]: r for r in csv.DictReader(
        (pkg / "records/reference_ray/test_metrics.csv").open(encoding="utf-8", newline=""))}

    cfg_target = node_grid(configured, tgt, "configured_target")

    # ------------------------------------------------ control 1: geometry
    print("control 1: re-traced geometry vs deposited geometry")
    worst = 0.0
    for tid in test_ids[:3]:
        mine = trace_geometry(cfg_target, observations(tid))
        dep = _load_geometry(
            pkg / ("records/operator_consistency_support/reference_ray_geometry/%s.npz" % tid))
        if not (np.array_equal(mine.row_ptr, dep.row_ptr)
                and np.array_equal(mine.cell_indices, dep.cell_indices)):
            raise SystemExit("control 1 failed: geometry structure differs for %s" % tid)
        worst = max(worst, float(np.abs(mine.path_lengths_km - dep.path_lengths_km).max()))
    if worst > 1e-9:
        raise SystemExit("control 1 failed: max |dL| %.3e km" % worst)
    print("   OK  max |dL| %.2e km" % worst)

    # ------------------------------------------------ control 2: end-to-end
    print("control 2: end-to-end configured run vs published test metrics")
    cfg_solver = TtcrpyRectilinearForwardSolver.from_cartesian_grid(
        node_grid(configured, fwd, "configured_forward"), FSM_CONFIG)
    cfg_cells = cell_centered_values_from_node_vector(
        np.asarray(cfg_target.p_velocity_km_per_s, dtype=float), node_shape)
    worst = 0.0
    for tid in test_ids:
        obs = observations(tid)
        with np.load(pkg / ("records/reference_ray/predictions/%s.npz" % tid)) as p:
            truth = p["target_cells"].astype(float)
        solved = solve_fixed_ray_case(
            case=Shim(obs), geometry=trace_geometry(cfg_target, obs),
            reference_cells=cfg_cells, target_cells=truth, damping=DAMPING,
            smoothing=SMOOTHING, cell_shape=CELL_SHAPE, velocity_bounds=VELOCITY_BOUNDS,
            reference_travel_times_s=fsm_times(cfg_solver, obs))
        worst = max(worst, abs(solved["all_cell_rmse"]
                               - float(published[tid]["direct_cell_rmse_km_per_s"])))
    if worst > 1e-7:
        raise SystemExit("control 2 failed: max deviation %.3e km/s" % worst)
    print("   OK  max deviation %.2e km/s (PCG tolerance)" % worst)

    # ------------------------------------------------ matched-prior run
    print("matched-prior run")
    trn_target = node_grid(training, tgt, "training_target")
    trn_solver = TtcrpyRectilinearForwardSolver.from_cartesian_grid(
        node_grid(training, fwd, "training_forward"), FSM_CONFIG)
    trn_cells = cell_centered_values_from_node_vector(
        np.asarray(trn_target.p_velocity_km_per_s, dtype=float), node_shape)

    rows = []
    for tid in test_ids:
        obs = observations(tid)
        with np.load(pkg / ("records/reference_ray/predictions/%s.npz" % tid)) as p:
            truth = p["target_cells"].astype(float)
        solved = solve_fixed_ray_case(
            case=Shim(obs), geometry=trace_geometry(trn_target, obs),
            reference_cells=trn_cells, target_cells=truth, damping=DAMPING,
            smoothing=SMOOTHING, cell_shape=CELL_SHAPE, velocity_bounds=VELOCITY_BOUNDS,
            reference_travel_times_s=fsm_times(trn_solver, obs))
        rows.append({
            "target_id": tid, "family": family[tid], "split": "test",
            "method_id": "reference_ray_training_prior",
            "direct_cell_rmse_km_per_s": repr(float(solved["all_cell_rmse"])),
            "direct_cell_mae_km_per_s": repr(float(solved["all_cell_mae"])),
            "pcg_iterations": solved["pcg_iterations"],
            "pcg_converged": solved["pcg_converged"],
            "clipped_fraction": repr(float(solved["clipped_fraction"])),
        })

    with (out_dir / "test_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    trn = {r["target_id"]: float(r["direct_cell_rmse_km_per_s"]) for r in rows}
    ref = {t: float(published[t]["direct_cell_rmse_km_per_s"]) for t in test_ids}
    node = {r["target_id"]: float(r["direct_cell_rmse_km_per_s"])
            for r in csv.DictReader((pkg / "records/node_native/case_metrics.csv").open(
                encoding="utf-8", newline=""))
            if r["split"] == "test" and r["method_id"] == "realistic_full"}
    cell = {r["target_id"]: float(r["direct_cell_rmse_km_per_s"])
            for r in csv.DictReader((pkg / "records/cell_native/case_metrics.csv").open(
                encoding="utf-8", newline="")) if r["split"] == "test"}
    fams = [family[t] for t in test_ids]

    comparisons = {
        "full_vs_training_prior": (node, trn),
        "cell_native_vs_training_prior": (cell, trn),
        "training_prior_vs_reference_ray": (trn, ref),
    }
    paired_rows = []
    summary_pairs = {}
    for cid, (first, second) in comparisons.items():
        deltas = [first[t] - second[t] for t in test_ids]
        lo, hi = bootstrap_ci(deltas, "matched-prior:" + cid)
        flo, fhi = family_ci(deltas, fams, "matched-prior-family:" + cid)
        tol = 1.0e-12
        summary_pairs[cid] = {
            "mean_delta": float(np.mean(deltas)),
            "ci95_lower": lo, "ci95_upper": hi,
            "family_stratified_ci95_lower": flo, "family_stratified_ci95_upper": fhi,
            "wins": int(sum(1 for d in deltas if d < -tol)),
            "losses": int(sum(1 for d in deltas if d > tol)),
            "ties": int(sum(1 for d in deltas if abs(d) <= tol)),
            "delta_definition": "first method direct-cell RMSE minus second; negative favors first",
        }
        for t, d in zip(test_ids, deltas):
            paired_rows.append({"comparison_id": cid, "target_id": t, "family": family[t],
                                "paired_delta": repr(float(d))})

    with (out_dir / "paired_deltas.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["comparison_id", "target_id", "family",
                                           "paired_delta"], lineterminator="\n")
        w.writeheader()
        w.writerows(paired_rows)

    by_family = {}
    for f in sorted(set(fams)):
        sel = [t for t in test_ids if family[t] == f]
        by_family[f] = {
            "targets": len(sel),
            "training_prior_rmse_mean": float(np.mean([trn[t] for t in sel])),
            "reference_ray_rmse_mean": float(np.mean([ref[t] for t in sel])),
            "mean_delta_training_minus_reference": float(np.mean([trn[t] - ref[t] for t in sel])),
        }

    summary = {
        "method_id": "reference_ray_training_prior",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "background_source": "mean over all nodes of the 175 training targets within each "
                             "configured depth layer",
        "layer_depth_boundaries_km": list(boundaries),
        "configured_layer_velocities_km_per_s": list(configured),
        "training_layer_velocities_km_per_s": training,
        "regularization_policy": "damping and smoothing held at the published selection; not "
                                 "reselected, so the background is the only difference",
        "selected_damping": DAMPING,
        "selected_smoothing": SMOOTHING,
        "residual_formula": "t_obs - t_FSM(s_0) with FSM reference times recomputed on the "
                            "forward grid under the training background",
        "retained_privilege": "the layer depth boundaries are retained from the configured "
                              "model; only the five layer velocities are estimated from data",
        "leakage_control": "only training targets contribute to the background; no validation "
                           "or test target is used",
        "test_rmse_mean_km_per_s": float(np.mean([trn[t] for t in test_ids])),
        "test_mae_mean_km_per_s": float(np.mean([float(r["direct_cell_mae_km_per_s"])
                                                 for r in rows])),
        "reference_ray_test_rmse_mean_km_per_s": float(np.mean([ref[t] for t in test_ids])),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_unit": "unique geological target",
        "paired_comparisons": summary_pairs,
        "family_resolved": by_family,
        "all_pcg_converged": all(r["pcg_converged"] for r in rows),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="")

    print("   test RMSE mean %.5f (published exact-background %.5f)"
          % (summary["test_rmse_mean_km_per_s"],
             summary["reference_ray_test_rmse_mean_km_per_s"]))
    for cid, s in summary_pairs.items():
        print("   %-32s %+.5f [%+.5f, %+.5f]  W/L/T %d/%d/%d"
              % (cid, s["mean_delta"], s["ci95_lower"], s["ci95_upper"],
                 s["wins"], s["losses"], s["ties"]))
    print("wrote %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
