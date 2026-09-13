"""Does the benchmark result depend on the arbitrary observation row ordering?

The primary Full-input representation is a fixed-length 3,072-value vector built
from 384 observations sorted by source, station and observation identifier
(Section 2.5). Because acquisition geometry is redrawn independently for every
target, position j in that vector denotes an identifier index and not a fixed
physical ray, so a linear map's coefficient at position j refers partly to an
arbitrary label. The third blind review asked whether the reported observation-
information effect depends on that ordering. The shuffled-time control does not
answer it: it destroys the target-acquisition association rather than re-ordering
an information-preserving encoding.

This runs two re-orderings under the published contract:

  geometry   Each case's 384 complete eight-field tuples are re-sorted by
             (source x, y, z, receiver x, y, z, distance). Tuples stay intact,
             so no information is added or removed, but feature positions
             acquire a geometric meaning that is comparable across targets
             instead of an arbitrary one. This is the scientifically
             informative variant.
  global     One fixed permutation of the 384 tuple slots, applied identically
             to every case. A linear model with per-column standardization
             should be invariant to this up to numerical precision, so it is an
             implementation sanity check rather than a scientific result.

Each variant is fitted on training targets only, has K and alpha selected on the
37 validation targets only, and is evaluated once on the 38 test targets. The
script refuses to proceed unless two controls pass first: the 110-row published
validation grid must reproduce, and the published Full-input test direct-cell
RMSE must reproduce. Without those, a new number from this code means nothing.

    python code/scripts/run_observation_ordering_sensitivity.py [--package DIR]
                                                                [--out FILE]
"""

import argparse
import csv
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = (REPO_ROOT / "outputs/generated/submission_readiness"
                   / "applied_sciences_final_submission")

FIELDS = ["source_x_km", "source_y_km", "source_z_km",
          "receiver_x_km", "receiver_y_km", "receiver_z_km",
          "euclidean_distance_km", "travel_time_s"]
KS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
LO, HI = 3.0, 8.0
NX, NY, NZ = 41, 41, 13
PUBLISHED_TEST_RMSE = 0.35341
GLOBAL_PERMUTATION_SEED = "observation-ordering:global-permutation"


def stable_seed(label):
    """The package's deterministic seed convention: sha256 of a stated label."""
    import hashlib
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8],
                          "little", signed=False) % (2 ** 32 - 1)


class Corpus:
    def __init__(self, package):
        self.pkg = package
        self.rec = package / "records"
        rows = list(csv.DictReader(
            (self.rec / "benchmark_v2_target_split_manifest.csv").read_text(
                encoding="utf-8").splitlines()))
        tid = next(c for c in rows[0] if "target_id" in c)
        spl = next(c for c in rows[0] if c == "split" or c.endswith("split"))
        self.split = {}
        for r in rows:
            self.split.setdefault(r[spl], []).append(r[tid])
        for k in self.split:
            self.split[k] = sorted(self.split[k])
        self._obs = {}

    def tuples(self, target_id):
        """The 384 eight-field tuples in published identifier order."""
        if target_id not in self._obs:
            p = self.rec / "corpus/observations" / ("%s_observations.csv" % target_id)
            rows = list(csv.DictReader(p.read_text(encoding="utf-8").splitlines()))
            rows.sort(key=lambda r: (r["earthquake_id"], r["station_id"], r["observation_id"]))
            self._obs[target_id] = np.array(
                [[float(r[f]) for f in FIELDS] for r in rows], dtype=float)
        return self._obs[target_id]

    def features(self, target_id, order, permutation=None):
        t = self.tuples(target_id)
        if order == "identifier":
            pass
        elif order == "geometry":
            # Lexicographic on the seven geometry fields; travel time excluded so
            # the ordering is a function of acquisition geometry alone.
            t = t[np.lexsort(tuple(t[:, i] for i in range(6, -1, -1)))]
        elif order == "global":
            t = t[permutation]
        else:
            raise ValueError(order)
        return t.ravel()

    def nodes(self, target_id):
        with np.load(self.rec / "corpus/targets" / ("%s.npz" % target_id)) as z:
            return np.asarray(z["p_velocity_km_per_s"], dtype=float)

    def direct_cell_truth(self):
        """Direct analytic cell-center truth per test target.

        The deposited directory holds 38 per-target files plus one stacked
        test_predictions.npz bundle; the bundle keys targets as `target_ids` and
        the per-target files as `target_id`. Both are read and required to agree,
        which is a free cross-check on the truth field.
        """
        out, bundle = {}, {}
        for p in sorted((self.rec / "node_native_direct_cell_selection/predictions").glob("*.npz")):
            with np.load(p) as z:
                if "direct_analytic_cell_truth_km_per_s" not in z.files:
                    continue
                truth = np.asarray(z["direct_analytic_cell_truth_km_per_s"], dtype=float)
                if "target_id" in z.files:
                    out[str(z["target_id"])] = truth
                elif "target_ids" in z.files:
                    for i, t in enumerate(z["target_ids"]):
                        bundle[str(t)] = truth[i]
        for t, v in bundle.items():
            if t in out:
                if not np.array_equal(out[t], v):
                    raise SystemExit("the per-target and bundled direct analytic cell truth "
                                     "disagree for %s" % t)
            else:
                out[t] = v
        return out


def to_cells(node_vector):
    """Eight-corner arithmetic average, 41x41x13 nodes -> 40x40x12 cells."""
    v = node_vector.reshape(NZ, NY, NX)
    out = np.zeros((NZ - 1, NY - 1, NX - 1))
    for a in range(2):
        for b in range(2):
            for c in range(2):
                out += v[a:a + NZ - 1, b:b + NY - 1, c:c + NX - 1]
    return (out / 8.0).ravel()


def standardize(train_raw, others):
    mu = train_raw.mean(axis=0)
    sd = train_raw.std(axis=0)                          # population sd
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return [(m - mu) / sd for m in [train_raw] + list(others)]


def coefficient_map(Xtr, Ctr, Xev, alpha):
    """Dual-form ridge (175 samples, 3,072 features); alpha None = min-norm lstsq."""
    if alpha is None:
        W, *_ = np.linalg.lstsq(Xtr, Ctr, rcond=None)
        return Xev @ W
    n = Xtr.shape[0]
    A = np.linalg.solve(Xtr @ Xtr.T + alpha * np.eye(n), Ctr)
    return (Xev @ Xtr.T) @ A


def run_variant(corpus, order, permutation=None):
    """Fit on train, select on validation, evaluate test once."""
    tr, va, te = corpus.split["train"], corpus.split["validation"], corpus.split["test"]
    X = {s: np.array([corpus.features(t, order, permutation) for t in names])
         for s, names in (("train", tr), ("validation", va), ("test", te))}
    Y = {s: np.array([corpus.nodes(t) for t in names])
         for s, names in (("train", tr), ("validation", va), ("test", te))}
    Xtr, Xva, Xte = standardize(X["train"], [X["validation"], X["test"]])
    ybar = Y["train"].mean(axis=0)
    _, _, Vt = np.linalg.svd(Y["train"] - ybar, full_matrices=False)

    grid = {}
    for K in KS:
        Uk = Vt[:K]
        Ctr = (Y["train"] - ybar) @ Uk.T
        for alpha in [None] + ALPHAS:
            pred = np.clip(ybar + coefficient_map(Xtr, Ctr, Xva, alpha) @ Uk, LO, HI)
            grid[("pca_linear" if alpha is None else "pca_ridge", K, alpha)] = float(
                np.mean(np.sqrt(((pred - Y["validation"]) ** 2).mean(axis=1))))

    best = min(grid, key=grid.get)
    Uk = Vt[:best[1]]
    Ctr = (Y["train"] - ybar) @ Uk.T
    node_pred = np.clip(ybar + coefficient_map(Xtr, Ctr, Xte, best[2]) @ Uk, LO, HI)
    truth = corpus.direct_cell_truth()
    missing = [t for t in te if t not in truth]
    if missing:
        raise SystemExit("no deposited direct analytic cell truth for %d test target(s): %s"
                         % (len(missing), missing[:3]))
    per_target = {t: float(np.sqrt(((to_cells(node_pred[i]) - truth[t]) ** 2).mean()))
                  for i, t in enumerate(te)}
    return {
        "selected_model": best[0],
        "selected_component_count": best[1],
        "selected_ridge_alpha": best[2],
        "validation_node_rmse_mean_km_per_s": grid[best],
        "test_direct_cell_rmse_mean_km_per_s": float(np.mean(list(per_target.values()))),
        "per_target_test_direct_cell_rmse_km_per_s": per_target,
        "validation_grid": {"%s|K=%d|alpha=%s" % k: v for k, v in sorted(
            grid.items(), key=lambda kv: (kv[0][0], kv[0][1], -1 if kv[0][2] is None
                                          else kv[0][2]))},
    }


def control(corpus):
    """Refuse to publish anything unless the published workflow reproduces."""
    print("CONTROL 1 - the published 110-row validation grid")
    published = {}
    for r in csv.DictReader(
            (corpus.rec / "node_native/validation_model_selection.csv").read_text(
                encoding="utf-8").splitlines()):
        if r["feature_variant"] == "realistic_full":
            published[(r["model_name"], int(r["pca_component_count"]),
                       None if not r["ridge_alpha"] else float(r["ridge_alpha"]))] = float(
                r["validation_node_rmse_mean_km_per_s"])
    ref = run_variant(corpus, "identifier")
    worst = max(abs(ref["validation_grid"]["%s|K=%d|alpha=%s" % k] - v)
                for k, v in published.items())
    print("   %d rows compared, worst absolute mismatch %.3e" % (len(published), worst))
    if worst > 1e-9:
        raise SystemExit("control 1 failed: the reimplementation does not reproduce the "
                         "published validation grid, so no ordering result may be published")

    print("CONTROL 2 - the published Full-input test direct-cell RMSE")
    got = ref["test_direct_cell_rmse_mean_km_per_s"]
    print("   re-derived %.5f against published %.5f" % (got, PUBLISHED_TEST_RMSE))
    if abs(round(got, 5) - PUBLISHED_TEST_RMSE) > 5e-6:
        raise SystemExit("control 2 failed: test endpoint %.5f does not reproduce the published "
                         "%.5f" % (got, PUBLISHED_TEST_RMSE))
    if (ref["selected_model"], ref["selected_component_count"],
            ref["selected_ridge_alpha"]) != ("pca_ridge", 1, 1000.0):
        raise SystemExit("control 2 failed: validation selected %s, not pca_ridge K=1 alpha=1000"
                         % str((ref["selected_model"], ref["selected_component_count"],
                                ref["selected_ridge_alpha"])))
    print("   selection reproduces: pca_ridge, K=1, alpha=1000")
    print("   BOTH CONTROLS PASS")
    print()
    return ref


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", type=pathlib.Path, default=DEFAULT_PACKAGE)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="where to deposit the record (default: inside the package records/)")
    args = ap.parse_args(argv)
    package = args.package.resolve()
    out = args.out or (package / "records/secondary_diagnostics/observation_ordering"
                       / "observation_ordering_sensitivity.json")
    print("package : %s" % package)

    corpus = Corpus(package)
    reference = control(corpus)

    rng = np.random.default_rng(stable_seed(GLOBAL_PERMUTATION_SEED))
    permutation = rng.permutation(384)

    results = {"identifier": reference}
    for order, perm, label in (("geometry", None, "geometry-canonical per-case ordering"),
                               ("global", permutation, "one fixed permutation for every case")):
        print("VARIANT %s - %s" % (order, label))
        r = run_variant(corpus, order, perm)
        results[order] = r
        print("   selected %s K=%d alpha=%s; validation node RMSE %.5f"
              % (r["selected_model"], r["selected_component_count"],
                 r["selected_ridge_alpha"], r["validation_node_rmse_mean_km_per_s"]))
        print("   test direct-cell RMSE %.5f (identifier-ordered %.5f, difference %+.5f)"
              % (r["test_direct_cell_rmse_mean_km_per_s"],
                 reference["test_direct_cell_rmse_mean_km_per_s"],
                 r["test_direct_cell_rmse_mean_km_per_s"]
                 - reference["test_direct_cell_rmse_mean_km_per_s"]))
        print()

    # Paired comparisons at target level, with intervals deposited as pairs so
    # that a published interval traces to a record rather than to a re-run.
    ref_t = reference["per_target_test_direct_cell_rmse_km_per_s"]
    rr = {r["target_id"]: float(r["direct_cell_rmse_km_per_s"])
          for r in csv.DictReader(
              (corpus.rec / "reference_ray/test_metrics.csv").read_text(
                  encoding="utf-8").splitlines())}
    comparisons = {}
    for name, first, second in (
            ("geometry_minus_identifier",
             results["geometry"]["per_target_test_direct_cell_rmse_km_per_s"], ref_t),
            ("geometry_minus_reference_ray",
             results["geometry"]["per_target_test_direct_cell_rmse_km_per_s"], rr),
            ("identifier_minus_reference_ray", ref_t, rr)):
        ids = sorted(set(first) & set(second))
        if len(ids) != 38:
            raise SystemExit("%s: paired on %d targets, expected 38" % (name, len(ids)))
        deltas = np.array([first[t] - second[t] for t in ids])
        label = "observation-ordering:%s" % name
        rng = np.random.default_rng(stable_seed(label))
        draws = np.array([deltas[rng.integers(0, 38, 38)].mean() for _ in range(10000)])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        comparisons[name] = {
            "n_targets": 38,
            "mean_delta_km_per_s": float(deltas.mean()),
            "ci95_lower": float(lo),
            "ci95_upper": float(hi),
            "seed_label": label,
            "bootstrap_resamples": 10000,
            "first_worse_count": int((deltas > 0).sum()),
            "second_worse_count": int((deltas < 0).sum()),
            "sign": "positive favors the second method",
        }
        print("PAIRED %-32s %+.5f km/s  CI [%+.5f, %+.5f]  (%d/%d)"
              % (name, deltas.mean(), lo, hi,
                 int((deltas < 0).sum()), 38))
    print()

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "observation_ordering_sensitivity_v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": "Does the Full-input PCA-ridge result depend on the arbitrary identifier "
                    "ordering of the fixed-length observation vector?",
        "contract": {
            "fields_per_observation": FIELDS,
            "observations_per_case": 384,
            "feature_length": 384 * len(FIELDS),
            "orderings": {
                "identifier": "source identifier, station identifier, observation identifier; "
                              "the published primary representation",
                "geometry": "lexicographic on source x, y, z then receiver x, y, z then "
                            "Euclidean distance, per case, with complete tuples kept intact",
                "global": "one fixed permutation of the 384 tuple slots applied identically to "
                          "every case",
            },
            "global_permutation_seed_label": GLOBAL_PERMUTATION_SEED,
            "selection": "K and ridge alpha chosen on the 37 validation targets by mean "
                         "validation node RMSE; test evaluated once",
            "evaluation": "mean over 38 test targets of direct analytic cell-center RMSE after "
                          "elementwise clipping to 3.0-8.0 km/s and eight-corner node-to-cell "
                          "conversion",
        },
        "controls": {
            "published_validation_grid_reproduced": True,
            "published_test_direct_cell_rmse": PUBLISHED_TEST_RMSE,
        },
        "variants": results,
        "paired_comparisons": comparisons,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n",
                   encoding="utf-8", newline="\n")
    print("wrote   : %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
