"""Shallow MLP coefficient-map baseline under the published Full-input contract.

Implements the protocol declared in
wiki/decisions/2026-09-23_mlp-coefficient-map-protocol.md (committed before this
model touched any benchmark_v2 data). Exactly one element of the declared
Full-input workflow changes: the linear ridge map from the 3,072 standardized
observation features to the PCA target coefficients is replaced by a one-hidden-
layer MLP. The split, features, standardization, training-only PCA, K grid,
clipping, node-to-cell conversion, metrics and bootstrap are the published ones.

The corpus loader and the two reproduction controls are imported from
run_observation_ordering_sensitivity.py rather than copied, so the MLP is judged
by the same code that reproduces the published PCA-ridge numbers. A third
control requires two same-seed fits to agree exactly. If any control fails,
no MLP number is produced.

The test set is evaluated once, for the selected configuration only: the grid
loop keeps only the current best configuration's networks, iterating in the
declared tie-break order, so no other configuration's test error is ever
computed.

    python code/scripts/run_mlp_coefficient_map_baseline.py --package DIR [--out FILE]
"""

import argparse
import csv
import json
import pathlib
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_observation_ordering_sensitivity as ros  # noqa: E402

PROTOCOL = "wiki/decisions/2026-09-23_mlp-coefficient-map-protocol.md"
KS = ros.KS
WIDTHS = [16, 64]
LAMBDAS_TIE_ORDER = [1000.0, 10.0, 0.1]   # larger lambda first: the declared tie-break
MEMBERS = 5
EPOCHS = 1000
LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8
BOOT = 10000


def seed_label(K, H, lam, member):
    return "mlp-coefficient-map:init:K=%d:H=%d:lambda=%s:member=%d" % (K, H, repr(lam), member)


def fit_mlp(X, T, H, lam, seed):
    """Full-batch Adam on (1/2n)*sum sq err + (lam/2n)*(|W1|^2+|W2|^2); tanh hidden layer."""
    n, d = X.shape
    k = T.shape[1]
    rng = np.random.default_rng(seed)
    l1 = np.sqrt(6.0 / (d + H))
    l2 = np.sqrt(6.0 / (H + k))
    W1 = rng.uniform(-l1, l1, (d, H))
    W2 = rng.uniform(-l2, l2, (H, k))
    b1 = np.zeros(H)
    b2 = np.zeros(k)
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for ep in range(1, EPOCHS + 1):
        A = np.tanh(X @ W1 + b1)
        E = (A @ W2 + b2 - T) / n
        gW2 = A.T @ E + (lam / n) * W2
        gb2 = E.sum(axis=0)
        dA = (E @ W2.T) * (1.0 - A * A)
        gW1 = X.T @ dA + (lam / n) * W1
        gb1 = dA.sum(axis=0)
        for i, g in enumerate((gW1, gb1, gW2, gb2)):
            m[i] = B1 * m[i] + (1.0 - B1) * g
            v[i] = B2 * v[i] + (1.0 - B2) * g * g
            params[i] -= LR * (m[i] / (1.0 - B1 ** ep)) / (np.sqrt(v[i] / (1.0 - B2 ** ep)) + EPS)
    return params


def predict(params, X):
    W1, b1, W2, b2 = params
    return np.tanh(X @ W1 + b1) @ W2 + b2


def node_rmse(pred, truth):
    return float(np.mean(np.sqrt(((pred - truth) ** 2).mean(axis=1))))


def pooled_ci(deltas, label):
    rng = np.random.default_rng(ros.stable_seed(label))
    n = len(deltas)
    draws = np.array([deltas[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def stratified_ci(deltas, families, label):
    """Resample targets within each family; equally weighted mean of the family means."""
    fams = sorted(set(families))
    idx = {f: np.array([i for i, g in enumerate(families) if g == f]) for f in fams}
    point = float(np.mean([deltas[idx[f]].mean() for f in fams]))
    rng = np.random.default_rng(ros.stable_seed(label))
    draws = np.empty(BOOT)
    for b in range(BOOT):
        draws[b] = np.mean([deltas[idx[f][rng.integers(0, len(idx[f]), len(idx[f]))]].mean()
                            for f in fams])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)
    pkg = args.package.resolve()
    if pkg.name == "applied_sciences_final_submission":
        raise SystemExit("refusing to write into the frozen submitted package")
    out = args.out or (pkg / "records/secondary_diagnostics/mlp_coefficient_map"
                       / "mlp_coefficient_map_baseline.json")
    print("package : %s" % pkg)
    print("python %s, numpy %s" % (platform.python_version(), np.__version__))
    print()

    corpus = ros.Corpus(pkg)
    ridge = ros.control(corpus)                      # controls 1 and 2

    tr, va, te = corpus.split["train"], corpus.split["validation"], corpus.split["test"]
    X = {s: np.array([corpus.features(t, "identifier") for t in names])
         for s, names in (("train", tr), ("validation", va), ("test", te))}
    Y = {s: np.array([corpus.nodes(t) for t in names])
         for s, names in (("train", tr), ("validation", va))}
    Xtr, Xva, Xte = ros.standardize(X["train"], [X["validation"], X["test"]])
    ybar = Y["train"].mean(axis=0)
    _, _, Vt = np.linalg.svd(Y["train"] - ybar, full_matrices=False)

    print("CONTROL 3 - determinism: two same-seed fits must agree exactly")
    U1 = Vt[:1]
    C1 = (Y["train"] - ybar) @ U1.T
    T1 = C1 / C1[:, 0].std()
    s3 = ros.stable_seed("mlp-coefficient-map:control-3")
    a = predict(fit_mlp(Xtr, T1, 16, 10.0, s3), Xtr)
    b = predict(fit_mlp(Xtr, T1, 16, 10.0, s3), Xtr)
    if not np.array_equal(a, b):
        raise SystemExit("control 3 failed: same-seed fits differ, so no MLP result may be produced")
    print("   identical")
    print("   ALL THREE CONTROLS PASS")
    print()

    print("GRID - %d configurations x %d members, %d epochs each"
          % (len(KS) * len(WIDTHS) * len(LAMBDAS_TIE_ORDER), MEMBERS, EPOCHS))
    grid, best, best_nets = [], None, None
    t0 = time.perf_counter()
    for K in KS:
        Uk = Vt[:K]
        Ck = (Y["train"] - ybar) @ Uk.T
        scale = float(Ck[:, 0].std())
        Tk = Ck / scale
        for H in WIDTHS:
            for lam in LAMBDAS_TIE_ORDER:
                nets = [fit_mlp(Xtr, Tk, H, lam, ros.stable_seed(seed_label(K, H, lam, m)))
                        for m in range(MEMBERS)]
                coef = np.mean([predict(p, Xva) for p in nets], axis=0) * scale
                score = node_rmse(np.clip(ybar + coef @ Uk, ros.LO, ros.HI), Y["validation"])
                grid.append({"K": K, "H": H, "lambda": lam, "validation_node_rmse_km_per_s": score})
                if best is None or score < best["validation_node_rmse_km_per_s"]:
                    best = {"K": K, "H": H, "lambda": lam, "validation_node_rmse_km_per_s": score,
                            "coefficient_scale": scale}
                    best_nets = nets
                print("   K=%3d H=%2d lambda=%-6s val %.5f   (%.0f s)"
                      % (K, H, lam, score, time.perf_counter() - t0), flush=True)
    print("SELECTED K=%d H=%d lambda=%s, validation node RMSE %.5f"
          % (best["K"], best["H"], best["lambda"], best["validation_node_rmse_km_per_s"]))
    print()

    # Test, exactly once, for the selected configuration only.
    Uk = Vt[:best["K"]]
    coef = np.mean([predict(p, Xte) for p in best_nets], axis=0) * best["coefficient_scale"]
    node_pred = np.clip(ybar + coef @ Uk, ros.LO, ros.HI)
    truth = corpus.direct_cell_truth()
    per_target = {t: float(np.sqrt(((ros.to_cells(node_pred[i]) - truth[t]) ** 2).mean()))
                  for i, t in enumerate(te)}
    test_mean = float(np.mean(list(per_target.values())))
    print("TEST direct-cell RMSE %.5f" % test_mean)

    fam = {r["target_id"]: r["family"] for r in csv.DictReader(
        (corpus.rec / "node_native/case_metrics.csv").read_text(encoding="utf-8").splitlines())
        if r["split"] == "test"}
    rr = {r["target_id"]: float(r["direct_cell_rmse_km_per_s"]) for r in csv.DictReader(
        (corpus.rec / "reference_ray/test_metrics.csv").read_text(encoding="utf-8").splitlines())}
    ridge_t = ridge["per_target_test_direct_cell_rmse_km_per_s"]
    comparisons = {}
    for name, other in (("mlp_minus_reference_ray", rr), ("mlp_minus_pca_ridge", ridge_t)):
        ids = sorted(set(per_target) & set(other))
        if len(ids) != 38:
            raise SystemExit("%s: paired on %d targets, expected 38" % (name, len(ids)))
        d = np.array([per_target[t] - other[t] for t in ids])
        fl = [fam[t] for t in ids]
        lo, hi = pooled_ci(d, "mlp-coefficient-map:%s" % name)
        ep, slo, shi = stratified_ci(d, fl, "mlp-coefficient-map:%s:stratified" % name)
        comparisons[name] = {
            "n_targets": 38, "mean_delta_km_per_s": float(d.mean()),
            "ci95_lower": lo, "ci95_upper": hi,
            "equal_family_mean_km_per_s": ep,
            "family_stratified_ci95_lower": slo, "family_stratified_ci95_upper": shi,
            "family_means_km_per_s": {f: float(np.mean([d[i] for i, g in enumerate(fl) if g == f]))
                                      for f in sorted(set(fl))},
            "mlp_better_count": int((d < 0).sum()), "mlp_worse_count": int((d > 0).sum()),
            "seed_labels": ["mlp-coefficient-map:%s" % name,
                            "mlp-coefficient-map:%s:stratified" % name],
            "bootstrap_resamples": BOOT,
            "sign": "positive favors the second method",
        }
        print("PAIRED %-26s %+.5f  CI [%+.5f, %+.5f]  equal-family %+.5f  CI [%+.5f, %+.5f]  "
              "MLP better on %d/38" % (name, d.mean(), lo, hi, ep, slo, shi, int((d < 0).sum())))

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "mlp_coefficient_map_baseline_v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": PROTOCOL,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "contract": {
            "changed_element": "coefficient map: ridge replaced by a one-hidden-layer tanh MLP",
            "held_fixed": "split, identifier-ordered 3,072-value features and their standardization, "
                          "training-only PCA, K grid, clipping 3.0-8.0 km/s, eight-corner "
                          "node-to-cell conversion, validation node RMSE selection, test "
                          "direct-cell RMSE, percentile bootstrap",
            "mlp": {"hidden_activation": "tanh", "init": "Glorot uniform, zero biases",
                    "loss": "(1/2n) sum sq err + (lambda/2n)(|W1|^2 + |W2|^2)",
                    "optimizer": "full-batch Adam", "learning_rate": LR, "beta1": B1,
                    "beta2": B2, "epsilon": EPS, "epochs": EPOCHS,
                    "output_scaling": "all K coefficients divided by the population sd of the "
                                      "first training coefficient",
                    "ensemble_members": MEMBERS,
                    "seed_label_pattern": "mlp-coefficient-map:init:K={K}:H={H}:lambda={lambda}"
                                          ":member={m}"},
            "grid": {"K": KS, "H": WIDTHS, "lambda": sorted(LAMBDAS_TIE_ORDER)},
            "tie_break": "smaller K, then smaller H, then larger lambda",
        },
        "controls": {"published_validation_grid_reproduced": True,
                     "published_test_direct_cell_rmse_reproduced": ros.PUBLISHED_TEST_RMSE,
                     "same_seed_fits_identical": True},
        "validation_grid": grid,
        "selected": best,
        "test_direct_cell_rmse_mean_km_per_s": test_mean,
        "per_target_test_direct_cell_rmse_km_per_s": per_target,
        "paired_comparisons": comparisons,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print()
    print("wrote   : %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
