"""
Student-level bootstrap for selection stability.

The repeated-CV stability values in temporal_design.py share one sample of
students, so their spread is partition noise only, not sampling variability.
Here each replicate half-samples students WITHOUT replacement (n/2 students;
all registrations of a drawn student come along). It then runs one 5-fold
StratifiedGroupKFold on the half-sample, ranks features with all five methods
on each training fold, and computes Nogueira stability for every k.

Why half-sampling, not the with-replacement bootstrap: duplicated rows put zero-distance neighbours into the kNN mutual-information
estimator and inflate MI for continuous features (+58% on day-30 data), biasing
MI's replicates only. Half-sampling has no duplicates and treats all five
rankers alike. It is the delete-d jackknife with d = n/2 (variance factor
(n - d)/d = 1; Shao & Wu 1989), applied heuristically: its consistency is proven
for smooth statistics and quantiles, not for this n-dependent selector
statistic. Each replicate also carries the noise of a single partition, so the
SE is expected to be conservative.

Reporting:
  - The point estimate is the CV estimate from `<name>_analysis.json` (mean of
    10 x 5-fold CV on all dev students). A replicate is one 5-fold split on n/2
    students (rankers fit on ~0.4n), a different estimand, so its mean is
    biased by design; the bias (bootstrap mean - CV estimate) is reported.
  - Inference uses the SE only, centred on the CV estimate: `ci95` =
    est +/- 1.96 SE; `robust` = it excludes 0. `robust_bonferroni` = est +/-
    z(1 - 0.05/20) SE excludes 0 (10 method pairs). The percentile interval of
    the replicates is descriptive only.
  - k-summaries: `grid` (K_GRID, fixed in advance) is the primary summary;
    `k_le_p3` and `all` are sensitivity summaries.

Only rankings are needed, so no evaluator is fitted. Integrity as in
temporal_design.py: clean tree, one code version (scripts/ tree hash) across
replicates and the CV analysis they are summarised against, checked at start
and at the end, the same data hash, a per-file lock, complete replicates. Run it
after temporal_design.py has written the analysis.
"""
import os
import json
import time
import argparse
from itertools import combinations

import numpy as np
from scipy.stats import norm
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from paths import DATA_DIR, RESULTS_DIR, require_data
from temporal_design import (RANKERS, METHODS, K_GRID, RANDOM_STATE, load_oulad_at,
                             split_dev_holdout, prepare_xapi, nogueira_stability, provenance,
                             fix_discrete, read_jsonl, lock, check_same_commit, data_manifest)

N_BOOT = 200


def k_sets(p):
    return {"grid": K_GRID, "k_le_p3": list(range(1, p // 3 + 1)), "all": list(range(1, p // 2 + 1))}


def replicate(X, y, groups, b, n_students=None, seed=None, rankers=None):
    """One half-sample replicate. stability_vs_n.py passes `n_students` (draw that
    many students instead of n/2; OULAD only), `seed` (student-draw seed, default
    RANDOM_STATE + b) and `rankers` ({name: fn}, default the five METHODS)."""
    rankers = rankers or {m: RANKERS[m] for m in METHODS}
    rng = np.random.default_rng(RANDOM_STATE + b if seed is None else seed)
    if groups is None:  # xAPI: one row per student
        assert n_students is None, "n_students is for OULAD"
        idx = np.sort(rng.choice(len(X), len(X) // 2, replace=False))
        g = pd.Series(idx)
    else:
        uniq = groups.unique()
        n = len(uniq) // 2 if n_students is None else n_students
        assert 0 < n <= len(uniq), f"n_students {n} not in 1..{len(uniq)}"
        pick = rng.choice(uniq, n, replace=False)
        rows = groups.groupby(groups).indices
        idx = np.concatenate([rows[s] for s in pick])
        g = groups.iloc[idx].reset_index(drop=True)
    Xb, yb = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=b)
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_STATE)
    rankings, n_entered, train_rows = {m: [] for m in rankers}, [], []
    for f, (tr, _) in enumerate(outer.split(Xb, yb, g)):
        train_rows.append(int(len(tr)))
        for m, fn in rankers.items():
            ranking, info = fn(Xb.iloc[tr], yb.iloc[tr], g.iloc[tr], inner)
            rankings[m].append(ranking)
            if m == "LASSO":
                n_entered.append([info["n_entered"], info["need"]])
    return rankings, n_entered, train_rows


def summarise(path, analysis_path, p, feats, n_boot=N_BOOT):
    recs = sorted(read_jsonl(path), key=lambda r: r["b"])
    analysis = json.load(open(analysis_path))
    assert [r["b"] for r in recs] == list(range(n_boot)), "replicates incomplete or duplicated"
    shas = {r["sha"] for r in recs}
    assert shas == {analysis["fold_sha"]} and analysis["n_folds"] == 50, \
        f"bootstrap {shas} vs CV analysis {analysis['fold_sha']} (n_folds {analysis['n_folds']})"
    assert analysis["provenance"]["data_sha256"] == data_manifest(), "data differ from the CV analysis"
    curve = analysis["stability_curve_nogueira"]
    ks_all = range(1, max(p // 2, max(K_GRID)) + 1)
    stab = {m: {k: np.array([nogueira_stability([r[:k] for r in rec["rankings"][m]], feats) for rec in recs])
                for k in ks_all} for m in METHODS}
    q = lambda a, lo, hi: (float(np.percentile(a, lo)), float(np.percentile(a, hi)))

    def describe(est, draws, n_pairs=1):
        lo, hi = q(draws, 2.5, 97.5)
        mu, se = float(draws.mean()), float(draws.std(ddof=1))
        ci = (est - 1.96 * se, est + 1.96 * se)
        out = {"estimate_cv": float(est), "boot_mean": mu, "bias": mu - float(est), "se": se,
               "ci95": list(ci), "percentile_replicates": [lo, hi]}
        if n_pairs > 1:
            z = norm.ppf(1 - 0.05 / n_pairs / 2)
            bonf = (est - z * se, est + z * se)
            excl = lambda c: c[0] > 0 or c[1] < 0
            out["ci_bonferroni"] = list(bonf)
            out["robust"] = bool(excl(ci))
            out["robust_bonferroni"] = bool(excl(bonf))
        return out

    n_ent = [n for r in recs for n in r["lasso_n_entered"]]
    out = {"n_boot": len(recs), "scheme": "half-sampling without replacement (n/2 students)",
           "lasso_folds_short": int(sum(e < need for e, need in n_ent)),
           "boot_sha": shas.pop(), "provenance": provenance(with_data=True), "summaries": {}, "per_k": {}}
    for name, ks in k_sets(p).items():
        avg = {m: np.mean([stab[m][k] for k in ks], axis=0) for m in METHODS}
        est = {m: float(np.mean([curve[m][str(k)] for k in ks])) for m in METHODS}
        out["summaries"][name] = {
            "role": "confirmatory" if name == "grid" else "sensitivity", "k": ks,
            "stability": {m: describe(est[m], avg[m]) for m in METHODS},
            "diff": {f"{a}_minus_{b}": describe(est[a] - est[b], avg[a] - avg[b], n_pairs=10)
                     for a, b in combinations(METHODS, 2)}}
    for k in ks_all:
        out["per_k"][str(k)] = {"stability": {m: describe(curve[m][str(k)], stab[m][k]) for m in METHODS}}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["oulad", "xapi"], required=True)
    ap.add_argument("--cutoff", type=int, choices=[30, 60, 90])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--tma-lag", type=int, default=0, help="sensitivity: as in temporal_design.py")
    ap.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "temporal_design"))
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args()
    require_data()

    if a.dataset == "oulad":
        name = f"oulad_day{a.cutoff}" + (f"_tmalag{a.tma_lag}" if a.tma_lag else "")
        X, y, meta, _ = load_oulad_at(a.cutoff, tma_lag=a.tma_lag)
        dev, _ = split_dev_holdout(meta)
        X = X.drop(columns=[c for c in X.columns if X.loc[dev, c].nunique() <= 1])
        X, y, g = X[dev].reset_index(drop=True), y[dev].reset_index(drop=True), \
            meta.loc[dev, "id_student"].reset_index(drop=True)
        fix_discrete(X)
    else:
        name = "xapi_endofterm"
        X, y, _ = prepare_xapi(DATA_DIR)
        g = None
        fix_discrete(X)

    path = os.path.join(a.output_dir, f"{name}_bootstrap.jsonl")
    apath = os.path.join(a.output_dir, f"{name}_analysis.json")
    if not a.summarise_only:
        assert not provenance()["git_dirty"], "uncommitted changes in scripts/: commit before a run"
        # Fail fast, not after 200 replicates: the CV analysis must exist and match this code.
        assert os.path.exists(apath), f"run temporal_design.py first: {apath} missing"
        check_same_commit([json.load(open(apath))["fold_sha"]], f"{name} CV analysis")
        _lk = lock(path)
        recs = read_jsonl(path, repair=True)
        check_same_commit([r["sha"] for r in recs], f"{name} bootstrap")
        done = {r["b"] for r in recs}
        sha = provenance()["code_sha"]
        t0 = time.time()
        for b in range(a.n_boot):
            if b in done:
                continue
            t = time.time()
            rankings, n_entered, _ = replicate(X, y, g, b)
            rec = {"b": b, "rankings": rankings, "lasso_n_entered": n_entered, "sha": sha}
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{name}] boot {b} {time.time()-t:.0f}s (total {time.time()-t0:.0f}s)", flush=True)

    out = summarise(path, apath, X.shape[1], sorted(X.columns), a.n_boot)
    with open(os.path.join(a.output_dir, f"{name}_bootstrap_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{name}] bootstrap done", flush=True)
