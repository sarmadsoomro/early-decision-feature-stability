"""
Selection stability against sample size (OULAD).

The cross-presentation analysis (cross_cohort.py) ranks features on sets of
1,650 students; the CV analysis (temporal_design.py) ranks them on training
folds of about 13,500 registrations. Agreement depends on n, so a method's
stability in the two designs cannot be compared directly. This script
measures CV stability at a chosen number of students, so the CV design can be
read at the disjoint-set size (n_students = 2,062 gives training folds of
about 1,650 students) and at points between.

Per replicate b (R = N_REP): draw n_students development students without
replacement (all registrations of a drawn student come along), run one 5-fold
StratifiedGroupKFold, rank features with all five methods on each training
fold and store the rankings. This is stability_bootstrap.replicate with
n_students in place of n/2, so the n/2 bootstrap is the same design at
n_students = n/2.

Reporting is descriptive: per method, the mean, SD, min and max over
replicates of the grid-mean (and k <= p/3-mean) Nogueira stability, the per-k
mean, and the method order at this n. Replicates overlap in students, so the
SD is resampling spread, not a standard error.

Integrity as in temporal_design.py: clean tree, one code version across
replicates, a per-file lock, complete replicates.
"""
import os
import json
import time
import argparse

import numpy as np

from paths import RESULTS_DIR, require_data
from temporal_design import (METHODS, K_GRID, load_oulad_at, split_dev_holdout, nogueira_stability,
                             provenance, fix_discrete, read_jsonl, lock, check_same_commit)
from stability_bootstrap import replicate

N_REP = 50


def summarise(recs, feats, p):
    ks_sets = {"grid": K_GRID, "k_le_p3": list(range(1, p // 3 + 1))}
    ks_all = sorted(set(K_GRID) | set(ks_sets["k_le_p3"]))
    stab = {m: {k: np.array([nogueira_stability([r[:k] for r in rec["rankings"][m]], feats) for rec in recs])
                for k in ks_all} for m in METHODS}
    d = lambda a: {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                   "min": float(a.min()), "max": float(a.max())}
    out = {"summaries": {}, "per_k_mean": {str(k): {m: float(stab[m][k].mean()) for m in METHODS} for k in ks_all}}
    for name, ks in ks_sets.items():
        avg = {m: np.mean([stab[m][k] for k in ks], axis=0) for m in METHODS}
        out["summaries"][name] = {"k": ks, "stability": {m: d(avg[m]) for m in METHODS},
                                  "order": sorted(METHODS, key=lambda m: -avg[m].mean())}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, choices=[30, 60, 90], required=True)
    ap.add_argument("--n-students", type=int, required=True)
    ap.add_argument("--n-rep", type=int, default=N_REP)
    ap.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "temporal_design"))
    a = ap.parse_args()
    require_data()
    assert not provenance()["git_dirty"], "uncommitted changes in scripts/: commit before a run"
    os.makedirs(a.output_dir, exist_ok=True)
    name = f"oulad_day{a.cutoff}_stability_n{a.n_students}"

    X, y, meta, used = load_oulad_at(a.cutoff)
    assert used["max_vle_date"] <= a.cutoff and used["max_assess_date"] <= a.cutoff, used
    dev, _ = split_dev_holdout(meta)
    X = X.drop(columns=[c for c in X.columns if X.loc[dev, c].nunique() <= 1])  # as temporal_design
    X, y, g = X[dev].reset_index(drop=True), y[dev].reset_index(drop=True), \
        meta.loc[dev, "id_student"].reset_index(drop=True)
    fix_discrete(X)
    n_dev = int(g.nunique())
    assert a.n_students <= n_dev, f"--n-students {a.n_students} > {n_dev} development students"

    path = os.path.join(a.output_dir, f"{name}.jsonl")
    _lk = lock(path)
    done = read_jsonl(path, repair=True)
    sha = provenance()["code_sha"]
    check_same_commit([r["sha"] for r in done] + [sha], name)
    have = {r["b"] for r in done}
    for b in range(a.n_rep):
        if b in have:
            continue
        t0 = time.time()
        rankings, n_entered = replicate(X, y, g, b, n_students=a.n_students)
        with open(path, "a") as f:
            f.write(json.dumps({"b": b, "sha": sha, "rankings": rankings, "lasso_n_entered": n_entered}) + "\n")
        print(f"[{name}] rep {b} {time.time() - t0:.0f}s", flush=True)

    recs = sorted(read_jsonl(path), key=lambda r: r["b"])
    assert [r["b"] for r in recs] == list(range(a.n_rep)), "replicates incomplete or duplicated"
    check_same_commit([r["sha"] for r in recs], name)
    n_ent = [n for r in recs for n in r["lasso_n_entered"]]
    res = {"n_rep": a.n_rep, "n_students": a.n_students, "n_dev_students": n_dev,
           "train_students_per_fold": round(a.n_students * 4 / 5), "p": X.shape[1],
           "lasso_folds_short": int(sum(e < need for e, need in n_ent)),
           "note": "descriptive: replicates overlap in students; SD is resampling spread, not an SE",
           "provenance": provenance(with_data=True), **summarise(recs, sorted(X.columns), X.shape[1])}
    with open(os.path.join(a.output_dir, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{name}] done", flush=True)
