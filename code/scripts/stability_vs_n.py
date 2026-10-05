"""
Selection stability against sample size, CV against disjoint sets (OULAD).

The cross-presentation analysis (cross_cohort.py) ranks features on disjoint
sets of 1,650 students; the CV analysis (temporal_design.py) ranks them on
overlapping training folds of about 13,500 registrations. Agreement depends on
n and on overlap, so the two cannot be compared directly. This script measures
both designs on the SAME population (pooled development students, as in the CV
analysis) at a chosen n:

  - CV arm: draw n_students development students without replacement (all
    registrations of a drawn student come along), run one 5-fold
    StratifiedGroupKFold and rank on each training fold (stability_bootstrap.
    replicate). Training folds overlap by 75% of their students.
  - Disjoint arm: draw two disjoint sets of m students each, with m the CV
    training-fold size round(4n/5), capped at half the development students,
    and rank each set on all its rows. Agreement is Nogueira with M = 2.

So n_students = 2,062 gives m = 1,650, the cross_cohort set size, and
n_students = all development students gives the CV arm at full n and the
largest disjoint halves. Within an arm, the difference from the other arm is
the effect of overlap at matched n and population; the difference from the
cross_cohort `within_<pres>` values is then population (one presentation,
single-presentation students), not n.

Rankers: the five METHODS plus GAIN, a one-shot ranking by the gain importance
of one default XGBoost fit (the importance RFE uses at its first step). GAIN
against RFE separates the recursion from the importance measure.

Seeds: both arms draw students with seeds derived from (RANDOM_STATE, n, b,
arm), so draws are independent across n and from the half-sample bootstrap.

Reporting is descriptive: per method and arm, the mean, SD, min and max over
replicates of the grid-mean (and k <= p/3-mean) Nogueira stability, the per-k
mean, and the method order. Replicates overlap in students, so the SD is
resampling spread, not a standard error. Nogueira with fixed k equals the mean
of its pairwise values, so M = 5 and M = 2 values are on the same scale.

Integrity as in temporal_design.py: clean tree, one code version across
replicates, the same data across replicates, a per-file lock, complete
replicates.
"""
import os
import json
import time
import argparse

import numpy as np

from paths import RESULTS_DIR, require_data
from temporal_design import (RANKERS, METHODS, K_GRID, RANDOM_STATE, load_oulad_at, split_dev_holdout,
                             nogueira_stability, provenance, fix_discrete, read_jsonl, lock, check_same_commit,
                             data_manifest, _xgb, _tie_rng, _order)
from stability_bootstrap import replicate

N_REP = 30
ALL_METHODS = METHODS + ["GAIN"]


def rank_gain(X, y, g, cv):
    """One-shot ranking by the gain importance of one default XGBoost fit (as RFE's first step)."""
    imp = _xgb().fit(X, y).feature_importances_
    return _order(imp, list(X.columns), _tie_rng(X)), {}


RANKERS_N = {**{m: RANKERS[m] for m in METHODS}, "GAIN": rank_gain}


def seed(n, b, arm):
    return int(np.random.SeedSequence([RANDOM_STATE, n, b, arm]).generate_state(1)[0])


def draw_disjoint(groups, m, rng):
    """Row indices of two disjoint sets of m students each."""
    ids = rng.choice(groups.unique(), 2 * m, replace=False)
    rows = groups.groupby(groups).indices
    return [np.sort(np.concatenate([rows[s] for s in half])) for half in (ids[:m], ids[m:])]


def disjoint_arm(X, y, groups, m, rng):
    sets = draw_disjoint(groups, m, rng)
    out = {meth: [fn(X.iloc[ix], y.iloc[ix], None, None) for ix in sets] for meth, fn in RANKERS_N.items()}
    rankings = {meth: [o[0] for o in res] for meth, res in out.items()}
    lasso = [[o[1]["n_entered"], o[1]["need"]] for o in out["LASSO"]]
    return rankings, [int(len(ix)) for ix in sets], lasso


def summarise(recs, feats, p):
    ks_sets = {"grid": K_GRID, "k_le_p3": list(range(1, p // 3 + 1))}
    ks_all = sorted(set(K_GRID) | set(ks_sets["k_le_p3"]))
    d = lambda a: {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                   "min": float(a.min()), "max": float(a.max())}
    out = {}
    for arm in ("cv", "disjoint"):
        if recs[0].get(arm) is None:
            continue
        stab = {m: {k: np.array([nogueira_stability([r[:k] for r in rec[arm]["rankings"][m]], feats)
                                 for rec in recs]) for k in ks_all} for m in ALL_METHODS}
        res = {"per_k_mean": {str(k): {m: float(stab[m][k].mean()) for m in ALL_METHODS} for k in ks_all},
               "summaries": {}}
        for name, ks in ks_sets.items():
            avg = {m: np.mean([stab[m][k] for k in ks], axis=0) for m in ALL_METHODS}
            res["summaries"][name] = {"k": ks, "stability": {m: d(avg[m]) for m in ALL_METHODS},
                                      "order": sorted(ALL_METHODS, key=lambda m: -avg[m].mean())}
        out[arm] = res
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, choices=[30, 60, 90], required=True)
    ap.add_argument("--n-students", type=int, required=True, help="0 = all development students")
    ap.add_argument("--n-rep", type=int, default=N_REP)
    ap.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "temporal_design"))
    a = ap.parse_args()
    require_data()
    assert not provenance()["git_dirty"], "uncommitted changes in scripts/: commit before a run"
    os.makedirs(a.output_dir, exist_ok=True)

    X, y, meta, used = load_oulad_at(a.cutoff)
    assert used["max_vle_date"] <= a.cutoff and used["max_assess_date"] <= a.cutoff, used
    dev, _ = split_dev_holdout(meta)
    X = X.drop(columns=[c for c in X.columns if X.loc[dev, c].nunique() <= 1])  # as temporal_design
    X, y, g = X[dev].reset_index(drop=True), y[dev].reset_index(drop=True), \
        meta.loc[dev, "id_student"].reset_index(drop=True)
    fix_discrete(X)
    n_dev = int(g.nunique())
    n = n_dev if a.n_students == 0 else a.n_students
    assert 0 < n <= n_dev, f"--n-students {n} not in 1..{n_dev}"
    m = min(round(4 * n / 5), n_dev // 2)
    name = f"oulad_day{a.cutoff}_stability_n{'all' if a.n_students == 0 else n}"
    data_sha = data_manifest()

    path = os.path.join(a.output_dir, f"{name}.jsonl")
    _lk = lock(path)
    done = read_jsonl(path, repair=True)
    sha = provenance()["code_sha"]
    check_same_commit([r["sha"] for r in done] + [sha], name)
    assert all(r["data_sha256"] == data_sha for r in done), "stored replicates used other data"
    have = {r["b"] for r in done}
    for b in range(a.n_rep):
        if b in have:
            continue
        t0 = time.time()
        cv_rank, n_entered, cv_rows = replicate(X, y, g, b, n_students=n, seed=seed(n, b, 0), rankers=RANKERS_N)
        dj_rank, dj_rows, dj_lasso = disjoint_arm(X, y, g, m, np.random.default_rng(seed(n, b, 1)))
        rec = {"b": b, "sha": sha, "data_sha256": data_sha,
               "cv": {"rankings": cv_rank, "train_rows": cv_rows, "lasso_n_entered": n_entered},
               "disjoint": {"rankings": dj_rank, "rows": dj_rows, "lasso_n_entered": dj_lasso}}
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{name}] rep {b} {time.time() - t0:.0f}s", flush=True)

    recs = sorted(read_jsonl(path), key=lambda r: r["b"])
    assert [r["b"] for r in recs] == list(range(a.n_rep)), "replicates incomplete or duplicated"
    check_same_commit([r["sha"] for r in recs], name)
    short = lambda arm: int(sum(e < need for r in recs for e, need in r[arm]["lasso_n_entered"]))
    rows = lambda key, arm: [x for r in recs for x in r[arm][key]]
    res = {"n_rep": a.n_rep, "n_students": n, "n_dev_students": n_dev, "disjoint_set_students": m,
           "cv_train_rows": {"min": min(rows("train_rows", "cv")), "max": max(rows("train_rows", "cv"))},
           "disjoint_rows": {"min": min(rows("rows", "disjoint")), "max": max(rows("rows", "disjoint"))},
           "p": X.shape[1], "methods": ALL_METHODS,
           "lasso_sets_short": {"cv": short("cv"), "disjoint": short("disjoint")},
           "note": "descriptive: replicates overlap in students; SD is resampling spread, not an SE",
           "provenance": provenance(with_data=True), **summarise(recs, sorted(X.columns), X.shape[1])}
    with open(os.path.join(a.output_dir, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{name}] done", flush=True)
