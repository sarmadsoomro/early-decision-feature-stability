"""
Cross-presentation ranking stability.

The CV and half-sampling analyses measure subsampling stability within the
pooled development presentations. This script asks whether a method's top-k
set agrees less between two presentations than within one.

Design, per repeat r (R = 20) and per cutoff:
  - Four presentations: 2013B, 2013J, 2014B (dev) and 2014J (holdout).
    Only students with exactly one presentation in the raw studentInfo.csv
    are used, in every presentation alike (single_presentation_pool). The
    rule is applied before the cutoff filter, so a student who withdrew early
    and re-took the module later is excluded. The eligibility rule is the
    same at every cutoff; the students available still shrink with the
    cutoff's registration filters.
  - Two disjoint sets of students per presentation, S1_<pres> and S2_<pres>.
    `--modules all` (primary): N_STUDENTS random students per set.
    `--modules common` (sensitivity): fixed per-module quotas from BBB, DDD
    and FFF, the modules run in all four presentations, so every set has the
    same module mix (COMMON_QUOTA).
    `--n-students` runs the all-modules design at another n (e.g. the
    common-mode n), because agreement depends on n: compare magnitudes across
    modes only at equal n.
    Quotas and N_STUDENTS fit the smallest pool at every cutoff (day 90).
  - Every method ranks every set on all of its rows (no CV). Sets are equal
    in students; a student registered on two modules brings both rows
    (rows are stored per set).
  - within_<pres>   = Nogueira(S1, S2) of that presentation.
  - between_<a|b>   = mean Nogueira over the 4 cross combinations of a and b.
  - loss_<a|b>      = mean(within_a, within_b) - between_<a|b>: how much less
    two presentations agree than each agrees with itself, at equal n.
Nogueira with M = 2 equals Kuncheva's index: chance-corrected (0 = random).

Pairs differ in time gap, season (B = February start, J = October) and,
under `--modules all`, module mix. Groups are reported per pair and pooled
by type; gap groups are matched on whether 2014J is involved. Four
presentations give six dependent pairs, so no trend is tested. Reporting is
descriptive: mean, SD, min and max over repeats, which are resampling noise
within one sample of each presentation, not intervals; plus, per pair, the
share of repeats with loss > 0 and the number of pairs with mean loss > 0.
Some VLE activity types do not exist in some presentations (constant columns,
stored per set), so "presentation" includes course design, not only cohort.

Rows keep their original index, so the per-set tie-breaking seed (_tie_rng)
differs between sets. Integrity as in temporal_design.py: clean tree, one code
version across repeats, a per-file lock, complete repeats.
"""
import os
import json
import time
import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from paths import DATA_DIR, RESULTS_DIR, require_data
from temporal_design import (RANKERS, METHODS, K_GRID, RANDOM_STATE, CUTOFFS, load_oulad_at,
                             split_dev_holdout, nogueira_stability, provenance, fix_discrete,
                             read_jsonl, lock, check_same_commit)

N_REP = 20
N_STUDENTS = 1650                                      # 2 x 1650 <= 3,396 (2013B, day 90)
COMMON_QUOTA = {"BBB": 560, "DDD": 330, "FFF": 480}    # 2 x quota < 1128/673/985 (2014B, day 90)
ORDER = ["2013B", "2013J", "2014B", "2014J"]
PAIRS = [f"{a}|{b}" for a, b in combinations(ORDER, 2)]
GROUPS = {
    "dev_dev": ["2013B|2013J", "2013B|2014B", "2013J|2014B"],
    "dev_2014J": ["2013B|2014J", "2013J|2014J", "2014B|2014J"],
    "same_season": ["2013B|2014B", "2013J|2014J"],
    "cross_season": ["2013B|2013J", "2013J|2014B", "2014B|2014J", "2013B|2014J"],
    "gap1_dev": ["2013B|2013J", "2013J|2014B"], "gap1_2014J": ["2014B|2014J"],
    "gap2_dev": ["2013B|2014B"], "gap2_2014J": ["2013J|2014J"],
    "gap3": ["2013B|2014J"],
}


def single_presentation_pool(meta, raw_info):
    """Rows of students with exactly one presentation in the raw registrations
    (studentInfo.csv), so cutoff-dropped registrations still count."""
    n = raw_info.groupby("id_student")["code_presentation"].nunique()
    return (meta["id_student"].isin(n.index[n == 1]) & meta["code_presentation"].isin(ORDER)).to_numpy()


def draw_sets(meta, pool, modules, rng, n_students=None):
    """Row masks for S1_<pres> and S2_<pres>: two disjoint student sets per presentation."""
    masks = {}
    for pres in ORDER:
        in_pres = pool & (meta["code_presentation"] == pres).to_numpy()
        if modules == "all":
            n = n_students or N_STUDENTS
            ids = rng.choice(meta.loc[in_pres, "id_student"].unique(), 2 * n, replace=False)
            halves = [ids[:n], ids[n:]]
        else:
            halves, taken = [[], []], set()
            for mod, q in COMMON_QUOTA.items():  # a student on two common modules is drawn once
                cand = meta.loc[in_pres & (meta["code_module"] == mod).to_numpy(), "id_student"].unique()
                ids = rng.choice(np.setdiff1d(cand, list(taken)), 2 * q, replace=False)
                taken.update(ids)
                halves[0] += list(ids[:q])
                halves[1] += list(ids[q:])
        for i, h in enumerate(halves, 1):
            m = in_pres & meta["id_student"].isin(h).to_numpy()
            if modules != "all":
                m &= meta["code_module"].isin(list(COMMON_QUOTA)).to_numpy()
            masks[f"S{i}_{pres}"] = m
    return masks


def composition(X, y, meta, mask):
    return {"rows": int(mask.sum()), "students": int(meta.loc[mask, "id_student"].nunique()),
            "pass_rate": float(y[mask].mean()), "no_submission_share": float(X.loc[mask, "no_submission"].mean()),
            "module_share": meta.loc[mask, "code_module"].value_counts(normalize=True).round(4).to_dict(),
            "constant_cols": [c for c in X.columns if X.loc[mask, c].nunique() <= 1]}


def repeat(X, y, meta, pool, modules, r, n_students=None):
    masks = draw_sets(meta, pool, modules, np.random.default_rng(RANDOM_STATE + r), n_students)
    ids = {s: set(meta.loc[m, "id_student"]) for s, m in masks.items()}
    assert not any(ids[a] & ids[b] for a, b in combinations(masks, 2)), "sets share students"
    rankings, info = {}, {}
    for s, m in masks.items():
        out = {meth: RANKERS[meth](X[m], y[m], None, None) for meth in METHODS}
        rankings[s] = {meth: o[0] for meth, o in out.items()}
        info[s] = {meth: o[1] for meth, o in out.items()}
    return rankings, info, {s: composition(X, y, meta, m) for s, m in masks.items()}


def summarise(recs, feats, p):
    agree = lambda a, b, k: nogueira_stability([a[:k], b[:k]], feats)
    ks_all = range(1, max(p // 2, max(K_GRID)) + 1)

    def per_rep(m, k):
        """Per-repeat within (per presentation), between and loss (per pair) for method m at k."""
        out = {}
        for r in recs:
            R = r["rankings"]
            w = {pr: agree(R[f"S1_{pr}"][m], R[f"S2_{pr}"][m], k) for pr in ORDER}
            row = {f"within_{pr}": v for pr, v in w.items()}
            for pair in PAIRS:
                a, b = pair.split("|")
                btw = np.mean([agree(R[f"S{i}_{a}"][m], R[f"S{j}_{b}"][m], k) for i in (1, 2) for j in (1, 2)])
                row[f"between_{pair}"] = btw
                row[f"loss_{pair}"] = (w[a] + w[b]) / 2 - btw
            for g, prs in GROUPS.items():
                row[f"within_{g}"] = np.mean([(w[x.split("|")[0]] + w[x.split("|")[1]]) / 2 for x in prs])
                row[f"between_{g}"] = np.mean([row[f"between_{x}"] for x in prs])
                row[f"loss_{g}"] = np.mean([row[f"loss_{x}"] for x in prs])
            for key, v in row.items():
                out.setdefault(key, []).append(v)
        return {key: np.array(v) for key, v in out.items()}

    vals = {m: {k: per_rep(m, k) for k in ks_all} for m in METHODS}
    d = lambda a: {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                   "min": float(a.min()), "max": float(a.max())}
    keys = list(vals[METHODS[0]][1])
    pos = lambda a: float((a > 0).mean())
    out = {"groups": GROUPS,
           "loss_positive": {
               name: {m: {"share_of_repeats": {pr: pos(np.mean([vals[m][k][f"loss_{pr}"] for k in ks], axis=0))
                                               for pr in PAIRS},
                          "pairs_with_mean_loss_gt0": int(sum(
                              np.mean([vals[m][k][f"loss_{pr}"] for k in ks]) > 0 for pr in PAIRS))}
                      for m in METHODS}
               for name, ks in {"grid": K_GRID, "k_le_p3": list(range(1, p // 3 + 1))}.items()},
           "per_k": {str(k): {key: {m: d(vals[m][k][key]) for m in METHODS} for key in keys} for k in ks_all},
           "summaries": {}}
    for name, ks in {"grid": K_GRID, "k_le_p3": list(range(1, p // 3 + 1))}.items():
        out["summaries"][name] = {
            "role": "primary" if name == "grid" else "sensitivity", "k": ks,
            **{key: {m: d(np.mean([vals[m][k][key] for k in ks], axis=0)) for m in METHODS} for key in keys}}
    return out


def lasso_shortfall(recs):
    pairs = [(i["LASSO"]["n_entered"], i["LASSO"]["need"]) for r in recs for i in r["info"].values()]
    return {"sets_short": int(sum(e < n for e, n in pairs)), "min_n_entered": int(min(e for e, _ in pairs))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, choices=CUTOFFS, required=True)
    ap.add_argument("--modules", choices=["all", "common"], default="all")
    ap.add_argument("--n-rep", type=int, default=N_REP)
    ap.add_argument("--n-students", type=int, default=None, help="all-modules only; default N_STUDENTS")
    ap.add_argument("--tma-lag", type=int, default=0, help="sensitivity: as in temporal_design.py")
    ap.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "temporal_design"))
    a = ap.parse_args()
    require_data()
    assert not provenance()["git_dirty"], "uncommitted changes in scripts/: commit before a run"
    os.makedirs(a.output_dir, exist_ok=True)
    assert a.n_students is None or a.modules == "all", "--n-students applies to --modules all"
    name = f"oulad_day{a.cutoff}_cross_cohort" + ("_common" if a.modules == "common" else "") \
        + (f"_n{a.n_students}" if a.n_students not in (None, N_STUDENTS) else "") \
        + (f"_tmalag{a.tma_lag}" if a.tma_lag else "")
    X, y, meta, used = load_oulad_at(a.cutoff, tma_lag=a.tma_lag)
    assert used["max_vle_date"] <= a.cutoff and used["max_assess_date"] <= a.cutoff, used
    dev, te = split_dev_holdout(meta)
    X = X.drop(columns=[c for c in X.columns if X.loc[dev, c].nunique() <= 1])  # as temporal_design
    fix_discrete(X[dev])
    pool = single_presentation_pool(
        meta, pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"), usecols=["id_student", "code_presentation"]))
    feats = list(X.columns)
    design = {"modules": a.modules, "n_students": (a.n_students or N_STUDENTS) if a.modules == "all" else COMMON_QUOTA, "tma_lag": a.tma_lag}

    path = os.path.join(a.output_dir, f"{name}.jsonl")
    _lk = lock(path)
    done = read_jsonl(path, repair=True)
    sha = provenance()["code_sha"]
    if done:
        check_same_commit([r["sha"] for r in done] + [sha], name)
        assert all(r["design"] == design for r in done), "stored repeats used another design"
    assert len(done) <= a.n_rep, f"{len(done)} repeats stored, --n-rep {a.n_rep}: raise --n-rep"
    have = {r["r"] for r in done}
    for r in range(a.n_rep):
        if r in have:
            continue
        t0 = time.time()
        rankings, info, comp = repeat(X, y, meta, pool, a.modules, r, a.n_students)
        rec = {"r": r, "sha": sha, "design": design, "rankings": rankings, "info": info, "composition": comp}
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{name}] rep {r} {time.time() - t0:.0f}s", flush=True)

    recs = sorted(read_jsonl(path), key=lambda r: r["r"])
    assert [r["r"] for r in recs] == list(range(a.n_rep)), "repeats incomplete or duplicated"
    check_same_commit([r["sha"] for r in recs], name)
    res = {"n_rep": a.n_rep, "design": design, "p": len(feats), "presentations": ORDER,
           "note": "descriptive: one sample per presentation; spread over repeats is resampling noise. "
                   "Pairs differ in time, season and (modules=all) module mix; per-set composition in the jsonl.",
           "lasso_path": lasso_shortfall(recs),
           "provenance": provenance(with_data=True), **summarise(recs, feats, len(feats))}
    with open(os.path.join(a.output_dir, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{name}] done", flush=True)
