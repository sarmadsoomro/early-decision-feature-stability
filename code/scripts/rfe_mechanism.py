"""
Why are RFE's selections less stable, and does it matter? (read-only)

Reads the stored CV fold checkpoints of temporal_design.py (rankings and
per-fold test AUCs); fits no model. Per OULAD cutoff, for each method and each
k in K_GRID:

1. Substitution. For every pair of the 50 CV folds, a feature that is in one
   fold's top-k set A but not in the other's set B is a "swapped" feature. Its
   substitution score is max |r| with the features of B, where r is the
   Pearson correlation on all development rows. The reference is the same
   score for every feature outside B (swapped or not). If a method's swapped
   features have much higher scores than the reference, its disagreements are
   mostly exchanges between near-duplicates. Reported: mean swapped score,
   mean reference score, mean number of swapped features per pair, and the
   share of swapped features with score >= 0.7.
2. Cost. Per fold, the XGBoost test-fold ROC-AUC of RFE's top-k minus that of
   each other method (all methods are scored on the same test fold). Reported:
   mean, min and max over the 50 folds. This is descriptive; the corrected
   tests are in temporal_design.py's `pairwise_roc_auc`.

Fold pairs share training rows, so nothing here is a test. Output:
`oulad_dayN_rfe_mechanism.json`.
"""
import os
import json
import argparse
from itertools import combinations

import numpy as np

from paths import RESULTS_DIR, require_data
from temporal_design import METHODS, K_GRID, load_oulad_at, split_dev_holdout, read_jsonl, provenance

HIGH_R = 0.7


def substitution(rankings, corr, k):
    """Mean swapped-feature score, mean reference score, mean swaps per pair, share >= HIGH_R."""
    swapped, ref, n_swaps = [], [], []
    for ra, rb in combinations(rankings, 2):
        for A, B in ((set(ra[:k]), set(rb[:k])), (set(rb[:k]), set(ra[:k]))):
            Bl = list(B)
            score = lambda f: float(corr.loc[f, Bl].max())
            sw = [score(f) for f in A - B]
            swapped += sw
            ref += [score(f) for f in corr.index if f not in B]
            n_swaps.append(len(sw))
    sw = np.array(swapped)
    return {"swapped_mean_maxabsr": float(sw.mean()) if len(sw) else float("nan"),
            "reference_mean_maxabsr": float(np.mean(ref)),
            "swaps_per_pair": float(np.mean(n_swaps)),
            "share_swapped_ge_0.7": float((sw >= HIGH_R).mean()) if len(sw) else float("nan")}


def cost(folds, k):
    auc = lambda f, m: f["methods"][m]["per_k"][str(k)]["xgb"]["roc_auc"]
    out = {}
    for m in METHODS:
        if m == "RFE":
            continue
        d = np.array([auc(f, "RFE") - auc(f, m) for f in folds])
        out[f"RFE_minus_{m}"] = {"mean": float(d.mean()), "min": float(d.min()), "max": float(d.max())}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, choices=[30, 60, 90], required=True)
    ap.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "temporal_design"))
    a = ap.parse_args()
    require_data()
    name = f"oulad_day{a.cutoff}"
    folds = read_jsonl(os.path.join(a.output_dir, f"{name}_fold_checkpoints.jsonl"))
    assert len(folds) == 50 and len({f["sha"] for f in folds}) == 1, "need 50 folds from one code version"

    X, y, meta, _ = load_oulad_at(a.cutoff)
    dev, _ = split_dev_holdout(meta)
    X = X.drop(columns=[c for c in X.columns if X.loc[dev, c].nunique() <= 1])  # as temporal_design
    feats = set(folds[0]["methods"]["MI"]["ranking"])
    assert feats == set(X.columns), "features differ from the stored run"
    corr = X[dev].astype(float).corr().abs()
    np.fill_diagonal(corr.values, np.nan)  # a feature is not its own substitute

    res = {"cutoff": a.cutoff, "fold_sha": folds[0]["sha"], "high_r": HIGH_R,
           "note": "descriptive: fold pairs share training rows",
           "provenance": provenance(with_data=True), "per_k": {}}
    for k in K_GRID:
        res["per_k"][str(k)] = {
            "substitution": {m: substitution([f["methods"][m]["ranking"] for f in folds], corr, k) for m in METHODS},
            "auc_cost_xgb": cost(folds, k)}
    with open(os.path.join(a.output_dir, f"{name}_rfe_mechanism.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{name}] rfe mechanism done", flush=True)
