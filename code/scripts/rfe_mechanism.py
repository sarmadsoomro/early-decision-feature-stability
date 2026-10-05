"""
Why are RFE's selections less stable, and does it matter? (read-only)

Reads the stored CV fold checkpoints of temporal_design.py (rankings and
per-fold test AUCs); fits no model. Per OULAD cutoff, for each method and each
k in K_GRID:

1. Substitution. For every ordered pair of the 50 CV folds with top-k sets A
   and B, a feature in A \ B is "swapped out"; the features in B \ A replaced
   it. Its substitution score is max |rho| with the features of B \ A, where
   rho is the Spearman correlation on all development rows (Pearson as a
   sensitivity). The reference is rank-matched: the features ranked k+1..2k in
   A's fold that are not in B, scored the same way against B \ A. Swapped
   features scoring well above the reference mean the disagreements are mostly
   exchanges of strongly correlated features. Reported: mean swapped and
   reference scores, mean swaps per pair, and the share of swapped features
   with score >= 0.7 ("strongly correlated", not "duplicate"). Dummies of one
   categorical are mutually exclusive, so they correlate negatively; |rho|
   counts that as association.
2. AUC gap. Per fold, the test-fold ROC-AUC of RFE's top-k minus that of each
   other method, for each evaluator (XGBoost, random forest, logistic
   regression; all scored on the same test fold). Reported: mean, min and max
   over the 50 folds. This is a level gap between methods, not the accuracy
   effect of instability; the corrected tests are temporal_design.py's
   `pairwise_roc_auc`.

Fold pairs share training rows, so nothing here is a test. Output:
`oulad_dayN_rfe_mechanism.json`.
"""
import os
import json
import argparse
from itertools import combinations

import numpy as np

from paths import RESULTS_DIR, require_data
from temporal_design import (METHODS, K_GRID, EVALUATORS, load_oulad_at, split_dev_holdout, read_jsonl,
                             provenance, data_manifest)

HIGH_R = 0.7


def substitution(rankings, corr, k):
    """Swapped-out features of each ordered fold pair, scored against their replacements."""
    swapped, ref, n_swaps = [], [], []
    for ia, ra in enumerate(rankings):
        for ib, rb in enumerate(rankings):
            if ia == ib:
                continue
            A, B = set(ra[:k]), set(rb[:k])
            repl = list(B - A)
            n_swaps.append(len(repl))
            if not repl:
                continue
            score = lambda f: float(corr.loc[f, repl].max())
            swapped += [score(f) for f in A - B]
            ref += [score(f) for f in ra[k:2 * k] if f not in B]
    sw, rf = np.array(swapped), np.array(ref)
    nan = float("nan")
    return {"swapped_mean_maxabsr": float(sw.mean()) if len(sw) else nan,
            "reference_mean_maxabsr": float(rf.mean()) if len(rf) else nan,
            "swaps_per_pair": float(np.mean(n_swaps)),
            "share_swapped_ge_0.7": float((sw >= HIGH_R).mean()) if len(sw) else nan}


def auc_gap(folds, k):
    out = {}
    for ev in EVALUATORS:
        auc = lambda f, m: f["methods"][m]["per_k"][str(k)][ev]["roc_auc"]
        for m in METHODS:
            if m != "RFE":
                d = np.array([auc(f, "RFE") - auc(f, m) for f in folds])
                out[f"{ev}_RFE_minus_{m}"] = {"mean": float(d.mean()), "min": float(d.min()), "max": float(d.max())}
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
    analysis = json.load(open(os.path.join(a.output_dir, f"{name}_analysis.json")))
    assert analysis["provenance"]["data_sha256"] == data_manifest(), "data differ from the stored run"

    X, y, meta, _ = load_oulad_at(a.cutoff)
    dev, _ = split_dev_holdout(meta)
    X = X.drop(columns=[c for c in X.columns if X.loc[dev, c].nunique() <= 1])  # as temporal_design
    feats = set(folds[0]["methods"]["MI"]["ranking"])
    assert feats == set(X.columns), "features differ from the stored run"
    corrs = {how: X[dev].astype(float).corr(method=how).abs() for how in ("spearman", "pearson")}
    for c in corrs.values():
        np.fill_diagonal(c.values, np.nan)  # a feature is not its own substitute

    res = {"cutoff": a.cutoff, "fold_sha": folds[0]["sha"], "high_r": HIGH_R,
           "note": "descriptive: fold pairs share training rows",
           "provenance": provenance(with_data=True), "per_k": {}}
    for k in K_GRID:
        res["per_k"][str(k)] = {
            **{f"substitution_{how}": {m: substitution([f["methods"][m]["ranking"] for f in folds], c, k)
                                       for m in METHODS} for how, c in corrs.items()},
            "auc_gap": auc_gap(folds, k)}
    with open(os.path.join(a.output_dir, f"{name}_rfe_mechanism.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{name}] rfe mechanism done", flush=True)
