"""
Re-run one stored cross-validation fold and compare it with the stored record.

The stored outputs record `code_sha`, the tree hash of the scripts that wrote
them. This check shows that the scripts in this repository reproduce those
outputs: it rebuilds the data exactly as temporal_design.py does, re-fits
the first fold (repeat 0, fold 0) and compares every method's ranking and
every evaluator's ROC-AUC with the stored fold record.

    python3 reproduce_fold.py --dataset oulad --cutoff 30   # ~2 min
    python3 reproduce_fold.py --dataset xapi                # ~1 min
"""
import argparse
import json
import os

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import temporal_design as T
from paths import DATA_DIR, RESULTS_DIR, require_data


def stored_fold(name, fold=0):
    with open(os.path.join(RESULTS_DIR, "temporal_design", f"{name}_fold_checkpoints.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            if rec["fold"] == fold:
                return rec
    raise SystemExit(f"fold {fold} not found for {name}")


def aucs(rec):
    """Flatten every ROC-AUC in a fold record to {path: value}."""
    out = {f"all/{ev}": v["roc_auc"] for ev, v in rec["baseline_all"].items()}
    for m, r in rec["methods"].items():
        for k, per_ev in r["per_k"].items():
            for ev, v in per_ev.items():
                if isinstance(v, dict) and "roc_auc" in v:
                    out[f"{m}/k{k}/{ev}"] = v["roc_auc"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["oulad", "xapi"], required=True)
    ap.add_argument("--cutoff", type=int, choices=T.CUTOFFS)
    a = ap.parse_args()
    require_data()

    if a.dataset == "oulad":
        name = f"oulad_day{a.cutoff}"
        X, y, meta, _ = T.load_oulad_at(a.cutoff)
        dev, _ = T.split_dev_holdout(meta)
        X = X.drop(columns=[c for c in X.columns if X.loc[dev, c].nunique() <= 1])
        X, y, g = X[dev].reset_index(drop=True), y[dev].reset_index(drop=True), \
            meta.loc[dev, "id_student"].reset_index(drop=True)
        outer = StratifiedGroupKFold(T.N_SPLITS, shuffle=True, random_state=0)
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=T.RANDOM_STATE)
    else:
        name = "xapi_endofterm"
        X, y, _ = T.prepare_xapi(DATA_DIR)
        g = None
        outer = StratifiedKFold(T.N_SPLITS, shuffle=True, random_state=0)
        inner = StratifiedKFold(3, shuffle=True, random_state=T.RANDOM_STATE)
    T.fix_discrete(X)

    tr, te = next(outer.split(X, y, g))
    rec = T.evaluate(X.iloc[tr], y.iloc[tr], None if g is None else g.iloc[tr], X.iloc[te], y.iloc[te], inner)
    ref = stored_fold(name)

    bad = [m for m in T.METHODS if rec["methods"][m]["ranking"] != ref["methods"][m]["ranking"]]
    new, old = aucs(rec), aucs(ref)
    worst = max(abs(new[k] - old[k]) for k in old)
    print(f"[{name}] rankings identical: {len(T.METHODS) - len(bad)}/{len(T.METHODS)}"
          + (f" (differ: {bad})" if bad else ""))
    print(f"[{name}] ROC-AUC values compared: {len(old)}; largest absolute difference: {worst:.2e}")
    raise SystemExit(1 if bad or worst > 1e-9 else 0)
