"""
Feature-selection stability at early decision points: data construction,
grouped cross-validation, the five rankers, evaluators and the later-presentation
holdout.

Design:
  - Prediction point. OULAD features are built from data available by day
    30, 60 or 90 of each presentation, never from full-course aggregates.
    Registrations withdrawn on or before the cutoff, or not yet registered
    by it, are dropped. Withdrawals with no recorded date are kept (label 0):
    dropping them would select on the outcome.
  - Unit of analysis. One row per registration (id_student, code_module,
    code_presentation); VLE and assessment data join on all three keys, so
    a student's other courses never leak into a registration.
  - Grouped splitting. Outer and inner CV are StratifiedGroupKFold on
    id_student, so a repeating student never sits on both sides of a split.
  - Temporal holdout. Train on the 2013B/2013J/2014B presentations, test on
    2014J; students who appear in 2014J are removed from the development set.
  - No full-frame preprocessing. No rows are filtered on feature values; encodings are
    stateless; fitted transforms (scalers) live in Pipelines fitted per
    training fold.
  - Non-circular k. Every selector returns a full ranking; stability and
    performance are reported over k, chosen before any run.
  - All five methods, ANOVA included, are scored on the holdout.

Comparators:
  - LASSO ranks features by entry on the L1 regularisation path (the largest
    penalty at which a coefficient becomes non-zero), ties broken by |coef|
    at the entry C.
  - MI declares binary/low-cardinality integer columns discrete, so the kNN
    estimator is not run on dummies.

Evaluators: three model families -- XGBoost, RandomForest and a scaled
logistic regression -- each tuned once per fold on all features (group-aware
inner CV) and refitted on each method's top-k, with columns always in the
data's own order (column-sampling learners depend on order). Every method
gets the same models, so differences come from the subset alone.

Run integrity: a run refuses to start from a dirty tree, refuses to resume or
reuse a holdout produced by a different commit, holds a per-file lock, and
analysis asserts one commit and a complete, duplicate-free set of folds. All-features baselines are
stored per fold and on the holdout.

xAPI-Edu-Data has no timestamps and one row per student, so it runs the same
loop with plain StratifiedKFold and is an end-of-term contrast, not an
early-warning test.
"""
import os
import sys
import json
import time
import argparse
import warnings
import fcntl
import hashlib
import zlib
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as student_t
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import l1_min_c
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, RESULTS_DIR, require_data  # noqa: E402

RANDOM_STATE = 42


def sanitize_columns(X):
    X = X.copy()
    X.columns = X.columns.str.replace(r"[\[\]<>]", "", regex=True)
    return X


def load_xapi(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "xapi_edu_data.csv"))
    # Binary reclassification consistent with OULAD framing: L (Low) -> 0 (Fail-analog),
    # M/H (Medium/High) -> 1 (Pass-analog), matching the xAPI context-dependence setup.
    target_col = "Class"
    df["target"] = df[target_col].apply(lambda x: 0 if x == "L" else 1)
    df = df.drop(columns=[target_col])
    for col in df.select_dtypes(include=["object"]).columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    X = sanitize_columns(df.drop(columns=["target"]))
    y = df["target"].astype(int)
    return X, y


def jaccard_index(s1, s2):
    s1, s2 = set(s1), set(s2)
    return len(s1 & s2) / len(s1 | s2)


def kuncheva_index(s1, s2, total_features):
    s1, s2 = set(s1), set(s2)
    k = len(s1)
    r = len(s1 & s2)
    numerator = (r * total_features) - (k ** 2)
    denominator = k * (total_features - k)
    return numerator / denominator if denominator != 0 else 0.0


def nadeau_bengio_ttest(diffs, n_test, n_train):
    """Corrected paired t-test for k-fold CV differences (Nadeau & Bengio,
    2003). diffs: array of per-fold (method_a - method_b) differences."""
    diffs = np.asarray(diffs)
    k = len(diffs)
    mean_diff = diffs.mean()
    sd_diff = diffs.std(ddof=1)
    n2_over_n1 = n_test / n_train
    var_nb = sd_diff ** 2 * (1.0 / k + n2_over_n1)
    se = np.sqrt(var_nb) if var_nb > 0 else 1e-12
    t_stat = mean_diff / se
    df = k - 1
    p_two_sided = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return {"mean_diff": float(mean_diff), "sd_diff": float(sd_diff),
            "t_stat": float(t_stat), "df": int(df), "p_value": float(p_two_sided),
            "se_nb": float(se)}


def benjamini_hochberg(p_values):
    """Manual BH correction (avoids adding a statsmodels dependency for one
    function). Returns adjusted p-values in the original order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p-value down
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out.tolist()


def concentration_gini(importances):
    """Gini coefficient of the (non-negative) importance mass across the
    selected subset -- how concentrated vs. diffuse the explanation is."""
    x = np.sort(np.abs(np.asarray(importances)))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def redundancy_mean_abs_corr(X_tr, selected_features):
    if len(selected_features) < 2:
        return 0.0
    corr = X_tr[selected_features].corr().abs()
    iu = np.triu_indices_from(corr, k=1)
    return float(np.nanmean(corr.values[iu]))


def nogueira_stability(feature_subsets, all_features):
    """Nogueira et al. (2018, JMLR 18:174) stability, eq. 4:
    1 - mean_f s_f^2 / ((k/d)(1 - k/d)), with the unbiased s_f^2 = M/(M-1) p_f(1-p_f)."""
    m, d = len(feature_subsets), len(all_features)
    idx = {f: i for i, f in enumerate(all_features)}
    Z = np.zeros((m, d))
    for i, subset in enumerate(feature_subsets):
        Z[i, [idx[f] for f in subset]] = 1
    p, k_bar = Z.mean(axis=0), Z.sum(axis=1).mean()
    if m < 2 or k_bar in (0, d):
        return float("nan")  # undefined, not "unstable"
    return 1 - (m / (m - 1)) * np.mean(p * (1 - p)) / ((k_bar / d) * (1 - k_bar / d))

METHODS = ["MI", "RFE", "LASSO", "ANOVA", "SHAP"]
K_GRID = [5, 10, 15, 20, 25, 30]  # fixed in advance
CUTOFFS = [30, 60, 90]
N_SPLITS, N_REPEATS = 5, 10
N_JOBS = int(os.environ.get("FS_N_JOBS", -1))
HOLDOUT_PRESENTATION = "2014J"
IMD_MAP = {"0-10%": 1, "10-20": 2, "20-30%": 3, "30-40%": 4, "40-50%": 5, "50-60%": 6,
           "60-70%": 7, "70-80%": 8, "80-90%": 9, "90-100%": 10}
KEYS = ["id_student", "code_module", "code_presentation"]
LASSO_CS = np.logspace(-4, 1, 30)

# Evaluator grids.
EVALUATORS = {
    "xgb": (XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          random_state=RANDOM_STATE, n_jobs=1),
            {"n_estimators": [150, 300, 600], "max_depth": [3, 4, 6], "learning_rate": [0.02, 0.03, 0.1],
             "min_child_weight": [5], "subsample": [0.8], "colsample_bytree": [0.8]}),
    "rf": (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
           {"min_samples_leaf": [1, 10, 50], "max_features": ["sqrt", 0.5]}),
    "lr": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
           {"logisticregression__C": [0.01, 0.1, 1, 10, 30, 100]}),
}


# ── Data ─────────────────────────────────────────────────────────────────
def load_oulad_at(day, data_dir=DATA_DIR, tma_lag=0):
    """Registration-level features using only data dated <= `day`.

    Returns X, y, meta (id_student, code_module, code_presentation) and the
    max VLE and assessment dates actually used (for the leakage assert)."""
    rd = lambda f: pd.read_csv(os.path.join(data_dir, f), na_values="?")
    info, reg = rd("studentInfo.csv"), rd("studentRegistration.csv")
    vle, vle_meta = rd("studentVle.csv"), rd("vle.csv")[["id_site", "activity_type"]]
    sa, assess = rd("studentAssessment.csv"), rd("assessments.csv")

    df = info.merge(reg[KEYS + ["date_registration", "date_unregistration"]], on=KEYS, how="left")
    # Not yet registered at the prediction point: cannot be scored then.
    # Missing registration date: kept, treated as registered; counted.
    n_no_reg_date = int(df["date_registration"].isna().sum())
    df = df[~(df["date_registration"] > day)].drop(columns="date_registration")
    # Withdrawals with no recorded date are kept: excluding them would select on the
    # final outcome, which is unknown at the cutoff.
    n_undated_wd = int(((df["final_result"] == "Withdrawn") & df["date_unregistration"].isna()).sum())
    df["_undated"] = df["date_unregistration"].isna()
    df = df[~(df["date_unregistration"] <= day)].drop(columns="date_unregistration")

    vle = vle[vle["date"] <= day].merge(vle_meta, on="id_site", how="left")
    clicks = vle.pivot_table(index=KEYS, columns="activity_type", values="sum_click",
                             aggfunc="sum", fill_value=0)
    clicks.columns = [f"vle_{c}" for c in clicks.columns]

    # A score counts only if both the submission and the assessment's deadline
    # fall on or before the cutoff -- marks are not returned before the deadline.
    # Banked scores (earlier-presentation results, date_submitted < 0) follow the same rule.
    sa = sa.merge(assess[["id_assessment", "code_module", "code_presentation", "date", "assessment_type"]],
                  on="id_assessment", how="left")
    # Sensitivity (`--tma-lag`): tutor-marked scores return after the deadline, so a TMA
    # counts only once deadline + lag <= day. CMAs are computer-marked: no lag.
    # Banked scores come from an earlier presentation and are already known: no lag.
    assert tma_lag >= 0, "a negative lag would admit marks before their deadline"
    lag = np.where((sa["assessment_type"] == "TMA") & (sa["is_banked"] == 0), tma_lag, 0)
    sa = sa[(sa["date_submitted"] <= day) & (sa["date"] + lag <= day)].dropna(subset=["score"])
    scores = sa.groupby(KEYS).agg(num_submissions=("id_assessment", "count"),
                                  avg_score=("score", "mean"), max_score=("score", "max"),
                                  min_score=("score", "min"))

    df = df.merge(clicks, on=KEYS, how="left").merge(scores, on=KEYS, how="left")
    feat_cols = list(clicks.columns) + list(scores.columns)
    df[feat_cols] = df[feat_cols].fillna(0)
    df["no_submission"] = (df["num_submissions"] == 0).astype(int)  # 0 score != no score
    df = df.dropna(subset=["studied_credits"])
    df["imd_missing"] = df["imd_band"].isna().astype(int)
    df["imd_band"] = df["imd_band"].map(IMD_MAP).fillna(0).astype(int)

    y = df["final_result"].isin(["Pass", "Distinction"]).astype(int)
    meta = df[KEYS].reset_index(drop=True)
    meta["undated_withdrawn"] = ((df["final_result"] == "Withdrawn") & df["_undated"]).to_numpy()
    X = df.drop(columns=KEYS + ["final_result", "_undated"])
    X = sanitize_columns(pd.get_dummies(X, drop_first=True).astype(float))
    used = {"max_vle_date": float(vle["date"].max()), "max_assess_date": float(sa["date"].max()),
            "n_undated_withdrawn_kept": n_undated_wd, "n_missing_registration_date_kept": n_no_reg_date}
    return X.reset_index(drop=True), y.reset_index(drop=True), meta, used


def prepare_xapi(data_dir=DATA_DIR):
    """xAPI frame used by every script: exact duplicate rows (X+y) dropped, then
    identical columns (Nationality_* == PlaceofBirth_* for five countries), which only
    create exact ties."""
    X, y = load_xapi(data_dir)
    keep = ~pd.concat([X, y], axis=1).duplicated()
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    dup_cols = list(X.columns[X.T.duplicated()])
    return X.drop(columns=dup_cols), y, {"dropped_duplicates": int((~keep).sum()), "dropped_identical_cols": dup_cols}


def split_dev_holdout(meta):
    """Temporal split: 2014J is the holdout; dev excludes any student seen in 2014J."""
    te = meta["code_presentation"] == HOLDOUT_PRESENTATION
    dev = ~te & ~meta["id_student"].isin(set(meta.loc[te, "id_student"]))
    assert not set(meta.loc[dev, "id_student"]) & set(meta.loc[te, "id_student"])
    return dev, te


# ── Selectors: each returns (FULL ranking best-first, info dict) ─────────
def _xgb():
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                         random_state=RANDOM_STATE, n_jobs=N_JOBS)


def _tie_rng(X):
    """Per-fold seed from the training rows, so tie-breaking is reproducible yet differs
    between folds. Column position must not break ties: it is the same in every fold and
    would make tied top-k sets agree by construction."""
    return np.random.default_rng(zlib.crc32(np.ascontiguousarray(X.index.to_numpy()).tobytes()))


def _max_tie_top(scores):
    """Largest group of exactly equal scores that reaches into the top max(K_GRID) ranks:
    the ties that can change a top-k set."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    top_vals = set(s[:max(K_GRID)])
    return int(max((s == v).sum() for v in top_vals))


def _order(scores, cols, rng):
    """Best first; exact ties broken at random (seeded per fold)."""
    scores = np.asarray(scores, dtype=float)
    return [cols[j] for j in np.lexsort((rng.random(len(scores)), -scores))]


DISCRETE_COLS = None  # set once per dataset by fix_discrete()


def discrete_mask(X):
    """Binary dummies and low-cardinality integer columns (<= 20 values) are discrete."""
    return np.array([(X[c] % 1 == 0).all() and X[c].nunique() <= 20 for c in X.columns])


def fix_discrete(X):
    """Fix the MI discrete/continuous split once from the whole dev frame (label-free),
    so borderline columns cannot flip between folds or replicates."""
    global DISCRETE_COLS
    DISCRETE_COLS = set(X.columns[discrete_mask(X)])
    return sorted(DISCRETE_COLS)


def rank_mi(X, y, g, cv):
    assert DISCRETE_COLS is not None, "call fix_discrete() on the dev frame first"
    mask = np.array([c in DISCRETE_COLS for c in X.columns])
    s = mutual_info_classif(X, y, discrete_features=mask, random_state=RANDOM_STATE)
    return _order(s, X.columns, _tie_rng(X)), {"n_nonzero": int((s > 0).sum()), "max_tie_top": _max_tie_top(s)}


def rank_anova(X, y, g, cv):
    # NaN F only for a column constant within the training fold: no signal, so rank it last.
    s = np.nan_to_num(f_classif(X, y)[0], nan=0.0)
    return _order(s, X.columns, _tie_rng(X)), {"n_nonzero": int((s > 0).sum()), "max_tie_top": _max_tie_top(s)}


def rank_rfe(X, y, g, cv):
    """Recursive feature elimination with XGB importances, one feature per step, as
    sklearn's RFE(step=1) but with exact importance ties broken at random (seeded per
    fold) instead of by column position. Returns the
    reverse elimination order (last survivor first)."""
    tie = dict(zip(X.columns, _tie_rng(X).random(X.shape[1])))
    remaining, eliminated, tied_steps = list(X.columns), [], 0
    while len(remaining) > 1:
        imp = _xgb().fit(X[remaining], y).feature_importances_
        if len(remaining) <= max(K_GRID) and (imp == imp.min()).sum() > 1:
            tied_steps += 1
        j = np.lexsort(([tie[c] for c in remaining], imp))[0]
        eliminated.append(remaining.pop(j))
    return remaining + eliminated[::-1], {"tied_steps_top": tied_steps}


def _l1_fit(Xs, y, C):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        coef = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000,
                                  random_state=RANDOM_STATE).fit(Xs, y).coef_[0]
    return np.abs(coef), sum(issubclass(x.category, ConvergenceWarning) for x in w)


def rank_lasso(X, y, g, cv):
    """Rank by order of entry on the L1 path (largest penalty first).

    Coarse-to-fine: a log grid from l1_min_c up to 1e6x (20 points per decade),
    swept in chunks until `need` features have entered; each interval in which
    several enter together is refined twice with 12 points. Ties left after
    that are ordered by |coef| at entry C, then at random (seeded per fold).
    On small folds (xAPI half-samples) the liblinear path is not monotone, so
    the sweep can need far more than 1000x. If `need`
    is still not reached, the tail is random, not column order, and the
    shortfall is recorded in `n_entered`."""
    Xs = StandardScaler().fit_transform(X)
    c0 = l1_min_c(Xs, y, loss="log")
    p = X.shape[1]
    entry_C = np.full(p, np.inf)
    entry_mag = np.zeros(p)
    n_conv = 0

    def sweep(Cs):
        nonlocal n_conv
        fits = Parallel(n_jobs=N_JOBS)(delayed(_l1_fit)(Xs, y, C) for C in Cs)
        for C, (mag, nw) in zip(Cs, fits):
            n_conv += nw
            new = (mag > 1e-10) & np.isinf(entry_C)
            entry_C[new] = C
            entry_mag[new] = mag[new]

    # Only the top max(K_GRID, p/2) ranks are ever used, so the sweep stops once
    # that many features have entered; later (slow, large-C) fits cannot change them.
    need = min(int((X.nunique() > 1).sum()), max(max(K_GRID), p // 2))
    coarse = c0 * np.logspace(0, 6, 121)
    for chunk in np.array_split(coarse, 12):
        sweep(chunk)
        if np.isfinite(entry_C).sum() >= need:
            break
    for _ in range(2):  # refine intervals holding tied entries
        vals, counts = np.unique(entry_C[np.isfinite(entry_C)], return_counts=True)
        tied = vals[counts > 1]
        if not len(tied):
            break
        ratio = coarse[1] / coarse[0]
        refine = np.concatenate([np.exp(np.linspace(np.log(C / ratio), np.log(C), 13)[1:]) for C in tied])
        entry_C[np.isin(entry_C, tied)] = np.inf
        sweep(np.sort(refine))
        ratio = ratio ** (1 / 12)
    n_entered = int(np.isfinite(entry_C).sum())
    # Never-entered features sort last; then entry C, magnitude at entry, random.
    tie = _tie_rng(X).random(p)
    order = sorted(range(p), key=lambda j: (entry_C[j], -entry_mag[j], tie[j]))
    _, counts = np.unique(entry_C[np.isfinite(entry_C)], return_counts=True)
    return [X.columns[j] for j in order], {
        "l1_min_c": float(c0), "n_entered": n_entered, "need": need, "n_never_enter": int(np.isinf(entry_C).sum()),
        "max_tie_group": int(counts.max()) if len(counts) else 0, "convergence_warnings": n_conv}


def rank_shap(X, y, g, cv):
    model = _xgb().fit(X, y)
    s = np.abs(shap.TreeExplainer(model).shap_values(X)).mean(axis=0)
    return _order(s, X.columns, _tie_rng(X)), {"n_nonzero": int((s > 0).sum()), "max_tie_top": _max_tie_top(s)}


RANKERS = {"MI": rank_mi, "RFE": rank_rfe, "LASSO": rank_lasso, "ANOVA": rank_anova, "SHAP": rank_shap}


# ── Interpretability axis (model's own SHAP order, training means) ─────────
def _deletion_curve(model, X_tr, X_te, features, order):
    """Per-instance probability of the step-0 predicted class as features are replaced
    by their training mean in `order` (Samek et al. 2017; Petsiuk et al. 2018)."""
    X_work = X_te[features].copy()
    means = X_tr[features].mean()
    p = model.predict_proba(X_work)[:, 1]
    cls = p >= 0.5
    curve = [np.where(cls, p, 1 - p).mean()]
    for f in order:
        X_work[f] = means[f]
        p = model.predict_proba(X_work)[:, 1]
        curve.append(np.where(cls, p, 1 - p).mean())
    return np.array(curve)


def _norm_deletion(c, min_span=0.01):
    """1 - (mean height - end) / (start - end): share of the total drop achieved on
    average along the curve. NaN when the total drop is below `min_span`."""
    span = c[0] - c[-1]
    if span < min_span:
        return float("nan")
    return float(1 - (np.trapezoid(c) / (len(c) - 1) - c[-1]) / span)


def faithfulness(model, X_tr, X_te, features, order, n_random=20, seed=None):
    """Normalised deletion score: the share of the total confidence drop
    (start -> all features replaced) achieved on average along the curve,
    1 - (mean height - end) / (start - end). Higher = the order removes what the
    model relies on sooner. Normalising per model removes model confidence, which
    otherwise dominated cross-method differences. A bespoke
    normalised deletion score, not AOPC (the curve follows Samek et al. 2017); its
    ceiling depends on k, so compare methods only within k. Also returns the same
    score for `n_random` random orders of the same model, seeded per fold and k
    (`seed`), and the gain over them. NaN when the total drop is < 0.01."""
    own_curve = _deletion_curve(model, X_tr, X_te, features, order)
    rng = np.random.default_rng(seed)
    rand = [_norm_deletion(_deletion_curve(model, X_tr, X_te, features, list(rng.permutation(order))))
            for _ in range(n_random)]
    own = _norm_deletion(own_curve)
    return {"faithfulness": own, "faithfulness_random": float(np.mean(rand)),
            "faithfulness_gain": own - float(np.mean(rand)),
            "deletion_start": float(own_curve[0]), "deletion_end": float(own_curve[-1])}


# ── One train/test evaluation ────────────────────────────────────────────
def _scores(y, prob):
    return {"roc_auc": float(roc_auc_score(y, prob)), "accuracy": float(accuracy_score(y, prob >= 0.5)),
            "f1": float(f1_score(y, prob >= 0.5))}


def evaluate(X_tr, y_tr, g_tr, X_te, y_te, inner_cv, keep_probs=False):
    """Tune each evaluator once on all features, rank with every method,
    then refit each evaluator on each method's top-k."""
    tuned, out = {}, {"params": {}, "baseline_all": {}, "methods": {}}
    probs = {}
    for ev, (est, grid) in EVALUATORS.items():
        gs = GridSearchCV(clone(est), grid, scoring="roc_auc", cv=inner_cv, n_jobs=N_JOBS)
        gs.fit(X_tr, y_tr, groups=g_tr)
        tuned[ev] = gs.best_params_
        out["params"][ev] = gs.best_params_
        p = gs.best_estimator_.predict_proba(X_te)[:, 1]
        out["baseline_all"][ev] = _scores(y_te, p)
        probs[("ALL", "all", ev)] = p
    for m in METHODS:
        ranking, info = RANKERS[m](X_tr, y_tr, g_tr, inner_cv)
        per_k = {}
        for k in K_GRID:
            top = set(ranking[:k])
            # The data's column order, not the ranking's: XGB colsample and RF max_features
            # sample by position, so order alone would change the AUC.
            sel = [c for c in X_tr.columns if c in top]
            rec = {}
            for ev, (est, _) in EVALUATORS.items():
                model = clone(est).set_params(**tuned[ev])
                if ev != "lr":
                    model.set_params(n_jobs=N_JOBS)
                model.fit(X_tr[sel], y_tr)
                p = model.predict_proba(X_te[sel])[:, 1]
                rec[ev] = _scores(y_te, p)
                probs[(m, str(k), ev)] = p
                if ev == "xgb":
                    imp = np.abs(shap.TreeExplainer(model).shap_values(X_tr[sel])).mean(axis=0)
                    own_order = [sel[i] for i in np.argsort(-imp, kind="stable")]
                    seed = [zlib.crc32(np.ascontiguousarray(X_tr.index.to_numpy()).tobytes()), k]
                    rec.update(faithfulness(model, X_tr, X_te, sel, own_order, seed=seed))
                    assert imp.sum() > 0, "all-zero SHAP importances: Gini undefined"
                    rec["concentration_gini"] = concentration_gini(imp)
            rec["redundancy"] = redundancy_mean_abs_corr(X_tr, sel)
            rec["redundancy_const_cols"] = int((X_tr[sel].nunique() <= 1).sum())  # skipped by nanmean
            per_k[str(k)] = rec
        out["methods"][m] = {"ranking": ranking, "info": info, "per_k": per_k}
    return (out, probs) if keep_probs else out


def read_jsonl(path, repair=False):
    """Records of a checkpoint file, ignoring a partial trailing line left by a kill
    mid-write. `repair=True` (only while holding the run's lock) also truncates it,
    so a resumed run appends cleanly; read-only callers never modify the file."""
    if not os.path.exists(path):
        return []
    with open(path, "rb+" if repair else "rb") as f:
        data = f.read()
        end = data.rfind(b"\n") + 1
        if repair and end < len(data):
            f.truncate(end)
    return [json.loads(l) for l in data[:end].decode().splitlines() if l.strip()]


def lock(path):
    """Exclusive lock on `path`; raises if another run holds it. The OS releases
    it when the process exits or is killed. Keep the returned handle alive."""
    fh = open(path + ".lock", "w")
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fh


def check_same_commit(shas, what):
    """Refuse to mix outputs of different code. Keyed on the
    scripts/ tree hash, so committing results or docs does not block a resume."""
    head = provenance()["code_sha"]
    stale = set(shas) - {head}
    assert not stale, f"{what} came from commit(s) {sorted(stale)}, HEAD is {head}: clear the output dir"


def run(name, X, y, groups, out_dir, n_repeats=N_REPEATS):
    """Repeated (grouped) CV with per-fold JSONL checkpoints and resume."""
    path = os.path.join(out_dir, f"{name}_fold_checkpoints.jsonl")
    recs = read_jsonl(path, repair=True)
    check_same_commit([r["sha"] for r in recs], f"{name} fold checkpoints")
    done = {r["fold"] for r in recs}
    sha = provenance()["code_sha"]
    t0 = time.time()
    for rep in range(n_repeats):
        if groups is None:
            outer = StratifiedKFold(N_SPLITS, shuffle=True, random_state=rep)
            inner = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
        else:
            outer = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=rep)
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_STATE)
        for i, (tr, te) in enumerate(outer.split(X, y, groups)):
            fold = rep * N_SPLITS + i
            if fold in done:
                continue
            g_tr = None if groups is None else groups.iloc[tr]
            if groups is not None:
                assert not set(g_tr) & set(groups.iloc[te]), "student on both sides of a fold"
            t = time.time()
            rec = evaluate(X.iloc[tr], y.iloc[tr], g_tr, X.iloc[te], y.iloc[te], inner)
            rec.update(fold=fold, repeat=rep, n_train=len(tr), n_test=len(te), sha=sha, git_dirty=False)
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{name}] fold {fold} {time.time()-t:.0f}s (total {time.time()-t0:.0f}s) "
                  f"lasso_conv_warn={rec['methods']['LASSO']['info']['convergence_warnings']}", flush=True)
    return path


# ── Holdout (baselines, stored probabilities, cluster-bootstrap CIs, per module) ─
def _cluster_boot_idx(groups, rng):
    """Row indices for one student-cluster bootstrap resample."""
    codes, uniq = pd.factorize(groups)
    rows_by = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([rows_by[i] for i in pick])


def holdout(X, y, meta, dev, te, out_dir, name, n_boot=1000):
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_STATE)
    rec, probs = evaluate(X[dev], y[dev], meta.loc[dev, "id_student"], X[te], y[te], inner,
                          keep_probs=True)
    y_te, g_te, mod_te = y[te].to_numpy(), meta.loc[te, "id_student"].to_numpy(), meta.loc[te, "code_module"].to_numpy()
    np.savez_compressed(os.path.join(out_dir, f"{name}_holdout_probs.npz"),
                        y=y_te, id_student=g_te, code_module=mod_te,
                        **{"|".join(k): v for k, v in probs.items()})

    rng = np.random.default_rng(RANDOM_STATE)
    boots = [_cluster_boot_idx(g_te, rng) for _ in range(n_boot)]
    auc_b = lambda p: np.array([roc_auc_score(y_te[b], p[b]) for b in boots])
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    boot = {}
    for ev in EVALUATORS:
        base = auc_b(probs[("ALL", "all", ev)])
        for k in K_GRID:
            ks = str(k)
            a = {m: auc_b(probs[(m, ks, ev)]) for m in METHODS}
            boot[f"{ev}_k{k}"] = {
                "auc_ci": {m: ci(a[m]) for m in METHODS},
                "diff_vs_all_ci": {m: ci(a[m] - base) for m in METHODS},
                "pairwise_diff_ci": {f"{m1}_vs_{m2}": ci(a[m1] - a[m2]) for m1, m2 in combinations(METHODS, 2)},
            }
    # Sensitivity: holdout AUC without the kept undated withdrawals.
    keep = ~meta.loc[te, "undated_withdrawn"].to_numpy()
    excl_uw = {"n_excluded": int((~keep).sum()),
               **{ev: {"all": float(roc_auc_score(y_te[keep], probs[("ALL", "all", ev)][keep])),
                       **{m: {str(k): float(roc_auc_score(y_te[keep], probs[(m, str(k), ev)][keep]))
                              for k in K_GRID} for m in METHODS}} for ev in EVALUATORS}}
    per_module = {}
    for mod in sorted(set(mod_te)):
        s = mod_te == mod
        per_module[mod] = {"n": int(s.sum()), "pass_rate": float(y_te[s].mean()),
                           "auc_xgb_k15": {m: float(roc_auc_score(y_te[s], probs[(m, "15", "xgb")][s]))
                                           for m in METHODS},
                           "auc_xgb_all": float(roc_auc_score(y_te[s], probs[("ALL", "all", "xgb")][s]))}
    dev_mod = meta.loc[dev, "code_module"].value_counts(normalize=True).round(4).to_dict()
    rec.update(provenance=provenance(with_data=True), n_dev=int(dev.sum()), n_holdout=int(te.sum()),
               n_dev_dropped_overlap=int((~te & ~dev).sum()), n_boot=n_boot, bootstrap=boot,
               bootstrap_note="percentile CIs, descriptive: no multiplicity adjustment",
               per_module=per_module, auc_excl_undated_withdrawn=excl_uw, dev_pass_rate=float(y[dev].mean()),
               holdout_pass_rate=float(y_te.mean()), dev_module_share=dev_mod)
    with open(os.path.join(out_dir, f"{name}_holdout.json"), "w") as f:
        json.dump(rec, f, indent=2)
    return rec


@lru_cache(maxsize=None)
def data_manifest(data_dir=DATA_DIR):
    """SHA-256 of every raw CSV used."""
    out = {}
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".csv"):
            h = hashlib.sha256()
            with open(os.path.join(data_dir, f), "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[f] = h.hexdigest()
    return out


def provenance(with_data=False):
    import subprocess, sklearn, xgboost
    here = os.path.dirname(os.path.abspath(__file__))
    git = lambda *a: subprocess.run(["git", "-C", here, *a], capture_output=True, text=True).stdout.strip()
    return {"git_sha": git("rev-parse", "HEAD"), "code_sha": git("rev-parse", "HEAD:./"), "git_dirty": bool(git("status", "--porcelain", "--", ".")),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                         "xgboost": xgboost.__version__, "shap": shap.__version__},
            "FS_N_JOBS": N_JOBS, **({"data_sha256": data_manifest()} if with_data else {})}


# ── Analysis ─────────────────────────────────────────────────────────────
def _pairwise(values, n_test, n_train, n_repeats):
    """NB-corrected tests on fold-level performance for all 10 method pairs,
    BH-adjusted as one family. A heuristic: df is set conservatively to
    n_repeats - 1 (Bouckaert & Frank) rather than n_folds - 1.
    Zero-variance pairs get p=None. BH is applied afterwards in analyze() over
    one declared family per evaluator: 10 pairs x K_GRID."""
    tests = {}
    for a, b in combinations(METHODS, 2):
        d = np.array(values[a]) - np.array(values[b])
        if np.abs(d).max() >= 1e-12 and d.std() > 0:  # identical subsets differ only by float noise
            t = nadeau_bengio_ttest(d, n_test, n_train)
            t["df"] = n_repeats - 1
            t["p_value"] = float(2 * (1 - student_t.cdf(abs(t["t_stat"]), t["df"])))
        else:
            t = {"mean_diff": float(d.mean()), "p_value": None}
        tests[f"{a}_vs_{b}"] = t
    return tests


def analyze(ckpt, name, extra=None, n_repeats=N_REPEATS):
    recs = sorted(read_jsonl(ckpt), key=lambda r: r["fold"])
    folds = [r["fold"] for r in recs]
    assert folds == list(range(N_SPLITS * n_repeats)), f"{name}: folds incomplete or duplicated"
    shas = {r["sha"] for r in recs}
    assert len(shas) == 1 and not any(r["git_dirty"] for r in recs), f"{name}: mixed commits {shas}"
    feats = sorted(recs[0]["methods"]["MI"]["ranking"])
    n_test = int(np.mean([r["n_test"] for r in recs]))
    n_train = int(np.mean([r["n_train"] for r in recs]))
    reps = sorted({r["repeat"] for r in recs})
    res = {"dataset": name, "n_folds": len(recs), "n_repeats": len(reps), "k_grid": K_GRID,
           "n_features": len(feats), "per_k": {}, "provenance": provenance(with_data=True),
           "fold_sha": shas.pop(), **(extra or {})}
    summ = lambda v: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}  # sample SD
    evs = list(EVALUATORS)

    res["baseline_all"] = {ev: {met: summ([r["baseline_all"][ev][met] for r in recs])
                                for met in recs[0]["baseline_all"][ev]} for ev in evs}
    assert max(K_GRID) < len(feats), "k >= p: Kuncheva/Nogueira degenerate"
    res["ranking_info"] = {m: {key: {**summ(v), "min": float(np.min(v))}
                               for key in recs[0]["methods"][m]["info"]
                               for v in [[r["methods"][m]["info"][key] for r in recs]]}
                           for m in METHODS}

    # Full stability curve k = 1..p/2 from the stored rankings.
    res["stability_curve_nogueira"] = {
        m: {str(k): float(np.mean([nogueira_stability(
            [r["methods"][m]["ranking"][:k] for r in recs if r["repeat"] == rep], feats) for rep in reps]))
            for k in range(1, max(len(feats) // 2, max(K_GRID)) + 1)} for m in METHODS}

    nog_by_k = {m: [] for m in METHODS}
    for k in K_GRID:
        ks = str(k)
        stab = {m: {"Jaccard": [], "Kuncheva": [], "Nogueira": []} for m in METHODS}
        for rep in reps:
            rr = [r for r in recs if r["repeat"] == rep]
            for m in METHODS:
                subs = [r["methods"][m]["ranking"][:k] for r in rr]
                stab[m]["Jaccard"].append(np.mean([jaccard_index(a, b) for a, b in combinations(subs, 2)]))
                stab[m]["Kuncheva"].append(np.mean([kuncheva_index(a, b, len(feats)) for a, b in combinations(subs, 2)]))
                stab[m]["Nogueira"].append(nogueira_stability(subs, feats))
        for m in METHODS:
            nog_by_k[m].append(stab[m]["Nogueira"])
        perf = {m: {ev: [r["methods"][m]["per_k"][ks][ev]["roc_auc"] for r in recs] for ev in evs}
                for m in METHODS}
        interp = {m: {met: summ([r["methods"][m]["per_k"][ks][met] for r in recs])
                      for met in ("faithfulness", "faithfulness_random", "faithfulness_gain",
                                  "concentration_gini", "redundancy")} for m in METHODS}
        res["per_k"][ks] = {
            "stability": {m: {i: summ(v) for i, v in stab[m].items()} for m in METHODS},
            "roc_auc": {m: {ev: summ(v) for ev, v in perf[m].items()} for m in METHODS},
            "interpretability": interp,
            "pairwise_roc_auc": {ev: _pairwise({m: perf[m][ev] for m in METHODS}, n_test, n_train, len(reps))
                                 for ev in evs},
        }

    # BH family, fixed in advance: per evaluator, all
    # 10 method pairs x every k in K_GRID. XGB is the primary evaluator; RF and LR
    # are robustness checks.
    for ev in evs:
        cells = [(ks, pair) for ks in map(str, K_GRID) for pair, t in res["per_k"][ks]["pairwise_roc_auc"][ev].items()
                 if t["p_value"] is not None]
        adj = benjamini_hochberg([res["per_k"][ks]["pairwise_roc_auc"][ev][pair]["p_value"] for ks, pair in cells])
        for (ks, pair), pa in zip(cells, adj):
            res["per_k"][ks]["pairwise_roc_auc"][ev][pair]["p_adjusted_bh"] = float(pa)
    res["bh_family"] = {"per_evaluator": "10 pairs x K_GRID", "primary_evaluator": "xgb"}

    # Descriptive only: inference on stability comes from stability_bootstrap.py
    # (student-level bootstrap), not from tests across CV repeats.
    curve = res["stability_curve_nogueira"]
    p = len(feats)
    k_sets = {"grid": K_GRID, "k_le_p3": list(range(1, p // 3 + 1)), "all": list(range(1, p // 2 + 1))}
    res["k_summaries_nogueira"] = {name: {"k": ks, "mean": {m: float(np.mean([curve[m][str(k)] for k in ks]))
                                                           for m in METHODS}}
                                   for name, ks in k_sets.items()}
    res["k_grid_repeat_range"] = {m: [float(min(v)), float(max(v))]
                                  for m, v in ((m, np.mean(nog_by_k[m], axis=0)) for m in METHODS)}
    with open(ckpt.replace("_fold_checkpoints.jsonl", "_analysis.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["oulad", "xapi"], required=True)
    ap.add_argument("--cutoff", type=int, choices=CUTOFFS)
    ap.add_argument("--tma-lag", type=int, default=0, help="sensitivity: days before a TMA mark counts")
    ap.add_argument("--n-repeats", type=int, default=N_REPEATS)
    ap.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "temporal_design"))
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    require_data()
    os.makedirs(a.output_dir, exist_ok=True)
    if not a.analyze_only:
        assert not provenance()["git_dirty"], "uncommitted changes in scripts/: commit before a run"

    if a.dataset == "oulad":
        name = f"oulad_day{a.cutoff}" + (f"_tmalag{a.tma_lag}" if a.tma_lag else "")
        X, y, meta, used = load_oulad_at(a.cutoff, tma_lag=a.tma_lag)
        assert used["max_vle_date"] <= a.cutoff and used["max_assess_date"] <= a.cutoff, used
        dev, te = split_dev_holdout(meta)
        const = [c for c in X.columns if X.loc[dev, c].nunique() <= 1]  # e.g. vle_htmlactivity
        X = X.drop(columns=const)
        extra = {"cutoff_day": a.cutoff, "tma_lag": a.tma_lag, "dates_used": used, "dropped_dev_constant": const}
        hpath = os.path.join(a.output_dir, f"{name}_holdout.json")
        if not a.analyze_only and os.path.exists(hpath):
            check_same_commit([json.load(open(hpath))["provenance"]["code_sha"]], f"{name} holdout")
        elif not a.analyze_only:
            _lk = lock(hpath)
            print(f"[{name}] n={len(X)} p={X.shape[1]} pass_rate={y.mean():.3f} dropped={const}", flush=True)
            fix_discrete(X[dev])
            holdout(X, y, meta, dev, te, a.output_dir, name)
        X, y, g = X[dev].reset_index(drop=True), y[dev].reset_index(drop=True), \
            meta.loc[dev, "id_student"].reset_index(drop=True)
        extra["mi_discrete_cols"] = fix_discrete(X)
    else:
        X, y, info = prepare_xapi(DATA_DIR)
        g, name = None, "xapi_endofterm"
        extra = {"note": "end-of-term aggregates; no prediction point", **info, "mi_discrete_cols": fix_discrete(X)}
    ck = os.path.join(a.output_dir, f"{name}_fold_checkpoints.jsonl")
    if not a.analyze_only:
        _lk = lock(ck)
        ck = run(name, X, y, g, a.output_dir, a.n_repeats)
    res = analyze(ck, name, extra, a.n_repeats)
    if a.dataset == "oulad":
        h_sha = json.load(open(hpath))["provenance"]["code_sha"]
        assert h_sha == res["fold_sha"], f"holdout ({h_sha}) and folds ({res['fold_sha']}) differ"
    print(f"[{name}] done", flush=True)
