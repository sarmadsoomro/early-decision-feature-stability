"""Hand-computed checks for every statistic behind a reported number."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import temporal_design as T  # noqa: E402
from temporal_design import (jaccard_index, kuncheva_index, nadeau_bengio_ttest,  # noqa: E402
                             benjamini_hochberg, concentration_gini, redundancy_mean_abs_corr)


def test_overlap_indices():
    assert T.nogueira_stability([["a", "b"], ["a", "c"]], list("abcd")) == 0.0  # M=2, d=4
    assert abs(kuncheva_index(list("abc"), list("abd"), 10) - 11 / 21) < 1e-12  # (rd - k^2)/(k(d-k))
    assert jaccard_index("abc", "abd") == 0.5


def test_nadeau_bengio_and_bh():
    d = np.array([.01, .02, -.005, .015, .03])
    t = d.mean() / np.sqrt((1 / 5 + 20 / 80) * d.var(ddof=1))  # corrected variance, J=5, n2/n1=1/4
    assert abs(nadeau_bengio_ttest(d, 20, 80)["t_stat"] - t) < 1e-12
    p = np.array([.01, .04, .03, .2, .001, .04])
    assert np.allclose(benjamini_hochberg(p), false_discovery_control(p))


def test_identical_subsets_get_no_p_value():
    vals = {m: [0.7 + 1e-17 * i, 0.71, 0.69] for i, m in enumerate(T.METHODS)}
    assert all(t["p_value"] is None for t in T._pairwise(vals, 20, 80, 3).values())


def test_gini_and_redundancy():
    assert abs(concentration_gini([0, 0, 0, 1]) - 0.75) < 1e-12 and abs(concentration_gini([1, 1, 1, 1])) < 1e-12
    X = pd.DataFrame({"a": [1, 2, 3, 4.], "b": [2, 4, 6, 8.], "c": [1, -1, 1, -1.]})
    expect = (1 + 2 * abs(np.corrcoef(X.a, X.c)[0, 1])) / 3
    assert abs(redundancy_mean_abs_corr(X, ["a", "b", "c"]) - expect) < 1e-12


def test_faithfulness_is_normalised_and_order_sensitive():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(4000, 3)), columns=["a", "b", "c"])

    def relies_on_a(slope):
        class M:
            def predict_proba(self, X):
                p = 1 / (1 + np.exp(-slope * X["a"].to_numpy()))
                return np.c_[1 - p, p]
        return M()

    tr, te = X.iloc[:2000], X.iloc[2000:]
    strong, weak = relies_on_a(10), relies_on_a(0.5)
    f = lambda m, order: T.faithfulness(m, tr, te, ["a", "b", "c"], order)["faithfulness"]
    # higher = better; the true order wins for a confident and an unconfident model alike,
    # and an unconfident model in the wrong order no longer beats a confident one in the right order
    assert f(strong, ["a", "b", "c"]) > f(strong, ["b", "c", "a"])
    assert f(weak, ["a", "b", "c"]) > f(weak, ["b", "c", "a"])
    assert f(strong, ["a", "b", "c"]) > f(weak, ["b", "c", "a"])
    assert abs(f(strong, ["a", "b", "c"]) - f(weak, ["a", "b", "c"])) < 1e-9  # confidence-invariant here
    assert T.faithfulness(strong, tr, te, ["a", "b", "c"], ["a", "b", "c"])["faithfulness_gain"] > 0


def test_ties_are_not_broken_by_column_position():
    cols = [f"f{i}" for i in range(20)]
    scores = np.r_[np.arange(5, 0, -1), np.zeros(15)]  # 15-way tie after rank 5
    orders = {tuple(T._order(scores, cols, T._tie_rng(pd.DataFrame(index=np.arange(i, i + 50)))))
              for i in range(10)}
    assert all(o[:5] == tuple(cols[:5]) for o in orders)  # strict order kept
    assert len({o[5:] for o in orders}) > 1  # tied block differs between folds


def test_rankers_put_signal_first():
    rng = np.random.default_rng(1)  # f0 and f1 each carry marginal signal; f2..f7 are noise
    Xc = pd.DataFrame(rng.normal(size=(600, 8)), columns=[f"f{i}" for i in range(8)])
    yc = pd.Series((1.5 * Xc.f0 + Xc.f1 + rng.normal(scale=0.5, size=600) > 0).astype(int))
    T.fix_discrete(Xc)
    old = T.K_GRID
    T.K_GRID = [4]  # p=8: the LASSO sweep needs max(K_GRID, p/2) entries
    try:
        for m, rank in T.RANKERS.items():
            assert set(rank(Xc, yc, None, None)[0][:2]) == {"f0", "f1"}, m
    finally:
        T.K_GRID = old


def test_read_jsonl_partial_trailing_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"fold": 0}\n{"fold": 1}\n{"fo')
    assert [r["fold"] for r in T.read_jsonl(str(p))] == [0, 1]
    assert p.read_text().endswith('{"fo')  # read-only callers never modify the file
    assert [r["fold"] for r in T.read_jsonl(str(p), repair=True)] == [0, 1]
    assert p.read_text() == '{"fold": 0}\n{"fold": 1}\n'


def test_normalised_deletion_hand_value():
    # trapezoid mean of [1, .6, .5, .5] = (0.8 + 0.55 + 0.5) / 3; 1 - (0.6167 - 0.5) / 0.5 = 0.7667
    assert abs(T._norm_deletion(np.array([1, .6, .5, .5])) - 0.76667) < 1e-4
    assert np.isnan(T._norm_deletion(np.array([.6, .6, .595])))  # total drop < 0.01


def test_faithfulness_random_baseline_differs_by_seed():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 6)), columns=list("abcdef"))

    class Lin:
        def predict_proba(self, X):
            p = 1 / (1 + np.exp(-X.to_numpy() @ np.array([3, 2, 1, .5, .2, .1])))
            return np.c_[1 - p, p]

    r = [T.faithfulness(Lin(), X, X, list("abcdef"), list("abcdef"), seed=[s, 6])["faithfulness_random"]
         for s in (1, 2)]
    assert r[0] != r[1]


def test_rfe_matches_sklearn_without_ties_and_randomises_ties():
    from sklearn.feature_selection import RFE
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(400, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X.f0 + 0.6 * X.f1 + 0.3 * X.f2 + rng.normal(scale=.5, size=400) > 0).astype(int))
    ours = T.rank_rfe(X, y, None, None)[0]
    sk = RFE(T._xgb(), n_features_to_select=1).fit(X, y).ranking_
    assert ours == list(pd.Series(sk, index=X.columns).sort_values(kind="stable").index)
    # 12 constant-in-practice noise columns (never split on) tie at zero importance
    Z = pd.concat([X[["f0"]], pd.DataFrame(0.0, index=X.index, columns=[f"z{i}" for i in range(12)])], axis=1)
    tails = {tuple(T.rank_rfe(Z.set_axis(np.arange(i, i + 400)), y.set_axis(np.arange(i, i + 400)), None, None)[0][1:])
             for i in range(5)}
    assert len(tails) > 1  # tied features are not eliminated in column order
