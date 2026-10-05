import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from rfe_mechanism import substitution  # noqa: E402
from stability_vs_n import draw_disjoint  # noqa: E402
from stability_bootstrap import replicate  # noqa: E402
from temporal_design import rank_anova  # noqa: E402


def _corr():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    X = pd.DataFrame({"a": a, "a2": a + 0.01 * rng.normal(size=500),
                      "b": rng.normal(size=500), "c": rng.normal(size=500), "d": rng.normal(size=500)})
    c = X.corr(method="spearman").abs()
    np.fill_diagonal(c.values, np.nan)
    return c


def test_substitution_detects_near_duplicate_swaps():
    # Folds agree on b and swap a <-> a2: each swapped-out feature is scored against its replacement.
    out = substitution([["a", "b", "c", "d"], ["a2", "b", "d", "c"]], _corr(), 2)
    assert out["swaps_per_pair"] == 1
    assert out["swapped_mean_maxabsr"] > 0.99
    assert out["share_swapped_ge_0.7"] == 1


def test_substitution_ignores_shared_features():
    # c replaces a; a is strongly correlated with the shared a2, but not with its replacement c.
    out = substitution([["a2", "a", "b", "d"], ["a2", "c", "b", "d"]], _corr(), 2)
    assert out["swapped_mean_maxabsr"] < 0.3
    assert out["share_swapped_ge_0.7"] == 0
    assert np.isnan(substitution([["a", "b"], ["a", "b"]], _corr(), 2)["swapped_mean_maxabsr"])


def test_draw_disjoint_sizes_and_overlap():
    g = pd.Series(np.repeat(np.arange(100), 2))  # 100 students, 2 rows each
    s1, s2 = draw_disjoint(g, 30, np.random.default_rng(1))
    assert len(s1) == len(s2) == 60
    assert not set(g[s1]) & set(g[s2]) and g[s1].nunique() == 30


def test_replicate_draws_n_students():
    rng = np.random.default_rng(2)
    g = pd.Series(np.repeat(np.arange(200), 2))
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("wxyz"))
    y = pd.Series(rng.integers(0, 2, 400))
    rankings, _, train_rows = replicate(X, y, g, 0, n_students=50, seed=7, rankers={"ANOVA": rank_anova})
    assert len(rankings["ANOVA"]) == 5 and sum(train_rows) == 4 * 100  # each row trains in 4 of 5 folds
