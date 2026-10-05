import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from rfe_mechanism import substitution  # noqa: E402


def test_substitution_detects_near_duplicate_swaps():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    X = pd.DataFrame({"a": a, "a2": a + 0.01 * rng.normal(size=500),
                      "b": rng.normal(size=500), "c": rng.normal(size=500)})
    corr = X.corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    # Two folds that agree on b and swap a <-> a2: the swap is between near-duplicates.
    out = substitution([["a", "b", "c"], ["a2", "b", "c"]], corr, 2)
    assert out["swaps_per_pair"] == 1
    assert out["swapped_mean_maxabsr"] > 0.99
    assert out["share_swapped_ge_0.7"] == 1
    assert out["reference_mean_maxabsr"] < out["swapped_mean_maxabsr"]
    # Identical rankings: no swaps.
    assert np.isnan(substitution([["a", "b"], ["a", "b"]], corr, 2)["swapped_mean_maxabsr"])
