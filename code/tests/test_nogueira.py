import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from temporal_design import nogueira_stability  # noqa: E402


def test_nogueira_matches_published_formula():
    feats = list("abcdefghij")
    # Hand-computed with eq. 4 of Nogueira et al. (2018): M=3, d=10, k=3.
    # p = [1, 2/3, 1/3, 1/3, 1/3, 0...]; mean p(1-p) = (2/9)*4/10 = 4/45
    # 1 - (3/2)(4/45)/(0.3*0.7) = 1 - (2/15)/0.21 = 23/63
    subsets = [["a", "b", "c"], ["a", "b", "d"], ["a", "c", "e"]]
    assert abs(nogueira_stability(subsets, feats) - 23 / 63) < 1e-12
    assert nogueira_stability([["a", "b"]] * 4, feats) == 1.0


def test_faithfulness_undefined_for_a_model_that_ignores_its_inputs():
    import math
    import numpy as np
    import pandas as pd
    from temporal_design import faithfulness

    class Const:
        def predict_proba(self, X):
            return np.tile([0.3, 0.7], (len(X), 1))

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 5)), columns=list("abcde"))
    assert math.isnan(faithfulness(Const(), X, X, list("abcde"), list("abcde"))["faithfulness"])
