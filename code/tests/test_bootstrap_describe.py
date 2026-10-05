"""Hand check of the half-sampling SE, 95% CI and Bonferroni interval (audit 20261005 A1)."""
import os, sys, json
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import stability_bootstrap as sb  # noqa: E402
from temporal_design import METHODS, nogueira_stability  # noqa: E402


def test_describe_se_ci_and_bonferroni(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "data_manifest", lambda: {"x.csv": "D"})
    feats = [f"f{i}" for i in range(60)]
    rng = np.random.default_rng(0)
    recs = [{"b": b, "sha": "S", "lasso_n_entered": [[30, 30]],
             "rankings": {m: [list(rng.permutation(feats)) for _ in range(5)] for m in METHODS}} for b in range(4)]
    path = tmp_path / "boot.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    curve = {m: {str(k): 0.5 + 0.01 * i for k in range(1, 31)} for i, m in enumerate(METHODS)}
    apath = tmp_path / "analysis.json"
    apath.write_text(json.dumps({"fold_sha": "S", "n_folds": 50, "provenance": {"data_sha256": {"x.csv": "D"}},
                                 "stability_curve_nogueira": curve}))
    out = sb.summarise(str(path), str(apath), 60, feats, n_boot=4)

    # Per-k: SE is the replicate SD (ddof=1), the CI is centred on the CV estimate.
    draws = np.array([nogueira_stability([r[:5] for r in rec["rankings"]["MI"]], feats) for rec in recs])
    s = out["per_k"]["5"]["stability"]["MI"]
    assert np.isclose(s["se"], draws.std(ddof=1))
    assert np.isclose(s["bias"], draws.mean() - 0.5)
    assert np.allclose(s["ci95"], [0.5 - 1.96 * s["se"], 0.5 + 1.96 * s["se"]])

    # Grid contrast: Bonferroni over 10 pairs, z = 2.807.
    z = norm.ppf(1 - 0.05 / 20)
    assert abs(z - 2.807) < 1e-3
    d = out["summaries"]["grid"]["diff"]["MI_minus_RFE"]
    assert np.isclose(d["estimate_cv"], -0.01)
    assert np.allclose(d["ci_bonferroni"], [-0.01 - z * d["se"], -0.01 + z * d["se"]])
    assert d["robust_bonferroni"] == (d["ci_bonferroni"][0] > 0 or d["ci_bonferroni"][1] < 0)
