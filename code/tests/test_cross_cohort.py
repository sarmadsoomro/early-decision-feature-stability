import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cross_cohort as cc  # noqa: E402
import temporal_design  # noqa: E402
from temporal_design import METHODS  # noqa: E402


def _recs(sets):
    return [{"rankings": {s: {m: v for m in METHODS} for s, v in sets.items()}}]


def test_agreement_matches_kuncheva_and_loss_is_within_minus_between():
    # Nogueira with M = 2 equals Kuncheva (r*d - k^2) / (k*(d - k)); d = 60, k = 5.
    f = [f"f{i}" for i in range(60)]
    overlap2 = f[3:5] + f[::-1][:3]                            # shares 2 of f[:5]
    overlap2 += [x for x in f if x not in overlap2]
    sets = {f"S{i}_{p}": f for i in (1, 2) for p in cc.ORDER}  # within = 1 everywhere
    sets["S1_2013J"] = sets["S2_2013J"] = f[::-1]              # 2013J disjoint from the rest
    sets["S1_2014B"] = sets["S2_2014B"] = overlap2
    out = cc.summarise(_recs(sets), f, len(f))
    k5 = out["per_k"]["5"]
    assert abs(k5["between_2013B|2013J"]["ANOVA"]["mean"] - (-25 / 275)) < 1e-12
    assert abs(k5["between_2013B|2014B"]["ANOVA"]["mean"] - 95 / 275) < 1e-12
    assert k5["within_2013B"]["ANOVA"]["mean"] == 1.0
    assert abs(k5["loss_2013B|2014B"]["ANOVA"]["mean"] - (1 - 95 / 275)) < 1e-12
    # group fields: within = mean of (w_a + w_b)/2 over pairs; loss = within - between
    btw = (-25 + 95 + 155) / 275 / 3                           # 2013J|2014B shares 3 of 5
    assert k5["within_dev_dev"]["ANOVA"]["mean"] == 1.0
    assert abs(k5["between_dev_dev"]["ANOVA"]["mean"] - btw) < 1e-12
    assert abs(k5["loss_dev_dev"]["ANOVA"]["mean"] - (1 - btw)) < 1e-12
    pos = out["loss_positive"]["grid"]["ANOVA"]
    assert pos["share_of_repeats"]["2013B|2014J"] == 0.0      # identical rankings: no loss
    assert pos["share_of_repeats"]["2013B|2013J"] == 1.0
    assert pos["pairs_with_mean_loss_gt0"] == 5


def test_groups_hold_the_intended_pairs():
    g = cc.GROUPS
    assert set(g["dev_dev"]) | set(g["dev_2014J"]) == set(cc.PAIRS)
    assert all("2014J" in x for x in g["dev_2014J"]) and not any("2014J" in x for x in g["dev_dev"])
    assert set(g["same_season"]) | set(g["cross_season"]) == set(cc.PAIRS)
    assert all(a[-1] == b[-1] for a, b in (x.split("|") for x in g["same_season"]))
    gap = lambda x: abs(cc.ORDER.index(x.split("|")[1]) - cc.ORDER.index(x.split("|")[0]))
    for name, n in [("gap1_dev", 1), ("gap1_2014J", 1), ("gap2_dev", 2), ("gap2_2014J", 2), ("gap3", 3)]:
        assert all(gap(x) == n for x in g[name])


def _toy(n=60):
    rng = np.random.default_rng(0)
    ids, pres, mods = [], [], []
    sid = 0
    for p in cc.ORDER:
        for m in cc.COMMON_QUOTA:
            for _ in range(n):
                ids.append(sid); pres.append(p); mods.append(m); sid += 1
    extra = pd.DataFrame({"id_student": [1, 2, 0], "code_presentation": ["2013B", "2013B", "2014J"],
                          "code_module": ["DDD", "AAA", "BBB"]})
    # student 1: BBB + DDD in 2013B (two common modules); student 2: BBB + AAA (a non-common row);
    # student 0: 2013B + 2014J (two presentations, must leave the pool)
    meta = pd.concat([pd.DataFrame({"id_student": ids, "code_presentation": pres, "code_module": mods}), extra],
                     ignore_index=True)
    X = pd.DataFrame(rng.normal(size=(len(meta), 6)), columns=[f"x{i}" for i in range(6)])
    X["no_submission"] = 0
    y = pd.Series(np.arange(len(meta)) % 2)
    return X, y, meta


def _raw(meta, extra_rows=()):
    return pd.concat([meta[["id_student", "code_presentation"]],
                      pd.DataFrame(list(extra_rows), columns=["id_student", "code_presentation"])])


def test_pool_uses_raw_registrations():
    _, _, meta = _toy()
    # student 5's 2013J registration was dropped by the cutoff filter but is in the raw file
    pool = cc.single_presentation_pool(meta, _raw(meta, [(5, "2013J")]))
    assert not pool[(meta["id_student"] == 0).to_numpy()].any()
    assert not pool[(meta["id_student"] == 5).to_numpy()].any()
    assert pool[(meta["id_student"] == 6).to_numpy()].all()


def test_sets_disjoint_whole_students_and_quota(monkeypatch):
    X, y, meta = _toy()
    pool = cc.single_presentation_pool(meta, _raw(meta))
    monkeypatch.setattr(cc, "N_STUDENTS", 50)
    monkeypatch.setattr(cc, "COMMON_QUOTA", {"BBB": 25, "DDD": 25, "FFF": 25})
    for modules in ("all", "common"):
        for seed in range(5):
            masks = cc.draw_sets(meta, pool, modules, np.random.default_rng(seed))
            ids = {s: set(meta.loc[m, "id_student"]) for s, m in masks.items()}
            assert all(not ids[a] & ids[b] for a, b in combinations(ids, 2))
            for s, m in masks.items():
                assert meta.loc[m, "code_presentation"].nunique() == 1
                rows = meta["id_student"].isin(ids[s]).to_numpy() & pool
                if modules == "all":
                    assert (rows == m).all() and len(ids[s]) == 50
                else:
                    common = meta["code_module"].isin(list(cc.COMMON_QUOTA)).to_numpy()
                    assert (rows & common == m).all()          # both common rows of student 1, never AAA
                    assert len(ids[s]) == 75                   # 25 students per module, each drawn once
                    assert "AAA" not in set(meta.loc[m, "code_module"])


def test_repeat_ranks_every_set_with_every_method(monkeypatch):
    X, y, meta = _toy()
    monkeypatch.setattr(cc, "N_STUDENTS", 50)
    temporal_design.fix_discrete(X)
    rankings, info, comp = cc.repeat(X, y, meta, cc.single_presentation_pool(meta, _raw(meta)), "all", 0)
    assert set(rankings) == {f"S{i}_{p}" for i in (1, 2) for p in cc.ORDER}
    assert all(sorted(rankings[s][m]) == sorted(X.columns) for s in rankings for m in METHODS)
    assert all(comp[s]["students"] == 50 for s in comp)
