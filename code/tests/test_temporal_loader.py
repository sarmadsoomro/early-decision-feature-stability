"""Fails if load_oulad_at leaks data past the cutoff or across a student's courses."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from temporal_design import load_oulad_at, split_dev_holdout  # noqa: E402


def _write(d, name, rows, cols):
    pd.DataFrame(rows, columns=cols).to_csv(os.path.join(d, name), index=False)


def test_cutoff_and_three_key_join(tmp_path):
    k = ["code_module", "code_presentation", "id_student"]
    # Student 1 takes AAA and BBB; student 2 withdraws on day 10.
    _write(tmp_path, "studentInfo.csv", [
        ("AAA", "2013J", 1, "M", "R", "HE", "10-20", "0-35", 0, 60, "N", "Pass"),
        ("BBB", "2013J", 1, "M", "R", "HE", "10-20", "0-35", 0, 60, "N", "Fail"),
        ("AAA", "2013J", 2, "F", "R", "HE", "?", "0-35", 0, 60, "N", "Withdrawn"),
        ("AAA", "2013J", 3, "F", "R", "HE", "?", "0-35", 0, 60, "N", "Withdrawn"),  # undated withdrawal
        ("AAA", "2013J", 4, "F", "R", "HE", "?", "0-35", 0, 60, "N", "Pass"),  # registers on day 40
    ], k + ["gender", "region", "highest_education", "imd_band", "age_band",
            "num_of_prev_attempts", "studied_credits", "disability", "final_result"])
    _write(tmp_path, "studentRegistration.csv",
           [("AAA", "2013J", 1, -10, "?"), ("BBB", "2013J", 1, -10, "?"), ("AAA", "2013J", 2, -10, 10),
            ("AAA", "2013J", 4, 40, "?")],
           k + ["date_registration", "date_unregistration"])
    _write(tmp_path, "vle.csv", [(100, "AAA", "2013J", "quiz", "?", "?"), (200, "BBB", "2013J", "quiz", "?", "?")],
           ["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"])
    _write(tmp_path, "studentVle.csv", [
        ("AAA", "2013J", 1, 100, 5, 3),     # counted
        ("AAA", "2013J", 1, 100, 31, 50),   # after cutoff
        ("BBB", "2013J", 1, 200, 5, 7),     # other course: must not reach AAA row
    ], k + ["id_site", "date", "sum_click"])
    _write(tmp_path, "assessments.csv", [("AAA", "2013J", 11, "TMA", 20, 10), ("AAA", "2013J", 12, "TMA", 40, 10),
                                         ("BBB", "2013J", 21, "TMA", 20, 10)],
           ["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"])
    _write(tmp_path, "studentAssessment.csv", [(11, 1, 18, 0, 80), (12, 1, 25, 0, 10), (21, 1, 19, 0, 30)],
           ["id_assessment", "id_student", "date_submitted", "is_banked", "score"])

    X, y, meta, used = load_oulad_at(30, str(tmp_path))

    # dated withdrawal (2) and post-cutoff registration (4) dropped; undated withdrawal (3) kept as label 0
    assert len(X) == 3 and list(meta["id_student"]) == [1, 1, 3]
    assert used["n_undated_withdrawn_kept"] == 1
    aaa = X.iloc[0]
    assert aaa["vle_quiz"] == 3                               # day-31 clicks and BBB clicks excluded
    assert aaa["num_submissions"] == 1 and aaa["avg_score"] == 80  # deadline-40 TMA excluded though submitted day 25
    assert X.iloc[1]["avg_score"] == 30 and list(y) == [1, 0, 0]
    assert used["max_vle_date"] <= 30 and used["max_assess_date"] <= 30


def test_holdout_excludes_2014j_students_from_dev():
    meta = pd.DataFrame({"id_student": [1, 1, 2, 3, 3],
                         "code_module": ["A"] * 5,
                         "code_presentation": ["2013J", "2014J", "2013J", "2014B", "2013B"]})
    dev, te = split_dev_holdout(meta)
    assert list(te) == [False, True, False, False, False]
    assert list(dev) == [False, False, True, True, True]  # student 1's 2013J row leaves dev


def test_tma_lag_delays_tutor_marked_scores(tmp_path):
    k = ["code_module", "code_presentation", "id_student"]
    _write(tmp_path, "studentInfo.csv", [("AAA", "2013J", 1, "M", "R", "HE", "10-20", "0-35", 0, 60, "N", "Pass")],
           k + ["gender", "region", "highest_education", "imd_band", "age_band",
                "num_of_prev_attempts", "studied_credits", "disability", "final_result"])
    _write(tmp_path, "studentRegistration.csv", [("AAA", "2013J", 1, -10, "?")],
           k + ["date_registration", "date_unregistration"])
    _write(tmp_path, "vle.csv", [(100, "AAA", "2013J", "quiz", "?", "?")],
           ["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"])
    _write(tmp_path, "studentVle.csv", [("AAA", "2013J", 1, 100, 5, 3)], k + ["id_site", "date", "sum_click"])
    _write(tmp_path, "assessments.csv", [("AAA", "2013J", 11, "TMA", 20, 10), ("AAA", "2013J", 12, "CMA", 20, 10)],
           ["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"])
    _write(tmp_path, "studentAssessment.csv", [(11, 1, 18, 0, 80), (12, 1, 19, 0, 40)],
           ["id_assessment", "id_student", "date_submitted", "is_banked", "score"])
    X0 = load_oulad_at(30, str(tmp_path))[0]
    import pytest
    with pytest.raises(AssertionError):
        load_oulad_at(30, str(tmp_path), tma_lag=-7)
    X14 = load_oulad_at(30, str(tmp_path), tma_lag=14)[0]
    assert X0.iloc[0]["num_submissions"] == 2                                    # both by the deadline
    assert X14.iloc[0]["num_submissions"] == 1 and X14.iloc[0]["avg_score"] == 40  # TMA (20+14 > 30) waits; CMA counts


def test_tma_lag_does_not_delay_banked_scores(tmp_path):
    test_tma_lag_delays_tutor_marked_scores(tmp_path)  # writes the toy files
    _write(tmp_path, "studentAssessment.csv", [(11, 1, -1, 1, 80), (12, 1, 19, 0, 40)],
           ["id_assessment", "id_student", "date_submitted", "is_banked", "score"])
    X14 = load_oulad_at(30, str(tmp_path), tma_lag=14)[0]
    assert X14.iloc[0]["num_submissions"] == 2  # banked TMA known from an earlier presentation
