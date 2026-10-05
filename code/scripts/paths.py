"""Canonical directory resolution for every script in this deposit.

All paths are derived from this file's own location, so the scripts run
correctly from any working directory. The two environment variables below
override the defaults, which is the supported way to point the pipeline at a
dataset stored outside the repository (the OULAD CSVs are ~440 MB and are not
redistributed here -- see dataset/README.md).

    FS_DATA_DIR     raw CSVs (studentInfo.csv, studentVle.csv, ...)
    FS_RESULTS_DIR  JSON/JSONL analysis artifacts written by the scripts

Layout assumed by the defaults:

    <repo root>/
      dataset/          <- FS_DATA_DIR
      code/
        scripts/        <- this file
        results/        <- FS_RESULTS_DIR
"""
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPTS_DIR)
ROOT = os.path.dirname(CODE_DIR)

DATA_DIR = os.environ.get("FS_DATA_DIR", os.path.join(ROOT, "dataset"))
RESULTS_DIR = os.environ.get("FS_RESULTS_DIR", os.path.join(CODE_DIR, "results"))


def require_data():
    """Fail early, with an actionable message, when the raw CSVs are absent.

    Without this the scripts die several seconds later inside pandas with a
    bare FileNotFoundError that does not say the data must be downloaded.
    """
    probe = os.path.join(DATA_DIR, "studentInfo.csv")
    if not os.path.exists(probe):
        raise SystemExit(
            f"OULAD CSVs not found in {DATA_DIR}\n"
            "Download them and unpack them there (see dataset/README.md), or set\n"
            "FS_DATA_DIR to wherever they already live."
        )


os.makedirs(RESULTS_DIR, exist_ok=True)
