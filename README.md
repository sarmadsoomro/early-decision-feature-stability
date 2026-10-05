# Feature-ranking stability at early decision points: analysis code

Analysis code and stored outputs for:

> **How Stable Are Feature Rankings at Early Decision Points? A Comparison of
> Five Feature-Selection Methods with Student-Grouped Cross-Validation and a
> Later-Presentation Holdout**
> Muhammad Bilal and Sarmad Soomro. Submitted to the *International Journal of
> Data Science and Analytics*.

Code is MIT licensed (`LICENSE`). The datasets are not ours and carry their own
terms; see `dataset/README.md`.

## What the study does

The study measures how reproducible the top-k feature sets of five
feature-selection methods are when student-outcome models are built at a fixed
decision point. The methods are an ANOVA F-test, mutual information (MI),
L1-regularised logistic regression (LASSO, ranked by order of entry on the
regularisation path), recursive feature elimination (RFE) with XGBoost
importances, and mean absolute TreeSHAP values. Each method returns a full
ranking, and k runs over the grid 5, 10, 15, 20, 25 and 30, fixed before any run.

Main data: the Open University Learning Analytics Dataset (OULAD).

- **Decision points.** Days 30, 60 and 90 of each presentation. Features use
  only data dated on or before the cutoff. Registrations withdrawn by the
  cutoff, or not yet registered by it, are dropped.
- **Unit of analysis.** One registration (student, module, presentation). VLE
  and assessment data are joined on all three keys.
- **Splits.** Presentation 2014J is the holdout, and students who appear in it
  are removed from the development data (2013B, 2013J, 2014B). Within the
  development data: 10 repeats of 5-fold `StratifiedGroupKFold` grouped by
  student, with a grouped inner 3-fold CV for tuning.
- **Stability.** Nogueira index per repeat, averaged over repeats. Standard
  errors come from 200 half-samples of students (one 5-fold grouped split per
  half-sample).
- **Agreement between presentations.** In each of 20 repeats, two disjoint sets
  of students are drawn from each of the four presentations, and every method
  ranks every set. Agreement within a presentation is compared with agreement
  between presentations.
- **Accuracy.** XGBoost, random forest and scaled logistic regression are tuned
  once per fold on all features and refitted on each method's top k. Pairs are
  compared with the Nadeau–Bengio corrected t-test and Benjamini–Hochberg
  adjustment. On the 2014J holdout, the cost of a selection is AUC(top k) minus
  AUC(all features), with student-cluster bootstrap intervals.
- **Sensitivity.** Tutor-marked scores counted only 14 days after the deadline
  (`--tma-lag 14`), at days 30 and 60.

xAPI-Edu-Data is analysed with the same loop as an end-of-term contrast. It has
no timestamps and one row per student, so folds are ungrouped and there is no
decision point.

## Quick start

### Environment

The stored outputs were produced with Python 3.11.9 on macOS (Apple silicon).
`requirements.txt` pins the library versions recorded in each output's
`provenance.versions`.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest            # for the tests only
```

### Data

The raw data are not redistributed. Download both datasets as described in
`dataset/README.md` and unpack the CSVs directly into `dataset/`, or set
`FS_DATA_DIR` to another directory. Keep all seven OULAD tables: the scripts
read six of them and record the SHA-256 of every `.csv` file in the data
directory.

### Tests

```bash
python3 -m pytest code/tests
```

The four test files use small synthetic frames and need no data. They cover
the cutoff and three-key join, the 2014J exclusion, the marking-lag option, the
Nogueira formula, the statistical helpers, tie handling in the rankers, and the
disjoint-set construction of the cross-presentation analysis.

### Regenerate the tables and figures from the stored outputs

```bash
cd code/scripts
python3 make_figures.py              # writes to code/results/temporal_design/paper/
python3 make_figures.py --out DIR    # or to another directory
```

This reads only the JSON and JSONL outputs in `code/results/temporal_design/`,
fits nothing, needs no raw data and finishes in seconds. It writes two PDF
figures and the LaTeX table bodies (rows only; captions and column headers are
in the manuscript). The `.tex` files it writes are identical to the stored ones
in `paper/`. `RESULTS-MAP.md` lists which output each table and figure comes
from.

### Check that this code reproduces the stored outputs

```bash
cd code/scripts
python3 reproduce_fold.py --dataset xapi                # about 1 min
python3 reproduce_fold.py --dataset oulad --cutoff 30   # about 2 min; also 60, 90
```

`reproduce_fold.py` needs the raw data. It rebuilds the data as
`temporal_design.py` does, re-fits the first stored CV fold (repeat 0, fold 0)
and compares every method's ranking and every evaluator's ROC-AUC with the
stored fold record. It exits with status 1 if any ranking differs or any AUC
differs by more than 1e-9. See [Provenance](#provenance) for the results.

### Full re-run

Every command below needs the raw data and is run from `code/scripts/`.
Runtimes are from the machine that produced the stored outputs and are rough
guides. `FS_N_JOBS` sets the number of parallel jobs (default: all cores).

```bash
# 1. Grouped CV and 2014J holdout, about 2 h per cutoff
python3 temporal_design.py --dataset oulad --cutoff 30
python3 temporal_design.py --dataset oulad --cutoff 60
python3 temporal_design.py --dataset oulad --cutoff 90
python3 temporal_design.py --dataset xapi                # about 16 min

# 2. Half-sampling bootstrap, 200 replicates. Needs the analysis JSON that
#    step 1 wrote for the same setting. About 2-3 h per OULAD setting.
python3 stability_bootstrap.py --dataset oulad --cutoff 30
python3 stability_bootstrap.py --dataset oulad --cutoff 60
python3 stability_bootstrap.py --dataset oulad --cutoff 90
python3 stability_bootstrap.py --dataset xapi            # about 10 min

# 3. Agreement within and between presentations, 20 repeats,
#    about 10 min per command
python3 cross_cohort.py --cutoff 30                      # all modules, 1,650 students per set
python3 cross_cohort.py --cutoff 30 --modules common     # BBB/DDD/FFF quotas, 1,370 students per set
python3 cross_cohort.py --cutoff 30 --n-students 1370    # all modules, 1,370 students per set
# ...and the same three commands with --cutoff 60 and --cutoff 90

# 4. Marking-lag sensitivity, days 30 and 60. The bootstrap reads the CV
#    analysis from its own --output-dir, so both go to the same directory.
#    The stored lag-14 bootstraps took about 4.5 h each.
python3 temporal_design.py     --dataset oulad --cutoff 30 --tma-lag 14 --output-dir ../results/temporal_design/lag14_boot
python3 stability_bootstrap.py --dataset oulad --cutoff 30 --tma-lag 14 --output-dir ../results/temporal_design/lag14_boot
# ...and the same two commands with --cutoff 60

# 5. Tables and figures
python3 make_figures.py
```

Outputs go to `code/results/temporal_design/` unless `--output-dir` is given.
Runs append one record per fold, replicate or repeat to a `.jsonl` file as they
go, so an interrupted run resumes where it stopped when restarted with the same
command. To re-run into the folders that hold the stored outputs, move those
outputs away first (see [Provenance](#provenance)).

Other options: `--n-repeats` (`temporal_design.py`, default 10), `--n-boot`
(`stability_bootstrap.py`, default 200), `--n-rep` (`cross_cohort.py`, default
20), `--analyze-only` (`temporal_design.py`: rebuild the analysis JSON from the
existing fold records) and `--summarise-only` (`stability_bootstrap.py`: rebuild
the summary from the existing replicates). Both of the last two still load the
raw data. `stability_bootstrap.py` summarises only against a 50-fold CV
analysis, so keep `--n-repeats` at 10 when a bootstrap will follow.

## Provenance

Each stored output records, in its `provenance` block:

- `provenance.code_sha`: the git tree hash of `code/scripts/` in the authors'
  working repository when the output was written. Per-record `.jsonl` files
  carry the same hash as `sha`; the analysis and bootstrap summaries repeat it
  as `fold_sha` and `boot_sha`.
- `provenance.git_sha`: a commit in that working repository.

Neither hash resolves in this repository, which was published as a fresh
history containing the final code. The scripts here differ from those working
versions only in comments, docstrings, two output label strings, and the move
of eight helper functions into `temporal_design.py`.

The stored outputs carry three `code_sha` values:

| `code_sha` | Outputs |
|---|---|
| `f9c197105763537e3440e35c28bd0973a645d82d` | CV, holdout and bootstrap at days 30/60/90 and xAPI; the lag-14 CV and holdout in `temporal_design/` |
| `0b339de1ea1c3453d88de8c18d3fe28fc0e5e007` | `oulad_day*_cross_cohort*` |
| `318d9c08304b03f41970381093a3c44d1e540557` | everything in `lag14_boot/` |

`reproduce_fold.py` re-fits the first stored fold with the code in this
repository. Results:

- xAPI: 5/5 rankings identical; 93 ROC-AUC values compared, maximum absolute
  difference 0.
- OULAD days 30, 60 and 90: 5/5 rankings identical at each cutoff; 93 ROC-AUC values each, maximum absolute difference 0.

New runs stamp the tree hash of `code/scripts/` in this repository, so they
must be run from a git clone with the scripts committed. Outside a git
repository the hashes are recorded as empty strings. The scripts refuse to mix
records from different code versions:

- `temporal_design.py` will not resume fold records, or reuse a holdout file,
  written by code with a different hash. With `--analyze-only` it checks only
  that all fold records share one hash and, for OULAD, that the holdout has
  the same hash.
- `stability_bootstrap.py` will not start replicates unless the CV analysis in
  its output directory was written by the current code, and will not resume
  replicates written by other code. When summarising, including with
  `--summarise-only`, it requires all replicates to share the analysis's
  `fold_sha` and the data checksums to match those recorded by the CV run.
- `cross_cohort.py` has no summary-only mode. It will not resume or summarise
  stored repeats written by code with a different hash.

In practice, `make_figures.py`, `reproduce_fold.py`,
`temporal_design.py --analyze-only` and `stability_bootstrap.py
--summarise-only` work with the stored outputs. A new CV, bootstrap or
cross-presentation run needs an empty output directory (or one holding only
outputs written by this repository's code). Each run except `--analyze-only`
and `--summarise-only` also refuses to start if `code/scripts/` has uncommitted
changes, and holds a `.lock` file so that two processes cannot write the same
output.

## Layout

```
├── README.md              this file
├── RESULTS-MAP.md         manuscript table/figure → script → command → output
├── CITATION.cff
├── LICENSE                MIT
├── requirements.txt       exact pins
├── dataset/               raw CSVs go here (not committed); see its README
└── code/
    ├── scripts/
    │   ├── temporal_design.py      data construction, grouped CV, five rankers, evaluators, holdout
    │   ├── stability_bootstrap.py  half-sampling bootstrap for the stability index
    │   ├── cross_cohort.py         agreement within and between presentations
    │   ├── make_figures.py         figures and table bodies from the stored outputs
    │   ├── reproduce_fold.py       re-fits the first stored fold and compares it
    │   └── paths.py                directory resolution
    ├── tests/             test_{temporal_loader,nogueira,stats,cross_cohort}.py
    └── results/
        ├── DATA-DICTIONARY.md      fields of the stored outputs
        └── temporal_design/        stored outputs and run logs
            ├── lag14_boot/         lag-14 CV and bootstrap
            └── paper/              output of make_figures.py
```

`code/scripts/paths.py` resolves every path from the script's own location, so
the scripts run from any working directory. `FS_DATA_DIR` and `FS_RESULTS_DIR`
override the data and results directories. The `.log` files next to the outputs
are the console output of each run.

## Limitations

- **One main dataset.** OULAD covers seven modules of one institution over two
  years. The three decision points are views of the same students, and four
  presentations give only six dependent pairs.
- **Not bit-exact across machines.** The runs are seeded, but XGBoost, the k-NN
  MI estimator and TreeSHAP can differ numerically across CPU architectures,
  BLAS builds and `FS_N_JOBS` settings. Expect small differences in stability
  values and AUCs on other machines.
- **Label-free steps use the whole development data.** One-hot encoding,
  removal of constant columns and the MI discrete/continuous split are decided
  once on the development data, not per fold.
- **One configuration per method.** RFE uses an untuned XGBoost model and
  removes one feature per step. LASSO uses the liblinear solver, which also
  penalises the intercept.
- **Heuristic standard errors.** The half-sampling standard error is not proven
  valid for this statistic. Replicates use half the students, so their mean is
  biased downwards; the bias is stored in the bootstrap summary.
- **Descriptive cross-presentation results.** The cross-presentation outputs
  describe resampling noise within one sample of each presentation. They carry
  no intervals or tests.
- **Marking dates are not recorded in OULAD.** By default a score counts once
  both the submission and the deadline fall on or before the cutoff;
  `--tma-lag` delays tutor-marked scores only. The lag-14 runs also write
  holdout files, but the manuscript reports only the lag-14 CV and bootstrap.
- **Outcome.** The binary outcome merges failure with later withdrawal.
  Registrations recorded as withdrawn without a date (93, 87 of them in 2014J)
  are kept as non-pass; `auc_excl_undated_withdrawn` in each holdout JSON gives
  the holdout AUCs without them.
- **Unreported outputs.** The interpretability measures (`faithfulness`,
  `concentration_gini`, `redundancy`) and the per-module holdout AUCs are stored
  but not reported in the manuscript.
- **Holdout probabilities.** `*_holdout_probs.npz` stores `code_module` as an
  object array, so load it with `numpy.load(path, allow_pickle=True)`.

## Authors

| | |
|---|---|
| **Muhammad Bilal** | School of Information Technology, Whitecliffe College, Christchurch, New Zealand · [0009-0004-8200-7835](https://orcid.org/0009-0004-8200-7835) · 20251211@mywhitecliffe.com |
| **Sarmad Soomro** (corresponding) | School of Information Technology, Whitecliffe College, Christchurch, New Zealand · [0000-0001-6677-1332](https://orcid.org/0000-0001-6677-1332) · sarmads@whitecliffe.ac.nz |

Contributions follow the manuscript's CRediT statement: Bilal: conceptualization,
methodology, software, formal analysis, investigation, data curation, writing
(original draft); Soomro: validation, supervision, writing (review and editing).

## Contact

Sarmad Soomro (corresponding author), School of Information Technology,
Whitecliffe College, Christchurch, New Zealand. sarmads@whitecliffe.ac.nz.
ORCID [0000-0001-6677-1332](https://orcid.org/0000-0001-6677-1332).

## Citation

Cite the article for the findings and this repository for the code;
`CITATION.cff` has the details.

Repository: <https://github.com/sarmadsoomro/early-decision-feature-stability>
