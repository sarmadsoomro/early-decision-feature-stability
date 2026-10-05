# Results map

This file maps each table and figure of the manuscript and of its
Supplementary Material to the script, command and output file that produce it.

Paths are relative to `code/results/temporal_design/` unless stated otherwise,
and commands are run from `code/scripts/`. `N` is the cutoff day (30, 60 or 90).

Every table and figure is written by `make_figures.py`, which reads only the
stored outputs and needs no raw data:

```bash
python3 make_figures.py            # writes to code/results/temporal_design/paper/
```

The `.tex` files hold the table rows only; captions and column headers are in
the manuscript source. The outputs that `make_figures.py` reads come from the
commands in the last column, which need the raw data. `README.md` gives the
runtimes.

## Main manuscript

| Item | `paper/` file | Inputs (fields) | Produced by |
|---|---|---|---|
| Table 1: registrations, features and pass rates | `tab-data.tex` | `oulad_dayN_holdout.json` (`n_dev`, `n_holdout`, `dev_pass_rate`, `holdout_pass_rate`); `oulad_dayN_analysis.json` (`n_features`, `dates_used.max_assess_date`) | `temporal_design.py --dataset oulad --cutoff N` |
| Table 2: CV stability, grid mean and k ≤ p/3 mean | `tab-stability.tex` | `oulad_dayN_analysis.json`, `xapi_endofterm_analysis.json` (`k_summaries_nogueira.grid.mean`, `k_summaries_nogueira.k_le_p3.mean`) | `temporal_design.py --dataset oulad --cutoff N`; `temporal_design.py --dataset xapi` |
| Fig. 1: CV stability against k | `fig-stability-k.pdf` | same files (`stability_curve_nogueira`, `n_features`) | as Table 2 |
| Table 3: RFE minus each other method, Bonferroni intervals | `tab-rfe-contrasts.tex` | `oulad_dayN_bootstrap_summary.json`, `xapi_endofterm_bootstrap_summary.json` (`summaries.grid.diff`: `estimate_cv`, `ci_bonferroni`, `robust_bonferroni`) | `stability_bootstrap.py --dataset oulad --cutoff N`; `stability_bootstrap.py --dataset xapi` |
| Table 4: agreement within and between presentations | `tab-cross.tex` | `oulad_dayN_cross_cohort.json` (`summaries.grid.within_dev_dev`, `loss_dev_dev`, `loss_dev_2014J`; `loss_positive.grid.<method>.pairs_with_mean_loss_gt0`) | `cross_cohort.py --cutoff N` |
| Table 5: all-feature AUC, CV and holdout | `tab-holdout-all.tex` | `oulad_dayN_analysis.json` (`baseline_all.<ev>.roc_auc.mean`); `oulad_dayN_holdout.json` (`baseline_all.<ev>.roc_auc`) | `temporal_design.py --dataset oulad --cutoff N` |
| Fig. 2: holdout cost of keeping k features | `fig-holdout-cost.pdf` | `oulad_dayN_holdout.json` (`methods.<method>.per_k.<k>.<ev>.roc_auc`, `baseline_all`, `bootstrap.<ev>_k<k>.diff_vs_all_ci`) | `temporal_design.py --dataset oulad --cutoff N` |

## Supplementary Material

| Item | `paper/` file | Inputs (fields) | Produced by |
|---|---|---|---|
| Table S1: stability contrasts whose interval excludes zero | `tabS-contrasts.tex` | `oulad_dayN_bootstrap_summary.json`, `xapi_endofterm_bootstrap_summary.json` (`summaries.grid.diff`, `summaries.k_le_p3.diff`: `estimate_cv`, `robust`, `robust_bonferroni`) | as Table 3 |
| Table S2: loss per presentation pair | `tabS-cross-pairs.tex` | `oulad_dayN_cross_cohort.json` and `oulad_dayN_cross_cohort_common.json` (`summaries.grid.loss_<a>\|<b>`, `loss_positive.grid.<method>.share_of_repeats`) | `cross_cohort.py --cutoff N` and `cross_cohort.py --cutoff N --modules common` |
| Table S3: agreement with the module mix held fixed | `tabS-cross-common.tex` | `oulad_dayN_cross_cohort_common.json` (fields as Table 4) | `cross_cohort.py --cutoff N --modules common` |
| Table S4(a): grid-mean CV stability, lag 0 and lag 14 | `tabS-lag.tex` | `oulad_dayN_analysis.json` and `oulad_dayN_tmalag14_analysis.json`, N = 30, 60 (`k_summaries_nogueira.grid.mean`) | `temporal_design.py --dataset oulad --cutoff N`, with and without `--tma-lag 14` |
| Table S4(b): RFE contrasts, lag 0 and lag 14 | `tabS-lag-contrasts.tex` | `oulad_dayN_bootstrap_summary.json` and `lag14_boot/oulad_dayN_tmalag14_bootstrap_summary.json`, N = 30, 60 (fields as Table 3) | lag 0: as Table 3. Lag 14: `temporal_design.py --dataset oulad --cutoff N --tma-lag 14 --output-dir ../results/temporal_design/lag14_boot`, then `stability_bootstrap.py --dataset oulad --cutoff N --tma-lag 14 --output-dir ../results/temporal_design/lag14_boot` |
| Table S5: CV ROC-AUC (XGBoost) and BH-significant pairs | `tabS-cv-auc.tex` | `oulad_dayN_analysis.json` (`per_k.<k>.roc_auc.<method>.xgb.mean`; `per_k.<k>.pairwise_roc_auc.<ev>.<a>_vs_<b>.p_adjusted_bh`) | `temporal_design.py --dataset oulad --cutoff N` |
| Table S6: grid-mean stability, CV estimate / bias / SE | `tabS-bias.tex` | `oulad_dayN_bootstrap_summary.json`, `xapi_endofterm_bootstrap_summary.json` (`summaries.grid.stability.<method>`: `estimate_cv`, `bias`, `se`) | as Table 3 |

The lag-14 CV is stored twice. Table S4(a) reads the copy in
`temporal_design/`. `stability_bootstrap.py` summarises only against a CV
analysis with the same code hash as its replicates, so the lag-14 CV was run
again into `lag14_boot/` before the bootstrap. The two copies have identical
`per_k` and `k_summaries_nogueira` values.

## Values quoted in the text without a table

| Value | Source |
|---|---|
| All-modules analysis at 1,370 students per set | `oulad_dayN_cross_cohort_n1370.json`, from `cross_cohort.py --cutoff N --n-students 1370` |
| Holdout AUC without the registrations withdrawn without a date (Supplementary Section S1) | `oulad_dayN_holdout.json`, `auc_excl_undated_withdrawn` |
| Registrations removed from development because the student appears in 2014J | `oulad_dayN_holdout.json`, `n_dev_dropped_overlap` |
| Undated withdrawals kept, registrations without a registration date | `oulad_dayN_analysis.json`, `dates_used` |
| Features removed as constant | `oulad_dayN_analysis.json`, `dropped_dev_constant` |
| xAPI duplicate records and identical columns removed | `xapi_endofterm_analysis.json`, `dropped_duplicates`, `dropped_identical_cols` |
| Tuned evaluator hyperparameters per fold and on the holdout (Supplementary Section S2) | `oulad_dayN_fold_checkpoints.jsonl` and `oulad_dayN_holdout.json`, `params` |
| Holdout cost of individual methods at individual k | `oulad_dayN_holdout.json`, `methods.<method>.per_k.<k>.<ev>.roc_auc` minus `baseline_all.<ev>.roc_auc` |
| LASSO path shortfalls | `ranking_info.LASSO` in the analysis JSON, `lasso_folds_short` in the bootstrap summary, `lasso_path` in the cross-presentation JSON |

## Other files

| File | Contents |
|---|---|
| `*_fold_checkpoints.jsonl` | One record per CV fold: rankings, metrics and tuned parameters. Every CV statistic is computed from these. |
| `*_bootstrap.jsonl` | One record per half-sample replicate: the five rankings on each of the five folds. |
| `oulad_dayN_cross_cohort*.jsonl` | One record per repeat: rankings and composition of every drawn set. |
| `oulad_dayN_holdout_probs.npz` | Holdout predicted probabilities for every method, k and evaluator. |
| `run_*.log`, `boot_*.log`, `cross_*.log`, `lag14_boot/*.log` | Console output of each run, with per-fold, per-replicate or per-repeat timings. |
| `oulad_dayN_tmalag14_holdout.json`, `oulad_dayN_tmalag14_holdout_probs.npz` and the lag-14 files in `lag14_boot/` other than the bootstrap summaries | Written by the lag-14 runs; not reported in the manuscript. |

`code/results/DATA-DICTIONARY.md` documents the fields of every output file.
