# Field reference for the stored outputs

This file documents the fields of the JSON, JSON Lines and NPZ outputs in
`temporal_design/`, so the outputs can be reused without reading the code.

## Conventions

- `N` is the cutoff day (30, 60 or 90). OULAD files are named
  `oulad_dayN_*`; with a 14-day tutor-marked-assessment lag they are named
  `oulad_dayN_tmalag14_*`. xAPI files are named `xapi_endofterm_*`.
- `<method>` is one of `MI`, `RFE`, `LASSO`, `ANOVA`, `SHAP`.
- `<ev>` is an evaluator: `xgb` (XGBoost), `rf` (random forest), `lr`
  (standardised logistic regression).
- `<k>` is a number of features kept, stored as a string key. The pre-set grid
  is 5, 10, 15, 20, 25, 30.
- A *ranking* is a list of feature names, best first, covering every feature.
  The top-k set of a method is the first k entries.
- Stability values are Nogueira indices using the unbiased per-feature variance
  M/(M−1)·p̂(1−p̂). They equal 1 for identical selections and have expectation 0
  for random ones, so they can be negative.
- `random_state` is 42 throughout; repeat r uses seed r for its outer split.

## `provenance` (in every summary JSON)

| Field | Meaning |
|---|---|
| `git_sha` | A commit in the authors' working repository when the output was written. It does not resolve in this repository. |
| `code_sha` | Git tree hash of `code/scripts/` in the authors' working repository when the output was written. It does not resolve in this repository, which was published as a fresh history containing the final code; see `README.md`, "Provenance". Outputs written by new runs record the tree hash of `code/scripts/` in this repository. |
| `git_dirty` | Whether `code/scripts/` had uncommitted changes. False in every stored output. |
| `versions` | `numpy`, `pandas`, `sklearn`, `xgboost`, `shap` versions. |
| `FS_N_JOBS` | Parallel jobs setting (`-1` means all cores). |
| `data_sha256` | SHA-256 of every `.csv` file in the data directory, keyed by file name. |

## `*_fold_checkpoints.jsonl`

JSON Lines, one record per outer CV fold, 50 records (10 repeats × 5 folds),
appended as the run goes.

| Field | Meaning |
|---|---|
| `fold` | 0–49; `fold = 5 × repeat + i`. |
| `repeat` | 0–9. Stability is computed within each repeat over its five folds. |
| `n_train`, `n_test` | Rows in the training and test parts of this fold. |
| `sha` | `code_sha` of the code that wrote the record. |
| `git_dirty` | Always false. |
| `params.<ev>` | Hyperparameters chosen by the inner grouped 3-fold CV on all features. Shared by every method in this fold. |
| `baseline_all.<ev>` | `roc_auc`, `accuracy`, `f1` of the tuned evaluator with all features on the test fold. Accuracy and F1 use a 0.5 threshold. |
| `methods.<method>.ranking` | The method's full ranking on the training fold. |
| `methods.<method>.info` | Diagnostics of the ranker; see below. |
| `methods.<method>.per_k.<k>.<ev>` | `roc_auc`, `accuracy`, `f1` of the evaluator refitted on the top-k set (columns in the data's own order). |
| `methods.<method>.per_k.<k>.faithfulness` | Normalised deletion score of the XGBoost model on the top-k set: features are replaced by their training mean in the order of that model's own mean absolute SHAP values, and the score is the share of the total confidence drop reached on average along the curve. Higher means the order removes what the model relies on sooner. NaN if the total drop is below 0.01. Compare methods only at equal k. |
| `faithfulness_random` | Mean of the same score over 20 random orders. |
| `faithfulness_gain` | `faithfulness` minus `faithfulness_random`. |
| `deletion_start`, `deletion_end` | Mean confidence in the initially predicted class before any and after all features are replaced. |
| `concentration_gini` | Gini coefficient of the XGBoost model's mean absolute SHAP values over the top-k set. |
| `redundancy` | Mean absolute pairwise correlation among the top-k features on the training fold. |
| `redundancy_const_cols` | Top-k features that are constant on the training fold and therefore left out of `redundancy`. |

The interpretability fields (`faithfulness` to `redundancy_const_cols`) are
stored but not reported in the manuscript.

`methods.<method>.info` by method:

| Method | Field | Meaning |
|---|---|---|
| `MI`, `ANOVA`, `SHAP` | `n_nonzero` | Features with a score above zero. |
| | `max_tie_top` | Largest group of exactly equal scores that reaches into the top 30. Exact ties are broken at random with a seed derived from the training rows. |
| `RFE` | `tied_steps_top` | Elimination steps, once 30 or fewer features remain, at which more than one feature had the minimum importance. |
| `LASSO` | `l1_min_c` | Smallest inverse penalty C at which any coefficient is non-zero. |
| | `n_entered` | Features that entered the L1 path before the sweep stopped. |
| | `need` | Features that had to enter: the smaller of the number of non-constant features and max(30, p/2). |
| | `n_never_enter` | Features that did not enter; they are ranked last, in random order. |
| | `max_tie_group` | Largest group of features entering at the same C after refinement. |
| | `convergence_warnings` | liblinear convergence warnings on this fold. |

## `*_analysis.json`

Summary of the 50 fold records, written by `temporal_design.py` (also by
`--analyze-only`).

| Field | Meaning |
|---|---|
| `dataset` | Output name, e.g. `oulad_day30`. |
| `n_folds`, `n_repeats` | 50 and 10. |
| `k_grid` | The pre-set grid of k. |
| `n_features` | p, the number of features after constant columns are removed. |
| `fold_sha` | `code_sha` shared by all fold records. |
| `cutoff_day`, `tma_lag` | OULAD only: the cutoff and the tutor-marked lag in days. |
| `dates_used` | OULAD only: `max_vle_date` and `max_assess_date` (latest VLE date and latest assessment deadline used), `n_undated_withdrawn_kept`, `n_missing_registration_date_kept`. |
| `dropped_dev_constant` | OULAD only: columns removed because they are constant in the development data. |
| `mi_discrete_cols` | Columns treated as discrete by MI (binary, or integer with at most 20 values), fixed once from the development data. |
| `note`, `dropped_duplicates`, `dropped_identical_cols` | xAPI only: the end-of-term note, duplicate records removed, and columns removed as identical to others. |
| `per_k.<k>.stability.<method>.{Jaccard,Kuncheva,Nogueira}` | `mean` and `std` (sample SD) over the 10 repeats of the per-repeat index. Jaccard and Kuncheva are averaged over the 10 fold pairs of a repeat; Nogueira is computed over the 5 folds at once. |
| `per_k.<k>.roc_auc.<method>.<ev>` | `mean`, `std` over the 50 folds. |
| `per_k.<k>.interpretability.<method>.<measure>` | `mean`, `std` over folds of `faithfulness`, `faithfulness_random`, `faithfulness_gain`, `concentration_gini`, `redundancy`. |
| `per_k.<k>.pairwise_roc_auc.<ev>.<a>_vs_<b>` | Nadeau–Bengio corrected paired t-test on fold AUCs of method a minus method b: `mean_diff`, `sd_diff`, `t_stat`, `df` (9), `p_value`, `se_nb`, `p_adjusted_bh`. If the two methods give identical fold AUCs, only `mean_diff` and `p_value` (null) are stored. |
| `bh_family` | The Benjamini–Hochberg family: per evaluator, 10 pairs × 6 values of k. |
| `baseline_all.<ev>.{roc_auc,accuracy,f1}` | `mean`, `std` over folds with all features. |
| `ranking_info.<method>.<field>` | `mean`, `std`, `min` over folds of each `info` field. |
| `stability_curve_nogueira.<method>.<k>` | Mean over repeats of the per-repeat Nogueira index, for k = 1 to 30. |
| `k_summaries_nogueira.{grid,k_le_p3,all}` | `k` (the values averaged) and `mean.<method>`: the curve averaged over the grid, over k ≤ p/3, and over k ≤ p/2. `grid` is the primary summary. |
| `k_grid_repeat_range.<method>` | Minimum and maximum over repeats of the grid-mean Nogueira index. |
| `provenance` | See above. |

## `oulad_dayN_holdout.json`

Development data (2013B, 2013J, 2014B without students who appear in 2014J)
against the 2014J holdout. Written by `temporal_design.py` for OULAD only.

| Field | Meaning |
|---|---|
| `params.<ev>` | Hyperparameters tuned on all development data. |
| `baseline_all.<ev>` | `roc_auc`, `accuracy`, `f1` on 2014J with all features. |
| `methods.<method>.ranking`, `.info` | Ranking on all development data, and its diagnostics (as in the fold records). |
| `methods.<method>.per_k.<k>` | Holdout metrics per evaluator and the interpretability fields, as in the fold records. |
| `n_dev`, `n_holdout` | Development and holdout rows. |
| `n_dev_dropped_overlap` | Development rows removed because the student appears in 2014J. |
| `dev_pass_rate`, `holdout_pass_rate` | Share of rows with outcome 1. |
| `dev_module_share` | Share of development rows per module. |
| `n_boot` | Student-cluster bootstrap resamples of 2014J (1,000). |
| `bootstrap.<ev>_k<k>.auc_ci.<method>` | 2.5th and 97.5th percentiles of the holdout AUC. |
| `bootstrap.<ev>_k<k>.diff_vs_all_ci.<method>` | Same for AUC(top k) minus AUC(all features). |
| `bootstrap.<ev>_k<k>.pairwise_diff_ci.<a>_vs_<b>` | Same for AUC of method a minus method b. |
| `bootstrap_note` | The intervals are descriptive percentile intervals without multiplicity adjustment. |
| `per_module.<module>` | Holdout `n`, `pass_rate`, XGBoost AUC with all features (`auc_xgb_all`) and with each method's top 15 (`auc_xgb_k15`). |
| `auc_excl_undated_withdrawn` | Holdout AUCs without the registrations recorded as withdrawn without a date: `n_excluded`, then per evaluator `all` and `<method>.<k>`. |
| `provenance` | See above. |

## `oulad_dayN_holdout_probs.npz`

Holdout predictions, one array per key, each with one entry per 2014J row:
`y` (outcome), `id_student`, `code_module`, and one array of predicted
probabilities per `"<method>|<k>|<ev>"`, plus `"ALL|all|<ev>"` for all
features. `code_module` is an object array, so load with
`numpy.load(path, allow_pickle=True)`.

## `*_bootstrap.jsonl`

One record per half-sample replicate, 200 records.

| Field | Meaning |
|---|---|
| `b` | Replicate index 0–199; its seed is 42 + b. |
| `rankings.<method>` | Five rankings, one per training fold of a single 5-fold grouped split of a random half of the development students (drawn without replacement). |
| `lasso_n_entered` | Per fold, `[n_entered, need]` for LASSO. |
| `sha` | `code_sha` of the code that wrote the record. |

## `*_bootstrap_summary.json`

Written by `stability_bootstrap.py` (also by `--summarise-only`).

| Field | Meaning |
|---|---|
| `n_boot` | 200. |
| `scheme` | Half-sampling without replacement (n/2 students). |
| `lasso_folds_short` | Replicate folds in which fewer LASSO features entered than needed. |
| `boot_sha` | `code_sha` shared by all replicates; equal to the analysis `fold_sha`. |
| `provenance` | See above. |
| `summaries.{grid,k_le_p3,all}.role` | `confirmatory` for `grid`; the other two are sensitivity summaries. |
| `summaries.<summary>.k` | The values of k averaged. |
| `summaries.<summary>.stability.<method>` | Interval record (below) for the method's mean stability over these k. |
| `summaries.<summary>.diff.<a>_minus_<b>` | Interval record for method a minus method b, for all 10 pairs, with Bonferroni fields. |
| `per_k.<k>.stability.<method>` | Interval record for one k, k = 1 to 30. |

Interval record:

| Field | Meaning |
|---|---|
| `estimate_cv` | The CV estimate from `*_analysis.json`. Intervals are centred on it. |
| `boot_mean` | Mean over replicates. |
| `bias` | `boot_mean` minus `estimate_cv`. Replicates use half the students, so this is expected to be negative for single methods. |
| `se` | Sample SD of the replicates, used as the standard error. |
| `ci95` | `estimate_cv` ± 1.96 `se`. |
| `percentile_replicates` | 2.5th and 97.5th percentiles of the replicates; descriptive only. |
| `ci_bonferroni` | `diff` only: `estimate_cv` ± z(1 − 0.05/20) `se`, for 10 pairs. |
| `robust`, `robust_bonferroni` | `diff` only: whether `ci95` or `ci_bonferroni` excludes zero. |

## `oulad_dayN_cross_cohort*.json` and `.jsonl`

Agreement within and between presentations, written by `cross_cohort.py`.
Suffix: none for all modules at 1,650 students per set; `_common` for
`--modules common` (BBB, DDD and FFF quotas of 560, 330 and 480 students);
`_n1370` for all modules at 1,370 students.

`.jsonl`, one record per repeat (20):

| Field | Meaning |
|---|---|
| `r` | Repeat index; its seed is 42 + r. |
| `sha` | `code_sha` of the code that wrote the record. |
| `design` | `modules` and `n_students` (a number, or the module quotas). |
| `rankings.S{1,2}_<pres>.<method>` | Ranking on each of the two disjoint student sets of each presentation (`2013B`, `2013J`, `2014B`, `2014J`), computed on all rows of the set. |
| `info.S{1,2}_<pres>.<method>` | Ranker diagnostics, as in the fold records. |
| `composition.S{1,2}_<pres>` | `rows`, `students`, `pass_rate`, `no_submission_share`, `module_share` and `constant_cols` of the set. |

`.json`:

| Field | Meaning |
|---|---|
| `n_rep`, `design`, `p`, `presentations` | Repeats, design, number of features, presentation order. |
| `note` | The results are descriptive; spread over repeats is resampling noise. |
| `lasso_path` | `sets_short` (sets in which fewer LASSO features entered than needed) and `min_n_entered`. |
| `groups` | Named groups of presentation pairs: `dev_dev`, `dev_2014J`, `same_season`, `cross_season`, `gap1_dev`, `gap1_2014J`, `gap2_dev`, `gap2_2014J`, `gap3`. |
| `per_k.<k>.<key>.<method>` | `mean`, `sd`, `min`, `max` over repeats at one k (1 to 30). |
| `summaries.{grid,k_le_p3}` | `role` (`primary` for `grid`), `k`, and the same keys averaged over those k before taking `mean`, `sd`, `min`, `max` over repeats. |
| `loss_positive.{grid,k_le_p3}.<method>` | `share_of_repeats.<a>\|<b>`: share of repeats with a positive loss for the pair; `pairs_with_mean_loss_gt0`: pairs (of 6) with a positive mean loss. |
| `provenance` | See above. |

Keys inside `per_k` and `summaries`, with agreement measured by the Nogueira
index for two sets (equal to Kuncheva's index):

| Key | Meaning |
|---|---|
| `within_<pres>` | Agreement between the two sets of one presentation. |
| `between_<a>\|<b>` | Mean agreement over the four cross pairs of sets from presentations a and b. |
| `loss_<a>\|<b>` | Mean of `within_<a>` and `within_<b>` minus `between_<a>\|<b>`. |
| `within_<group>`, `between_<group>`, `loss_<group>` | The same averaged over the pairs of a named group, e.g. `loss_dev_dev`, `loss_dev_2014J`. |
