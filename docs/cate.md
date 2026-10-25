# CATE implementation and population

CATE uses independent propensity screening and EconML XLearner.

## Inputs and procedure

1. Read the current threshold's analysis frame. Keep observed treatment and the selected mortality outcome. Missing X uses the imputation rules.
2. Require both treatment arms and binary A/Y. Fit the preprocessing on this complete CATE screening sample.
3. Compute class sampling probabilities from the full complete sample. Fit the balanced treatment model with the current family and seed + 40. Apply the case-control probability correction.
4. Select finite corrected PS values inside the inclusive configured bounds. Require both arms after selection. The default bounds are 0.01 and 0.99. The numerical bounds use analysis.trimming. Selection is independent of the ATE OOF mask.
5. Fit preprocessing again on the CATE-retained sample. Missing indicators and keep_empty_features reach both fits. Empty training columns are dropped by default; patient rows remain.
6. Construct EconML XLearner with the two outcome-probability wrappers at seed/seed + 1, two effect regressors at seed + 2/seed + 3, and propensity classifier at seed + 4. Set categories=[0,1]. Call fit(y,a,X=x), then effect(x,T0=0,T1=1).
7. Require one finite effect per retained patient. Join the full-reference subgroup labels and compute means, medians, individual P25/P75, subgroup N, and treatment rate.

The [official XLearner API](https://www.pywhy.org/EconML/_autosummary/econml.metalearners.XLearner.html) defines the constructor and fit/effect interface. Binary-outcome wrappers predict probabilities. Raw classifier labels do not replace those probabilities.

## Distinct propensity roles

| Role | Fit population | Use |
|---|---|---|
| Main ATE PS | balanced outer training folds; full complete-sample correction | main OOF mask and AIPW |
| CATE screening PS | balanced full A/Y-complete CATE sample; full complete-sample correction | independent CATE mask only |
| EconML propensity | full CATE-retained sample, fitted by XLearner | combine second-stage effect predictions |

The saved CATE propensity column is the corrected screening PS. It is not the library's combining probability. CATE is an in-sample exploratory fit. It is not an OOF prediction or a calibration assessment.

## Outputs

- cate_private/{family}_{outcome}.parquet contains patient IDs, effects, corrected screening PS, engine, population rule, bounds, class sampling rates, and before/after arm counts.
- cate_subgroups.csv retains the current subgroup summaries and exploratory labels. P25/P75 describe individual predictions. They are not confidence intervals.
- cate_status.csv reports every requested CATE cell, including failures. Successful rows state CATE_in_sample_corrected_propensity and its population counts.
- Supplement_CATE_population_audit carries that status table into report output. Captions identify the independent CATE population.

Four design fields are excluded from CATE model X but remain available for subgroup descriptions.

The default remains CATE for the 500/2000 primary pair, all declared families and outcomes. Other pairs remain available through cate_primary_only=false. Full-reference quantile cutpoints, structural-zero rules, figure forms, and failure isolation remain.

The two branches run from the same validated frame. An ATE failure does not suppress a valid CATE fit. A CATE failure does not change an ATE result. Completed CATE outputs are listed in the run inventory and checkpoint records.
