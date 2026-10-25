# Analysis methods

## Population and time origin

The analysis includes adult patients in their first ICU stay, with at least one day of ICU follow-up and no death during the first 24 hours. Time zero is the first valid urine-output threshold crossing in the index CRRT episode, including a urine record exactly at the episode end. The landmark is time zero plus 24 hours. The cohort excludes deaths in (time zero, landmark] and requires classifiable treatment status.

CRRT intervals are start-inclusive and end-exclusive. Treatment is active CRRT at the landmark; restarts count if treatment is active then. The comparison does not require uninterrupted continuation. The episode merge gap is six hours. Loop-diuretic exposure uses item IDs 228340, 221794 and 229639 in the preceding 24 hours.

Mortality horizons start at the landmark. Patient `dod` supplies death dates, retaining their recorded precision. Hospital `deathtime` is not substituted. The standalone outcome-labeling helper includes deaths equal to the landmark; cohort eligibility excludes those deaths before estimation.

## Covariates and preprocessing

Repeated measurements use CRRT-start-to-time-zero summaries: latest value, mean and declared extrema. Later measurements do not fill earlier missing values. First-day SOFA and SAPS II are static ICU-stay scores joined without collection-window restrictions. CRRT modality is the latest valid record at or before time zero in the index episode. Ventilation summaries exclude None and missing support labels.

Continuous variables receive training-fold median imputation and scaling. Binary event absence uses a declared structural-zero rule; absent laboratory measurements remain missing. All-missing columns are removed using training data only. An intercept-only fallback retains rows when no predictor remains. Optional missing indicators apply to all continuous measurements.

Fluid input uses interval overlap and explicit unit conversion. Laboratory result records and specimen labels are resolved separately at each time zero. Lactate uses derived `bg`. See [covariates](covariates.md).

## ATE estimation

The five probability model families are ridge logistic regression, lasso, elastic net, XGBoost and random forest. Random forest uses 1000 trees, maximum depth 4, square-root feature selection and `max(3, ceil(0.03*n_training))` observations per leaf.

Each outer training fold uses an exact-count balanced subset for propensity fitting. Full complete-sample class sampling probabilities correct those predictions. Treatment-stratified folds are shuffled with a fixed random seed. Every row receives held-out outcome and propensity predictions; incomplete predictions cause an error.

The AIPW score is averaged over observations with corrected propensity in [0.01, 0.99], inclusively. Numerical probability clipping is a separate operation. Different model families and threshold pairs can select different target populations.

## Bootstrap

Each replicate samples analysis rows with replacement and refits preprocessing, nuisance models, undersampling, correction, trimming and AIPW. Folds are assigned to sampled rows; copies of a source row can enter different folds.

The main interval uses the 2.5th and 97.5th percentiles of successful bootstrap estimates. At least 95% of 1000 replicates must succeed. Otherwise the result is marked incomplete. Influence-score and empirical bootstrap standard errors are recorded separately.

## CATE and sensitivity

CATE uses its own full-sample propensity screening followed by EconML XLearner. It does not use the main ATE mask. Subgroup cutpoints use the complete landmark cohort. Individual-effect quartiles are distributions, not confidence intervals. See [CATE](cate.md).

Point E-values, IRM robustness values and feature-block benchmarks use the declared retained populations. Separate bounded TMLE estimates are labeled separately. See [sensitivity calculations](sensitivity.md).

## Limitations

Conditioning on landmark survival restricts the target population and does not eliminate selection bias. Conditional exchangeability remains an assumption. Laboratory concentrations during CRRT reflect treatment and illness; creatinine does not directly measure recovered intrinsic filtration. Missingness and measurement frequency may be informative. The analysis is single-center and exploratory across specifications. A confidence interval containing zero does not establish absence of a clinical effect.
