# Point sensitivity calculations

## E-value

Compute two adjusted risks from the same OOF predictions and retained population as the main AIPW estimate.

```text
p1 = mean(mu1 + A/e * (Y - mu1))
p0 = mean(mu0 + (1-A)/(1-e) * (Y - mu0))
ATE = p1 - p0
RR = p1 / p0
R = max(RR, 1/RR)
E = R + sqrt(R*(R-1))
```

Require positive risks within the probability range. Do not transform a risk difference directly. Do not use crude control mortality as the adjusted reference risk.

Finite-sample AIPW risks can be invalid. Mark their direct E-value unavailable. A separate two-coordinate logistic TMLE supplies a bounded supplementary estimate when supported.

TMLE risks, risk difference, risk ratio, and E-value have separate columns. Its E-value does not belong to the original AIPW point estimate. No final-risk clipping or arbitrary constant is used.

The targeting regression clips initial nuisance probabilities before applying the logit. This standard numerical safeguard does not clip the final adjusted risks. Constant-outcome treatment arms receive an unavailable targeting status.

## IRM RV

```text
g = A*mu1 + (1-A)*mu0
alpha = A/e - (1-A)/(1-e)
sigma2 = mean((Y-g)^2)
nu2_raw = mean(2*(1/e + 1/(1-e)) - alpha^2)
```

The main sensitivity moment uses `nu2_raw` when positive. DoubleML uses `mean(alpha^2)` when that estimated moment is not positive. This package follows that explicit fallback.

Both the raw moment and the method label are saved. The fallback is non-orthogonal and can indicate poor finite-sample behavior. It is not an arbitrary positive floor.

For `rho=1`, calculate equal-strength point RV:

```text
b = abs(ATE) / sqrt(sigma2*nu2)
RV = 2*b / (sqrt(b*b + 4) + b)
RV_percent = 100*RV
```

The point bias bound is:

```text
bias = abs(rho)*sqrt(sigma2*nu2)*sqrt(cf_y*cf_d/(1-cf_d))
point_bounds = [ATE-bias, ATE+bias]
```

`cf_y` concerns residual outcome variation. IRM `cf_d` concerns gains in conditional treatment precision through the Riesz representer. It is not a generic treatment-residual partial R-squared.

The default official-package check uses external OOF predictions. Main and reference ATEs, moments, and point RVs must agree on the same retained population. It does not replace main bootstrap inference. An explicit no-reference-doubleml flag can disable the reference for a stated supplementary run.

## Benchmarks

Refit after dropping a complete clinical feature block. Keep the long-model overlap mask, rows, folds, and seeds fixed. Do not retrim with the short-model propensity.

Gain statistics follow the pinned DoubleML implementation. The treatment-precision gain is not the bounded fraction used in the point-bound formula. Convert the gain explicitly:

```text
gain_d = (nu2_long - nu2_short) / nu2_short
cf_d = gain_d / (1 + gain_d)
cf_y = (sigma2_short - sigma2_long) / sigma2_long
```

The audit saves `cf_d_gain_raw`, `cf_d_raw`, and the accepted fraction separately. For 1x and 2x scenarios, multiply the precision gain and outcome fraction first. Convert each multiplied gain to its own fraction. Invalid or unstable strengths receive an unavailable status. They are not clipped into an interpretable benchmark.

Each block removes latest, mean, extreme, and source-related proxy features. Blocks cover fluid status, electrolytes, acid-base status, renal markers, vasopressors, and severity.

Electrolyte blocks include accepted calcium, magnesium, and phosphate features. Acid-base blocks include accepted pCO2, base excess, and anion gap features. These variables remain optional in the design. Their presence cannot escape a block benchmark through an incomplete name list.

The compact RV table labels non-orthogonal moment fallback explicitly. The detailed audit retains both moments. A positive fallback supports calculation but does not establish good estimator calibration.

## Interpretation

Only point E-values and point RVs appear in the sensitivity table. Confidence-limit E-values and RVa are omitted. Main ATE CIs remain visible.

These statistics concern point-estimate robustness. They do not create significance or establish a stable causal direction. Sensitivity contours show confounding point bounds, not confidence intervals.

Method sources include VanderWeele and Ding (2017) and *Long Story Short: Omitted Variable Bias in Causal Machine Learning*. Official documentation: [DoubleML sensitivity](https://docs.doubleml.org/stable/guide/sensitivity.html).

## Default official comparison

DoubleML is a required dependency. The sensitivity command uses it by default with saved retained OOF predictions. It checks ATE, sigma2, nu2, and point RV. RV absolute tolerance is 1e-5; a mismatch stops sensitivity registration. Reference values, status, and difference are saved. Main nuisance models are not retrained. An explicit no-reference-doubleml flag disables this comparison for a declared supplementary run.
