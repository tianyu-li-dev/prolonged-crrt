# Laboratory covariates

| Variable | Main item ID | Canonical unit | Summary |
|---|---:|---|---|
| Potassium | 50971 | mmol/L | Latest, mean, minimum, maximum |
| Sodium | 50983 | mmol/L | Latest, mean, minimum, maximum |
| Chloride | 50902 | mmol/L | Latest, mean, minimum, maximum |
| Bicarbonate | 50882 | mmol/L | Latest, mean, minimum, maximum |
| Blood pH | 50820 | Unitless | Latest, mean, minimum, maximum |
| Serum creatinine | 50912 | mg/dL | Latest, mean, maximum |
| BUN | 51006 | mg/dL | Latest, mean, maximum |

The source dictionary must confirm blood fluid for required item IDs. pH extraction does not require pO2. Specimen item 52033 provides recorded arterial or venous status.

Main pH uses all documented blood specimens and a specimen-status covariate. Arterial-only analysis is a declared sensitivity. Unknown source status remains explicit.

Main chemistry variables use chemistry sources. Optional fallback uses blood-gas item IDs 50822, 50824, 50806, and 50803. Chemistry takes priority at the same timestamp.

Calculated blood-gas bicarbonate differs from chemistry bicarbonate. Source changes require explicit configuration. The event audit retains excluded-source counts.

Optional variables include pCO2, base excess, reported anion gap, calcium, magnesium, and phosphate. They do not automatically enter the main model. Accepted optional variables use the same summaries.

Lactate uses the derived bg source. Raw biochemical laboratory extraction does not replace it. Urine output, vital signs, supports, medications, and fluid balance retain their separate feature roles.

The baseline insurance indicator groups Medicare and Medicaid. Its column name is `Insurance, Medicare`. Interpret it as that combined grouping.

## Missingness

The audit reports completeness by treatment group. It reports counts and measurement ages before imputation. A required variable without observations blocks estimation.

Completeness below 50% receives a sparse status. Partially sparse laboratories enter estimation by default. An explicit --no-allow-sparse-labs requires at least 50% completeness. Zero-observation required laboratories retain the separate unavailable status.

These percentages are operational review rules. They are not clinical validity thresholds. Inspect treatment-specific missingness even when overall completeness is adequate.

## Availability sensitivities

The main convention uses measurement time (`charttime`). The main window includes CRRT start and time zero. The strict configuration also requires a known result storage timestamp (`storetime`). Both the measurement and its result must be available by time zero.

The strict result clock is `max(charttime, storetime)`. This clock controls availability only. It does not move an old measurement into the baseline window. Latest-value summaries sort measurement time.

Result versions remain separate until the patient-specific time zero is known. The snapshot keeps the latest eligible version of each specimen, item, and measurement timestamp. A correction stored after time zero cannot remove an earlier eligible result.

Specimen labels have their own availability check. The label must be recorded by time zero. The strict configuration also requires its storage time by time zero. Arterial restriction occurs after this check. A future arterial label cannot classify an earlier unknown specimen.

This availability sensitivity applies only to the biochemical laboratory block. Existing lactate, vital signs, and scores use their respective source definitions.

A final-24-hour configuration changes only the biochemical laboratory window. Other repeated covariates keep their existing windows. First-day scores are static and ignore collection timestamps.

## Other time-dependent sources

The model uses the latest valid CRRT modality record at or before t0 within the index episode. It compares procedure labels and derived CRRT modes. The episode-level label is an audit field. Ventilation excludes None and missing labels before interval summaries.

Input fluid features use the overlap of the infusion interval and baseline window. Recognized volume units convert to mL. Recognized volume rates integrate over the overlap. Drug mass never becomes volume.

Without a valid volume rate, the code prorates the documented volume across its interval. This assumes a constant rate within that interval. Review this assumption against the actual input source. Bolus volume uses its start timestamp.

The input audit identifies invalid units, cancelled records, duplicate versions, and ambiguous solution components. Other output and CRRT-removal features retain their source definitions. Actual CRRT-removal units still require clinical-source review.
