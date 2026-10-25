"""X-learner distribution summaries. These intervals are not confidence intervals."""
import numpy as np
import pandas as pd
from .preprocessing import fit_transformer
from .models import (_make_outcome_probability_model, _make_effect_regressor, _make_probability_classifier,
                     _fit_undersampled_propensity, _predict_positive_probability, _undersampling_sample_probs,
                     _calibrate_case_control_propensity, _crump_trim_mask, _normalize_crump_trimming_bounds,
                     _with_cate_subgroup_helpers, _subgroup_definitions)
from .models import STRUCTURAL_ZERO_FEATURE_COLUMNS, STRUCTURAL_ZERO_FEATURE_PREFIXES


def subgroup_reference(frame):
    """Fix subgroup cutpoints before estimator-specific patient exclusion."""
    working = frame.copy()
    for column in working:
        if column in STRUCTURAL_ZERO_FEATURE_COLUMNS or column.startswith(STRUCTURAL_ZERO_FEATURE_PREFIXES):
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)
    working = _with_cate_subgroup_helpers(working)
    definitions = _subgroup_definitions(working)
    variables = {"urine_output_tertile": "time_zero_urineoutput_24hr", "net_fluid_balance_tertile": "net_fluid_balance_ml_start_to_tz",
                 "early_vs_late_time_zero": "hours_crrt_start_to_time_zero",
                 "sofa_strata": "tz_last_sofa" if "tz_last_sofa" in working else "tz_last_baseline_sofa",
                 "sapsii_strata": "tz_last_sapsii" if "tz_last_sapsii" in working else "tz_last_baseline_sapsii"}
    boundaries = []
    for group, column in variables.items():
        if column not in working:
            continue
        numeric = pd.to_numeric(working[column], errors="coerce")
        try:
            _, raw_bins = pd.qcut(numeric, 3, labels=False, retbins=True, duplicates="drop")
            # Interpolated quantiles can differ from a tied value by roundoff only.
            tolerance = 64 * np.finfo(float).eps * max(1.0, np.max(np.abs(raw_bins)))
            bins = []
            for boundary in raw_bins:
                if bins and abs(boundary - bins[-1]) <= tolerance:
                    bins[-1] = boundary
                else:
                    bins.append(boundary)
            if len(bins) < 2:
                raise ValueError("No distinct quantile boundaries.")
            bins[0], bins[-1] = raw_bins[0], raw_bins[-1]
            codes = pd.cut(numeric, bins, labels=False, include_lowest=True)
            precision = next((digits for digits in range(3, 16)
                              if len({f"{value:.{digits}g}" for value in bins}) == len(bins)), 16)
            labels = {i: f"Q{i+1} {'[' if i == 0 else '('}{bins[i]:.{precision}g}, {bins[i+1]:.{precision}g}]"
                      for i in range(len(bins) - 1)}
            definitions[group] = codes.map(labels)
            boundaries.extend({"subgroup": group, "level": labels[i], "lower": bins[i], "upper": bins[i+1],
                               "reference_n": len(working), "reference_population": "eligible_landmark_cohort"}
                              for i in labels)
        except ValueError:
            definitions[group] = pd.Series(pd.NA, index=working.index, dtype="object")
    if "sex" in definitions:
        definitions["sex"] = definitions["sex"].map({"yes": "Female", "no": "Male"})
    return {name: pd.Series(series.to_numpy(), index=working.subject_id) for name, series in definitions.items()}, pd.DataFrame(boundaries)


def xlearner(frame, outcome, family, seed=42, reference_frame=None, missing_indicators=False,
             trimming_bounds=(0.01, 0.99), keep_empty_features=False):
    """Fit EconML after independent CATE PS trimming; never consume an ATE mask."""
    from econml.metalearners import XLearner
    bounds = _normalize_crump_trimming_bounds(trimming_bounds)
    complete = frame.loc[frame[outcome].notna() & frame.intervention_status.notna()].copy()
    a, y = complete.intervention_status.to_numpy(dtype=int), complete[outcome].to_numpy(dtype=float)
    if not np.isin(a, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Treatment and mortality must be binary for X-learner.")
    if len(np.unique(a)) != 2:
        raise ValueError("Both treatment arms are required for X-learner.")
    # Screen the CATE population independently of ATE OOF predictions.
    _, x_propensity, _ = fit_transformer(complete, missing_indicators, keep_empty_features)
    sample_probs = _undersampling_sample_probs(a)
    prop = _fit_undersampled_propensity(x_propensity, a, family, seed + 40)
    propensity = _calibrate_case_control_propensity(_predict_positive_probability(prop, x_propensity), sample_probs)
    trim = _crump_trim_mask(propensity, a, bounds)
    if not trim["treated_after"] or not trim["control_after"]:
        raise ValueError("Independent CATE PS trimming must retain both treatment arms.")
    selected = complete.loc[trim["mask"]].copy()
    a, y = a[trim["mask"]], y[trim["mask"]]
    _, x, _ = fit_transformer(selected, missing_indicators, keep_empty_features)
    learner = XLearner(
        models=[_make_outcome_probability_model(family, seed), _make_outcome_probability_model(family, seed + 1)],
        cate_models=[_make_effect_regressor(family, len(a), seed + 2), _make_effect_regressor(family, len(a), seed + 3)],
        propensity_model=_make_probability_classifier(family, len(a), seed + 4), categories=[0, 1],
    )
    learner.fit(y, a, X=x)
    effects = np.asarray(learner.effect(x, T0=0, T1=1), dtype=float).reshape(-1)
    if len(effects) != len(selected) or not np.isfinite(effects).all():
        raise ValueError("EconML returned incomplete or non-finite CATE predictions.")
    population = {"cate_engine": "econml.metalearners.XLearner", "cate_population": "CATE_specific_PS_overlap",
                  "trimming_source": "CATE_in_sample_corrected_propensity", "trimming_method": "crump",
                  "trimming_lower": trim["lower"], "trimming_upper": trim["upper"],
                  "n_before_trimming": trim["n_before"], "n_after_trimming": trim["n_after"],
                  "treated_after_trimming": trim["treated_after"], "control_after_trimming": trim["control_after"],
                  "sampling_probability_0": sample_probs[0], "sampling_probability_1": sample_probs[1]}
    individual = selected[["subject_id", "hadm_id", "stay_id"]].copy()
    individual["outcome_name"], individual["estimator_name"] = outcome, family
    individual["cate"] = effects
    individual["propensity"] = propensity[trim["mask"]]
    for name, value in population.items():
        individual[name] = value
    definitions, boundaries = subgroup_reference(reference_frame if reference_frame is not None else frame)
    rows = []
    for group, reference in definitions.items():
        levels = selected.subject_id.map(reference)
        for level in levels.dropna().unique():
            idx = levels.eq(level).to_numpy()
            values = effects[idx]
            rows.append({"outcome_name": outcome, "subgroup": group, "level": str(level), "n": int(idx.sum()),
                         "mean_cate": float(values.mean()), "median_cate": float(np.median(values)),
                         "cate_p25": float(np.quantile(values, .25)), "cate_p75": float(np.quantile(values, .75)),
                         "treatment_rate": float(a[idx].mean())})
    subgroup = pd.DataFrame(rows)
    subgroup["interval_type"] = "individual_CATE_p25_p75"
    subgroup["analysis_role"] = "exploratory"
    for name, value in population.items():
        subgroup[name] = value
    return individual, subgroup
