"""Probability and effect models from the manuscript specification."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Iterable

import numpy as np

import pandas as pd

from .constants import (
    ADMISSION_AGE,
    DOD,
    EPISODE_START,
    HADM_ID,
    INTERVENTION_STATUS,
    STAY_ID,
    SUBJECT_ID,
)

OUTCOME_COLUMNS = ("mortality_28days", "mortality_60days", "mortality_90days")

ESTIMATOR_NAMES = ("logistic", "xgboost", "rf")

EXTRA_ESTIMATOR_NAMES = ("lasso", "elastic_net", "mlp")

SUPPORTED_ESTIMATOR_NAMES = ESTIMATOR_NAMES + EXTRA_ESTIMATOR_NAMES

CRUMP_TRIMMING_BOUNDS = ((0.01, 0.99), (0.05, 0.95), (0.10, 0.90))

DEFAULT_CRUMP_TRIMMING_BOUNDS = (0.05, 0.95)

ANTIBIOTIC_COLUMNS = ("Glycopeptide", "Beta-lactams", "Carbapenems", "Aminoglycosides")

CATE_HELPER_COLUMNS = {"hours_crrt_start_to_time_zero"}

CATEGORICAL_FEATURE_COLUMNS = {
    "anchor_year_group",
    "crrt_primary_modality",
    "latest_pre_tz_rrt_modality_status",
}

STRUCTURAL_ZERO_FEATURE_COLUMNS = {
    *ANTIBIOTIC_COLUMNS,
    "positive_fluid_balance_start_to_tz",
}

STRUCTURAL_ZERO_FEATURE_PREFIXES = (
    "active_at_tz_",
    "any_pre_tz_",
    "event_count_start_to_tz_",
    "exposure_hours_start_to_tz_",
    "fluid_crrt_removal_ml_",
    "fluid_derived_urine_output_ml_",
    "fluid_input_ml_",
    "fluid_outputevents_output_ml_",
)

NON_FEATURE_COLUMNS = {
    "time_zero_threshold_ml_24hr", "threshold_margin_ml_24hr", "time_zero_uo_tm_24hr", "anchor_year_group",
    SUBJECT_ID,
    HADM_ID,
    STAY_ID,
    DOD,
    INTERVENTION_STATUS,
    "early_discontinuation",
    "continued_crrt_at_landmark",
    "treatment_classifiable",
    "treatment_label",
    "landmark_hours",
    "time_zero",
    "landmark_time",
    EPISODE_START,
    *OUTCOME_COLUMNS,
}

def summarize_cate_subgroups(frame: pd.DataFrame, cate: np.ndarray, outcome_name: str) -> pd.DataFrame:
    """Aggregate individual CATEs across clinically relevant subgroups."""

    working = _with_cate_subgroup_helpers(frame)
    working["_cate"] = cate

    # Subgroup definitions are centralized so CATE outputs remain consistent
    # across all outcomes.
    subgroup_defs = _subgroup_definitions(working)
    rows = []
    for group_name, series in subgroup_defs.items():
        # Each subgroup is summarized level-by-level; missing subgroup labels are
        # left out rather than treated as clinical strata.
        for level, idx in series.dropna().groupby(series.dropna()).groups.items():
            vals = working.loc[idx, "_cate"].dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "outcome_name": outcome_name,
                    "subgroup": group_name,
                    "level": str(level),
                    "n": int(vals.shape[0]),
                    "mean_cate": float(vals.mean()),
                    "median_cate": float(vals.median()),
                    "cate_p25": float(vals.quantile(0.25)),
                    "cate_p75": float(vals.quantile(0.75)),
                    "treatment_rate": float(working.loc[idx, INTERVENTION_STATUS].mean()),
                }
            )
    return pd.DataFrame(rows)

class _ProbabilityOutcomeModel:
    """Sklearn-compatible wrapper whose predict method returns P(Y=1|X)."""

    def __init__(self, estimator_name: str = "logistic", random_state: int = 42):
        self.estimator_name = estimator_name
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> dict:
        return {"estimator_name": self.estimator_name, "random_state": self.random_state}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, x, y):
        from sklearn.dummy import DummyRegressor

        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) < 4 or np.unique(y_arr[np.isfinite(y_arr)]).size < 2:
            self.model_ = DummyRegressor(strategy="mean")
            self.model_.fit(x, y_arr)
        else:
            self.model_ = _make_probability_classifier(
                self.estimator_name,
                n_samples=len(y_arr),
                random_state=self.random_state,
            )
            self.model_.fit(x, y_arr.astype(int))
        return self

    def predict(self, x):
        return _predict_probability(self.model_, x)

def _normalize_estimator_name(estimator_name: str) -> str:
    name = str(estimator_name).strip().lower()
    aliases = {
        "xgb": "xgboost",
        "random_forest": "rf",
        "forest": "rf",
        "regularized_lr": "logistic",
        "elasticnet": "elastic_net",
        "enet": "elastic_net",
        "multilayer_perceptron": "mlp",
        "neural_net": "mlp",
    }
    name = aliases.get(name, name)
    if name not in SUPPORTED_ESTIMATOR_NAMES:
        raise ValueError(
            f"Unsupported landmark estimator {estimator_name!r}; expected one of {SUPPORTED_ESTIMATOR_NAMES}."
        )
    return name

def _make_outcome_probability_model(estimator_name: str, random_state: int):
    return _ProbabilityOutcomeModel(estimator_name=_normalize_estimator_name(estimator_name), random_state=random_state)

def _make_probability_classifier(estimator_name: str, n_samples: int, random_state: int):
    estimator_name = _normalize_estimator_name(estimator_name)
    if estimator_name == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, solver="lbfgs")
    if estimator_name == "lasso":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            penalty="l1",
            solver="saga",
            C=1.0,
            max_iter=5000,
            random_state=random_state,
        )
    if estimator_name == "elastic_net":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=5000,
            random_state=random_state,
        )
    if estimator_name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=1e-4,
            max_iter=400,
            random_state=random_state,
        )
    if estimator_name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=1000,
            max_depth=4,
            min_samples_leaf=_small_sample_min_leaf(n_samples),
            max_features="sqrt",
            random_state=random_state,
            n_jobs=1,
        )
    if estimator_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.05,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=3,
            reg_alpha=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(f"Unsupported landmark estimator {estimator_name!r}.")

def _make_effect_regressor(estimator_name: str, n_samples: int, random_state: int):
    estimator_name = _normalize_estimator_name(estimator_name)
    if estimator_name == "logistic":
        from sklearn.linear_model import Ridge

        return Ridge(tol=1e-3)
    if estimator_name == "lasso":
        from sklearn.linear_model import Lasso

        return Lasso(alpha=0.01, max_iter=10000)
    if estimator_name == "elastic_net":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
    if estimator_name == "mlp":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=(32, 16),
            alpha=1e-4,
            max_iter=400,
            random_state=random_state,
        )
    if estimator_name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=1000,
            max_depth=4,
            min_samples_leaf=_small_sample_min_leaf(n_samples),
            max_features="sqrt",
            random_state=random_state,
            n_jobs=1,
        )
    if estimator_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.05,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=3,
            reg_alpha=0.1,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(f"Unsupported landmark estimator {estimator_name!r}.")

def _small_sample_min_leaf(n_samples: int) -> int:
    return max(3, int(np.ceil(0.03 * max(1, int(n_samples)))))

def _fit_outcome_probability_model(x: np.ndarray, y: np.ndarray, estimator_name: str, random_state: int):
    """Fit a robust nuisance outcome probability model."""

    model = _make_outcome_probability_model(estimator_name, random_state)
    model.fit(x, y)
    return model

def _fit_undersampled_propensity(x: np.ndarray, a: np.ndarray, estimator_name: str, random_state: int):
    """Fit propensity after balancing treatment groups by random undersampling."""

    from sklearn.dummy import DummyClassifier

    if len(np.unique(a)) < 2:
        model = DummyClassifier(strategy="prior")
        model.fit(x, a)
        return model
    sampled = _undersampling_indices(a, random_state)
    model = _make_probability_classifier(estimator_name, n_samples=len(sampled), random_state=random_state)
    model.fit(x[sampled], a[sampled])
    return model


def _undersampling_indices(a, random_state):
    """Select a balanced training subset with deterministic row order."""
    rng = np.random.default_rng(random_state)
    treated = np.where(a == 1)[0]
    control = np.where(a == 0)[0]
    n_treated, n_control = len(treated), len(control)
    if n_treated == 0 or n_control == 0:
        # This should usually be caught earlier, but a dummy model makes the
        # helper total and easier to test.
        return np.arange(len(a))

    if n_treated < n_control:
        # Keep all treated patients and randomly sample controls to match them.
        sampled_control = rng.choice(control, size=n_treated, replace=False)
        sampled = np.concatenate([treated, sampled_control])
    elif n_control < n_treated:
        # Symmetric case: keep all controls and sample treated patients.
        sampled_treated = rng.choice(treated, size=n_control, replace=False)
        sampled = np.concatenate([sampled_treated, control])
    else:
        # No imbalance correction is needed when group sizes already match.
        sampled = np.arange(len(a))

    return sampled

def _predict_probability(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return _predict_positive_probability(model, x)
    return np.asarray(model.predict(x), dtype=float).reshape(-1)

def _predict_positive_probability(model, x: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x), dtype=float)
    if proba.ndim == 1:
        return proba.reshape(-1)
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.size:
        matches = np.where(classes == 1)[0]
        if matches.size:
            return proba[:, int(matches[0])]
        return np.zeros(proba.shape[0], dtype=float)
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, 1]

def _normalize_crump_trimming_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    """Validate absolute propensity-score trimming bounds."""

    if len(bounds) != 2:
        raise ValueError("Crump trimming bounds must contain exactly two values.")
    lower, upper = (float(bounds[0]), float(bounds[1]))
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError(f"Invalid Crump trimming bounds: {(lower, upper)!r}.")
    return lower, upper

def _crump_trim_mask(propensity: np.ndarray, treatment: np.ndarray, bounds: tuple[float, float]) -> dict:
    """Return Crump-style absolute propensity-score trimming metadata and mask."""

    e = np.asarray(propensity, dtype=float)
    a = np.asarray(treatment, dtype=int)
    lower, upper = _normalize_crump_trimming_bounds(bounds)
    finite = np.isfinite(e)
    mask = finite & (e >= lower) & (e <= upper)
    return {
        "mask": mask,
        "lower": float(lower),
        "upper": float(upper),
        "n_before": int(len(e)),
        "n_after": int(mask.sum()),
        "treated_after": int(np.sum(mask & (a == 1))),
        "control_after": int(np.sum(mask & (a == 0))),
    }

def _undersampling_sample_probs(a: np.ndarray) -> dict[int, float]:
    """Return class sampling probabilities implied by balancing the full sample."""

    n_treated = int(np.sum(a == 1))
    n_control = int(np.sum(a == 0))
    if n_treated == 0 or n_control == 0:
        return {1: 1.0, 0: 1.0}
    if n_treated < n_control:
        return {1: 1.0, 0: n_treated / n_control}
    if n_control < n_treated:
        return {1: n_control / n_treated, 0: 1.0}
    return {1: 1.0, 0: 1.0}

def _calibrate_case_control_propensity(ps_sample: np.ndarray, sample_probs: dict[int, float]) -> np.ndarray:
    """Calibrate propensity from undersampled data back to the original sample."""

    # If q is the propensity in the case-control/undersampled data, this formula
    # uses class-specific sampling probabilities to recover the original-scale
    # propensity P(A=1|X).
    q = np.clip(np.asarray(ps_sample, dtype=float), 1e-6, 1 - 1e-6)
    s1 = float(sample_probs.get(1, 1.0))
    s0 = float(sample_probs.get(0, 1.0))
    numerator = q * s0
    denominator = numerator + (1 - q) * s1
    return numerator / denominator

def _is_model_feature_column(column: str) -> bool:
    """Return whether a frame column may enter nuisance or CATE models."""

    if column in NON_FEATURE_COLUMNS or column in CATE_HELPER_COLUMNS:
        return False
    if column.startswith("subgroup_"):
        return False
    if "aki_stage" in column.lower():
        return False
    return True

def _with_cate_subgroup_helpers(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive CATE subgroup helpers on a temporary copy only."""

    out = frame.copy()
    if ADMISSION_AGE in out:
        age = pd.to_numeric(out[ADMISSION_AGE], errors="coerce")
        out["subgroup_age_ge_65"] = _float_condition(age >= 65, age.notna())
    if "threshold_margin_ml_24hr" in out:
        margin = pd.to_numeric(out["threshold_margin_ml_24hr"], errors="coerce")
        out["subgroup_near_urine_threshold"] = _float_condition(margin.between(0, 250), margin.notna())
    if "time_zero" in out and EPISODE_START in out:
        out["hours_crrt_start_to_time_zero"] = (
            pd.to_datetime(out["time_zero"]) - pd.to_datetime(out[EPISODE_START])
        ).dt.total_seconds() / 3600.0
    if "net_fluid_balance_ml_start_to_tz" in out:
        net_balance = pd.to_numeric(out["net_fluid_balance_ml_start_to_tz"], errors="coerce")
        out["subgroup_positive_fluid_balance"] = _float_condition(net_balance > 0, net_balance.notna())
    return out

def _subgroup_definitions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Build clinically motivated subgroup labels for CATE summaries."""

    out: dict[str, pd.Series] = {}
    if "time_zero_urineoutput_24hr" in frame:
        # Tertiles summarize heterogeneity across urine-output recovery levels.
        out["urine_output_tertile"] = _safe_qcut(frame["time_zero_urineoutput_24hr"], 3)
    if "subgroup_near_urine_threshold" in frame:
        out["near_urine_threshold"] = _yes_no_from_binary(frame["subgroup_near_urine_threshold"])
    if "threshold_margin_ml_24hr" in frame:
        out["urine_threshold_margin_band"] = _threshold_margin_band(frame["threshold_margin_ml_24hr"])
    if "diuretic_prior_24h" in frame:
        out["diuretic_prior_24h"] = _yes_no_from_binary(frame["diuretic_prior_24h"])
    if "active_at_tz_vasopressor" in frame:
        out["vasopressor_active_at_tz"] = _yes_no_from_binary(frame["active_at_tz_vasopressor"])
    if "active_at_tz_ventilation" in frame:
        out["ventilation_active_at_tz"] = _yes_no_from_binary(frame["active_at_tz_ventilation"])
    if "worst_start_to_tz_mbp" in frame:
        mbp = pd.to_numeric(frame["worst_start_to_tz_mbp"], errors="coerce")
        out["low_map"] = _yes_no_from_condition(mbp < 65, mbp.notna())
    if "worst_start_to_tz_spo2" in frame:
        spo2 = pd.to_numeric(frame["worst_start_to_tz_spo2"], errors="coerce")
        out["low_spo2"] = _yes_no_from_condition(spo2 < 90, spo2.notna())
    if "tz_last_lactate" in frame:
        # Lactate cut points reflect common clinical severity thresholds.
        lactate = pd.to_numeric(frame["tz_last_lactate"], errors="coerce")
        out["lactate_ge_2"] = _yes_no_from_condition(lactate >= 2, lactate.notna())
        out["lactate_ge_4"] = _yes_no_from_condition(lactate >= 4, lactate.notna())
    sofa_col = "tz_last_sofa" if "tz_last_sofa" in frame else "tz_last_baseline_sofa" if "tz_last_baseline_sofa" in frame else None
    if sofa_col:
        # SOFA is stratified by the observed distribution because absolute
        # categories vary by cohort and source table.
        out["sofa_strata"] = _safe_qcut(frame[sofa_col], 3)
    sapsii_col = (
        "tz_last_sapsii"
        if "tz_last_sapsii" in frame
        else "tz_last_baseline_sapsii"
        if "tz_last_baseline_sapsii" in frame
        else None
    )
    if sapsii_col:
        out["sapsii_strata"] = _safe_qcut(frame[sapsii_col], 3)
    if "subgroup_positive_fluid_balance" in frame:
        out["positive_fluid_balance"] = _yes_no_from_binary(frame["subgroup_positive_fluid_balance"])
    if "net_fluid_balance_ml_start_to_tz" in frame:
        out["net_fluid_balance_tertile"] = _safe_qcut(frame["net_fluid_balance_ml_start_to_tz"], 3)
        out["high_fluid_balance"] = _top_tertile_label(frame["net_fluid_balance_ml_start_to_tz"])
    if "any_pre_tz_suspected_infection" in frame:
        out["suspected_infection"] = _yes_no_from_binary(frame["any_pre_tz_suspected_infection"])
    for antibiotic in ANTIBIOTIC_COLUMNS:
        if antibiotic in frame:
            out[f"antibiotic_{_slug(antibiotic)}"] = _yes_no_from_binary(frame[antibiotic])
    antibiotic_cols = [col for col in ANTIBIOTIC_COLUMNS if col in frame]
    if antibiotic_cols:
        out["broad_antibiotic_exposure"] = _any_binary_label(frame[antibiotic_cols])
    if "hours_crrt_start_to_time_zero" in frame:
        # Tertiles separate early, middle, and late readiness signals.
        out["early_vs_late_time_zero"] = _safe_qcut(frame["hours_crrt_start_to_time_zero"], 3)
    if "crrt_primary_modality" in frame:
        out["crrt_modality"] = frame["crrt_primary_modality"].astype("string")
    if "latest_pre_tz_rrt_modality_status" in frame:
        out["rrt_modality_status"] = frame["latest_pre_tz_rrt_modality_status"].astype("string")
    if ADMISSION_AGE in frame:
        age = pd.to_numeric(frame[ADMISSION_AGE], errors="coerce")
        out["age_ge_65"] = _yes_no_from_condition(age >= 65, age.notna())
    if "Female" in frame:
        out["sex"] = _yes_no_from_binary(frame["Female"])
    if "White" in frame:
        out["race_white"] = _yes_no_from_binary(frame["White"])
    if "Emergency admission" in frame:
        out["emergency_admission"] = _yes_no_from_binary(frame["Emergency admission"])
    if "Insurance, Medicare" in frame:
        out["insurance_medicare"] = _yes_no_from_binary(frame["Insurance, Medicare"])
    if "anchor_year_group" in frame:
        out["admission_era"] = frame["anchor_year_group"].astype("string")
    return out

def _yes_no_from_binary(series: pd.Series) -> pd.Series:
    """Map binary-like values to yes/no while preserving missingness."""

    values = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(pd.NA, index=series.index, dtype="string")
    labels.loc[values.eq(0)] = "no"
    labels.loc[values.gt(0)] = "yes"
    return labels

def _yes_no_from_condition(condition: pd.Series, known: pd.Series) -> pd.Series:
    """Map a boolean condition to yes/no while preserving unknown rows."""

    labels = pd.Series(pd.NA, index=condition.index, dtype="string")
    labels.loc[known & ~condition] = "no"
    labels.loc[known & condition] = "yes"
    return labels

def _float_condition(condition: pd.Series, known: pd.Series) -> pd.Series:
    """Convert a boolean condition to float flags while preserving unknown rows."""

    values = pd.Series(np.nan, index=condition.index, dtype=float)
    values.loc[known] = condition.loc[known].astype(float)
    return values

def _threshold_margin_band(series: pd.Series) -> pd.Series:
    """Band urine-output threshold margins into clinically readable bins."""

    values = pd.to_numeric(series, errors="coerce")
    return pd.cut(values, bins=[-np.inf, 250, 750, np.inf], labels=["0-250", "250-750", ">750"]).astype("string")

def _top_tertile_label(series: pd.Series) -> pd.Series:
    """Label the highest observed tertile of a numeric series."""

    values = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(pd.NA, index=series.index, dtype="string")
    finite = values.dropna()
    if finite.empty:
        return labels
    cutoff = finite.quantile(2 / 3)
    labels.loc[values.notna() & values.lt(cutoff)] = "no"
    labels.loc[values.notna() & values.ge(cutoff)] = "yes"
    return labels

def _any_binary_label(frame: pd.DataFrame) -> pd.Series:
    """Return yes/no when any binary column is positive, missing if all unknown."""

    values = frame.apply(pd.to_numeric, errors="coerce")
    known = values.notna().any(axis=1)
    exposed = values.fillna(0).gt(0).any(axis=1)
    labels = exposed.map({False: "no", True: "yes"}).astype("string")
    labels.loc[~known] = pd.NA
    return labels

def _slug(value: str) -> str:
    """Normalize display labels for output column names."""

    return value.lower().replace("-", "_").replace(" ", "_").replace(",", "")

def _safe_qcut(series: pd.Series, q: int) -> pd.Series:
    """Quantile-bin a series, returning all-missing labels when bins collapse."""

    values = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(values, q=q, duplicates="drop").astype("string")
    except ValueError:
        return pd.Series(pd.NA, index=series.index, dtype="string")
