"""Pre-treatment feature construction for the landmark CRRT experiment.

The feature layer deliberately separates static baseline covariates from
pre-treatment summaries of repeated measurements. This avoids forcing every
confounder into a time-varying form while still using time-zero information
when source measurements have repeated timestamps or intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import numpy as np
import pandas as pd

from .constants import (
    ADMISSION_AGE,
    DOD,
    EPISODE_START,
    HADM_ID,
    ICU_INTIME,
    INTERVENTION_STATUS,
    MORTALITY_TEMPLATE,
    STAY_ID,
    SUBJECT_ID,
)


# Mortality outcomes are copied through the analysis frame but excluded from
# model features later by the estimator module.
OUTCOME_COLUMNS = tuple(MORTALITY_TEMPLATE.format(days=day) for day in (28, 60, 90))

ANTIBIOTIC_NAME_PATTERNS = {
    "Glycopeptide": ("vancomycin",),
    "Beta-lactams": ("ceftriaxone", "cefotaxime", "cefepime"),
    "Carbapenems": ("meropenem",),
    "Aminoglycosides": ("gentamicin", "amikacin"),
}


@dataclass(frozen=True)
class AnalysisFrameResult:
    """Feature outputs used by the analysis and sensitivity paths."""

    analysis_frame: pd.DataFrame
    analysis_frame_without_fluid_balance: pd.DataFrame
    feature_metadata: pd.DataFrame


def build_analysis_frames(
    cohort: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
) -> AnalysisFrameResult:
    """Build model-ready analysis frames from the landmark cohort and sources."""

    # Begin with one-row-per-stay baseline information and treatment/outcome
    # columns. All later feature blocks merge onto these same identifiers.
    base = _static_baseline_features(cohort)
    metadata_rows: list[dict] = []

    features = base.copy()
    features["_audit_episode_crrt_primary_modality"] = features.pop("crrt_primary_modality") if "crrt_primary_modality" in features else np.nan
    features = _merge_features(features, latest_crrt_modality(cohort, sources.get("crrt_records", pd.DataFrame()),
                                                            sources.get("crrt", pd.DataFrame())))
    features = _merge_features(features, static_first_day_scores(cohort, sources))
    metadata_rows.extend({"feature_group": name, "source_type": "static_first_day_score",
                          "source_table": table, "worst_rule": "", "window_policy": "static_icu_stay_score"}
                         for name, table in [("baseline_sofa", "mimiciv_derived.first_day_sofa"),
                                             ("baseline_sapsii", "mimiciv_derived.sapsii")])
    metadata_rows.append({"feature_group": "crrt_primary_modality", "source_type": "latest_pre_tz_status",
                          "source_table": "crrt.crrt_mode / procedureevents.itemid", "worst_rule": "",
                          "window_policy": "index_episode_start <= record_time <= t0"})

    # Repeated numeric sources use these summaries: closest value at or
    # before time zero, mean from CRRT start to time zero, and a clinically
    # directed worst value where applicable.
    for name, frame in _numeric_feature_sources(sources).items():
        summary = summarize_numeric_events(
            cohort,
            frame=frame["frame"],
            value_col=frame["value_col"],
            time_col=frame["time_col"],
            prefix=name,
            worst=frame["worst"],
            join_key=frame.get("join_key"),
        )
        features = _merge_features(features, summary)
        metadata_rows.append(
            {
                "feature_group": name,
                "source_type": "time_stamped_numeric",
                "source_table": frame.get("source_table", ""),
                "worst_rule": frame["worst"],
            }
        )

    # Interval-like exposures, such as ventilation or vasopressors, are not
    # averaged as numbers; they are summarized as any/active/hours/count.
    interval_sources = _interval_feature_sources(sources)
    for name, frame in interval_sources.items():
        summary = summarize_interval_events(
            cohort,
            frame=frame["frame"],
            start_col=frame["start_col"],
            end_col=frame.get("end_col"),
            prefix=name,
            join_key=frame.get("join_key"),
        )
        if name == "ventilation":
            summary = cohort[[SUBJECT_ID, HADM_ID, STAY_ID]].merge(summary, how="left", on=[SUBJECT_ID, HADM_ID, STAY_ID])
            for col in ["any_pre_tz_ventilation", "active_at_tz_ventilation", "exposure_hours_start_to_tz_ventilation",
                        "event_count_start_to_tz_ventilation"]:
                summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0) if col in summary else 0
        features = _merge_features(features, summary)
        metadata_rows.append(
            {
                "feature_group": name,
                "source_type": "pre_treatment_interval",
                "source_table": frame.get("source_table", ""),
                "worst_rule": "",
            }
        )

    # Categorical/status data use the most recent pre-time-zero status. Model
    # preprocessing handles one-hot encoding later.
    categorical = summarize_categorical_status(
        cohort,
        frame=sources.get("rrt", pd.DataFrame()),
        value_col="dialysis_type",
        time_col="charttime",
        prefix="rrt_modality_status",
    )
    features = _merge_features(features, categorical)

    antibiotics = build_antibiotic_features(
        cohort,
        prescriptions=sources.get("prescriptions", pd.DataFrame()),
    )
    features = _merge_features(features, antibiotics)
    metadata_rows.append(
        {
            "feature_group": "antibiotics",
            "source_type": "pre_treatment_binary_exposure",
            "source_table": "mimiciv_hosp.prescriptions",
            "worst_rule": "",
        }
    )

    # Fluid balance summarizes inputs and outputs over the baseline window.
    fluid = build_fluid_balance_features(
        cohort,
        inputevents=sources.get("inputevents", pd.DataFrame()),
        outputevents=sources.get("outputevents", pd.DataFrame()),
        urine_output=sources.get("urine_output", pd.DataFrame()),
        crrt=sources.get("crrt", pd.DataFrame()),
    )
    features = _merge_features(features, fluid)
    metadata_rows.append(
        {
            "feature_group": "fluid_balance",
            "source_type": "fluid_balance_covariate",
            "source_table": "inputevents/outputevents/urine_output/crrt",
            "worst_rule": "positive net balance is worse",
        }
    )

    metadata_rows.extend({"feature_group": "input_volume_quality", "source_type": "audit_only", **row}
                         for row in fluid.attrs.get("input_audit", []))

    # The no-fluid-balance frame supports fluid covariate sensitivity analysis.
    no_fluid = features.drop(
        columns=[col for col in features.columns if "fluid_balance" in col or col.startswith("fluid_")],
        errors="ignore",
    )
    return AnalysisFrameResult(
        analysis_frame=features,
        analysis_frame_without_fluid_balance=no_fluid,
        feature_metadata=pd.DataFrame(metadata_rows),
    )


def static_first_day_scores(cohort, sources):
    """Join first-day scores by ICU stay without any score-time restriction."""
    out = cohort[[SUBJECT_ID, HADM_ID, STAY_ID]].copy()
    for source, value, feature in [("first_day_sofa", "sofa", "tz_last_baseline_sofa"),
                                   ("first_day_sapsii", "sapsii", "tz_last_baseline_sapsii")]:
        frame = sources.get(source, pd.DataFrame())
        if frame.empty or STAY_ID not in frame or value not in frame:
            out[feature] = np.nan
            continue
        score = frame[[STAY_ID, value]].copy()
        score[value] = pd.to_numeric(score[value], errors="coerce")
        score = score.drop_duplicates()
        if score[STAY_ID].duplicated().any():
            raise ValueError(f"Conflicting first-day scores in {source}; one value per stay is required.")
        out = out.merge(score.rename(columns={value: feature}), on=STAY_ID, how="left", validate="one_to_one")
    return out


def latest_crrt_modality(cohort, procedures, crrt):
    """Choose the closest valid CRRT modality record at or before t0."""
    from .cohort import CRRT_ITEMIDS, CRRT_MODALITY_LABELS
    keys = [SUBJECT_ID, HADM_ID, STAY_ID]
    candidates = []
    if not procedures.empty and {STAY_ID, "starttime", "itemid"}.issubset(procedures):
        events = procedures.loc[procedures.itemid.isin(CRRT_ITEMIDS), [STAY_ID, "starttime", "itemid"]].copy()
        events["record_time"] = pd.to_datetime(events.starttime, errors="coerce")
        events["modality"] = events.itemid.map(CRRT_MODALITY_LABELS)
        events["priority"] = events.modality.ne("CRRT").astype(int)
        events["source"] = "procedure_itemid"
        candidates.append(events[[STAY_ID, "record_time", "modality", "priority", "source"]])
    if not crrt.empty and {STAY_ID, "charttime", "crrt_mode"}.issubset(crrt):
        events = crrt[[STAY_ID, "charttime", "crrt_mode"]].copy()
        events["record_time"] = pd.to_datetime(events.charttime, errors="coerce")
        events["modality"] = events.crrt_mode.astype("string").str.strip()
        events = events.loc[events.modality.notna() & ~events.modality.str.casefold().isin(["", "none", "null", "nan"])]
        events["priority"], events["source"] = 2, "derived_crrt_mode"
        candidates.append(events[[STAY_ID, "record_time", "modality", "priority", "source"]])
    out = cohort[keys].copy()
    out["crrt_primary_modality"] = np.nan
    out["_audit_crrt_modality_record_time"] = pd.NaT
    out["_audit_crrt_modality_source"] = pd.NA
    if not candidates:
        return out
    events = pd.concat(candidates, ignore_index=True).merge(cohort[[STAY_ID, EPISODE_START, "time_zero"]], on=STAY_ID)
    events = events.loc[events.record_time.ge(events[EPISODE_START]) & events.record_time.le(events.time_zero)]
    latest = events.sort_values([STAY_ID, "record_time", "priority", "modality"], kind="stable").drop_duplicates(STAY_ID, keep="last")
    if latest.empty:
        return out
    latest = latest.rename(columns={"modality": "crrt_primary_modality", "record_time": "_audit_crrt_modality_record_time",
                                    "source": "_audit_crrt_modality_source"})
    return cohort[keys].merge(latest[[STAY_ID, "crrt_primary_modality", "_audit_crrt_modality_record_time",
                                    "_audit_crrt_modality_source"]], on=STAY_ID, how="left", validate="one_to_one")


def summarize_numeric_events(
    cohort: pd.DataFrame,
    frame: pd.DataFrame,
    value_col: str,
    time_col: str,
    prefix: str,
    worst: str = "max",
    join_key: str | None = None,
) -> pd.DataFrame:
    """Summarize repeated numeric measurements before or at time zero."""

    keys = [SUBJECT_ID, HADM_ID, STAY_ID]
    # Missing source tables should not break the pipeline; they simply produce
    # no new feature columns.
    if frame is None or frame.empty or value_col not in frame or time_col not in frame:
        return cohort[keys].copy()

    # Restrict measurements to the pre-treatment window from CRRT start through
    # time zero, inclusive.
    joined = _join_to_window(cohort, frame, time_col=time_col, join_key=join_key)
    if joined.empty:
        return cohort[keys].copy()
    joined[value_col] = pd.to_numeric(joined[value_col], errors="coerce")
    joined = joined.loc[joined[value_col].notna()].copy()
    if joined.empty:
        return cohort[keys].copy()

    # "last" after sorting by time is the value closest to time zero.
    grouped = joined.sort_values([SUBJECT_ID, HADM_ID, STAY_ID, time_col]).groupby(keys, dropna=False)
    out = grouped[value_col].agg(
        **{
            f"tz_last_{prefix}": "last",
            f"mean_start_to_tz_{prefix}": "mean",
        }
    ).reset_index()
    if worst == "min":
        # Lower values are clinically worse for variables such as MAP or SpO2.
        out[f"worst_start_to_tz_{prefix}"] = grouped[value_col].min().to_numpy()
    elif worst == "both":
        # Some physiologic variables can be harmful at either extreme, so keep
        # both minima and maxima rather than choosing a single direction.
        extrema = grouped[value_col].agg(["min", "max"]).reset_index(drop=True)
        out[f"min_start_to_tz_{prefix}"] = extrema["min"].to_numpy()
        out[f"max_start_to_tz_{prefix}"] = extrema["max"].to_numpy()
    else:
        # Higher values are clinically worse for variables such as lactate,
        # SOFA, or SAPSII.
        out[f"worst_start_to_tz_{prefix}"] = grouped[value_col].max().to_numpy()
    return out


def summarize_interval_events(
    cohort: pd.DataFrame,
    frame: pd.DataFrame,
    start_col: str,
    prefix: str,
    end_col: str | None = None,
    join_key: str | None = None,
) -> pd.DataFrame:
    """Summarize binary or interval exposures before time zero."""

    keys = [SUBJECT_ID, HADM_ID, STAY_ID]
    if frame is None or frame.empty or start_col not in frame:
        return cohort[keys].copy()
    working = frame.copy()
    if end_col is None or end_col not in working:
        # Point events are treated as zero-duration intervals for any/count
        # summaries while still allowing active_at_tz to be evaluated.
        end_col = "_event_end"
        working[end_col] = working[start_col]

    # Keep only intervals that overlap CRRT start through time zero.
    joined = _join_to_window(cohort, working, time_col=start_col, join_key=join_key, end_col=end_col)
    if joined.empty:
        return cohort[keys].copy()
    joined[start_col] = pd.to_datetime(joined[start_col], errors="coerce")
    joined[end_col] = pd.to_datetime(joined[end_col], errors="coerce").fillna(joined[start_col])

    # Clip each interval to the pre-treatment window before summing exposure
    # hours, so post-time-zero duration cannot leak into baseline covariates.
    joined["_overlap_start"] = joined[start_col].clip(lower=joined[EPISODE_START])
    joined["_overlap_end"] = joined[end_col].clip(upper=joined["time_zero"])
    joined["_exposure_hours"] = (
        (joined["_overlap_end"] - joined["_overlap_start"]).dt.total_seconds().clip(lower=0) / 3600.0
    )
    joined["_active_at_tz"] = (joined[start_col] <= joined["time_zero"]) & (joined[end_col] >= joined["time_zero"])
    grouped = joined.groupby(keys, dropna=False)
    return grouped.agg(
        **{
            f"any_pre_tz_{prefix}": (start_col, lambda s: int(s.notna().any())),
            f"active_at_tz_{prefix}": ("_active_at_tz", "max"),
            f"exposure_hours_start_to_tz_{prefix}": ("_exposure_hours", "sum"),
            f"event_count_start_to_tz_{prefix}": (start_col, "count"),
        }
    ).reset_index()


def summarize_categorical_status(
    cohort: pd.DataFrame,
    frame: pd.DataFrame,
    value_col: str,
    time_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Use the latest categorical status before time zero."""

    keys = [SUBJECT_ID, HADM_ID, STAY_ID]
    if frame is None or frame.empty or value_col not in frame or time_col not in frame:
        return cohort[keys].copy()
    joined = _join_to_window(cohort, frame, time_col=time_col, join_key=STAY_ID)
    if joined.empty:
        return cohort[keys].copy()

    # The most recent pre-treatment status is often the clinically interpretable
    # status at the decision point.
    latest = (
        joined.sort_values(keys + [time_col])
        .drop_duplicates(keys, keep="last")
        [keys + [value_col]]
        .rename(columns={value_col: f"latest_pre_tz_{prefix}"})
    )
    return latest


def build_fluid_balance_features(
    cohort: pd.DataFrame,
    inputevents: pd.DataFrame,
    outputevents: pd.DataFrame,
    urine_output: pd.DataFrame | None = None,
    crrt: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build transparent fluid-balance components.

    Net balance is input minus output. The component columns are kept so the
    definition can be audited and replaced if a site-specific fluid concept is
    preferred later.
    """

    keys = [SUBJECT_ID, HADM_ID, STAY_ID]
    out = cohort[keys].copy()

    # Inputs and outputs can live in separate MIMIC tables. Each component is
    # summarized over the same CRRT-start-to-time-zero window.
    from .fluids import input_volumes
    input_sums, input_audit = input_volumes(cohort, inputevents)
    output_sums = _sum_events_in_windows(cohort, outputevents, "value", "charttime", "fluid_outputevents_output_ml")
    urine_sums = _sum_events_in_windows(cohort, urine_output, "urineoutput", "charttime", "fluid_derived_urine_output_ml")
    crrt_removal = _crrt_removal_sums(cohort, crrt)

    for part in [input_sums, output_sums, urine_sums, crrt_removal]:
        out = _merge_features(out, part)

    # Create absent components explicitly so the net-balance formula has a
    # stable schema across the supported source tables.
    for col in [
        "fluid_input_ml_start_to_tz",
        "fluid_outputevents_output_ml_start_to_tz",
        "fluid_crrt_removal_ml_start_to_tz",
    ]:
        if col not in out:
            out[col] = np.nan

    # Conservative definition: net input minus documented non-urine outputevents
    # and CRRT fluid removal. Derived urine output is retained separately because
    # it is already central to time-zero definition and may be used in sensitivity
    # checks before adding it to the main net-balance formula.
    out["net_fluid_balance_ml_start_to_tz"] = (
        out["fluid_input_ml_start_to_tz"].fillna(0)
        - out["fluid_outputevents_output_ml_start_to_tz"].fillna(0)
        - out["fluid_crrt_removal_ml_start_to_tz"].fillna(0)
    )
    out["positive_fluid_balance_start_to_tz"] = (out["net_fluid_balance_ml_start_to_tz"] > 0).astype(int)
    out.attrs["input_audit"] = input_audit.to_dict("records")
    return out


def build_antibiotic_features(cohort: pd.DataFrame, prescriptions: pd.DataFrame | None) -> pd.DataFrame:
    """Create binary pre-time-zero antibiotic class exposures from prescriptions."""

    keys = [SUBJECT_ID, HADM_ID, STAY_ID]
    out = cohort[keys].copy()
    for drug_class in ANTIBIOTIC_NAME_PATTERNS:
        out[drug_class] = np.nan

    if prescriptions is None:
        return out
    required = {"drug", "starttime"}
    if not required.issubset(prescriptions.columns):
        return out

    # A successful but empty source extract means no observed class exposure in
    # the cohort; a failed/missing extract lacks the required columns and stays
    # all-missing above.
    for drug_class in ANTIBIOTIC_NAME_PATTERNS:
        out[drug_class] = 0
    if prescriptions.empty:
        return out

    end_col = "stoptime" if "stoptime" in prescriptions else None
    joined = _join_to_window(
        cohort,
        prescriptions,
        time_col="starttime",
        join_key=HADM_ID if HADM_ID in prescriptions else None,
        end_col=end_col,
    )
    if joined.empty:
        return out

    drug_text = joined["drug"].fillna("").astype(str).str.lower()
    for drug_class, names in ANTIBIOTIC_NAME_PATTERNS.items():
        pattern = "|".join(re.escape(name) for name in names)
        exposed = joined.loc[drug_text.str.contains(pattern, regex=True, na=False), keys].drop_duplicates()
        if exposed.empty:
            continue
        marker = f"_{drug_class}_exposed"
        out = out.merge(exposed.assign(**{marker: 1}), on=keys, how="left")
        out[drug_class] = out.pop(marker).fillna(out[drug_class]).astype(int)
    return out


def _static_baseline_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """Copy static baseline covariates without forcing time-varying summaries."""

    keys = [SUBJECT_ID, HADM_ID, STAY_ID]
    # These columns are either identifiers, treatment/outcome definitions, or
    # one-row baseline fields that should pass through unchanged.
    keep = [
        SUBJECT_ID,
        HADM_ID,
        STAY_ID,
        INTERVENTION_STATUS,
        "early_discontinuation",
        "continued_crrt_at_landmark",
        "treatment_classifiable",
        "treatment_label",
        "landmark_hours",
        "time_zero",
        "landmark_time",
        EPISODE_START,
        "time_zero_urineoutput_24hr",
        "time_zero_uo_tm_24hr",
        "time_zero_threshold_ml_24hr",
        "threshold_margin_ml_24hr",
        "diuretic_prior_24h",
        "crrt_primary_modality",
        "anchor_year_group",
        ADMISSION_AGE,
        DOD,
        ICU_INTIME,
        *OUTCOME_COLUMNS,
    ]
    frame = cohort[[col for col in keep if col in cohort]].copy()

    # Recreate common binary baseline covariates used by the reproduction script
    # when the raw categorical fields are available.
    if "gender" in cohort and "Female" not in frame:
        frame["Female"] = (cohort["gender"] == "F").astype(int)
    if "race" in cohort and "White" not in frame:
        frame["White"] = cohort["race"].fillna("").str.lower().str.contains("white").astype(int)
    if "admission_type" in cohort and "Emergency admission" not in frame:
        frame["Emergency admission"] = cohort["admission_type"].isin(["DIRECT EMER.", "EW EMER."]).astype(int)
    if "insurance" in cohort and "Insurance, Medicare" not in frame:
        frame["Insurance, Medicare"] = cohort["insurance"].isin(["Medicare", "Medicaid"]).astype(int)
    for col in ["Female", "White", "Emergency admission", "Insurance, Medicare"]:
        if col in cohort and col not in frame:
            frame[col] = cohort[col]
    return frame.drop_duplicates(keys)


def _numeric_feature_sources(sources: Mapping[str, pd.DataFrame]) -> dict[str, dict]:
    """Declare numeric repeated-measurement sources and their worst directions."""

    out: dict[str, dict] = {}
    if "vitalsign" in sources:
        # Vital signs can be harmful in different directions, so each variable
        # explicitly declares how worst-case summaries should be constructed.
        vital_specs = {
            "mbp": "min",
            "spo2": "min",
            "heart_rate": "both",
            "resp_rate": "both",
            "temperature": "both",
        }
        for col, worst in vital_specs.items():
            if col in sources["vitalsign"]:
                out[col] = {
                    "frame": sources["vitalsign"],
                    "value_col": col,
                    "time_col": "charttime",
                    "worst": worst,
                    "source_table": "mimiciv_derived.vitalsign",
                }
    if "bg" in sources and "lactate" in sources["bg"]:
        # Lactate is a hospital-admission-level lab in many MIMIC tables, so use
        # hadm_id when stay_id is unavailable.
        out["lactate"] = {
            "frame": sources["bg"],
            "value_col": "lactate",
            "time_col": "charttime",
            "worst": "max",
            "join_key": HADM_ID if HADM_ID in sources["bg"] else None,
            "source_table": "mimiciv_derived.bg",
        }
    if "kdigo_uo" in sources and "urineoutput_24hr" in sources["kdigo_uo"]:
        # Urine output remains useful as a baseline severity/renal recovery
        # feature after it has defined eligibility.
        out["urineoutput_24hr"] = {
            "frame": sources["kdigo_uo"],
            "value_col": "urineoutput_24hr",
            "time_col": "charttime",
            "worst": "min",
            "source_table": "mimiciv_derived.kdigo_uo",
        }
    return out


def _interval_feature_sources(sources: Mapping[str, pd.DataFrame]) -> dict[str, dict]:
    """Declare interval or point-event sources for pre-treatment exposure summaries."""

    out: dict[str, dict] = {}
    if "diuretics" in sources:
        # Diuretics are included as a covariate even though they also affect the
        # time-zero threshold, because recent use may proxy clinician intent.
        out["diuretic"] = {
            "frame": sources["diuretics"],
            "start_col": "starttime",
            "end_col": "endtime",
            "source_table": "mimiciv_icu.inputevents",
        }
    if "ventilation" in sources:
        ventilation = sources["ventilation"].copy()
        if not ventilation.empty:
            if "ventilation_status" not in ventilation:
                raise ValueError("ventilation_status is required to distinguish None/missing labels from exposure.")
            status = ventilation.ventilation_status.astype("string").str.strip().str.casefold()
            ventilation = ventilation.loc[status.notna() & ~status.isin(["none", "null", "nan", ""])].copy()
        out["ventilation"] = {
            "frame": ventilation,
            "start_col": "starttime",
            "end_col": "endtime",
            "source_table": "mimiciv_derived.ventilation",
        }
    if "suspected_infection" in sources:
        out["suspected_infection"] = {
            "frame": sources["suspected_infection"],
            "start_col": "suspected_infection_time",
            "end_col": "suspected_infection_time",
            "join_key": HADM_ID,
            "source_table": "mimiciv_derived.suspicion_of_infection",
        }
    vaso = _prepare_vasopressor_intervals(sources.get("vasoactive_agent", pd.DataFrame()))
    if not vaso.empty:
        # Vasoactive-agent rows are first reduced to intervals with any positive
        # agent dose, then summarized as an exposure.
        out["vasopressor"] = {
            "frame": vaso,
            "start_col": "starttime",
            "end_col": "endtime",
            "source_table": "mimiciv_derived.vasoactive_agent",
        }
    return out


def _prepare_vasopressor_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with at least one positive vasoactive-agent dose."""

    if frame is None or frame.empty:
        return pd.DataFrame()
    vaso_cols = [
        "dopamine",
        "epinephrine",
        "norepinephrine",
        "phenylephrine",
        "vasopressin",
        "dobutamine",
        "milrinone",
    ]
    existing = [col for col in vaso_cols if col in frame]
    if not existing:
        return pd.DataFrame()
    out = frame.copy()
    # Any listed agent with a positive dose marks vasopressor exposure.
    out["_any_vaso"] = out[existing].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0).any(axis=1)
    return out.loc[out["_any_vaso"]].copy()


def _join_to_window(
    cohort: pd.DataFrame,
    frame: pd.DataFrame,
    time_col: str,
    join_key: str | None = None,
    end_col: str | None = None,
) -> pd.DataFrame:
    """Attach cohort landmarks to event rows and keep only pre-time-zero data."""

    working = frame.copy()
    working[time_col] = pd.to_datetime(working[time_col], errors="coerce")
    if end_col is not None and end_col in working:
        working[end_col] = pd.to_datetime(working[end_col], errors="coerce").fillna(working[time_col])
    key = join_key or _choose_join_key(working)
    if key is None:
        return pd.DataFrame()

    # Keep the lookup schema unique because the chosen key may also be one of
    # subject_id/hadm_id/stay_id.
    lookup_cols = [key, SUBJECT_ID, HADM_ID, STAY_ID, EPISODE_START, "time_zero"]
    lookup_cols = list(dict.fromkeys(col for col in lookup_cols if col in cohort))
    joined = working.merge(cohort[lookup_cols].drop_duplicates(), on=key, how="inner", suffixes=("", "_cohort"))
    for col in [SUBJECT_ID, HADM_ID, STAY_ID]:
        # When an event table joins by hadm_id or subject_id, fill missing
        # stay-level identifiers from the landmark cohort.
        cohort_col = f"{col}_cohort"
        if cohort_col in joined:
            joined[col] = joined[col].fillna(joined[cohort_col]) if col in joined else joined[cohort_col]
    if end_col is not None and end_col in joined:
        # Interval rows are retained if any part of the interval overlaps the
        # pre-treatment window.
        mask = (joined[time_col] <= joined["time_zero"]) & (joined[end_col] >= joined[EPISODE_START])
    else:
        # Point rows must fall directly inside the pre-treatment window.
        mask = (joined[time_col] >= joined[EPISODE_START]) & (joined[time_col] <= joined["time_zero"])
    return joined.loc[mask].copy()


def _choose_join_key(frame: pd.DataFrame) -> str | None:
    """Choose the most specific available identifier for joining event tables."""

    for key in [STAY_ID, HADM_ID, SUBJECT_ID]:
        if key in frame:
            return key
    return None


def _sum_events_in_windows(
    cohort: pd.DataFrame,
    frame: pd.DataFrame | None,
    value_col: str,
    time_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Sum fluid-like event values over CRRT-start-to-time-zero and pre-24h windows."""

    keys = [SUBJECT_ID, HADM_ID, STAY_ID]
    if frame is None or frame.empty or value_col not in frame or time_col not in frame:
        return cohort[keys].copy()
    joined = _join_to_window(cohort, frame, time_col=time_col)
    if joined.empty:
        return cohort[keys].copy()
    joined[value_col] = pd.to_numeric(joined[value_col], errors="coerce")
    # The pre-24h column is useful for sensitivity analyses around the
    # discontinuation decision even when the main feature uses the full window.
    joined["_pre24"] = joined[time_col] >= (joined["time_zero"] - pd.Timedelta(hours=24))
    grouped = joined.groupby(keys, dropna=False)
    totals = grouped[value_col].sum(min_count=1).rename(f"{prefix}_start_to_tz").reset_index()
    pre24 = (
        joined.loc[joined["_pre24"]]
        .groupby(keys, dropna=False)[value_col]
        .sum(min_count=1)
        .rename(f"{prefix}_pre24h")
        .reset_index()
    )
    return totals.merge(pre24, on=keys, how="left")


def _crrt_removal_sums(cohort: pd.DataFrame, crrt: pd.DataFrame | None) -> pd.DataFrame:
    """Summarize CRRT fluid removal when derived CRRT rows expose removal fields."""

    keys = [SUBJECT_ID, HADM_ID, STAY_ID]
    if crrt is None or crrt.empty:
        return cohort[keys].copy()
    value_cols = [col for col in ["hourly_patient_fluid_removal", "ultrafiltrate_output"] if col in crrt]
    if not value_cols or "charttime" not in crrt:
        return cohort[keys].copy()
    tmp = crrt.copy()
    # MIMIC-derived CRRT tables may expose different removal column names; sum
    # whichever are available rather than failing on a missing optional field.
    tmp["crrt_removal_value"] = tmp[value_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    return _sum_events_in_windows(cohort, tmp, "crrt_removal_value", "charttime", "fluid_crrt_removal_ml")


def _merge_features(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Left-merge a feature block onto the one-row-per-stay analysis frame."""

    keys = [SUBJECT_ID, HADM_ID, STAY_ID]
    if right is None or right.empty:
        return left
    payload_cols = [col for col in right.columns if col not in keys]
    if not payload_cols:
        return left
    return left.merge(right[keys + payload_cols].drop_duplicates(keys), on=keys, how="left")
