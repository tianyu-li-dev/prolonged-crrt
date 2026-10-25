"""Cohort, time-zero, treatment, and outcome construction.

This module defines the target-trial style cohort before any causal model is
fit. The important design choice is that all eligibility, treatment, and
outcome definitions are anchored on a landmark time: the first active-CRRT
time when urine output reaches the discontinuation threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    ADMISSION_AGE,
    DOD,
    END_TIME,
    EPISODE_END,
    EPISODE_ID,
    EPISODE_START,
    GAP_FROM_PREVIOUS_EPISODE,
    HADM_ID,
    HAS_MULTIPLE_EPISODES,
    ICU_INTIME,
    INTERVENTION_STATUS,
    MORTALITY_TEMPLATE,
    N_EPISODES_FOR_STAY,
    START_TIME,
    STAY_ID,
    SUBJECT_ID,
)


DEFAULT_LANDMARK_HOURS = 24.0

# MIMIC-IV procedureevents itemids used by the official RRT extension for CRRT
# and closely related continuous modalities.
CRRT_ITEMIDS = (225802, 225803, 225809, 225955)

# Inputevents itemids for loop diuretics that change the urine-output threshold.
DIURETIC_ITEMIDS = (228340, 221794, 229639)

# Katayama-style CRRT discontinuation thresholds: a higher urine-output
# threshold is required when loop diuretics were recently administered.
DIURETIC_THRESHOLD_ML_24HR = 2000.0
NO_DIURETIC_THRESHOLD_ML_24HR = 500.0
DIURETIC_WINDOW_HOURS = 24.0

# Human-readable modality names are kept for diagnostics and CATE summaries.
CRRT_MODALITY_LABELS = {
    225802: "CRRT",
    225803: "CVVHD",
    225809: "CVVHDF",
    225955: "SCUF",
}


@dataclass(frozen=True)
class LandmarkCohortResult:
    """Named outputs from the cohort stage.

    Keeping diagnostics next to the cohort makes it harder to silently lose
    sample-size reductions, multi-episode patterns, or pre-landmark deaths.
    """

    cohort: pd.DataFrame
    episodes: pd.DataFrame
    cohort_flow: pd.DataFrame
    multi_episode_summary: pd.DataFrame
    episode_gap_distribution: pd.DataFrame
    deaths_timezero_to_landmark: pd.DataFrame
    threshold_exclusion_summary: pd.DataFrame
    threshold_exclusion_records: pd.DataFrame


@dataclass(frozen=True)
class TimeZeroEvaluation:
    """Time-zero successes plus threshold-related exclusions."""

    time_zero: pd.DataFrame
    threshold_exclusion_summary: pd.DataFrame
    threshold_exclusion_records: pd.DataFrame


def build_landmark_cohort(
    stays: pd.DataFrame,
    crrt_records: pd.DataFrame,
    urine: pd.DataFrame,
    diuretics: pd.DataFrame | None = None,
    merge_gap_hours: float = 6.0,
    landmark_hours: float = DEFAULT_LANDMARK_HOURS,
    uo_threshold_no_diuretic: float = NO_DIURETIC_THRESHOLD_ML_24HR,
    uo_threshold_diuretic: float = DIURETIC_THRESHOLD_ML_24HR,
    outcome_horizons_days: Iterable[int] = (28, 60, 90),
) -> LandmarkCohortResult:
    """Build the landmark cohort from source tables.

    ``intervention_status`` is coded as 1 for receiving CRRT at the landmark
    and 0 for early discontinuation before the landmark.
    """

    flow = CohortFlow()

    # Start from ICU stays rather than CRRT rows so every cohort-reducing
    # patient/stay restriction can be recorded in a CONSORT-style flow table.
    base = _prepare_stays(stays)
    flow.add("raw ICU stays", len(base), len(base))

    # Restrict to adult target population. If the source table lacks age, the
    # helper leaves the frame unchanged so external cohorts can still run.
    base = _filter_if_present(base, ADMISSION_AGE, lambda s: s >= 18)
    flow.add("adult patients", flow.current_n, len(base))

    # Prefer the MIMIC-derived first_icu_stay flag when it is available.
    if "first_icu_stay" in base:
        base = base.loc[base["first_icu_stay"].fillna(False).astype(bool)].copy()
        flow.add("first ICU stay flag", flow.current_n, len(base))

    # Enforce one record per patient. When multiple ICU stays remain, the
    # earliest ICU admission is kept to avoid within-patient dependence.
    base = _keep_earliest_stay_per_subject(base)
    flow.add("one earliest ICU stay per patient", flow.current_n, len(base))

    # Exclude stays too short to observe the landmark design meaningfully.
    los_col = "los_icu" if "los_icu" in base else "los" if "los" in base else None
    if los_col is not None:
        base = base.loc[pd.to_numeric(base[los_col], errors="coerce") >= 1.0].copy()
        flow.add("ICU LOS at least 1 day", flow.current_n, len(base))

    # Exclude deaths during the first 24 hours after ICU admission.
    if DOD in base and ICU_INTIME in base:
        alive_one_day = base[DOD].isna() | ((base[DOD] - base[ICU_INTIME]) >= pd.Timedelta(days=1))
        base = base.loc[alive_one_day].copy()
        flow.add("exclude death within 24h of ICU admission", flow.current_n, len(base))

    # Collapse raw procedure rows into clinically interpretable CRRT episodes.
    # The configured merge gap only joins adjacent records from the same stay.
    episodes = build_crrt_episodes(crrt_records, merge_gap_hours=merge_gap_hours)

    # Exposure is defined only from the first episode in each ICU stay. Later
    # episodes are retained in the diagnostics but do not define treatment.
    index_episodes = select_first_episode(episodes)
    cohort = base.merge(index_episodes, on=STAY_ID, how="inner", suffixes=("", "_episode"))
    flow.add("has qualifying CRRT episode", flow.current_n, len(cohort))

    # Time-zero and treatment classification require a documented end for the
    # index episode; records with open-ended CRRT cannot be classified reliably.
    cohort = cohort.loc[cohort[EPISODE_END].notna()].copy()
    flow.add("documented CRRT episode end", flow.current_n, len(cohort))

    # Time zero is the first urine-output threshold crossing during active CRRT.
    # Urine recorded exactly at the episode end is included.
    # The main rule requires urineoutput_24hr to be documented; uo_tm_24hr is
    # carried forward only as a diagnostic/sensitivity variable.
    time_zero_result = evaluate_time_zero(
        cohort,
        urine=urine,
        diuretics=diuretics,
        uo_threshold_no_diuretic=uo_threshold_no_diuretic,
        uo_threshold_diuretic=uo_threshold_diuretic,
    )
    n_no_documented = _threshold_reason_count(
        time_zero_result.threshold_exclusion_summary,
        "no_documented_urineoutput_24hr_during_active_crrt",
    )
    n_never_met = _threshold_reason_count(
        time_zero_result.threshold_exclusion_summary,
        "documented_urineoutput_but_never_met_threshold",
    )
    flow.add("documented urineoutput_24hr during active index CRRT", flow.current_n, flow.current_n - n_no_documented)
    flow.add("met configured urine-output threshold before CRRT stop", flow.current_n, flow.current_n - n_never_met)

    time_zero = time_zero_result.time_zero
    cohort = cohort.merge(
        time_zero,
        on=[SUBJECT_ID, HADM_ID, STAY_ID, EPISODE_ID],
        how="inner",
        suffixes=("", "_time_zero"),
    )
    flow.current_n = int(len(cohort))

    # Patients must be alive at time zero to enter the landmark target trial.
    if DOD in cohort:
        alive_at_tz = cohort[DOD].isna() | (cohort[DOD] > cohort["time_zero"])
        cohort = cohort.loc[alive_at_tz].copy()
        flow.add("alive at time zero", flow.current_n, len(cohort))

    # Treatment is assigned using CRRT status exactly at tL, where
    # tL = time_zero + landmark_hours. This supports sensitivity analyses with
    # 6h, 12h, 24h, or 48h landmarks without changing the estimators.
    cohort["landmark_hours"] = float(landmark_hours)
    cohort["landmark_time"] = cohort["time_zero"] + pd.to_timedelta(float(landmark_hours), unit="h")
    treatment = classify_treatment_at_landmark(cohort, episodes)
    cohort = cohort.merge(treatment, on=[SUBJECT_ID, HADM_ID, STAY_ID, EPISODE_ID], how="left")
    cohort = cohort.loc[cohort["treatment_classifiable"].fillna(False)].copy()
    cohort[INTERVENTION_STATUS] = cohort["continued_crrt_at_landmark"].astype(int)
    cohort["treatment_label"] = np.where(
        cohort[INTERVENTION_STATUS] == 1,
        "continued_crrt_at_landmark",
        "early_discontinuation",
    )
    flow.add("classifiable CRRT status at landmark", flow.current_n, len(cohort))

    # Deaths between time zero and the landmark are counted for reporting but
    # excluded from causal estimation because treatment status is not fully
    # observable through the landmark interval.
    deaths_to_landmark = pd.DataFrame()
    if DOD in cohort and not cohort.empty:
        dead_before_landmark = (
            cohort[DOD].notna()
            & (cohort[DOD] > cohort["time_zero"])
            & (cohort[DOD] <= cohort["landmark_time"])
        )
        deaths_to_landmark = cohort.loc[
            dead_before_landmark,
            [col for col in [SUBJECT_ID, HADM_ID, STAY_ID, DOD, "time_zero", "landmark_time"] if col in cohort],
        ].copy()
        cohort = cohort.loc[~dead_before_landmark].copy()
        flow.add("exclude death between time zero and landmark", flow.current_n, len(cohort))

    # Outcomes are counted from landmark_time rather than from ICU admission or
    # CRRT start, preserving the landmark interpretation.
    cohort = add_landmark_mortality_outcomes(cohort, outcome_horizons_days)

    return LandmarkCohortResult(
        cohort=cohort.reset_index(drop=True),
        episodes=episodes.reset_index(drop=True),
        cohort_flow=flow.to_frame(),
        multi_episode_summary=summarize_multi_episode_patterns(episodes),
        episode_gap_distribution=episode_gap_distribution(episodes),
        deaths_timezero_to_landmark=deaths_to_landmark.reset_index(drop=True),
        threshold_exclusion_summary=time_zero_result.threshold_exclusion_summary.reset_index(drop=True),
        threshold_exclusion_records=time_zero_result.threshold_exclusion_records.reset_index(drop=True),
    )




def build_crrt_episodes(crrt_records: pd.DataFrame, merge_gap_hours: float = 6.0) -> pd.DataFrame:
    """Merge adjacent CRRT procedure records into stay-level episodes.

    A short gap can reflect charting fragmentation rather than a real clinical
    stop. A longer gap is treated as a distinct CRRT episode and reported.
    """

    records = crrt_records.copy()
    if records.empty:
        return _empty_episodes()
    # Include all four CRRT-related itemids rather than only the generic CRRT
    # code, matching the MIMIC extension design.
    records = records.loc[records["itemid"].isin(CRRT_ITEMIDS)].copy()
    if records.empty:
        return _empty_episodes()
    records[START_TIME] = pd.to_datetime(records[START_TIME], errors="coerce")
    records[END_TIME] = pd.to_datetime(records[END_TIME], errors="coerce")
    records = records.loc[records[STAY_ID].notna() & records[START_TIME].notna()].copy()
    records["crrt_modality"] = records["itemid"].map(CRRT_MODALITY_LABELS).fillna(records["itemid"].astype(str))

    rows: list[dict] = []
    for stay_id, group in records.sort_values([STAY_ID, START_TIME]).groupby(STAY_ID, sort=False):
        # These variables track the episode currently being assembled inside a
        # stay. They reset only when the inter-record gap exceeds merge_gap_hours.
        current_records: list[pd.Series] = []
        current_start = None
        current_end_for_gap = None
        previous_episode_end = pd.NaT
        episode_id = 0

        for _, record in group.iterrows():
            start = record[START_TIME]
            end = record[END_TIME]
            # Open-ended records use the start time for gap calculations so they
            # do not accidentally absorb all later records.
            end_for_gap = end if pd.notna(end) else start
            start_new = not current_records
            if current_records and pd.notna(current_end_for_gap):
                gap = start - current_end_for_gap
                start_new = gap > pd.Timedelta(hours=merge_gap_hours)
            if start_new:
                # Close the previous episode before starting a new one.
                if current_records:
                    episode_id += 1
                    rows.append(_episode_row(stay_id, episode_id, current_records, previous_episode_end))
                    previous_episode_end = _max_timestamp([rec[END_TIME] for rec in current_records])
                current_records = [record]
                current_start = start
                current_end_for_gap = end_for_gap
            else:
                # Continue the current episode and extend its observed end used
                # for the next gap calculation.
                current_records.append(record)
                current_end_for_gap = max(current_end_for_gap, end_for_gap)
                current_start = min(current_start, start)

        if current_records:
            episode_id += 1
            rows.append(_episode_row(stay_id, episode_id, current_records, previous_episode_end))

    episodes = pd.DataFrame(rows)
    if episodes.empty:
        return _empty_episodes()

    # Add stay-level episode counts back onto every episode row for diagnostics.
    episode_counts = episodes.groupby(STAY_ID)[EPISODE_ID].transform("max")
    episodes[N_EPISODES_FOR_STAY] = episode_counts.astype(int)
    episodes[HAS_MULTIPLE_EPISODES] = episodes[N_EPISODES_FOR_STAY] > 1
    return episodes.sort_values([STAY_ID, EPISODE_ID]).reset_index(drop=True)


def select_first_episode(episodes: pd.DataFrame) -> pd.DataFrame:
    """Return the first CRRT episode per stay."""

    if episodes.empty:
        return episodes.copy()
    # Sorting by episode start is more robust than assuming episode_id order
    # survives every upstream transformation.
    return (
        episodes.sort_values([STAY_ID, EPISODE_START])
        .drop_duplicates(STAY_ID, keep="first")
        .reset_index(drop=True)
    )


def find_time_zero(
    index_episodes: pd.DataFrame,
    urine: pd.DataFrame,
    diuretics: pd.DataFrame | None = None,
    uo_threshold_no_diuretic: float = NO_DIURETIC_THRESHOLD_ML_24HR,
    uo_threshold_diuretic: float = DIURETIC_THRESHOLD_ML_24HR,
) -> pd.DataFrame:
    """Find the first urine-threshold crossing during active index CRRT."""

    return evaluate_time_zero(
        index_episodes,
        urine=urine,
        diuretics=diuretics,
        uo_threshold_no_diuretic=uo_threshold_no_diuretic,
        uo_threshold_diuretic=uo_threshold_diuretic,
    ).time_zero


def evaluate_time_zero(
    index_episodes: pd.DataFrame,
    urine: pd.DataFrame,
    diuretics: pd.DataFrame | None = None,
    uo_threshold_no_diuretic: float = NO_DIURETIC_THRESHOLD_ML_24HR,
    uo_threshold_diuretic: float = DIURETIC_THRESHOLD_ML_24HR,
) -> TimeZeroEvaluation:
    """Find time zero and record threshold-related cohort exclusions."""

    base_cols = [SUBJECT_ID, HADM_ID, STAY_ID, EPISODE_ID, EPISODE_START, EPISODE_END, "crrt_primary_modality"]
    if index_episodes.empty or urine.empty:
        records = _threshold_exclusion_records(index_episodes, None, "no_documented_urineoutput_24hr_during_active_crrt")
        return TimeZeroEvaluation(
            time_zero=_empty_time_zero(),
            threshold_exclusion_summary=_threshold_exclusion_summary(records),
            threshold_exclusion_records=records,
        )

    # Only the selected index episode can generate time zero for a stay.
    episodes = index_episodes[[col for col in base_cols if col in index_episodes]].copy()
    uo = urine.copy()
    if "charttime" not in uo or "urineoutput_24hr" not in uo:
        records = _threshold_exclusion_records(episodes, None, "no_documented_urineoutput_24hr_during_active_crrt")
        return TimeZeroEvaluation(
            time_zero=_empty_time_zero(),
            threshold_exclusion_summary=_threshold_exclusion_summary(records),
            threshold_exclusion_records=records,
        )
    uo["charttime"] = pd.to_datetime(uo["charttime"], errors="coerce")
    uo["urineoutput_24hr"] = pd.to_numeric(uo["urineoutput_24hr"], errors="coerce")
    if "uo_tm_24hr" in uo:
        uo["uo_tm_24hr"] = pd.to_numeric(uo["uo_tm_24hr"], errors="coerce")
    else:
        uo["uo_tm_24hr"] = np.nan

    # Candidate time-zero records must be documented urine-output records that
    # occur while the index CRRT episode is active.
    candidates = uo.merge(episodes, on=STAY_ID, how="inner", suffixes=("", "_episode"))
    for col in [SUBJECT_ID, HADM_ID]:
        episode_col = f"{col}_episode"
        if episode_col in candidates:
            candidates[col] = candidates[col].fillna(candidates[episode_col]) if col in candidates else candidates[episode_col]
    candidates = candidates.loc[
        candidates["charttime"].notna()
        & candidates["urineoutput_24hr"].notna()
        & (candidates["charttime"] >= candidates[EPISODE_START])
        & (candidates["charttime"] <= candidates[EPISODE_END])
    ].copy()

    if not candidates.empty:
        # Diuretic exposure is evaluated inside the fixed 24 hours before each
        # candidate urine-output timestamp and controls the threshold used at
        # that timestamp.
        candidates["diuretic_prior_24h"] = _diuretic_prior_24h(candidates, diuretics)
        candidates["time_zero_threshold_ml_24hr"] = np.where(
            candidates["diuretic_prior_24h"],
            float(uo_threshold_diuretic),
            float(uo_threshold_no_diuretic),
        )
        candidates["threshold_margin_ml_24hr"] = (
            candidates["urineoutput_24hr"] - candidates["time_zero_threshold_ml_24hr"]
        )

    # The first non-negative margin is the earliest threshold crossing.
    eligible = candidates.loc[candidates["threshold_margin_ml_24hr"] >= 0].copy() if not candidates.empty else candidates.copy()
    if eligible.empty:
        no_doc_records = _threshold_exclusion_records(
            episodes,
            candidates,
            "no_documented_urineoutput_24hr_during_active_crrt",
            only_missing_documentation=True,
        )
        never_met_records = _threshold_exclusion_records(
            episodes,
            candidates,
            "documented_urineoutput_but_never_met_threshold",
            only_with_documentation=True,
        )
        records = pd.concat([no_doc_records, never_met_records], ignore_index=True)
        return TimeZeroEvaluation(
            time_zero=_empty_time_zero(),
            threshold_exclusion_summary=_threshold_exclusion_summary(records),
            threshold_exclusion_records=records,
        )

    eligible = eligible.sort_values([STAY_ID, "charttime"])
    first = eligible.drop_duplicates(STAY_ID, keep="first").copy()
    first = first.rename(
        columns={
            "charttime": "time_zero",
            "urineoutput_24hr": "time_zero_urineoutput_24hr",
            "uo_tm_24hr": "time_zero_uo_tm_24hr",
        }
    )
    keep_cols = [
        SUBJECT_ID,
        HADM_ID,
        STAY_ID,
        EPISODE_ID,
        "time_zero",
        "time_zero_urineoutput_24hr",
        "time_zero_uo_tm_24hr",
        "diuretic_prior_24h",
        "time_zero_threshold_ml_24hr",
        "threshold_margin_ml_24hr",
        "crrt_primary_modality",
    ]
    time_zero = first[[col for col in keep_cols if col in first]].reset_index(drop=True)

    eligible_keys = time_zero[[STAY_ID]].drop_duplicates()
    remaining_episodes = episodes.merge(eligible_keys.assign(_has_time_zero=True), on=STAY_ID, how="left")
    remaining_episodes = remaining_episodes.loc[remaining_episodes["_has_time_zero"].isna()].drop(columns="_has_time_zero")
    no_doc_records = _threshold_exclusion_records(
        remaining_episodes,
        candidates,
        "no_documented_urineoutput_24hr_during_active_crrt",
        only_missing_documentation=True,
    )
    never_met_records = _threshold_exclusion_records(
        remaining_episodes,
        candidates,
        "documented_urineoutput_but_never_met_threshold",
        only_with_documentation=True,
    )
    records = pd.concat([no_doc_records, never_met_records], ignore_index=True)
    return TimeZeroEvaluation(
        time_zero=time_zero,
        threshold_exclusion_summary=_threshold_exclusion_summary(records),
        threshold_exclusion_records=records,
    )


def classify_treatment_at_landmark(cohort: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Classify treatment by whether any CRRT episode is active at tL."""

    keys = [SUBJECT_ID, HADM_ID, STAY_ID, EPISODE_ID]
    if cohort.empty:
        return pd.DataFrame(columns=keys + ["early_discontinuation", "continued_crrt_at_landmark", "treatment_classifiable"])
    episodes_by_stay = {stay_id: group for stay_id, group in episodes.groupby(STAY_ID)} if not episodes.empty else {}
    rows = []
    for _, row in cohort.iterrows():
        landmark_time = row["landmark_time"]
        group = episodes_by_stay.get(row[STAY_ID], pd.DataFrame())
        treatment_classifiable = True
        continued = False
        if group.empty or pd.isna(landmark_time):
            treatment_classifiable = False
        else:
            starts = pd.to_datetime(group[EPISODE_START], errors="coerce")
            ends = pd.to_datetime(group[EPISODE_END], errors="coerce")
            open_at_landmark = (starts <= landmark_time) & ends.isna()
            active_at_landmark = (starts <= landmark_time) & (ends > landmark_time)
            if open_at_landmark.any():
                treatment_classifiable = False
            else:
                continued = bool(active_at_landmark.any())
        rows.append(
            {
                SUBJECT_ID: row.get(SUBJECT_ID, np.nan),
                HADM_ID: row.get(HADM_ID, np.nan),
                STAY_ID: row.get(STAY_ID, np.nan),
                EPISODE_ID: row.get(EPISODE_ID, np.nan),
                "early_discontinuation": int(not continued) if treatment_classifiable else np.nan,
                "continued_crrt_at_landmark": int(continued) if treatment_classifiable else np.nan,
                "treatment_classifiable": bool(treatment_classifiable),
            }
        )
    return pd.DataFrame(rows)


def add_landmark_mortality_outcomes(
    cohort: pd.DataFrame,
    horizons_days: Iterable[int] = (28, 60, 90),
) -> pd.DataFrame:
    """Add mortality outcomes counted from the configured landmark."""

    out = cohort.copy()
    if DOD not in out:
        raise ValueError(f"cohort must contain {DOD!r}.")
    out[DOD] = pd.to_datetime(out[DOD], errors="coerce")
    if out.empty:
        for horizon in horizons_days:
            out[MORTALITY_TEMPLATE.format(days=horizon)] = pd.Series(dtype="int64")
        return out
    for horizon in horizons_days:
        # A death exactly at the landmark is counted as post-landmark mortality;
        # deaths before the landmark were removed by build_landmark_cohort().
        out[MORTALITY_TEMPLATE.format(days=horizon)] = (
            out[DOD].notna()
            & (out[DOD] >= out["landmark_time"])
            & (out[DOD] <= out["landmark_time"] + pd.Timedelta(days=int(horizon)))
        ).astype(int)
    return out


def summarize_multi_episode_patterns(episodes: pd.DataFrame) -> pd.DataFrame:
    """Report how often a stay contains more than one CRRT episode."""

    if episodes.empty:
        return pd.DataFrame(
            [{"n_stays": 0, "n_multi_episode_stays": 0, "multi_episode_rate": 0.0}]
        )
    by_stay = episodes.groupby(STAY_ID)[EPISODE_ID].max()
    return pd.DataFrame(
        [
            {
                "n_stays": int(by_stay.shape[0]),
                "n_multi_episode_stays": int((by_stay > 1).sum()),
                "multi_episode_rate": float((by_stay > 1).mean()),
            }
        ]
    )


def episode_gap_distribution(episodes: pd.DataFrame) -> pd.DataFrame:
    """Return gaps between separate CRRT episodes for sensitivity reporting."""

    if episodes.empty or GAP_FROM_PREVIOUS_EPISODE not in episodes:
        return pd.DataFrame(columns=[STAY_ID, EPISODE_ID, "gap_hours", "gap_days"])
    gaps = episodes.loc[episodes[GAP_FROM_PREVIOUS_EPISODE].notna(), [STAY_ID, EPISODE_ID, GAP_FROM_PREVIOUS_EPISODE]].copy()
    # Empty/all-NaT episode gaps can be inferred as datetime rather than timedelta.
    gaps[GAP_FROM_PREVIOUS_EPISODE] = pd.Series(gaps[GAP_FROM_PREVIOUS_EPISODE].tolist(), index=gaps.index,
                                               dtype="timedelta64[ns]")
    gaps["gap_hours"] = gaps[GAP_FROM_PREVIOUS_EPISODE].dt.total_seconds() / 3600.0
    gaps["gap_days"] = gaps["gap_hours"] / 24.0
    return gaps[[STAY_ID, EPISODE_ID, "gap_hours", "gap_days"]].reset_index(drop=True)


def _threshold_exclusion_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize threshold-related exclusions by reason."""

    if records is None or records.empty or "threshold_exclusion_reason" not in records:
        return pd.DataFrame(columns=["threshold_exclusion_reason", "n"])
    return (
        records.groupby("threshold_exclusion_reason", dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values("threshold_exclusion_reason")
        .reset_index(drop=True)
    )


def _threshold_reason_count(summary: pd.DataFrame, reason: str) -> int:
    """Return the exclusion count for one threshold-exclusion reason."""

    if summary is None or summary.empty or "threshold_exclusion_reason" not in summary:
        return 0
    match = summary.loc[summary["threshold_exclusion_reason"] == reason, "n"]
    return int(match.iloc[0]) if not match.empty else 0


def _threshold_exclusion_records(
    episodes: pd.DataFrame,
    candidates: pd.DataFrame | None,
    reason: str,
    only_missing_documentation: bool = False,
    only_with_documentation: bool = False,
) -> pd.DataFrame:
    """Build stay-level records for urine-threshold exclusions."""

    schema = [
        SUBJECT_ID,
        HADM_ID,
        STAY_ID,
        EPISODE_ID,
        EPISODE_START,
        EPISODE_END,
        "threshold_exclusion_reason",
        "n_documented_urine_records",
        "max_urineoutput_24hr",
        "max_threshold_margin_ml_24hr",
        "min_applicable_threshold_ml_24hr",
        "max_applicable_threshold_ml_24hr",
        "any_diuretic_prior_24h",
    ]
    if episodes is None or episodes.empty:
        return pd.DataFrame(columns=schema)

    records = []
    candidate_groups = {}
    if candidates is not None and not candidates.empty:
        candidate_groups = {stay_id: group for stay_id, group in candidates.groupby(STAY_ID)}

    for _, episode in episodes.iterrows():
        group = candidate_groups.get(episode[STAY_ID], pd.DataFrame())
        has_documentation = not group.empty
        if only_missing_documentation and has_documentation:
            continue
        if only_with_documentation and not has_documentation:
            continue
        record = {
            SUBJECT_ID: episode.get(SUBJECT_ID, np.nan),
            HADM_ID: episode.get(HADM_ID, np.nan),
            STAY_ID: episode.get(STAY_ID, np.nan),
            EPISODE_ID: episode.get(EPISODE_ID, np.nan),
            EPISODE_START: episode.get(EPISODE_START, pd.NaT),
            EPISODE_END: episode.get(EPISODE_END, pd.NaT),
            "threshold_exclusion_reason": reason,
            "n_documented_urine_records": int(len(group)) if has_documentation else 0,
            "max_urineoutput_24hr": np.nan,
            "max_threshold_margin_ml_24hr": np.nan,
            "min_applicable_threshold_ml_24hr": np.nan,
            "max_applicable_threshold_ml_24hr": np.nan,
            "any_diuretic_prior_24h": False,
        }
        if has_documentation:
            record["max_urineoutput_24hr"] = float(pd.to_numeric(group["urineoutput_24hr"], errors="coerce").max())
            if "threshold_margin_ml_24hr" in group:
                record["max_threshold_margin_ml_24hr"] = float(pd.to_numeric(group["threshold_margin_ml_24hr"], errors="coerce").max())
            if "time_zero_threshold_ml_24hr" in group:
                thresholds = pd.to_numeric(group["time_zero_threshold_ml_24hr"], errors="coerce")
                record["min_applicable_threshold_ml_24hr"] = float(thresholds.min())
                record["max_applicable_threshold_ml_24hr"] = float(thresholds.max())
            if "diuretic_prior_24h" in group:
                record["any_diuretic_prior_24h"] = bool(group["diuretic_prior_24h"].fillna(False).any())
        records.append(record)
    return pd.DataFrame(records, columns=schema)


class CohortFlow:
    """Small helper for recording every cohort-reducing operation."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.current_n = 0

    def add(self, step: str, before: int, after: int) -> None:
        # Store both counts and removals so the table can feed a flow diagram.
        self.rows.append(
            {
                "step": step,
                "n_before": int(before),
                "n_after": int(after),
                "n_removed": int(before - after),
            }
        )
        self.current_n = int(after)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _prepare_stays(stays: pd.DataFrame) -> pd.DataFrame:
    """Normalize datetime columns used by eligibility rules."""

    out = stays.copy()
    for col in [ICU_INTIME, "icu_outtime", DOD]:
        if col in out:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _filter_if_present(frame: pd.DataFrame, column: str, predicate) -> pd.DataFrame:
    """Apply an eligibility filter only when the required source column exists."""

    if column not in frame:
        return frame.copy()
    values = pd.to_numeric(frame[column], errors="coerce")
    return frame.loc[predicate(values).fillna(False)].copy()


def _keep_earliest_stay_per_subject(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest ICU admission for each patient."""

    if SUBJECT_ID not in frame or ICU_INTIME not in frame:
        return frame.copy()
    return (
        frame.sort_values([SUBJECT_ID, ICU_INTIME])
        .drop_duplicates(SUBJECT_ID, keep="first")
        .reset_index(drop=True)
    )


def _episode_row(stay_id, episode_id: int, records: list[pd.Series], previous_end) -> dict:
    """Create one episode-level row from the raw procedure rows in an episode."""

    frame = pd.DataFrame(records)
    episode_start = frame[START_TIME].min()
    episode_end = frame[END_TIME].max()
    gap = episode_start - previous_end if pd.notna(previous_end) else pd.NaT
    first = frame.sort_values(START_TIME).iloc[0]
    # Multiple itemids inside an episode are collapsed into both a primary label
    # and a pipe-delimited full modality list for later subgroup summaries.
    modalities = sorted({str(value) for value in frame["crrt_modality"].dropna().unique()})
    return {
        SUBJECT_ID: first.get(SUBJECT_ID, np.nan),
        HADM_ID: first.get(HADM_ID, np.nan),
        STAY_ID: stay_id,
        EPISODE_ID: int(episode_id),
        EPISODE_START: episode_start,
        EPISODE_END: episode_end,
        "episode_record_count": int(len(frame)),
        GAP_FROM_PREVIOUS_EPISODE: gap,
        "crrt_primary_modality": modalities[0] if modalities else np.nan,
        "crrt_modalities": "|".join(modalities),
    }


def _max_timestamp(values) -> pd.Timestamp | pd.NaT:
    """Return the latest valid timestamp or NaT when all values are missing."""

    series = pd.to_datetime(pd.Series(values), errors="coerce")
    if series.notna().any():
        return series.max()
    return pd.NaT


def _diuretic_prior_24h(candidates: pd.DataFrame, diuretics: pd.DataFrame | None) -> pd.Series:
    """Flag whether a loop diuretic overlapped the 24 hours before each row."""

    if diuretics is None or diuretics.empty:
        return pd.Series(False, index=candidates.index)
    # Restrict to the pre-specified loop diuretic itemids only.
    meds = diuretics.loc[diuretics["itemid"].isin(DIURETIC_ITEMIDS)].copy()
    if meds.empty:
        return pd.Series(False, index=candidates.index)
    meds[START_TIME] = pd.to_datetime(meds[START_TIME], errors="coerce")
    meds[END_TIME] = pd.to_datetime(meds.get(END_TIME, meds[START_TIME]), errors="coerce")
    meds[END_TIME] = meds[END_TIME].fillna(meds[START_TIME])
    meds = meds.loc[meds[STAY_ID].notna() & meds[START_TIME].notna()].copy()
    meds_by_stay = {stay_id: group for stay_id, group in meds.groupby(STAY_ID)}

    exposed = []
    for _, row in candidates.iterrows():
        group = meds_by_stay.get(row[STAY_ID])
        if group is None:
            exposed.append(False)
            continue
        window_start = row["charttime"] - pd.Timedelta(hours=24)
        window_end = row["charttime"]
        # Interval overlap handles both bolus-like and infusion-like medication
        # records without assuming zero-duration administration.
        overlap = (group[START_TIME] <= window_end) & (group[END_TIME] >= window_start)
        exposed.append(bool(overlap.any()))
    return pd.Series(exposed, index=candidates.index)


def _empty_episodes() -> pd.DataFrame:
    """Return an empty episode frame with the expected schema."""

    return pd.DataFrame(
        columns=[
            SUBJECT_ID,
            HADM_ID,
            STAY_ID,
            EPISODE_ID,
            EPISODE_START,
            EPISODE_END,
            "episode_record_count",
            GAP_FROM_PREVIOUS_EPISODE,
            N_EPISODES_FOR_STAY,
            HAS_MULTIPLE_EPISODES,
            "crrt_primary_modality",
            "crrt_modalities",
        ]
    )


def _empty_time_zero() -> pd.DataFrame:
    """Return an empty time-zero frame with the expected schema."""

    return pd.DataFrame(
        columns=[
            SUBJECT_ID,
            HADM_ID,
            STAY_ID,
            EPISODE_ID,
            "time_zero",
            "time_zero_urineoutput_24hr",
            "time_zero_uo_tm_24hr",
            "diuretic_prior_24h",
            "time_zero_threshold_ml_24hr",
            "threshold_margin_ml_24hr",
            "crrt_primary_modality",
        ]
    )
