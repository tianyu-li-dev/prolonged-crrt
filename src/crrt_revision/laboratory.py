"""Extract blood laboratory events without requiring an oxygen result."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .features import build_analysis_frames, summarize_numeric_events, summarize_categorical_status, _join_to_window


@dataclass(frozen=True)
class Lab:
    itemid: int
    unit: str
    worst: str
    lower: float
    upper: float
    required: bool = False


LABS = {
    "potassium": Lab(50971, "mmol/L", "both", 0, 40, True),
    "sodium": Lab(50983, "mmol/L", "both", 20, 350, True),
    "chloride": Lab(50902, "mmol/L", "both", 10, 350, True),
    "bicarbonate": Lab(50882, "mmol/L", "both", 0, 100, True),
    "ph_blood": Lab(50820, "unitless", "both", 4, 9.5, True),
    "creatinine": Lab(50912, "mg/dL", "max", 0, 150, True),
    "bun": Lab(51006, "mg/dL", "max", 0, 400, True),
    "pco2": Lab(50818, "mmHg", "both", 0, 300),
    "base_excess": Lab(50802, "mmol/L", "both", -150, 150),
    "anion_gap": Lab(50868, "mmol/L", "both", -200, 200),
    "calcium_total": Lab(50893, "mg/dL", "both", 0, 50),
    "calcium_ionized": Lab(50808, "mmol/L", "both", 0, 10),
    "magnesium": Lab(50960, "mg/dL", "both", 0, 30),
    "phosphate": Lab(50970, "mg/dL", "both", 0, 50),
    "lactate": Lab(50813, "mmol/L", "max", 0, 50),
}
MAIN_LABS = tuple(name for name, spec in LABS.items() if spec.required)
GAS_ALTERNATIVES = {50822: "potassium", 50824: "sodium", 50806: "chloride", 50803: "bicarbonate"}
ALL_ITEMIDS = tuple([spec.itemid for spec in LABS.values()] + list(GAS_ALTERNATIVES) + [52033])


def _convert(value, unit, name):
    value = np.nan if pd.isna(value) else value
    u = ("" if pd.isna(unit) else str(unit)).strip().lower().replace(" ", "").replace("μ", "u").replace("µ", "u")
    target = LABS[name].unit.lower()
    if name == "ph_blood" and u in {"", "units", "unit", "unitless", "ph"}:
        return value
    if u == target or (target == "mmol/l" and u in {"meq/l", "mmol/l"}):
        return value
    if name == "creatinine" and u in {"umol/l", "micromol/l"}:
        return value / 88.4
    if name == "bun" and u == "mmol/l":
        return value * 2.801
    if name == "pco2" and u == "kpa":
        return value * 7.50062
    if name == "magnesium" and u == "mmol/l":
        return value * 2.4305
    if name == "phosphate" and u == "mmol/l":
        return value * 3.0974
    if name == "calcium_total" and u == "mmol/l":
        return value * 4.0078
    if name == "calcium_ionized" and u == "mg/dl":
        return value / 4.0078
    return np.nan


def normalize_labs(events, dictionary, chemistry_fallback=False, arterial_only=False, strict_storetime=False):
    """Return validated numeric blood events and aggregate exclusion counts."""
    required_columns = {"subject_id", "hadm_id", "charttime", "itemid", "valuenum", "valueuom"}
    if not required_columns.issubset(events.columns):
        raise ValueError("Laboratory event schema is incomplete.")
    lookup = dictionary.set_index("itemid")
    for spec in LABS.values():
        if spec.required and (spec.itemid not in lookup.index or
                              str(lookup.loc[spec.itemid, "fluid"]).lower() != "blood"):
            raise ValueError(f"Required blood item {spec.itemid} is absent or has a different fluid.")
    out = events.loc[events.itemid.isin(ALL_ITEMIDS)].copy()
    out["charttime"] = pd.to_datetime(out.charttime, errors="coerce")
    out["storetime"] = pd.to_datetime(out.get("storetime", pd.Series(pd.NaT, index=out.index)), errors="coerce")
    out["available_time"] = out.charttime
    if strict_storetime:
        # Unknown availability remains unknown. Never substitute a future value.
        out["available_time"] = out[["charttime", "storetime"]].max(axis=1)
        out.loc[out.storetime.isna() | out.charttime.isna(), "available_time"] = pd.NaT
    out["specimen_type"] = "unknown"
    specimen = out.loc[out.itemid.eq(52033)].copy() if "specimen_id" in out and "value" in out else pd.DataFrame()
    reverse = {spec.itemid: name for name, spec in LABS.items()}
    out["variable"] = out.itemid.map({**reverse, **GAS_ALTERNATIVES})
    out = out.loc[out.variable.notna()].copy()
    out["source_kind"] = np.where(out.itemid.isin(GAS_ALTERNATIVES), "blood_gas_alternative", "primary")
    values, statuses = [], []
    for row in out.itertuples():
        blood = row.itemid in lookup.index and str(lookup.loc[row.itemid, "fluid"]).lower() == "blood"
        v = pd.to_numeric(row.valuenum, errors="coerce")
        converted = _convert(v, row.valueuom, row.variable)
        spec = LABS[row.variable]
        status = "ok"
        if not blood:
            status = "non_blood"
        elif not np.isfinite(converted):
            status = "missing_or_unknown_unit"
        elif not spec.lower <= converted <= spec.upper:
            status = "broad_data_error_range"
        elif pd.isna(row.available_time):
            status = "unknown_available_time"
        elif not chemistry_fallback and row.source_kind != "primary":
            status = "alternative_source_excluded"
        values.append(converted); statuses.append(status)
    out["value_numeric"], out["quality_status"] = values, statuses
    audit = out.groupby(["variable", "source_kind", "quality_status"], dropna=False).size().rename("event_count").reset_index()
    valid = out.loc[out.quality_status.eq("ok")].copy()
    # Preserve versions and label histories until each cohort's cutoff is known.
    valid.attrs.update(strict_storetime=bool(strict_storetime), arterial_only=bool(arterial_only),
                       specimen_labels=specimen.to_dict("records"))
    return valid, audit


def laboratory_snapshot(cohort, events):
    """Select distinct measurements and labels available at each patient's t0."""
    if events.empty:
        return events.copy()
    strict, arterial = events.attrs.get("strict_storetime", False), events.attrs.get("arterial_only", False)
    joined = _join_to_window(cohort, events, "charttime", join_key="hadm_id")
    joined = joined.loc[pd.to_datetime(joined.available_time) <= joined.time_zero].copy()
    keys = ["stay_id", "hadm_id", "itemid", "charttime"]
    if "specimen_id" in joined:
        keys.insert(2, "specimen_id")
    joined = joined.sort_values("storetime", na_position="first", kind="stable").drop_duplicates(keys, keep="last")
    joined["_source_priority"] = joined.source_kind.ne("primary").astype(int)
    timestamp_keys = ["stay_id", "hadm_id", "charttime", "variable"]
    if not joined.empty:
        best = joined.groupby(timestamp_keys)._source_priority.transform("min")
        joined = joined.loc[joined._source_priority.eq(best)].copy()
    labels = pd.DataFrame(events.attrs.get("specimen_labels", []))
    if not labels.empty and "specimen_id" in joined:
        labels = labels.loc[labels.hadm_id.notna() & labels.specimen_id.notna()]
        labels = labels.merge(cohort[["stay_id", "hadm_id", "time_zero"]], on="hadm_id", how="inner")
        labels = labels.loc[pd.to_datetime(labels.charttime).le(labels.time_zero) &
                            pd.to_datetime(labels.available_time).le(labels.time_zero)]
        label_keys = ["stay_id", "hadm_id", "specimen_id"]
        labels = labels.sort_values(["available_time", "storetime"], na_position="first", kind="stable").drop_duplicates(label_keys, keep="last")
        labels["specimen_type"] = labels.value.fillna("unknown").astype(str).str.lower()
        joined = joined.drop(columns="specimen_type").merge(labels[label_keys + ["specimen_type"]], on=label_keys, how="left")
        joined["specimen_type"] = joined.specimen_type.fillna("unknown")
    if arterial:
        acid_base = joined.variable.isin(["ph_blood", "pco2", "base_excess"])
        joined = joined.loc[~acid_base | joined.specimen_type.isin(["art", "arterial"])].copy()
    # Downstream shared summaries rejoin by stay_id and sort actual measurement time.
    keep = [c for c in events.columns if c not in {"subject_id", "hadm_id", "stay_id"}] + ["subject_id", "hadm_id", "stay_id"]
    return joined.reindex(columns=keep).reset_index(drop=True)


def build_features(cohort, sources, variables=MAIN_LABS, window_hours=None):
    """Apply a shared numeric summarizer to the specified measurement sources."""
    sources = {name: frame.copy() for name, frame in sources.items()}
    design = cohort.copy()
    if window_hours is not None:
        design["episode_start"] = pd.concat([pd.to_datetime(design.episode_start),
            pd.to_datetime(design.time_zero) - pd.Timedelta(hours=window_hours)], axis=1).max(axis=1)
    events = laboratory_snapshot(design, sources.get("laboratory", pd.DataFrame()))
    # Clinical covariates use their declared source tables and episode window.
    # Laboratory-window sensitivity applies only to the biochemical laboratory block.
    result = build_analysis_frames(cohort, sources)
    features = result.analysis_frame
    audit_rows, metadata = [], result.feature_metadata.to_dict("records")
    for name in variables:
        if name not in LABS:
            raise ValueError(f"Unknown declared laboratory variable: {name}")
        if name == "lactate":
            raise ValueError("Lactate is read from the derived bg source.")
        spec = LABS[name]
        frame = events.loc[events.variable.eq(name)].copy() if not events.empty else pd.DataFrame()
        frame["event_time"] = pd.to_datetime(frame.get("charttime", pd.Series(dtype="datetime64[ns]")))
        summary = summarize_numeric_events(design, frame, "value_numeric", "event_time", name, spec.worst, "stay_id")
        expected = [f"tz_last_{name}", f"mean_start_to_tz_{name}"] + (
            [f"min_start_to_tz_{name}", f"max_start_to_tz_{name}"] if spec.worst == "both" else [f"worst_start_to_tz_{name}"])
        for c in expected:
            if c not in summary:
                summary[c] = np.nan
        features = features.merge(summary, on=["subject_id", "hadm_id", "stay_id"], how="left", validate="one_to_one")
        joined = _join_to_window(design, frame, "event_time", join_key="stay_id") if not frame.empty else pd.DataFrame()
        for level, group in [("all", features), ("continued", features.loc[features.intervention_status.eq(1)]),
                             ("discontinued", features.loc[features.intervention_status.eq(0)])]:
            coverage = float(group[f"tz_last_{name}"].notna().mean()) if len(group) else np.nan
            audit_rows.append({"variable": name, "treatment_group": level, "n": len(group),
                               "n_observed": int(group[f"tz_last_{name}"].notna().sum()),
                               "completeness": coverage, "missing_fraction": 1 - coverage,
                               "review_status": "unavailable" if coverage == 0 else "review_sparse" if coverage < .5 else "ok"})
        if not joined.empty:
            counts = joined.groupby("stay_id").size()
            ages = ((joined.time_zero - joined.event_time).dt.total_seconds() / 3600).groupby(joined.stay_id).min()
            audit_rows[-3].update(median_measurement_count=float(counts.median()),
                                  repeated_measurement_fraction=float((counts > 1).mean()),
                                  measurement_age_median_hours=float(ages.median()))
        metadata.append({"feature_group": name, "source_type": "time_stamped_numeric", "unit": spec.unit,
                         "worst_rule": spec.worst, "source_table": "mimiciv_hosp.labevents"})
    if not events.empty and "ph_blood" in variables:
        ph = events.loc[events.variable.eq("ph_blood")].copy()
        ph["event_time"] = ph.charttime
        labels = summarize_categorical_status(design, ph, "specimen_type", "event_time", "ph_specimen")
        features = features.merge(labels, on=["subject_id", "hadm_id", "stay_id"], how="left", validate="one_to_one")
    # Preserve the original episode start as an audit field after a window sensitivity.
    features["episode_start"] = cohort.set_index("stay_id").loc[features.stay_id, "episode_start"].to_numpy()
    return features, pd.DataFrame(metadata), pd.DataFrame(audit_rows, columns=list(dict.fromkeys(
        ["variable", "treatment_group", "n", "n_observed", "completeness", "missing_fraction", "review_status"] +
        [k for row in audit_rows for k in row])))
