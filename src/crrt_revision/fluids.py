"""Documented input volumes in the intersection of infusion and baseline windows."""
import numpy as np
import pandas as pd

KEYS = ["subject_id", "hadm_id", "stay_id"]
VOLUME_UNITS = {"ml": 1., "milliliter": 1., "milliliters": 1., "l": 1000., "liter": 1000., "liters": 1000.}
RATE_UNITS = {"ml/hr": 1., "ml/h": 1., "ml/hour": 1., "ml/min": 60., "ml/sec": 3600.,
              "l/hr": 1000., "l/h": 1000., "l/min": 60000.}


def input_volumes(cohort, events):
    """Return mL summaries and a source-quality audit; never convert drug mass."""
    columns = ["fluid_input_ml_start_to_tz", "fluid_input_ml_pre24h"]
    out = cohort[KEYS].copy()
    for column in columns:
        out[column] = 0.
    if events is None or events.empty:
        return out, pd.DataFrame(columns=["quality_status", "event_count"])
    if not {"stay_id", "starttime"}.issubset(events) or not ({"amountuom", "rateuom"} & set(events)):
        raise ValueError("Input source requires timing and volume units. Re-extract amountuom/rateuom.")
    raw = events.copy().reset_index(drop=True)
    for column in ["amount", "rate"]:
        raw[column] = pd.to_numeric(raw.get(column, pd.Series(np.nan, index=raw.index)), errors="coerce").astype(float)
    for column in ["amountuom", "rateuom", "statusdescription", "ordercategorydescription", "ordercomponenttypedescription"]:
        raw[column] = raw.get(column, pd.Series("", index=raw.index)).fillna("").astype(str).str.strip().str.lower()
    for column in ["starttime", "endtime", "storetime"]:
        raw[column] = pd.to_datetime(raw.get(column, pd.Series(pd.NaT, index=raw.index)), errors="coerce")
    raw["_amount_factor"] = raw.amountuom.map(VOLUME_UNITS)
    raw["_rate_factor"] = raw.rateuom.map(RATE_UNITS)
    raw["_row"] = np.arange(len(raw))
    raw["quality_status"] = "ok"
    raw.loc[raw._amount_factor.isna() & raw._rate_factor.isna(), "quality_status"] = "nonvolume_or_unknown_unit"
    raw.loc[raw.statusdescription.str.contains("rewritten|cancel", regex=True), "quality_status"] = "cancelled_or_rewritten"
    raw.loc[raw.starttime.isna(), "quality_status"] = "unknown_start"
    if "orderid" in raw:
        named = raw.orderid.notna()
        version_keys = ["stay_id", "orderid", "starttime", "endtime"] + (["itemid"] if "itemid" in raw else [])
        duplicates = raw.loc[named].sort_values("storetime", na_position="first", kind="stable").duplicated(version_keys, keep="last")
        raw.loc[duplicates.index[duplicates], "quality_status"] = "duplicate_input_version"
        candidates = raw.loc[named & raw.quality_status.eq("ok")]
        group_keys = ["stay_id", "orderid", "starttime", "endtime"]
        group_size = candidates.groupby(group_keys, dropna=False)["_row"].transform("size")
        carrier = candidates.ordercomponenttypedescription.str.contains("mixed solution", regex=False)
        carrier_count = carrier.groupby([candidates[key] for key in group_keys], dropna=False).transform("sum")
        excluded = candidates.index[group_size.gt(1) & (carrier_count.ne(1) | ~carrier)]
        raw.loc[excluded, "quality_status"] = "ambiguous_volume_component"
    lookup = cohort[KEYS + ["episode_start", "time_zero"]].drop_duplicates("stay_id")
    records = []
    for row in raw.merge(lookup, on="stay_id", suffixes=("", "_cohort"), how="inner").to_dict("records"):
        status = row["quality_status"]
        volumes = []
        for lower in [row["episode_start"], max(row["episode_start"], row["time_zero"] - pd.Timedelta(hours=24))]:
            volume = 0.
            start, end = row["starttime"], row["endtime"]
            amount = row["amount"] * row["_amount_factor"]
            rate = row["rate"] * row["_rate_factor"]
            bolus = "bolus" in row["ordercategorydescription"] or (pd.notna(end) and end == start)
            if status == "ok":
                if bolus and np.isfinite(amount) and amount >= 0:
                    volume = amount if lower <= start <= row["time_zero"] else 0.
                elif pd.isna(end) or end < start:
                    status = "invalid_infusion_interval"
                else:
                    duration = (end - start).total_seconds() / 3600
                    overlap = max(0., (min(end, row["time_zero"]) - max(start, lower)).total_seconds() / 3600)
                    if np.isfinite(rate) and rate >= 0:
                        volume = rate * overlap
                    elif np.isfinite(amount) and amount >= 0 and duration > 0:
                        volume = amount * overlap / duration
                    else:
                        status = "invalid_volume_or_rate"
            volumes.append(float(volume))
        # _row is the unchanged RangeIndex assigned before source processing.
        raw.at[row["_row"], "quality_status"] = status
        records.append({"stay_id": row["stay_id"], **dict(zip(columns, volumes))})
    if records:
        sums = pd.DataFrame(records).groupby("stay_id")[columns].sum()
        for column in columns:
            out[column] = out.stay_id.map(sums[column]).fillna(0.)
    audit = raw.groupby("quality_status").size().rename("event_count").reset_index()
    return out, audit
