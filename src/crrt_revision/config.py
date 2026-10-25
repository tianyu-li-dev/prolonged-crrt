"""Analysis configuration and threshold grid."""
from __future__ import annotations
import json
from pathlib import Path
import yaml
from .estimation import FAMILIES, OUTCOMES
from .laboratory import MAIN_LABS, LABS

DEFAULT = {
    "data": {"release": "2.1", "postgres_uri_env": "MIMIC_POSTGRES_URI", "source_directory": None},
    "design": {"no_diuretic_thresholds": [400, 500, 600], "diuretic_thresholds": [1000, 2000],
               "primary_pair": [500, 2000], "landmark_hours": 24, "merge_gap_hours": 6},
    "features": {"laboratories": list(MAIN_LABS), "chemistry_fallback": False, "arterial_only": False,
                 "strict_storetime": False, "window_hours": None, "missing_indicators": False, "keep_empty_features": False},
    "analysis": {"families": list(FAMILIES), "outcomes": list(OUTCOMES), "folds": 5, "seed": 42,
                 "trimming": [0.01, 0.99], "bootstrap_samples": 1000, "bootstrap_jobs": 1,
                 "minimum_bootstrap_success": 0.95, "cate_primary_only": True, "allow_sparse_labs": True},
}


def load_config(path=None):
    result = json.loads(json.dumps(DEFAULT))
    if path:
        custom = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for section, values in custom.items():
            if section not in result or not isinstance(values, dict):
                raise ValueError(f"Unknown configuration section: {section}")
            unknown = set(values) - set(result[section])
            if unknown:
                raise ValueError(f"Unknown fields in {section}: {sorted(unknown)}")
            result[section].update(values)
    d, a, f = result["design"], result["analysis"], result["features"]
    if not all(x > 0 for x in d["no_diuretic_thresholds"] + d["diuretic_thresholds"]):
        raise ValueError("Thresholds must be positive.")
    if any(len(set(d[k])) != len(d[k]) for k in ["no_diuretic_thresholds", "diuretic_thresholds"]):
        raise ValueError("Duplicate thresholds are not permitted.")
    if not set(a["families"]).issubset(FAMILIES) or not a["families"]:
        raise ValueError("Declare one or more estimator families.")
    if not set(a["outcomes"]).issubset(OUTCOMES) or not a["outcomes"]:
        raise ValueError("Unsupported outcome horizon.")
    if not set(f["laboratories"]).issubset(LABS):
        raise ValueError("Unknown laboratory covariate.")
    if "lactate" in f["laboratories"]:
        raise ValueError("Lactate is read from the derived bg source.")
    if not 0 <= a["trimming"][0] < a["trimming"][1] <= 1:
        raise ValueError("Invalid trimming bounds.")
    if a["folds"] < 2 or a["bootstrap_samples"] < 0 or not 0 < a["minimum_bootstrap_success"] <= 1:
        raise ValueError("Invalid fold or bootstrap configuration.")
    if d["landmark_hours"] <= 0 or d["merge_gap_hours"] < 0:
        raise ValueError("Invalid landmark or episode definition.")
    if f["window_hours"] is not None and f["window_hours"] <= 0:
        raise ValueError("The feature window must be positive.")
    primary = d["primary_pair"]
    if len(primary) != 2 or primary[0] not in d["no_diuretic_thresholds"] or primary[1] not in d["diuretic_thresholds"]:
        raise ValueError("The primary pair must belong to the declared grid.")
    return result


def pairs(config):
    for no in config["design"]["no_diuretic_thresholds"]:
        for di in config["design"]["diuretic_thresholds"]:
            yield float(no), float(di), f"uo_nodiu_{no:g}_diu_{di:g}"
