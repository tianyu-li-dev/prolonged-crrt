"""Fit covariate transformations on a training sample only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .models import (
    _is_model_feature_column, STRUCTURAL_ZERO_FEATURE_COLUMNS,
    STRUCTURAL_ZERO_FEATURE_PREFIXES, CATEGORICAL_FEATURE_COLUMNS,
)


def covariates(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in frame if _is_model_feature_column(c)
               and not c.startswith("_audit_")
               and not pd.api.types.is_datetime64_any_dtype(frame[c])]
    out = frame[columns].copy()
    for c in out:
        if c in STRUCTURAL_ZERO_FEATURE_COLUMNS or c.startswith(STRUCTURAL_ZERO_FEATURE_PREFIXES):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(float)
    return out


def fit_transformer(training: pd.DataFrame, missing_indicators: bool = False, keep_empty_features: bool = False):
    """Return a fitted transformer, training matrix, and feature names."""
    x = covariates(training)
    empty_columns = [c for c in x if x[c].isna().all()]
    if not keep_empty_features:
        x = x.drop(columns=empty_columns)
    if not len(x.columns):
        x["intercept_only"] = 0.0
    cat = [c for c in x if c in CATEGORICAL_FEATURE_COLUMNS
           or not pd.api.types.is_numeric_dtype(x[c])]
    continuous_prefixes = ("tz_last_", "mean_start_to_tz_", "worst_start_to_tz_", "min_start_to_tz_", "max_start_to_tz_")
    binary = [c for c in x if c not in cat and not c.startswith(continuous_prefixes) and
              set(x[c].dropna().unique()).issubset({0, 1}) and x[c].notna().any()]
    numeric = [c for c in x if c not in cat + binary]
    for c in cat:
        x[c] = x[c].astype(object).where(x[c].notna(), np.nan)
    transformer = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=keep_empty_features,
                                            add_indicator=missing_indicators), StandardScaler()), numeric),
        ("bin", SimpleImputer(strategy="most_frequent", keep_empty_features=keep_empty_features), binary),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=keep_empty_features),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
    ], sparse_threshold=0)
    matrix = transformer.fit_transform(x)
    transformer.input_columns_ = list(x.columns)
    transformer.categorical_columns_ = cat
    transformer.empty_training_columns_ = empty_columns
    transformer.keep_empty_features_ = keep_empty_features
    return transformer, np.asarray(matrix, dtype=float), transformer.get_feature_names_out().tolist()


def transform(transformer, holdout: pd.DataFrame) -> np.ndarray:
    x = covariates(holdout).reindex(columns=transformer.input_columns_)
    if "intercept_only" in x:
        x["intercept_only"] = 0.0
    for c in transformer.categorical_columns_:
        x[c] = x[c].astype(object).where(x[c].notna(), np.nan)
    return np.asarray(transformer.transform(x), dtype=float)
