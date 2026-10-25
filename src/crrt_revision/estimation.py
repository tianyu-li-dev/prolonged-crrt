"""Cross-fitted CU-DML with full-sample propensity correction."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .models import (
    _fit_outcome_probability_model, _fit_undersampled_propensity,
    _predict_probability, _predict_positive_probability,
    _undersampling_sample_probs, _calibrate_case_control_propensity, _crump_trim_mask,
    _undersampling_indices,
)
from .preprocessing import fit_transformer, transform
from .checkpoints import Checkpoints, CheckpointError

FAMILIES = ("logistic", "lasso", "elastic_net", "xgboost", "rf")
OUTCOMES = ("mortality_28days", "mortality_60days", "mortality_90days")


def folds(a, n_folds=5, seed=42):
    """Create shuffled treatment-stratified row folds."""
    a = np.asarray(a)
    k = min(int(n_folds), int(np.sum(a == 0)), int(np.sum(a == 1)))
    if k < 2:
        raise ValueError("At least two rows per treatment group are required.")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    yield from cv.split(np.zeros_like(a), a)


def crossfit(frame, outcome, family, n_folds=5, seed=42, missing_indicators=False,
             keep_empty_features=False,
             checkpoint=None):
    """Return complete OOF predictions. Never fill them from an in-sample fit."""
    complete = frame.loc[frame[outcome].notna() & frame.intervention_status.notna()].copy().reset_index(drop=True)
    if complete.empty:
        raise ValueError("No observed treatment/outcome pairs.")
    a = complete.intervention_status.to_numpy(dtype=int)
    y = complete[outcome].to_numpy(dtype=float)
    if not np.isin(a, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Treatment and mortality must be binary.")
    if len(np.unique(a)) != 2:
        raise ValueError("Both treatment groups are required.")
    e, q, m0, m1 = [np.full(len(complete), np.nan) for _ in range(4)]
    fold_id = np.full(len(complete), -1, dtype=int)
    names = set()
    empty_by_fold = {}
    # Use class sampling probabilities from the full complete sample.
    sampling_probs = _undersampling_sample_probs(a)
    store = None
    if checkpoint is not None:
        store = Checkpoints(checkpoint, {"task": "crossfit", "n_rows": len(complete),
                            "columns": complete.columns.tolist(), "outcome": outcome, "family": family,
                            "folds": n_folds, "seed": seed, "missing_indicators": missing_indicators,
                            "keep_empty_features": keep_empty_features})
        store.prepare(f"fold:{k}" for k in range(min(int(n_folds), int(np.sum(a == 0)), int(np.sum(a == 1)))))
    for k, (tr, te) in enumerate(folds(a, n_folds, seed)):
        cached = store.load(f"fold:{k}") if store else None
        if cached:
            payload = cached["payload"]
            if payload["test_indices"] != te.tolist():
                raise CheckpointError("Saved OOF fold differs from the current row split.")
            for target, field in [(e, "e"), (q, "q"), (m0, "mu0"), (m1, "mu1")]:
                values = np.asarray(payload[field], dtype=float)
                if len(values) != len(te) or not np.isfinite(values).all():
                    raise CheckpointError("Saved OOF fold has invalid predictions.")
                target[te] = values
            names.update(payload["features"])
            empty_by_fold[k] = payload["empty_training_columns"]
            fold_id[te] = k
            continue
        transformer, xtr, feature_names = fit_transformer(complete.iloc[tr], missing_indicators, keep_empty_features)
        empty_by_fold[k] = transformer.empty_training_columns_
        xte = transform(transformer, complete.iloc[te])
        names.update(feature_names)
        at, yt = a[tr], y[tr]
        model0 = _fit_outcome_probability_model(xtr[at == 0], yt[at == 0], family, seed + 1)
        model1 = _fit_outcome_probability_model(xtr[at == 1], yt[at == 1], family, seed)
        prop = _fit_undersampled_propensity(xtr, at, family, seed + 2)
        m0[te], m1[te] = _predict_probability(model0, xte), _predict_probability(model1, xte)
        q[te] = _predict_positive_probability(prop, xte)
        e[te] = _calibrate_case_control_propensity(q[te], sampling_probs)
        fold_id[te] = k
        if store:
            if not all(np.isfinite(x[te]).all() for x in [e, q, m0, m1]):
                raise ValueError("Non-finite OOF fold predictions cannot be checkpointed.")
            store.save(f"fold:{k}", {"test_indices": te.tolist(), "e": e[te], "q": q[te],
                        "mu0": m0[te], "mu1": m1[te], "features": feature_names,
                        "empty_training_columns": empty_by_fold[k]})
    if not all(np.isfinite(x).all() for x in [e, q, m0, m1]) or (fold_id < 0).any():
        raise ValueError("Incomplete OOF predictions. No in-sample fallback is permitted.")
    predictions = complete[[c for c in ["subject_id", "hadm_id", "stay_id"] if c in complete]].copy()
    predictions["y"], predictions["a"] = y, a
    predictions["mu0"], predictions["mu1"] = m0, m1
    predictions["ps_sampled"], predictions["propensity"] = q, e
    predictions["fold_id"] = fold_id
    return {"predictions": predictions, "frame": complete, "feature_names": sorted(names),
            "sampling_probs": sampling_probs, "empty_training_columns_by_fold": empty_by_fold}


def estimate(predictions: pd.DataFrame, bounds=(0.01, 0.99), fixed_mask=None):
    p = predictions
    mask = _crump_trim_mask(p.propensity.to_numpy(), p.a.to_numpy(), bounds)["mask"]
    if fixed_mask is not None:
        mask = np.asarray(fixed_mask, dtype=bool)
        if len(mask) != len(p):
            raise ValueError("Fixed mask has the wrong length.")
    selected = p.loc[mask].copy()
    if selected.empty or selected.a.nunique() != 2:
        raise ValueError("Overlap population must contain both treatment groups.")
    e = np.clip(selected.propensity.to_numpy(), 1e-6, 1 - 1e-6)
    a, y = selected.a.to_numpy(), selected.y.to_numpy()
    c1 = selected.mu1.to_numpy() - selected.mu0.to_numpy()
    c2 = a / e * (y - selected.mu1.to_numpy())
    c3 = -(1 - a) / (1 - e) * (y - selected.mu0.to_numpy())
    selected["c1"], selected["c2"], selected["c3"] = c1, c2, c3
    selected["score"] = c1 + c2 + c3
    selected["weight"] = np.where(a == 1, 1 / e, 1 / (1 - e))
    ate = float(selected.score.mean())
    se = float(selected.score.std(ddof=1) / np.sqrt(len(selected)))
    return {"ATE": ate, "if_standard_error": se, "if_ci_lower": ate - 1.96 * se,
            "if_ci_upper": ate + 1.96 * se, "n_before_trimming": len(p), "n": len(selected),
            "treated": int(a.sum()), "control": int((1 - a).sum()),
            "trimming_lower": bounds[0], "trimming_upper": bounds[1]}, selected, mask


def bootstrap(frame, outcome, family, count, n_folds=5, seed=42, bounds=(0.01, 0.99),
              n_jobs=1, missing_indicators=False, keep_empty_features=False, checkpoint=None):
    """Resample analysis rows and refit the entire estimation procedure."""
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(frame), len(frame)) for _ in range(count)]
    store = Checkpoints(checkpoint, {"task": "bootstrap", "n_rows": len(frame), "columns": frame.columns.tolist(),
                        "outcome": outcome, "family": family, "count": count, "folds": n_folds, "seed": seed,
                        "bounds": list(bounds), "missing_indicators": missing_indicators,
                        "keep_empty_features": keep_empty_features}) if checkpoint is not None else None

    if store:
        store.prepare(f"bootstrap:{i}" for i in range(count))

    def one(i, draw):
        cached = store.load(f"bootstrap:{i}") if store else None
        if cached and cached["payload"]["status"] == "ok":
            return cached["payload"]
        try:
            sample = frame.iloc[draw].reset_index(drop=True)
            result = crossfit(sample, outcome, family, n_folds, seed + 1000 + i, missing_indicators,
                              keep_empty_features=keep_empty_features,
                              checkpoint=store.directory(f"bootstrap:{i}") / "folds" if store else None)
            row, _, _ = estimate(result["predictions"], bounds)
            item = {"bootstrap_id": i, "status": "ok", **row}
        except CheckpointError:
            raise
        except Exception as exc:
            item = {"bootstrap_id": i, "status": "failed", "reason": type(exc).__name__ + ": " + str(exc)}
        if store:
            store.save(f"bootstrap:{i}", item)
        return item

    if n_jobs == 1:
        rows = [one(i, draw) for i, draw in enumerate(draws)]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(delayed(one)(i, draw) for i, draw in enumerate(draws))
    return pd.DataFrame(rows)
