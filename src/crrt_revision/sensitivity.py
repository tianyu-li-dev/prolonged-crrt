"""Point E-values and IRM robustness values on a fixed analysis population."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from .estimation import crossfit, estimate


def evalue(rr):
    if not np.isfinite(rr) or rr <= 0:
        raise ValueError("A point E-value requires a positive finite risk ratio.")
    magnitude = max(float(rr), 1 / float(rr))
    return magnitude + math.sqrt(magnitude * (magnitude - 1))


def adjusted_risks(p):
    e = np.clip(p.propensity.to_numpy(), 1e-6, 1 - 1e-6)
    y, a = p.y.to_numpy(), p.a.to_numpy()
    m0, m1 = p.mu0.to_numpy(), p.mu1.to_numpy()
    return float(np.mean(m1 + a / e * (y - m1))), float(np.mean(m0 + (1 - a) / (1 - e) * (y - m0)))


def bounded_risks(p):
    """Two-coordinate logistic TMLE. Do not alter the main AIPW estimates."""
    e = np.clip(p.propensity.to_numpy(), 1e-6, 1 - 1e-6)
    y, a = p.y.to_numpy(), p.a.to_numpy()
    means = []
    for treatment, initial, probability in [(1, p.mu1.to_numpy(), e), (0, p.mu0.to_numpy(), 1 - e)]:
        offset = logit(np.clip(initial, 1e-6, 1 - 1e-6))
        h = 1 / probability
        arm = a == treatment
        if not arm.any() or len(np.unique(y[arm])) != 2:
            raise ValueError("Bounded targeting requires both outcomes in each retained treatment arm.")
        def gradient(epsilon):
            return float(np.sum(h[arm] * (y[arm] - expit(offset[arm] + epsilon * h[arm]))))
        epsilon = brentq(gradient, -100, 100, xtol=1e-12)
        means.append(float(expit(offset + epsilon * h).mean()))
    return means[0], means[1]


def moment_details(p):
    e = np.clip(p.propensity.to_numpy(), 1e-6, 1 - 1e-6)
    a, y = p.a.to_numpy(), p.y.to_numpy()
    g = a * p.mu1.to_numpy() + (1 - a) * p.mu0.to_numpy()
    alpha = a / e - (1 - a) / (1 - e)
    sigma2 = float(np.mean((y - g) ** 2))
    raw_nu2 = float(np.mean(2 * (1 / e + 1 / (1 - e)) - alpha ** 2))
    # DoubleML uses this non-orthogonal moment when
    # the estimated orthogonal moment is not positive. This is not a floor.
    fallback = raw_nu2 <= 0
    nu2 = float(np.mean(alpha ** 2)) if fallback else raw_nu2
    return {"sigma2": sigma2, "nu2": nu2, "nu2_orthogonal_raw": raw_nu2,
            "nu2_method": "riesz_squared_fallback" if fallback else "orthogonal_moment"}


def moments(p):
    details = moment_details(p)
    return details["sigma2"], details["nu2"]


def point_bounds(theta, sigma2, nu2, cf_y, cf_d, rho=1.0):
    if not 0 <= cf_y < 1 or not 0 <= cf_d < 1 or not 0 <= abs(rho) <= 1:
        raise ValueError("Sensitivity strengths must be fractions below one.")
    if sigma2 <= 0 or nu2 <= 0:
        raise ValueError("Invalid IRM sensitivity moments. No arbitrary positive floor is applied.")
    bias = abs(rho) * math.sqrt(sigma2 * nu2 * cf_y * cf_d / (1 - cf_d))
    return theta - bias, theta + bias


def robustness_value(theta, sigma2, nu2, rho=1.0):
    if sigma2 <= 0 or nu2 <= 0 or not 0 < abs(rho) <= 1:
        raise ValueError("Invalid moments or error alignment for point RV.")
    b = abs(theta) / (abs(rho) * math.sqrt(sigma2 * nu2))
    return 2 * b / (math.sqrt(b * b + 4) + b)


def summarize(p, theta, rho=1.0):
    p1, p0 = adjusted_risks(p)
    if not np.isclose(p1 - p0, theta, atol=1e-8, rtol=0):
        raise ValueError("Adjusted risks do not match the main risk difference.")
    row = {"adjusted_risk_1": p1, "adjusted_risk_0": p0, "rho": rho,
           "risk_ratio": np.nan, "E_value_point": np.nan, "E_value_method": "AIPW",
           "E_value_status": "ok", "E_value_reason": "", "RV_point": np.nan,
           "RV_percent": np.nan, "RV_status": "ok", "RV_reason": ""}
    if 0 < p1 <= 1 and 0 < p0 <= 1:
        row["risk_ratio"] = p1 / p0
        row["E_value_point"] = evalue(p1 / p0)
    else:
        row["E_value_status"] = "unavailable"
        row["E_value_reason"] = "AIPW adjusted risks are outside the valid positive probability range."
        try:
            t1, t0 = bounded_risks(p)
            row.update(TMLE_risk_1=t1, TMLE_risk_0=t0, TMLE_risk_difference=t1 - t0,
                       TMLE_risk_ratio=t1 / t0, TMLE_E_value_point=evalue(t1 / t0))
        except ValueError as exc:
            row["TMLE_status"] = str(exc)
    details = moment_details(p)
    sigma2, nu2 = details["sigma2"], details["nu2"]
    row.update(details)
    try:
        rv = robustness_value(theta, sigma2, nu2, rho)
        row.update(RV_point=rv, RV_percent=100 * rv)
    except ValueError as exc:
        row.update(RV_status="unavailable", RV_reason=str(exc))
    return row


BENCHMARKS = {
    "fluid": ("fluid_", "net_fluid_balance", "positive_fluid_balance"),
    "electrolytes": ("potassium", "sodium", "chloride", "calcium_total", "calcium_ionized", "magnesium", "phosphate"),
    "acid_base": ("ph_blood", "bicarbonate", "ph_specimen", "pco2", "base_excess", "anion_gap"),
    "renal": ("creatinine", "bun"),
    "vasopressor": ("vasopressor",),
    "severity": ("sofa", "sapsii"),
}


def benchmark(frame, outcome, family, full_predictions, mask, config, checkpoint=None):
    """Drop entire blocks and refit; freeze the long-model overlap population."""
    long_row, long_p, _ = estimate(full_predictions, config["analysis"]["trimming"], mask)
    long_sigma, long_nu = moments(long_p)
    from .checkpoints import Checkpoints
    store = Checkpoints(checkpoint, {"task": "benchmark", "n_rows": len(frame),
                        "columns": frame.columns.tolist(), "mask": np.asarray(mask).tolist(),
                        "outcome": outcome, "family": family, "config": config}) if checkpoint is not None else None
    rows = []
    for block, tokens in BENCHMARKS.items():
        cached = store.load("benchmark:" + block) if store else None
        if cached:
            rows.append(cached["payload"])
            continue
        drop = [c for c in frame if any(token in c for token in tokens)]
        if not drop:
            rows.append({"benchmark_block": block, "status": "absent"}); continue
        try:
            a = config["analysis"]
            short = crossfit(frame.drop(columns=drop), outcome, family, a["folds"], a["seed"],
                             config["features"]["missing_indicators"],
                             keep_empty_features=config["features"]["keep_empty_features"],
                             checkpoint=store.directory("benchmark:" + block) / "folds" if store else None)
            short_row, short_p, _ = estimate(short["predictions"], a["trimming"], mask)
            short_sigma, short_nu = moments(short_p)
            # Observed long/short comparison on the SAME fixed target.
            # These are empirical strength estimates, not guaranteed population quantities.
            cf_y_raw = (short_sigma - long_sigma) / long_sigma
            cf_d_gain = (long_nu - short_nu) / short_nu
            cf_d_raw = cf_d_gain / (1 + cf_d_gain) if cf_d_gain > -1 else np.nan
            cf_y, cf_d = np.clip(cf_y_raw, 0, 1), np.clip(cf_d_raw, 0, 1)
            product = (short_sigma - long_sigma) * (long_nu - short_nu)
            delta = short_row["ATE"] - long_row["ATE"]
            denominator = math.sqrt(product) if short_sigma > long_sigma and long_nu > short_nu else 0
            rho_raw = delta / denominator if denominator > 0 else np.sign(delta)
            rho = np.sign(delta) * np.clip(abs(rho_raw), 0, 1)
            valid = 0 <= cf_y_raw < 1 and 0 <= cf_d_raw < 1 and np.isfinite(rho_raw) and abs(rho_raw) <= 1
            rows.append({"benchmark_block": block, "dropped_features": "|".join(drop),
                         "cf_y": cf_y, "cf_d": cf_d, "rho_observed": rho,
                         "cf_y_raw": cf_y_raw, "cf_d_raw": cf_d_raw, "rho_raw": rho_raw,
                         "cf_d_gain_raw": cf_d_gain,
                         "benchmark_definition": "DoubleML_gain_statistics_converted_to_fraction",
                         "delta_theta": short_row["ATE"] - long_row["ATE"],
                         "status": "ok" if valid else "finite_sample_invalid"})
            if valid:
                for multiplier in [1, 2]:
                    fy = cf_y * multiplier
                    gain = cf_d_gain * multiplier
                    fd = gain / (1 + gain)
                    if 0 <= fy < 1 and 0 <= fd < 1:
                        low, high = point_bounds(long_row["ATE"], long_sigma, long_nu, fy, fd)
                        rows[-1][f"bound_{multiplier}x_lower"] = low
                        rows[-1][f"bound_{multiplier}x_upper"] = high
                        rows[-1][f"bound_{multiplier}x_status"] = "ok"
                    else:
                        rows[-1][f"bound_{multiplier}x_status"] = "strength_outside_fraction_range"
            if store:
                store.save("benchmark:" + block, rows[-1])
        except (ValueError, ZeroDivisionError) as exc:
            rows.append({"benchmark_block": block, "status": "failed", "reason": str(exc)})
    return pd.DataFrame(rows)


def doubleml_check(p):
    """Official-package reference using the same retained OOF predictions."""
    import doubleml as dml
    from sklearn.dummy import DummyClassifier
    data = pd.DataFrame({"y": p.y.to_numpy(), "a": p.a.to_numpy(), "x": np.zeros(len(p))})
    fold_values = sorted(p.fold_id.unique()) if "fold_id" in p else None
    obj = dml.DoubleMLIRM(dml.DoubleMLData(data, "y", "a"), DummyClassifier(), DummyClassifier(),
                         n_folds=len(fold_values) if fold_values else 2, normalize_ipw=False,
                         trimming_threshold=1e-6, draw_sample_splitting=not bool(fold_values))
    if fold_values:
        obj.set_sample_splitting([(np.flatnonzero(p.fold_id.to_numpy() != fold),
                                   np.flatnonzero(p.fold_id.to_numpy() == fold)) for fold in fold_values])
    predictions = {"a": {"ml_g0": p.mu0.to_numpy()[:, None], "ml_g1": p.mu1.to_numpy()[:, None],
                         "ml_m": p.propensity.to_numpy()[:, None]}}
    obj.fit(external_predictions=predictions)
    obj.sensitivity_analysis(rho=1.0)
    return {"ATE": float(obj.coef[0]), "sigma2": float(obj.sensitivity_elements["sigma2"].ravel()[0]),
            "nu2": float(obj.sensitivity_elements["nu2"].ravel()[0]),
            "RV_point": float(obj.sensitivity_params["rv"][0])}


def validate_doubleml_reference(local, theta, official):
    """Require numerical agreement before registering point-sensitivity output."""
    expected = {"ATE": theta, "sigma2": local["sigma2"], "nu2": local["nu2"], "RV_point": local["RV_point"]}
    tolerances = {"ATE": 1e-8, "sigma2": 1e-10, "nu2": 1e-10, "RV_point": 1e-5}
    failed = [name for name, value in expected.items()
              if not np.isclose(value, official[name], atol=tolerances[name], rtol=1e-8)]
    if failed:
        raise ValueError("DoubleML reference mismatch for " + ", ".join(failed) + "; sensitivity output is not registered.")
    return {**{"reference_" + name: value for name, value in official.items()},
            "reference_doubleml_status": "ok",
            "reference_RV_absolute_difference": abs(local["RV_point"] - official["RV_point"]),
            "reference_RV_absolute_tolerance": tolerances["RV_point"]}
