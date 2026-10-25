"""Generate manuscript tables and figures from one verified run."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .storage import write_table, write_json, private_output
from .preprocessing import covariates
from .estimation import FAMILIES
from .integrity import verified_run, verified_table, result_inventory, validate_result, registered, stage_state, verify_artifact
from .checkpoints import exclusive_run

LABELS = {"logistic": "Ridge", "lasso": "Lasso", "elastic_net": "Elastic-net", "xgboost": "XGBoost", "rf": "Random forest"}
COLORS = dict(zip(FAMILIES, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]))
plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})

# Descriptive rows are independent of the model feature filter. Outcomes belong
# in Table 2 even though they must never enter propensity or outcome predictors.
TABLE2_VARIABLES = (
    "admission_age", "Female", "White", "Emergency admission", "Insurance, Medicare",
    "tz_last_urineoutput_24hr", "tz_last_mbp", "tz_last_spo2", "tz_last_heart_rate",
    "tz_last_temperature", "tz_last_resp_rate", "tz_last_lactate",
    "tz_last_baseline_sapsii", "tz_last_baseline_sofa", "net_fluid_balance_ml_start_to_tz",
    "diuretic_prior_24h", "active_at_tz_vasopressor", "active_at_tz_ventilation",
    "any_pre_tz_suspected_infection", "positive_fluid_balance_start_to_tz",
    "Glycopeptide", "Beta-lactams", "Carbapenems", "Aminoglycosides",
    *["tz_last_" + name for name in ["potassium", "sodium", "chloride", "bicarbonate", "ph_blood", "creatinine", "bun"]],
    "mortality_28days", "mortality_60days", "mortality_90days",
)


def _save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def table(frame, path):
    write_table(frame, Path(path).with_suffix(".csv"))


def characteristics(frame, columns=None):
    """Describe observed values; explicit rows may include non-model outcomes.

    Apply the structural-zero exposure convention without imputing laboratories.
    Selected but absent variables remain visible with unavailable values and
    missing counts. This selection does not change the model feature filter.
    """
    x = covariates(frame)
    if columns is not None:
        for name in columns:
            if name not in x and name in frame:
                x[name] = frame[name]
        x = x.reindex(columns=columns)
    groups = [pd.Series(True, index=frame.index), frame.intervention_status.eq(0), frame.intervention_status.eq(1)]
    rows = []
    for name in x:
        series = x[name]
        numeric = pd.api.types.is_numeric_dtype(series)
        binary = numeric and series.dropna().isin([0, 1]).all() and not name.startswith(("tz_last_", "mean_", "min_", "max_", "worst_"))
        levels = [None] if numeric and not binary else [1] if binary else sorted(series.dropna().astype(str).unique())
        for level in levels:
            row = {"variable": name, "category": "" if level is None else str(level)}
            means, variances = [], []
            for mask, label in zip(groups, ["overall", "discontinued", "continued"]):
                vals = series.loc[mask].dropna()
                row[f"{label}_n_observed"] = len(vals)
                row[f"{label}_n_missing"] = int(mask.sum()) - len(vals)
                if level is None:
                    row[label] = f"{vals.mean():.2f} ({vals.std():.2f}); median {vals.median():.2f} [{vals.quantile(.25):.2f}, {vals.quantile(.75):.2f}]" if len(vals) else "unavailable"
                    if label != "overall":
                        means.append(vals.mean()); variances.append(vals.var())
                else:
                    binary_values = vals.eq(level) if binary else vals.astype(str).eq(str(level))
                    proportion = binary_values.mean() if len(vals) else np.nan
                    row[label] = f"{binary_values.sum()} ({100*proportion:.1f}%)" if len(vals) else "unavailable"
                    if label != "overall":
                        means.append(proportion); variances.append(proportion * (1 - proportion))
            denominator = np.sqrt(np.mean(variances))
            row["absolute_standardized_difference"] = abs(means[1] - means[0]) / denominator if denominator > 0 else 0.0 if means[1] == means[0] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def treatment_diagram(path, prefix=""):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot([-24, 0, 24, 48], [1, 1, 1, 1], lw=3, color=COLORS["logistic"])
    ax.plot([-24, 0, 12], [0, 0, 0], lw=3, color=COLORS["rf"])
    ax.scatter([12], [0], color=COLORS["rf"])
    for time, label in [(0, "t0: eligibility"), (24, "tL: treatment classification")]:
        ax.axvline(time, color="grey", linestyle="--"); ax.text(time, 1.35, label, ha="center")
    ax.text(49, 1, "CRRT active at landmark: A=1", va="center")
    ax.text(13, 0, "CRRT stopped; inactive at landmark: A=0", va="center")
    ax.set(xlim=(-30, 85), ylim=(-.7, 1.8), yticks=[], xlabel="Hours from first eligibility crossing",
           title=prefix + "Landmark treatment definition (schematic)")
    fig.tight_layout(); _save(fig, path)


def flow_plot(frame, path, prefix=""):
    n_column = next((c for c in ["n_after", "n_remaining", "n"] if c in frame), None)
    text_column = next((c for c in ["step", "stage"] if c in frame), frame.columns[0])
    if n_column is None:
        raise ValueError("Cohort-flow count schema is not recognized.")
    fig, ax = plt.subplots(figsize=(8, max(3, .45 * len(frame))))
    for i, row in frame.reset_index(drop=True).iterrows():
        y = len(frame) - i
        ax.text(.5, y, f"{row[text_column]}\nN={int(row[n_column])}", ha="center", va="center",
                bbox={"boxstyle": "round", "facecolor": "#EEF4F8", "edgecolor": "#698A9F"})
    ax.set(xlim=(0, 1), ylim=(0, len(frame) + 1), title=prefix + "Primary-pair cohort flow")
    ax.axis("off"); fig.tight_layout(); _save(fig, path)


def ate_plot(results, path, prefix=""):
    diuretics = sorted(results.threshold_diuretic.unique())
    outcomes = [c for c in ["mortality_28days", "mortality_60days", "mortality_90days"] if c in set(results.outcome_name)]
    fig, axes = plt.subplots(len(outcomes), len(diuretics), figsize=(4.8 * len(diuretics), 3.7 * len(outcomes)), squeeze=False)
    parts = [pd.to_numeric(results[c], errors="coerce") for c in ["ATE", "ci_lower", "ci_upper"] if c in results]
    values = pd.concat(parts).dropna() if parts else pd.Series(dtype=float)
    lower = min(0, values.min()) if len(values) else -.5
    upper = max(0, values.max()) if len(values) else .5
    padding = max(.02, .08 * (upper - lower))
    for i, outcome in enumerate(outcomes):
        for j, diuretic in enumerate(diuretics):
            ax = axes[i, j]
            part = results.loc[results.outcome_name.eq(outcome) & results.threshold_diuretic.eq(diuretic)]
            thresholds = sorted(part.threshold_no_diuretic.unique())
            positions, labels = [], []
            for t, threshold in enumerate(thresholds):
                for k, family in enumerate(FAMILIES):
                    row = part.loc[part.threshold_no_diuretic.eq(threshold) & part.estimator_name.eq(family)]
                    if row.empty:
                        continue
                    y = t * 6 + k
                    positions.append(y); labels.append(f"{threshold:g}: {LABELS[family]}")
                    item = row.iloc[0]
                    if pd.notna(item.get("ATE")):
                        ax.plot(item.ATE, y, "o", color=COLORS[family], markersize=4)
                    if pd.notna(item.get("ci_lower")):
                        ax.hlines(y, item.ci_lower, item.ci_upper, color=COLORS[family])
                    elif item.status not in {"ok", "point_only"}:
                        ax.text(.02, y, "unavailable", fontsize=6, transform=ax.get_yaxis_transform())
            ax.axvline(0, color="grey", linestyle="--", lw=.8)
            ax.set(yticks=positions, yticklabels=labels, title=f"{outcome.split('_')[1]} after landmark; diuretic UO {diuretic:g}",
                   xlabel="Mortality risk difference")
            ax.invert_yaxis(); ax.tick_params(labelsize=7)
            ax.set_xlim(lower - padding, upper + padding)
    fig.suptitle(prefix + "Estimator-specific ATE and full-procedure bootstrap 95% CI", fontsize=11)
    fig.tight_layout(); _save(fig, path)


def cate_plot(frame, path, prefix=""):
    frame = frame.copy(); frame["label"] = frame.subgroup.str.replace("_", " ", regex=False) + ": " + frame.level.astype(str)
    labels = list(frame.label.drop_duplicates())
    fig, ax = plt.subplots(figsize=(10, max(4, .3 * len(labels))))
    for k, family in enumerate(FAMILIES):
        part = frame.loc[frame.estimator_name.eq(family)]
        for _, row in part.iterrows():
            y = labels.index(row.label) + (k - 2) * .12
            ax.hlines(y, row.cate_p25, row.cate_p75, color=COLORS[family], alpha=.7)
            ax.plot(row.mean_cate, y, "o", color=COLORS[family], markersize=3)
        ax.plot([], [], "o", color=COLORS[family], label=LABELS[family])
    ax.axvline(0, color="grey", linestyle="--")
    ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel="CATE mean and individual CATE 25th–75th percentiles",
           title=prefix + "Exploratory subgroup distributions; bars are not confidence intervals")
    ax.invert_yaxis(); ax.legend(fontsize=7); fig.tight_layout(); _save(fig, path)


def sensitivity_plots(frame, path, prefix=""):
    from .sensitivity import point_bounds
    valid = frame.loc[frame.RV_status.eq("ok")]
    for _, row in valid.iterrows():
        maximum = min(.95, max(.20, float(row.RV_point) * 1.3))
        strengths = np.linspace(0, maximum, 80)
        x, y = np.meshgrid(strengths, strengths)
        bias = np.sqrt(row.sigma2 * row.nu2 * x * y / (1 - y))
        bound = row.ATE - bias if row.ATE >= 0 else row.ATE + bias
        fig, ax = plt.subplots(figsize=(6, 4.5))
        mesh = ax.contourf(x * 100, y * 100, bound, levels=15, cmap="coolwarm")
        if bound.min() <= 0 <= bound.max():
            ax.contour(x * 100, y * 100, bound, levels=[0], colors="black")
        ax.plot(strengths * 100, strengths * 100, "k--", lw=.7)
        ax.set(xlabel="cf_y: residual outcome variation (%)", ylabel="cf_d: treatment precision gain (%)",
               title=prefix + LABELS[row.estimator_name] + " | " + row.outcome_name + " | point bounds")
        fig.colorbar(mesh, ax=ax, label="Confounding-adjusted point bound, not a CI")
        fig.tight_layout(); _save(fig, Path(path) / f"RV_{row.estimator_name}_{row.outcome_name}")


@exclusive_run("output")
def report(output, allow_incomplete=False, resume=True):
    output, manifest = verified_run(output)
    results = result_inventory(output, manifest)
    expected = manifest["expected_cells"]
    if len(results) != expected or results.duplicated(["analysis_id", "outcome_name", "estimator_name"]).any():
        raise ValueError("The result inventory is incomplete or has duplicate cells.")
    if not allow_incomplete and not results.status.eq("ok").all():
        raise ValueError("Incomplete cells exist. Use --allow-incomplete only to prepare a labeled review report.")
    for _, row in results.iterrows():
        validate_result(output, row, manifest["configuration"])
    prefix = ""
    config = manifest["configuration"]
    no, di = config["design"]["primary_pair"]
    key = f"uo_nodiu_{no:g}_diu_{di:g}"
    required_inputs = [output / "threshold_cohorts.csv"]
    for file in [output / "cate_subgroups.csv", output / "cate_status.csv", output / "sensitivity.csv", output / "sensitivity_benchmarks.csv"]:
        if file.exists():
            required_inputs.append(file)
    for name in ["analysis_frame.parquet", "subgroup_boundaries.csv", "cohort_flow.csv"]:
        file = output / key / name
        if registered(output, file):
            required_inputs.append(file)
    for file in required_inputs:
        verify_artifact(output, file)
    from .checkpoints import Checkpoints, retire_stage_markers
    store = Checkpoints(output / "_checkpoints" / "report", {"task": "report",
                        "inputs": stage_state(output), "allow_incomplete": allow_incomplete}, resume, owner=output)
    completed = store.load("report_stage:complete")
    if completed:
        print("Resume: verified publication outputs.", flush=True)
        return output / "publication_outputs"
    import uuid
    dest = output / ("_publication_stage_" + uuid.uuid4().hex)
    dest.mkdir()
    cohort_table = verified_table(output, output / "threshold_cohorts.csv")
    published_cohorts = cohort_table.reindex(columns=["threshold_no_diuretic", "threshold_diuretic", "n_landmark",
        "treated_landmark", "control_landmark", "status"]).rename(columns={"threshold_no_diuretic": "UO no diuretic",
        "threshold_diuretic": "UO diuretic", "n_landmark": "Landmark N", "treated_landmark": "Continued N",
        "control_landmark": "Stopped N", "status": "Status"})
    table(published_cohorts, dest / "Table1_threshold_cohorts")
    table(cohort_table, dest / "Supplement_cohort_audit")
    if registered(output, output / "cate_status.csv"):
        table(verified_table(output, output / "cate_status.csv"), dest / "Supplement_CATE_population_audit")
    primary_frame = output / key / "analysis_frame.parquet"
    if primary_frame.exists() and registered(output, primary_frame):
        frame = verified_table(output, primary_frame)
        all_characteristics = characteristics(frame)
        write_table(all_characteristics, dest / "Supplement_all_covariate_characteristics.csv")
        published = characteristics(frame, TABLE2_VARIABLES)
        published["variable"] = published.variable.str.replace("tz_last_", "", regex=False).str.replace("_", " ", regex=False)
        published["variable"] = published.variable.replace({"baseline sofa": "First-day SOFA", "baseline sapsii": "First-day SAPS II",
            "diuretic prior 24h": "Diuretic use in previous 24 h", "any pre tz suspected infection": "Suspected infection before time zero",
            "mortality 28days": "28-day death", "mortality 60days": "60-day death", "mortality 90days": "90-day death"})
        for col in ["overall", "continued", "discontinued"]:
            published[col] = published[col].str.split(";", n=1).str[0]
        published = published[["variable", "overall", "continued", "discontinued", "overall_n_missing", "continued_n_missing", "discontinued_n_missing",
                               "absolute_standardized_difference"]].rename(columns={"variable": "Variable", "continued": "Continued",
            "overall": "Overall", "overall_n_missing": "Missing overall",
            "discontinued": "Stopped", "continued_n_missing": "Missing continued", "discontinued_n_missing": "Missing stopped",
            "absolute_standardized_difference": "Absolute SMD"})
        table(published, dest / "Table2_characteristics")
    if registered(output, output / key / "subgroup_boundaries.csv"):
        table(verified_table(output, output / key / "subgroup_boundaries.csv"), dest / "Supplement_subgroup_boundaries")
    write_table(results, dest / "Supplement_all_ATE_audit.csv")
    compact_results = results.copy()
    compact_results["Family"] = compact_results.estimator_name.map(LABELS)
    compact_results["Days"] = compact_results.outcome_name.str.extract(r"(\d+)")
    compact_results["95% bootstrap CI"] = [f"[{row.get('ci_lower', np.nan):.3f}, {row.get('ci_upper', np.nan):.3f}]"
        if pd.notna(row.get("ci_lower")) else "unavailable" for _, row in compact_results.iterrows()]
    table(compact_results.reindex(columns=["threshold_no_diuretic", "threshold_diuretic", "Days", "Family", "ATE",
          "95% bootstrap CI", "n", "treated", "control", "status"]).rename(columns={"threshold_no_diuretic": "UO no diuretic",
          "threshold_diuretic": "UO diuretic", "n": "N", "treated": "Continued", "control": "Stopped", "status": "Status"}),
          dest / "Supplement_all_ATE_grid")
    compact_results["Primary pair"] = np.where(compact_results.analysis_id.eq(key), "yes", "no")
    published_ate = compact_results.reindex(columns=["threshold_no_diuretic", "threshold_diuretic", "Primary pair", "Days", "Family", "ATE", "95% bootstrap CI", "n", "treated", "control", "status"]).rename(
        columns={"threshold_no_diuretic": "UO no diuretic", "threshold_diuretic": "UO diuretic", "n": "N retained", "treated": "N continued", "control": "N stopped", "status": "Status"})
    table(published_ate, dest / "Table3_ATE")
    treatment_diagram(dest / "Fig1_treatment", prefix)
    if registered(output, output / key / "cohort_flow.csv"):
        flow_plot(verified_table(output, output / key / "cohort_flow.csv"), dest / "Fig2_flow", prefix)
    ate_plot(results, dest / "Fig3_ATE", prefix)
    for diuretic in sorted(results.threshold_diuretic.unique()):
        for outcome in results.outcome_name.drop_duplicates():
            part = results.loc[results.threshold_diuretic.eq(diuretic) & results.outcome_name.eq(outcome)]
            ate_plot(part, dest / f"Supplement_ATE_diu_{diuretic:g}_{outcome}", prefix)
    from .overlap import outputs
    outputs(output, results, dest, key, prefix)
    cate_path = output / "cate_subgroups.csv"
    if cate_path.exists() and cate_path.stat().st_size > 4:
        cates = verified_table(output, cate_path)
        table(cates, dest / "Supplement_CATE_distributions")
        for i, outcome in enumerate(["mortality_28days", "mortality_60days", "mortality_90days"], 4):
            part = cates.loc[cates.analysis_id.eq(key) & cates.outcome_name.eq(outcome)]
            if not part.empty:
                cate_plot(part, dest / f"Fig{i}_CATE_{outcome}", prefix)
    if (output / "sensitivity.csv").exists():
        sensitivity = verified_table(output, output / "sensitivity.csv")
        write_table(sensitivity, dest / "Supplement_point_sensitivity_audit.csv")
        compact = sensitivity.copy()
        compact["Days"] = compact.outcome_name.str.extract(r"(\d+)")
        compact["Family"] = compact.estimator_name.map(LABELS)
        if "nu2_method" in compact:
            fallback = compact.nu2_method.eq("riesz_squared_fallback")
            compact.loc[fallback, "RV_status"] = compact.loc[fallback, "RV_status"].astype(str) + "; fallback"
        compact = compact.reindex(columns=["analysis_id", "Days", "Family", "risk_ratio", "E_value_point", "RV_percent",
                                           "E_value_status", "RV_status"]).rename(columns={"risk_ratio": "Adjusted RR",
            "E_value_point": "Point E-value", "RV_percent": "Point RV (%)", "E_value_status": "E status", "RV_status": "RV status"})
        table(compact, dest / "Supplement_point_sensitivity")
        table(compact.loc[compact.analysis_id.eq(key)].drop(columns="analysis_id"), dest / "Table4_primary_point_sensitivity")
        if "TMLE_E_value_point" in sensitivity:
            tmle = sensitivity.loc[sensitivity.TMLE_E_value_point.notna()].copy()
            tmle["Days"] = tmle.outcome_name.str.extract(r"(\d+)")
            tmle["Family"] = tmle.estimator_name.map(LABELS)
            tmle["Method"] = "Separate bounded TMLE"
            table(tmle.reindex(columns=["analysis_id", "Days", "Family", "Method", "TMLE_risk_difference",
                                        "TMLE_risk_ratio", "TMLE_E_value_point"]).rename(columns={
                "TMLE_risk_difference": "Risk difference", "TMLE_risk_ratio": "Adjusted RR", "TMLE_E_value_point": "Point E-value"}),
                dest / "Supplement_separate_TMLE_sensitivity")
        primary = sensitivity.loc[sensitivity.analysis_id.eq(key)]
        if "RV_status" in primary:
            sensitivity_plots(primary, dest / "sensitivity_figures", prefix)
    if (output / "sensitivity_benchmarks.csv").exists():
        benchmarks = verified_table(output, output / "sensitivity_benchmarks.csv")
        write_table(benchmarks, dest / "Supplement_confounding_benchmarks_audit.csv")
        benchmarks["Days"] = benchmarks.outcome_name.str.extract(r"(\d+)")
        benchmarks["Family"] = benchmarks.estimator_name.map(LABELS)
        for multiplier in [1, 2]:
            benchmarks[f"{multiplier}x point bounds"] = [
                f"[{row.get(f'bound_{multiplier}x_lower', np.nan):.3f}, {row.get(f'bound_{multiplier}x_upper', np.nan):.3f}]"
                if pd.notna(row.get(f"bound_{multiplier}x_lower")) else "unavailable" for _, row in benchmarks.iterrows()]
        table(benchmarks.reindex(columns=["Days", "Family", "benchmark_block", "delta_theta", "cf_y_raw", "cf_d_raw",
                                         "1x point bounds", "2x point bounds", "status"]).rename(columns={
            "benchmark_block": "Block", "delta_theta": "ATE change", "cf_y_raw": "Raw cf_y", "cf_d_raw": "Raw cf_d", "status": "Status"}),
            dest / "Supplement_confounding_benchmarks")
    captions = (prefix + "Data origin: " + manifest["data_kind"] + "\n"
                "Fig1: Landmark status defines treatment. The trajectory is schematic.\n"
                "Fig2: Primary-pair eligibility, landmark survival, and classifiability.\n"
                "Fig3: Mortality horizons start at tL. Bars show full-procedure bootstrap 95% confidence intervals.\n"
                "Fig4–6: Exploratory EconML X-learner subgroup means in the independent CATE PS-trimmed population. "
                "The main ATE OOF mask is not used. Bars show individual CATE quartiles, not confidence intervals.\n"
                "Table2: Overall and treatment-arm observed values and missing counts before PS trimming. "
                "The descriptive row list includes vital signs, pre-time-zero exposures and all three mortality horizons, independent of model feature selection. "
                "First-day scores are static, with no collection-window restriction. "
                "Standardized differences compare the two arms. No P values are used.\n"
                "Table3: All configured threshold pairs, horizons and model families; the primary pair is marked.\n"
                "Table4: Point E-values and IRM RV. These do not restore statistical evidence when the ATE CI includes zero. "
                "RV status 'fallback' identifies the documented non-orthogonal Riesz moment.\n")
    (dest / "captions.txt").write_text(captions, encoding="utf-8")
    write_json({"configuration": config, "data_kind": manifest["data_kind"],
                "files": sorted(str(p.relative_to(dest)) for p in dest.rglob("*")
                                if p.is_file() and p.name != "publication_manifest.json")},
               dest / "publication_manifest.json")
    from .storage import commit_publication
    retire_stage_markers(output, "report_stage:complete")
    dest = commit_publication(output, dest)
    store.save("report_stage:complete", {}, [path for path in dest.rglob("*") if path.is_file()])
    return dest
