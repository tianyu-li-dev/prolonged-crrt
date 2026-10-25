"""Routine overlap displays. No empirical recalibration is applied."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .reporting import LABELS, COLORS, _save
from .storage import write_table


def outputs(run, results, destination, primary_key, prefix=""):
    rows, primary = [], {}
    for _, row in results.iterrows():
        file = Path(run) / row.analysis_id / "predictions_private" / f"{row.estimator_name}_{row.outcome_name}.parquet"
        if row.status == "failed" or not file.exists():
            continue
        p = pd.read_parquet(file)
        e = p.propensity.to_numpy()
        groups = [p.loc[p.a.eq(t), "propensity"].to_numpy() for t in [0, 1]]
        overlap = min(groups[0].max(), groups[1].max()) - max(groups[0].min(), groups[1].min())
        rows.append({k: row[k] for k in ["analysis_id", "threshold_no_diuretic", "threshold_diuretic", "outcome_name", "estimator_name"]} |
                    {"n_oof": len(p), "n_retained": int(p.retained.sum()), "empirical_range_overlap": float(overlap),
                     "ps_min": float(e.min()), "ps_max": float(e.max()),
                     "tail_001": float(((e < .01) | (e > .99)).mean()),
                     "tail_005": float(((e < .05) | (e > .95)).mean())})
        if row.analysis_id == primary_key and row.outcome_name == results.outcome_name.iloc[0]:
            primary[row.estimator_name] = p
    summary = pd.DataFrame(rows)
    write_table(summary, Path(destination) / "Supplement_overlap.csv")
    if not summary.empty:
        reference = summary.loc[summary.outcome_name.eq(results.outcome_name.iloc[0])]
        fig, axes = plt.subplots(1, len(reference.threshold_diuretic.unique()), figsize=(12, 3.8), squeeze=False)
        for ax, di in zip(axes.ravel(), sorted(reference.threshold_diuretic.unique())):
            for family, group in reference.loc[reference.threshold_diuretic.eq(di)].groupby("estimator_name"):
                group = group.sort_values("threshold_no_diuretic")
                ax.plot(group.threshold_no_diuretic, 100 * group.tail_005, "o-", color=COLORS[family], label=LABELS[family])
            ax.set(title=f"Diuretic threshold {di:g} mL/day", xlabel="No-diuretic threshold (mL/day)",
                   ylabel="OOF propensity outside [0.05, 0.95] (%)")
            ax.legend(fontsize=6)
        fig.suptitle(prefix + "Empirical overlap diagnostics, before main trimming")
        fig.tight_layout(); _save(fig, Path(destination) / "Fig7_overlap")
    if primary:
        fig, axes = plt.subplots(1, len(primary), figsize=(3.5 * len(primary), 3.5), squeeze=False)
        for ax, (family, p) in zip(axes.ravel(), primary.items()):
            for a, label, color in [(0, "Discontinued", "#0072B2"), (1, "Continued", "#D55E00")]:
                ax.hist(p.loc[p.a.eq(a), "propensity"], bins=np.linspace(0, 1, 16), density=True,
                        alpha=.5, color=color, label=label)
            ax.axvline(.01, linestyle="--", color="grey"); ax.axvline(.99, linestyle="--", color="grey")
            ax.set(title=LABELS[family], xlabel="Corrected OOF propensity", ylabel="Density")
            ax.legend(fontsize=7)
        fig.suptitle(prefix + "Primary-pair propensity distributions; no additional calibration")
        fig.tight_layout(); _save(fig, Path(destination) / "Fig8_primary_overlap")
    return summary
