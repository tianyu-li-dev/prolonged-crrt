"""Track completed run stages and validate numerical result structure."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .storage import write_json, read_table, private_output


def registry(run):
    path = Path(run) / "artifact_registry.json"
    if not path.is_file():
        raise ValueError("The run has no artifact inventory. Build into a new output directory.")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record.get("artifacts"), list) or "updates" not in record:
        raise ValueError("This run uses an unsupported checkpoint layout. Use a new output directory.")
    return record


def reset_registry(run):
    write_json({"artifacts": [], "stages": {}, "updates": {}}, Path(run) / "artifact_registry.json")


def register_stage(run, stage, files):
    root = Path(run).resolve()
    record = registry(root)
    names = []
    for file in files:
        file = Path(file).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError("A stage output must be an existing file inside its run directory.")
        names.append(file.relative_to(root).as_posix())
    names = sorted(set(names))
    if record["stages"].get(stage) != names:
        record["updates"][stage] = record["updates"].get(stage, 0) + 1
    record["stages"][stage] = names
    record["artifacts"] = sorted(set().union(*map(set, record["stages"].values())))
    write_json(record, root / "artifact_registry.json")


def invalidate_stage(run, stage):
    root = Path(run)
    record = registry(root)
    affected = {stage}
    if stage == "design":
        affected.update(["estimation", "sensitivity"])
    if "estimation" in affected:
        affected.add("sensitivity")
        affected.update(k for k in record["stages"] if k.startswith(("point:", "cell:", "cate:")))
    for name in affected:
        record["stages"].pop(name, None)
        record["updates"][name] = record["updates"].get(name, 0) + 1
    record["artifacts"] = sorted(set().union(*map(set, record["stages"].values())))
    write_json(record, root / "artifact_registry.json")


def stage_state(run, stages=("design", "estimation", "sensitivity")):
    record = registry(run)
    return {name: record["updates"].get(name, 0) for name in stages}


def registered(run, file):
    name = Path(file).resolve().relative_to(Path(run).resolve()).as_posix()
    return name in registry(run)["artifacts"]


def verify_artifact(run, file):
    root, file = Path(run).resolve(), Path(file).resolve()
    if not file.is_relative_to(root):
        raise ValueError("Artifact escapes its run directory.")
    name = file.relative_to(root).as_posix()
    if name not in registry(root)["artifacts"] or not file.is_file():
        raise ValueError("Stage output is missing or is not registered: " + name)
    return file


def verified_table(run, file):
    return read_table(verify_artifact(run, file))


def verified_run(run):
    from .pipeline import _restore_run_manifest
    root = private_output(run)
    registry(root)
    initial = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if initial.get("checkpoint_storage") != "named_tasks":
        raise ValueError("This run uses an unsupported checkpoint layout. Use a new output directory.")
    _restore_run_manifest(root, initial["configuration"])
    return root, json.loads(verify_artifact(root, root / "run_manifest.json").read_text(encoding="utf-8"))


def result_inventory(run, manifest):
    from .config import pairs
    path = verify_artifact(run, Path(run) / "all_ate_results.csv")
    rows = pd.read_csv(path)
    config = manifest["configuration"]
    expected = {(key, outcome, family) for _, _, key in pairs(config)
                for outcome in config["analysis"]["outcomes"] for family in config["analysis"]["families"]}
    keys = ["analysis_id", "outcome_name", "estimator_name"]
    actual = set(rows[keys].itertuples(index=False, name=None))
    if actual != expected or len(rows) != len(expected) or rows.duplicated(keys).any():
        raise ValueError("Result inventory does not match the complete declared grid.")
    return rows


def validate_result(run, row, config):
    from .estimation import estimate
    if row.status == "failed":
        return None
    stem = f"{row.estimator_name}_{row.outcome_name}"
    file = Path(run) / row.analysis_id / "predictions_private" / (stem + ".parquet")
    p = verified_table(run, file)
    point, selected, mask = estimate(p, config["analysis"]["trimming"], p.retained.to_numpy())
    for field in ["ATE", "if_standard_error", "n", "treated", "control", "n_before_trimming"]:
        if not np.isclose(point[field], row[field], rtol=0, atol=1e-9):
            raise ValueError(f"Main result differs from saved predictions: {field}")
    if row.bootstrap_requested:
        boot = verified_table(run, Path(run) / row.analysis_id / "bootstrap_private" / (stem + ".csv"))
        successful = boot.loc[boot.status.eq("ok"), "ATE"] if "ATE" in boot else pd.Series(dtype=float)
        if len(boot) != int(row.bootstrap_requested) or len(successful) != int(row.bootstrap_completed):
            raise ValueError("Bootstrap inventory differs from the main result.")
        if row.status == "ok":
            for column, quantile in [("ci_lower", .025), ("ci_upper", .975)]:
                if not np.isclose(row[column], successful.quantile(quantile), rtol=0, atol=1e-9):
                    raise ValueError("Main confidence interval differs from the saved bootstrap.")
    return p, point, selected, mask


