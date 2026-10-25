"""Local commands. Importing the package does not run an analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from .config import load_config, pairs
from .storage import write_table
from .integrity import verified_run, verified_table, result_inventory, validate_result, register_stage, invalidate_stage, stage_state
from .checkpoints import Checkpoints, exclusive_run, stage_files, retire_stage_markers


def _verified_run(path):
    output, manifest = verified_run(path)
    return output, manifest["configuration"]


@exclusive_run("path")
def sensitivity_run(path, benchmarks=False, reference=True, resume=True):
    from .estimation import estimate
    from .sensitivity import summarize, benchmark, doubleml_check, validate_doubleml_reference
    output, config = _verified_run(path)
    results = result_inventory(output, verified_run(output)[1])
    stage_files(output, "design"); stage_files(output, "estimation")
    store = Checkpoints(output / "_checkpoints" / "sensitivity", {"task": "sensitivity",
                        "inputs": stage_state(output, ("design", "estimation")), "benchmarks": benchmarks, "reference": reference},
                        resume, owner=output)
    cached_stage = store.load("sensitivity_stage:complete")
    if cached_stage and not cached_stage["payload"]["failed_benchmarks"]:
        for _, row in results.iterrows():
            validate_result(output, row, config)
        register_stage(output, "sensitivity", [output / name for name in cached_stage["files"]])
        print("Resume: verified sensitivity stage.", flush=True)
        return pd.read_csv(output / "sensitivity.csv")
    invalidate_stage(output, "sensitivity")
    rows, benchmark_rows = [], []
    for _, row in results.iterrows():
        context = row.to_dict()
        cell = row.analysis_id + ":" + row.estimator_name + "_" + row.outcome_name
        cached = store.load("sensitivity:" + cell)
        file = output / row.analysis_id / "predictions_private" / f"{row.estimator_name}_{row.outcome_name}.parquet"
        if row.status == "failed":
            rows.append({**context, "E_value_status": "unavailable", "RV_status": "unavailable",
                         "sensitivity_reason": "Main estimate is unavailable."}); continue
        p, point, selected, mask = validate_result(output, row, config)
        if cached:
            sensitivity = cached["payload"]
        else:
            sensitivity = summarize(selected, point["ATE"])
            if reference and pd.notna(sensitivity["RV_point"]):
                official = doubleml_check(selected)
                sensitivity.update(validate_doubleml_reference(sensitivity, point["ATE"], official))
            else:
                sensitivity["reference_doubleml_status"] = "local_RV_unavailable" if reference else "explicitly_disabled"
            store.save("sensitivity:" + cell, sensitivity)
        rows.append({**context, **sensitivity})
        no, di = config["design"]["primary_pair"]
        if benchmarks and row.analysis_id == f"uo_nodiu_{no:g}_diu_{di:g}":
            frame = verified_table(output, output / row.analysis_id / "analysis_frame.parquet")
            item = benchmark(frame, row.outcome_name, row.estimator_name, p, mask, config,
                             checkpoint=store.directory("sensitivity:" + cell) / "benchmarks")
            for key in ["analysis_id", "outcome_name", "estimator_name"]:
                item[key] = context[key]
            benchmark_rows.append(item)
    retire_stage_markers(output, "sensitivity_stage:complete")
    write_table(pd.DataFrame(rows), output / "sensitivity.csv")
    files = [output / "sensitivity.csv"]
    if benchmark_rows:
        write_table(pd.concat(benchmark_rows, ignore_index=True), output / "sensitivity_benchmarks.csv")
        files.append(output / "sensitivity_benchmarks.csv")
    elif (output / "sensitivity_benchmarks.csv").exists():
        import uuid
        archive = output / "sensitivity_archive" / uuid.uuid4().hex
        archive.mkdir(parents=True)
        (output / "sensitivity_benchmarks.csv").replace(archive / "sensitivity_benchmarks.csv")
    register_stage(output, "sensitivity", files)
    store.save("sensitivity_stage:complete", {"failed_benchmarks": any(item.status.eq("failed").any() for item in benchmark_rows)}, files)
    return pd.DataFrame(rows)


def _dispatch(argv=None):
    parser = argparse.ArgumentParser(description="Landmark causal analysis of prolonged CRRT.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["plan", "build", "estimate", "run"]:
        item = commands.add_parser(name)
        item.add_argument("--config", required=True)
        if name != "plan":
            item.add_argument("--output", required=True)
            item.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
        if name in {"estimate", "run"}:
            item.add_argument("--allow-sparse-labs", action=argparse.BooleanOptionalAction, default=None)
    item = commands.add_parser("inspect-sources", help="Read-only check of existing relations or local source files.")
    item.add_argument("--config", required=True)
    item = commands.add_parser("sensitivity")
    item.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    item.add_argument("--run", required=True)
    item.add_argument("--benchmarks", action="store_true")
    item.add_argument("--reference-doubleml", action=argparse.BooleanOptionalAction, default=True)
    item = commands.add_parser("report")
    item.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    item.add_argument("--run", required=True)
    item.add_argument("--allow-incomplete", action="store_true")
    item = commands.add_parser("status", help="Read local checkpoint counts without running an analysis.")
    item.add_argument("--run", required=True)
    item = commands.add_parser("check-release")
    item.add_argument("--root", default=str(Path.cwd()))
    args = parser.parse_args(argv)
    if args.command == "status":
        from .checkpoints import progress
        print(json.dumps(progress(args.run), indent=2)); return
    if args.command == "plan":
        c = load_config(args.config)
        print(json.dumps({"threshold_pairs": list(pairs(c)), "ate_cells": len(list(pairs(c))) *
                          len(c["analysis"]["outcomes"]) * len(c["analysis"]["families"]),
                          "configuration": c}, indent=2)); return
    if args.command == "inspect-sources":
        from .data import inspect_sources
        print(json.dumps(inspect_sources(load_config(args.config)), indent=2)); return
    if args.command in {"build", "estimate", "run"}:
        from .pipeline import build, analyze
        config = load_config(args.config)
        if args.command in {"build", "run"}:
            build(config, args.output, resume=args.resume)
        if args.command in {"estimate", "run"}:
            analyze(config, args.output, args.allow_sparse_labs, resume=args.resume)
        print("Local output:", str(Path(args.output).resolve()))
        return
    if args.command == "sensitivity":
        sensitivity_run(args.run, args.benchmarks, args.reference_doubleml, resume=args.resume)
    elif args.command == "report":
        from .reporting import report
        print(report(args.run, args.allow_incomplete, resume=args.resume))
    elif args.command == "check-release":
        from .release import check
        print(json.dumps(check(args.root), indent=2))


def main(argv=None):
    try:
        return _dispatch(argv)
    except KeyboardInterrupt:
        print("Stopped. Committed checkpoints are saved. Repeat the same command and output path to resume.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
