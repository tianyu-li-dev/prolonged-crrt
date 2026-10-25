"""Build clinical cohorts and estimate treatment effects."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import pairs
from .cohort import build_landmark_cohort
from .data import load_sources, admission_fields
from .laboratory import normalize_labs, build_features, MAIN_LABS
from .estimation import crossfit, estimate, bootstrap
from .cate import xlearner, subgroup_reference
from .storage import private_output, write_table, write_json
from .integrity import reset_registry, register_stage, invalidate_stage, verify_artifact, stage_state, registry
from .checkpoints import Checkpoints, CheckpointError, exclusive_run, stage_files


def provenance(config):
    return {"configuration": config, "checkpoint_storage": "named_tasks", "data_kind": "credentialed_MIMIC_IV",
            "outcome_origin": "landmark", "propensity_correction": "full_complete_sample_class_probabilities",
            "database_policy": "read_only", "death_source": "patient_dod",
            "cohort_rules": "urine_episode_end_inclusive",
            "existing_covariates": "static_scores_latest_modality_filtered_ventilation",
            "keep_empty_features": config["features"]["keep_empty_features"],
            "cate_engine": "econml.metalearners.XLearner", "cate_trimming_source": "independent_CATE_full_sample_PS",
            "started_utc": datetime.now(timezone.utc).isoformat(), "status": "started"}


def _save_run_manifest(output, manifest):
    """Commit a recovery receipt before updating the manifest and inventory."""
    store = Checkpoints(Path(output) / "_checkpoints" / "run_state",
                        {"configuration": manifest["configuration"]}, owner=output)
    store.save("run_state:desired", manifest)
    write_json(manifest, Path(output) / "run_manifest.json")
    register_stage(output, "run", [Path(output) / "run_manifest.json"])


def _restore_run_manifest(output, config):
    root = Path(output)
    path = root / "run_manifest.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if current and (current.get("checkpoint_storage") != "named_tasks" or current["configuration"] != config):
        raise ValueError("Run settings or checkpoint layout differ. Use a new output directory.")
    if (root / "artifact_registry.json").exists():
        registry(root)
    else:
        reset_registry(root)
    store = Checkpoints(root / "_checkpoints" / "run_state", {"configuration": config}, owner=root)
    saved = store.load("run_state:desired")
    if saved:
        desired = saved["payload"]
        if desired["configuration"] != config:
            raise CheckpointError("Saved run settings differ from the requested configuration.")
        if current != desired:
            write_json(desired, path)
        register_stage(root, "run", [path])


@exclusive_run("output")
def build(config, output, supplied_sources=None, resume=True):
    output = private_output(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = provenance(config)
    existing = output / "run_manifest.json"
    _restore_run_manifest(output, config)
    if existing.exists():
        manifest = json.loads(existing.read_text(encoding="utf-8"))
        if manifest["configuration"] != config:
            raise ValueError("This output directory belongs to another configuration. Use a new output path.")
        if resume:
            if supplied_sources is not None:
                raise ValueError("Resume uses the saved source snapshot. Supply new data in a new output directory.")
            completed = stage_files(output, "design")
            if completed and not pd.read_csv(output / "threshold_cohorts.csv").status.eq("failed").any():
                print("Resume: completed design; source extraction and threshold builds are skipped.", flush=True)
                return output
    written = []
    _save_run_manifest(output, manifest)
    context = {"configuration": config}
    source_store = Checkpoints(output / "_checkpoints" / "sources", context, resume, owner=output)
    frozen = source_store.load("sources:complete")
    if frozen:
        s = {name: pd.read_parquet(output / path) for name, path in frozen["payload"]["paths"].items()}
        print("Resume: using the saved local source snapshot.", flush=True)
    else:
        s = supplied_sources if supplied_sources is not None else load_sources(config, output)
        paths = {name: "sources/" + name + ".parquet" for name in s}
        for name, path in paths.items():
            write_table(s[name], output / path)
        source_store.save("sources:complete", {"paths": paths}, [output / path for path in paths.values()])
    invalidate_stage(output, "design")
    store = Checkpoints(output / "_checkpoints" / "design", context, resume, owner=output)
    labs, lab_audit = normalize_labs(s["labevents"], s["d_labitems"], **{
        k: config["features"][k] for k in ["chemistry_fallback", "arterial_only", "strict_storetime"]})
    s = {**s, "laboratory": labs}
    write_table(lab_audit, output / "laboratory_event_audit.csv")
    written.append(output / "laboratory_event_audit.csv")
    cohort_rows = []
    for no, di, key in pairs(config):
        dest = output / key
        cached = store.load("design:" + key)
        if cached:
            cohort_rows.append(cached["payload"])
            written.extend(output / path for path in cached["files"])
            print("Resume: design " + key, flush=True)
            continue
        pair_files = []
        try:
            d = config["design"]
            stays = admission_fields(s["stays"], s["admissions"])
            cohort = build_landmark_cohort(stays, s["crrt_records"], s["kdigo_uo"], s["diuretics"],
                                           uo_threshold_no_diuretic=no, uo_threshold_diuretic=di,
                                           merge_gap_hours=d["merge_gap_hours"], landmark_hours=d["landmark_hours"])
            if cohort.cohort.empty:
                raise ValueError("No patients survive and remain classifiable at the landmark.")
            frame, metadata, audit = build_features(cohort.cohort, s, config["features"]["laboratories"],
                                                    config["features"]["window_hours"])
            _, subgroup_boundaries = subgroup_reference(frame)
            write_table(subgroup_boundaries, dest / "subgroup_boundaries.csv")
            written.append(dest / "subgroup_boundaries.csv")
            pair_files.append(dest / "subgroup_boundaries.csv")
            for name, item in [("analysis_frame", frame), ("feature_dictionary", metadata), ("laboratory_audit", audit),
                               ("cohort_flow", cohort.cohort_flow), ("episode_summary", cohort.multi_episode_summary),
                               ("early_deaths_private", cohort.deaths_timezero_to_landmark),
                               ("threshold_exclusions", cohort.threshold_exclusion_summary)]:
                write_table(item, dest / (name + (".parquet" if name == "analysis_frame" else ".csv")))
                written.append(dest / (name + (".parquet" if name == "analysis_frame" else ".csv")))
                pair_files.append(dest / (name + (".parquet" if name == "analysis_frame" else ".csv")))
            quality = audit.loc[audit.treatment_group.eq("all") & audit.variable.isin(MAIN_LABS)]
            unavailable = quality.loc[quality.completeness.eq(0), "variable"].tolist()
            sparse = quality.loc[quality.completeness.lt(.5), "variable"].tolist()
            row = {"analysis_id": key, "threshold_no_diuretic": no, "threshold_diuretic": di,
                   "n_landmark": len(frame), "treated_landmark": int(frame.intervention_status.sum()),
                   "control_landmark": int(frame.intervention_status.eq(0).sum()),
                   "unavailable_laboratories": "|".join(unavailable), "sparse_laboratories": "|".join(sparse),
                   "status": "unavailable_labs" if unavailable else "sparse_labs" if sparse else "ready"}
            write_json(row, dest / "design_status.json")
            written.append(dest / "design_status.json")
            pair_files.append(dest / "design_status.json")
            store.save("design:" + key, row, pair_files)
            cohort_rows.append(row)
        except ValueError as exc:
            row = {"analysis_id": key, "threshold_no_diuretic": no, "threshold_diuretic": di,
                   "status": "failed", "reason": str(exc)}
            write_json(row, dest / "design_status.json"); cohort_rows.append(row)
            written.append(dest / "design_status.json")
        write_table(pd.DataFrame(cohort_rows), output / "threshold_cohorts.csv")
    write_table(pd.DataFrame(cohort_rows), output / "threshold_cohorts.csv")
    manifest["status"] = "design_built"
    _save_run_manifest(output, manifest)
    register_stage(output, "design", written + [output / "threshold_cohorts.csv"])
    return output


@exclusive_run("output")
def analyze(config, output, allow_sparse=None, resume=True):
    output = private_output(output)
    _restore_run_manifest(output, config)
    verify_artifact(output, output / "run_manifest.json")
    manifest = json.loads((output / "run_manifest.json").read_text())
    if manifest["configuration"] != config:
        raise ValueError("Analysis configuration does not match the built design.")
    a = config["analysis"]
    allow_sparse = a["allow_sparse_labs"] if allow_sparse is None else bool(allow_sparse)
    previous_stage = stage_files(output, "estimation") if resume else []
    store = Checkpoints(output / "_checkpoints" / "estimation", {"configuration": config,
                        "design": stage_state(output, ("design",)), "allow_sparse": allow_sparse}, resume, owner=output)
    changed = False
    main_changed = False

    def begin_work(main):
        nonlocal changed, main_changed
        if not changed:
            invalidate_stage(output, "estimation")
            manifest["status"] = "analysis_in_progress"
            _save_run_manifest(output, manifest)
            changed = True
        if main and not main_changed:
            invalidate_stage(output, "sensitivity")
            main_changed = True
    written = []
    results, cates, cate_status = [], [], []
    ordered_pairs = sorted(pairs(config), key=lambda item: [item[0], item[1]] != config["design"]["primary_pair"])
    for no, di, key in ordered_pairs:
        dest = output / key
        design = json.loads(verify_artifact(output, dest / "design_status.json").read_text())
        ready = design["status"] == "ready" or (allow_sparse and design["status"] == "sparse_labs")
        frame = pd.read_parquet(dest / "analysis_frame.parquet") if ready else None
        for outcome in a["outcomes"]:
            for family in a["families"]:
                context = {"analysis_id": key, "threshold_no_diuretic": no, "threshold_diuretic": di,
                           "outcome_name": outcome, "estimator_name": family, "outcome_origin": "landmark",
                           "landmark_hours": config["design"]["landmark_hours"], "bootstrap_requested": a["bootstrap_samples"]}
                stem = f"{family}_{outcome}"
                cell = key + ":" + stem
                cached = store.load("ate:" + cell)
                point_cache = store.load("point:" + cell)
                try:
                    if not ready:
                        raise ValueError("Design is not ready: " + design["status"] + "; " + design.get("reason", ""))
                    if cached and cached["payload"]["status"] in {"ok", "point_only"}:
                        from .integrity import validate_result
                        cell_files = [output / path for path in cached["files"]]
                        register_stage(output, "cell:" + cell, cell_files)
                        validate_result(output, pd.Series(cached["payload"]), config)
                        results.append(cached["payload"])
                        written.extend(cell_files)
                        print("Resume: ATE " + cell, flush=True)
                    else:
                        begin_work(True)
                        point_files = [dest / "predictions_private" / (stem + ".parquet"), dest / "feature_roles" / (stem + ".json")]
                        if point_cache:
                            row = point_cache["payload"]
                            register_stage(output, "point:" + cell, point_files)
                            from .integrity import validate_result
                            validate_result(output, pd.Series({**context, **row, "status": "point_only", "bootstrap_requested": 0}), config)
                        else:
                            fitted = crossfit(frame, outcome, family, a["folds"], a["seed"], config["features"]["missing_indicators"],
                                              keep_empty_features=config["features"]["keep_empty_features"],
                                              checkpoint=store.directory("point:" + cell) / "folds")
                            row, selected, mask = estimate(fitted["predictions"], a["trimming"])
                            prediction = fitted["predictions"].copy()
                            prediction["retained"] = mask
                            write_table(prediction, point_files[0])
                            write_json({"features": fitted["feature_names"], "sampling_probs_full_sample": fitted["sampling_probs"],
                                        "empty_training_columns_by_fold": fitted["empty_training_columns_by_fold"],
                                        "keep_empty_features": config["features"]["keep_empty_features"]}, point_files[1])
                            register_stage(output, "point:" + cell, point_files)
                            store.save("point:" + cell, row, point_files)
                        cell_files = list(point_files)
                        row.update(status="point_only", ci_method="not_computed", ci_lower=np.nan, ci_upper=np.nan,
                                   bootstrap_completed=0, bootstrap_standard_error=np.nan)
                        if a["bootstrap_samples"]:
                            draws = bootstrap(frame, outcome, family, a["bootstrap_samples"], a["folds"], a["seed"],
                                              a["trimming"], a["bootstrap_jobs"], config["features"]["missing_indicators"],
                                              config["features"]["keep_empty_features"],
                                              checkpoint=store.directory("ate:" + cell) / "bootstrap")
                            boot_path = dest / "bootstrap_private" / (stem + ".csv")
                            write_table(draws, boot_path)
                            cell_files.append(boot_path)
                            valid = draws.loc[draws.status.eq("ok"), "ATE"] if "ATE" in draws else pd.Series(dtype=float)
                            row["bootstrap_completed"] = len(valid)
                            row["bootstrap_standard_error"] = float(valid.std(ddof=1))
                            if len(valid) >= max(2, int(np.ceil(a["minimum_bootstrap_success"] * a["bootstrap_samples"]))):
                                row.update(ci_lower=float(valid.quantile(.025)), ci_upper=float(valid.quantile(.975)),
                                           ci_method="full_procedure_percentile", status="ok")
                            else:
                                row["status"] = "bootstrap_incomplete"
                        result = {**context, **row}
                        results.append(result)
                        written.extend(cell_files)
                        register_stage(output, "cell:" + cell, cell_files)
                        store.save("ate:" + cell, result, cell_files)
                except ValueError as exc:
                    if cached or point_cache:
                        raise CheckpointError("Cached ATE failed mathematical verification: " + cell) from exc
                    begin_work(True)
                    results.append({**context, "status": "failed", "reason": str(exc)})
                if changed or not previous_stage:
                    write_table(pd.DataFrame(results), output / "all_ate_results.csv")
                # CATE is independent of ATE success, OOF predictions and retained mask.
                if not a["cate_primary_only"] or [no, di] == config["design"]["primary_pair"]:
                    cate_cache = store.load("cate:" + cell)
                    try:
                        if not ready:
                            raise ValueError("Design is not ready: " + design["status"])
                        subgroup_file = dest / "cate_private" / (stem + "_subgroups.parquet")
                        if cate_cache:
                            cates.append(pd.read_parquet(subgroup_file))
                            cate_status.append(cate_cache["payload"])
                            written.extend(output / path for path in cate_cache["files"])
                            print("Resume: CATE " + cell, flush=True)
                        else:
                            begin_work(False)
                            individual, subgroup = xlearner(frame, outcome, family, a["seed"], reference_frame=frame,
                                                           missing_indicators=config["features"]["missing_indicators"],
                                                           trimming_bounds=a["trimming"],
                                                           keep_empty_features=config["features"]["keep_empty_features"])
                            individual_file = dest / "cate_private" / (stem + ".parquet")
                            write_table(individual, individual_file)
                            for k, v in context.items():
                                subgroup[k] = v
                            write_table(subgroup, subgroup_file)
                            cates.append(subgroup)
                            info = individual.iloc[0]
                            status = {**context, "status": "ok", **{k: info[k] for k in [
                                "cate_engine", "cate_population", "trimming_source", "trimming_method", "trimming_lower",
                                "trimming_upper", "n_before_trimming", "n_after_trimming", "treated_after_trimming",
                                "control_after_trimming", "sampling_probability_0", "sampling_probability_1"]}}
                            cate_status.append(status)
                            files = [individual_file, subgroup_file]
                            written.extend(files)
                            register_stage(output, "cate:" + cell, files)
                            store.save("cate:" + cell, status, files)
                    except (ValueError, ImportError) as exc:
                        begin_work(False)
                        failed = {**context, "status": "failed", "reason": str(exc),
                                  "cate_engine": "econml.metalearners.XLearner",
                                  "trimming_source": "CATE_in_sample_corrected_propensity"}
                        failure_path = dest / "cate_private" / (stem + ".json")
                        write_json(failed, failure_path)
                        written.append(failure_path)
                        cate_status.append(failed)
                if changed or not previous_stage:
                    write_table(pd.DataFrame(cate_status), output / "cate_status.csv")
    if not changed and previous_stage:
        return pd.read_csv(output / "all_ate_results.csv")
    write_table(pd.DataFrame(results), output / "all_ate_results.csv")
    write_table(pd.concat(cates, ignore_index=True) if cates else pd.DataFrame(), output / "cate_subgroups.csv")
    write_table(pd.DataFrame(cate_status), output / "cate_status.csv")
    manifest.update(status="analysis_complete" if all(r["status"] == "ok" for r in results) else "analysis_has_incomplete_cells",
                    expected_cells=len(list(pairs(config))) * len(a["outcomes"]) * len(a["families"]),
                    completed_cells=sum(r["status"] == "ok" for r in results),
                    cate_completed_cells=sum(r["status"] == "ok" for r in cate_status),
                    cate_expected_cells=len(cate_status),
                    sparse_laboratory_override=bool(allow_sparse), completed_utc=datetime.now(timezone.utc).isoformat())
    _save_run_manifest(output, manifest)
    register_stage(output, "estimation", written + [output / "all_ate_results.csv", output / "cate_subgroups.csv",
                                                  output / "cate_status.csv"])
    return pd.DataFrame(results)
