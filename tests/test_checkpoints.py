"""Task continuation, concurrent allocation and output-directory boundaries."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

import pytest

from crrt_revision.checkpoints import Checkpoints, CheckpointError, run_lock, retire_stage_markers
from crrt_revision.integrity import reset_registry, register_stage, invalidate_stage, registry, stage_state
from crrt_revision.storage import private_output


def test_round_trip_and_distinct_task_names(tmp_path):
    store = Checkpoints(tmp_path, {"seed": 42})
    store.save("fit:a/b", {"value": float("nan"), "count": 2})
    other = Checkpoints(tmp_path, {"seed": 42})
    assert math.isnan(other.load("fit:a/b")["payload"]["value"])
    assert other.load("fit:a:b") is None
    assert Checkpoints(tmp_path, {"seed": 43}).load("fit:a/b") is None


def test_missing_and_uncommitted_outputs(tmp_path):
    store = Checkpoints(tmp_path, {})
    file = tmp_path / "output.txt"
    file.write_text("complete")
    store.save("fit", {"status": "ok"}, [file])
    assert store.load("fit")["files"] == ["output.txt"]
    file.unlink()
    with pytest.raises(CheckpointError, match="missing"):
        store.load("fit")
    partial = store.directory("unfinished") / "completed.json.partial"
    partial.parent.mkdir()
    partial.write_text("incomplete")
    assert store.load("unfinished") is None


def test_malformed_completion_is_rejected(tmp_path):
    store = Checkpoints(tmp_path, {})
    store.save("fit", {})
    (store.directory("fit") / "completed.json").write_text("{")
    with pytest.raises(CheckpointError, match="Cannot read"):
        store.load("fit")


def test_files_cannot_escape_owner(tmp_path):
    root = tmp_path / "run"
    store = Checkpoints(root, {})
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    with pytest.raises(CheckpointError, match="escapes"):
        store.save("fit", {}, [outside])
    store.save("fit", {})
    marker = store.directory("fit") / "completed.json"
    marker.write_text(json.dumps({"key": "fit", "payload": {}, "files": ["../outside.txt"]}))
    with pytest.raises(CheckpointError, match="escapes"):
        store.load("fit")


def test_parallel_task_allocation(tmp_path):
    def commit(number):
        store = Checkpoints(tmp_path, {"task": "parallel"})
        store.save(f"unit:{number}", {"number": number})
        return store.path, store.directory(f"unit:{number}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(commit, range(24)))
    assert len({p[0] for p in paths}) == 1
    assert len({p[1] for p in paths}) == 24
    store = Checkpoints(tmp_path, {"task": "parallel"})
    assert [store.load(f"unit:{i}")["payload"]["number"] for i in range(24)] == list(range(24))


def test_forced_recompute_retires_descendant_tasks(tmp_path):
    store = Checkpoints(tmp_path, {"task": "estimate"})
    store.save("point", {})
    child = Checkpoints(store.directory("point") / "folds", {"folds": 5})
    child.save("fold:0", {"done": True})
    old_parent, old_child = store.path, child.path
    replacement = Checkpoints(tmp_path, {"task": "estimate"}, resume=False)
    assert replacement.load("point") is None
    assert not old_parent.exists() and not old_child.exists()
    assert list((tmp_path / "_checkpoint_archive").rglob("completed.json"))


def test_run_lock_rejects_second_writer_and_releases(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(CheckpointError, match="Another process"):
            with run_lock(tmp_path):
                pass
    with run_lock(tmp_path):
        pass


def test_upstream_invalidation_and_noop_registration(tmp_path):
    reset_registry(tmp_path)
    file = tmp_path / "output.txt"
    file.write_text("complete")
    for name in ("design", "estimation", "cell:one", "sensitivity"):
        register_stage(tmp_path, name, [file])
    state = stage_state(tmp_path)
    register_stage(tmp_path, "sensitivity", [file])
    assert stage_state(tmp_path) == state
    invalidate_stage(tmp_path, "design")
    assert not registry(tmp_path)["artifacts"]
    assert all(stage_state(tmp_path)[k] > state[k] for k in state)


def test_repository_cannot_be_used_as_output():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="outside"):
        private_output(root / "results")


def test_stage_retirement_preserves_fitted_tasks(tmp_path, monkeypatch):
    label = "report_stage:complete"
    stages = [Checkpoints(tmp_path / "_checkpoints" / "report", {"inputs": i}, owner=tmp_path)
              for i in range(2)]
    for stage in stages:
        stage.save(label, {})
    nested = Checkpoints(stages[0].directory("fit") / "folds", {"folds": 5})
    nested.save("fold:0", {"done": True})
    fitted_marker = nested.directory("fold:0") / "completed.json"
    content = fitted_marker.read_bytes()
    original = Path.read_text
    def read_text(path, *args, **kwargs):
        if path == fitted_marker:
            raise AssertionError("Stage retirement must not read fitted-task records.")
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", read_text)
        retire_stage_markers(tmp_path, label)
    assert all(stage.load(label) is None for stage in stages)
    assert fitted_marker.read_bytes() == content
    assert all(list(stage.directory(label).glob("retired_*.json")) for stage in stages)
