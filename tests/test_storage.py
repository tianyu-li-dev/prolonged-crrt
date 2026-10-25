"""Atomic output publication under temporary and persistent file locks."""
import json
from pathlib import Path

import pandas as pd
import pytest

from crrt_revision import storage


@pytest.mark.parametrize("extension", ["json", "csv", "parquet"])
def test_temporary_lock_retries_before_replacing_complete_artifact(tmp_path, monkeypatch, extension):
    target = tmp_path / ("result." + extension)
    original = Path.replace
    calls, waits = [], []
    def replace(path, destination):
        if Path(destination) == target:
            calls.append(path)
            if len(calls) < 3:
                raise PermissionError("Temporary lock")
        return original(path, destination)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(storage.time, "sleep", waits.append)
    if extension == "json":
        storage.write_json({"value": 7}, target)
        assert json.loads(target.read_text()) == {"value": 7}
    else:
        expected = pd.DataFrame({"value": [7]})
        storage.write_table(expected, target)
        actual = pd.read_csv(target) if extension == "csv" else pd.read_parquet(target)
        pd.testing.assert_frame_equal(actual, expected)
    assert len(calls) == 3 and waits == [0.05, 0.1]
    assert not target.with_name(target.name + ".partial").exists()


def test_persistent_lock_preserves_previous_artifact_and_reports_failure(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    target.write_text('{"value": 1}', encoding="utf-8")
    calls = []
    def locked(path, destination):
        calls.append(path)
        raise PermissionError("Persistent lock")
    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(storage.time, "sleep", lambda delay: None)
    with pytest.raises(PermissionError, match="Persistent lock"):
        storage.write_json({"value": 2}, target)
    assert len(calls) == 8
    assert json.loads(target.read_text()) == {"value": 1}
    assert json.loads(target.with_name(target.name + ".partial").read_text()) == {"value": 2}
