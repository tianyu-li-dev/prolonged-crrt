"""Atomic checkpoints for named analysis tasks in a private run directory."""
from __future__ import annotations
from contextlib import contextmanager
from functools import wraps
import inspect
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

from .storage import private_output, write_json


class CheckpointError(RuntimeError):
    """A checkpoint is incomplete, unreadable, or incompatible with its task."""


def _encode(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {"__checkpoint_float__": str(value)}
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_encode(v) for v in value]
    return value


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"__checkpoint_float__"}:
            return float(value["__checkpoint_float__"])
        return {k: _decode(v) for k, v in value.items()}
    return [_decode(v) for v in value] if isinstance(value, list) else value


@contextmanager
def _file_lock(path, wait=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + 60
        while True:
            handle.seek(0)
            try:
                if __import__("os").name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if not wait or time.monotonic() >= deadline:
                    raise CheckpointError("Another process is using this checkpoint or run directory.") from exc
                time.sleep(.05)
        try:
            yield
        finally:
            handle.seek(0)
            if __import__("os").name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def run_lock(output):
    root = private_output(output)
    with _file_lock(root / ".execution.lock"):
        yield root


def exclusive_run(parameter):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = inspect.signature(function).bind(*args, **kwargs)
            bound.apply_defaults()
            with run_lock(bound.arguments[parameter]):
                return function(*args, **kwargs)
        return wrapped
    return decorate


class Checkpoints:
    """Map task names to short directories; commit completion metadata last."""

    def __init__(self, root, contract, resume=True, owner=None):
        logical = private_output(root)
        anchors = [p for p in [logical, *logical.parents] if p.name == "_checkpoints"]
        self.owner = private_output(owner) if owner is not None else anchors[0].parent if anchors else logical
        self.root = self.owner / "_checkpoints"
        if not logical.is_relative_to(self.owner):
            raise CheckpointError("Checkpoint scope escapes the private run.")
        self.contract = {"scope": logical.relative_to(self.owner).as_posix(), "settings": _encode(contract)}
        pointer = logical / "namespaces.json"
        with _file_lock(logical / ".namespace.lock", wait=True):
            entries = json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else []
            match = next((item for item in entries if item["contract"] == self.contract), None)
            if match is not None and not resume:
                self._archive(match["directory"])
                entries.remove(match)
                match = None
            if match is None:
                with _file_lock(self.root / ".allocation.lock", wait=True):
                    counter = self.root / "allocation.json"
                    number = json.loads(counter.read_text(encoding="utf-8"))["next"] if counter.exists() else 1
                    write_json({"next": number + 1}, counter)
                    directory = f"task_{number:06d}"
                    path = self.root / directory
                    path.mkdir(exist_ok=False)
                    write_json(self.contract, path / "contract.json")
                match = {"contract": self.contract, "directory": directory}
                entries.append(match)
                write_json(entries, pointer)
            self.path = self._within(self.root / match["directory"], self.root)
            saved = json.loads((self.path / "contract.json").read_text(encoding="utf-8"))
            if saved != self.contract:
                raise CheckpointError("Checkpoint settings do not match the named task.")
        self._units = {}

    @staticmethod
    def _within(path, root):
        path = Path(path).resolve()
        root = Path(root).resolve()
        if path == root or not path.is_relative_to(root):
            raise CheckpointError("Checkpoint path escapes its private directory.")
        return path

    def _archive(self, directory):
        source = self._within(self.root / directory, self.root)
        names = {source.name}
        scopes = {}
        for file in self.root.glob("task_*/contract.json"):
            item = json.loads(file.read_text(encoding="utf-8"))
            parts = Path(item["scope"]).parts
            if len(parts) > 1 and parts[0] == "_checkpoints":
                scopes[file.parent.name] = parts[1]
        while True:
            expanded = names | {name for name, parent in scopes.items() if parent in names}
            if expanded == names:
                break
            names = expanded
        destination = self._within(self.owner / "_checkpoint_archive" / uuid.uuid4().hex, self.owner)
        destination.mkdir(parents=True)
        for name in sorted(names):
            item = self._within(self.root / name, self.root)
            if item.exists():
                item.replace(destination / name)

    def prepare(self, keys):
        """Allocate independent task directories before starting parallel workers."""
        keys = list(keys)
        with _file_lock(self.path / ".units.lock", wait=True):
            path = self.path / "units.json"
            units = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            changed = False
            for key in keys:
                if not isinstance(key, str):
                    raise TypeError("Checkpoint task names must be strings.")
                if key not in units:
                    units[key] = f"unit_{len(units) + 1:06d}"
                    changed = True
            if changed:
                write_json(units, path)
            self._units.update(units)

    def directory(self, key):
        if key not in self._units:
            self.prepare([key])
        return self._within(self.path / self._units[key], self.path)

    def load(self, key):
        marker = self.directory(key) / "completed.json"
        if not marker.exists():
            return None
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record["key"] != key or not isinstance(record["files"], list):
                raise CheckpointError("Checkpoint does not describe the requested task.")
            for name in record["files"]:
                file = self._within(self.owner / name, self.owner)
                if not file.is_file():
                    raise CheckpointError("Checkpoint output is missing: " + name)
            return {**record, "payload": _decode(record["payload"])}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CheckpointError("Cannot read completed checkpoint: " + key) from exc

    def save(self, key, payload, files=()):
        names = []
        for file in files:
            file = self._within(file, self.owner)
            if not file.is_file():
                raise CheckpointError("Cannot complete a task with a missing output.")
            names.append(file.relative_to(self.owner).as_posix())
        record = {"key": key, "payload": _encode(payload), "files": sorted(set(names))}
        write_json(record, self.directory(key) / "completed.json")
        return record


def stage_files(run, stage):
    from .integrity import registry, verify_artifact
    return [verify_artifact(run, Path(run) / name) for name in registry(run)["stages"].get(stage, [])]


def retire_stage_markers(run, label):
    root = private_output(run) / "_checkpoints"
    # Stage namespaces are direct children; fitted models have nested scopes.
    for pointer in root.glob("*/namespaces.json"):
        for entry in json.loads(pointer.read_text(encoding="utf-8")):
            task = Checkpoints._within(root / entry["directory"], root)
            index = task / "units.json"
            if not index.exists():
                continue
            units = json.loads(index.read_text(encoding="utf-8"))
            if label not in units:
                continue
            marker = Checkpoints._within(task / units[label], task) / "completed.json"
            if marker.exists():
                record = json.loads(marker.read_text(encoding="utf-8"))
                if record["key"] != label:
                    raise CheckpointError("Stage checkpoint does not describe the requested task.")
                marker.replace(marker.with_name("retired_" + uuid.uuid4().hex + ".json"))


def progress(output):
    """Read task counts without loading patient records or running a model."""
    root = private_output(output)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    counts = {}
    for marker in (root / "_checkpoints").glob("task_*/unit_*/completed.json"):
        item = json.loads(marker.read_text(encoding="utf-8"))
        label = item["key"].split(":", 1)[0]
        counts[label] = counts.get(label, 0) + 1
    return {"run": str(root), "status": manifest["status"], "committed_checkpoints": counts,
            "note": "Counts include saved task settings and failed attempts. Successful completed tasks are reused; unfinished fits restart."}
