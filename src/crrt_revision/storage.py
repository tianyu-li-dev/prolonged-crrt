"""Atomic artifacts and clinical output boundaries."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import tomllib
import os
import time


def _replace(temporary, path):
    """Replace an artifact after bounded retries for a temporary file lock."""
    for delay in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(delay)
    temporary.replace(path)


def write_json(data, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding="utf-8")
    _replace(temporary, path)


def write_table(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    _replace(temporary, path)


def resolved_path(path):
    """Resolve links, then normalize residual dot segments on Windows."""
    return Path(os.path.normpath(str(Path(path).resolve())))


def private_output(path, repository_root=None):
    resolved = resolved_path(path)
    for ancestor in [resolved, *resolved.parents]:
        marker = ancestor / "pyproject.toml"
        if marker.is_file():
            project = tomllib.loads(marker.read_text(encoding="utf-8")).get("project", {}).get("name")
            if project == "crrt-revision":
                raise ValueError("Store clinical outputs outside the publication repository.")
    if repository_root is not None:
        repository = resolved_path(repository_root)
        if resolved == repository or repository in resolved.parents:
            raise ValueError("Store clinical outputs outside the publication repository.")
    return resolved


def read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def commit_publication(run, staging):
    """Preserve the previous complete report and publish a fresh directory."""
    return commit_output_directory(run, staging, "publication_outputs")


def commit_output_directory(run, staging, target_name):
    """Replace a complete report directory, with recovery after a failed commit."""
    import uuid
    root, staging = resolved_path(run), resolved_path(staging)
    allowed = {"publication_outputs": ("_publication_stage_", "publication_archive")}
    if target_name not in allowed:
        raise ValueError("Unsupported generated output directory.")
    prefix, archive_name = allowed[target_name]
    target = root / target_name
    archive = root / archive_name / uuid.uuid4().hex
    if not all(path.is_relative_to(root) and path != root for path in [staging, target, archive]):
        raise ValueError("Publication paths must remain inside the declared private run.")
    if not staging.name.startswith(prefix):
        raise ValueError("Only a generated staging directory can be committed.")
    if target.exists():
        archive.parent.mkdir(exist_ok=True)
        target.replace(archive)
    try:
        staging.replace(target)
    except OSError:
        if archive.exists() and not target.exists():
            archive.replace(target)
        raise
    return target
