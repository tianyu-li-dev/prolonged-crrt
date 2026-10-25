"""Public analysis configuration and command interfaces."""
from pathlib import Path

from crrt_revision.config import load_config, pairs
from crrt_revision.cli import main


def test_default_grid_and_primary():
    config = load_config()
    assert len(list(pairs(config))) == 6
    assert config["design"]["primary_pair"] == [500, 2000]
    assert len(config["analysis"]["families"]) == 5
    assert len(config["analysis"]["outcomes"]) == 3
    assert config["analysis"]["bootstrap_samples"] == 1000


def test_all_configurations_load():
    for path in (Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"):
        load_config(path)


def test_plan_does_not_connect_to_sources(monkeypatch, capsys):
    from crrt_revision import data
    monkeypatch.setattr(data, "engine", lambda *args: (_ for _ in ()).throw(AssertionError("Unexpected connection")))
    path = Path(__file__).resolve().parents[1] / "configs/main.yaml"
    main(["plan", "--config", str(path)])
    assert '"ate_cells": 90' in capsys.readouterr().out
