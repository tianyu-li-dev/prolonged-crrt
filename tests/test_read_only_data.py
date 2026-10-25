"""No database is contacted. Verify write rejection and source-only extraction."""
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from crrt_revision import data
from crrt_revision.config import load_config
from crrt_revision.cli import main


@pytest.mark.parametrize("statement", [
    "CREATE SCHEMA mimiciv_derived", "CREATE TEMP TABLE scratch(x int)", "DROP TABLE mimiciv_derived.crrt",
    "UPDATE mimiciv_hosp.admissions SET insurance='Other'", "DELETE FROM mimiciv_hosp.admissions",
    "INSERT INTO scratch VALUES (1)", "ALTER TABLE scratch ADD COLUMN x int", "COPY scratch FROM STDIN",
    "SET TRANSACTION READ WRITE", "SELECT * INTO scratch FROM mimiciv_hosp.admissions",
    "SELECT set_config('transaction_read_only', 'off', false)", "SELECT nextval('sequence')",
    "SELECT 1; CREATE TABLE scratch(x int)", "WITH changed AS (DELETE FROM scratch RETURNING *) SELECT * FROM changed",
])
def test_modifying_statements_are_rejected_before_driver_execution(statement):
    with pytest.raises(PermissionError, match="single non-modifying SELECT"):
        data._guard_query(None, None, statement, None, None, False)


def test_unsupported_database_commands_are_rejected(capsys):
    for args in [["prepare-derived", "--config", "configs/main.yaml", "--execute"],
                 ["inspect-sources", "--config", "configs/main.yaml", "--execute"]]:
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 2
    assert "--execute" in capsys.readouterr().err


def test_engine_requests_driver_and_transaction_read_only(monkeypatch):
    sa = pytest.importorskip("sqlalchemy")
    monkeypatch.setenv("MIMIC_POSTGRES_URI", "postgresql://reader@localhost/fixture")
    captured = {}
    def factory(url, **kwargs):
        captured.update(kwargs)
        return object()
    monkeypatch.setattr(sa, "create_engine", factory)
    monkeypatch.setattr(sa.event, "listen", lambda db, name, handler: captured.update(event=name, handler=handler))
    data.engine(load_config())
    assert captured["connect_args"]["options"] == "-c default_transaction_read_only=on"
    assert captured["execution_options"]["postgresql_readonly"] is True
    assert captured["isolation_level"] == "REPEATABLE READ"
    assert captured["event"] == "before_cursor_execute"
    with pytest.raises(PermissionError):
        captured["handler"](None, None, "CREATE TABLE scratch(x int)", None, None, False)


@pytest.mark.parametrize("mode", ["on", "off"])
def test_read_only_status_is_checked_and_engine_always_closed(monkeypatch, mode):
    pytest.importorskip("sqlalchemy")
    trace = []
    class Connection:
        def execution_options(self, **kwargs):
            trace.append(kwargs)
            return self
        @contextmanager
        def begin(self):
            yield
        def execute(self, sql):
            trace.append(str(sql))
            class Result:
                def scalar(self):
                    return mode
            return Result()
    class Engine:
        @contextmanager
        def connect(self):
            yield Connection()
        def dispose(self):
            trace.append("disposed")
    monkeypatch.setattr(data, "engine", lambda config: Engine())
    if mode == "off":
        with pytest.raises(PermissionError, match="server did not enable"):
            with data.read_only_connection(load_config()):
                pytest.fail("An unprotected connection was yielded.")
    else:
        with data.read_only_connection(load_config()):
            trace.append("read allowed")
    assert trace[0] == {"postgresql_readonly": True}
    assert trace[1] == "SELECT current_setting('transaction_read_only')"
    assert trace[-1] == "disposed"


def test_missing_file_inspection_does_not_create_sources_or_contact_db(monkeypatch, tmp_path):
    config = load_config(); config["data"]["source_directory"] = str(tmp_path)
    monkeypatch.setattr(data, "engine", lambda config: pytest.fail("Local file mode must not connect."))
    result = data.inspect_sources(config)
    assert result["status"] == "missing_sources" and len(result["missing_sources"]) == 20
    assert not list(tmp_path.iterdir())
