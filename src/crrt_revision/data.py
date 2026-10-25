"""Read-only MIMIC extraction. No database preparation or write operations."""
from __future__ import annotations
import os
from pathlib import Path
import re
from contextlib import contextmanager
import pandas as pd

from .cohort import CRRT_ITEMIDS, DIURETIC_ITEMIDS
from .laboratory import ALL_ITEMIDS
from .storage import write_table, write_json, read_table


SOURCE_TABLES = {
    "stays": "mimiciv_derived.icustay_detail", "admissions": "mimiciv_hosp.admissions",
    "crrt_records": "mimiciv_icu.procedureevents", "kdigo_uo": "mimiciv_derived.kdigo_uo",
    "diuretics": "mimiciv_icu.inputevents", "vitalsign": "mimiciv_derived.vitalsign",
    "bg": "mimiciv_derived.bg", "first_day_sapsii": "mimiciv_derived.sapsii",
    "first_day_sofa": "mimiciv_derived.first_day_sofa", "urine_output": "mimiciv_derived.urine_output",
    "ventilation": "mimiciv_derived.ventilation", "vasoactive_agent": "mimiciv_derived.vasoactive_agent",
    "suspected_infection": "mimiciv_derived.suspicion_of_infection", "rrt": "mimiciv_derived.rrt",
    "inputevents": "mimiciv_icu.inputevents", "outputevents": "mimiciv_icu.outputevents",
    "crrt": "mimiciv_derived.crrt", "prescriptions": "mimiciv_hosp.prescriptions",
    "labevents": "mimiciv_hosp.labevents", "d_labitems": "mimiciv_hosp.d_labitems",
}


def require_read_query(statement):
    """Fail closed on non-SELECT, stacked or modifying extraction statements."""
    sql = str(statement).strip()
    forbidden = r"\b(CREATE|DROP|ALTER|TRUNCATE|INSERT|UPDATE|DELETE|MERGE|COPY|GRANT|REVOKE|INTO|SET_CONFIG|NEXTVAL|SETVAL|LO_IMPORT|LO_UNLINK)\b"
    if (not re.match(r"^SELECT\b", sql, re.I) or any(token in sql for token in [";", "--", "/*"])
            or re.search(forbidden, sql, re.I)):
        raise PermissionError("Read-only extraction permits a single non-modifying SELECT only.")


def _guard_query(conn, cursor, statement, parameters, context, executemany):
    """Reject a prohibited query before SQLAlchemy sends it to the database."""
    require_read_query(statement)


def engine(config):
    from sqlalchemy import create_engine, event
    from sqlalchemy.engine import make_url
    name = config["data"]["postgres_uri_env"]
    uri = os.environ.get(name)
    if not uri:
        raise ValueError(f"Set environment variable {name}; do not put credentials in a configuration file.")
    try:
        url = make_url(uri)
    except Exception:
        raise ValueError("The database URI is invalid. Credentials are not displayed.") from None
    if url.drivername not in {"postgresql", "postgresql+psycopg2"}:
        raise ValueError("Read-only extraction requires PostgreSQL with the psycopg2 driver.")
    db = create_engine(url, connect_args={"options": "-c default_transaction_read_only=on"},
                       isolation_level="REPEATABLE READ", execution_options={"postgresql_readonly": True})
    event.listen(db, "before_cursor_execute", _guard_query)
    return db


@contextmanager
def read_only_connection(config):
    """Verify server-enforced read-only mode before inspecting or reading sources."""
    from sqlalchemy import text
    db = engine(config)
    try:
        with db.connect() as conn:
            conn = conn.execution_options(postgresql_readonly=True)
            with conn.begin():
                mode = conn.execute(text("SELECT current_setting('transaction_read_only')")).scalar()
                if str(mode).lower() != "on":
                    raise PermissionError("The server did not enable a read-only transaction. Extraction stopped.")
                yield conn
    finally:
        db.dispose()


def _source_files(directory):
    root = Path(directory)
    return {name: root / (name + (".parquet" if (root / (name + ".parquet")).is_file() else ".csv"))
            for name in SOURCE_TABLES}


def inspect_sources(config):
    """Check existing relations or local exports without reading patient rows."""
    directory = config["data"].get("source_directory")
    if directory:
        files = _source_files(directory)
        items = {name: path.is_file() for name, path in files.items()}
        mode = "local_files"
    else:
        from sqlalchemy import text
        with read_only_connection(config) as conn:
            items = {name: conn.execute(text("SELECT to_regclass(:name)"), {"name": table}).scalar() is not None
                     for name, table in SOURCE_TABLES.items()}
        mode = "read_only_PostgreSQL"
    return {"mode": mode, "sources": items, "missing_sources": [name for name, present in items.items() if not present],
            "status": "ready" if all(items.values()) else "missing_sources"}


def _in(column, identifiers):
    values = sorted(set(int(x) for x in identifiers if pd.notna(x)))
    return f"{column} IN ({','.join(map(str, values))})" if values else "false"


def load_sources(config, output):
    """Read declared local sources, or extract them without silent omission."""
    directory = config["data"].get("source_directory")
    if directory:
        out = {}
        for name, path in _source_files(directory).items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing declared source: {name}")
            out[name] = read_table(path)
        return out
    ids = ",".join(map(str, CRRT_ITEMIDS)); medids = ",".join(map(str, DIURETIC_ITEMIDS))
    core = {
        "stays": "SELECT * FROM mimiciv_derived.icustay_detail",
        "admissions": "SELECT hadm_id, admission_type, insurance FROM mimiciv_hosp.admissions",
        "crrt_records": "SELECT subject_id, hadm_id, stay_id, itemid, starttime, endtime, value "
                        f"FROM mimiciv_icu.procedureevents WHERE itemid IN ({ids})",
    }
    out = {}
    with read_only_connection(config) as conn:
        for name, query in core.items():
            out[name] = pd.read_sql_query(query, conn)
        stays, admissions = out["crrt_records"].stay_id, out["crrt_records"].hadm_id
        queries = {
            "kdigo_uo": f"SELECT * FROM mimiciv_derived.kdigo_uo WHERE {_in('stay_id', stays)}",
            "diuretics": "SELECT subject_id,hadm_id,stay_id,itemid,starttime,endtime,amount "
                         f"FROM mimiciv_icu.inputevents WHERE itemid IN ({medids}) AND {_in('stay_id', stays)}",
            "bg": f"SELECT * FROM mimiciv_derived.bg WHERE {_in('hadm_id', admissions)}",
            "d_labitems": f"SELECT itemid,label,fluid,category FROM mimiciv_hosp.d_labitems WHERE {_in('itemid', ALL_ITEMIDS)}",
            "labevents": "SELECT subject_id,hadm_id,specimen_id,itemid,charttime,storetime,value,valuenum,valueuom "
                         f"FROM mimiciv_hosp.labevents WHERE {_in('itemid', ALL_ITEMIDS)} AND {_in('hadm_id', admissions)}",
        }
        for name, table in {"vitalsign": "vitalsign", "first_day_sapsii": "sapsii", "first_day_sofa": "first_day_sofa",
                            "urine_output": "urine_output", "ventilation": "ventilation",
                            "vasoactive_agent": "vasoactive_agent", "rrt": "rrt", "crrt": "crrt"}.items():
            queries[name] = f"SELECT * FROM mimiciv_derived.{table} WHERE {_in('stay_id', stays)}"
        queries.update({
            "suspected_infection": f"SELECT * FROM mimiciv_derived.suspicion_of_infection WHERE {_in('hadm_id', admissions)}",
            "inputevents": "SELECT subject_id,hadm_id,stay_id,starttime,endtime,storetime,amount,amountuom,rate,rateuom,"
                           "itemid,orderid,linkorderid,ordercomponenttypedescription,ordercategorydescription,statusdescription "
                           f"FROM mimiciv_icu.inputevents WHERE {_in('stay_id', stays)}",
            "outputevents": f"SELECT subject_id,hadm_id,stay_id,charttime,value,itemid FROM mimiciv_icu.outputevents WHERE {_in('stay_id', stays)}",
            "prescriptions": f"SELECT subject_id,hadm_id,drug,starttime,stoptime FROM mimiciv_hosp.prescriptions WHERE {_in('hadm_id', admissions)}",
        })
        for name, query in queries.items():
            try:
                out[name] = pd.read_sql_query(query, conn)
            except Exception as exc:
                raise RuntimeError(f"Cannot read required source {name} ({type(exc).__name__}). "
                                   "Use existing authorized relations or local exports; this code cannot prepare them.") from None
    for name, frame in out.items():
        write_table(frame, Path(output) / "sources" / (name + ".parquet"))
    write_json({"release": config["data"]["release"], "sources": {k: len(v) for k, v in out.items()}},
               Path(output) / "sources" / "source_manifest.json")
    return out


def admission_fields(cohort: pd.DataFrame, admissions: pd.DataFrame | None) -> pd.DataFrame:
    """Join admission covariates without overwriting patient dod."""

    out = cohort.copy()
    if admissions is None or admissions.empty or "hadm_id" not in admissions:
        return out
    covariate_fields = [col for col in ["admission_type", "insurance"] if col in admissions]
    if not covariate_fields:
        return out
    source = admissions[["hadm_id", *covariate_fields]].drop_duplicates("hadm_id").rename(
        columns={col: f"_admission_{col}" for col in covariate_fields}
    )
    out = out.merge(source, on="hadm_id", how="left")
    for col in covariate_fields:
        source_col = f"_admission_{col}"
        if col in out:
            out[col] = out[col].where(out[col].notna(), out[source_col])
        else:
            out[col] = out[source_col]
        out = out.drop(columns=source_col)
    return out
