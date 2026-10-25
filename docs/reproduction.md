# Reproduction procedure

Use Python 3.12 and the pinned dependencies in `requirements.lock`. EconML supplies CATE fitting and DoubleML supplies the reference point-sensitivity calculation.

## Local inputs

The supported clinical release is MIMIC-IV 2.1. Raw tables must exist in `mimiciv_hosp` and `mimiciv_icu`, and required derived relations must exist in `mimiciv_derived`.

The package reads existing raw and derived relations only. It never creates schemas, functions, tables, views, indexes, or missing concepts.

Run `inspect-sources --config configs/main.yaml` to check existing sources. Missing sources remain missing and cause an explicit report/error. Use authorized existing relations or local exports. This workflow must not alter the MIMIC-IV database, its users, or its configuration.

The psycopg2 connection starts with default_transaction_read_only=on. SQLAlchemy also sets postgresql_readonly=True. A transaction-state query must confirm on before reading sources. The fixed extraction query guard rejects non-SELECT, stacked, and modifying statements. All source reads share one read-only snapshot. See [SQLAlchemy read-only transactions](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#setting-read-only-deferrable) and [PostgreSQL transaction modes](https://www.postgresql.org/docs/15/sql-set-transaction.html).

The alternative input mode reads one CSV or Parquet file per source. Set `data.source_directory` in the configuration. Relative paths use the current working directory.

| File stem | Required source |
|---|---|
| stays | `icustay_detail` |
| admissions | Admission type and insurance only; patient dod comes from stays |
| crrt_records | Selected CRRT `procedureevents` |
| kdigo_uo | Timed 24-hour urine output |
| diuretics | Loop-diuretic `inputevents` |
| vitalsign | Time-stamped vital signs |
| bg | Derived blood-gas lactate source |
| first_day_sofa | First-day SOFA |
| first_day_sapsii | First-day SAPS II; collection timestamps do not restrict its static join |
| urine_output | Time-stamped urine volumes |
| ventilation | Support intervals |
| vasoactive_agent | Vasoactive intervals |
| suspected_infection | Suspected infection timestamps |
| rrt | RRT status events |
| inputevents | Input amount, units, rate, interval, order, component, and record status |
| outputevents | Output amount and timestamp |
| crrt | CRRT removal measurements |
| prescriptions | Drug name and interval |
| labevents | Selected raw blood tests and specimen labels |
| d_labitems | Item label, fluid, and category |

The extraction queries in `data.py` define exact columns. Empty event sources must retain their documented schema. A missing source file or failed required query causes an explicit error.

Nonempty input events must supply `stay_id`, `starttime`, and volume-unit information. Preserve `amount`, `amountuom`, `rate`, `rateuom`, `endtime`, `storetime`, `orderid`, `itemid`, `ordercategorydescription`, `ordercomponenttypedescription`, and `statusdescription` when available. Unitless historical extracts require re-extraction.

## Execution order

1. Inspect the configuration and declared source schemas.
2. Run `plan` and `inspect-sources`.
3. Run `build` to construct all configured cohorts and features.
4. Inspect missingness and exclusion reports in the private output directory.
5. Run `estimate` for ATE, bootstrap and CATE.
6. Run `sensitivity --benchmarks` for point sensitivity and feature-block comparisons.
7. Run `report` for CSV tables and SVG, PNG and TIFF figures.

Each threshold pair rebuilds time zero, landmark, outcomes and covariates. The output directory stores a local source snapshot for continuation. Use the same configuration for build and estimation. Changed input data or code requires a new output directory. See [resume behavior](resume.md).

## Supplementary configurations

- `no_new_labs.yaml` excludes the biochemical feature block.
- `acid_base.yaml` includes pCO2 and base excess.
- `extended_electrolytes.yaml` includes calcium, magnesium and phosphate.
- `anion_gap.yaml` includes reported anion gap.
- `arterial_only.yaml` restricts blood-gas acid-base sources.
- `strict_availability.yaml` uses laboratory storage timestamps.
- `last_24_hours.yaml` restricts the biochemical laboratory window.
- `missing_indicators.yaml` includes indicators for continuous measurements.

Use a separate output directory for each configuration. Compare results with explicit attention to target-population differences.

## Computation

The bootstrap refits the complete procedure for every replicate. Random forest uses 1000 trees. `analysis.bootstrap_jobs` controls replicate workers and defaults to one; choose a value consistent with available memory. Model-level workers remain limited to avoid nested oversubscription.
