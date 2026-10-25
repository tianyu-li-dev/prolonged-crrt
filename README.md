# Landmark causal analysis of prolonged CRRT

This repository contains code for estimating the causal effect of CRRT continuation at a 24-hour landmark on subsequent mortality. It includes cohort construction, clinical features, cross-fitted AIPW estimation, bootstrap intervals, CATE and point sensitivity analysis.

## Analysis

- Urine-output thresholds without diuretics: 400, 500 and 600 mL/day.
- Thresholds with diuretic exposure: 1000 and 2000 mL/day.
- Primary pair: 500/2000 mL/day.
- Mortality horizons: 28, 60 and 90 days from the landmark.
- Base learners for propensity-score and binary-outcome models: L2-regularized logistic regression, L1-regularized logistic regression, elastic-net logistic regression, XGBoost classifiers, and random-forest classifiers.
- Main inference: 1000 repetitions of the full estimation procedure, with percentile confidence intervals.

Each threshold pair defines its own eligibility time, landmark, cohort and covariates. Main ATE preprocessing is fitted within training folds. CATE uses an independent propensity-screened population and EconML XLearner. For X-learner effect regression, the corresponding learners are ridge, lasso, elastic net, XGBoost regression, and random-forest regression. [Methods](docs/methods.md) describe the estimands and assumptions.

## Installation

Use Python 3.12 and the dependencies in `requirements.lock`.

```text
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

For PostgreSQL access, install the `postgres` extra. Keep credentials in the `MIMIC_POSTGRES_URI` environment variable. Authorized MIMIC-IV 2.1 sources must be available locally. [Input requirements](docs/reproduction.md) describe database and exported-file inputs.

```text
python -m pip install -e .[postgres,test]
python -m crrt_revision plan --config configs/main.yaml
python -m crrt_revision inspect-sources --config configs/main.yaml
```

`plan` reads configuration only. `inspect-sources` checks existing sources. Database access is read-only.

## Execution

Choose an output directory outside the repository.

```text
python -m crrt_revision build --config configs/main.yaml --output ../crrt_output
python -m crrt_revision estimate --config configs/main.yaml --output ../crrt_output
python -m crrt_revision sensitivity --run ../crrt_output --benchmarks
python -m crrt_revision report --run ../crrt_output
```

`run` combines `build` and `estimate`. Sensitivity analysis and report generation are separate commands. Reports contain CSV tables and SVG, PNG and TIFF figures.

Press Ctrl+C and wait for the process to exit before resuming with the same command, configuration and output directory. Successful completed tasks are reused. `--no-resume` recomputes the selected stage. Use a new output directory after changing code or input data. See [checkpoint behavior](docs/resume.md).

```text
python -m crrt_revision status --run ../crrt_output
```

## Public contents

The repository contains source code, configuration, method documentation and tests. Clinical records, predictions, results, checkpoints and validation logs belong outside it. Clinical inputs require separate access authorization.

```text
python -m pytest -p no:cacheprovider --basetemp ../crrt_tests
python -m crrt_revision check-release --root .
```

## Documentation

- [Analysis methods](docs/methods.md)
- [Clinical and laboratory covariates](docs/covariates.md)
- [CATE population and estimation](docs/cate.md)
- [Threshold definitions](docs/thresholds.md)
- [Sensitivity calculations](docs/sensitivity.md)
- [Reproduction procedure](docs/reproduction.md)
- [Pause and resume](docs/resume.md)
- [Public package inspection](docs/release.md)

The estimand concerns eligible landmark survivors under the stated causal assumptions. These analyses do not provide clinical stopping guidance.
