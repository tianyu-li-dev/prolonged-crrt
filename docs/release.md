# Public package inspection

Run the public tests and the content inspection command with all temporary files outside the source directory. Build and install the package from a separate staging copy to avoid leaving build artifacts in the public tree.

The public tree contains source code, configuration and documentation. Do not include source exports, patient frames, model predictions, experiment results, checkpoints, logs, credentials, local execution paths or generated figures. Review both the directory and the final package contents.

```text
python -m pytest -p no:cacheprovider --basetemp ../crrt_tests
python -m crrt_revision check-release --root .
```

`check-release` checks required files, excluded artifact types and credential patterns. It supplements manual inspection. `.gitignore` prevents accidental tracking of common private artifacts.

Access to MIMIC patient data is governed separately.
