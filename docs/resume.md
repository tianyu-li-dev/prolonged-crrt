# Pause and resume

Resume is enabled by default for build, estimation, sensitivity and reporting. Press Ctrl+C and wait for the process to exit. Repeat the same command with the same configuration and output directory to continue. The currently unfinished fit restarts.

Completed source extraction, threshold builds, OOF folds, bootstrap replicates, ATE cells, CATE cells, sensitivity blocks and reports have named completion records. The records store task settings and output paths. Completion is committed after outputs are written. Short numbered directories keep paths within Windows limits.

The run configuration must match exactly. Stored OOF fold indices, prediction lengths and finite values are checked before reuse. Listed output files must exist and be readable by their consumers. The package does not detect arbitrary edits to input values or completed files. Keep a run's saved sources and outputs unchanged; use a new output directory for changed data or code.

Bootstrap sample indices and fit seeds are deterministic. Successful completed draws are reused and failed draws are retried. Parallel workers write separate units; index allocation is protected by file locks. A run lock prevents two commands from writing the same run concurrently.

Use `--no-resume` to recompute a stage. Its task checkpoints and descendant caches are archived. Rebuilding the design invalidates estimation and sensitivity registrations. Re-estimation invalidates sensitivity; downstream calculations use explicit stage-update counters. Refresh sensitivity before reporting after a main re-estimation. Reports are written into a staging directory and replace the complete publication directory only after successful rendering.

A run-manifest recovery receipt handles interruption between writing the manifest and updating the artifact inventory. Files without completion records are not reused. Unsupported checkpoint layouts require a new output directory.

```text
python -m crrt_revision run --config configs/main.yaml --output ../crrt_output
python -m crrt_revision status --run ../crrt_output
python -m crrt_revision estimate --config configs/main.yaml --output ../crrt_output --no-resume
```
