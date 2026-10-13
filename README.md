# Sydney TMD V2

Sydney-only transport-mode experiments driven by one JSON file. The pipeline
prepares a frozen input snapshot, compares configurable window lengths using
nested validation, and saves held-out metrics and charts to local run storage
and MLflow. It implements [PLAN.md](PLAN.md), building on the sensor processing
of `../sydney-tmd-v1`.

The CLI, Docker image, and Compose services are implemented. The synthetic
end-to-end test passes. A full production experiment and container launch have
not yet been run in this repository.

## Docker workflow

Docker Compose mounts the downloaded Sydney sessions read-only. Download or
sync them separately, then set `SYDNEY_TMD_INPUT_DIR` to their directory.
From this project directory in PowerShell:

```powershell
$env:SYDNEY_TMD_INPUT_DIR = 'D:\path\to\sydney-sessions'
docker compose up -d --build mlflow
docker compose run --rm study
```

Open <http://127.0.0.1:5000> for the MLflow UI. The default experiment name
is `Sydney TMD V2`. Set `SYDNEY_TMD_MLFLOW_PORT` if port 5000 is occupied.
The local `runs/` and `mlflow-data/` directories retain outputs across
container restarts. `runs/logs/` contains flushed stage logs. To use other
paths, set `SYDNEY_TMD_RUNS_DIR`, `SYDNEY_TMD_MLFLOW_DIR`, and
`SYDNEY_TMD_CONFIG_FILE` before invoking Compose. The config file defaults to
the tracked [`experiment.json`](experiment.json), which contains `{}` and
therefore selects the baseline defaults.

Two complete example configs are available:

| File | Study |
|---|---|
| [`examples/participant-lopo-xgboost.json`](examples/participant-lopo-xgboost.json) | Explicit baseline: participant leave-one-out, XGBoost, macro F1. |
| [`examples/session-split-model-comparison.json`](examples/session-split-model-comparison.json) | Session train/test split, both model families, balanced-accuracy selection, and 80% of bus training sessions. |

To run one with Compose, set its absolute path before starting the study:

```powershell
$env:SYDNEY_TMD_CONFIG_FILE = (Resolve-Path .\examples\session-split-model-comparison.json).Path
docker compose run --rm study
```

Both examples use 100 completed trials per tuning study and require enough
Sydney sessions for their class and inner-fold coverage checks.

The study service generates a UTC run name such as
`20261013T043015Z-a1b2c3d4`. To resume an interrupted run, pass its ID with
the same inputs and settings:

```powershell
docker compose run --rm study all --input /input --config /config/experiment.json --runs-dir /runs --logs-dir /runs/logs --tracking-uri http://mlflow:5000 --run-id YOUR_RUN_ID
```

You can also invoke `prepare`, `evaluate`, or `report` separately through the
same entry point. Later stages read the frozen configuration and snapshot from
the run directory. A changed configuration, code version, dependency version,
or original input inventory requires a new run ID when preparing; later stages
can run from the snapshot after the original input directory is removed.

## Experiment settings

The baseline uses 100/120/140-second windows, seeds 42/43/44, participant
leave-one-out evaluation, XGBoost, macro F1 selection, 100 completed Optuna
trials per study, three inner folds, 5,000 bootstrap draws, and at least 10
valid windows per participant at **each** size. The full JSON schema and
search spaces are in [PLAN.md](PLAN.md).

For a smaller custom experiment, copy `experiment.json` and set fields such
as these:

```json
{
  "windows_seconds": [100, 120],
  "seeds": [42],
  "model_families": ["xgboost", "random_forest"],
  "selection_metric": "balanced_accuracy",
  "optuna_trials": 20,
  "bootstrap_iterations": null,
  "evaluation_strategy": "session_train_test_split",
  "session_train_test_split_ratio": 0.8
}
```

The other supported strategies are `participant_lopo` and
`manual_participant_group`. The latter requires a
`manual_participant_group` object with disjoint `train` and `test` participant
ID lists. Splits and transport training fractions keep whole sessions
together. Custom `labels` map selected Sydney `vehicle_type` values to
contiguous integer codes starting at zero. `feature_names` selects registered
sensor features; new feature computations require code.

## Local development and outputs

Python 3.12+ is required. Install the package and run tests from this
directory:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e . pytest
.venv/Scripts/python -m pytest -q
```

On Linux or macOS, use `.venv/bin/python`. To run without Docker, start a
reachable MLflow server, then use
`python -m sydney_tmd_v2 all --input <sessions> --config <json> --tracking-uri <server-url>`.
The `--no-mlflow` option is for local tests only.

Each `runs/<run-id>/` contains the resolved experiment, source fingerprints,
eligible raw sessions and sidecars under `snapshot/sha256/`, exclusion and
cohort manifests, frozen session splits, feature tables, Optuna study
histories, held-out predictions, and `reports/`. The report includes metric,
per-class, participant, confusion-matrix, and prediction CSVs; a
machine-readable summary; and `macro-f1.png`. For each fixed window and the
selected-window result, `reports/confusion-<result>-normalized.png` shows
row-normalized percentages and `reports/confusion-<result>-counts.png` shows
raw window counts. Each chart has one panel per seed, with the configured
class names on both axes. The charts are also available in the MLflow parent
run under `reports/` and in the matching result child run under `charts/`.
When bootstrap is enabled, the report also includes per-window sample CSVs
and 95% intervals on the macro-F1 chart. MLflow
shows a parent experiment run and child tuning/result runs with metrics and
artifacts. Optuna trials remain histories within study runs.
