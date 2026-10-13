# Sydney TMD V2

**Status: planning.** This directory currently contains documentation only.
The V2 experiment script, Docker image, and Compose services have not been
implemented, so there are no runnable V2 setup or experiment commands yet.
The full intended behavior is in [PLAN.md](PLAN.md).

Sydney TMD V2 will train and compare transport-mode classifiers using only
Sydney sessions with an explicit Australian country code. It builds on the
sensor processing and study structure of `../sydney-tmd-v1` while adding a
JSON-driven experiment, Docker execution, random forest alongside XGBoost,
and live MLflow tracking.

## Intended inputs

- A locally downloaded directory of Sydney session JSON or `.json.gz` files
  and any metadata sidecars. Downloading and synchronization happen separately.
- One experiment JSON file. Defaults reproduce the V1 study windows
  (100/120/140 seconds), seeds (42/43/44), participant leave-one-out
  evaluation, XGBoost, macro F1 selection, 100 trials, three inner folds,
  5,000 bootstrap draws, and a minimum of 10 valid windows per participant
  at each window size.
- Optional JSON settings select registered features, transport labels,
  training-session fractions, window lengths, seeds, the evaluation strategy,
  the selection metric, and `xgboost` and/or `random_forest`. The manual
  participant and session train/test strategies are specified in the plan.

## Planned workflow and outputs

The future Compose project will start a persistent MLflow server and run a
one-shot experiment service with the input directory mounted read-only. The
service will validate the JSON, copy eligible sessions and sidecars into a
hashed per-run snapshot, extract quality-valid windows, tune models, evaluate
held-out data, and stream parent and child run progress to MLflow.

Each run will retain the resolved configuration, snapshot manifest, split
decisions, metric CSVs, predictions, trial histories, a machine-readable
summary, and a macro-F1-versus-window line chart. Enabled bootstrap reporting
will add 95% confidence intervals to each evaluated fixed-window result.
The MLflow parent name will contain a UTC start timestamp and six random
lowercase hex characters. Exact Docker and experiment commands will be added
when their interfaces exist.
