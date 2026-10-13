# Sydney TMD V2 implementation plan

## Purpose and status

Build a Sydney-only transport-mode study from the structure and sensor semantics
of `../sydney-tmd-v1`. V2 will run JSON-defined experiments in Docker Compose,
compare configurable windows and model families, and show live progress and
results in a Docker-hosted MLflow server. The initial pipeline and Compose
services are implemented; a production run and container launch are still
pending. See [README.md](README.md) for operator commands and current status.

Each execution receives an immutable run identity. The MLflow parent run name
uses the UTC execution start time and eight randomly generated lowercase hex
characters, for example `20261013T043015Z-a1b2c3d4`. Child runs represent tuning
studies and evaluated results; Optuna trials are recorded within study runs,
not as individual MLflow runs. Log progress while the experiment runs. Save
metric CSVs, predictions, trial histories, a machine-readable study summary,
and a macro-F1-versus-window line chart locally and as MLflow artifacts.

## Experiment JSON

One JSON file defines one experiment. The intended baseline uses the V1 study
windows, seeds, features, and XGBoost model. Random forest is an additional V2
option. The defaults below apply when a field is omitted; strategy-specific
fields are required only when that strategy uses them.

| Field | Default | Meaning |
|---|---|---|
| `feature_names` | All V1 registered features | Nonempty list of registered feature extractor names, such as `accelerometer#mean` and `gyroscope#standard_deviation`. Unknown names fail validation; adding an extractor requires code. |
| `windows_seconds` | `[100, 120, 140]` | Nonempty list of distinct positive window lengths. The longest configured window is the minimum session duration before quality checks. |
| `seeds` | `[42, 43, 44]` | Nonempty list of distinct integer seeds for repeatable splits, sampling, and model training. |
| `individual_transport_training_ratios` | `1.0` for every configured class | Per-class fractions in `(0, 1]` of whole **training sessions** to retain for each seed. Validation and test sessions are never sampled. |
| `labels` | `{"bus": 0, "car": 1, "train": 2}` | Mapping from selected Sydney `vehicle_type` values to distinct, contiguous integer codes starting at zero. Unlisted types are excluded. |
| `evaluation_strategy` | `"participant_lopo"` | One of `participant_lopo`, `manual_participant_group`, or `session_train_test_split`. |
| `selection_metric` | `"macro_f1"` | `macro_f1` or `balanced_accuracy`; used for both Optuna trials and window selection. |
| `bootstrap_iterations` | `5000` | Positive draw count for 95% percentile intervals; `null` disables intervals. |
| `optuna_trials` | `100` | Positive number of completed trials per tuning study. |
| `model_families` | `["xgboost"]` | Nonempty selection from `xgboost` and `random_forest`. Trials choose among the requested families. |
| `cv_fold_count` | `3` | Number of inner folds for every strategy; integer at least 2 and no greater than the available training groups. |
| `minimum_window_count_per_participant` | `10` | Exclude participants below this quality-valid window count at **any** configured size. |
| `manual_participant_group` | Omitted | Required only for the matching strategy: `{"train": ["participant_001"], "test": ["participant_004"]}` with nonempty, disjoint participant ID lists. |
| `session_train_test_split_ratio` | Omitted | Required only for `session_train_test_split`: fraction in `(0, 1)` targeting training windows per participant; for example, `0.8`. |

For the baseline experiment, the JSON can contain `{}` and use these defaults.
An explicit manual split, for example, sets `evaluation_strategy` to
`manual_participant_group` and supplies `manual_participant_group.train` and
`manual_participant_group.test`. Validate unknown fields, duplicate values,
unknown participant IDs, missing classes in a training partition, and infeasible
inner folds before expensive tuning. Record the resolved config in the run.

### Evaluation strategies and data separation

Extract and label complete windows first, preserving participant, session, and
transport metadata. Keep every window from a session together in all training,
validation, and test assignments. This matters because windows overlap.

- `participant_lopo`: Hold out each eligible participant in turn. Tune using
  participant-disjoint inner folds on the other participants, refit on the
  outer training participants, and evaluate the held-out participant. Report
  each held-out result and the aggregate across participants and seeds.
- `manual_participant_group`: Use exactly the listed train and test
  participants; unlisted participants do not enter this evaluation. Tune with
  participant-disjoint inner folds in the train group, then evaluate the fixed
  test group once per seed and window.
- `session_train_test_split`: Make a seeded, session-disjoint train/test assignment
  that approximates `session_train_test_split_ratio` by **window count for each
  participant**, considering every configured window size. Whole sessions
  make an exact target such as four of five windows impossible in some cases.
  A participant with too few sessions may appear on only one side. Use
  session-disjoint inner folds in the training pool. Require nonempty train
  and test sets and all configured classes in every training partition;
  report a clear diagnostic if a viable assignment or fold is impossible.

Freeze split identities and sampled training-session IDs per seed so that
window-size comparisons use the same underlying sessions. For a requested
transport ratio, select whole sessions from the current training partition
using the given seed, retain at least one session per configured class, and
keep all windows of selected sessions. Fit imputation and class-balanced sample
weights only on the selected training partition. Never use held-out data for
selection, imputation, or training weights.

## Data snapshot and features

Read locally downloaded Sydney session JSON or compressed JSON files; fetching
or syncing data is a separate prerequisite. Require an explicit
`collection_country_code == "AU"`. Exclude records with missing or non-AU
country labels, including legacy records admitted by older studies. Apply the
configured `labels` mapping to `vehicle_type` and exclude unlisted types.

At preparation time, copy every eligible source session and its metadata
sidecars into a per-run, content-addressed input snapshot. Save file hashes,
original paths, session and participant identities, exclusion reasons, class
counts, and participant/class coverage in a manifest. Record the resolved JSON,
code/dependency versions, and split and sampling decisions so the experiment
can be reproduced even if the original download is removed. Derive participant
counts from this run's cohort; do not assume V1's seven participants.

Retain a session only if it produces at least one quality-valid window at
**every configured window size**. Window counts can differ by size, but the
session cohort remains common. Then apply
`minimum_window_count_per_participant` at every size and record participant
exclusions. Use complete, half-open windows with 50% overlap. For the baseline
windows, the steps are 50, 60, and 70 seconds respectively.

Preserve the V1 sensor-magnitude calculations, feature ordering and quality
checks. Minimum sampling rates are 30 Hz for accelerometer, 4 Hz for gyroscope,
and 2 Hz for magnetometer; the maximum permitted sample gap is 500 ms. The
baseline registered features are the V1 magnitude statistics: mean, population
standard deviation, range, and minimum for accelerometer and gyroscope; and
standard deviation, range, and minimum for magnetometer. A new feature name
requires a registered implementation, not an expression in JSON.

## Tuning and model selection

Run sequential Optuna trials on CPU with configurable model threads. Disable
pruning and early stopping. A trial chooses a family from `model_families` and
its family-specific parameters. Use the V1 XGBoost histogram-tree multiclass
classifier and median-imputation pipeline with missing-value indicators.
Random forest uses the same imputation and indicators. Compute class-balanced
sample weights only from the current training partition. Apply no duration
balancing.

Retain the V1 nine-parameter XGBoost search space:

| Parameter | XGBoost name | Search range |
|---|---|---|
| Trees | `n_estimators` | Integers 200–1,200, step 100 |
| Learning rate | `learning_rate` | 0.02–0.2, logarithmic |
| Maximum depth | `max_depth` | Integers 3–8 |
| Minimum child weight | `min_child_weight` | 1–10 |
| Row subsampling | `subsample` | 0.6–1.0 |
| Column subsampling | `colsample_bytree` | 0.5–1.0 |
| Gamma | `gamma` | 0–5 |
| L1 regularization | `reg_alpha` | 0–2 |
| L2 regularization | `reg_lambda` | 0.5–10 |

For random forest, adapt the existing `../all-tmd-v2/all_tmd/models.py`
search: `n_estimators` 200–800 in steps of 100; `max_depth` from
`[null, 8, 12, 16, 24]`; `min_samples_split` from `[2, 4, 8, 16]`;
`min_samples_leaf` from `[1, 2, 4, 8]`; and `max_features` from
`["sqrt", "log2", 0.5, 0.8]`. XGBoost float ranges are linear unless marked
logarithmic. Use the same resolved labels and training partition for either
family.

For each trial, pool inner out-of-fold predictions before calculating the
configured `selection_metric`; do not average fold-level scores. Macro F1
uses all configured labels, with zero for undefined class scores. Choose the
winning trial and window from inner validation only, then refit before held-out
evaluation. Report both selection metrics even when only one drives selection.

## Evaluation, tracking, and operation

Report macro F1, balanced accuracy, per-class precision/recall/F1/support,
and confusion matrices for each fixed window and applicable selected-window
result. Save supporting CSV tables, held-out predictions, trial histories,
the study summary, and a macro-F1-versus-window line chart. If
`bootstrap_iterations` is enabled, draw seeds with replacement, participants
within sampled seeds, and sessions within sampled participants while retaining
each sampled session's windows. Recompute pooled metrics and generate 95%
percentile intervals for **every evaluated fixed window**, showing them on
the chart. Keep selected-window results separate. If bootstrap is `null`,
omit intervals and bands.

Docker Compose will provide a one-shot experiment service and a persistent
MLflow tracking server. Mount session inputs read-only and keep the run
snapshot, reports, and MLflow store in persistent writable volumes. The
experiment entry point accepts a JSON path, validates and freezes its resolved
settings, prepares the snapshot, evaluates the strategies, writes local
outputs, and streams progress to the MLflow parent and child runs. Publish
the output artifacts to MLflow as stages finish. The CLI and Compose commands
are documented in the README.
