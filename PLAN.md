# Sydney TMD V2

## Plan

Create a new project (`sydney-tmd-v2`) based on `ml/sydney-tmd-v1`
The project should:
- Refer most of its project structure from `sydney-tmd-v1`
- Run on docker
- Only use Sydney data
- The project should have a script which accepts a `.json` file and run the experiment.
- Each experiment run should record a reproducible snapshot of the dataset when running the experiment.
- Each run should be saved to mlflow and viewable on docker MLflow. The run name should contain the experiment execution datetime and randomly generated 6 hex characters.
- Each run produce a set of `.csv` files and a line

## Configuration File Description
- `feature_names`: e.g. ["accelerometer#mean", "accelerometer#standard_deviation", "gyroscope#standard_deviation"]
- `windows_seconds`: e.g. [100, 120, 140]
    - The maximum window length should be the minimum session duration in the preprocessing step.
- `seeds`: e.g. [42, 43, 44]
- `individual_transport_training_ratios`: e.g. { "bus": 1.0, "car": 1.0, "train": 1.0 }
- `labels`: e.g. {"bus": 0, "car": 1, "train": 2}
- `evaluation_strategy` e.g. "participant_lopo", "manual_participant_selection", "train_test_split"
- `selection_metric`: e.g. "macro_f1", "balanced_accuracy"
- `bootstrap_iterations`: null, or a number e.g. 5000
- `optuna_trials`: e.g. 100,
- `model_families`: e.g. ["xgboost"],
- `train_test_split_ratio`: e.g. null, or 0.8
- `cv_fold_count`: e.g. null, or a number 5
- `minimum_window_count_per_participant`: e.g. 10
    - If a participant contains total windows less than `minimum_window_count_per_participant`, the participant should be excluded from the experiment.

### `evaluation_strategy` Description

#### `participant_lopo`
Leave one participant out. For example, if there are total 7 participants, leave 1 out, train on other 6 and evaluate on the excluded participant, repeating 7 times and average the metric.

#### `manual_participant_group`
User inputs participants manually, for example:
{
    "train": ["participant_001", "participant_003", ...]
    "test": ["participant_004", "participant_005", ...]
}

#### `train_test_split`
If this strategy is selected, "train_test_split_ratio" and "cv_fold_count" is required.

The session data should be sliced into the given window length and the features should be extracted first, with the participant and transport mode metadata label to each extracted data.

Split the extracted features into training and testing set. `train_test_split_ratio` should be the training set ratio and `1 - train_test_split_ratio` should be the testing set ratio.

The split should aim to maintain the equal ratio for the window for each participant. For example, If participant A contains 5 windows after feature extraction, and `train_test_split_ratio` = 0.8, 4 (= 5 * 0.8) windows should go into the training set and 1 (= 5 * 0.2) window should go into the testing set. If the `train_test_split_ratio` * `total_window_count` is not an integer, use floor rounding.


### `bootstrap_iterations`
If `bootstrap_iterations` is not null, create `bootstrap_iterations` amount of sets with all participants and all seed values provided.
For example, if 3 seed values are provided and there are total 7 participants,
```
sample 3 training seeds with replacement
sample 7 participants with replacement
sample sessions within each selected participant
recompute pooled macro F1 for every sampled seed
average the 3 sampled-seed scores
```

### `individual_transport_training_ratios`
Each fraction value denotes the actual ratio of training data to use from the training set. Therefore, 1.0 should use the full training set. 0.5 should use half of the training set to train models. Picking the data from the training set should use the given seed for reproducibility. Default value is 1.0 for each transport mode.


## Data

Freeze a new input manifest from the locally downloaded Sydney sessions available
when preparation runs. The input directory contains session JSON or compressed
JSON files; downloading or synchronizing data is a separate prerequisite.

- Require an explicit `collection_country_code == "AU"`. Exclude missing country
  labels and non-AU sessions, including legacy records that the reference study
  admitted without a country label.
- Use the same session cohort for all window sizes: retain a session only if it
  produces at least one quality-valid window at each of 100, 120, and 140 seconds.
  Window counts may differ across sizes even though session membership is fixed.
- Record input fingerprints, session and participant identities, exclusion
  reasons, class counts, and participant/class coverage. Derive participant counts
  from this cohort; do not inherit the previous snapshot or its seven-participant
  assumption.






TODO


## Features


Use complete windows within each session, with 50% overlap:

| Window length | Step |
|---|---|
| 100 seconds | 50 seconds |
| 120 seconds | 60 seconds |
| 140 seconds | 70 seconds |

