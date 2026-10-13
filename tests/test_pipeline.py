import json
import shutil

import numpy as np
import pytest

from sydney_tmd_v2.cli import main
from sydney_tmd_v2.config import validate_config
from sydney_tmd_v2.core import read_json
from sydney_tmd_v2.data import make_splits
from sydney_tmd_v2.training import model


def test_config_rejects_invalid_cross_validation_and_unknown_features():
    assert validate_config({"cv_fold_count": 2})["cv_fold_count"] == 2
    with pytest.raises(ValueError, match="cv_fold_count"):
        validate_config({"cv_fold_count": 1})
    with pytest.raises(ValueError, match="Unregistered features"):
        validate_config({"feature_names": ["unknown#mean"]})
    with pytest.raises(ValueError, match="overlap"):
        validate_config({"evaluation_strategy": "manual_participant_group",
                         "manual_participant_group": {"train": ["a"], "test": ["a"]}})


def test_session_split_keeps_sessions_together_and_samples_per_seed():
    config = validate_config({"windows_seconds": [4, 6], "seeds": [7],
                              "labels": {"bus": 0, "car": 1},
                              "cv_fold_count": 2, "minimum_window_count_per_participant": 1,
                              "evaluation_strategy": "session_train_test_split",
                              "session_train_test_split_ratio": 0.7})
    cohort = [{"session_id": f"{person}-{mode}-{i}", "participant_id": person,
               "label": code, "window_counts": {"4": 3, "6": 1}}
              for person in ("a", "b") for mode, code in config["labels"].items()
              for i in range(3)]
    splits = make_splits(cohort, config)
    fold = splits["outer"][0]
    assert set(fold["train_pool_sessions"]).isdisjoint(fold["test_sessions"])
    assert set(fold["train_pool_sessions"]) | set(fold["test_sessions"]) == {x["session_id"] for x in cohort}
    assert all(set(inner["train_sessions"]).isdisjoint(inner["validation_sessions"])
               for inner in fold["inner"])
    assert splits == make_splits(cohort, config)


def test_both_model_families_accept_binary_labels():
    for params in ({"family": "xgboost", "n_estimators": 2, "max_depth": 2},
                   {"family": "random_forest", "n_estimators": 2}):
        estimator = model(params, 7, 1, 2)
        estimator.fit(np.array([[0.], [1.], [0.1], [1.1]]), [0, 1, 0, 1])
        assert estimator.predict([[0.], [1.]]).shape == (2,)


def _session(path, person, mode, session_id):
    count = 401
    ts = np.arange(count) * 20
    amplitude = 1 if mode == "bus" else 5
    samples = [{"ts": int(t), "ax": float(amplitude + np.sin(t / 300)), "ay": 1., "az": 1.,
                "gx": float(amplitude + np.cos(t / 400)), "gy": 1., "gz": 1.,
                "mx": float(amplitude), "my": 1., "mz": 1.} for t in ts]
    path.write_text(json.dumps({"session_id": session_id, "participant_id": person,
                                "collection_country_code": "AU", "vehicle_type": mode,
                                "samples": samples}), encoding="utf-8")


def test_small_end_to_end_experiment_with_snapshot_and_report(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    for person in ("a", "b", "c"):
        for mode in ("bus", "car"):
            _session(source / f"{person}-{mode}.json", person, mode, f"{person}-{mode}")
    foreign = source / "foreign-bus.json"
    _session(foreign, "foreign", "bus", "foreign-bus")
    payload = json.loads(foreign.read_text(encoding="utf-8"))
    payload["collection_country_code"] = "KR"
    foreign.write_text(json.dumps(payload), encoding="utf-8")
    _session(source / "unlisted-walk.json", "unlisted", "walk", "unlisted-walk")
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps({
        "feature_names": ["accelerometer#mean", "gyroscope#standard_deviation"],
        "windows_seconds": [4, 6], "seeds": [7], "labels": {"bus": 0, "car": 1},
        "evaluation_strategy": "manual_participant_group",
        "manual_participant_group": {"train": ["a", "b"], "test": ["c"]},
        "minimum_window_count_per_participant": 1, "cv_fold_count": 2,
        "optuna_trials": 1, "bootstrap_iterations": 8,
    }), encoding="utf-8")
    runs, logs = tmp_path / "runs", tmp_path / "logs"
    run_id = "20261013T000000Z-1234abcd"
    assert main(["all", "--input", str(source), "--config", str(config_path),
                 "--run-id", run_id, "--runs-dir", str(runs),
                 "--logs-dir", str(logs), "--no-mlflow"]) == 0
    run = runs / run_id
    snapshot = read_json(run / "snapshot.json")
    assert len(snapshot) == 6
    assert all((run / file["snapshot"]).exists() for item in snapshot for file in item["files"])
    assert {x["exclusion"] for x in read_json(run / "cohort.json")["exclusions"]} == {
        "non_AU", "unsupported_class"}
    report = read_json(run / "reports" / "study-summary.json")
    assert set(report["results"]) == {"4", "6", "selected"}
    assert report["results"]["4"]["intervals"]["macro_f1"]["low"] >= 0
    assert (run / "reports" / "macro-f1.png").exists()
    shutil.rmtree(source)
    assert main(["report", "--run-id", run_id, "--runs-dir", str(runs),
                 "--logs-dir", str(logs), "--no-mlflow"]) == 0
