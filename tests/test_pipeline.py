import json
import shutil
from contextlib import contextmanager

import numpy as np
import pytest
from matplotlib.figure import Figure

from sydney_tmd_v2.cli import main
from sydney_tmd_v2.config import validate_config
from sydney_tmd_v2.core import read_json
from sydney_tmd_v2.data import make_splits
from sydney_tmd_v2.reporting import confusion_chart, confusion_chart_names, report
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


def test_confusion_charts_show_each_seed_dynamic_labels_and_zero_support(tmp_path, monkeypatch):
    matrices = {43: np.array([[0, 2], [3, 1]]),
                42: np.array([[1, 1], [0, 0]])}
    labels = {"ferry": 1, "tram": 0}
    captured = []
    savefig = Figure.savefig

    def capture(self, path, *args, **kwargs):
        captured.append([{
            "title": ax.get_title(),
            "labels": [tick.get_text() for tick in ax.get_xticklabels()],
            "values": np.asarray(ax.images[0].get_array()).copy(),
            "annotations": [text.get_text() for text in ax.texts],
        } for ax in self.axes])
        return savefig(self, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    for normalized, filename in zip((True, False), confusion_chart_names("4")):
        with np.errstate(divide="raise", invalid="raise"):
            confusion_chart(matrices, labels, "4", tmp_path / filename, normalized)
        assert (tmp_path / filename).stat().st_size > 0
    assert [panel["title"] for panel in captured[0]] == ["Seed 42", "Seed 43"]
    assert all(panel["labels"] == ["tram", "ferry"] for panel in captured[0])
    np.testing.assert_array_equal(captured[0][0]["values"], [[0.5, 0.5], [0, 0]])
    assert captured[0][0]["annotations"] == ["50.0%", "50.0%", "0.0%", "0.0%"]
    np.testing.assert_array_equal(captured[1][1]["values"], matrices[43])
    assert captured[1][1]["annotations"] == ["0", "2", "3", "1"]


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
    summary = read_json(run / "reports" / "study-summary.json")
    assert set(summary["results"]) == {"4", "6", "selected"}
    assert summary["results"]["4"]["intervals"]["macro_f1"]["low"] >= 0
    assert (run / "reports" / "macro-f1.png").exists()
    marker = read_json(run / "reports" / "complete.json")
    chart_names = {name for result in ("4", "6", "selected")
                   for name in confusion_chart_names(result)}
    assert chart_names <= marker["files"].keys()
    assert all((run / "reports" / name).stat().st_size > 0 for name in chart_names)
    shutil.rmtree(source)
    assert main(["report", "--run-id", run_id, "--runs-dir", str(runs),
                 "--logs-dir", str(logs), "--no-mlflow"]) == 0
    assert read_json(run / "reports" / "complete.json") == marker

    class RecordingTracking:
        def __init__(self):
            self.child_name = None
            self.files = []

        @contextmanager
        def child(self, name, kind):
            self.child_name = name
            yield self
            self.child_name = None

        def file(self, path, artifact_path):
            self.files.append((self.child_name, path.name, artifact_path))

    class SilentLog:
        def event(self, *args, **kwargs):
            pass

    tracking = RecordingTracking()
    report(run, SilentLog(), tracking)
    for result in ("4", "6", "selected"):
        for name in confusion_chart_names(result):
            assert (None, name, "reports") in tracking.files
            assert (f"summary-{result}", name, "charts") in tracking.files
