"""Resolve and validate the public experiment JSON."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .core import DEFAULT_LABELS, DEFAULT_SEEDS, DEFAULT_WINDOWS, FEATURES

STRATEGIES = ("participant_lopo", "manual_participant_group", "session_train_test_split")
METRICS = ("macro_f1", "balanced_accuracy")
FAMILIES = ("xgboost", "random_forest")
DEFAULTS = {
    "feature_names": FEATURES,
    "windows_seconds": list(DEFAULT_WINDOWS),
    "seeds": list(DEFAULT_SEEDS),
    "individual_transport_training_ratios": None,
    "labels": DEFAULT_LABELS,
    "evaluation_strategy": "participant_lopo",
    "selection_metric": "macro_f1",
    "bootstrap_iterations": 5000,
    "optuna_trials": 100,
    "model_families": ["xgboost"],
    "cv_fold_count": 3,
    "minimum_window_count_per_participant": 10,
    "manual_participant_group": None,
    "session_train_test_split_ratio": None,
}


def _positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _distinct_list(value, name, element_type):
    if not isinstance(value, list) or not value or any(type(x) is not element_type for x in value):
        raise ValueError(f"{name} must be a nonempty list of {element_type.__name__} values")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicates")
    return value


def validate_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("Experiment JSON must be an object")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown experiment fields: {sorted(unknown)}")
    config = {**DEFAULTS, **raw}
    config["feature_names"] = list(_distinct_list(config["feature_names"], "feature_names", str))
    unknown_features = sorted(set(config["feature_names"]) - set(FEATURES))
    if unknown_features:
        raise ValueError(f"Unregistered features: {unknown_features}")
    config["windows_seconds"] = list(_distinct_list(config["windows_seconds"], "windows_seconds", int))
    for value in config["windows_seconds"]:
        _positive_int(value, "windows_seconds item")
    config["seeds"] = list(_distinct_list(config["seeds"], "seeds", int))
    for value in config["seeds"]:
        if value < 0:
            raise ValueError("seeds must be nonnegative")
    labels = config["labels"]
    if (not isinstance(labels, dict) or len(labels) < 2
            or any(not isinstance(k, str) or not k.strip() or type(v) is not int for k, v in labels.items())
            or sorted(labels.values()) != list(range(len(labels)))):
        raise ValueError("labels must map at least two vehicle types to contiguous codes starting at zero")
    config["labels"] = dict(labels)
    ratios = config["individual_transport_training_ratios"]
    if ratios is None:
        ratios = {name: 1.0 for name in labels}
    if not isinstance(ratios, dict) or set(ratios) != set(labels):
        raise ValueError("individual_transport_training_ratios must cover exactly the configured labels")
    for name, value in ratios.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"Invalid training ratio for {name}: expected (0, 1]")
    config["individual_transport_training_ratios"] = {k: float(v) for k, v in ratios.items()}
    if config["evaluation_strategy"] not in STRATEGIES:
        raise ValueError(f"evaluation_strategy must be one of {STRATEGIES}")
    if config["selection_metric"] not in METRICS:
        raise ValueError(f"selection_metric must be one of {METRICS}")
    if config["bootstrap_iterations"] is not None:
        _positive_int(config["bootstrap_iterations"], "bootstrap_iterations")
    _positive_int(config["optuna_trials"], "optuna_trials")
    _positive_int(config["cv_fold_count"], "cv_fold_count", 2)
    _positive_int(config["minimum_window_count_per_participant"], "minimum_window_count_per_participant")
    families = _distinct_list(config["model_families"], "model_families", str)
    if set(families) - set(FAMILIES):
        raise ValueError(f"model_families may contain only {FAMILIES}")
    config["model_families"] = list(families)
    groups, ratio, strategy = (config["manual_participant_group"],
                               config["session_train_test_split_ratio"], config["evaluation_strategy"])
    if strategy == "manual_participant_group":
        if not isinstance(groups, dict) or set(groups) != {"train", "test"}:
            raise ValueError("manual_participant_group requires train and test lists")
        for name in ("train", "test"):
            _distinct_list(groups[name], f"manual_participant_group.{name}", str)
        if set(groups["train"]) & set(groups["test"]):
            raise ValueError("Manual train and test participants overlap")
    elif groups is not None:
        raise ValueError("manual_participant_group is only valid for its evaluation strategy")
    if strategy == "session_train_test_split":
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 < ratio < 1:
            raise ValueError("session_train_test_split_ratio must be in (0, 1)")
        config["session_train_test_split_ratio"] = float(ratio)
    elif ratio is not None:
        raise ValueError("session_train_test_split_ratio is only valid for its evaluation strategy")
    return config


def load_config(path):
    try:
        return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid experiment JSON: {exc}") from exc
