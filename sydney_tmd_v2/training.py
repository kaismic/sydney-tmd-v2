"""Nested tuning and held-out predictions for configurable Sydney studies."""
from __future__ import annotations

import json
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .core import atomic_write, digest, file_hash, read_json, verify_prepared, write_json


def score(actual, predicted, config):
    if config["selection_metric"] == "macro_f1":
        return float(f1_score(actual, predicted, labels=list(config["labels"].values()),
                              average="macro", zero_division=0))
    return float(balanced_accuracy_score(actual, predicted))


def suggest(trial, config):
    family = trial.suggest_categorical("family", config["model_families"])
    prefix = f"{family}__"
    if family == "xgboost":
        return {"family": family,
                "n_estimators": trial.suggest_int(prefix + "n_estimators", 200, 1200, step=100),
                "learning_rate": trial.suggest_float(prefix + "learning_rate", .02, .2, log=True),
                "max_depth": trial.suggest_int(prefix + "max_depth", 3, 8),
                "min_child_weight": trial.suggest_float(prefix + "min_child_weight", 1, 10),
                "subsample": trial.suggest_float(prefix + "subsample", .6, 1),
                "colsample_bytree": trial.suggest_float(prefix + "colsample_bytree", .5, 1),
                "gamma": trial.suggest_float(prefix + "gamma", 0, 5),
                "reg_alpha": trial.suggest_float(prefix + "reg_alpha", 0, 2),
                "reg_lambda": trial.suggest_float(prefix + "reg_lambda", .5, 10)}
    return {"family": family,
            "n_estimators": trial.suggest_int(prefix + "n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_categorical(prefix + "max_depth", [None, 8, 12, 16, 24]),
            "min_samples_split": trial.suggest_categorical(prefix + "min_samples_split", [2, 4, 8, 16]),
            "min_samples_leaf": trial.suggest_categorical(prefix + "min_samples_leaf", [1, 2, 4, 8]),
            "max_features": trial.suggest_categorical(prefix + "max_features", ["sqrt", "log2", 0.5, 0.8])}


def model(params, seed, threads, class_count):
    family = params["family"]
    options = {k: v for k, v in params.items() if k != "family"}
    if family == "xgboost":
        options.update(objective="multi:softprob" if class_count > 2 else "binary:logistic",
                       eval_metric="mlogloss" if class_count > 2 else "logloss",
                       tree_method="hist", device="cpu", random_state=seed, n_jobs=threads)
        estimator = XGBClassifier(**options)
    elif family == "random_forest":
        estimator = RandomForestClassifier(**options, random_state=seed, n_jobs=threads,
                                           class_weight=None)
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                     ("model", estimator)])


def fit(frame, params, seed, threads, config, log, timing_path):
    expected = set(config["labels"].values())
    if set(frame.label) != expected:
        raise ValueError("Training partition is missing configured classes")
    estimator = model(params, seed, threads, len(expected))
    start = time.monotonic()
    estimator.fit(frame[config["feature_names"]], frame.label,
                  model__sample_weight=compute_sample_weight("balanced", frame.label))
    seconds = time.monotonic() - start
    with timing_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"seed": seed, "family": params["family"],
                                 "training_windows": len(frame), "fit_seconds": seconds}) + "\n")
    log.event("model fit complete", family=params["family"], fit_seconds=seconds,
              training_windows=len(frame))
    return estimator


def objective(trial, frame, folds, seed, threads, config, log, timing_path):
    params = suggest(trial, config)
    actual, predicted = [], []
    for index, fold in enumerate(folds, 1):
        train_ids, validation_ids = set(fold["train_sessions"]), set(fold["validation_sessions"])
        if train_ids & validation_ids:
            raise ValueError("Session leakage in inner split")
        train = frame[frame.session_id.isin(train_ids)]
        validation = frame[frame.session_id.isin(validation_ids)]
        if validation.empty:
            raise ValueError("Empty inner validation fold")
        log.task("inner fit", trial=trial.number, inner_fold=index, inner_folds=len(folds))
        estimator = fit(train, params, seed, threads, config, log, timing_path)
        actual.extend(validation.label.tolist())
        predicted.extend(estimator.predict(validation[config["feature_names"]]).tolist())
    return score(actual, predicted, config)


def _history(study, path):
    trials = [{"number": t.number, "state": t.state.name, "value": t.value,
               "params": t.params, "duration_seconds": t.duration.total_seconds() if t.duration else None}
              for t in study.trials]
    write_json(path, {"trials": trials,
                      "completed": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)})
    atomic_write(path.with_suffix(".csv"),
                 lambda p: pd.DataFrame(trials).to_csv(p, index=False))


def search(run, name, frame, folds, seed, threads, config, log, tracking):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    directory = run / "studies"
    directory.mkdir(exist_ok=True)
    storage = f"sqlite:///{(directory / 'studies.sqlite3').resolve().as_posix()}"
    study = optuna.create_study(storage=storage, study_name=name, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=int(digest([seed, name])[:8], 16),
                                                                  n_startup_trials=10),
                                pruner=optuna.pruners.NopPruner())
    identity = {"seed": seed, "folds": folds, "prepared": file_hash(run / "prepared.json"),
                "selection_metric": config["selection_metric"], "families": config["model_families"]}
    if study.user_attrs.get("identity", identity) != identity:
        raise ValueError(f"Study identity mismatch: {name}")
    study.set_user_attr("identity", identity)
    budget = config["optuna_trials"]
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if completed > budget:
        raise ValueError(f"Study exceeds trial budget: {name}")
    if completed < budget:
        with tracking.child(name, "tuning_study"):
            tracking.params({"seed": seed, "window": name.split("-w")[-1].split("-")[0],
                             "selection_metric": config["selection_metric"]})
            def on_trial(current, trial):
                _history(current, directory / f"{name}.json")
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    tracking.metrics({"inner.score": trial.value,
                                      "inner.best_score": current.best_value}, step=trial.number)
                log.event("trial complete", study=name, completed_trials=sum(
                    t.state == optuna.trial.TrialState.COMPLETE for t in current.trials), trial_budget=budget)

            while sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) < budget:
                study.optimize(lambda trial: objective(trial, frame, folds, seed, threads, config, log,
                                                        run / "fit-times.jsonl"),
                               n_trials=1, n_jobs=1, callbacks=[on_trial])
            _history(study, directory / f"{name}.json")
            tracking.file(directory / f"{name}.json", "studies")
            tracking.file(directory / f"{name}.csv", "studies")
    if not study.best_trials:
        raise ValueError(f"Study has no successful trials: {name}")
    tracking.file(directory / f"{name}.json", "studies")
    tracking.file(directory / f"{name}.csv", "studies")
    best = min((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
               key=lambda t: (-t.value, t.number))
    params = {k.split("__", 1)[-1]: v for k, v in best.params.items() if k != "family"}
    return {"trial": best.number, "score": best.value,
            "params": {"family": best.params["family"], **params}}


def _bundle(directory):
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    for name, sha in read_json(marker)["files"].items():
        if file_hash(directory / name) != sha:
            raise ValueError(f"Corrupt completed artifact: {directory / name}")
    return True


def evaluate(run, log, tracking):
    identity, _ = verify_prepared(run)
    config = identity["configuration"]
    windows = config["windows_seconds"]
    frames = {w: pd.read_parquet(run / f"features-{w}.parquet") for w in windows}
    splits = read_json(run / "splits.json")
    bundles = []
    for index, fold in enumerate(splits["outer"]):
        seed, fold_id = fold["seed"], fold["id"]
        directory = run / "outer" / f"fold-{index:03d}-seed-{seed}"
        if _bundle(directory):
            log.event("outer result cache hit", fold=fold_id, seed=seed)
            with tracking.child(f"result-{fold_id}-seed-{seed}", "heldout_result"):
                tracking.file(directory / "selection.json", "results")
                tracking.file(directory / "predictions.parquet", "results")
        else:
            winners = {}
            for w in windows:
                train = frames[w][frames[w].session_id.isin(fold["train_pool_sessions"])]
                winners[w] = search(run, f"outer-{index:03d}-w{w}-s{seed}", train,
                                    fold["inner"], seed, identity["threads"], config, log, tracking)
            selected = min(windows, key=lambda w: (-winners[w]["score"], w))
            rows = []
            for w in windows:
                log.task("outer refit", fold=fold_id, seed=seed, window=w)
                train = frames[w][frames[w].session_id.isin(fold["train_sessions"])]
                test = frames[w][frames[w].session_id.isin(fold["test_sessions"])]
                estimator = fit(train, winners[w]["params"], seed, identity["threads"],
                                config, log, run / "fit-times.jsonl")
                prediction = test[["session_id", "participant_id", "label",
                                   "window_start_ms", "window_end_ms"]].copy()
                prediction["prediction"] = estimator.predict(test[config["feature_names"]]).astype(int)
                probabilities = estimator.predict_proba(test[config["feature_names"]])
                for code in config["labels"].values():
                    prediction[f"probability_{code}"] = probabilities[:, code]
                prediction["seed"], prediction["window"] = seed, w
                prediction["outer_fold"] = fold_id
                rows.append(prediction.assign(result=str(w)))
                if w == selected:
                    rows.append(prediction.assign(result="selected"))
            write_json(directory / "selection.json", {"fold": fold_id, "seed": seed,
                                                       "selected_window": selected, "winners": winners})
            atomic_write(directory / "predictions.parquet",
                         lambda p: pd.concat(rows, ignore_index=True).to_parquet(p, index=False))
            write_json(directory / "complete.json", {"files": {
                name: file_hash(directory / name) for name in ("selection.json", "predictions.parquet")}})
            with tracking.child(f"result-{fold_id}-seed-{seed}", "heldout_result"):
                tracking.params({"seed": seed, "outer_fold": fold_id, "selected_window": selected})
                tracking.file(directory / "selection.json", "results")
                tracking.file(directory / "predictions.parquet", "results")
        bundles.append(directory.relative_to(run).as_posix())
        log.event("outer fold complete", completed=index + 1, total=len(splits["outer"]),
                  fold=fold_id, seed=seed)
    write_json(run / "evaluated.json", {"prepared": file_hash(run / "prepared.json"),
                                        "bundles": bundles})
    tracking.file(run / "evaluated.json")
