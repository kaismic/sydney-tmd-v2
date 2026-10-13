"""Metrics, hierarchical intervals, CSV tables, and result charts."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .core import atomic_write, file_hash, read_json, verify_prepared, write_json
from .training import _bundle


def matrix_metrics(cm, labels):
    cm = np.asarray(cm, dtype=float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    correct = np.diag(cm)
    precision = np.divide(correct, predicted, out=np.zeros_like(correct), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros_like(correct), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros_like(correct),
                   where=support + predicted > 0)
    return {"macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall[support > 0].mean()) if any(support > 0) else 0.0,
            "per_class": [{"class": name, "label": code, "precision": float(precision[code]),
                           "recall": float(recall[code]), "f1": float(f1[code]),
                           "support": int(support[code])} for name, code in labels.items()]}


def pooled_metrics(rows, labels):
    return matrix_metrics(confusion_matrix(rows.label, rows.prediction,
                                           labels=list(range(len(labels)))), labels)


def bootstrap(rows, labels, draws, seed=42):
    """Resample seeds, then participants, then sessions; preserve session windows."""
    clusters = {}
    codes = list(range(len(labels)))
    for (run_seed, person, session), group in rows.groupby(["seed", "participant_id", "session_id"]):
        clusters.setdefault(int(run_seed), {}).setdefault(person, []).append(
            confusion_matrix(group.label, group.prediction, labels=codes))
    seeds = sorted(clusters)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        seed_scores = []
        for chosen_seed in rng.choice(seeds, len(seeds), replace=True):
            people = sorted(clusters[chosen_seed])
            cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
            for person in rng.choice(people, len(people), replace=True):
                sessions = np.asarray(clusters[chosen_seed][person])
                cm += sessions[rng.integers(len(sessions), size=len(sessions))].sum(axis=0)
            metrics = matrix_metrics(cm, labels)
            seed_scores.append([metrics["macro_f1"], metrics["balanced_accuracy"]])
        samples.append(np.mean(seed_scores, axis=0))
    samples = np.asarray(samples)
    bounds = np.percentile(samples, [2.5, 97.5], axis=0)
    return {name: {"low": float(bounds[0, i]), "high": float(bounds[1, i])}
            for i, name in enumerate(("macro_f1", "balanced_accuracy"))}, samples


def chart(summary, windows, path, intervals_enabled):
    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = sorted(windows)
    means = [summary[str(w)]["mean"]["macro_f1"] for w in ordered]
    ax.plot(ordered, means, "-o", color="#2266aa", label="Mean over seeds")
    if intervals_enabled:
        lower = [summary[str(w)]["intervals"]["macro_f1"]["low"] for w in ordered]
        upper = [summary[str(w)]["intervals"]["macro_f1"]["high"] for w in ordered]
        ax.fill_between(ordered, lower, upper, color="#2266aa", alpha=.2,
                        label="95% hierarchical interval")
    ax.set_xticks(ordered)
    ax.set(xlabel="Window (seconds)", ylabel="Held-out macro F1", ylim=(0, 1),
           title="Sydney transport-mode window comparison")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    atomic_write(path, lambda p: fig.savefig(p, format="png", dpi=180))
    plt.close(fig)


def confusion_chart(matrices, labels, result, path, normalized):
    """Plot each seed separately because its held-out windows may recur in other seeds."""
    names = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
    count = len(names)
    width = max(4, count * 1.1 + 1.5)
    fig, axes = plt.subplots(1, len(matrices), figsize=(width * len(matrices), width),
                             squeeze=False)
    try:
        maximum = max(1, *(int(matrix.max()) for matrix in matrices.values()))
        for ax, (seed, counts) in zip(axes.flat, sorted(matrices.items())):
            if normalized:
                totals = counts.sum(axis=1, keepdims=True)
                values = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float),
                                   where=totals != 0)
            else:
                values = counts
            ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else maximum)
            for actual in range(count):
                for predicted in range(count):
                    value = values[actual, predicted]
                    label = f"{value:.1%}" if normalized else str(value)
                    ax.text(predicted, actual, label, ha="center", va="center",
                            color="white" if value > (0.5 if normalized else maximum / 2)
                            else "black")
            ax.set(xticks=range(count), yticks=range(count), xticklabels=names,
                   yticklabels=names, xlabel="Predicted", ylabel="Actual", title=f"Seed {seed}")
        title = "Selected window" if result == "selected" else f"{result}-second windows"
        fig.suptitle(title + (" — row-normalized" if normalized else " — window counts"))
        fig.tight_layout()
        atomic_write(path, lambda p: fig.savefig(p, format="png", dpi=160))
    finally:
        plt.close(fig)


def confusion_chart_names(result):
    return [f"confusion-{result}-normalized.png", f"confusion-{result}-counts.png"]


def report(run, log, tracking):
    identity, _ = verify_prepared(run)
    config = identity["configuration"]
    evaluation = read_json(run / "evaluated.json")
    if evaluation["prepared"] != file_hash(run / "prepared.json"):
        raise ValueError("Evaluation does not match prepared snapshot")
    for bundle in evaluation["bundles"]:
        if not _bundle(run / bundle):
            raise ValueError(f"Incomplete result bundle: {bundle}")
    directory = run / "reports"
    directory.mkdir(exist_ok=True)
    results = [str(w) for w in config["windows_seconds"]] + ["selected"]
    if _bundle(directory):
        marker = read_json(directory / "complete.json")
        if marker["evaluation_sha256"] != file_hash(run / "evaluated.json"):
            raise ValueError("Report does not match evaluation")
        for name in marker["files"]:
            tracking.file(directory / name, "reports")
        for result in results:
            charts = confusion_chart_names(result)
            if all(name in marker["files"] for name in charts):
                with tracking.child(f"summary-{result}", "heldout_result"):
                    for name in charts:
                        tracking.file(directory / name, "charts")
        log.event("report cache hit", output=str(directory))
        return
    rows = pd.concat([pd.read_parquet(run / b / "predictions.parquet")
                      for b in evaluation["bundles"]], ignore_index=True)
    atomic_write(directory / "predictions.csv", lambda p: rows.to_csv(p, index=False))
    labels = config["labels"]
    seeds = config["seeds"]
    summary, seed_rows, class_rows, participant_rows, matrices, samples_files, chart_files = (
        {}, [], [], [], [], [], [])
    for result in results:
        group = rows[rows.result == result]
        by_seed, result_matrices = {}, {}
        for seed in seeds:
            data = group[group.seed == seed]
            if data.empty or data.duplicated(["session_id", "window_start_ms"]).any():
                raise ValueError(f"Empty or duplicate held-out predictions for {result}, seed {seed}")
            metrics = pooled_metrics(data, labels)
            by_seed[str(seed)] = {k: metrics[k] for k in ("macro_f1", "balanced_accuracy")}
            seed_rows.append({"result": result, "seed": seed, **by_seed[str(seed)]})
            class_rows.extend({"result": result, "seed": seed, **item} for item in metrics["per_class"])
            cm = confusion_matrix(data.label, data.prediction, labels=list(range(len(labels))))
            result_matrices[seed] = cm
            matrices.extend({"result": result, "seed": seed, "actual": actual,
                             "predicted": predicted, "windows": int(cm[actual, predicted])}
                            for actual in range(len(labels)) for predicted in range(len(labels)))
            for person, subset in data.groupby("participant_id"):
                personal = pooled_metrics(subset, labels)
                participant_rows.append({"result": result, "seed": seed, "participant_id": person,
                                         "windows": len(subset), "macro_f1": personal["macro_f1"],
                                         "balanced_accuracy": personal["balanced_accuracy"]})
        means = {metric: float(np.mean([x[metric] for x in by_seed.values()]))
                 for metric in ("macro_f1", "balanced_accuracy")}
        intervals = None
        if config["bootstrap_iterations"] is not None and result != "selected":
            intervals, samples = bootstrap(group, labels, config["bootstrap_iterations"])
            filename = f"bootstrap-{result}.csv"
            atomic_write(directory / filename,
                         lambda p: pd.DataFrame(samples, columns=["macro_f1", "balanced_accuracy"])
                         .to_csv(p, index=False))
            samples_files.append(filename)
        summary[result] = {"seed_scores": by_seed, "mean": means, "intervals": intervals}
        for normalized, filename in zip((True, False), confusion_chart_names(result)):
            confusion_chart(result_matrices, labels, result, directory / filename, normalized)
            chart_files.append(filename)
        with tracking.child(f"summary-{result}", "heldout_result"):
            tracking.params({"result": result, "selection_metric": config["selection_metric"]})
            tracking.metrics(means)
            for filename in confusion_chart_names(result):
                tracking.file(directory / filename, "charts")
        log.event("result metrics complete", result=result, **means)
    metric_rows = [{"result": result, **item["mean"],
                    "macro_f1_low": item["intervals"]["macro_f1"]["low"] if item["intervals"] else None,
                    "macro_f1_high": item["intervals"]["macro_f1"]["high"] if item["intervals"] else None}
                   for result, item in summary.items()]
    tables = {"metrics.csv": metric_rows, "seed-metrics.csv": seed_rows,
              "per-class.csv": class_rows, "participant-metrics.csv": participant_rows,
              "confusion-matrices.csv": matrices}
    for name, data in tables.items():
        atomic_write(directory / name, lambda p: pd.DataFrame(data).to_csv(p, index=False))
    chart(summary, config["windows_seconds"], directory / "macro-f1.png",
          config["bootstrap_iterations"] is not None)
    write_json(directory / "study-summary.json", {"configuration": config,
               "results": summary, "run_name": run.name,
               "cohort": read_json(run / "cohort.json"),
               "splits": read_json(run / "splits.json"),
               "bootstrap_note": "Intervals resample saved predictions, not model training or selection."})
    files = ["predictions.csv", *tables, *samples_files, *chart_files,
             "macro-f1.png", "study-summary.json"]
    write_json(directory / "complete.json", {"evaluation_sha256": file_hash(run / "evaluated.json"),
                 "files": {name: file_hash(directory / name) for name in files}})
    for name in files:
        tracking.file(directory / name, "reports")
    log.event("report complete", output=str(directory))
