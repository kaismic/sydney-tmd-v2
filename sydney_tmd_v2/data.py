"""Collector-only ingestion and reference-compatible sensor features."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .core import (FEATURES, PROTOCOL, RATES, SENSORS, atomic_write,
                   code_fingerprint, digest, environment, file_hash, read_json,
                   verify_prepared, write_json)


def discover(root):
    return sorted(p for p in root.rglob("*") if p.is_file()
                  and (p.name.endswith(".json") or p.name.endswith(".json.gz"))
                  and not p.name.endswith(".metadata.json")
                  and not any(part.startswith(".") for part in p.relative_to(root).parts))


def sidecars(path):
    return [p for p in (path.with_suffix(f"{path.suffix}.metadata.json"),
                       path.with_suffix(".metadata.json")) if p.exists()]


def load_session(path, labels, minimum_seconds):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    metadata = read_json(sidecars(path)[0]) if sidecars(path) else {}
    if not isinstance(payload, dict) or not isinstance(metadata, dict):
        raise ValueError("Session and metadata must be JSON objects")
    # Reject conflicting identities instead of placing the same person on both sides of a fold.
    for field in ("participant_id", "session_id", "collection_country_code"):
        a, b = payload.get(field), metadata.get(field)
        if a is not None and b is not None and a != b:
            raise ValueError(f"Conflicting {field} in payload and sidecar")
    info = {**payload, **metadata}
    country = info.get("collection_country_code")
    record = {"session_id": str(info.get("session_id") or path.name.removesuffix(".gz").removesuffix(".json")),
              "participant_id": info.get("participant_id"), "country": country,
              "vehicle_type": str(payload.get("vehicle_type") or metadata.get("vehicle_type") or "").lower()}
    if country != "AU":
        return record, None, "missing_country" if country is None else "non_AU"
    if record["vehicle_type"] not in labels:
        return record, None, "unsupported_class"
    if not isinstance(record["participant_id"], str) or not record["participant_id"].strip():
        return record, None, "missing_participant"
    if "raw" in path.parts:
        index = path.parts.index("raw")
        if index + 2 < len(path.parts) and path.parts[index + 1] != record["participant_id"]:
            raise ValueError("Participant sidecar/path mismatch")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        return record, None, "no_samples"
    frame = pd.DataFrame(samples)
    if "ts" not in frame:
        raise ValueError("Session samples lack ts")
    frame = frame.reindex(columns=["ts", *[c for cols in SENSORS.values() for c in cols]])
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["ts"])
    frame["ts"] = frame.ts.astype("int64")
    # Payload trim bounds have the same precedence as the reference; zero is a valid bound.
    bounds = [payload.get(k) if payload.get(k) is not None else metadata.get(k)
              for k in ("trimmed_start_ms", "trimmed_end_ms")]
    if all(v is not None for v in bounds):
        frame = frame[frame.ts.between(int(bounds[0]), int(bounds[1]))]
    frame = frame.sort_values("ts", kind="stable").reset_index(drop=True)
    if frame.empty:
        return record, None, "no_usable_samples"
    record["duration_seconds"] = float((frame.ts.max() - frame.ts.min()) / 1000)
    if record["duration_seconds"] < minimum_seconds:
        return record, None, "duration_below_maximum_window"
    record["label"] = labels[record["vehicle_type"]]
    return record, frame, None


def extract(frame, seconds, feature_names=FEATURES):
    """Preserve reference boundaries, gap checks, feature ordering and float32 storage."""
    times = frame.ts.to_numpy(dtype=np.int64)
    diffs = np.diff(times)
    interval = int(np.median(diffs[diffs > 0])) if (diffs > 0).any() else seconds * 500
    end = int(times[-1]) + interval
    rows, generated = [], 0
    sensors = {}
    for sensor, columns in SENSORS.items():
        vectors = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
        valid = np.isfinite(vectors).all(axis=1)
        sensors[sensor] = (times[valid], np.linalg.norm(vectors[valid], axis=1))
    for start in range(int(times[0]), end - seconds * 1000 + 1, seconds * 500):
        generated += 1
        stop, features = start + seconds * 1000, []
        for sensor, (timestamps, magnitudes) in sensors.items():
            left, right = np.searchsorted(timestamps, [start, stop], side="left")
            t, values = timestamps[left:right], magnitudes[left:right]
            if len(t) < RATES[sensor] * seconds or np.diff(np.r_[start, t, stop]).max() > 500:
                break
            values = values[np.r_[True, np.diff(t) > 0]]
            stats = [np.mean(values), np.std(values, ddof=0), np.ptp(values), np.min(values)]
            features.extend(stats if sensor != "magnetometer" else stats[1:])
        else:
            rows.append(dict(window_start_ms=start, window_end_ms=stop,
                             **dict(zip(FEATURES, np.asarray(features, dtype=np.float32)))))
    return (pd.DataFrame(rows, columns=["window_start_ms", "window_end_ms", *FEATURES])
            [["window_start_ms", "window_end_ms", *feature_names]], generated)


def inventory(root):
    paths = discover(root)
    if not paths:
        raise ValueError(f"No session JSON files found under {root}")
    return [{"path": path.relative_to(root).as_posix(), "sha256": file_hash(path),
             "sidecars": {p.relative_to(root).as_posix(): file_hash(p) for p in sidecars(path)}}
            for path in paths]


def _seed(seed, *parts):
    return int(digest([seed, *parts])[:16], 16)


def make_splits(cohort, config):
    """Freeze all outer/inner session IDs before tuning or examining predictions."""
    by_id = {record["session_id"]: record for record in cohort}
    if len(by_id) != len(cohort):
        raise ValueError("Duplicate session ID in cohort")
    sessions = sorted(by_id)
    people = sorted({r["participant_id"] for r in cohort})
    labels = config["labels"]
    folds = config["cv_fold_count"]

    def require_classes(ids, where):
        present = {by_id[x]["label"] for x in ids}
        missing = [name for name, code in labels.items() if code not in present]
        if missing:
            raise ValueError(f"Missing training classes {missing} in {where}")

    def sample(ids, seed, where):
        require_classes(ids, where)
        chosen = []
        for name, code in labels.items():
            options = sorted(x for x in ids if by_id[x]["label"] == code)
            count = max(1, math.floor(len(options) * config["individual_transport_training_ratios"][name]))
            rng = np.random.default_rng(_seed(seed, where, name))
            chosen.extend(rng.permutation(options)[:count].tolist())
        return sorted(chosen)

    def inner(pool, seed, where, unit):
        if unit == "participant":
            groups = sorted({by_id[x]["participant_id"] for x in pool})
            if len(groups) < folds:
                raise ValueError(f"{where}: {folds} folds require at least {folds} training participants")
            assignment = KFold(folds, shuffle=True, random_state=seed % (2 ** 32)).split(groups)
            result = []
            for i, (a, b) in enumerate(assignment):
                train_people, valid_people = set(np.asarray(groups)[a]), set(np.asarray(groups)[b])
                train = [x for x in pool if by_id[x]["participant_id"] in train_people]
                valid = [x for x in pool if by_id[x]["participant_id"] in valid_people]
                result.append({"train_sessions": sample(train, seed, f"{where}-inner-{i}"),
                               "validation_sessions": sorted(valid)})
        else:
            classes = [by_id[x]["label"] for x in pool]
            for name, code in labels.items():
                if classes.count(code) < folds:
                    raise ValueError(f"{where}: {folds} session folds need {folds} sessions of {name}")
            assignment = StratifiedKFold(folds, shuffle=True,
                                         random_state=seed % (2 ** 32)).split(pool, classes)
            result = [{"train_sessions": sample([pool[j] for j in a], seed, f"{where}-inner-{i}"),
                       "validation_sessions": sorted(pool[j] for j in b)}
                      for i, (a, b) in enumerate(assignment)]
        for fold in result:
            if not fold["validation_sessions"] or set(fold["train_sessions"]) & set(fold["validation_sessions"]):
                raise ValueError(f"Invalid inner fold in {where}")
        return result

    def session_split(seed):
        """Seeded session assignment minimizing per-participant window-ratio error."""
        wanted = 1 - config["session_train_test_split_ratio"]
        windows = config["windows_seconds"]
        by_person = defaultdict(list)
        for sid in sessions:
            by_person[by_id[sid]["participant_id"]].append(sid)
        candidates = []
        for attempt in range(128):
            test = set()
            error = 0.0
            for person, ids in sorted(by_person.items()):
                rng = np.random.default_rng(_seed(seed, "outer", attempt, person))
                ordered = rng.permutation(sorted(ids)).tolist()
                total = np.array([sum(by_id[x]["window_counts"][str(w)] for x in ids)
                                  for w in windows], dtype=float)
                selected = np.zeros(len(windows), dtype=float)
                target = total * wanted
                for sid in ordered:
                    size = np.array([by_id[sid]["window_counts"][str(w)] for w in windows], dtype=float)
                    if np.sum(((selected + size - target) / total) ** 2) < np.sum(((selected - target) / total) ** 2):
                        test.add(sid)
                        selected += size
                error += float(np.sum(((selected - target) / total) ** 2))
            train = sorted(set(sessions) - test)
            if not train or not test:
                continue
            try:
                require_classes(train, "session split")
                inner(train, seed, "session split", "session")
            except ValueError:
                continue
            candidates.append((error, sorted(test), train))
        if not candidates:
            raise ValueError("No viable session train/test split with class coverage and inner folds")
        _, test, train = min(candidates)
        return train, test

    strategy = config["evaluation_strategy"]
    if strategy == "manual_participant_group":
        groups = config["manual_participant_group"]
        unknown = (set(groups["train"]) | set(groups["test"])) - set(people)
        if unknown:
            raise ValueError(f"Unknown manual participants: {sorted(unknown)}")
    outer = []
    for seed in config["seeds"]:
        if strategy == "participant_lopo":
            assignments = [(person, [x for x in sessions if by_id[x]["participant_id"] != person],
                            [x for x in sessions if by_id[x]["participant_id"] == person], "participant")
                           for person in people]
        elif strategy == "manual_participant_group":
            groups = config["manual_participant_group"]
            assignments = [("manual", [x for x in sessions if by_id[x]["participant_id"] in groups["train"]],
                            [x for x in sessions if by_id[x]["participant_id"] in groups["test"]], "participant")]
        else:
            train, test = session_split(seed)
            assignments = [("session-split", train, test, "session")]
        for name, train, test, unit in assignments:
            if not train or not test or set(train) & set(test):
                raise ValueError(f"Invalid or empty outer split: {name}, seed {seed}")
            require_classes(train, f"outer {name}")
            where = f"{name}-seed-{seed}"
            outer.append({"id": name, "seed": seed, "train_pool_sessions": sorted(train),
                          "train_sessions": sample(train, seed, f"{where}-refit"),
                          "test_sessions": sorted(test), "inner": inner(train, seed, where, unit)})
    return {"schema_version": 1, "evaluation_strategy": strategy,
            "outer": outer, "participants": people}


def _snapshot_file(root, relative, sha256, run):
    source = root / relative
    suffix = ".json.gz" if source.name.endswith(".json.gz") else ".json"
    destination = run / "snapshot" / "sha256" / f"{sha256}{suffix}"
    if destination.exists():
        if file_hash(destination) != sha256:
            raise ValueError(f"Corrupt snapshot file: {destination}")
    else:
        atomic_write(destination, lambda temp: shutil.copyfile(source, temp))
        if file_hash(destination) != sha256:
            raise ValueError(f"Source changed while copying snapshot: {source}")
    return destination.relative_to(run).as_posix()


def prepare(root, run, config, threads, log):
    root, run = Path(root).resolve(), Path(run).resolve()
    inputs = inventory(root)
    identity = {"protocol": PROTOCOL, "configuration": config, "threads": threads,
                "inputs": inputs, "code": code_fingerprint(), "environment": environment()}
    identity_path = run / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Input, config, code, or dependencies changed; use a new run ID")
    if (run / "prepared.json").exists():
        verify_prepared(run)
        log.event("prepared snapshot cache hit", output=str(run))
        return
    write_json(identity_path, identity)
    write_json(run / "experiment.json", config)
    write_json(run / "inputs.json", inputs)
    records, feature_tables, seen = [], {}, set()
    windows = config["windows_seconds"]
    for i, item in enumerate(inputs, 1):
        log.task("ingest and extract", current_session=item["path"])
        path = root / item["path"]
        record, frame, reason = load_session(path, config["labels"], max(windows))
        record.update(path=item["path"], exclusion=reason)
        if record["session_id"] in seen:
            raise ValueError(f"Duplicate session ID: {record['session_id']}")
        seen.add(record["session_id"])
        if frame is not None:
            extracted = {w: extract(frame, w, config["feature_names"])[0] for w in windows}
            record["window_counts"] = {str(w): len(extracted[w]) for w in windows}
            if not all(record["window_counts"].values()):
                record["exclusion"] = "no_valid_window_at_every_size"
            else:
                feature_tables[record["session_id"]] = extracted
        records.append(record)
        log.event("session scanned", completed=i, total=len(inputs), exclusion=record["exclusion"])
    initial = [r for r in records if not r["exclusion"]]
    totals = defaultdict(Counter)
    for record in initial:
        totals[record["participant_id"]].update(record["window_counts"])
    minimum = config["minimum_window_count_per_participant"]
    excluded_people = {person for person, counts in totals.items()
                       if any(counts[str(w)] < minimum for w in windows)}
    for record in initial:
        if record["participant_id"] in excluded_people:
            record["exclusion"] = "participant_below_minimum_windows"
    cohort = [r for r in records if not r["exclusion"]]
    if not cohort:
        raise ValueError("No eligible Sydney sessions remain after quality and participant filters")
    coverage = {person: {name: sum(r["participant_id"] == person and r["label"] == code for r in cohort)
                         for name, code in config["labels"].items()}
                for person in sorted({r["participant_id"] for r in cohort})}
    write_json(run / "cohort.json", {"sessions": cohort,
               "exclusions": [r for r in records if r["exclusion"]],
               "participant_class_coverage": coverage,
               "class_counts": dict(Counter(r["vehicle_type"] for r in cohort))})
    splits = make_splits(cohort, config)
    write_json(run / "splits.json", splits)
    snapshot = []
    for record in cohort:
        item = next(x for x in inputs if x["path"] == record["path"])
        files = {item["path"]: item["sha256"], **item["sidecars"]}
        snapshot.append({"session_id": record["session_id"], "participant_id": record["participant_id"],
                         "files": [{"original": relative, "sha256": sha,
                                    "snapshot": _snapshot_file(root, relative, sha, run)}
                                   for relative, sha in files.items()]})
    write_json(run / "snapshot.json", snapshot)
    for w in windows:
        frames = []
        for record in cohort:
            frame = feature_tables[record["session_id"]][w].copy()
            frame["session_id"] = record["session_id"]
            frame["participant_id"] = record["participant_id"]
            frame["label"] = record["label"]
            frames.append(frame)
        atomic_write(run / f"features-{w}.parquet", lambda p: pd.concat(frames, ignore_index=True).to_parquet(p, index=False))
    names = ["experiment.json", "inputs.json", "cohort.json", "splits.json", "snapshot.json",
             *[f"features-{w}.parquet" for w in windows],
             *[file["snapshot"] for item in snapshot for file in item["files"]]]
    write_json(run / "prepared.json", {"identity": digest(identity),
                                      "files": {name: file_hash(run / name) for name in sorted(set(names))}})
    log.event("preparation complete", participants=len(coverage), sessions=len(cohort), output=str(run))
