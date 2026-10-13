"""Experiment CLI used by the Compose study service and local development."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import secrets
import sys

from .config import load_config
from .core import read_json, run_lock
from .data import prepare
from .progress import Attempt
from .tracking import NullTracking, Tracking


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sydney-only Docker/MLflow transport-mode study")
    parser.add_argument("command", choices=("prepare", "evaluate", "report", "all"))
    parser.add_argument("--input", type=Path, help="Downloaded Sydney sessions (prepare/all)")
    parser.add_argument("--config", type=Path, help="Experiment JSON; defaults to project experiment.json on prepare/all")
    parser.add_argument("--run-id", help="Existing ID to resume; default UTC start plus 8 hex digits")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--threads", type=int, help="Model CPU threads; defaults to 1 on prepare/all")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--experiment", default="Sydney TMD V2")
    parser.add_argument("--no-mlflow", action="store_true", help="Local test execution only")
    args = parser.parse_args(argv)
    if args.command in ("prepare", "all") and args.input is None:
        parser.error("--input is required for prepare/all")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    name = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)
    if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", name):
        parser.error("Run ID must be a UTC timestamp followed by eight lowercase hex characters")
    run = args.runs_dir.resolve() / name
    if args.input:
        source = args.input.resolve()
        if not source.is_dir():
            parser.error(f"Input directory does not exist: {source}")
        if run.is_relative_to(source) or args.logs_dir.resolve().is_relative_to(source):
            parser.error("Run and log directories must be outside the input tree")
    attempt = Attempt(args.logs_dir, name)
    tracking = (NullTracking() if args.no_mlflow else
                Tracking(args.tracking_uri, args.experiment, name, run))
    print(f"Run: {run}", flush=True)
    try:
        with run_lock(run), tracking as logged:
            if args.command in ("prepare", "all"):
                config = load_config(args.config or Path("experiment.json"))
                with attempt.stage("preparation") as log:
                    prepare(args.input, run, config, args.threads or 1, log)
                    logged.params({"evaluation_strategy": config["evaluation_strategy"],
                                   "selection_metric": config["selection_metric"],
                                   "windows_seconds": config["windows_seconds"],
                                   "model_families": config["model_families"]})
                    for name in ("experiment.json", "inputs.json", "cohort.json", "splits.json",
                                 "snapshot.json", "prepared.json"):
                        logged.file(run / name, "preparation")
            else:
                saved = read_json(run / "identity.json")
                if args.config is not None and load_config(args.config) != saved["configuration"]:
                    raise ValueError("Config changed; use a new run ID")
                if args.threads is not None and args.threads != saved["threads"]:
                    raise ValueError("Threads changed; use a new run ID")
            if args.command in ("evaluate", "all"):
                from .training import evaluate
                with attempt.stage("evaluation") as log:
                    evaluate(run, log, logged)
            if args.command in ("report", "all"):
                from .reporting import report
                with attempt.stage("reporting") as log:
                    report(run, log, logged)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}. Run state: {run}", file=sys.stderr, flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
