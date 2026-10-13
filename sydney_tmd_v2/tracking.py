"""Live MLflow parent/child tracking; artifacts remain available locally too."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from .core import read_json, write_json


class Tracking:
    def __init__(self, uri, experiment, name, run_dir):
        self.uri, self.experiment, self.name = uri, experiment, name
        self.run_dir = Path(run_dir)

    def __enter__(self):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self.experiment)
        marker = self.run_dir / "mlflow.json"
        if marker.exists():
            existing = read_json(marker)
            if existing["tracking_uri"] != self.uri or existing["experiment"] != self.experiment:
                raise ValueError("MLflow destination changed for this run")
            self.parent = mlflow.start_run(run_id=existing["run_id"])
        else:
            self.parent = mlflow.start_run(run_name=self.name)
            write_json(marker, {"run_id": self.parent.info.run_id, "tracking_uri": self.uri,
                                "experiment": self.experiment})
        mlflow.set_tag("study.kind", "experiment")
        return self

    def __exit__(self, kind, value, traceback):
        self.mlflow.end_run(status="FAILED" if kind else "FINISHED")

    @contextmanager
    def child(self, name, kind):
        marker = self.run_dir / "mlflow-children.json"
        children = read_json(marker) if marker.exists() else {}
        key = f"{kind}:{name}"
        options = ({"run_id": children[key]} if key in children else {"run_name": name})
        with self.mlflow.start_run(nested=True, **options) as run:
            if key not in children:
                children[key] = run.info.run_id
                write_json(marker, children)
            self.mlflow.set_tag("study.kind", kind)
            yield self

    def params(self, values):
        self.mlflow.log_params({str(k): str(v) for k, v in values.items()})

    def metrics(self, values, step=None):
        clean = {k: float(v) for k, v in values.items() if v is not None}
        if clean:
            self.mlflow.log_metrics(clean, step=step)

    def file(self, path, artifact_path=None):
        self.mlflow.log_artifact(str(Path(path)), artifact_path=artifact_path)


class NullTracking:
    """Local execution adapter for direct stage tests; the Compose CLI uses MLflow."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @contextmanager
    def child(self, *args):
        yield self

    def params(self, values):
        pass

    def metrics(self, values, step=None):
        pass

    def file(self, path, artifact_path=None):
        pass
