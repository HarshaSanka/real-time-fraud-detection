"""MLflow experiment and registry integration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def log_run(
    model_name: str,
    model: Any,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path],
    tags: dict[str, str],
) -> str | None:
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(os.getenv("FRAUD_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("real-time-fraud-detection")
        with mlflow.start_run(run_name=model_name, tags=tags) as run:
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            for artifact in artifacts:
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))
            mlflow.sklearn.log_model(model, name="model")
            return str(run.info.run_id)
    except Exception:
        return None
