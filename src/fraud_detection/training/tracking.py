"""MLflow experiment and registry integration."""

from __future__ import annotations

import json
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
            safe_parameters = {
                key: json.dumps(value, default=str)[:500]
                if isinstance(value, (dict, list, tuple))
                else str(value)[:500]
                for key, value in parameters.items()
            }
            mlflow.log_params(safe_parameters)
            mlflow.log_metrics(metrics)
            for artifact in artifacts:
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))
            try:
                mlflow.sklearn.log_model(model, name="model")
            except Exception as error:
                mlflow.set_tag("model_artifact_error", str(error)[:500])
            return str(run.info.run_id)
    except Exception:
        return None
