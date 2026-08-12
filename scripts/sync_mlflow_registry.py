"""Persist native candidate models and register the calibrated-bundle champion evidence."""

from __future__ import annotations

import json
import os

import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import pandas as pd
from scipy.sparse import vstack

from fraud_detection.models.factory import build_model
from fraud_detection.training.data import build_encoder, load_splits, xy
from fraud_detection.utils.config import load_settings, project_root


def main() -> None:
    root = project_root()
    settings = load_settings("portfolio")
    mlflow.set_tracking_uri(os.getenv("FRAUD_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("real-time-fraud-detection")
    summary_path = root / "reports/benchmark_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    splits = load_splits(settings)
    encoder = build_encoder()
    train_x, train_y = xy(splits.train)
    tune_x, tune_y = xy(splits.tune)
    encoder.fit(pd.concat([train_x, tune_x]))
    matrix = vstack([encoder.transform(train_x), encoder.transform(tune_x)])
    labels = np.concatenate([train_y.to_numpy(), tune_y.to_numpy()])
    native_runs: dict[str, str] = {}
    for name in ["xgboost", "lightgbm", "catboost"]:
        tuning_path = root / f"reports/tuning/{name}.json"
        parameters = (
            json.loads(tuning_path.read_text(encoding="utf-8"))["best_parameters"]
            if tuning_path.exists()
            else {}
        )
        model = build_model(name, settings, parameters)
        model.fit(matrix, labels)
        with mlflow.start_run(
            run_name=f"{name}_native_full_refit",
            tags={
                "dataset": "sparkov",
                "profile": "portfolio",
                "status": "candidate",
                "purpose": "native model artifact for measured benchmark",
            },
        ) as run:
            result = summary["results"][name]
            mlflow.log_params({key: str(value)[:500] for key, value in model.get_params().items()})
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in result.items()
                    if isinstance(value, (float, int)) and not isinstance(value, bool)
                }
            )
            mlflow.log_artifact(str(root / f"reports/threshold_grid_{name}.csv"))
            if tuning_path.exists():
                mlflow.log_artifact(str(tuning_path))
            if name == "xgboost":
                mlflow.xgboost.log_model(model, name="model")
            elif name == "lightgbm":
                mlflow.lightgbm.log_model(model, name="model")
            else:
                mlflow.catboost.log_model(model, name="model")
            native_runs[name] = run.info.run_id
            summary["results"][name]["mlflow_run_id"] = run.info.run_id

    client = mlflow.MlflowClient()
    experiment = mlflow.get_experiment_by_name("real-time-fraud-detection")
    if experiment is None:
        raise RuntimeError("MLflow experiment was not created")
    logged = client.search_logged_models(experiment_ids=[experiment.experiment_id])
    champion_run = summary["results"][summary["champion"]]["mlflow_run_id"]
    champion_logged = next(
        model
        for model in logged
        if model.source_run_id == champion_run and model.status.name == "READY"
    )
    model_version = mlflow.register_model(champion_logged.model_uri, "fraud-detection")
    client.set_registered_model_alias("fraud-detection", "champion", model_version.version)
    for path in (root / "artifacts/model/current").iterdir():
        client.log_artifact(champion_run, str(path), artifact_path="calibrated_bundle")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    registry_path = root / "reports/mlflow_champion_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry.update(
        {
            "registered_model": "fraud-detection",
            "registered_version": model_version.version,
            "alias": "champion",
            "champion_mlflow_run_id": champion_run,
            "native_candidate_run_ids": native_runs,
        }
    )
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
