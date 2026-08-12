"""Optuna tuning on pre-test chronological data only."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import average_precision_score

from fraud_detection.models.factory import build_model
from fraud_detection.utils.config import Settings, project_root


def _stratified_sample(
    matrix: Any, labels: np.ndarray, maximum_rows: int, seed: int
) -> tuple[Any, np.ndarray]:
    if len(labels) <= maximum_rows:
        return matrix, labels
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    remaining = max(maximum_rows - len(positive), 0)
    chosen_negative = rng.choice(negative, min(remaining, len(negative)), replace=False)
    indices = rng.permutation(np.concatenate([positive, chosen_negative]))
    return matrix[indices], labels[indices]


def tune_models(
    train_x: Any,
    train_y: np.ndarray,
    tune_x: Any,
    tune_y: np.ndarray,
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    seed = int(settings.project["seed"])
    sampled_x, sampled_y = _stratified_sample(
        train_x, train_y, settings.model.tuning_sample_rows, seed
    )
    output: dict[str, dict[str, Any]] = {}
    reports = project_root() / "reports/tuning"
    reports.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for model_name in ["xgboost", "lightgbm"]:

        def objective(trial: optuna.Trial, selected_model: str = model_name) -> float:
            common = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 550),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            }
            if selected_model == "xgboost":
                common.update(
                    {
                        "max_depth": trial.suggest_int("max_depth", 4, 10),
                        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                        "scale_pos_weight": trial.suggest_float(
                            "scale_pos_weight", 25.0, 175.0, log=True
                        ),
                    }
                )
            else:
                common.update(
                    {
                        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                    }
                )
            model = build_model(selected_model, settings, common)
            model.fit(sampled_x, sampled_y)
            return float(average_precision_score(tune_y, model.predict_proba(tune_x)[:, 1]))

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=settings.model.tuning_trials)
        trials = [
            {"number": trial.number, "value": trial.value, "parameters": trial.params}
            for trial in study.trials
        ]
        result = {
            "model": model_name,
            "objective": "validation_pr_auc",
            "training_rows": len(sampled_y),
            "validation_rows": len(tune_y),
            "best_value": study.best_value,
            "best_parameters": study.best_params,
            "trials": trials,
            "test_data_used": False,
        }
        (reports / f"{model_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        output[model_name] = study.best_params
    return output
