"""Deterministic model construction."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fraud_detection.utils.config import Settings


def build_model(name: str, settings: Settings, parameters: dict[str, Any] | None = None) -> Any:
    seed = int(settings.project["seed"])
    threads = int(settings.execution.get("max_threads", 4))
    overrides = parameters or {}
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced", max_iter=500, solver="liblinear", random_state=seed
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=settings.model.random_forest_estimators,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=threads,
            random_state=seed,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        defaults = dict(
            n_estimators=350,
            max_depth=7,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            n_jobs=threads,
            random_state=seed,
            scale_pos_weight=100.0,
        )
        defaults.update(overrides)
        return XGBClassifier(**defaults)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        defaults = dict(
            n_estimators=350,
            num_leaves=63,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            reg_lambda=2.0,
            n_jobs=threads,
            random_state=seed,
            verbosity=-1,
        )
        defaults.update(overrides)
        return LGBMClassifier(**defaults)  # type: ignore[arg-type]
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=350,
            depth=8,
            learning_rate=0.07,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            random_seed=seed,
            thread_count=threads,
            verbose=False,
        )
    raise ValueError(f"unknown model: {name}")


MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]
