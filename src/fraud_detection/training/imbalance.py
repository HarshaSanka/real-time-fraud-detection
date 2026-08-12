"""Training-only imbalance strategy comparison; no validation resampling."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from fraud_detection.utils.config import Settings, project_root


def _sample_indices(labels: np.ndarray, strategy: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if strategy == "undersampling":
        keep_negative = rng.choice(
            negative, size=min(len(negative), len(positive) * 20), replace=False
        )
        return rng.permutation(np.concatenate([positive, keep_negative]))
    if strategy == "oversampling":
        extra = rng.choice(
            positive,
            size=max(0, min(len(negative) // 10, len(positive) * 10) - len(positive)),
            replace=True,
        )
        return rng.permutation(np.concatenate([np.arange(len(labels)), extra]))
    return np.arange(len(labels))


def compare_imbalance_strategies(
    train_x: Any,
    train_y: np.ndarray,
    tune_x: Any,
    tune_y: np.ndarray,
    settings: Settings,
) -> list[dict[str, Any]]:
    seed = int(settings.project["seed"])
    if len(train_y) > settings.model.tuning_sample_rows:
        rng = np.random.default_rng(seed)
        positive = np.flatnonzero(train_y == 1)
        negative = np.flatnonzero(train_y == 0)
        remaining = max(settings.model.tuning_sample_rows - len(positive), 0)
        negative = rng.choice(negative, size=min(remaining, len(negative)), replace=False)
        base_indices = rng.permutation(np.concatenate([positive, negative]))
        train_x = train_x[base_indices]
        train_y = train_y[base_indices]
    results: list[dict[str, Any]] = []
    for strategy, class_weight in [
        ("unweighted", None),
        ("class_weight", "balanced"),
        ("undersampling", None),
        ("oversampling", None),
    ]:
        indices = _sample_indices(train_y, strategy, seed)
        model = LogisticRegression(
            class_weight=class_weight,
            max_iter=300,
            solver="liblinear",
            random_state=seed,
        )
        model.fit(train_x[indices], train_y[indices])
        probabilities = model.predict_proba(tune_x)[:, 1]
        results.append(
            {
                "strategy": strategy,
                "training_rows": len(indices),
                "training_fraud_rate": float(train_y[indices].mean()),
                "tune_pr_auc": float(average_precision_score(tune_y, probabilities)),
            }
        )
    reports = project_root() / "reports"
    if settings.profile != "portfolio":
        reports = reports / settings.profile
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "imbalance_comparison.json").write_text(
        json.dumps(
            {
                "results": results,
                "smote": {
                    "status": "excluded",
                    "reason": "Interpolating mixed categorical, geographic, and historical state can create impossible financial transactions.",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results
