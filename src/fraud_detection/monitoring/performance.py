"""Delayed-label concept performance with minimum-support guards."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fraud_detection.evaluation.metrics import classification_metrics
from fraud_detection.utils.config import MonitoringConfig


def labeled_performance(
    frame: pd.DataFrame,
    review_threshold: float,
    config: MonitoringConfig,
) -> dict[str, Any]:
    if (
        len(frame) < config.minimum_labels
        or int(frame["is_fraud"].sum()) < config.minimum_fraud_labels
    ):
        return {
            "status": "insufficient_labels",
            "labels": len(frame),
            "fraud_labels": int(frame["is_fraud"].sum()),
        }
    return {
        "status": "ready",
        **classification_metrics(
            frame["is_fraud"].to_numpy(),
            frame["fraud_probability"].to_numpy(),
            review_threshold,
        ),
    }
