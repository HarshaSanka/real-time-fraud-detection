"""Two-threshold policy search with declared capacity constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.evaluation.cost import PolicyResult, evaluate_policy
from fraud_detection.utils.config import RiskConfig


def optimize_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    config: RiskConfig,
    grid_size: int = 25,
) -> tuple[PolicyResult, pd.DataFrame]:
    quantiles = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, grid_size),
                np.quantile(probabilities, np.linspace(0.50, 0.999, grid_size)),
                np.array([config.fallback_review_threshold, config.fallback_block_threshold]),
            ]
        )
    )
    results: list[PolicyResult] = []
    for review_threshold in quantiles:
        for block_threshold in quantiles:
            if review_threshold >= block_threshold:
                continue
            results.append(
                evaluate_policy(
                    labels,
                    probabilities,
                    amounts,
                    float(review_threshold),
                    float(block_threshold),
                    config,
                )
            )
    feasible = [result for result in results if result.feasible]
    candidates = feasible or [
        result
        for result in results
        if result.review_rate <= config.maximum_review_rate
        and result.block_rate <= config.maximum_block_rate
    ]
    if not candidates:
        candidates = results
    best = min(
        candidates,
        key=lambda result: (
            result.expected_cost,
            result.review_rate,
            -result.fraud_dollar_capture,
        ),
    )
    return best, pd.DataFrame([result.as_dict() for result in results])
