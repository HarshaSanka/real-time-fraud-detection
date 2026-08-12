"""Simulated and configurable three-way decision economics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraud_detection.utils.config import RiskConfig


@dataclass(frozen=True)
class PolicyResult:
    review_threshold: float
    block_threshold: float
    expected_cost: float
    review_rate: float
    block_rate: float
    fraud_dollar_capture: float
    fraud_dollars_captured: float
    feasible: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "review_threshold": self.review_threshold,
            "block_threshold": self.block_threshold,
            "expected_cost": self.expected_cost,
            "review_rate": self.review_rate,
            "block_rate": self.block_rate,
            "fraud_dollar_capture": self.fraud_dollar_capture,
            "fraud_dollars_captured": self.fraud_dollars_captured,
            "feasible": self.feasible,
        }


def evaluate_policy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    review_threshold: float,
    block_threshold: float,
    config: RiskConfig,
) -> PolicyResult:
    if review_threshold >= block_threshold:
        raise ValueError("review threshold must be below block threshold")
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    approve = probabilities < review_threshold
    review = (probabilities >= review_threshold) & (probabilities < block_threshold)
    block = probabilities >= block_threshold
    fraud = labels == 1
    legitimate = ~fraud
    approve_loss = amounts[approve & fraud].sum()
    review_loss = config.review_cost * review.sum() + (
        amounts[review & fraud].sum() * (1 - config.review_capture_rate)
    )
    block_loss = config.false_block_cost * (block & legitimate).sum()
    total_fraud_amount = amounts[fraud].sum()
    captured = amounts[block & fraud].sum() + (
        config.review_capture_rate * amounts[review & fraud].sum()
    )
    review_rate = float(review.mean())
    block_rate = float(block.mean())
    capture_rate = float(captured / max(total_fraud_amount, 1e-12))
    feasible = (
        review_rate <= config.maximum_review_rate
        and block_rate <= config.maximum_block_rate
        and capture_rate >= config.minimum_fraud_dollar_capture
    )
    return PolicyResult(
        review_threshold=float(review_threshold),
        block_threshold=float(block_threshold),
        expected_cost=float(approve_loss + review_loss + block_loss),
        review_rate=review_rate,
        block_rate=block_rate,
        fraud_dollar_capture=capture_rate,
        fraud_dollars_captured=float(captured),
        feasible=feasible,
    )
