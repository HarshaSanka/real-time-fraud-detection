import numpy as np
import pytest

from fraud_detection.contracts import Decision, RiskLevel
from fraud_detection.evaluation.cost import evaluate_policy
from fraud_detection.evaluation.thresholds import optimize_thresholds
from fraud_detection.inference.decision_engine import decide
from fraud_detection.utils.config import load_settings


def test_decision_boundaries_and_degraded_mode() -> None:
    assert decide(0.1, 0.3, 0.75) == (RiskLevel.LOW, Decision.APPROVE)
    assert decide(0.3, 0.3, 0.75) == (RiskLevel.MEDIUM, Decision.REVIEW)
    assert decide(0.75, 0.3, 0.75) == (RiskLevel.HIGH, Decision.BLOCK)
    assert decide(0.1, 0.3, 0.75, degraded_mode=True)[1] == Decision.REVIEW


def test_cost_evaluation_is_dollar_weighted() -> None:
    config = load_settings("ci").risk
    result = evaluate_policy(
        np.array([1, 1, 0]),
        np.array([0.1, 0.5, 0.9]),
        np.array([100.0, 100.0, 50.0]),
        0.3,
        0.75,
        config,
    )
    assert result.expected_cost == pytest.approx(100 + 5 + 20 + 25)
    assert result.fraud_dollars_captured == pytest.approx(80)


def test_threshold_optimizer_respects_order() -> None:
    labels = np.array([0] * 980 + [1] * 20)
    probabilities = np.linspace(0, 1, 1000)
    amounts = np.ones(1000) * 100
    result, grid = optimize_thresholds(labels, probabilities, amounts, load_settings("ci").risk, 12)
    assert result.review_threshold < result.block_threshold
    assert not grid.empty
