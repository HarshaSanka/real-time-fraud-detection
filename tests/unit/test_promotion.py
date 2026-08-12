from fraud_detection.training.promotion import evaluate_promotion


def test_promotion_requires_business_and_model_improvement() -> None:
    baseline = {"pr_auc": 0.5, "expected_cost": 100.0}
    good = {
        "pr_auc": 0.6,
        "expected_cost": 90.0,
        "feasible": True,
        "expected_calibration_error": 0.01,
    }
    assert evaluate_promotion(good, baseline).accepted
    bad = {
        "pr_auc": 0.6,
        "expected_cost": 90.0,
        "feasible": False,
        "expected_calibration_error": 0.01,
    }
    assert not evaluate_promotion(bad, baseline).accepted


def test_promotion_does_not_require_or_consume_test_calibration_metric() -> None:
    baseline = {"pr_auc": 0.5, "expected_cost": 100.0}
    candidate = {"pr_auc": 0.6, "expected_cost": 90.0, "feasible": True}
    assert evaluate_promotion(candidate, baseline).accepted
