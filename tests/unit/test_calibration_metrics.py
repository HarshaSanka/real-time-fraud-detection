import numpy as np
import pytest

from fraud_detection.evaluation.metrics import classification_metrics
from fraud_detection.models.calibration import ProbabilityCalibrator, expected_calibration_error


def test_isotonic_and_platt_calibration() -> None:
    probabilities = np.linspace(0.001, 0.999, 500)
    labels = (probabilities > 0.8).astype(int)
    isotonic = ProbabilityCalibrator(minimum_positives=50).fit(probabilities, labels)
    assert isotonic.method == "isotonic"
    assert np.all((isotonic.predict(probabilities) >= 0) & (isotonic.predict(probabilities) <= 1))
    platt = ProbabilityCalibrator(minimum_positives=200).fit(probabilities, labels)
    assert platt.method == "platt"


def test_metrics_emphasize_pr_auc_and_calibration() -> None:
    labels = np.array([0, 0, 0, 1])
    probabilities = np.array([0.01, 0.1, 0.2, 0.9])
    result = classification_metrics(labels, probabilities, 0.5)
    assert result["pr_auc"] == 1.0
    assert result["recall"] == 1.0
    assert expected_calibration_error(labels, labels.astype(float)) == pytest.approx(0)
