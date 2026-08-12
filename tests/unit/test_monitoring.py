import numpy as np
import pandas as pd

from fraud_detection.monitoring.drift import categorical_js_divergence, population_stability_index
from fraud_detection.monitoring.performance import labeled_performance
from fraud_detection.utils.config import load_settings


def test_drift_statistics() -> None:
    reference = np.arange(1000)
    assert population_stability_index(reference, reference.copy()) == 0
    assert population_stability_index(reference, reference + 1000) > 0.2
    assert categorical_js_divergence(pd.Series(["a", "a"]), pd.Series(["a", "a"])) == 0
    assert categorical_js_divergence(pd.Series(["a"]), pd.Series(["b"])) > 0.5


def test_delayed_performance_requires_support() -> None:
    frame = pd.DataFrame({"is_fraud": [0, 1], "fraud_probability": [0.1, 0.9]})
    result = labeled_performance(frame, 0.5, load_settings("ci").monitoring)
    assert result["status"] == "insufficient_labels"
