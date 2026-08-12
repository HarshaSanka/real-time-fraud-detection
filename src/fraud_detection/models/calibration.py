"""Chronological probability calibration with an explicit fallback."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, minimum_positives: int = 100) -> None:
        self.minimum_positives = minimum_positives
        self.method = "unfit"
        self.calibrator: IsotonicRegression | LogisticRegression | None = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
        values = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        targets = np.asarray(labels, dtype=int)
        if int(targets.sum()) >= self.minimum_positives and len(np.unique(values)) >= 10:
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(values, targets)
            self.method = "isotonic"
        else:
            model = LogisticRegression(solver="lbfgs")
            model.fit(logit(values).reshape(-1, 1), targets)
            self.calibrator = model
            self.method = "platt"
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("calibrator is not fitted")
        values = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            return np.asarray(self.calibrator.predict(values), dtype=float)
        return np.asarray(self.calibrator.predict_proba(logit(values).reshape(-1, 1))[:, 1])


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    boundaries = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lower, upper in pairwise(boundaries):
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1 else probabilities < upper
        )
        if mask.any():
            error += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(error)
