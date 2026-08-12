"""Interpretable rules baseline returning a risk probability proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit


def rule_probabilities(frame: pd.DataFrame) -> np.ndarray:
    score = (
        -4.0
        + 0.65 * np.log1p(frame["amount_vs_customer_avg_30d"].clip(lower=0))
        + 0.20 * frame["customer_transactions_30m"].clip(upper=10)
        + 1.5 * frame["impossible_travel_flag"]
        + 80.0 * frame["merchant_confirmed_fraud_rate"]
    )
    return np.asarray(expit(score), dtype=float)
