"""Transparent PSI and Jensen-Shannon drift calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=edges)[0] / max(len(reference), 1)
    actual = np.histogram(current, bins=edges)[0] / max(len(current), 1)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def categorical_js_divergence(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.dropna()) | set(current.dropna()))
    expected = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    actual = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return float(jensenshannon(expected, actual, base=2) ** 2)
