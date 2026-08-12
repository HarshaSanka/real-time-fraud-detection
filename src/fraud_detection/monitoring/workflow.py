"""Generate a reproducible data-drift report from feature windows."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from fraud_detection.features.pipeline import CATEGORICAL_FEATURES, FEATURE_NAMES
from fraud_detection.monitoring.drift import categorical_js_divergence, population_stability_index
from fraud_detection.utils.config import Settings, project_root


def run_drift_monitor(settings: Settings) -> dict[str, Any]:
    source = project_root() / "data/processed" / f"features_{settings.profile}.parquet"
    frame = pd.read_parquet(source)
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
    reference = frame[frame.event_timestamp < pd.Timestamp(settings.splits.tune_start)]
    current = frame[frame.event_timestamp >= pd.Timestamp(settings.splits.test_start)]
    numeric = {
        feature: population_stability_index(
            reference[feature].to_numpy(), current[feature].to_numpy()
        )
        for feature in FEATURE_NAMES
    }
    categorical = {
        feature: categorical_js_divergence(reference[feature], current[feature])
        for feature in CATEGORICAL_FEATURES
    }
    status = (
        "critical"
        if any(value >= settings.monitoring.psi_critical for value in numeric.values())
        else "warning"
        if any(value >= settings.monitoring.psi_warning for value in numeric.values())
        or any(value >= settings.monitoring.js_warning for value in categorical.values())
        else "stable"
    )
    report = {
        "status": status,
        "reference_window": "2019-01-01 through 2019-12-31",
        "current_window": "2020-07-01 through 2020-12-31",
        "numeric_psi": numeric,
        "categorical_js_divergence": categorical,
        "concept_drift_note": "Concept drift is assessed only after delayed labels arrive; feature drift alone does not prove performance loss.",
    }
    reports = project_root() / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
