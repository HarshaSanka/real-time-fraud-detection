"""Generate a reproducible data-drift report from feature windows."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from fraud_detection.features.pipeline import CATEGORICAL_FEATURES, FEATURE_NAMES
from fraud_detection.monitoring.drift import categorical_js_divergence, population_stability_index
from fraud_detection.utils.config import Settings, project_root

MATURING_STATE_FEATURES = {
    "customer_confirmed_fraud_rate",
    "merchant_confirmed_fraud_rate",
    "customer_cold_start",
    "merchant_cold_start",
}


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
    external_numeric = {
        key: value for key, value in numeric.items() if key not in MATURING_STATE_FEATURES
    }
    status = (
        "critical"
        if any(value >= settings.monitoring.psi_critical for value in external_numeric.values())
        else "warning"
        if any(value >= settings.monitoring.psi_warning for value in external_numeric.values())
        or any(value >= settings.monitoring.js_warning for value in categorical.values())
        else "stable"
    )
    report = {
        "status": status,
        "reference_window": "2019-01-01 through 2019-12-31",
        "current_window": "2020-07-01 through 2020-12-31",
        "numeric_psi": numeric,
        "external_numeric_psi": external_numeric,
        "maturing_state_psi": {
            key: value for key, value in numeric.items() if key in MATURING_STATE_FEATURES
        },
        "categorical_js_divergence": categorical,
        "concept_drift_note": "Concept drift is assessed only after delayed labels arrive; feature drift alone does not prove performance loss.",
        "state_maturation_note": "Confirmed-fraud and cold-start features accumulate history by design; they are reported but excluded from the external-drift status.",
    }
    reports = project_root() / "reports"
    if settings.profile != "portfolio":
        reports = reports / settings.profile
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
