"""Delayed-label ingestion joins outcomes back to original predictions."""

from __future__ import annotations

from fraud_detection.contracts import FraudLabelEventV1
from fraud_detection.inference.feature_store import OnlineFeatureStore
from fraud_detection.storage.database import PredictionStore


def apply_label(
    label: FraudLabelEventV1,
    store: PredictionStore,
    feature_store: OnlineFeatureStore,
) -> None:
    prediction = store.save_label(label)
    feature_store.apply_label(
        label.transaction_id,
        prediction.customer_id,
        prediction.merchant_id,
        label.is_fraud,
    )
