"""Shared synchronous scoring service used by HTTP and Kafka consumers."""

from __future__ import annotations

import time

from prometheus_client import Counter, Histogram

from fraud_detection.contracts import ScoreResponseV1, TransactionEventV1
from fraud_detection.exceptions import FeatureStoreUnavailableError
from fraud_detection.inference.decision_engine import decide
from fraud_detection.inference.feature_store import (
    OnlineFeatureStore,
    degraded_features,
)
from fraud_detection.inference.predictor import Predictor
from fraud_detection.storage.database import PredictionStore, response_from_prediction

REQUESTS = Counter(
    "fraud_score_requests_total", "Scoring requests", ["source", "decision", "status"]
)
LATENCY = Histogram(
    "fraud_score_latency_seconds",
    "End-to-end scoring latency",
    ["source"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
SCORES = Histogram(
    "fraud_probability",
    "Calibrated fraud probability distribution",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9, 0.99, 1.0),
)


class ScoringService:
    def __init__(
        self,
        predictor: Predictor,
        feature_store: OnlineFeatureStore,
        prediction_store: PredictionStore,
    ) -> None:
        self.predictor = predictor
        self.feature_store = feature_store
        self.prediction_store = prediction_store

    def score(self, event: TransactionEventV1, source: str = "api") -> ScoreResponseV1:
        existing = self.prediction_store.get(event.transaction_id)
        if existing is not None:
            return response_from_prediction(existing, self.predictor.metadata.threshold_version)
        started = time.perf_counter()
        degraded = False
        try:
            feature_result = self.feature_store.get_and_update(event)
            features = feature_result.features
            degraded = feature_result.degraded
        except FeatureStoreUnavailableError:
            features = degraded_features(event)
            degraded = True
        probability = self.predictor.predict(event, features)
        risk_level, decision = decide(
            probability,
            self.predictor.metadata.review_threshold,
            self.predictor.metadata.block_threshold,
            degraded,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        response = ScoreResponseV1(
            transaction_id=event.transaction_id,
            fraud_probability=probability,
            risk_level=risk_level,
            decision=decision,
            model_version=self.predictor.metadata.model_version,
            threshold_version=self.predictor.metadata.threshold_version,
            latency_ms=latency_ms,
            reason_codes=(
                self.predictor.reason_codes(features) if decision.value != "APPROVE" else []
            ),
            degraded_mode=degraded,
        )
        self.prediction_store.save_prediction(event, response)
        REQUESTS.labels(source=source, decision=decision.value, status="ok").inc()
        LATENCY.labels(source=source).observe(latency_ms / 1000)
        SCORES.observe(probability)
        return response
