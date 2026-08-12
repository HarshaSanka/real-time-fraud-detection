"""Retry-safe in-memory and Redis online feature stores."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Any, Protocol, cast

from fraud_detection.contracts import TransactionEventV1
from fraud_detection.exceptions import FeatureStoreUnavailableError
from fraud_detection.features.engine import StreamingFeatureEngine


@dataclass(frozen=True)
class FeatureResult:
    features: dict[str, float]
    degraded: bool = False


class OnlineFeatureStore(Protocol):
    def get_and_update(self, event: TransactionEventV1) -> FeatureResult: ...
    def apply_label(
        self, transaction_id: str, customer_id: str, merchant_id: str, is_fraud: bool
    ) -> None: ...


def event_payload(event: TransactionEventV1) -> dict[str, Any]:
    return {
        "transaction_id": event.transaction_id,
        "customer_id": event.customer_id,
        "merchant_id": event.merchant_id,
        "event_timestamp": event.timestamp.replace(tzinfo=None),
        "amount": event.amount,
        "merchant_category": event.merchant_category.value,
        "customer_state": event.customer_state or "NA",
        "customer_home_latitude": event.customer_home_latitude,
        "customer_home_longitude": event.customer_home_longitude,
        "merchant_latitude": event.merchant_latitude,
        "merchant_longitude": event.merchant_longitude,
        "city_population": event.city_population or 0,
    }


class InMemoryFeatureStore:
    def __init__(self, label_delay_days: int = 7, degraded: bool = False) -> None:
        self.engine = StreamingFeatureEngine(label_delay_days)
        self.snapshots: dict[str, dict[str, float]] = {}
        self.degraded = degraded

    def get_and_update(self, event: TransactionEventV1) -> FeatureResult:
        if event.transaction_id not in self.snapshots:
            self.snapshots[event.transaction_id] = self.engine.process(event_payload(event))
        features = self.snapshots[event.transaction_id]
        if self.degraded:
            features = {**features, "history_unavailable": 1.0}
        return FeatureResult(features=features, degraded=self.degraded)

    def apply_label(
        self, transaction_id: str, customer_id: str, merchant_id: str, is_fraud: bool
    ) -> None:
        self.engine.apply_observed_label(customer_id, merchant_id, int(is_fraud))


class RedisFeatureStore:
    """Reference Redis implementation with atomic lock and idempotent snapshots.

    The single-node demo serializes the state engine for exact offline parity. A scaled
    deployment would partition the same state by customer and execute its update in Lua/Flink.
    """

    def __init__(self, redis_url: str, label_delay_days: int = 7) -> None:
        import redis

        self.client = redis.Redis.from_url(redis_url, socket_timeout=1, decode_responses=False)
        self.label_delay_days = label_delay_days

    def get_and_update(self, event: TransactionEventV1) -> FeatureResult:
        snapshot_key = f"fraud:snapshot:{event.transaction_id}"
        try:
            existing = self.client.get(snapshot_key)
            if existing:
                return FeatureResult(features=json.loads(cast(bytes, existing)))
            with self.client.lock("fraud:feature-engine:lock", timeout=10, blocking_timeout=2):
                existing = self.client.get(snapshot_key)
                if existing:
                    return FeatureResult(features=json.loads(cast(bytes, existing)))
                payload = self.client.get("fraud:feature-engine:v1")
                engine = (
                    pickle.loads(cast(bytes, payload))
                    if payload
                    else StreamingFeatureEngine(self.label_delay_days)
                )
                features = engine.process(event_payload(event))
                pipeline = self.client.pipeline(transaction=True)
                pipeline.set("fraud:feature-engine:v1", pickle.dumps(engine))
                pipeline.setex(snapshot_key, 30 * 24 * 3600, json.dumps(features))
                pipeline.execute()
                return FeatureResult(features=features)
        except Exception as error:
            raise FeatureStoreUnavailableError("Redis online history is unavailable") from error

    def apply_label(
        self, transaction_id: str, customer_id: str, merchant_id: str, is_fraud: bool
    ) -> None:
        try:
            with self.client.lock("fraud:feature-engine:lock", timeout=10, blocking_timeout=2):
                payload = self.client.get("fraud:feature-engine:v1")
                engine = (
                    pickle.loads(cast(bytes, payload))
                    if payload
                    else StreamingFeatureEngine(self.label_delay_days)
                )
                engine.apply_observed_label(customer_id, merchant_id, int(is_fraud))
                self.client.set("fraud:feature-engine:v1", pickle.dumps(engine))
        except Exception as error:
            raise FeatureStoreUnavailableError("Redis label update failed") from error


def degraded_features(event: TransactionEventV1) -> dict[str, float]:
    engine = StreamingFeatureEngine()
    features = engine.process(event_payload(event))
    features["history_unavailable"] = 1.0
    return features
