from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from fraud_detection.contracts import FraudLabelEventV1
from fraud_detection.inference.predictor import Predictor
from fraud_detection.streaming.labels import apply_label, run_label_consumer
from fraud_detection.streaming.outbox import publish_pending
from fraud_detection.streaming.producer import row_to_live_event
from fraud_detection.training.retraining import create_retraining_request
from fraud_detection.utils.config import load_settings


def test_row_to_live_event_and_reason_codes() -> None:
    timestamp = datetime.now(UTC)
    event = row_to_live_event(
        {
            "transaction_id": "txn_ops",
            "customer_id": "cust_ops",
            "merchant_id": "merchant_ops",
            "amount": 999.0,
            "merchant_category": "shopping_net",
            "merchant_latitude": 51.5,
            "merchant_longitude": -0.12,
            "customer_home_latitude": 41.88,
            "customer_home_longitude": -87.63,
            "customer_state": "IL",
            "city_population": 100_000,
        },
        timestamp,
    )
    assert event.timestamp == timestamp
    assert event.amount == 999.0
    reasons = Predictor.reason_codes(
        {
            "amount_vs_customer_avg_30d": 8,
            "customer_transactions_30m": 9,
            "travel_speed_kmh": 1000,
            "merchant_confirmed_fraud_rate": 0.2,
            "customer_confirmed_fraud_rate": 0.3,
            "history_unavailable": 1,
        }
    )
    assert len(reasons) == 4
    assert "CONFIRMED_CUSTOMER_HISTORY" in reasons


def test_delayed_label_updates_persistence_and_online_history() -> None:
    label = FraudLabelEventV1(
        transaction_id="txn_label",
        is_fraud=True,
        observed_at=datetime.now(UTC),
        source="chargeback",
    )
    prediction = SimpleNamespace(customer_id="cust_label", merchant_id="merchant_label")

    class Store:
        def save_label(self, received: FraudLabelEventV1) -> Any:
            assert received == label
            return prediction

    calls: list[tuple[object, ...]] = []

    class Features:
        def apply_label(self, *args: object) -> None:
            calls.append(args)

    apply_label(label, Store(), Features())  # type: ignore[arg-type]
    assert calls == [("txn_label", "cust_label", "merchant_label", True)]


def test_retraining_request_requires_human_approval(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("fraud_detection.training.retraining.project_root", lambda: tmp_path)
    output = create_retraining_request("concept_drift", {"pr_auc_change": -0.1}, "challenger-v2")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "awaiting_human_approval"
    assert payload["automatic_promotion"] is False


def test_outbox_publishes_then_marks_transaction(monkeypatch) -> None:
    produced: list[tuple[str, bytes, bytes]] = []
    marked: list[int] = []
    event = SimpleNamespace(
        id=7,
        topic="fraud_predictions.v1",
        message_key="txn_outbox",
        payload_json='{"transaction_id":"txn_outbox"}',
    )

    class FakeStore:
        def __init__(self, _url: str) -> None:
            pass

        def pending_outbox(self) -> list[Any]:
            return [event]

        def mark_published(self, event_id: int) -> None:
            marked.append(event_id)

    class FakeProducer:
        def __init__(self, _config: dict[str, object]) -> None:
            pass

        def produce(self, topic: str, *, key: bytes, value: bytes) -> None:
            produced.append((topic, key, value))

        def flush(self, _timeout: float) -> None:
            pass

    monkeypatch.setattr("fraud_detection.streaming.outbox.PredictionStore", FakeStore)
    monkeypatch.setitem(sys.modules, "confluent_kafka", SimpleNamespace(Producer=FakeProducer))
    assert publish_pending(load_settings("ci"), once=True) == 1
    assert marked == [7]
    assert produced[0][0] == "fraud_predictions.v1"


def test_label_consumer_commits_valid_and_dlqs_invalid(monkeypatch) -> None:
    label = FraudLabelEventV1(
        transaction_id="txn_label_consumer",
        is_fraud=True,
        observed_at=datetime.now(UTC),
        source="simulation",
    )
    callbacks: list[Any] = []
    commits: list[bytes] = []
    dlq_values: list[bytes] = []
    applied: list[tuple[object, ...]] = []

    class Message:
        def __init__(self, value: bytes) -> None:
            self._value = value

        def error(self) -> None:
            return None

        def value(self) -> bytes:
            return self._value

        def key(self) -> bytes:
            return b"txn_label_consumer"

    messages = [Message(label.model_dump_json().encode()), Message(b"not-json")]

    class FakeConsumer:
        def __init__(self, _config: dict[str, object]) -> None:
            pass

        def subscribe(self, _topics: list[str]) -> None:
            pass

        def poll(self, _timeout: float) -> Message | None:
            if messages:
                return messages.pop(0)
            callbacks[0](0, None)
            return None

        def commit(self, *, message: Message, asynchronous: bool) -> None:
            assert asynchronous is False
            commits.append(message.value())

        def close(self) -> None:
            commits.append(b"closed")

    class FakeProducer:
        def __init__(self, _config: dict[str, object]) -> None:
            pass

        def produce(self, _topic: str, *, key: bytes, value: bytes) -> None:
            dlq_values.append(value)

        def flush(self, _timeout: float) -> None:
            pass

    class FakeStore:
        def __init__(self, _url: str) -> None:
            pass

        def save_label(self, _label: FraudLabelEventV1) -> Any:
            return SimpleNamespace(customer_id="cust_label", merchant_id="merchant_label")

    class FakeFeatures:
        def __init__(self, _url: str, _delay: int) -> None:
            pass

        def apply_label(self, *args: object) -> None:
            applied.append(args)

    monkeypatch.setattr("fraud_detection.streaming.labels.PredictionStore", FakeStore)
    monkeypatch.setattr("fraud_detection.streaming.labels.RedisFeatureStore", FakeFeatures)
    monkeypatch.setattr(
        "fraud_detection.streaming.labels.signal.signal",
        lambda _signal, callback: callbacks.append(callback),
    )
    monkeypatch.setitem(
        sys.modules,
        "confluent_kafka",
        SimpleNamespace(Consumer=FakeConsumer, Producer=FakeProducer),
    )
    run_label_consumer(load_settings("ci"))
    assert applied == [("txn_label_consumer", "cust_label", "merchant_label", True)]
    assert len(dlq_values) == 1
    assert b"not-json" not in dlq_values[0]
    assert commits[-1] == b"closed"
