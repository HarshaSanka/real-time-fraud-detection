"""Delayed-label ingestion joins outcomes back to original predictions."""

from __future__ import annotations

import signal

from sqlalchemy.exc import SQLAlchemyError

from fraud_detection.contracts import FraudLabelEventV1
from fraud_detection.exceptions import FeatureStoreUnavailableError
from fraud_detection.inference.feature_store import OnlineFeatureStore, RedisFeatureStore
from fraud_detection.storage.database import PredictionStore
from fraud_detection.streaming.serialization import dead_letter
from fraud_detection.streaming.topics import DEAD_LETTER_TOPIC, LABELS_TOPIC
from fraud_detection.utils.config import Settings


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


def run_label_consumer(settings: Settings) -> None:
    """Persist delayed labels and update retry-safe confirmed-history state."""
    from confluent_kafka import Consumer, Producer

    consumer = Consumer(
        {
            "bootstrap.servers": settings.serving.kafka_bootstrap_servers,
            "group.id": "fraud-labels-v1",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    producer = Producer(
        {"bootstrap.servers": settings.serving.kafka_bootstrap_servers, "enable.idempotence": True}
    )
    store = PredictionStore(settings.serving.database_url)
    feature_store = RedisFeatureStore(settings.serving.redis_url, settings.data.label_delay_days)
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    consumer.subscribe([LABELS_TOPIC])
    try:
        while running:
            message = consumer.poll(1.0)
            if message is None or message.error():
                continue
            payload = message.value()
            if payload is None:
                continue
            try:
                label = FraudLabelEventV1.model_validate_json(payload)
                apply_label(label, store, feature_store)
                consumer.commit(message=message, asynchronous=False)
            except (FeatureStoreUnavailableError, SQLAlchemyError, KeyError):
                # Leave the offset uncommitted and let the supervisor restart/replay.
                raise
            except Exception as error:
                dlq = dead_letter(LABELS_TOPIC, payload, error)
                producer.produce(
                    DEAD_LETTER_TOPIC,
                    key=message.key(),
                    value=dlq.model_dump_json().encode(),
                )
                producer.flush(5)
                consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()
