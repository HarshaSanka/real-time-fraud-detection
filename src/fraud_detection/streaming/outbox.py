"""Transactional outbox delivery isolates predictions from Kafka outages."""

from __future__ import annotations

import time

from fraud_detection.storage.database import PredictionStore
from fraud_detection.utils.config import Settings


def publish_pending(settings: Settings, once: bool = False) -> int:
    from confluent_kafka import Producer

    store = PredictionStore(settings.serving.database_url)
    producer = Producer(
        {"bootstrap.servers": settings.serving.kafka_bootstrap_servers, "enable.idempotence": True}
    )
    published = 0
    while True:
        events = store.pending_outbox()
        for event in events:
            producer.produce(
                event.topic,
                key=event.message_key.encode(),
                value=event.payload_json.encode(),
            )
            producer.flush(5)
            store.mark_published(event.id)
            published += 1
        if once:
            return published
        time.sleep(1 if events else 5)
