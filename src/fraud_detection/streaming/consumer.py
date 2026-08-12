"""At-least-once Kafka scoring consumer with idempotent persistence."""

from __future__ import annotations

import signal

from fraud_detection.api.main import build_service
from fraud_detection.streaming.serialization import dead_letter, deserialize_event
from fraud_detection.streaming.topics import DEAD_LETTER_TOPIC, TRANSACTIONS_TOPIC
from fraud_detection.utils.config import Settings


def run_consumer(settings: Settings) -> None:
    from confluent_kafka import Consumer, Producer

    consumer = Consumer(
        {
            "bootstrap.servers": settings.serving.kafka_bootstrap_servers,
            "group.id": "fraud-scoring-v1",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    producer = Producer(
        {"bootstrap.servers": settings.serving.kafka_bootstrap_servers, "enable.idempotence": True}
    )
    scoring_service = build_service()
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    consumer.subscribe([TRANSACTIONS_TOPIC])
    try:
        while running:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                continue
            payload = message.value()
            if payload is None:
                continue
            try:
                event = deserialize_event(payload)
                event.validate_live_time()
                scoring_service.score(event, source="kafka")
                consumer.commit(message=message, asynchronous=False)
            except Exception as error:
                dlq = dead_letter(TRANSACTIONS_TOPIC, payload, error)
                producer.produce(
                    DEAD_LETTER_TOPIC,
                    key=message.key(),
                    value=dlq.model_dump_json().encode(),
                )
                producer.flush(5)
                consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()
