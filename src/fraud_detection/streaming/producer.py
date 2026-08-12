"""Replay pseudonymized Sparkov events while preserving relative order."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

from fraud_detection.contracts import MerchantCategory, TransactionEventV1
from fraud_detection.streaming.serialization import serialize_event
from fraud_detection.streaming.topics import TRANSACTIONS_TOPIC
from fraud_detection.utils.config import Settings, project_root


def row_to_live_event(row: dict[str, Any], timestamp: datetime) -> TransactionEventV1:
    return TransactionEventV1(
        transaction_id=row["transaction_id"],
        customer_id=row["customer_id"],
        merchant_id=row["merchant_id"],
        timestamp=timestamp,
        amount=row["amount"],
        merchant_category=MerchantCategory(row["merchant_category"]),
        merchant_latitude=row["merchant_latitude"],
        merchant_longitude=row["merchant_longitude"],
        customer_home_latitude=row["customer_home_latitude"],
        customer_home_longitude=row["customer_home_longitude"],
        customer_state=row["customer_state"],
        city_population=row["city_population"],
    )


def replay(settings: Settings, limit: int = 1000, rate_per_second: float = 50.0) -> int:
    from confluent_kafka import Producer

    source = project_root() / "data/processed" / f"transactions_{settings.profile}.parquet"
    frame = pl.read_parquet(source).sort("event_timestamp").head(limit)
    producer = Producer(
        {
            "bootstrap.servers": settings.serving.kafka_bootstrap_servers,
            "enable.idempotence": True,
            "acks": "all",
            "client.id": "fraud-transaction-replay",
        }
    )
    first = frame["event_timestamp"][0]
    started = datetime.now(UTC) - timedelta(seconds=1)
    delay = 1.0 / max(rate_per_second, 0.1)
    for row in frame.iter_rows(named=True):
        original = row["event_timestamp"]
        shifted = started + (original - first) / 86400
        event = row_to_live_event(row, shifted)
        producer.produce(
            TRANSACTIONS_TOPIC,
            key=event.transaction_id.encode(),
            value=serialize_event(event),
        )
        producer.poll(0)
        time.sleep(delay)
    producer.flush(30)
    return frame.height
