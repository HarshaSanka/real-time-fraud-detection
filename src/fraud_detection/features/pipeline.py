"""Materialize reference features in bounded-memory Parquet batches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from fraud_detection.features.engine import FEATURE_NAMES as ENGINE_FEATURE_NAMES
from fraud_detection.features.engine import StreamingFeatureEngine
from fraud_detection.utils.config import Settings, project_root

CATEGORICAL_FEATURES = ["merchant_category", "customer_state"]
FEATURE_NAMES = ENGINE_FEATURE_NAMES
MODEL_FEATURES = FEATURE_NAMES + CATEGORICAL_FEATURES


def feature_schema_hash() -> str:
    return hashlib.sha256(json.dumps(MODEL_FEATURES).encode()).hexdigest()


def build_features(settings: Settings, batch_size: int = 50_000) -> Path:
    root = project_root()
    source = root / "data/processed" / f"transactions_{settings.profile}.parquet"
    output = root / "data/processed" / f"features_{settings.profile}.parquet"
    engine = StreamingFeatureEngine(settings.data.label_delay_days)
    frame = pl.read_parquet(source).sort(["event_timestamp", "transaction_id"])
    writer: pq.ParquetWriter | None = None
    rows: list[dict[str, Any]] = []
    try:
        for event in frame.iter_rows(named=True):
            features = engine.process(event, int(event["is_fraud"]))
            rows.append(
                {
                    "transaction_id": event["transaction_id"],
                    "event_timestamp": event["event_timestamp"],
                    "merchant_category": event["merchant_category"],
                    "customer_state": event["customer_state"],
                    "is_fraud": int(event["is_fraud"]),
                    **features,
                }
            )
            if len(rows) >= batch_size:
                table = pa.Table.from_pylist(rows)
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema, compression="zstd")
                writer.write_table(table)
                rows.clear()
        if rows:
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return output
