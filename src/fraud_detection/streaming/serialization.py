"""Strict JSON serialization with sanitized dead letters."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fraud_detection.contracts import DeadLetterEventV1, TransactionEventV1


def serialize_event(event: TransactionEventV1) -> bytes:
    return event.model_dump_json().encode()


def deserialize_event(payload: bytes) -> TransactionEventV1:
    return TransactionEventV1.model_validate_json(payload)


def dead_letter(
    source_topic: str,
    payload: bytes,
    error: Exception,
    trace_id: str | None = None,
) -> DeadLetterEventV1:
    sanitized: dict[str, Any] = {}
    try:
        parsed = json.loads(payload)
        for key in ("schema_version", "transaction_id"):
            if key in parsed:
                sanitized[key] = parsed[key]
    except Exception:
        sanitized["payload_bytes"] = len(payload)
    return DeadLetterEventV1(
        source_topic=source_topic,
        error_type=type(error).__name__,
        error_message=str(error)[:500],
        received_at=datetime.now(UTC),
        trace_id=trace_id,
        sanitized_payload=sanitized,
    )
