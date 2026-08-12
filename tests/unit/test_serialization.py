from fraud_detection.streaming.serialization import dead_letter, deserialize_event, serialize_event
from fraud_detection.streaming.topics import TRANSACTIONS_TOPIC


def test_stream_round_trip(event) -> None:
    assert deserialize_event(serialize_event(event)) == event


def test_dead_letter_is_sanitized() -> None:
    payload = b'{"transaction_id":"txn_123","amount":999,"customer_id":"secret"}'
    result = dead_letter(TRANSACTIONS_TOPIC, payload, ValueError("bad event"))
    assert result.sanitized_payload == {"transaction_id": "txn_123"}
    assert "999" not in result.model_dump_json()
