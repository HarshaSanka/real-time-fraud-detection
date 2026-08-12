from datetime import datetime, timedelta

import pytest

from fraud_detection.features.engine import StreamingFeatureEngine, haversine_km


def row(index: int, timestamp: datetime, amount: float = 100.0) -> dict[str, object]:
    return {
        "transaction_id": f"txn_{index}",
        "customer_id": "cust_1",
        "merchant_id": f"merchant_{index % 2}",
        "event_timestamp": timestamp,
        "amount": amount,
        "merchant_category": "shopping_net",
        "customer_state": "IL",
        "customer_home_latitude": 41.88,
        "customer_home_longitude": -87.63,
        "merchant_latitude": 41.88 if index < 2 else 51.50,
        "merchant_longitude": -87.63 if index < 2 else -0.12,
        "city_population": 1000,
    }


def test_velocity_is_strictly_pre_event() -> None:
    engine = StreamingFeatureEngine()
    start = datetime(2020, 1, 1, 10)
    first = engine.process(row(0, start), label=0)
    second = engine.process(row(1, start + timedelta(minutes=3)), label=0)
    assert first["customer_transactions_5m"] == 0
    assert second["customer_transactions_5m"] == 1
    assert second["customer_avg_amount_30d"] == 100


def test_delayed_label_not_available_before_seven_days() -> None:
    engine = StreamingFeatureEngine(label_delay_days=7)
    start = datetime(2020, 1, 1)
    engine.process(row(0, start), label=1)
    early = engine.process(row(1, start + timedelta(days=6, hours=23)))
    available = engine.process(row(2, start + timedelta(days=7, seconds=1)))
    assert early["customer_confirmed_fraud_rate"] == pytest.approx(1 / 200)
    assert available["customer_confirmed_fraud_rate"] == pytest.approx(2 / 201)


def test_impossible_travel_and_haversine() -> None:
    engine = StreamingFeatureEngine()
    start = datetime(2020, 1, 1, 10)
    engine.process(row(0, start))
    features = engine.process(row(2, start + timedelta(minutes=20)))
    assert haversine_km(41.88, -87.63, 51.50, -0.12) > 6000
    assert features["impossible_travel_flag"] == 1
