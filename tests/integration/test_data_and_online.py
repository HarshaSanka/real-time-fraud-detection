from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime, timedelta

import fakeredis
import pytest

from fraud_detection.contracts import MerchantCategory, TransactionEventV1
from fraud_detection.data.ingestion import verify_source_files
from fraud_detection.data.preprocessing import _pseudonym
from fraud_detection.data.validation import validate_raw_data
from fraud_detection.inference.feature_store import InMemoryFeatureStore, RedisFeatureStore
from fraud_detection.utils.config import load_settings


@pytest.mark.integration
@pytest.mark.full_data
def test_full_source_validation_and_privacy_transform() -> None:
    settings = load_settings("portfolio")
    verify_source_files(settings)
    report = validate_raw_data(settings, write_report=False)
    assert report["checks_passed"]
    assert report["rows"] == 1_852_394
    assert _pseudonym("1234", "cust", "salt").startswith("cust_")
    assert _pseudonym("1234", "cust", "salt") != _pseudonym("1234", "merchant", "salt")


@pytest.mark.integration
def test_redis_and_memory_feature_parity(monkeypatch) -> None:
    fake = fakeredis.FakeRedis(decode_responses=False)
    monkeypatch.setattr(fake, "lock", lambda *_args, **_kwargs: nullcontext())
    redis_store = RedisFeatureStore("redis://unused")
    redis_store.client = fake
    memory_store = InMemoryFeatureStore()
    start = datetime.now(UTC) - timedelta(minutes=10)
    for index in range(5):
        event = TransactionEventV1(
            transaction_id=f"txn_parity_{index}",
            customer_id="cust_parity",
            merchant_id=f"merchant_{index % 2}",
            timestamp=start + timedelta(minutes=index),
            amount=100 + index,
            merchant_category=MerchantCategory.HOME,
            merchant_latitude=41.88,
            merchant_longitude=-87.63,
            customer_state="IL",
        )
        assert (
            redis_store.get_and_update(event).features
            == memory_store.get_and_update(event).features
        )
    redis_store.apply_label("txn_parity_0", "cust_parity", "merchant_0", True)
    memory_store.apply_label("txn_parity_0", "cust_parity", "merchant_0", True)
    # Kafka redelivery must not count the same observed outcome twice.
    redis_store.apply_label("txn_parity_0", "cust_parity", "merchant_0", True)
    memory_store.apply_label("txn_parity_0", "cust_parity", "merchant_0", True)
    after_label = TransactionEventV1(
        transaction_id="txn_parity_after_label",
        customer_id="cust_parity",
        merchant_id="merchant_0",
        timestamp=start + timedelta(minutes=6),
        amount=110,
        merchant_category=MerchantCategory.HOME,
        merchant_latitude=41.88,
        merchant_longitude=-87.63,
        customer_state="IL",
    )
    redis_features = redis_store.get_and_update(after_label).features
    memory_features = memory_store.get_and_update(after_label).features
    assert redis_features == memory_features
    assert redis_features["customer_confirmed_fraud_rate"] > 0
