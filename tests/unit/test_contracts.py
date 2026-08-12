from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from fraud_detection.contracts import MerchantCategory, ModelBundleMetadata, TransactionEventV1


def test_event_rejects_pan_like_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TransactionEventV1(
            transaction_id="txn_123",
            customer_id="cust_123",
            merchant_id="merchant_123",
            timestamp=datetime.now(UTC),
            amount=10,
            merchant_category=MerchantCategory.HOME,
            merchant_latitude=0,
            merchant_longitude=0,
            card_number="4111111111111111",  # type: ignore[call-arg]
        )


def test_live_time_rejects_stale_and_future(event: TransactionEventV1) -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="older"):
        event.model_copy(update={"timestamp": now - timedelta(days=2)}).validate_live_time(now)
    with pytest.raises(ValueError, match="future"):
        event.model_copy(update={"timestamp": now + timedelta(minutes=6)}).validate_live_time(now)


def test_metadata_requires_ordered_thresholds() -> None:
    with pytest.raises(ValidationError):
        ModelBundleMetadata(
            model_name="fraud",
            model_version="v1",
            model_type="xgboost",
            dataset_hash="a",
            feature_schema_hash="b",
            feature_names=["amount"],
            training_cutoff=datetime.now(UTC),
            calibration_method="isotonic",
            review_threshold=0.8,
            block_threshold=0.7,
            threshold_version="v1",
            metrics={},
            created_at=datetime.now(UTC),
            git_commit="test",
        )
