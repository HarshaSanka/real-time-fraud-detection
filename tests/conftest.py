from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fraud_detection.contracts import MerchantCategory, TransactionEventV1


@pytest.fixture
def event() -> TransactionEventV1:
    return TransactionEventV1(
        transaction_id="txn_test_001",
        customer_id="cust_test_001",
        merchant_id="merchant_test_001",
        timestamp=datetime.now(UTC),
        amount=125.0,
        merchant_category=MerchantCategory.SHOPPING_NET,
        merchant_latitude=41.88,
        merchant_longitude=-87.63,
        customer_home_latitude=41.87,
        customer_home_longitude=-87.62,
        customer_state="IL",
        city_population=2_700_000,
    )
