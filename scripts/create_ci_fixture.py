"""Create a deterministic synthetic fixture with both classes and temporal coverage."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from fraud_detection.utils.config import project_root


def main() -> None:
    root = project_root()
    output = root / "data/processed/transactions_ci.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    start = datetime(2019, 1, 1)
    for index in range(10_000):
        timestamp = start + timedelta(minutes=index * 105)
        fraud = int(index % 151 == 0)
        rows.append(
            {
                "transaction_id": f"ci_{index}",
                "customer_id": f"cust_{index % 100}",
                "merchant_id": f"merchant_{index % 30}",
                "event_timestamp": timestamp,
                "amount": float((index % 250) + 20 + fraud * 900),
                "merchant_category": "shopping_net" if fraud else "food_dining",
                "customer_state": "IL",
                "customer_home_latitude": 41.88,
                "customer_home_longitude": -87.63,
                "merchant_latitude": 41.89 if not fraud else 51.50,
                "merchant_longitude": -87.64 if not fraud else -0.12,
                "city_population": 2_746_388,
                "is_fraud": fraud,
            }
        )
    pl.DataFrame(rows).write_parquet(output)


if __name__ == "__main__":
    main()
