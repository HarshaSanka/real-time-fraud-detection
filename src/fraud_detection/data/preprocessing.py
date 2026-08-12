"""Privacy-preserving conversion of raw Sparkov data to typed Parquet."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from fraud_detection.utils.config import Settings, project_root


def _pseudonym(value: str, namespace: str, salt: str) -> str:
    payload = f"{salt}:{namespace}:{value}".encode()
    return f"{namespace}_{hashlib.sha256(payload).hexdigest()[:20]}"


def process_raw_data(settings: Settings) -> Path:
    root = project_root()
    raw_dir = root / "data/raw"
    output_dir = root / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [pl.read_csv(raw_dir / name, try_parse_dates=False) for name in settings.data.files]
    raw = pl.concat(frames, how="vertical")
    salt = str(settings.project["pseudonym_salt"])
    customer_values = raw["cc_num"].unique().cast(pl.String).to_list()
    merchant_values = raw["merchant"].unique().to_list()
    customer_map = {value: _pseudonym(value, "cust", salt) for value in customer_values}
    merchant_map = {value: _pseudonym(value, "merchant", salt) for value in merchant_values}
    processed = (
        raw.with_columns(
            pl.col("cc_num").cast(pl.String).replace_strict(customer_map).alias("customer_id"),
            pl.col("merchant").replace_strict(merchant_map).alias("merchant_id"),
            pl.col("trans_num").alias("transaction_id"),
            pl.col("trans_date_trans_time")
            .str.to_datetime("%Y-%m-%d %H:%M:%S")
            .alias("event_timestamp"),
            pl.col("amt").cast(pl.Float64).alias("amount"),
            pl.col("category").alias("merchant_category"),
            pl.col("state").alias("customer_state"),
            pl.col("lat").cast(pl.Float64).alias("customer_home_latitude"),
            pl.col("long").cast(pl.Float64).alias("customer_home_longitude"),
            pl.col("merch_lat").cast(pl.Float64).alias("merchant_latitude"),
            pl.col("merch_long").cast(pl.Float64).alias("merchant_longitude"),
            pl.col("city_pop").cast(pl.Int64).alias("city_population"),
            pl.col("is_fraud").cast(pl.Int8),
        )
        .select(
            "transaction_id",
            "customer_id",
            "merchant_id",
            "event_timestamp",
            "amount",
            "merchant_category",
            "customer_state",
            "customer_home_latitude",
            "customer_home_longitude",
            "merchant_latitude",
            "merchant_longitude",
            "city_population",
            "is_fraud",
        )
        .sort(["event_timestamp", "transaction_id"])
    )
    if settings.data.maximum_rows is not None:
        fraud = processed.filter(pl.col("is_fraud") == 1)
        legitimate_count = max(settings.data.maximum_rows - fraud.height, 0)
        legitimate = processed.filter(pl.col("is_fraud") == 0).sample(
            n=min(legitimate_count, processed.height - fraud.height),
            seed=int(settings.project["seed"]),
        )
        processed = pl.concat([fraud, legitimate]).sort(["event_timestamp", "transaction_id"])
    output = output_dir / f"transactions_{settings.profile}.parquet"
    processed.write_parquet(output, compression="zstd", statistics=True)
    return output
