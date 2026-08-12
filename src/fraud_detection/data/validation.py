"""Batch validation for the original Sparkov CSVs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from fraud_detection.contracts import MerchantCategory
from fraud_detection.exceptions import DataValidationError
from fraud_detection.utils.config import Settings, project_root

RAW_COLUMNS = {
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
}


def _scan(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, try_parse_dates=False).select([pl.col(name) for name in RAW_COLUMNS])


def validate_raw_data(settings: Settings, write_report: bool = True) -> dict[str, Any]:
    raw_dir = project_root() / "data/raw"
    frames = [_scan(raw_dir / name) for name in settings.data.files]
    data = pl.concat(frames).with_columns(
        pl.col("trans_date_trans_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ts"),
        pl.col("amt").cast(pl.Float64),
        pl.col("is_fraud").cast(pl.Int8),
    )
    allowed = {item.value for item in MerchantCategory if item != MerchantCategory.OTHER}
    summary = (
        data.select(
            pl.len().alias("rows"),
            pl.col("is_fraud").sum().alias("fraud_rows"),
            pl.col("trans_num").n_unique().alias("unique_transactions"),
            pl.col("cc_num").n_unique().alias("customers"),
            pl.col("merchant").n_unique().alias("merchants"),
            pl.col("ts").min().alias("minimum_timestamp"),
            pl.col("ts").max().alias("maximum_timestamp"),
            pl.col("amt").min().alias("minimum_amount"),
            pl.col("amt").max().alias("maximum_amount"),
            pl.sum_horizontal(pl.all().null_count()).alias("null_values"),
        )
        .collect()
        .to_dicts()[0]
    )
    categories = set(data.select("category").unique().collect()["category"].to_list())
    failures: list[str] = []
    if summary["rows"] != settings.data.expected_rows:
        failures.append(f"expected {settings.data.expected_rows} rows, found {summary['rows']}")
    if summary["fraud_rows"] != settings.data.expected_fraud:
        failures.append(
            f"expected {settings.data.expected_fraud} fraud rows, found {summary['fraud_rows']}"
        )
    if summary["unique_transactions"] != summary["rows"]:
        failures.append("duplicate transaction IDs detected")
    if summary["minimum_amount"] <= 0:
        failures.append("non-positive transaction amount detected")
    if summary["null_values"]:
        failures.append("missing values detected")
    if categories != allowed:
        failures.append(f"invalid category registry: {sorted(categories ^ allowed)}")
    minimum = summary["minimum_timestamp"]
    maximum = summary["maximum_timestamp"]
    if minimum < datetime(2019, 1, 1) or maximum >= datetime(2021, 1, 1):
        failures.append("timestamp outside declared Sparkov window")
    report = {
        **{k: str(v) if isinstance(v, datetime) else v for k, v in summary.items()},
        "fraud_rate": summary["fraud_rows"] / summary["rows"],
        "categories": sorted(categories),
        "checks_passed": not failures,
        "failures": failures,
    }
    if write_report:
        reports = project_root() / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "data_validation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    if failures:
        raise DataValidationError("; ".join(failures))
    return report
