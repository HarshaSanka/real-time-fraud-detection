"""Generated exploratory analysis; every value comes from processed data."""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from fraud_detection.utils.config import Settings, project_root


def generate_eda(settings: Settings) -> dict[str, Any]:
    root = project_root()
    source = root / "data/processed" / f"transactions_{settings.profile}.parquet"
    reports = root / "reports"
    if settings.profile != "portfolio":
        reports = reports / settings.profile
    output = reports / "eda"
    output.mkdir(parents=True, exist_ok=True)
    data = pl.read_parquet(source).with_columns(
        pl.col("event_timestamp").dt.hour().alias("hour"),
        pl.col("event_timestamp").dt.weekday().alias("weekday"),
        pl.col("amount")
        .cut([10, 25, 50, 100, 250, 500, 1000])
        .cast(pl.String)
        .alias("amount_bucket"),
    )
    summary = {
        "rows": data.height,
        "fraud_rows": int(data["is_fraud"].sum()),
        "fraud_rate": float(data["is_fraud"].sum()) / data.height,
        "fraud_amount": float(data.filter(pl.col("is_fraud") == 1)["amount"].sum()),
        "legitimate_amount": float(data.filter(pl.col("is_fraud") == 0)["amount"].sum()),
        "customers": data["customer_id"].n_unique(),
        "merchants": data["merchant_id"].n_unique(),
        "minimum_timestamp": str(data["event_timestamp"].min()),
        "maximum_timestamp": str(data["event_timestamp"].max()),
        "limitations": [
            "Sparkov is simulated and does not represent bank production traffic.",
            "The dataset has no channel, payment method, device, chargeback workflow, or country field.",
            "Labels are generator outputs; the platform simulates a seven-day availability delay.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sampled = (
        data.select("amount", "is_fraud")
        .sample(n=min(200_000, data.height), seed=int(settings.project["seed"]))
        .to_pandas()
    )
    for label, name, color in [(0, "Legitimate", "#2563eb"), (1, "Fraud", "#dc2626")]:
        values = sampled.loc[sampled["is_fraud"] == label, "amount"].clip(upper=1000)
        sns.histplot(
            values, bins=60, stat="density", alpha=0.45, ax=axes[0], label=name, color=color
        )
    axes[0].set_title("Transaction amount distribution (clipped at $1,000)")
    axes[0].legend()
    class_counts = data.group_by("is_fraud").len().sort("is_fraud").to_pandas()
    sns.barplot(data=class_counts, x="is_fraud", y="len", ax=axes[1], hue="is_fraud", legend=False)
    axes[1].set_yscale("log")
    axes[1].set_title("Extreme class imbalance (log scale)")
    fig.tight_layout()
    fig.savefig(output / "class_imbalance_and_amount.png", dpi=160)
    plt.close(fig)

    aggregations: dict[str, pl.DataFrame] = {}
    for column in ["hour", "weekday", "merchant_category", "customer_state", "amount_bucket"]:
        aggregations[column] = (
            data.group_by(column)
            .agg(pl.len().alias("transactions"), pl.col("is_fraud").mean().alias("fraud_rate"))
            .sort(column)
        )
        aggregations[column].write_csv(output / f"fraud_by_{column}.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, column, title in zip(
        axes.ravel(),
        ["hour", "weekday", "merchant_category", "amount_bucket"],
        [
            "Fraud by hour",
            "Fraud by weekday",
            "Fraud by merchant category",
            "Fraud by amount bucket",
        ],
        strict=True,
    ):
        frame = aggregations[column].to_pandas()
        sns.barplot(data=frame, x=column, y="fraud_rate", ax=axis, color="#dc2626")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output / "fraud_segments.png", dpi=160)
    plt.close(fig)
    return summary
