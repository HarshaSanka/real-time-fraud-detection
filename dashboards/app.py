"""Fraud Operations Command Center backed by persisted predictions and reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

from fraud_detection.utils.config import load_settings, project_root

st.set_page_config(page_title="Fraud Operations", page_icon="🛡️", layout="wide")
settings = load_settings()
root = project_root()

st.title("🛡️ Fraud Operations Command Center")
st.caption("Synthetic Sparkov demonstration — no real cardholder or banking data")


@st.cache_data(ttl=10)
def load_predictions(database_url: str) -> pd.DataFrame:
    try:
        return pd.read_sql(
            "SELECT * FROM predictions ORDER BY created_at DESC", create_engine(database_url)
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


predictions = load_predictions(settings.serving.database_url)
benchmark = load_json(root / "reports/benchmark_summary.json")
drift = load_json(root / "reports/drift_report.json")

overview, model_tab, drift_tab, transaction_tab = st.tabs(
    ["Overview", "Model", "Drift", "Transaction"]
)
with overview:
    if predictions.empty:
        st.info(
            "No live predictions have been persisted yet. Replay synthetic transactions to populate this view."
        )
    else:
        total = len(predictions)
        alerts = int(predictions.decision.isin(["REVIEW", "BLOCK"]).sum())
        amount_at_risk = predictions.loc[
            predictions.decision.isin(["REVIEW", "BLOCK"]), "amount"
        ].sum()
        block_rate = float((predictions.decision == "BLOCK").mean())
        columns = st.columns(4)
        columns[0].metric("Transactions", f"{total:,}")
        columns[1].metric("Fraud alerts", f"{alerts:,}")
        columns[2].metric("Amount at risk", f"${amount_at_risk:,.0f}")
        columns[3].metric("Block rate", f"{block_rate:.2%}")
        decision_counts = (
            predictions.decision.value_counts().rename_axis("decision").reset_index(name="count")
        )
        st.plotly_chart(
            px.bar(decision_counts, x="decision", y="count", color="decision"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.histogram(predictions, x="fraud_probability", nbins=40, color="decision"),
            use_container_width=True,
        )

with model_tab:
    if not benchmark:
        st.info("Measured model results will appear after the benchmark workflow completes.")
    else:
        st.subheader(f"Champion: {benchmark.get('champion')}")
        results = pd.DataFrame(benchmark.get("results", {}).values())
        available = [
            column
            for column in [
                "model",
                "pr_auc",
                "recall",
                "precision",
                "expected_cost",
                "review_rate",
                "block_rate",
            ]
            if column in results
        ]
        st.dataframe(results[available], use_container_width=True, hide_index=True)
        pr_plot = root / "reports/plots/precision_recall_curve.png"
        if pr_plot.exists():
            st.image(str(pr_plot), caption="Sealed-test precision-recall curves")

with drift_tab:
    if not drift:
        st.info("Run the monitoring workflow to generate drift evidence.")
    else:
        st.metric("Drift status", str(drift.get("status", "unknown")).upper())
        numeric = pd.DataFrame(
            [
                {"feature": key, "PSI": value}
                for key, value in dict(drift.get("numeric_psi", {})).items()
            ]
        ).sort_values("PSI", ascending=False)
        st.plotly_chart(
            px.bar(numeric.head(15), x="PSI", y="feature", orientation="h"),
            use_container_width=True,
        )
        st.caption(str(drift.get("concept_drift_note", "")))

with transaction_tab:
    if predictions.empty:
        st.info("No scored transaction is available.")
    else:
        selected = st.selectbox("Transaction ID", predictions.transaction_id.tolist())
        row = predictions[predictions.transaction_id == selected].iloc[0]
        left, right = st.columns(2)
        left.metric("Fraud probability", f"{row.fraud_probability:.2%}")
        right.metric("Decision", row.decision)
        st.write("Reason codes", json.loads(row.reason_codes_json))
        st.json(
            {
                "transaction_id": row.transaction_id,
                "event_timestamp": str(row.event_timestamp),
                "merchant_category": row.merchant_category,
                "model_version": row.model_version,
                "degraded_mode": bool(row.degraded_mode),
            }
        )
