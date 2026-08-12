"""Evaluate champion performance across supported sealed-test segments."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from fraud_detection.training.data import load_splits, xy
from fraud_detection.utils.config import load_settings, project_root


def main() -> None:
    root = project_root()
    settings = load_settings("portfolio")
    bundle = root / "artifacts/model/current"
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    encoder = joblib.load(bundle / "encoder.joblib")
    model = joblib.load(bundle / "model.joblib")
    calibrator = joblib.load(bundle / "calibrator.joblib")
    test = load_splits(settings).test.copy()
    features, _ = xy(test)
    raw = model.predict_proba(encoder.transform(features))[:, 1]
    probabilities = np.asarray(calibrator.predict(raw), dtype=float)
    decisions = probabilities >= float(metadata["review_threshold"])
    test["amount_bucket"] = pd.cut(
        test.amount,
        bins=[0, 25, 75, 200, 500, 2_000, np.inf],
        labels=["0-25", "25-75", "75-200", "200-500", "500-2000", "2000+"],
    ).astype(str)
    test["hour_bucket"] = pd.cut(
        test.event_timestamp.dt.hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["00-05", "06-11", "12-17", "18-23"],
    ).astype(str)
    test["probability"] = probabilities
    test["prediction"] = decisions.astype(int)

    rows: list[dict[str, object]] = []
    for feature in ["merchant_category", "amount_bucket", "hour_bucket", "customer_state"]:
        for value, segment in test.groupby(feature, observed=True):
            segment_labels = segment.is_fraud.to_numpy(dtype=int)
            support = len(segment)
            frauds = int(segment_labels.sum())
            row: dict[str, object] = {
                "segment_feature": feature,
                "segment_value": str(value),
                "transactions": support,
                "frauds": frauds,
                "fraud_rate": float(segment_labels.mean()),
                "minimum_support_met": support >= 1_000 and frauds >= 10,
            }
            if frauds >= 2:
                row.update(
                    {
                        "pr_auc": float(
                            average_precision_score(segment_labels, segment.probability)
                        ),
                        "precision": float(
                            precision_score(segment_labels, segment.prediction, zero_division=0)
                        ),
                        "recall": float(
                            recall_score(segment_labels, segment.prediction, zero_division=0)
                        ),
                    }
                )
            else:
                row.update({"pr_auc": None, "precision": None, "recall": None})
            rows.append(row)
    output = root / "reports/segment_performance.csv"
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
