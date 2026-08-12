"""Load a versioned bundle and return calibrated predictions with reason codes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fraud_detection.contracts import ModelBundleMetadata, TransactionEventV1
from fraud_detection.exceptions import ModelNotReadyError
from fraud_detection.features.pipeline import FEATURE_NAMES


class Predictor:
    def __init__(self, bundle_path: Path) -> None:
        if not bundle_path.exists():
            raise ModelNotReadyError(f"model bundle does not exist: {bundle_path}")
        try:
            self.encoder = joblib.load(bundle_path / "encoder.joblib")
            self.model = joblib.load(bundle_path / "model.joblib")
            self.calibrator = joblib.load(bundle_path / "calibrator.joblib")
            self.metadata = ModelBundleMetadata.model_validate_json(
                (bundle_path / "metadata.json").read_text(encoding="utf-8")
            )
        except Exception as error:
            raise ModelNotReadyError("invalid model bundle") from error

    def predict(self, event: TransactionEventV1, features: dict[str, float]) -> float:
        row: dict[str, Any] = {name: features[name] for name in FEATURE_NAMES}
        row["merchant_category"] = event.merchant_category.value
        row["customer_state"] = event.customer_state or "NA"
        matrix = self.encoder.transform(pd.DataFrame([row]))
        raw = self.model.predict_proba(matrix)[:, 1]
        return float(self.calibrator.predict(raw)[0])

    @staticmethod
    def reason_codes(features: dict[str, float], limit: int = 4) -> list[str]:
        candidates = [
            (features.get("amount_vs_customer_avg_30d", 0), "AMOUNT_ABOVE_CUSTOMER_BASELINE"),
            (features.get("customer_transactions_30m", 0) / 3, "HIGH_TRANSACTION_VELOCITY"),
            (features.get("travel_speed_kmh", 0) / 900, "IMPOSSIBLE_TRAVEL"),
            (features.get("merchant_confirmed_fraud_rate", 0) * 100, "ELEVATED_MERCHANT_RISK"),
            (features.get("customer_confirmed_fraud_rate", 0) * 100, "CONFIRMED_CUSTOMER_HISTORY"),
            (features.get("history_unavailable", 0) * 10, "ONLINE_HISTORY_UNAVAILABLE"),
        ]
        return [code for value, code in sorted(candidates, reverse=True) if value >= 1][:limit]
