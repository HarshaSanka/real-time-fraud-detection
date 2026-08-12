"""Chronological feature loading and shared encoding."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_detection.features.pipeline import CATEGORICAL_FEATURES, FEATURE_NAMES, MODEL_FEATURES
from fraud_detection.utils.config import Settings, project_root


@dataclass
class DataSplits:
    train: pd.DataFrame
    tune: pd.DataFrame
    calibration: pd.DataFrame
    threshold: pd.DataFrame
    test: pd.DataFrame


def load_splits(settings: Settings) -> DataSplits:
    source = project_root() / "data/processed" / f"features_{settings.profile}.parquet"
    frame = pd.read_parquet(source)
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
    split = settings.splits
    tune = pd.Timestamp(split.tune_start)
    calibration = pd.Timestamp(split.calibration_start)
    threshold = pd.Timestamp(split.threshold_start)
    test = pd.Timestamp(split.test_start)
    end = pd.Timestamp(split.end)
    return DataSplits(
        train=frame[
            (frame.event_timestamp >= pd.Timestamp(split.train_start))
            & (frame.event_timestamp < tune)
        ].copy(),
        tune=frame[(frame.event_timestamp >= tune) & (frame.event_timestamp < calibration)].copy(),
        calibration=frame[
            (frame.event_timestamp >= calibration) & (frame.event_timestamp < threshold)
        ].copy(),
        threshold=frame[
            (frame.event_timestamp >= threshold) & (frame.event_timestamp < test)
        ].copy(),
        test=frame[(frame.event_timestamp >= test) & (frame.event_timestamp < end)].copy(),
    )


def build_encoder() -> Pipeline:
    numeric = [name for name in FEATURE_NAMES]
    transformer = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline([("columns", transformer), ("scale", StandardScaler(with_mean=False))])


def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame[MODEL_FEATURES], frame["is_fraud"].astype(int)
