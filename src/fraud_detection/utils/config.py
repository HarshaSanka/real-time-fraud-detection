"""Typed configuration loading with deterministic profile merging."""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    archive_sha256: str
    files: dict[str, str]
    expected_rows: int
    expected_fraud: int
    label_delay_days: int = 7
    maximum_rows: int | None = None


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train_start: datetime
    tune_start: datetime
    calibration_start: datetime
    threshold_start: datetime
    test_start: datetime
    end: datetime


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_cost: float = 5.0
    review_capture_rate: float = Field(0.80, ge=0, le=1)
    false_block_cost: float = 25.0
    maximum_review_rate: float = Field(0.05, ge=0, le=1)
    maximum_block_rate: float = Field(0.01, ge=0, le=1)
    minimum_fraud_dollar_capture: float = Field(0.80, ge=0, le=1)
    fallback_review_threshold: float = 0.30
    fallback_block_threshold: float = 0.75


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tuning_trials: int = 20
    tuning_sample_rows: int = 250_000
    random_forest_estimators: int = 300
    calibration_method: str = "isotonic"
    minimum_calibration_positives: int = 100
    primary_metric: str = "pr_auc"


class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_bundle: str
    database_url: str
    redis_url: str
    kafka_bootstrap_servers: str
    batch_limit: int = 100
    live_max_age_hours: int = 24
    future_tolerance_minutes: int = 5


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    psi_warning: float = 0.10
    psi_critical: float = 0.20
    js_warning: float = 0.10
    minimum_labels: int = 200
    minimum_fraud_labels: int = 20
    performance_relative_drop: float = 0.10
    cost_relative_increase: float = 0.15


class Settings(BaseModel):
    model_config = ConfigDict(extra="allow")
    project: dict[str, Any]
    data: DataConfig
    splits: SplitConfig
    risk: RiskConfig
    model: ModelConfig
    serving: ServingConfig
    monitoring: MonitoringConfig
    profile: str = "base"
    execution: dict[str, Any] = Field(default_factory=dict)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_settings(profile: str | None = None) -> Settings:
    root = project_root()
    base_path = Path(os.getenv("FRAUD_CONFIG", root / "configs/base.yaml"))
    with base_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    selected = profile or os.getenv("FRAUD_PROFILE", "demo")
    profile_path = root / "configs/profiles" / f"{selected}.yaml"
    if profile_path.exists():
        with profile_path.open(encoding="utf-8") as handle:
            payload = _merge(payload, yaml.safe_load(handle))
    payload["serving"]["database_url"] = os.getenv(
        "FRAUD_DATABASE_URL", payload["serving"]["database_url"]
    )
    payload["serving"]["redis_url"] = os.getenv("FRAUD_REDIS_URL", payload["serving"]["redis_url"])
    payload["serving"]["kafka_bootstrap_servers"] = os.getenv(
        "FRAUD_KAFKA_BOOTSTRAP_SERVERS", payload["serving"]["kafka_bootstrap_servers"]
    )
    payload["serving"]["model_bundle"] = os.getenv(
        "FRAUD_MODEL_BUNDLE", payload["serving"]["model_bundle"]
    )
    return Settings.model_validate(payload)
