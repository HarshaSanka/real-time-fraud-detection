"""Versioned public contracts shared by API and streaming services."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MerchantCategory(StrEnum):
    ENTERTAINMENT = "entertainment"
    FOOD_DINING = "food_dining"
    GAS_TRANSPORT = "gas_transport"
    GROCERY_NET = "grocery_net"
    GROCERY_POS = "grocery_pos"
    HEALTH_FITNESS = "health_fitness"
    HOME = "home"
    KIDS_PETS = "kids_pets"
    MISC_NET = "misc_net"
    MISC_POS = "misc_pos"
    PERSONAL_CARE = "personal_care"
    SHOPPING_NET = "shopping_net"
    SHOPPING_POS = "shopping_pos"
    TRAVEL = "travel"
    OTHER = "other"


class Decision(StrEnum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TransactionEventV1(BaseModel):
    """Authorization-time event; no PAN, CVV, name, or street address is accepted."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    transaction_id: str = Field(min_length=3, max_length=128)
    customer_id: str = Field(min_length=3, max_length=128)
    merchant_id: str = Field(min_length=3, max_length=128)
    timestamp: datetime
    amount: float = Field(gt=0, le=1_000_000)
    currency: Literal["USD"] = "USD"
    merchant_category: MerchantCategory
    merchant_latitude: float = Field(ge=-90, le=90)
    merchant_longitude: float = Field(ge=-180, le=180)
    customer_home_latitude: float | None = Field(default=None, ge=-90, le=90)
    customer_home_longitude: float | None = Field(default=None, ge=-180, le=180)
    customer_state: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    city_population: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_aware_timestamp(self) -> TransactionEventV1:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include a UTC offset")
        return self

    def validate_live_time(self, now: datetime | None = None) -> None:
        reference = now or datetime.now(UTC)
        event_time = self.timestamp.astimezone(UTC)
        if event_time > reference + timedelta(minutes=5):
            raise ValueError("timestamp is more than five minutes in the future")
        if event_time < reference - timedelta(hours=24):
            raise ValueError("timestamp is older than the live scoring window")


class ScoreResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    transaction_id: str
    fraud_probability: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    decision: Decision
    model_version: str
    threshold_version: str
    latency_ms: float = Field(ge=0)
    reason_codes: list[str] = Field(default_factory=list)
    degraded_mode: bool = False


class FraudLabelEventV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    transaction_id: str = Field(min_length=3, max_length=128)
    is_fraud: bool
    observed_at: datetime
    source: Literal["chargeback", "investigation", "simulation"]


class ModelBundleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    model_version: str
    model_type: str
    dataset_hash: str
    feature_schema_hash: str
    feature_names: list[str]
    training_cutoff: datetime
    calibration_method: str
    review_threshold: float = Field(ge=0, le=1)
    block_threshold: float = Field(ge=0, le=1)
    threshold_version: str
    metrics: dict[str, float]
    created_at: datetime
    git_commit: str

    @model_validator(mode="after")
    def ordered_thresholds(self) -> ModelBundleMetadata:
        if self.review_threshold >= self.block_threshold:
            raise ValueError("review_threshold must be below block_threshold")
        return self


class DeadLetterEventV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    source_topic: str
    error_type: str
    error_message: str
    received_at: datetime
    trace_id: str | None = None
    sanitized_payload: dict[str, Any] = Field(default_factory=dict)
