"""SQLAlchemy persistence shared by SQLite development and PostgreSQL Compose."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from fraud_detection.contracts import (
    Decision,
    FraudLabelEventV1,
    RiskLevel,
    ScoreResponseV1,
    TransactionEventV1,
)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    transaction_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    event_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    customer_id: Mapped[str] = mapped_column(String(128), index=True)
    merchant_id: Mapped[str] = mapped_column(String(128), index=True)
    amount: Mapped[float] = mapped_column(Float)
    merchant_category: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(128), index=True)
    fraud_probability: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(16), index=True)
    risk_level: Mapped[str] = mapped_column(String(16))
    latency_ms: Mapped[float] = mapped_column(Float)
    degraded_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    reason_codes_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class FraudLabel(Base):
    __tablename__ = "fraud_labels"
    transaction_id: Mapped[str] = mapped_column(
        ForeignKey("predictions.transaction_id"), primary_key=True
    )
    is_fraud: Mapped[bool] = mapped_column(Boolean)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String(32))


class OutboxEvent(Base):
    __tablename__ = "outbox_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic: Mapped[str] = mapped_column(String(128), index=True)
    message_key: Mapped[str] = mapped_column(String(128))
    payload_json: Mapped[str] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class PredictionStore:
    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args)
        Base.metadata.create_all(self.engine)

    def get(self, transaction_id: str) -> Prediction | None:
        with Session(self.engine, expire_on_commit=False) as session:
            return session.get(Prediction, transaction_id)

    def save_prediction(self, event: TransactionEventV1, response: ScoreResponseV1) -> Prediction:
        with Session(self.engine, expire_on_commit=False) as session:
            existing = session.get(Prediction, event.transaction_id)
            if existing is not None:
                return existing
            prediction = Prediction(
                transaction_id=event.transaction_id,
                event_timestamp=event.timestamp,
                customer_id=event.customer_id,
                merchant_id=event.merchant_id,
                amount=event.amount,
                merchant_category=event.merchant_category.value,
                model_version=response.model_version,
                fraud_probability=response.fraud_probability,
                decision=response.decision.value,
                risk_level=response.risk_level.value,
                latency_ms=response.latency_ms,
                degraded_mode=response.degraded_mode,
                reason_codes_json=json.dumps(response.reason_codes),
            )
            session.add(prediction)
            topic = (
                "fraud_alerts.v1"
                if response.decision.value in {"REVIEW", "BLOCK"}
                else "fraud_predictions.v1"
            )
            session.add(
                OutboxEvent(
                    topic=topic,
                    message_key=event.transaction_id,
                    payload_json=response.model_dump_json(),
                )
            )
            session.commit()
            session.refresh(prediction)
            return prediction

    def save_label(self, label: FraudLabelEventV1) -> Prediction:
        with Session(self.engine, expire_on_commit=False) as session:
            prediction = session.get(Prediction, label.transaction_id)
            if prediction is None:
                raise KeyError(f"prediction not found: {label.transaction_id}")
            existing = session.get(FraudLabel, label.transaction_id)
            if existing is None:
                session.add(
                    FraudLabel(
                        transaction_id=label.transaction_id,
                        is_fraud=label.is_fraud,
                        observed_at=label.observed_at,
                        source=label.source,
                    )
                )
                session.commit()
            return prediction

    def pending_outbox(self, limit: int = 100) -> list[OutboxEvent]:
        with Session(self.engine, expire_on_commit=False) as session:
            return list(
                session.scalars(
                    select(OutboxEvent)
                    .where(OutboxEvent.published_at.is_(None))
                    .order_by(OutboxEvent.id)
                    .limit(limit)
                )
            )

    def mark_published(self, event_id: int) -> None:
        with Session(self.engine, expire_on_commit=False) as session:
            event = session.get(OutboxEvent, event_id)
            if event:
                event.published_at = datetime.now(UTC)
                event.attempts += 1
                session.commit()

    def counts(self) -> dict[str, int]:
        from sqlalchemy import func

        with Session(self.engine, expire_on_commit=False) as session:
            rows = session.execute(
                select(Prediction.decision, func.count()).group_by(Prediction.decision)
            ).all()
        return {str(decision): int(count) for decision, count in rows}


def response_from_prediction(prediction: Prediction, threshold_version: str) -> ScoreResponseV1:
    return ScoreResponseV1(
        transaction_id=prediction.transaction_id,
        fraud_probability=prediction.fraud_probability,
        risk_level=RiskLevel(prediction.risk_level),
        decision=Decision(prediction.decision),
        model_version=prediction.model_version,
        threshold_version=threshold_version,
        latency_ms=prediction.latency_ms,
        reason_codes=json.loads(prediction.reason_codes_json),
        degraded_mode=prediction.degraded_mode,
    )
