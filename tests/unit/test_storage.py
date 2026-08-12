from datetime import UTC, datetime

from fraud_detection.contracts import Decision, FraudLabelEventV1, RiskLevel, ScoreResponseV1
from fraud_detection.storage.database import PredictionStore, response_from_prediction


def test_idempotent_prediction_label_and_outbox(tmp_path, event) -> None:
    store = PredictionStore(f"sqlite:///{tmp_path / 'test.db'}")
    response = ScoreResponseV1(
        transaction_id=event.transaction_id,
        fraud_probability=0.8,
        risk_level=RiskLevel.HIGH,
        decision=Decision.BLOCK,
        model_version="v1",
        threshold_version="t1",
        latency_ms=10,
        reason_codes=["HIGH_TRANSACTION_VELOCITY"],
    )
    first = store.save_prediction(event, response)
    second = store.save_prediction(event, response)
    assert first.transaction_id == second.transaction_id
    assert len(store.pending_outbox()) == 1
    restored = response_from_prediction(first, "t1")
    assert restored.decision == Decision.BLOCK
    prediction = store.save_label(
        FraudLabelEventV1(
            transaction_id=event.transaction_id,
            is_fraud=True,
            observed_at=datetime.now(UTC),
            source="simulation",
        )
    )
    assert prediction.customer_id == event.customer_id
    store.mark_published(store.pending_outbox()[0].id)
    assert not store.pending_outbox()
