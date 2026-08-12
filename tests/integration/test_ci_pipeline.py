from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from fraud_detection.api.main import app
from fraud_detection.data.eda import generate_eda
from fraud_detection.features.pipeline import build_features
from fraud_detection.monitoring.workflow import run_drift_monitor
from fraud_detection.training.train import run_benchmark
from fraud_detection.utils.config import load_settings, project_root


@pytest.fixture(scope="module")
def ci_bundle() -> dict[str, object]:
    root = project_root()
    subprocess.run(
        [sys.executable, "scripts/create_ci_fixture.py"],
        cwd=root,
        check=True,
    )
    settings = load_settings("ci")
    eda = generate_eda(settings)
    assert eda["rows"] == 10_000
    build_features(settings, batch_size=2000)
    result = run_benchmark(settings)
    run_drift_monitor(settings)
    return result


@pytest.mark.integration
def test_end_to_end_ci_training_has_champion_and_sealed_test(ci_bundle) -> None:
    assert ci_bundle["champion"]
    assert ci_bundle["test_is_reporting_only"] is True
    assert set(ci_bundle["results"]) >= {
        "always_legitimate",
        "rules",
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
    }
    metadata = project_root() / "artifacts/model/ci/metadata.json"
    assert metadata.exists()


@pytest.mark.integration
def test_api_readiness_score_batch_duplicate_and_validation(
    ci_bundle, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FRAUD_PROFILE", "ci")
    monkeypatch.setenv("FRAUD_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("FRAUD_REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setenv("FRAUD_MODEL_BUNDLE", "artifacts/model/ci")
    event = {
        "transaction_id": "txn_integration_001",
        "customer_id": "cust_integration_001",
        "merchant_id": "merchant_integration_001",
        "timestamp": datetime.now(UTC).isoformat(),
        "amount": 1500.0,
        "currency": "USD",
        "merchant_category": "shopping_net",
        "merchant_latitude": 51.5,
        "merchant_longitude": -0.12,
        "customer_home_latitude": 41.88,
        "customer_home_longitude": -87.63,
        "customer_state": "IL",
        "city_population": 2_700_000,
    }
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/model/info").status_code == 200
        first = client.post("/score", json=event)
        assert first.status_code == 200
        assert first.json()["degraded_mode"] is True
        assert first.json()["decision"] in {"REVIEW", "BLOCK"}
        duplicate = client.post("/score", json=event)
        assert duplicate.json()["transaction_id"] == event["transaction_id"]
        event["transaction_id"] = "txn_integration_002"
        batch = client.post("/score/batch", json={"transactions": [event]})
        assert batch.status_code == 200
        assert len(batch.json()["predictions"]) == 1
        invalid = client.post("/score", json={**event, "amount": -1})
        assert invalid.status_code == 422
        assert client.get("/metrics").status_code == 200
