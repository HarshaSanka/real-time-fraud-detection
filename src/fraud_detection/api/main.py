"""FastAPI application with separate liveness and readiness semantics."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fraud_detection.api.schemas import BatchScoreRequest, BatchScoreResponse
from fraud_detection.contracts import ScoreResponseV1, TransactionEventV1
from fraud_detection.inference.feature_store import (
    InMemoryFeatureStore,
    OnlineFeatureStore,
    RedisFeatureStore,
)
from fraud_detection.inference.predictor import Predictor
from fraud_detection.inference.service import ScoringService
from fraud_detection.storage.database import PredictionStore
from fraud_detection.utils.config import load_settings, project_root


def build_service() -> ScoringService:
    settings = load_settings()
    bundle = Path(settings.serving.model_bundle)
    if not bundle.is_absolute():
        bundle = project_root() / bundle
    predictor = Predictor(bundle)
    prediction_store = PredictionStore(settings.serving.database_url)
    try:
        redis_store = RedisFeatureStore(settings.serving.redis_url, settings.data.label_delay_days)
        redis_store.client.ping()
        feature_store: OnlineFeatureStore = redis_store
    except Exception:
        feature_store = InMemoryFeatureStore(settings.data.label_delay_days, degraded=True)
    return ScoringService(predictor, feature_store, prediction_store)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        app.state.scoring_service = build_service()
        app.state.readiness_error = None
    except Exception as error:
        app.state.scoring_service = None
        app.state.readiness_error = str(error)
    yield


app = FastAPI(
    title="Real-Time Fraud Detection API",
    version="1.0.0",
    description="Calibrated fraud risk scoring with APPROVE, REVIEW, and BLOCK decisions.",
    lifespan=lifespan,
)


def service(request: Request) -> ScoringService:
    scoring_service = getattr(request.app.state, "scoring_service", None)
    if scoring_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model service is not ready",
        )
    return scoring_service  # type: ignore[no-any-return]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    service(request)
    return {"status": "ready"}


@app.get("/model/info")
def model_info(request: Request) -> dict[str, object]:
    scoring_service = service(request)
    metadata = scoring_service.predictor.metadata
    return metadata.model_dump(mode="json")


@app.post("/score", response_model=ScoreResponseV1)
def score(event: TransactionEventV1, request: Request) -> ScoreResponseV1:
    try:
        event.validate_live_time()
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return service(request).score(event, source="api")


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(payload: BatchScoreRequest, request: Request) -> BatchScoreResponse:
    scoring_service = service(request)
    predictions: list[ScoreResponseV1] = []
    for event in payload.transactions:
        try:
            event.validate_live_time()
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        predictions.append(scoring_service.score(event, source="batch"))
    return BatchScoreResponse(predictions=predictions)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
