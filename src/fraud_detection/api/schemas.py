"""HTTP-specific batch contracts."""

from pydantic import BaseModel, ConfigDict, Field

from fraud_detection.contracts import ScoreResponseV1, TransactionEventV1


class BatchScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[TransactionEventV1] = Field(min_length=1, max_length=100)


class BatchScoreResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictions: list[ScoreResponseV1]
