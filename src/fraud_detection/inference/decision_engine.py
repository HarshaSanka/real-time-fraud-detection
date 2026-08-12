"""Three-way decision mapping."""

from fraud_detection.contracts import Decision, RiskLevel


def decide(
    probability: float,
    review_threshold: float,
    block_threshold: float,
    degraded_mode: bool = False,
) -> tuple[RiskLevel, Decision]:
    if probability >= block_threshold:
        return RiskLevel.HIGH, Decision.BLOCK
    if probability >= review_threshold or degraded_mode:
        return RiskLevel.MEDIUM, Decision.REVIEW
    return RiskLevel.LOW, Decision.APPROVE
