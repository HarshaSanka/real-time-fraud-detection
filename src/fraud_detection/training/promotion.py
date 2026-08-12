"""Human-gated champion/challenger decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reasons: list[str]


def evaluate_promotion(
    candidate: dict[str, float | bool],
    baseline: dict[str, float | bool],
) -> PromotionDecision:
    reasons: list[str] = []
    if not bool(candidate.get("feasible", False)):
        reasons.append("business capacity or fraud-dollar capture constraint failed")
    if float(candidate["pr_auc"]) < float(baseline["pr_auc"]) * 1.02:
        reasons.append("PR-AUC did not improve at least 2% relative to logistic baseline")
    if float(candidate["expected_cost"]) > float(baseline["expected_cost"]) * 0.98:
        reasons.append("simulated expected cost did not improve at least 2%")
    if float(candidate.get("expected_calibration_error", 1.0)) > 0.03:
        reasons.append("expected calibration error exceeded 3%")
    return PromotionDecision(accepted=not reasons, reasons=reasons)
