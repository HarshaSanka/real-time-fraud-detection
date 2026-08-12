"""Rebuild promotion evidence exclusively from pre-test artifacts."""

from __future__ import annotations

import json

import pandas as pd

from fraud_detection.training.promotion import evaluate_promotion
from fraud_detection.utils.config import project_root


def main() -> None:
    root = project_root()
    path = root / "reports/benchmark_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    results = summary["results"]
    names = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]
    for name in names:
        result = results[name]
        grid = pd.read_csv(root / f"reports/threshold_grid_{name}.csv")
        selected = grid[(grid.review_threshold - result["review_threshold"]).abs() < 1e-12]
        selected = selected[
            (selected.block_threshold - result["block_threshold"]).abs() < 1e-12
        ].iloc[0]
        result["selection_metrics"] = {
            "pr_auc": result["tune_pr_auc"],
            "expected_cost": float(selected.expected_cost),
            "review_rate": float(selected.review_rate),
            "block_rate": float(selected.block_rate),
            "fraud_dollar_capture": float(selected.fraud_dollar_capture),
            "feasible": bool(selected.feasible),
        }
    baseline = results["logistic_regression"]["selection_metrics"]
    promotion = {}
    accepted = []
    for name in names[1:]:
        decision = evaluate_promotion(results[name]["selection_metrics"], baseline)
        promotion[name] = {"accepted": decision.accepted, "reasons": decision.reasons}
        if decision.accepted:
            accepted.append(name)
    champion = min(
        accepted or ["logistic_regression"],
        key=lambda name: results[name]["selection_metrics"]["expected_cost"],
    )
    summary["champion"] = champion
    summary["promotion"] = promotion
    summary["selection_basis"] = "validation PR-AUC plus June threshold-window business policy only"
    summary["sealed_test_excluded_from_selection"] = True
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    registry = {
        "champion": champion,
        "version": summary["champion_version"],
        "status": "champion",
        "selection_basis": summary["selection_basis"],
        "selection_metrics": results[champion]["selection_metrics"],
        "test_is_reporting_only": True,
        "human_approval_required_for_replacement": True,
    }
    (root / "reports/mlflow_champion_registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
