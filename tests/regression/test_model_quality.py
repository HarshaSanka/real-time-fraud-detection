import json
from pathlib import Path


def test_committed_benchmark_is_measured_and_selection_is_pretest() -> None:
    summary = json.loads(Path("reports/benchmark_summary.json").read_text())
    assert summary["profile"] == "portfolio"
    assert summary["split_rows"]["test"] == 525_661
    assert summary["sealed_test_excluded_from_selection"] is True
    champion = summary["results"][summary["champion"]]
    assert champion["pr_auc"] > summary["results"]["logistic_regression"]["pr_auc"]
    assert champion["fraud_dollar_capture"] >= 0.80
    assert champion["review_rate"] <= 0.05
    assert champion["block_rate"] <= 0.01


def test_always_legitimate_baseline_catches_no_fraud() -> None:
    summary = json.loads(Path("reports/benchmark_summary.json").read_text())
    baseline = summary["results"]["always_legitimate"]
    assert baseline["recall"] == 0
    assert baseline["false_negative_rate"] == 1
