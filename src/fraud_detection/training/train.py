"""End-to-end chronological benchmark, calibration, policy selection, and export."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

from fraud_detection.contracts import ModelBundleMetadata
from fraud_detection.evaluation.cost import evaluate_policy
from fraud_detection.evaluation.metrics import classification_metrics
from fraud_detection.evaluation.thresholds import optimize_thresholds
from fraud_detection.features.pipeline import MODEL_FEATURES, feature_schema_hash
from fraud_detection.models.calibration import ProbabilityCalibrator
from fraud_detection.models.factory import MODEL_NAMES, build_model
from fraud_detection.models.rules import rule_probabilities
from fraud_detection.training.data import build_encoder, load_splits, xy
from fraud_detection.training.imbalance import compare_imbalance_strategies
from fraud_detection.training.promotion import evaluate_promotion
from fraud_detection.training.tracking import log_run
from fraud_detection.training.tuning import tune_models
from fraud_detection.utils.config import Settings, project_root


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "uncommitted"


def _predict(model: Any, matrix: Any) -> np.ndarray:
    return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)


def _plot_pr(labels: np.ndarray, probabilities: dict[str, np.ndarray], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for name, values in probabilities.items():
        PrecisionRecallDisplay.from_predictions(labels, values, name=name, ax=axis)
    axis.set_title("Sealed-test precision-recall curves")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_calibration(labels: np.ndarray, probabilities: np.ndarray, output: Path) -> None:
    bins = pd.DataFrame({"label": labels, "probability": probabilities})
    bins["bin"] = pd.cut(bins["probability"], bins=list(np.linspace(0, 1, 11)), include_lowest=True)
    grouped = (
        bins.groupby("bin", observed=False)
        .agg(predicted=("probability", "mean"), observed=("label", "mean"), count=("label", "size"))
        .dropna()
    )
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    axis.plot(grouped.predicted, grouped.observed, marker="o", label="champion")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed fraud rate", title="Calibration")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run_benchmark(settings: Settings) -> dict[str, Any]:
    root = project_root()
    reports = root / "reports"
    if settings.profile != "portfolio":
        reports = reports / settings.profile
    plots = reports / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    data = load_splits(settings)
    combined_fit = pd.concat([data.train, data.tune], ignore_index=True)
    encoder = build_encoder()
    fit_x, _ = xy(combined_fit)
    encoder.fit(fit_x)
    matrices: dict[str, Any] = {}
    labels: dict[str, np.ndarray] = {}
    for split_name in ["train", "tune", "calibration", "threshold", "test"]:
        split_frame = getattr(data, split_name)
        x, y = xy(split_frame)
        matrices[split_name] = encoder.transform(x)
        labels[split_name] = y.to_numpy()

    compare_imbalance_strategies(
        matrices["train"], labels["train"], matrices["tune"], labels["tune"], settings
    )
    tuned_parameters = tune_models(
        matrices["train"], labels["train"], matrices["tune"], labels["tune"], settings
    )
    amounts = {
        name: getattr(data, name)["amount"].to_numpy(dtype=float)
        for name in ["calibration", "threshold", "test"]
    }
    results: dict[str, dict[str, Any]] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    fitted: dict[str, tuple[Any, ProbabilityCalibrator]] = {}

    all_legitimate = np.zeros(len(data.test), dtype=float)
    all_legit_metrics = classification_metrics(labels["test"], all_legitimate, 0.5)
    results["always_legitimate"] = {**all_legit_metrics, "model": "always_legitimate"}

    rule_calibrator = ProbabilityCalibrator(settings.model.minimum_calibration_positives)
    rule_calibrator.fit(rule_probabilities(data.calibration), labels["calibration"])
    rule_threshold_probs = rule_calibrator.predict(rule_probabilities(data.threshold))
    rule_policy, _ = optimize_thresholds(
        labels["threshold"], rule_threshold_probs, amounts["threshold"], settings.risk
    )
    rule_test_probs = rule_calibrator.predict(rule_probabilities(data.test))
    rule_metrics = classification_metrics(
        labels["test"], rule_test_probs, rule_policy.review_threshold
    )
    rule_test_policy = evaluate_policy(
        labels["test"],
        rule_test_probs,
        amounts["test"],
        rule_policy.review_threshold,
        rule_policy.block_threshold,
        settings.risk,
    )
    results["rules"] = {
        "model": "rules",
        **rule_metrics,
        **rule_test_policy.as_dict(),
        "calibration_method": rule_calibrator.method,
    }
    test_probabilities["rules"] = rule_test_probs

    for name in MODEL_NAMES:
        started = time.perf_counter()
        model = build_model(name, settings, tuned_parameters.get(name))
        model.fit(matrices["train"], labels["train"])
        tune_pr_auc = average_precision_score(labels["tune"], _predict(model, matrices["tune"]))
        model.fit(
            vstack([matrices["train"], matrices["tune"]]),
            np.concatenate([labels["train"], labels["tune"]]),
        )
        calibrator = ProbabilityCalibrator(settings.model.minimum_calibration_positives)
        calibrator.fit(_predict(model, matrices["calibration"]), labels["calibration"])
        threshold_probabilities = calibrator.predict(_predict(model, matrices["threshold"]))
        policy, grid = optimize_thresholds(
            labels["threshold"], threshold_probabilities, amounts["threshold"], settings.risk
        )
        test_probability = calibrator.predict(_predict(model, matrices["test"]))
        metrics = classification_metrics(labels["test"], test_probability, policy.review_threshold)
        test_policy = evaluate_policy(
            labels["test"],
            test_probability,
            amounts["test"],
            policy.review_threshold,
            policy.block_threshold,
            settings.risk,
        )
        training_seconds = time.perf_counter() - started
        result = {
            "model": name,
            "tune_pr_auc": float(tune_pr_auc),
            **metrics,
            **test_policy.as_dict(),
            "threshold_window_feasible": policy.feasible,
            "calibration_method": calibrator.method,
            "training_seconds": training_seconds,
            "review_threshold": policy.review_threshold,
            "block_threshold": policy.block_threshold,
            "selection_metrics": {
                "pr_auc": float(tune_pr_auc),
                "expected_cost": policy.expected_cost,
                "review_rate": policy.review_rate,
                "block_rate": policy.block_rate,
                "fraud_dollar_capture": policy.fraud_dollar_capture,
                "feasible": policy.feasible,
            },
        }
        results[name] = result
        test_probabilities[name] = test_probability
        fitted[name] = (model, calibrator)
        grid.to_csv(reports / f"threshold_grid_{name}.csv", index=False)
        numeric_metrics = {
            key: float(value)
            for key, value in result.items()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
        }
        run_id = log_run(
            name,
            model,
            model.get_params(),
            numeric_metrics,
            [reports / f"threshold_grid_{name}.csv"],
            {"dataset": "sparkov", "profile": settings.profile, "status": "candidate"},
        )
        result["mlflow_run_id"] = run_id

    logistic = results["logistic_regression"]["selection_metrics"]
    accepted: list[str] = []
    promotion: dict[str, Any] = {}
    for name in MODEL_NAMES[1:]:
        decision = evaluate_promotion(results[name]["selection_metrics"], logistic)
        promotion[name] = {"accepted": decision.accepted, "reasons": decision.reasons}
        if decision.accepted:
            accepted.append(name)
    champion_name = min(
        accepted or ["logistic_regression"],
        key=lambda name: float(results[name]["selection_metrics"]["expected_cost"]),
    )
    champion = results[champion_name]
    champion_model, champion_calibrator = fitted[champion_name]
    version = f"fraud-{champion_name}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    feature_path = root / "data/processed" / f"features_{settings.profile}.parquet"
    metadata = ModelBundleMetadata(
        model_name="fraud-detection",
        model_version=version,
        model_type=champion_name,
        dataset_hash=_hash_file(feature_path),
        feature_schema_hash=feature_schema_hash(),
        feature_names=MODEL_FEATURES,
        training_cutoff=settings.splits.calibration_start,
        calibration_method=champion_calibrator.method,
        review_threshold=float(champion["review_threshold"]),
        block_threshold=float(champion["block_threshold"]),
        threshold_version="business-policy-v1",
        metrics={
            key: float(value)
            for key, value in champion.items()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
        },
        created_at=datetime.now(UTC),
        git_commit=_git_commit(root),
    )
    version_dir = root / "artifacts/model" / version
    version_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, version_dir / "encoder.joblib")
    joblib.dump(champion_model, version_dir / "model.joblib")
    joblib.dump(champion_calibrator, version_dir / "calibrator.joblib")
    (version_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    current_name = "ci" if settings.profile == "ci" else "current"
    current = root / "artifacts/model" / current_name
    if current.exists():
        shutil.rmtree(current)
    shutil.copytree(version_dir, current)

    comparison = pd.DataFrame(results.values())
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    _plot_pr(labels["test"], test_probabilities, plots / "precision_recall_curve.png")
    _plot_calibration(
        labels["test"], test_probabilities[champion_name], plots / "calibration_curve.png"
    )
    summary = {
        "profile": settings.profile,
        "split_rows": {
            name: len(getattr(data, name))
            for name in ["train", "tune", "calibration", "threshold", "test"]
        },
        "champion": champion_name,
        "champion_version": version,
        "promotion": promotion,
        "results": results,
        "test_is_reporting_only": True,
        "selection_basis": "validation PR-AUC plus June threshold-window business policy only",
        "sealed_test_excluded_from_selection": True,
    }
    (reports / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (reports / "mlflow_champion_registry.json").write_text(
        json.dumps(
            {
                "champion": champion_name,
                "version": version,
                "status": "champion",
                "promotion_source": "threshold window only",
                "human_approval_required_for_replacement": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary
