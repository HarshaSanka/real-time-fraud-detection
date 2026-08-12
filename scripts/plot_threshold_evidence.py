"""Render the committed champion threshold-search evidence."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from fraud_detection.utils.config import project_root


def main() -> None:
    root = project_root()
    summary = json.loads((root / "reports/benchmark_summary.json").read_text(encoding="utf-8"))
    champion = str(summary["champion"])
    grid = pd.read_csv(root / f"reports/threshold_grid_{champion}.csv")
    feasible = grid[grid.feasible].copy()
    if feasible.empty:
        raise RuntimeError("the champion has no feasible threshold policies")
    selected = summary["results"][champion]["selection_metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter = axes[0].scatter(
        feasible.review_rate * 100,
        feasible.expected_cost,
        c=feasible.fraud_dollar_capture * 100,
        cmap="viridis",
        s=28,
        alpha=0.75,
    )
    axes[0].scatter(
        float(selected["review_rate"]) * 100,
        float(selected["expected_cost"]),
        marker="*",
        s=260,
        color="#dc2626",
        label="selected June policy",
    )
    axes[0].set(
        xlabel="Review volume (%)",
        ylabel="Simulated expected cost ($)",
        title="Feasible policy frontier",
    )
    axes[0].legend()
    fig.colorbar(scatter, ax=axes[0], label="Fraud dollars captured (%)")

    by_review = (
        feasible.sort_values("expected_cost")
        .groupby("review_threshold", as_index=False)
        .first()
        .sort_values("review_threshold")
    )
    axes[1].plot(by_review.review_threshold, by_review.expected_cost, color="#1d4ed8")
    axes[1].axvline(
        float(summary["results"][champion]["review_threshold"]),
        color="#dc2626",
        linestyle="--",
        label="selected review threshold",
    )
    axes[1].set(
        xlabel="Calibrated review threshold",
        ylabel="Best simulated expected cost ($)",
        title=f"{champion.replace('_', ' ').title()} threshold search",
    )
    axes[1].legend()
    fig.suptitle("June 2020 promotion window — sealed test excluded")
    fig.tight_layout()
    output = root / "reports/plots/threshold_optimization.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
