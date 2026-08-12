"""Build the readable EDA notebook from generated, committed evidence."""

# mypy: disable-error-code="no-untyped-call"

from __future__ import annotations

import nbformat as nbf

from fraud_detection.utils.config import project_root


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Sparkov Fraud EDA\n\nThis notebook reads the privacy-safe portfolio Parquet, or the sanitized CI fixture when full data is unavailable, and reproduces the main imbalance and segment evidence. Sparkov is synthetic and is not bank production traffic."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport polars as pl\n\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nportfolio = ROOT / 'data/processed/transactions_portfolio.parquet'\nsource = portfolio if portfolio.exists() else ROOT / 'data/processed/transactions_ci.parquet'\ndata = pl.read_parquet(source)\n{'source': source.name, 'shape': data.shape}"
        ),
        nbf.v4.new_code_cell(
            "summary = data.select(pl.len().alias('transactions'), pl.col('is_fraud').sum().alias('fraud'), pl.col('is_fraud').mean().alias('fraud_rate'), pl.col('amount').sum().alias('total_amount'))\nsummary"
        ),
        nbf.v4.new_code_cell(
            "by_category = data.group_by('merchant_category').agg(pl.len().alias('transactions'), pl.col('is_fraud').mean().alias('fraud_rate')).sort('fraud_rate', descending=True)\nby_category"
        ),
        nbf.v4.new_code_cell(
            "plot = by_category.to_pandas().plot.bar(x='merchant_category', y='fraud_rate', figsize=(12,4), color='#dc2626', title='Fraud rate by synthetic merchant category')\nplot.set_ylabel('Fraud rate')\nplt.tight_layout()"
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation\n\nAccuracy is misleading at this class ratio. Merchant category and amount patterns are generator-specific, so the production design emphasizes chronological validation, calibrated probability, capacity-aware thresholds, delayed labels, and drift monitoring rather than treating this EDA as real bank behavior."
        ),
    ]
    output = project_root() / "notebooks/01_fraud_eda.ipynb"
    output.parent.mkdir(exist_ok=True)
    nbf.write(notebook, output)


if __name__ == "__main__":
    main()
