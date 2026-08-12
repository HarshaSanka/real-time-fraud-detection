"""Generate measured SHAP summaries for the registered champion when supported."""

from __future__ import annotations

import json
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np

from fraud_detection.training.data import load_splits, xy
from fraud_detection.utils.config import Settings, project_root


def generate_shap_artifacts(settings: Settings, sample_rows: int = 1000) -> dict[str, Any]:
    import shap

    root = project_root()
    bundle = root / settings.serving.model_bundle
    encoder = joblib.load(bundle / "encoder.joblib")
    model = joblib.load(bundle / "model.joblib")
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    test = load_splits(settings).test
    sample = test.sample(n=min(sample_rows, len(test)), random_state=int(settings.project["seed"]))
    x, _ = xy(sample)
    matrix = encoder.transform(x)
    feature_names = list(encoder.get_feature_names_out())
    output = root / "reports/explainability"
    output.mkdir(parents=True, exist_ok=True)
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(matrix)
        if isinstance(values, list):
            values = values[-1]
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        importance = np.abs(np.asarray(values)).mean(axis=0)
        top = np.argsort(importance)[-20:]
        ranking = [
            {"feature": feature_names[index], "mean_absolute_shap": float(importance[index])}
            for index in top[::-1]
        ]
        (output / "global_shap_importance.json").write_text(
            json.dumps(ranking, indent=2), encoding="utf-8"
        )
        fig, axis = plt.subplots(figsize=(9, 7))
        axis.barh([feature_names[index] for index in top], importance[top], color="#dc2626")
        axis.set_title(f"Global SHAP importance — {metadata['model_type']}")
        axis.set_xlabel("Mean |SHAP value|")
        fig.tight_layout()
        fig.savefig(output / "global_shap_importance.png", dpi=170)
        plt.close(fig)
        local_index = int(np.argmax(model.predict_proba(matrix)[:, 1]))
        local = sorted(
            [
                {
                    "feature": feature_names[index],
                    "value": float(dense[local_index, index]),
                    "shap_value": float(np.asarray(values)[local_index, index]),
                }
                for index in range(len(feature_names))
            ],
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )[:10]
        (output / "local_shap_example.json").write_text(
            json.dumps(local, indent=2), encoding="utf-8"
        )
        return {"status": "generated", "sample_rows": len(sample), "top_features": ranking}
    except Exception as error:
        report = {
            "status": "unsupported",
            "reason": str(error),
            "model_type": metadata["model_type"],
        }
        (output / "shap_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
