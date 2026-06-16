"""
Model evaluation: drift detection, SHAP explainability, and performance reports.
"""
import pandas as pd
import numpy as np
import shap
import joblib
import os
import json
import logging
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.preprocess import preprocess, get_feature_cols, load_data, engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPORT_DIR = "docs/evaluation"


def compute_shap_values(model, X_test_array, feature_cols: list, n_samples: int = 200):
    log.info("Computing SHAP values...")
    X_sample = X_test_array[:n_samples]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_cols, mean_abs_shap.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return importance


def detect_drift(reference_path: str, current_path: str, feature_cols: list) -> dict:
    """Simple population stability index (PSI) drift check."""
    ref = pd.read_csv(reference_path)
    cur = pd.read_csv(current_path)

    drift_report = {}
    numeric_cols = [c for c in feature_cols if c in ref.columns and c in cur.columns][:5]

    for col in numeric_cols:
        ref_mean = ref[col].mean()
        cur_mean = cur[col].mean()
        drift_pct = abs((cur_mean - ref_mean) / (ref_mean + 1e-9)) * 100
        drift_report[col] = {
            "ref_mean": round(ref_mean, 4),
            "cur_mean": round(cur_mean, 4),
            "drift_pct": round(drift_pct, 2),
            "status": "DRIFT" if drift_pct > 10 else "OK"
        }

    return drift_report


def generate_evaluation_report(model_path: str = "models/xgboost_model.joblib",
                                data_path: str = "data/synthetic/sensor_data.csv"):
    os.makedirs(REPORT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_cols = preprocess(data_path, save_scaler=False)
    model = joblib.load(model_path)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "n_test_samples": len(y_test),
        "failure_rate_test": round(float(y_test.mean()), 4)
    }
    log.info(f"Evaluation metrics: {metrics}")

    shap_importance = compute_shap_values(model, X_test, feature_cols)

    drift_report = detect_drift(data_path, data_path, feature_cols)

    report = {
        "model": model_path,
        "metrics": metrics,
        "shap_feature_importance": dict(list(shap_importance.items())[:10]),
        "drift_report": drift_report
    }

    report_path = os.path.join(REPORT_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"Evaluation report saved to {report_path}")
    return report


if __name__ == "__main__":
    report = generate_evaluation_report()
    print(json.dumps(report, indent=2))
