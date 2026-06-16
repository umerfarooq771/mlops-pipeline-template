"""
Model training pipeline with MLflow experiment tracking.
Trains XGBoost classifier and logs params, metrics, and artifacts.
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
import numpy as np
import joblib
import os
import logging
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.preprocess import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "mlops-pipeline-failure-detection"
MODEL_REGISTRY_NAME = "equipment-failure-detector"
DATA_PATH = "data/synthetic/sensor_data.csv"


def get_model_configs():
    return {
        "xgboost": {
            "model": xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=20,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            ),
            "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42
            ),
            "params": {"n_estimators": 100, "class_weight": "balanced"}
        },
        "logistic_regression": {
            "model": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            "params": {"class_weight": "balanced", "max_iter": 1000}
        }
    }


def evaluate(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
    }

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["precision_class1"] = round(report.get("1", {}).get("precision", 0), 4)
    metrics["recall_class1"] = round(report.get("1", {}).get("recall", 0), 4)
    metrics["f1_class1"] = round(report.get("1", {}).get("f1-score", 0), 4)

    log.info(f"\n[{model_name}] Metrics: {metrics}")
    return metrics


def train_all():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    X_train, X_test, y_train, y_test, feature_cols = preprocess(DATA_PATH)

    best_run_id = None
    best_auc = 0

    for name, config in get_model_configs().items():
        with mlflow.start_run(run_name=name) as run:
            log.info(f"\nTraining {name}...")
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("dataset", DATA_PATH)
            mlflow.log_params(config["params"])
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("feature_list", str(feature_cols[:10]))

            model = config["model"]
            model.fit(X_train, y_train)

            metrics = evaluate(model, X_test, y_test, name)
            mlflow.log_metrics(metrics)

            if name == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            joblib.dump(model, f"models/{name}_model.joblib")
            log.info(f"Model saved to models/{name}_model.joblib")

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_run_id = run.info.run_id
                best_model_name = name

    log.info(f"\nBest model: {best_model_name} | ROC-AUC: {best_auc} | Run ID: {best_run_id}")
    return best_run_id


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_all()
