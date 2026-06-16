"""
Model monitoring: tracks prediction drift, data distribution shifts,
and performance degradation over time.
"""
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MONITOR_LOG = "docs/monitoring/drift_log.jsonl"
ALERT_THRESHOLD_PSI = 0.2
ALERT_THRESHOLD_PRED_SHIFT = 0.15


def population_stability_index(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Compute PSI between reference and current distributions."""
    eps = 1e-4
    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + eps
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual) + eps
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def compute_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                           feature_cols: List[str]) -> Dict:
    drift_report = {}
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values
        ref_scaled = (ref_vals - ref_vals.min()) / (ref_vals.ptp() + 1e-9)
        cur_scaled = (cur_vals - ref_vals.min()) / (ref_vals.ptp() + 1e-9)
        cur_scaled = np.clip(cur_scaled, 0, 1)
        psi = population_stability_index(ref_scaled, cur_scaled)
        drift_report[col] = {
            "psi": psi,
            "status": "DRIFT" if psi > ALERT_THRESHOLD_PSI else "STABLE",
            "ref_mean": round(float(ref_vals.mean()), 4),
            "cur_mean": round(float(cur_vals.mean()), 4),
        }
    return drift_report


def log_prediction_batch(predictions: List[float], model_version: str = "v1.0"):
    os.makedirs(os.path.dirname(MONITOR_LOG), exist_ok=True)
    avg_prob = round(float(np.mean(predictions)), 4)
    high_risk_pct = round(float(np.mean([p > 0.6 for p in predictions])), 4)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "n_predictions": len(predictions),
        "avg_failure_probability": avg_prob,
        "high_risk_pct": high_risk_pct,
        "alert": high_risk_pct > ALERT_THRESHOLD_PRED_SHIFT
    }

    with open(MONITOR_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    if entry["alert"]:
        log.warning(f"ALERT: High risk predictions {high_risk_pct:.1%} exceed threshold!")
    else:
        log.info(f"Monitoring OK — avg failure prob: {avg_prob:.3f}")

    return entry


def read_monitoring_log() -> pd.DataFrame:
    if not os.path.exists(MONITOR_LOG):
        return pd.DataFrame()
    records = []
    with open(MONITOR_LOG, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


if __name__ == "__main__":
    import random
    random.seed(42)
    log.info("Simulating monitoring log entries...")
    for _ in range(10):
        preds = [random.uniform(0.0, 0.9) for _ in range(50)]
        log_prediction_batch(preds)

    df = read_monitoring_log()
    print(df.tail())
