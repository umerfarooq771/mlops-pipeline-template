"""
Data ingestion & preprocessing pipeline.
Loads raw sensor data, engineers features, and splits train/test sets.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

NUMERIC_FEATURES = ["temperature_c", "vibration_mms", "pressure_bar", "rpm", "oil_level"]
TARGET = "failure"
SCALER_PATH = "models/scaler.joblib"


def load_data(path: str) -> pd.DataFrame:
    log.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    log.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Engineering rolling features...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in NUMERIC_FEATURES:
        df[f"{col}_roll_mean_6h"] = df[col].rolling(6, min_periods=1).mean()
        df[f"{col}_roll_std_6h"] = df[col].rolling(6, min_periods=1).std().fillna(0)

    df["temp_vibration_ratio"] = df["temperature_c"] / (df["vibration_mms"] + 1e-6)
    df["pressure_rpm_ratio"] = df["pressure_bar"] / (df["rpm"] + 1e-6)

    log.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"timestamp", "machine_id", TARGET}
    return [c for c in df.columns if c not in exclude]


def preprocess(data_path: str, test_size: float = 0.2, save_scaler: bool = True):
    df = load_data(data_path)
    df = engineer_features(df)

    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        log.info(f"Scaler saved to {SCALER_PATH}")

    log.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    log.info(f"Failure rate — Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols


if __name__ == "__main__":
    preprocess("data/synthetic/sensor_data.csv")
