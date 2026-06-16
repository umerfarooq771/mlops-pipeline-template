"""
FastAPI model serving endpoint.
Exposes /predict and /health endpoints for the trained failure detection model.
"""

import logging
import os
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Pipeline — Equipment Failure Detection API",
    description="Predicts equipment failure probability from sensor readings.",
    version="1.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost_model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")

model = None
scaler = None


@app.on_event("startup")
def load_model():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        log.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        log.warning(f"Could not load model at startup: {e}")


class SensorReading(BaseModel):
    temperature_c: float = Field(..., example=78.5, description="Temperature in Celsius")
    vibration_mms: float = Field(..., example=0.6, description="Vibration in mm/s")
    pressure_bar: float = Field(..., example=102.0, description="Pressure in bar")
    rpm: int = Field(..., example=2950, description="Rotations per minute")
    oil_level: float = Field(..., ge=0, le=1, example=0.75, description="Oil level 0–1")
    machine_id: Optional[str] = Field(None, example="M01")


class PredictionResponse(BaseModel):
    machine_id: Optional[str]
    failure_probability: float
    risk_level: str
    recommendation: str
    timestamp: str


def build_features(reading: SensorReading) -> np.ndarray:
    base = {
        "temperature_c": reading.temperature_c,
        "vibration_mms": reading.vibration_mms,
        "pressure_bar": reading.pressure_bar,
        "rpm": reading.rpm,
        "oil_level": reading.oil_level,
    }
    features = list(base.values())
    # Rolling features — approximate with base values for single readings
    for v in list(base.values()):
        features.append(v)  # roll_mean_6h ≈ current
        features.append(0.0)  # roll_std_6h ≈ 0 for single reading

    features.append(reading.temperature_c / (reading.vibration_mms + 1e-6))
    features.append(reading.pressure_bar / (reading.rpm + 1e-6))

    return np.array(features).reshape(1, -1)


def risk_label(prob: float) -> tuple:
    if prob < 0.3:
        return "LOW", "Continue normal operations. Schedule next routine inspection."
    elif prob < 0.6:
        return "MEDIUM", "Increase monitoring frequency. Inspect within 48 hours."
    elif prob < 0.8:
        return "HIGH", "Immediate inspection recommended. Reduce load if possible."
    else:
        return "CRITICAL", "Halt operations and perform emergency maintenance immediately."


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training pipeline first.")

    features = build_features(reading)
    if scaler:
        features = scaler.transform(features)

    prob = float(model.predict_proba(features)[0][1])
    risk, recommendation = risk_label(prob)

    return PredictionResponse(
        machine_id=reading.machine_id,
        failure_probability=round(prob, 4),
        risk_level=risk,
        recommendation=recommendation,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/")
def root():
    return {"message": "MLOps Pipeline API. Visit /docs for interactive API documentation."}
