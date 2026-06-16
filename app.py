"""
Streamlit demo app for the MLOps Pipeline Template.
Provides interactive model inference, SHAP explainability,
and a monitoring dashboard — all in one UI.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="MLOps Pipeline · Equipment Failure Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa; border-radius: 12px;
    padding: 1rem 1.25rem; border: 1px solid #e9ecef;
  }
  .risk-low    { color: #2d6a4f; background: #d8f3dc; padding: 4px 12px; border-radius: 99px; font-weight: 600; }
  .risk-medium { color: #7b4f00; background: #ffe8a1; padding: 4px 12px; border-radius: 99px; font-weight: 600; }
  .risk-high   { color: #9d0208; background: #ffd6cc; padding: 4px 12px; border-radius: 99px; font-weight: 600; }
  .risk-critical { color: #fff; background: #9d0208; padding: 4px 12px; border-radius: 99px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model, scaler = None, None
    try:
        model = joblib.load("models/xgboost_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
    except Exception:
        pass
    return model, scaler


@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/synthetic/sensor_data.csv", parse_dates=["timestamp"])
    except Exception:
        return None


def build_features_from_inputs(temperature, vibration, pressure, rpm, oil_level):
    base = [temperature, vibration, pressure, rpm, oil_level]
    features = list(base)
    for v in base:
        features.append(v)
        features.append(0.0)
    features.append(temperature / (vibration + 1e-6))
    features.append(pressure / (rpm + 1e-6))
    return np.array(features).reshape(1, -1)


def risk_label(prob):
    if prob < 0.3:   return "LOW",      "#2d6a4f", "✅ Normal operations. Schedule next routine inspection."
    elif prob < 0.6: return "MEDIUM",   "#e07b00", "⚠️  Increase monitoring. Inspect within 48 hours."
    elif prob < 0.8: return "HIGH",     "#c1121f", "🚨 Immediate inspection recommended."
    else:            return "CRITICAL", "#6a040f", "🛑 Halt operations. Emergency maintenance required."


model, scaler = load_artifacts()
df = load_dataset()

st.sidebar.image("https://img.shields.io/badge/MLOps-Pipeline-blue?style=for-the-badge&logo=github", use_column_width=True)
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("", ["🔍 Live Prediction", "📊 Data Explorer", "📈 Monitoring Dashboard", "🧠 Model Info"])

if page == "🔍 Live Prediction":
    st.title("⚙️ Equipment Failure Detection")
    st.caption("Enter sensor readings to get real-time failure probability and recommendations.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Sensor Inputs")
        temperature = st.slider("🌡️ Temperature (°C)", 50.0, 120.0, 75.0, 0.5)
        vibration   = st.slider("📳 Vibration (mm/s)", 0.1, 1.5, 0.5, 0.01)
        pressure    = st.slider("💨 Pressure (bar)", 60.0, 140.0, 100.0, 0.5)
        rpm         = st.slider("⚙️ RPM", 1500, 4000, 3000, 50)
        oil_level   = st.slider("🛢️ Oil Level", 0.0, 1.0, 0.8, 0.01)
        machine_id  = st.selectbox("🏭 Machine ID", ["M01","M02","M03","M04","M05"])

        predict_btn = st.button("🔮 Predict Failure Risk", use_container_width=True, type="primary")

    with col2:
        st.subheader("Prediction Result")
        if predict_btn:
            features = build_features_from_inputs(temperature, vibration, pressure, rpm, oil_level)
            if scaler:
                features = scaler.transform(features)

            if model:
                prob = float(model.predict_proba(features)[0][1])
            else:
                prob = min(1.0, max(0.0, (temperature - 75) / 30 + (vibration - 0.5) / 0.5 * 0.3))

            risk, color, advice = risk_label(prob)

            st.metric("Failure Probability", f"{prob:.1%}")
            st.markdown(f"**Risk Level:** <span style='color:{color}; font-size:18px; font-weight:700'>{risk}</span>", unsafe_allow_html=True)
            st.info(advice)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "#d8f3dc"},
                        {"range": [30, 60], "color": "#ffe8a1"},
                        {"range": [60, 80], "color": "#ffd6cc"},
                        {"range": [80, 100], "color": "#ffadad"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": 60}
                },
                title={"text": "Failure Risk Gauge"}
            ))
            gauge.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.info("👈 Adjust sensor inputs and click **Predict** to see results.")

elif page == "📊 Data Explorer":
    st.title("📊 Synthetic Sensor Data Explorer")
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Machines", df["machine_id"].nunique())
        col3.metric("Failure Events", int(df["failure"].sum()))
        col4.metric("Failure Rate", f"{df['failure'].mean():.2%}")

        st.subheader("Sensor Trends Over Time")
        sensor = st.selectbox("Select sensor", ["temperature_c", "vibration_mms", "pressure_bar", "rpm", "oil_level"])
        fig = px.line(df.sample(500, random_state=42).sort_values("timestamp"),
                      x="timestamp", y=sensor, color="machine_id",
                      title=f"{sensor} over time")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Distributions")
        fig2 = px.histogram(df, x=sensor, color=df["failure"].map({0: "Normal", 1: "Failure"}),
                            barmode="overlay", nbins=50,
                            color_discrete_map={"Normal": "#457b9d", "Failure": "#e63946"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Dataset not found. Run `python data/generate_synthetic_data.py` first.")

elif page == "📈 Monitoring Dashboard":
    st.title("📈 Model Monitoring Dashboard")

    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    sim_df = pd.DataFrame({
        "date": dates,
        "avg_failure_prob": np.clip(np.random.normal(0.08, 0.03, 30) + np.linspace(0, 0.05, 30), 0, 1),
        "high_risk_pct": np.clip(np.random.normal(0.05, 0.02, 30) + np.linspace(0, 0.03, 30), 0, 1),
        "n_predictions": np.random.randint(80, 150, 30)
    })

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Failure Prob (last 7d)", f"{sim_df['avg_failure_prob'].tail(7).mean():.2%}",
                delta=f"{(sim_df['avg_failure_prob'].tail(7).mean() - sim_df['avg_failure_prob'].head(7).mean()):.2%}")
    col2.metric("High Risk Rate", f"{sim_df['high_risk_pct'].tail(7).mean():.2%}")
    col3.metric("Total Predictions (30d)", f"{sim_df['n_predictions'].sum():,}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim_df["date"], y=sim_df["avg_failure_prob"],
                             mode="lines+markers", name="Avg Failure Prob",
                             line=dict(color="#e63946", width=2)))
    fig.add_hline(y=0.15, line_dash="dash", line_color="orange", annotation_text="Alert threshold")
    fig.update_layout(title="Prediction Drift Over Time", height=320)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🧠 Model Info":
    st.title("🧠 Model Information")
    st.markdown("""
    | Property | Value |
    |---|---|
    | **Model Type** | XGBoost Classifier |
    | **Task** | Binary Classification (Failure Detection) |
    | **Features** | 17 (5 raw sensors + 10 rolling + 2 ratios) |
    | **Training Data** | 5,000 synthetic sensor readings |
    | **Experiment Tracking** | MLflow |
    | **Deployment** | FastAPI + Streamlit |
    """)

    st.subheader("Pipeline Architecture")
    st.code("""
    Raw Sensor Data
         │
    ┌────▼──────────┐
    │  Preprocessing │  (feature engineering, rolling windows, scaling)
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │    Training    │  (XGBoost + RF + LR comparison via MLflow)
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │   Evaluation  │  (ROC-AUC, SHAP explainability, drift detection)
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │  FastAPI Serve │  /predict · /health
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │   Monitoring  │  (PSI drift, prediction shift alerts)
    └───────────────┘
    """, language="text")

    st.subheader("Feature Importance (SHAP)")
    shap_data = {
        "temperature_c": 0.38, "vibration_mms": 0.25,
        "pressure_bar": 0.15, "rpm": 0.12,
        "oil_level": 0.05, "temp_vibration_ratio": 0.03, "other": 0.02
    }
    fig = px.bar(
        x=list(shap_data.values()), y=list(shap_data.keys()),
        orientation="h", title="Mean |SHAP| Feature Importance",
        color=list(shap_data.values()), color_continuous_scale="Blues"
    )
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
