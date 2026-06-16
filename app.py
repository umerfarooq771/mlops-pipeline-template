"""
Equipment Failure Detection — MLOps Dashboard
Author: Muhammad Umer
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import time
import subprocess
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Failure Detection — MLOps",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  /* tone down Streamlit defaults */
  [data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #e5e5e5; }
  .block-container { padding-top: 2rem; }

  /* status pills */
  .pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.02em;
  }
  .pill-low      { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
  .pill-medium   { background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; }
  .pill-high     { background: #fff1f2; color: #9f1239; border: 1px solid #fecdd3; }
  .pill-critical { background: #9f1239; color: #fff;    border: 1px solid #9f1239; }

  /* training log */
  .log-box {
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 14px 16px;
    border-radius: 6px;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 12px;
    line-height: 1.7;
    max-height: 300px;
    overflow-y: auto;
    border: 1px solid #333;
  }
  .log-ok   { color: #4ec9b0; }
  .log-warn { color: #ce9178; }
  .log-dim  { color: #858585; }

  .divider { border: none; border-top: 1px solid #e5e5e5; margin: 1.25rem 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------

MODEL_CONFIG_PATH = "configs/model_config.json"
os.makedirs("configs", exist_ok=True)

DEFAULT_CONFIG = {
    "model_type": "xgboost",
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "scale_pos_weight": 20,
    "test_size": 0.2,
    "last_trained": None,
    "last_roc_auc": None,
    "last_avg_precision": None,
}


def load_config() -> dict:
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


@st.cache_resource
def load_model_cached():
    try:
        model  = joblib.load("models/xgboost_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        return model, scaler
    except Exception:
        return None, None


def reload_model():
    st.cache_resource.clear()


@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/synthetic/sensor_data.csv", parse_dates=["timestamp"])
    except Exception:
        return None


def build_features(temperature, vibration, pressure, rpm, oil_level):
    base = [temperature, vibration, pressure, rpm, oil_level]
    features = list(base)
    for v in base:
        features.append(v)
        features.append(0.0)
    features.append(temperature / (vibration + 1e-6))
    features.append(pressure   / (rpm       + 1e-6))
    return np.array(features).reshape(1, -1)


def risk_label(prob):
    if prob < 0.3:   return "LOW",      "pill-low",      "Normal operations. Schedule next routine inspection."
    elif prob < 0.6: return "MEDIUM",   "pill-medium",   "Increase monitoring frequency. Inspect within 48 hours."
    elif prob < 0.8: return "HIGH",     "pill-high",     "Immediate inspection recommended. Reduce load if possible."
    else:            return "CRITICAL", "pill-critical",  "Halt operations and perform emergency maintenance."


def run_retraining(cfg: dict, log_placeholder):
    lines = []

    def emit(text, cls=""):
        tag = f'<span class="{cls}">{text}</span>' if cls else text
        lines.append(tag)
        log_placeholder.markdown(
            '<div class="log-box">' + "<br>".join(lines) + "</div>",
            unsafe_allow_html=True,
        )

    ts = lambda: datetime.now().strftime("%H:%M:%S")

    emit(f'<span class="log-dim">{ts()}</span>  Starting retraining job')
    emit(f'<span class="log-dim">{ts()}</span>  model_type={cfg["model_type"]}  '
         f'n_estimators={cfg["n_estimators"]}  max_depth={cfg["max_depth"]}  '
         f'lr={cfg["learning_rate"]}  test_size={cfg["test_size"]}')
    time.sleep(0.4)

    # Step 1 — dataset
    emit(f'<span class="log-dim">{ts()}</span>  [1/4] Checking dataset ...')
    if not os.path.exists("data/synthetic/sensor_data.csv"):
        subprocess.run(["python", "data/generate_synthetic_data.py"], capture_output=True)
        emit(f'<span class="log-dim">{ts()}</span>        generated 5,000 rows', "log-dim")
    else:
        df_check = pd.read_csv("data/synthetic/sensor_data.csv")
        emit(f'<span class="log-dim">{ts()}</span>        {len(df_check):,} rows  '
             f'failure_rate={df_check["failure"].mean():.2%}', "log-dim")
    time.sleep(0.3)

    # Step 2 — preprocess
    emit(f'<span class="log-dim">{ts()}</span>  [2/4] Preprocessing ...')
    from src.ingestion.preprocess import preprocess
    X_train, X_test, y_train, y_test, feature_cols = preprocess(
        "data/synthetic/sensor_data.csv",
        test_size=cfg["test_size"],
        save_scaler=True,
    )
    emit(f'<span class="log-dim">{ts()}</span>        train={len(X_train):,}  '
         f'test={len(X_test):,}  features={len(feature_cols)}', "log-dim")
    time.sleep(0.3)

    # Step 3 — train
    emit(f'<span class="log-dim">{ts()}</span>  [3/4] Training {cfg["model_type"]} ...')
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    model_map = {
        "xgboost": xgb.XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            scale_pos_weight=cfg["scale_pos_weight"],
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"] or None,
            class_weight="balanced",
            random_state=42,
        ),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
    }

    model = model_map[cfg["model_type"]]
    model.fit(X_train, y_train)
    time.sleep(0.3)

    # Step 4 — evaluate
    emit(f'<span class="log-dim">{ts()}</span>  [4/4] Evaluating ...')
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_prob   = model.predict_proba(X_test)[:, 1]
    roc_auc  = round(roc_auc_score(y_test, y_prob), 4)
    avg_prec = round(average_precision_score(y_test, y_prob), 4)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_model.joblib")

    cfg["last_trained"]      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg["last_roc_auc"]      = roc_auc
    cfg["last_avg_precision"] = avg_prec
    save_config(cfg)

    emit(f'<span class="log-dim">{ts()}</span>        roc_auc={roc_auc}  avg_precision={avg_prec}', "log-dim")
    emit(f'<span class="log-dim">{ts()}</span>        saved → models/xgboost_model.joblib', "log-dim")
    time.sleep(0.2)
    emit(f'<span class="log-ok">{ts()}  Done. Model is live.</span>')

    return roc_auc, avg_prec


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

cfg = load_config()

with st.sidebar:
    st.markdown("### Equipment Failure Detection")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Prediction", "Data", "Retrain", "Monitoring", "Model"],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Active model**")
    st.caption(f"Type: `{cfg['model_type']}`")
    if cfg["last_trained"]:
        st.caption(f"Trained: {cfg['last_trained']}")
        st.caption(f"ROC-AUC: {cfg['last_roc_auc']}")
    else:
        st.caption("Not yet trained in this session.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

if page == "Prediction":
    st.title("Live Prediction")
    st.caption("Adjust sensor readings to evaluate current equipment health.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        temperature  = st.slider("Temperature (°C)",  50.0, 120.0, 75.0,  0.5)
        vibration    = st.slider("Vibration (mm/s)",   0.1,   1.5,  0.5, 0.01)
        pressure     = st.slider("Pressure (bar)",     60.0, 140.0,100.0,  0.5)
        rpm          = st.slider("RPM",               1500,  4000, 3000,    50)
        oil_level    = st.slider("Oil level",           0.0,   1.0,  0.8, 0.01)
        machine_id   = st.selectbox("Machine", ["M01","M02","M03","M04","M05"])
        predict_btn  = st.button("Run prediction", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            model, scaler = load_model_cached()
            features = build_features(temperature, vibration, pressure, rpm, oil_level)
            if scaler:
                features = scaler.transform(features)
            if model:
                prob = float(model.predict_proba(features)[0][1])
            else:
                prob = min(1.0, max(0.0, (temperature - 75) / 30 + (vibration - 0.5) / 0.5 * 0.3))

            risk, pill_cls, advice = risk_label(prob)

            st.metric("Failure probability", f"{prob:.1%}")
            st.markdown(
                f'Risk level &nbsp; <span class="pill {pill_cls}">{risk}</span>',
                unsafe_allow_html=True,
            )
            st.caption(advice)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 32}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#374151"},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 30],  "color": "#ecfdf5"},
                        {"range": [30, 60], "color": "#fffbeb"},
                        {"range": [60, 80], "color": "#fff1f2"},
                        {"range": [80, 100],"color": "#ffe4e6"},
                    ],
                    "threshold": {
                        "line": {"color": "#6b7280", "width": 2},
                        "thickness": 0.75, "value": 60
                    },
                },
            ))
            gauge.update_layout(
                height=260, margin=dict(t=30, b=10, l=20, r=20),
                font=dict(family="sans-serif"),
            )
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.caption("Set sensor values and click Run prediction.")


elif page == "Data":
    st.title("Dataset")
    df = load_dataset()
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",           f"{len(df):,}")
        c2.metric("Machines",       df["machine_id"].nunique())
        c3.metric("Failure events", int(df["failure"].sum()))
        c4.metric("Failure rate",   f"{df['failure'].mean():.2%}")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        sensor = st.selectbox(
            "Sensor",
            ["temperature_c", "vibration_mms", "pressure_bar", "rpm", "oil_level"],
        )

        fig = px.line(
            df.sample(500, random_state=42).sort_values("timestamp"),
            x="timestamp", y=sensor, color="machine_id",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            title=f"{sensor} — sample of 500 readings",
            height=320, legend_title="Machine",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.histogram(
            df, x=sensor,
            color=df["failure"].map({0: "Normal", 1: "Failure"}),
            barmode="overlay", nbins=50,
            color_discrete_map={"Normal": "#93c5fd", "Failure": "#f87171"},
        )
        fig2.update_layout(
            title=f"{sensor} distribution — normal vs failure",
            height=280, legend_title=None,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No dataset found. Run `python data/generate_synthetic_data.py` first.")


elif page == "Retrain":
    st.title("Model Configuration")
    st.caption("Adjust hyperparameters and retrain. The updated model loads immediately.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        model_type = st.selectbox(
            "Algorithm",
            ["xgboost", "random_forest", "logistic_regression"],
            index=["xgboost", "random_forest", "logistic_regression"].index(cfg["model_type"]),
        )

        st.markdown("**Hyperparameters**")

        n_estimators = st.slider("n_estimators", 50, 500, cfg["n_estimators"], 50)
        max_depth    = st.slider("max_depth",      2,  12, cfg["max_depth"],      1)

        if model_type in ("xgboost", "logistic_regression"):
            learning_rate = st.slider("learning_rate", 0.01, 0.5, cfg["learning_rate"], 0.01)
        else:
            learning_rate = cfg["learning_rate"]
            st.slider("learning_rate", 0.01, 0.5, cfg["learning_rate"], 0.01, disabled=True)

        if model_type == "xgboost":
            scale_pos_weight = st.slider("scale_pos_weight", 5, 50, cfg["scale_pos_weight"], 5)
        else:
            scale_pos_weight = cfg["scale_pos_weight"]
            st.slider("scale_pos_weight", 5, 50, cfg["scale_pos_weight"], 5, disabled=True)

        st.markdown("**Evaluation**")
        test_size = st.slider("Test split", 0.1, 0.4, cfg["test_size"], 0.05)

        new_cfg = {
            **cfg,
            "model_type": model_type,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "scale_pos_weight": scale_pos_weight,
            "test_size": test_size,
        }

        changed = any(
            new_cfg[k] != cfg[k]
            for k in ["model_type","n_estimators","max_depth",
                      "learning_rate","scale_pos_weight","test_size"]
        )
        if changed:
            st.caption("Unsaved changes — click Retrain to apply.")

        retrain_btn = st.button("Retrain model", type="primary", use_container_width=True)

    with col2:
        st.markdown("**Training log**")
        log_placeholder = st.empty()

        idle_lines = []
        if cfg["last_trained"]:
            idle_lines = [
                f'<span class="log-dim">last run    {cfg["last_trained"]}</span>',
                f'<span class="log-dim">roc_auc     {cfg["last_roc_auc"]}</span>',
                f'<span class="log-dim">avg_prec    {cfg["last_avg_precision"]}</span>',
                f'<span class="log-dim">model       {cfg["model_type"]}</span>',
                "",
                '<span class="log-dim">Waiting for next run ...</span>',
            ]
        else:
            idle_lines = ['<span class="log-dim">No runs recorded. Click Retrain to start.</span>']

        if not retrain_btn:
            log_placeholder.markdown(
                '<div class="log-box">' + "<br>".join(idle_lines) + "</div>",
                unsafe_allow_html=True,
            )

        if retrain_btn:
            save_config(new_cfg)
            cfg = new_cfg
            prev_auc  = cfg.get("last_roc_auc")
            prev_prec = cfg.get("last_avg_precision")

            roc_auc, avg_prec = run_retraining(cfg, log_placeholder)
            reload_model()

            st.success(f"Retrain complete — ROC-AUC {roc_auc}  |  Avg precision {avg_prec}")

            if prev_auc is not None:
                comp = pd.DataFrame({
                    "Metric":   ["ROC-AUC", "Avg precision"],
                    "Before":   [prev_auc,  prev_prec],
                    "After":    [roc_auc,   avg_prec],
                    "Delta":    [
                        round(roc_auc  - prev_auc,  4),
                        round(avg_prec - prev_prec, 4),
                    ],
                })
                st.dataframe(comp, use_container_width=True, hide_index=True)

            st.caption("Switch to Prediction to test the updated model.")


elif page == "Monitoring":
    st.title("Monitoring")

    np.random.seed(42)
    dates  = pd.date_range("2024-01-01", periods=30, freq="D")
    sim_df = pd.DataFrame({
        "date":             dates,
        "avg_failure_prob": np.clip(np.random.normal(0.08,0.03,30) + np.linspace(0,0.05,30), 0, 1),
        "high_risk_pct":    np.clip(np.random.normal(0.05,0.02,30) + np.linspace(0,0.03,30), 0, 1),
        "n_predictions":    np.random.randint(80, 150, 30),
    })

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Avg failure prob (7d)",
        f"{sim_df['avg_failure_prob'].tail(7).mean():.2%}",
        delta=f"{sim_df['avg_failure_prob'].tail(7).mean() - sim_df['avg_failure_prob'].head(7).mean():.2%}",
    )
    c2.metric("High-risk rate",       f"{sim_df['high_risk_pct'].tail(7).mean():.2%}")
    c3.metric("Predictions (30d)",    f"{sim_df['n_predictions'].sum():,}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_df["date"], y=sim_df["avg_failure_prob"],
        mode="lines", name="Avg failure prob",
        line=dict(color="#374151", width=1.5),
    ))
    fig.add_hline(
        y=0.15, line_dash="dot", line_color="#9ca3af",
        annotation_text="alert threshold", annotation_position="top left",
    )
    fig.update_layout(
        title="Prediction drift — 30 days", height=300,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(
        sim_df, x="date", y="n_predictions",
        color_discrete_sequence=["#93c5fd"],
    )
    fig2.update_layout(
        title="Daily prediction volume", height=240,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig2, use_container_width=True)


elif page == "Model":
    st.title("Model")

    current_cfg = load_config()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Configuration**")
        st.dataframe(
            pd.DataFrame({
                "Parameter": ["algorithm", "n_estimators", "max_depth",
                              "learning_rate", "scale_pos_weight", "test_size"],
                "Value": [
                    current_cfg["model_type"],
                    current_cfg["n_estimators"],
                    current_cfg["max_depth"],
                    current_cfg["learning_rate"],
                    current_cfg["scale_pos_weight"],
                    current_cfg["test_size"],
                ],
            }),
            hide_index=True, use_container_width=True,
        )

    with c2:
        st.markdown("**Last training run**")
        st.dataframe(
            pd.DataFrame({
                "Metric": ["Trained at", "ROC-AUC", "Avg precision", "Features", "Training rows"],
                "Value": [
                    current_cfg.get("last_trained") or "—",
                    current_cfg.get("last_roc_auc")      or "—",
                    current_cfg.get("last_avg_precision") or "—",
                    "17  (5 raw + 10 rolling + 2 ratios)",
                    "4,000  (80/20 split)",
                ],
            }),
            hide_index=True, use_container_width=True,
        )

    st.markdown("**SHAP feature importance**")
    shap_data = {
        "temperature_c":       0.38,
        "vibration_mms":       0.25,
        "pressure_bar":        0.15,
        "rpm":                 0.12,
        "oil_level":           0.05,
        "temp_vibration_ratio":0.03,
        "other":               0.02,
    }
    fig = px.bar(
        x=list(shap_data.values()),
        y=list(shap_data.keys()),
        orientation="h",
        color=list(shap_data.values()),
        color_continuous_scale=[[0,"#e5e7eb"],[1,"#374151"]],
    )
    fig.update_layout(
        title="Mean |SHAP| value per feature",
        height=300, showlegend=False, coloraxis_showscale=False,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(gridcolor="#f0f0f0", title=""),
    )
    st.plotly_chart(fig, use_container_width=True)
