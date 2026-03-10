"""
Traffic Volume Prediction — Streamlit App
Run with: streamlit run app.py
"""

import os

# Auto-train model if missing
if not os.path.exists("traffic_model.pkl"):
    os.system("python train.py")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🚦 Traffic Volume Predictor",
    page_icon="🚦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DARK UI CSS (FIXED)
# ─────────────────────────────────────────────
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(135deg,#0e1117,#111827);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* TITLE */
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    color: white;
}

.sub-title {
    text-align:center;
    color:#9ca3af;
    margin-bottom:20px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* RESULT CARD */
.metric-card {
    background: rgba(255,255,255,0.06);
    padding: 28px;
    border-radius: 18px;
    text-align:center;
    margin-top: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.metric-value {
    font-size: 3.2rem;
    font-weight: 800;
    color: #22c55e;
}

.metric-label {
    font-size: 1rem;
    color: #cbd5e1;
    margin-bottom: 10px;
}

/* BADGES */
.badge {
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
}

.badge-low { background:#065f46; color:#34d399; }
.badge-medium { background:#78350f; color:#facc15; }
.badge-high { background:#7f1d1d; color:#f87171; }
.badge-very-high { background:#450a0a; color:#ef4444; }

/* TIP BOX */
.tip-box {
    background: rgba(59,130,246,0.15);
    padding: 14px;
    border-radius: 12px;
    margin-top: 15px;
    color: #e5e7eb;
}

/* SECTION HEADER */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 10px;
    color: white;
}

/* BUTTON */
.stButton>button {
    border-radius: 10px;
    background: linear-gradient(90deg,#22c55e,#16a34a);
    color:white;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = joblib.load("traffic_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    weather_enc = joblib.load("weather_encoder.pkl")
    weather_classes = joblib.load("weather_classes.pkl")
    return model, feature_cols, weather_enc, weather_classes

model, feature_cols, weather_enc, weather_classes = load_assets()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def traffic_level(vol):
    if vol < 1500:
        return "🟢 Low", "badge-low", "Light traffic — smooth drive ahead!"
    elif vol < 3500:
        return "🟡 Moderate", "badge-medium", "Moderate traffic — allow extra time."
    elif vol < 5500:
        return "🔴 High", "badge-high", "Heavy traffic — consider alternate routes."
    else:
        return "🚨 Very High", "badge-very-high", "Severe congestion — delays expected."


def predict_traffic(temp_c, rain, snow, clouds, weather,
                    hour, day, month, year, weekday, is_holiday):

    temp_k = temp_c + 273.15
    is_weekend = int(weekday >= 5)
    is_rush_hour = int((7 <= hour <= 9) or (16 <= hour <= 18))
    w_enc = weather_enc.transform([weather])[0]

    row = pd.DataFrame([[
        temp_k, rain, snow, clouds, w_enc,
        hour, day, month, year,
        weekday, is_weekend, is_rush_hour, is_holiday
    ]], columns=feature_cols)

    return int(model.predict(row)[0])

# ─────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Prediction Inputs")

    now = datetime.now()
    sel_date = st.date_input("Date", value=now.date())
    sel_hour = st.slider("Hour of Day", 0, 23, now.hour)

    st.markdown("### 🌤️ Weather")
    sel_weather = st.selectbox("Weather", weather_classes)
    sel_temp = st.slider("Temperature (°C)", -30, 45, 15)
    sel_clouds = st.slider("Cloud Cover (%)", 0, 100, 30)
    sel_rain = st.number_input("Rain (mm/h)", 0.0, 100.0, 0.0)
    sel_snow = st.number_input("Snow (mm/h)", 0.0, 50.0, 0.0)

    sel_holiday = st.checkbox("Public Holiday")

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🚦 Traffic Volume Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Metro Interstate I-94 · Random Forest Model</p>', unsafe_allow_html=True)

weekday = sel_date.weekday()

volume = predict_traffic(
    sel_temp, sel_rain, sel_snow, sel_clouds,
    sel_weather, sel_hour,
    sel_date.day, sel_date.month, sel_date.year,
    weekday, int(sel_holiday)
)

level_label, level_class, level_tip = traffic_level(volume)

# RESULT CARD
st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{volume:,}</div>
    <div class="metric-label">Predicted Vehicles / Hour</div>
    <span class="badge {level_class}">{level_label}</span>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="tip-box">💡 {level_tip}</div>', unsafe_allow_html=True)

# INPUT SUMMARY
st.markdown('<p class="section-header">📋 Input Summary</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.metric("Date", sel_date.strftime("%b %d, %Y"))
    st.metric("Hour", f"{sel_hour:02d}:00")
    st.metric("Temperature", f"{sel_temp}°C")

with col2:
    st.metric("Weather", sel_weather)
    st.metric("Clouds", f"{sel_clouds}%")
    st.metric("Holiday", "Yes" if sel_holiday else "No")

st.markdown("---")
st.caption("Built with ❤️ using Scikit-learn + Streamlit")
