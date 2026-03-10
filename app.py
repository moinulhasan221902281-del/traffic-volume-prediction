"""
Traffic Volume Prediction — Streamlit App
Run with:  streamlit run app.py
"""
import os

# create model automatically if missing
if not os.path.exists("traffic_model.pkl"):
    os.system("python Train.py")
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
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
# CSS — clean dark-accent card style
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #f0f4f8; }

    /* Title */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.6rem;
        font-weight: 800;
        color: #e74c3c;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #777;
        margin-top: 0.2rem;
    }

    /* Traffic level badge */
    .badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    .badge-low    { background:#d4edda; color:#155724; }
    .badge-medium { background:#fff3cd; color:#856404; }
    .badge-high   { background:#f8d7da; color:#721c24; }
    .badge-very-high { background:#f5c6cb; color:#491217; }

    /* Tip box */
    .tip-box {
        background: #eaf2ff;
        border-left: 4px solid #3498db;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-top: 1rem;
        font-size: 0.92rem;
        color: #1a3c5e;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 1.4rem 0 0.6rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.3rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #aaa !important;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL & ENCODERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    """Load model and encoders, training if needed."""
    if not os.path.exists("traffic_model.pkl"):
        st.info("🔧 Model not found — training now (takes ~30 s)…")
        import subprocess, sys
        subprocess.run([sys.executable, "train.py"], check=True)

    model         = joblib.load("traffic_model.pkl")
    feature_cols  = joblib.load("feature_cols.pkl")
    weather_enc   = joblib.load("weather_encoder.pkl")
    weather_classes = joblib.load("weather_classes.pkl")
    return model, feature_cols, weather_enc, weather_classes


model, feature_cols, weather_enc, weather_classes = load_assets()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def traffic_level(vol):
    if vol < 1500:
        return "🟢 Low",       "badge-low",       "Light traffic — smooth drive ahead!"
    elif vol < 3500:
        return "🟡 Moderate",  "badge-medium",    "Moderate traffic — allow extra time."
    elif vol < 5500:
        return "🔴 High",      "badge-high",      "Heavy traffic — consider alternate routes."
    else:
        return "🚨 Very High", "badge-very-high", "Severe congestion — significant delays expected."


def predict_traffic(temp_c, rain, snow, clouds, weather,
                    hour, day, month, year, weekday, is_holiday):
    temp_k       = temp_c + 273.15
    is_weekend   = int(weekday >= 5)
    is_rush_hour = int((7 <= hour <= 9) or (16 <= hour <= 18))
    w_enc        = weather_enc.transform([weather])[0]

    row = pd.DataFrame([[
        temp_k, rain, snow, clouds, w_enc,
        hour, day, month, year,
        weekday, is_weekend, is_rush_hour, is_holiday
    ]], columns=feature_cols)

    return int(model.predict(row)[0])


# ─────────────────────────────────────────────
# SIDEBAR — INPUT CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Prediction Inputs")
    st.markdown("---")

    st.markdown("**📅 Date & Time**")
    now = datetime.now()
    sel_date = st.date_input("Date", value=now.date())
    sel_hour = st.slider("Hour of Day", 0, 23, now.hour,
                         format="%d:00")

    st.markdown("---")
    st.markdown("**🌤️ Weather**")
    sel_weather = st.selectbox("Weather Condition", weather_classes)
    sel_temp    = st.slider("Temperature (°C)", -30, 45, 15)
    sel_clouds  = st.slider("Cloud Cover (%)", 0, 100, 30)
    sel_rain    = st.number_input("Rain (mm/h)", 0.0, 100.0, 0.0, step=0.1)
    sel_snow    = st.number_input("Snow (mm/h)", 0.0, 50.0,  0.0, step=0.1)

    st.markdown("---")
    st.markdown("**🏖️ Holiday?**")
    sel_holiday = st.checkbox("Public Holiday", value=False)

    st.markdown("---")
    predict_btn = st.button("🚦 Predict Traffic", use_container_width=True)


# ─────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🚦 Traffic Volume Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Metro Interstate I-94 · Random Forest ML Model</p>', unsafe_allow_html=True)

# Auto-predict on load (or on button press)
weekday = sel_date.weekday()
volume = predict_traffic(
    temp_c=sel_temp,
    rain=sel_rain,
    snow=sel_snow,
    clouds=sel_clouds,
    weather=sel_weather,
    hour=sel_hour,
    day=sel_date.day,
    month=sel_date.month,
    year=sel_date.year,
    weekday=weekday,
    is_holiday=int(sel_holiday)
)

level_label, level_class, level_tip = traffic_level(volume)

# ── Result card ──
st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{volume:,}</div>
    <div class="metric-label">Predicted Vehicles / Hour</div>
    <div>
        <span class="badge {level_class}">{level_label}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="tip-box">💡 {level_tip}</div>', unsafe_allow_html=True)

# ── Input summary ──
st.markdown('<p class="section-header">📋 Input Summary</p>', unsafe_allow_html=True)
day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
col1, col2 = st.columns(2)
with col1:
    st.metric("📅 Date", f"{sel_date.strftime('%b %d, %Y')}")
    st.metric("🕐 Hour", f"{sel_hour:02d}:00")
    st.metric("📆 Day", day_names[weekday])
    st.metric("🌡️ Temperature", f"{sel_temp}°C")
with col2:
    st.metric("🌤️ Weather", sel_weather)
    st.metric("☁️ Clouds", f"{sel_clouds}%")
    st.metric("🌧️ Rain", f"{sel_rain} mm/h")
    st.metric("🏖️ Holiday", "Yes" if sel_holiday else "No")

# ── 24-hour forecast ──
st.markdown('<p class="section-header">📈 24-Hour Forecast</p>', unsafe_allow_html=True)

hourly = []
for h in range(24):
    is_rh = int((7 <= h <= 9) or (16 <= h <= 18))
    v = predict_traffic(
        temp_c=sel_temp, rain=sel_rain, snow=sel_snow,
        clouds=sel_clouds, weather=sel_weather,
        hour=h, day=sel_date.day, month=sel_date.month,
        year=sel_date.year, weekday=weekday,
        is_holiday=int(sel_holiday)
    )
    hourly.append({"Hour": f"{h:02d}:00", "Traffic Volume": v, "Rush Hour": "🔴" if is_rh else ""})

chart_df = pd.DataFrame(hourly).set_index("Hour")
st.line_chart(chart_df["Traffic Volume"], height=220)

# Show peak/trough
peak_row   = chart_df["Traffic Volume"].idxmax()
trough_row = chart_df["Traffic Volume"].idxmin()
c1, c2 = st.columns(2)
c1.metric("🔺 Peak Hour",   peak_row,   f"{chart_df.loc[peak_row,'Traffic Volume']:,} vehicles")
c2.metric("🔻 Quietest Hour", trough_row, f"{chart_df.loc[trough_row,'Traffic Volume']:,} vehicles")

# ── Model info ──
with st.expander("ℹ️ About This Model"):
    st.markdown("""
**Dataset:** Metro Interstate Traffic Volume (I-94 ATR 301, MN DOT)

**Algorithm:** Random Forest Regressor  
- 150 decision trees  
- Max depth: 20  
- Trained on 80% of data, tested on 20%  

**Features used:**
- 🌡️ Temperature, Rain, Snow, Cloud Cover, Weather Type  
- 🕐 Hour, Day, Month, Year, Weekday  
- 🏷️ Is Weekend, Is Rush Hour, Is Holiday  

**Target:** Hourly traffic volume (vehicles per hour)

**Typical accuracy:** R² ≈ 0.94 · MAE ≈ 250 vehicles
    """)

st.markdown("---")
st.caption("Built with ❤️ using Scikit-learn + Streamlit · Metro I-94 Dataset")
