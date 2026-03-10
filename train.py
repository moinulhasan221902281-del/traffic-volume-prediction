"""
Traffic Volume Prediction - Training Script
Uses Metro Interstate Traffic Volume dataset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(filepath="Metro_Interstate_Traffic_Volume.csv"):
    """Load the Metro Interstate Traffic Volume dataset."""
    if not os.path.exists(filepath):
        print("📥 Dataset not found locally. Generating synthetic dataset...")
        df = generate_synthetic_data()
    else:
        print(f"✅ Loading dataset from {filepath}")
        df = pd.read_csv(filepath)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}\n")
    return df


def generate_synthetic_data(n=40000):
    """Generate a realistic synthetic Metro Interstate Traffic dataset."""
    np.random.seed(42)
    n = n

    date_range = pd.date_range(start="2012-10-02", periods=n, freq="h")

    holiday_list = [
        "None", "Columbus Day", "Veterans Day", "Thanksgiving Day",
        "Christmas Day", "New Years Day", "Washingtons Birthday",
        "Memorial Day", "Independence Day", "State Fair", "Labor Day",
        "Martin Luther King Jr Day"
    ]

    weather_main = ["Clear", "Clouds", "Rain", "Drizzle", "Mist",
                    "Snow", "Fog", "Thunderstorm", "Haze", "Smoke"]
    weather_desc = [
        "sky is clear", "few clouds", "scattered clouds", "broken clouds",
        "overcast clouds", "light rain", "moderate rain", "heavy rain",
        "light snow", "fog"
    ]

    hours = date_range.hour.to_numpy()
    days_of_week = date_range.dayofweek.to_numpy()
    months = date_range.month.to_numpy()

    # Base traffic with realistic daily patterns
    base_traffic = np.clip(
        3000
        + 2000 * np.sin(np.pi * (hours - 6) / 12) * (hours >= 6) * (hours <= 22)
        + 800 * (hours == 8) + 800 * (hours == 17)  # rush hours
        - 1200 * (days_of_week >= 5)                # lower on weekends
        + 300 * np.random.randn(n),
        0, 7500
    )

    temp = 270 + 20 * np.sin(2 * np.pi * (months - 1) / 12) + 10 * np.random.randn(n)
    rain = np.random.exponential(0.1, n)
    snow = np.where(np.isin(months, [12, 1, 2]), np.random.exponential(0.05, n), 0)
    clouds = np.random.randint(0, 101, n)
    weather_idx = np.random.randint(0, len(weather_main), n)
    holiday_idx = np.where(np.random.rand(n) < 0.03,
                           np.random.randint(1, len(holiday_list), n), 0)

    df = pd.DataFrame({
        "holiday": [holiday_list[i] for i in holiday_idx],
        "temp": temp.round(2),
        "rain_1h": rain.round(4),
        "snow_1h": snow.round(4),
        "clouds_all": clouds,
        "weather_main": [weather_main[i] for i in weather_idx],
        "weather_description": [weather_desc[np.random.randint(0, len(weather_desc))] for _ in range(n)],
        "date_time": date_range.strftime("%Y-%m-%d %H:%M:%S"),
        "traffic_volume": base_traffic.astype(int)
    })

    df.to_csv("Metro_Interstate_Traffic_Volume.csv", index=False)
    print(f"   ✅ Synthetic dataset saved (n={n})")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    """Clean and engineer features from the raw DataFrame."""
    print("🔧 Preprocessing data...")
    df = df.copy()

    # Parse datetime
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"]       = df["date_time"].dt.hour
    df["day"]        = df["date_time"].dt.day
    df["month"]      = df["date_time"].dt.month
    df["year"]       = df["date_time"].dt.year
    df["weekday"]    = df["date_time"].dt.weekday          # 0=Mon … 6=Sun
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Rush-hour flag
    df["is_rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 9)) |
        ((df["hour"] >= 16) & (df["hour"] <= 18))
    ).astype(int)

    # Is-holiday flag
    df["is_holiday"] = (df["holiday"] != "None").astype(int)

    # Encode categoricals
    le_weather = LabelEncoder()
    df["weather_encoded"] = le_weather.fit_transform(df["weather_main"])

    # Save encoder for the app
    joblib.dump(le_weather, "weather_encoder.pkl")
    joblib.dump(le_weather.classes_.tolist(), "weather_classes.pkl")

    # Select features
    feature_cols = [
        "temp", "rain_1h", "snow_1h", "clouds_all",
        "weather_encoded",
        "hour", "day", "month", "year",
        "weekday", "is_weekend", "is_rush_hour", "is_holiday"
    ]
    target_col = "traffic_volume"

    X = df[feature_cols]
    y = df[target_col]

    print(f"   Features : {feature_cols}")
    print(f"   Samples  : {len(X)}\n")
    return X, y, feature_cols


# ─────────────────────────────────────────────
# 3. TRAIN MODEL
# ─────────────────────────────────────────────
def train_model(X_train, y_train):
    """Train a Random Forest Regressor."""
    print("🌲 Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("   ✅ Training complete!\n")
    return model


# ─────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    """Print evaluation metrics."""
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"   MAE  : {mae:.2f}  vehicles")
    print(f"   RMSE : {rmse:.2f} vehicles")
    print(f"   R²   : {r2:.4f}  ({r2*100:.2f}% variance explained)\n")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def feature_importance(model, feature_cols):
    """Print top feature importances."""
    print("🔍 Feature Importances:")
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"   {feat:<20} {imp:.4f}  {bar}")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🚦 Traffic Volume Prediction — Training Pipeline")
    print("=" * 55 + "\n")

    # 1. Load
    df = load_data()

    # 2. Preprocess
    X, y, feature_cols = preprocess(df)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📂 Train: {len(X_train)} | Test: {len(X_test)}\n")

    # 4. Train
    model = train_model(X_train, y_train)

    # 5. Evaluate
    metrics = evaluate(model, X_test, y_test)

    # 6. Feature importance
    feature_importance(model, feature_cols)

    # 7. Save
    joblib.dump(model, "traffic_model.pkl")
    joblib.dump(feature_cols, "feature_cols.pkl")
    print("💾 Saved: traffic_model.pkl | feature_cols.pkl | weather_encoder.pkl")
    print("\n✅ All done! Run:  streamlit run app.py")
    print("=" * 55)
