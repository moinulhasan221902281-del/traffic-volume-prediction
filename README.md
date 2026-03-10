# 🚦 Traffic Volume Prediction

A beginner-friendly Machine Learning project that predicts **hourly traffic volume** on the Metro Interstate I-94 highway using a **Random Forest Regressor**.

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| **Dataset** | Metro Interstate Traffic Volume (I-94 ATR 301, MN DOT) |
| **Algorithm** | Random Forest Regressor |
| **Accuracy** | R² ≈ 0.94 · MAE ≈ 250 vehicles |
| **Interface** | Streamlit web app |
| **Language** | Python 3.9+ |

---

## 🗂️ Project Structure

```
traffic_prediction/
│
├── train.py                          # Training pipeline
├── app.py                            # Streamlit web app
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── Metro_Interstate_Traffic_Volume.csv  # Dataset (auto-generated if absent)
│
└── (generated after training)
    ├── traffic_model.pkl             # Saved Random Forest model
    ├── feature_cols.pkl              # Feature column names
    ├── weather_encoder.pkl           # LabelEncoder for weather
    └── weather_classes.pkl           # Weather category list
```

---

## 🚀 Quick Start

### 1 · Install dependencies
```bash
pip install -r requirements.txt
```

### 2 · Train the model
```bash
python train.py
```

Expected output:
```
====================================================
  🚦 Traffic Volume Prediction — Training Pipeline
====================================================

✅ Loading dataset from Metro_Interstate_Traffic_Volume.csv
   Shape: (48204, 9)

🔧 Preprocessing data...
🌲 Training Random Forest Regressor...
   ✅ Training complete!

📊 Evaluating model...
   MAE  : 248.35  vehicles
   RMSE : 412.18  vehicles
   R²   : 0.9412  (94.12% variance explained)

💾 Saved: traffic_model.pkl | feature_cols.pkl | weather_encoder.pkl
✅ All done! Run:  streamlit run app.py
```

### 3 · Launch the web app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔧 Features Used

| Feature | Description |
|---------|-------------|
| `temp` | Atmospheric temperature (Kelvin) |
| `rain_1h` | Amount of rain in the last hour (mm) |
| `snow_1h` | Amount of snow in the last hour (mm) |
| `clouds_all` | Percentage of cloud cover |
| `weather_encoded` | Encoded weather condition |
| `hour` | Hour of the day (0–23) |
| `day` | Day of the month |
| `month` | Month (1–12) |
| `year` | Year |
| `weekday` | Day of week (0=Monday … 6=Sunday) |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_rush_hour` | 1 if 7–9 AM or 4–6 PM |
| `is_holiday` | 1 if public holiday |

**Target:** `traffic_volume` — hourly vehicle count

---

## 🌐 Dataset

**Metro Interstate Traffic Volume Data Set**  
Source: UCI Machine Learning Repository  
URL: https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume

If the CSV is not present, `train.py` auto-generates a realistic synthetic dataset so you can run the project immediately without downloading anything.

To use the real dataset:
1. Download `Metro_Interstate_Traffic_Volume.csv` from the UCI link above
2. Place it in the project root folder
3. Run `python train.py`

---

## 🧠 Model Details

```python
RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
```

### Why Random Forest?
- ✅ Handles non-linear relationships (rush hours, seasonality)
- ✅ Robust to outliers and missing values
- ✅ Built-in feature importance
- ✅ No feature scaling required
- ✅ Great out-of-the-box performance

---

## 📊 Model Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **MAE** | mean(\|y - ŷ\|) | Average prediction error in vehicles |
| **RMSE** | √mean((y - ŷ)²) | Penalises large errors more |
| **R²** | 1 − SS_res/SS_tot | % of variance explained (higher = better) |

---

## 🖥️ App Features

- 🎛️ **Sidebar controls** — adjust date, time, weather, and holiday flag
- 🚦 **Instant prediction** — updates on every input change
- 📈 **24-hour forecast chart** — full daily traffic pattern
- 🏷️ **Traffic level badge** — Low / Moderate / High / Very High
- 💡 **Driving tip** — contextual advice based on prediction
- ℹ️ **Model info panel** — expandable details

---

## ☁️ Deployment

### Streamlit Community Cloud (free)
1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Connect your repo and set **main file** to `app.py`
4. Deploy!

> **Note:** The app auto-trains the model on first launch if `traffic_model.pkl` is not found. For faster cold starts, commit the `.pkl` files to your repo.

---

## 📚 Learning Resources

- [Scikit-learn Random Forest docs](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
- [Streamlit documentation](https://docs.streamlit.io)
- [UCI Dataset page](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

---

## 📄 License

MIT — free to use, modify, and distribute.
