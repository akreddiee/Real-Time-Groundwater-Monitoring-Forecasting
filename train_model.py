import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load Data
data_path = "Telemetry_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset {data_path} not found.")

data = pd.read_csv(data_path)
required_cols = ["date", "Groundwatelevel_m", "rainfall_mm", "temperature_c", "humidity_pct"]
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

data["date"] = pd.to_datetime(data["date"], errors="coerce")
data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# 2. Evaporation Feature Calculation
def calculate_evaporation(temp, humidity):
    humidity = np.clip(humidity, 0, 100)
    return 0.0023 * (temp + 17.8) * np.sqrt(100 - humidity)

data["evaporation_mm"] = calculate_evaporation(data["temperature_c"], data["humidity_pct"])

# 3. Feature Engineering
def create_features(df):
    df = df.copy()
    for window in [7, 14, 30]:
        df[f"gw_rolling_mean_{window}"] = df["Groundwatelevel_m"].rolling(window).mean()
        df[f"rainfall_sum_{window}"] = df["rainfall_mm"].rolling(window).sum()

    for lag in [1, 7, 14]:
        df[f"gw_lag_{lag}"] = df["Groundwatelevel_m"].shift(lag)
        df[f"rain_lag_{lag}"] = df["rainfall_mm"].shift(lag)

    df["day_of_year"] = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    return df

data_clean = create_features(data).dropna().reset_index(drop=True)

FEATURES = [
    "gw_rolling_mean_7", "gw_rolling_mean_14", "gw_rolling_mean_30",
    "rainfall_sum_7", "rainfall_sum_14", "rainfall_sum_30",
    "gw_lag_1", "gw_lag_7", "gw_lag_14",
    "rain_lag_1", "rain_lag_7", "rain_lag_14",
    "rainfall_mm", "temperature_c", "humidity_pct",
    "evaporation_mm", "day_of_year", "month_sin", "month_cos"
]

X = data_clean[FEATURES]
y_gw = data_clean["Groundwatelevel_m"]
y_rain = data_clean["rainfall_mm"]

split = int(len(X) * 0.8)

# 4. Train Groundwater Model
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_gw_train, y_gw_test = y_gw.iloc[:split], y_gw.iloc[split:]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_gw_train_scaled = scaler_y.fit_transform(y_gw_train.values.reshape(-1, 1)).ravel()

gw_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gw_model.fit(X_train_scaled, y_gw_train_scaled)

# Evaluate Groundwater Model
gw_preds_scaled = gw_model.predict(X_test_scaled)
gw_preds = scaler_y.inverse_transform(gw_preds_scaled.reshape(-1, 1)).ravel()

gw_r2 = round(float(r2_score(y_gw_test, gw_preds)), 4)
gw_rmse = round(float(np.sqrt(mean_squared_error(y_gw_test, gw_preds))), 4)
gw_mae = round(float(mean_absolute_error(y_gw_test, gw_preds)), 4)

# 5. Train Rainfall Model
y_rain_train, y_rain_test = y_rain.iloc[:split], y_rain.iloc[split:]

scaler_X_r = StandardScaler()
scaler_y_r = StandardScaler()

X_train_r_scaled = scaler_X_r.fit_transform(X_train)
X_test_r_scaled = scaler_X_r.transform(X_test)
y_rain_train_scaled = scaler_y_r.fit_transform(y_rain_train.values.reshape(-1, 1)).ravel()

rain_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
rain_model.fit(X_train_r_scaled, y_rain_train_scaled)

# Evaluate Rainfall Model
rain_preds_scaled = rain_model.predict(X_test_r_scaled)
rain_preds = scaler_y_r.inverse_transform(rain_preds_scaled.reshape(-1, 1)).ravel()

rain_r2 = round(float(r2_score(y_rain_test, rain_preds)), 4)
rain_rmse = round(float(np.sqrt(mean_squared_error(y_rain_test, rain_preds))), 4)
rain_mae = round(float(mean_absolute_error(y_rain_test, rain_preds)), 4)

# 6. Feature Importances
importances = gw_model.feature_importances_
feature_importances = [
    {"feature": feat, "importance": round(float(imp), 4)}
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
]

# 7. Save Artifacts
joblib.dump(gw_model, os.path.join(MODELS_DIR, "gw_model.joblib"))
joblib.dump(rain_model, os.path.join(MODELS_DIR, "rain_model.joblib"))
joblib.dump(scaler_X, os.path.join(MODELS_DIR, "scaler_X.joblib"))
joblib.dump(scaler_y, os.path.join(MODELS_DIR, "scaler_y.joblib"))
joblib.dump(scaler_X_r, os.path.join(MODELS_DIR, "scaler_X_r.joblib"))
joblib.dump(scaler_y_r, os.path.join(MODELS_DIR, "scaler_y_r.joblib"))

import sys
import sklearn

metrics_data = {
    "environment": {
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__
    },
    "groundwater_model": {
        "r2_score": gw_r2,
        "rmse": gw_rmse,
        "mae": gw_mae
    },
    "rainfall_model": {
        "r2_score": rain_r2,
        "rmse": rain_rmse,
        "mae": rain_mae
    },
    "feature_importances": feature_importances
}

with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_data, f, indent=2)

print("Models trained and serialized successfully to models/")
print(f"Groundwater Model Performance -> R2: {gw_r2}, RMSE: {gw_rmse}, MAE: {gw_mae}")
print(f"Rainfall Model Performance    -> R2: {rain_r2}, RMSE: {rain_rmse}, MAE: {rain_mae}")
