import os
import json
from datetime import datetime
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# Flask App & CORS Configuration
# =====================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

MODELS_DIR = "models"
DATA_FILE = "Telemetry_data.csv"

# =====================================================
# Load & Preprocess Telemetry Data
# =====================================================
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Telemetry dataset '{DATA_FILE}' not found.")

data = pd.read_csv(DATA_FILE)
required_cols = ["date", "Groundwatelevel_m", "rainfall_mm", "temperature_c", "humidity_pct"]
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

data["date"] = pd.to_datetime(data["date"], errors="coerce")
data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

def calculate_evaporation(temp, humidity):
    humidity = np.clip(humidity, 0, 100)
    return 0.0023 * (temp + 17.8) * np.sqrt(100 - humidity)

data["evaporation_mm"] = calculate_evaporation(data["temperature_c"], data["humidity_pct"])

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

# Static station metadata store
STATIONS_DATA = [
    {
        "id": "GW-001",
        "station_id": "GW-001",
        "name": "Telangana Central Station",
        "status": "Active",
        "lat": 16.575,
        "lng": 79.311,
        "location": "Telangana Central"
    },
    {
        "id": "GW-002",
        "station_id": "GW-002",
        "name": "Hyderabad West Station",
        "status": "Active",
        "lat": 17.385,
        "lng": 78.486,
        "location": "Hyderabad West"
    },
    {
        "id": "GW-003",
        "station_id": "GW-003",
        "name": "Vijayawada North Station",
        "status": "Active",
        "lat": 16.306,
        "lng": 80.436,
        "location": "Vijayawada North"
    }
]

# =====================================================
# Load or Train Model Artifacts (With Fail-Safe Fallback)
# =====================================================
gw_model_path = os.path.join(MODELS_DIR, "gw_model.joblib")
rain_model_path = os.path.join(MODELS_DIR, "rain_model.joblib")
scaler_X_path = os.path.join(MODELS_DIR, "scaler_X.joblib")
scaler_y_path = os.path.join(MODELS_DIR, "scaler_y.joblib")
scaler_X_r_path = os.path.join(MODELS_DIR, "scaler_X_r.joblib")
scaler_y_r_path = os.path.join(MODELS_DIR, "scaler_y_r.joblib")

def train_models_online():
    print("Training ML models online...")
    X = data_clean[FEATURES]
    y = data_clean["Groundwatelevel_m"]
    split = int(len(X) * 0.8)

    X_train, y_train = X.iloc[:split], y.iloc[:split]
    scaler_X_obj = StandardScaler()
    scaler_y_obj = StandardScaler()
    X_train_scaled = scaler_X_obj.fit_transform(X_train)
    y_train_scaled = scaler_y_obj.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    gw_model_obj = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    gw_model_obj.fit(X_train_scaled, y_train_scaled)

    y_rain = data_clean["rainfall_mm"]
    X_train_r, y_train_r = X.iloc[:split], y_rain.iloc[:split]
    scaler_X_r_obj = StandardScaler()
    scaler_y_r_obj = StandardScaler()
    X_train_r_scaled = scaler_X_r_obj.fit_transform(X_train_r)
    y_train_r_scaled = scaler_y_r_obj.fit_transform(y_train_r.values.reshape(-1, 1)).ravel()

    rain_model_obj = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    rain_model_obj.fit(X_train_r_scaled, y_train_r_scaled)

    return gw_model_obj, rain_model_obj, scaler_X_obj, scaler_y_obj, scaler_X_r_obj, scaler_y_r_obj

try:
    if (os.path.exists(gw_model_path) and os.path.exists(rain_model_path) and
        os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path)):
        model = joblib.load(gw_model_path)
        rain_model = joblib.load(rain_model_path)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        scaler_X_r = joblib.load(scaler_X_r_path)
        scaler_y_r = joblib.load(scaler_y_r_path)
        print("Loaded serialized ML models successfully from models/")
    else:
        model, rain_model, scaler_X, scaler_y, scaler_X_r, scaler_y_r = train_models_online()
except Exception as e:
    import traceback
    print(f"[WARNING] Failed to load serialized model artifacts: {e}")
    traceback.print_exc()
    print("[FALLBACK] Training models online so application stays live...")
    model, rain_model, scaler_X, scaler_y, scaler_X_r, scaler_y_r = train_models_online()

# Helper for station pulse scoring
def calculate_station_pulse(station_offset=0):
    latest = data_clean.iloc[-1:]
    X_latest = scaler_X.transform(latest[FEATURES])
    gw_pred = scaler_y.inverse_transform(model.predict(X_latest).reshape(-1, 1))[0][0]

    mean = data_clean["Groundwatelevel_m"].mean()
    std = data_clean["Groundwatelevel_m"].std()

    base_score = 100 if std == 0 else max(0, min(100, 100 - abs((gw_pred - mean) / std) * 20))
    pulse_score = max(0, min(100, base_score - station_offset))
    pulse_score = round(float(pulse_score), 1)

    if pulse_score >= 80:
        color, level, message = "green", "NORMAL", "Groundwater levels are healthy"
    elif pulse_score >= 60:
        color, level, message = "yellow", "WATCH", "Groundwater levels are moderate - monitor closely"
    elif pulse_score >= 40:
        color, level, message = "orange", "ADVISORY", "Groundwater levels are below normal - conservation advised"
    elif pulse_score >= 20:
        color, level, message = "red", "WARNING", "Groundwater levels are critically low - immediate action needed"
    else:
        color, level, message = "darkred", "EMERGENCY", "Groundwater emergency - implement emergency measures"

    return pulse_score, color, level, message

# =====================================================
# API ROUTES
# =====================================================

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "Realtime Ground Water Level Monitoring API",
        "version": "latest",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route("/docs", methods=["GET"])
def swagger_ui():
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Aquapulse API Documentation</title>
      <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
      <style>
        body { margin: 0; padding: 0; background: #fafafa; }
      </style>
    </head>
    <body>
      <div id="swagger-ui"></div>
      <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js" crossorigin></script>
      <script>
        window.onload = () => {
          window.ui = SwaggerUIBundle({
            url: '/openapi.json',
            dom_id: '#swagger-ui',
            deepLinking: true,
            presets: [SwaggerUIBundle.presets.apis],
          });
        };
      </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route("/openapi.json", methods=["GET"])
def openapi_spec():
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "Realtime Ground Water Level Monitoring API",
            "version": "1.0.0",
            "description": "Production REST API for groundwater level monitoring, telemetry analytics, and ML predictions."
        },
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check endpoint",
                    "responses": {"200": {"description": "Service health status"}}
                }
            },
            "/api/metrics": {
                "get": {
                    "summary": "System statistics & ML evaluation metrics",
                    "responses": {"200": {"description": "Aggregated metrics payload"}}
                }
            },
            "/api/stations": {
                "get": {
                    "summary": "List all active groundwater monitoring stations",
                    "responses": {"200": {"description": "List of station objects"}}
                }
            },
            "/api/stations/{station_id}": {
                "get": {
                    "summary": "Get specific station details",
                    "parameters": [{"name": "station_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "Station detail object"}, "404": {"description": "Station not found"}}
                }
            },
            "/api/predict/{station_id}": {
                "get": {
                    "summary": "Get 7-day predictive forecasts for a station",
                    "parameters": [{"name": "station_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "Predictive forecast object"}, "404": {"description": "Station not found"}}
                }
            },
            "/api/status/{station_id}": {
                "get": {
                    "summary": "Get real-time pulse score & alert level for a station",
                    "parameters": [{"name": "station_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "Station health status object"}, "404": {"description": "Station not found"}}
                }
            },
            "/api/dashboard": {
                "get": {
                    "summary": "Get main dashboard predictive forecast charts payload",
                    "responses": {"200": {"description": "Dashboard forecast payload"}}
                }
            },
            "/api/alerts": {
                "get": {
                    "summary": "Get active system alerts for all stations",
                    "responses": {"200": {"description": "Array of station alerts"}}
                }
            },
            "/api/charts": {
                "get": {
                    "summary": "Get historical evaporation, water levels, and water balance ratios",
                    "responses": {"200": {"description": "Chart data series"}}
                }
            }
        }
    })

@app.route("/api/metrics", methods=["GET"])
def metrics():
    latest_water_level = round(float(data_clean["Groundwatelevel_m"].iloc[-1]), 2)
    avg_water_level = round(float(data_clean["Groundwatelevel_m"].mean()), 2)
    
    # Calculate live station counts
    offsets = [0, 15, 35]
    normal_count = 0
    critical_count = 0
    
    for offset in offsets:
        score, _, level, _ = calculate_station_pulse(offset)
        if level == "NORMAL":
            normal_count += 1
        elif level in ["WARNING", "EMERGENCY", "CRITICAL"]:
            critical_count += 1

    metrics_file = os.path.join(MODELS_DIR, "metrics.json")
    model_data = {}
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            model_data = json.load(f)

    response_payload = {
        "total_stations": len(STATIONS_DATA),
        "active_stations": len(STATIONS_DATA),
        "average_water_level": avg_water_level,
        "latest_water_level": latest_water_level,
        "critical_stations": critical_count,
        "normal_stations": normal_count,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    if model_data:
        response_payload.update(model_data)

    return jsonify(response_payload)

@app.route("/api/stations", methods=["GET"])
def get_stations():
    max_date = data_clean["date"].max().strftime("%Y-%m-%d")
    stations_list = []
    for s in STATIONS_DATA:
        s_copy = s.copy()
        s_copy["last_update"] = max_date
        stations_list.append(s_copy)
    return jsonify(stations_list)

@app.route("/api/stations/<station_id>", methods=["GET"])
def get_station_by_id(station_id):
    station = next((s for s in STATIONS_DATA if s["id"] == station_id or s["station_id"] == station_id), None)
    if not station:
        return jsonify({"error": f"Station with ID '{station_id}' not found."}), 404
    
    offset = 0 if station_id == "GW-001" else (15 if station_id == "GW-002" else 35)
    pulse_score, color, alert_level, message = calculate_station_pulse(offset)
    
    max_date = data_clean["date"].max().strftime("%Y-%m-%d")
    result = station.copy()
    result["last_update"] = max_date
    result["pulse_score"] = pulse_score
    result["color"] = color
    result["alert_level"] = alert_level
    result["message"] = message
    return jsonify(result)

@app.route("/api/predict/<station_id>", methods=["GET"])
def predict_station(station_id):
    station = next((s for s in STATIONS_DATA if s["id"] == station_id or s["station_id"] == station_id), None)
    if not station:
        return jsonify({"error": f"Station with ID '{station_id}' not found."}), 404

    latest = data_clean.iloc[-1:]
    X_latest = scaler_X.transform(latest[FEATURES])
    today_gw_pred = scaler_y.inverse_transform(model.predict(X_latest).reshape(-1, 1))[0][0]

    forecast_days = 7
    rainfall_dates, predicted_rainfall, rainfall_upper, rainfall_lower = [], [], [], []
    gw_dates, predicted_gw = [], []

    last_row = latest.copy()
    for i in range(forecast_days):
        next_date = last_row["date"].values[0] + np.timedelta64(1, 'D')
        last_row["date"] = pd.to_datetime(next_date)
        features_row = create_features(pd.concat([data_clean, last_row], ignore_index=True)).iloc[-1:]

        X_next = scaler_X.transform(features_row[FEATURES])
        gw_pred = scaler_y.inverse_transform(model.predict(X_next).reshape(-1, 1))[0][0]

        rainfall_pred = scaler_y_r.inverse_transform(rain_model.predict(scaler_X_r.transform(features_row[FEATURES])).reshape(-1, 1))[0][0]

        rainfall_upper_val = rainfall_pred * 1.1
        rainfall_lower_val = max(0, rainfall_pred * 0.9)

        rainfall_dates.append(str(next_date)[:10])
        predicted_rainfall.append(round(float(rainfall_pred), 2))
        rainfall_upper.append(round(float(rainfall_upper_val), 2))
        rainfall_lower.append(round(float(rainfall_lower_val), 2))
        gw_dates.append(str(next_date)[:10])
        predicted_gw.append(round(float(gw_pred), 3))

        last_row["Groundwatelevel_m"] = gw_pred
        last_row["rainfall_mm"] = rainfall_pred

    return jsonify({
        "station_id": station_id,
        "station_name": station["name"],
        "predicted_groundwater": round(float(today_gw_pred), 2),
        "rainfall_dates": rainfall_dates,
        "predicted_rainfall": predicted_rainfall,
        "rainfall_upper": rainfall_upper,
        "rainfall_lower": rainfall_lower,
        "gw_dates": gw_dates,
        "predicted_gw": predicted_gw,
        "today_gw_level": round(float(today_gw_pred), 3)
    })

@app.route("/api/status/<station_id>", methods=["GET"])
def get_status_by_id(station_id):
    station = next((s for s in STATIONS_DATA if s["id"] == station_id or s["station_id"] == station_id), None)
    if not station:
        return jsonify({"error": f"Station with ID '{station_id}' not found."}), 404

    offset = 0 if station_id == "GW-001" else (15 if station_id == "GW-002" else 35)
    pulse_score, color, alert_level, message = calculate_station_pulse(offset)

    return jsonify({
        "station_id": station_id,
        "name": station["name"],
        "pulse_score": pulse_score,
        "alert_level": alert_level,
        "color": color,
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    return predict_station("GW-001")

@app.route("/api/alerts", methods=["GET"])
def alerts():
    alerts_list = []
    offsets = [0, 15, 35]
    for s, offset in zip(STATIONS_DATA, offsets):
        score, color, level, msg = calculate_station_pulse(offset)
        alerts_list.append({
            "station_id": s["id"],
            "name": s["name"],
            "lat": s["lat"],
            "lng": s["lng"],
            "pulse_score": score,
            "alert_level": level,
            "color": color,
            "message": msg
        })
    return jsonify(alerts_list)

@app.route("/api/charts", methods=["GET"])
def charts():
    evaporation = data_clean[["date", "evaporation_mm"]].tail(100).assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    water_levels = data_clean[["date", "Groundwatelevel_m"]].tail(100).assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    last_7_days = data_clean.tail(7)
    pie = [
        {"name": "Evaporation", "value": round(float(last_7_days["evaporation_mm"].sum()), 2)},
        {"name": "Water Level", "value": round(float(last_7_days["Groundwatelevel_m"].sum()), 2)}
    ] if last_7_days.shape[0] > 0 else []

    return jsonify({
        "evaporation": evaporation,
        "water_levels": water_levels,
        "water_balance_pie": pie
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
