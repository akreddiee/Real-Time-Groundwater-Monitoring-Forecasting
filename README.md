# 🌊 Aquapulse — Realtime Groundwater Level Monitoring & Predictive System

![React](https://img.shields.io/badge/React-18.2-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-7.1-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Leaflet](https://img.shields.io/badge/Leaflet-1.9-199900?style=for-the-badge&logo=leaflet&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-4.1-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**Aquapulse** is an end-to-end hydro-informatics platform designed for real-time groundwater monitoring, time-series forecasting, and spatial telemetry management. The system processes environmental telemetry data (groundwater levels, rainfall, temperature, humidity), calculates thermodynamic evaporation rates, and utilizes **Gradient Boosting Regressors** and **Facebook Prophet** to forecast groundwater availability up to 7 days in advance.

---

## 📌 Key Features

- 🛰️ **Interactive Spatial GIS Map**: Dynamic Leaflet GIS mapping displaying monitoring stations with real-time, color-coded health indicators based on calculated pulse scores.
- 🔮 **Predictive Machine Learning**: 7-day future groundwater level and rainfall predictions with confidence bands ($90\%$ interval) using Gradient Boosting & Prophet time-series models.
- ⚡ **Dynamic Station Pulse & Health Scoring**: Automatic Z-score standard deviation analysis converting raw water level changes into intuitive pulse scores ($0-100$) and multi-tier alerts (`NORMAL`, `WATCH`, `ADVISORY`, `WARNING`, `EMERGENCY`).
- 📊 **Advanced Analytics & Charts**: Interactive Plotly and Recharts visualizations for water balance breakdown (7-day average water level vs rainfall accumulation vs evaporation loss).
- ⚙️ **Production-Ready REST API**: Modular Python Flask backend serving lightweight JSON payloads with automated model artifact caching.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        CSV[Telemetry Sensors / CSV Data]
        DB[(PostgreSQL / SQLite Database)]
    end

    subgraph Feature Engineering & ML Engine
        FE[Thermodynamic Evaporation & Rolling Feature Extraction]
        GBR[Gradient Boosting Regressors]
        Prophet[Facebook Prophet Time-Series Model]
        ZScore[Pulse Score & Health Alert Evaluator]
    end

    subgraph Flask REST API Backend
        API1[/api/dashboard]
        API2[/api/alerts]
        API3[/api/stations]
        API4[/api/charts]
        API5[/api/metrics]
    end

    subgraph React Frontend SPA
        Leaflet[Interactive Leaflet GIS Map]
        ForecastChart[Plotly 7-Day Forecast Charts]
        PulseGauge[Station Pulse Score Gauge]
        MetricsUI[ML Evaluation Metrics UI]
    end

    CSV --> FE
    DB --> FE
    FE --> GBR & Prophet & ZScore
    GBR & Prophet & ZScore --> API1 & API2 & API3 & API4 & API5
    API1 & API2 & API3 & API4 & API5 --> React Frontend SPA
    React Frontend SPA --> Leaflet & ForecastChart & PulseGauge & MetricsUI
```

---

## 📁 Repository Structure

```
Realtime-Ground-water-level-monitoring/
├── app.py                   # Flask REST API server
├── train_model.py           # ML Model training & serialization script
├── model.py                 # Offline EDA & standalone plotting script
├── Telemetry_data.csv       # Environmental telemetry dataset
├── requirements.txt         # Python backend dependencies
├── models/                  # Serialized ML model artifacts (.joblib, .json)
│   ├── gw_model.joblib
│   ├── rain_model.joblib
│   ├── scalers.joblib
│   └── metrics.json
├── src/                     # React Single Page Application (Vite)
│   ├── App.jsx              # Main React Router setup
│   ├── Home.jsx             # Interactive Leaflet GIS Map & Navigation
│   ├── Dashboard.jsx        # 7-day predictive forecast charts (Plotly)
│   ├── Stations.jsx         # Station management list & real-time telemetry
│   ├── Alerts.jsx           # Multi-level advisory & alert center
│   ├── Charts.jsx           # Water balance & evaporation analytics
│   ├── ModelMetrics.jsx     # ML Model evaluation & feature importance UI
│   └── Login.jsx            # User authentication interface
├── public/                  # Static web assets & station logos
├── tests/                   # Automated pytest backend test suite
│   └── test_api.py
├── Dockerfile               # Backend container configuration
├── docker-compose.yml       # Full-stack Docker orchestration
└── README.md                # Project documentation
```

---

## 🔌 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/dashboard` | `GET` | Returns today's predicted groundwater level & 7-day rainfall/groundwater forecast |
| `/api/alerts` | `GET` | Returns pulse scores, alert severity levels, and advisory messages for all stations |
| `/api/stations` | `GET` | Returns spatial metadata, status, coordinates, and telemetry timestamps for stations |
| `/api/charts` | `GET` | Returns 100-day evaporation, water level trends, and 7-day water balance ratios |
| `/api/metrics` | `GET` | Returns ML evaluation metrics ($R^2$, RMSE, MAE) and feature importance rankings |

---

## 🚀 Quick Start & Installation Guide

### Prerequisites
- **Node.js** (v18 or higher) & **npm**
- **Python** (v3.10 or higher) & **pip**
- *(Optional)* **Docker & Docker Compose**

---

### Method 1: Local Manual Setup

#### 1. Backend Setup (Flask API)
```bash
# Clone the repository
git clone https://github.com/Pardhu643/Realtime-Ground-water-level-monitoring.git
cd Realtime-Ground-water-level-monitoring

# Create a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train models & generate artifacts (Optional if models/ already exists)
python train_model.py

# Start Flask backend API
python app.py
```
*Backend runs on `http://127.0.0.1:5000`*

#### 2. Frontend Setup (React SPA)
```bash
# In a new terminal window at the project root
npm install

# Start Vite development server
npm run dev
```
*Frontend runs on `http://localhost:5173`*

---

### Method 2: Single-Command Docker Deployment

```bash
docker-compose up --build
```
This builds and launches both the Flask backend server and Vite frontend inside containerized environments.

---

## 🧪 Running Automated Tests

Run the backend test suite using `pytest`:

```bash
pytest tests/
```

---

## 🤖 Machine Learning Model Overview

The system extracts **19 predictive features** from raw environmental telemetry:
- **Thermodynamic Evaporation**: Calculated using ambient temperature and relative humidity ($E = 0.0023 \cdot (T + 17.8) \cdot \sqrt{100 - H}$).
- **Rolling Windows**: 7, 14, and 30-day rolling means for groundwater and accumulation totals for rainfall.
- **Lag Features**: 1, 7, and 14-day historical shifts.
- **Cyclical Temporal Encoding**: Sine and Cosine transformations of the month (`month_sin`, `month_cos`) and day of year.

### Model Evaluation Performance

| Target Metric | Model Algorithm | $R^2$ Score | RMSE | MAE |
| :--- | :--- | :---: | :---: | :---: |
| **Groundwater Level ($m$)** | Gradient Boosting Regressor | `> 0.92` | `~ 0.04m` | `~ 0.03m` |
| **Rainfall Accumulation ($mm$)** | Gradient Boosting Regressor | `> 0.85` | `~ 1.2mm` | `~ 0.8mm` |

---

## 👨‍💻 Author & Contact

Developed by **Pardhu** — [GitHub Repository](https://github.com/Pardhu643/Realtime-Ground-water-level-monitoring)