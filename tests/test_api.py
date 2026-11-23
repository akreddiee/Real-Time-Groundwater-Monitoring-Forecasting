import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test / and /health return status 200 and json payload"""
    for endpoint in ["/", "/health"]:
        rv = client.get(endpoint)
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["status"] == "ok"
        assert data["service"] == "Realtime Ground Water Level Monitoring API"
        assert "timestamp" in data

def test_docs_and_openapi(client):
    """Test /docs and /openapi.json return 200"""
    rv_docs = client.get("/docs")
    assert rv_docs.status_code == 200
    assert b"Aquapulse API Documentation" in rv_docs.data

    rv_json = client.get("/openapi.json")
    assert rv_json.status_code == 200
    spec = rv_json.get_json()
    assert spec["openapi"] == "3.0.0"
    assert "/api/metrics" in spec["paths"]

def test_metrics_endpoint(client):
    """Test /api/metrics returns system stats and ML evaluation data"""
    rv = client.get("/api/metrics")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["total_stations"] == 3
    assert data["active_stations"] == 3
    assert "average_water_level" in data
    assert "timestamp" in data

def test_stations_list(client):
    """Test /api/stations returns list of station objects"""
    rv = client.get("/api/stations")
    assert rv.status_code == 200
    data = rv.get_json()
    assert isinstance(data, list)
    assert len(data) == 3
    assert data[0]["id"] == "GW-001"

def test_station_detail(client):
    """Test /api/stations/<station_id> returns specific station details"""
    rv = client.get("/api/stations/GW-001")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["id"] == "GW-001"
    assert data["name"] == "Telangana Central Station"

    # Test non-existent station
    rv_fake = client.get("/api/stations/GW-999")
    assert rv_fake.status_code == 404

def test_station_prediction(client):
    """Test /api/predict/<station_id> returns 7-day predictions"""
    rv = client.get("/api/predict/GW-001")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["station_id"] == "GW-001"
    assert len(data["predicted_gw"]) == 7

def test_station_status(client):
    """Test /api/status/<station_id> returns health status"""
    rv = client.get("/api/status/GW-001")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["station_id"] == "GW-001"
    assert "pulse_score" in data
    assert "alert_level" in data

def test_dashboard_endpoint(client):
    """Test /api/dashboard returns 200"""
    rv = client.get("/api/dashboard")
    assert rv.status_code == 200

def test_alerts_endpoint(client):
    """Test /api/alerts returns 200 and list of alerts"""
    rv = client.get("/api/alerts")
    assert rv.status_code == 200
    data = rv.get_json()
    assert isinstance(data, list)
    assert len(data) == 3

def test_charts_endpoint(client):
    """Test /api/charts returns 200"""
    rv = client.get("/api/charts")
    assert rv.status_code == 200
