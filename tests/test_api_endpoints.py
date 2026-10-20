from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_health_check():
    res = client.get("/healthstatus")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_api_predict_valid_input():
    payload = {
        "age": 57.0, "sex": 1, "cp": 4, "trestbps": 140.0, "chol": 241.0,
        "fbs": 0, "restecg": 0, "thalach": 123.0, "exang": 1, "oldpeak": 0.2,
        "slope": 2, "ca": 0.0, "thal": 7.0
    }
    res = client.post("/evaluate-risk", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert "heart_disease_risk" in data
    assert "confidence" in data
    assert data["prediction"] in [0, 1]