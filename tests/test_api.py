
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["api_status"] == "healthy"
    assert response.json()["model_status"] == "loaded"


def test_prediction_endpoint():
    payload = {
        "machine_type": "M",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 108,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["machine_failure"] in [0, 1]
    assert result["prediction"] in [
        "Failure detected",
        "No failure detected",
    ]
    assert 0 <= result["failure_probability"] <= 1


def test_prediction_invalid_input():
    payload = {
        "machine_type": "INVALID",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 108,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
