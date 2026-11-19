
from pathlib import Path

import joblib
import pandas as pd


# Locate the saved model
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"


def test_model_exists():
    assert MODEL_PATH.exists()


def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    # Sample machine sensor data
    sample = pd.DataFrame([{
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 108,
    }])

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_model_probability():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 108,
    }])

    probability = model.predict_proba(sample)[0][1]

    assert 0 <= probability <= 1
