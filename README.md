# Predictive Maintenance with MLOps

A small end-to-end machine-learning project that predicts machine failure from operating sensor readings.

## Dataset we will use

**AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

- Official source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- DOI: https://doi.org/10.24432/C5HS5C
- Original size: 10,000 rows and 14 columns; approximately 510 KB.
- License: CC BY 4.0.
- The dataset is synthetic and was designed to resemble industrial predictive-maintenance data.

The project uses six input columns: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`. The target is `Machine failure`.

We intentionally exclude `UDI`, `Product ID`, and the failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`). The failure-mode columns reveal information about the target and would cause target leakage.

### Dataset included in this starter ZIP

`data/dataset.csv` is a small, locally generated demo dataset (600 rows) so the project can be tested immediately without a download. It is **not** the official UCI records.

To replace it with a stratified 3,000-row sample from the official UCI dataset, run:

```bash
python train.py --download-data
```

The downloaded sample is saved to the same `data/dataset.csv` path. You can also use all 10,000 original records by changing the `sample_size` in `download_small_dataset()` to `10000`.

## Tech stack

- Pandas and NumPy: data handling
- Scikit-learn and XGBoost: model training
- MLflow: experiment tracking and model artifacts
- Joblib: local model persistence
- FastAPI: prediction API
- Docker: containerization
- GitHub Actions: automated training and tests

No Kafka or Airflow is used.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

Optional: use the official UCI dataset sample:
```bash
python train.py --download-data
```

Train both models, log experiments to MLflow, and save the best pipeline:
```bash
python train.py
```

Start the API:
```bash
uvicorn main:app --reload
```

Open API documentation:
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

Run tests:
```bash
pytest -q
```

Open the MLflow UI in a second terminal:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```
Then visit http://127.0.0.1:5000.

## Example prediction request

POST `/predict`:

```json
{
  "type": "M",
  "air_temperature": 300.0,
  "process_temperature": 310.0,
  "rotational_speed": 1500,
  "torque": 40.0,
  "tool_wear": 100
}
```

## Docker

Train first so `models/model.joblib` exists:

```bash
python train.py
docker build -t predictive-maintenance .
docker run --rm -p 8000:8000 predictive-maintenance
```

## Project workflow

1. Read the CSV dataset.
2. Split the data into training and testing sets.
3. Train Random Forest and XGBoost pipelines.
4. Log parameters and evaluation metrics in MLflow.
5. Select the model with the best F1 score.
6. Save the selected pipeline as `models/model.joblib`.
7. Load that model in FastAPI and serve predictions.
8. GitHub Actions trains the model and runs tests on pushes and pull requests.

## Important note

This is a learning/portfolio project, not a certified industrial safety system. The UCI dataset is synthetic, so evaluation results should not be treated as real-world machine failure guarantees.
