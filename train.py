from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
MLFLOW_DB = ROOT / "mlflow.db"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Feature and target configuration
# --------------------------------------------------

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET = "Machine failure"

CATEGORICAL_FEATURES = ["Type"]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


# --------------------------------------------------
# 3. Load and validate dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)

required_columns = FEATURES + [TARGET]

missing_columns = set(required_columns) - set(data.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

X = data[FEATURES]
y = data[TARGET].astype(int)

if y.nunique() != 2:
    raise ValueError("Target must contain both 0 and 1.")

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")


# --------------------------------------------------
# 4. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())


# --------------------------------------------------
# 5. Build preprocessing pipeline
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
        (
            "numerical",
            "passthrough",
            NUMERIC_FEATURES,
        ),
    ]
)


# --------------------------------------------------
# 6. Configure Random Forest
# --------------------------------------------------

classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# --------------------------------------------------
# 7. Create complete ML pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)


# --------------------------------------------------
# 8. Configure MLflow
# --------------------------------------------------

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment("predictive-maintenance")


# --------------------------------------------------
# 9. Train and evaluate model
# --------------------------------------------------

with mlflow.start_run(run_name="Random-Forest"):

    print("\nStarting Random Forest training...")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # Generate predictions
    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
    }

    # Log model parameters
    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42,
        "test_size": 0.20,
    })

    mlflow.log_param("dataset_rows", len(data))
    mlflow.log_metrics(metrics)

    # Log trained model to MLflow
    mlflow.sklearn.log_model(
    sk_model=pipeline,
    name="random_forest_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # Display metrics
    print("\n========== MODEL EVALUATION ==========")

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Display confusion matrix
    print("\n========== CONFUSION MATRIX ==========")
    print(confusion_matrix(y_test, predictions))

    # Display classification report
    print("\n========== CLASSIFICATION REPORT ==========")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # Save trained model locally
    joblib.dump(pipeline, MODEL_PATH)

    print("\nModel logged to MLflow.")
    print("Model saved to:", MODEL_PATH)


print("\n========== TRAINING COMPLETED ==========")
