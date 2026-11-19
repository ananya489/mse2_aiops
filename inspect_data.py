
import pandas as pd
from pathlib import Path

# Get project directory
BASE_DIR = Path(__file__).resolve().parent

# Load dataset
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
df = pd.read_csv(DATA_PATH)

# 1. Dataset dimensions
print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 3. First five records
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# 4. Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# 5. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 6. Duplicate records
print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

# 7. Target distribution
print("\n========== TARGET DISTRIBUTION ==========")
print(df["Machine failure"].value_counts())

# 8. Target distribution in percentage
print("\n========== TARGET PERCENTAGE ==========")
print(df["Machine failure"].value_counts(normalize=True) * 100)

# 9. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== INSPECTION COMPLETED ==========")
