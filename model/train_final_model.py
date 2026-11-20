
from pathlib import Path
import json
import joblib
import pandas as pd

from xgboost import XGBClassifier
from preprocess_relink import load_relink_arff


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "ReLink"
MODEL_FOLDER = PROJECT_ROOT / "model" / "saved_model"

MODEL_PATH = MODEL_FOLDER / "xgboost_defect_model.pkl"
METADATA_PATH = MODEL_FOLDER / "model_metadata.json"


# 1. Load all ReLink datasets
datasets = []

for file in sorted(DATA_FOLDER.glob("*.arff")):
    df = load_relink_arff(file)

    for col in df.columns:
        if col != "isDefective":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["isDefective"] = df["isDefective"].map({
        "FALSE": 0,
        "TRUE": 1,
    })

    df["project"] = file.stem
    datasets.append(df)

data = pd.concat(datasets, ignore_index=True)


# 2. Select features and target
feature_columns = [
    col for col in data.columns
    if col not in ["isDefective", "project"]
]

X = data[feature_columns].copy()
y = data["isDefective"].astype(int)


# 3. Handle invalid AvgLine values
X.loc[X["AvgLine"] > 1000, "AvgLine"] = pd.NA

# Calculate imputation values from training data
imputation_values = X.median()

X = X.fillna(imputation_values).astype(float)


# 4. Configure selected XGBoost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=1,
)


# 5. Train on all available ReLink samples
model.fit(X, y)


# 6. Save model and metadata
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

metadata = {
    "model": "XGBoost",
    "dataset": "ReLink",
    "samples": len(data),
    "features": feature_columns,
    "target": "isDefective",
    "positive_class": 1,
    "decision_threshold": 0.50,
    "imputation_values": imputation_values.to_dict(),
    "invalid_avgline_rule": "Values greater than 1000 treated as missing",
    "parameters": {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
    },
}

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("=" * 65)
print("FINAL XGBOOST MODEL TRAINED AND SAVED")
print("=" * 65)
print("Samples:", len(data))
print("Features:", len(feature_columns))
print("Defective files:", int(y.sum()))
print("Non-defective files:", int((y == 0).sum()))
print("Model saved:", MODEL_PATH)
print("Metadata saved:", METADATA_PATH)
