
import json
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"

MODEL_PATH = MODEL_DIR / "xgboost_defect_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def load_model():
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata


def predict_file_metrics(csv_path):
    model, metadata = load_model()

    features = metadata["features"]
    medians = metadata["imputation_values"]
    threshold = metadata["decision_threshold"]

    df = pd.read_csv(csv_path)

    missing_columns = [col for col in features if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required metrics: {missing_columns}"
        )

    X = df[features].copy()

    for column in features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X.loc[X["AvgLine"] > 1000, "AvgLine"] = pd.NA
    X = X.fillna(medians)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = pd.DataFrame({
        "defect_probability": probabilities,
        "predicted_class": predictions,
        "risk": [
            "High" if p >= threshold else "Low"
            for p in probabilities
        ]
    })

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python model/predict.py <metrics.csv>")
        raise SystemExit(1)

    results = predict_file_metrics(sys.argv[1])

    print("\nPrediction Results:")
    print(results.to_string(index=False))
