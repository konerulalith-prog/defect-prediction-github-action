
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"

MODEL_PATH = MODEL_DIR / "xgboost_defect_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def explain_predictions(csv_path):
    model = joblib.load(MODEL_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    features = metadata["features"]
    medians = metadata["imputation_values"]

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Input CSV is empty")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Match the saved training preprocessing rule.
    X.loc[X["AvgLine"] > 1000, "AvgLine"] = np.nan

    X = X.fillna(pd.Series(medians))
    X = X.astype(float)

    if X.isna().any().any():
        raise ValueError("Unresolved missing feature values")

    probabilities = model.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(model)
    explanations = explainer(X)

    if explanations.values.ndim != 2:
        raise ValueError("Unexpected SHAP output shape")

    for row_index, probability in enumerate(probabilities):
        print(f"\nSample {row_index + 1}")
        print(f"Defect probability: {probability:.4f}")

        contributions = sorted(
            zip(features, explanations.values[row_index]),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        print("Top 5 SHAP contributions:")

        for feature, value in contributions[:5]:
            direction = (
                "increases" if value > 0 else "decreases"
            )
            print(
                f"  {feature}: {value:+.4f} "
                f"({direction} model output)"
            )

    print(
        "\nNote: SHAP contributions may be in raw model "
        "output units rather than probability units."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()

    explain_predictions(args.csv_path)
