
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "saved_model"


def generate_report(input_csv, output_json):
    model = joblib.load(
        MODEL_DIR / "xgboost_defect_model.pkl"
    )

    with open(
        MODEL_DIR / "model_metadata.json",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    features = metadata["features"]
    medians = metadata["imputation_values"]
    threshold = float(
        metadata.get("decision_threshold", 0.5)
    )

    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError("Input CSV is empty")

    missing = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    X = df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace([np.inf, -np.inf], np.nan)

    if "AvgLine" in X.columns:
        X.loc[X["AvgLine"] > 1000, "AvgLine"] = np.nan

    X = X.fillna(pd.Series(medians))
    X = X.astype(float)

    if X.isna().any().any():
        raise ValueError(
            "Missing values remain after imputation"
        )

    probabilities = model.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(model)
    explanations = explainer(X)

    if explanations.values.ndim != 2:
        raise ValueError("Unexpected SHAP output shape")

    results = []

    for i, probability in enumerate(probabilities):
        contributions = sorted(
            zip(features, explanations.values[i]),
            key=lambda item: abs(item[1]),
            reverse=True
        )

        top_features = [
            {
                "feature": name,
                "shap_value": round(float(value), 6),
                "direction": (
                    "increases_risk"
                    if value > 0
                    else "decreases_risk"
                    if value < 0
                    else "neutral"
                )
            }
            for name, value in contributions[:5]
        ]

        results.append({
            "sample": i + 1,
            "defect_probability": round(
                float(probability), 6
            ),
            "predicted_class": int(
                probability >= threshold
            ),
            "risk": (
                "High"
                if probability >= threshold
                else "Low"
            ),
            "top_features": top_features
        })

    report = {
        "model": "XGBoost",
        "decision_threshold": threshold,
        "feature_count": len(features),
        "metric_schema": "ReLink",
        "validation_status": (
            "Experimental: source metrics must be "
            "verified as ReLink-compatible"
        ),
        "results": results
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(f"Report generated: {output_path}")
    print(f"Samples analyzed: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_csv",
        help="CSV containing all 26 ReLink features"
    )

    parser.add_argument(
        "--output",
        default="reports/defect_report.json"
    )

    args = parser.parse_args()

    generate_report(args.input_csv, args.output)
