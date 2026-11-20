
import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT / "model" / "saved_model" / "model_metadata.json"


def validate_metrics(csv_path):
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    required = metadata["features"]
    df = pd.read_csv(csv_path)

    print(f"Required features: {len(required)}")
    print(f"CSV rows: {len(df)}")

    if df.empty:
        raise ValueError("CSV contains no data rows")

    if len(required) != 26 or len(set(required)) != 26:
        raise ValueError("Unexpected model feature schema")

    if len(df.columns) != len(set(df.columns)):
        raise ValueError("Duplicate CSV column names")

    missing = [col for col in required if col not in df.columns]

    if missing:
        print("\nMissing required features:")
        for col in missing:
            print(f" - {col}")
        raise ValueError(
            f"Validation failed: {len(missing)} features missing"
        )

    numeric = df[required].apply(pd.to_numeric, errors="coerce")

    # Reject infinite values and invalid non-numeric entries.
    import numpy as np
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite metric values detected")

    invalid = numeric.isna() & df[required].notna()
    if invalid.any().any():
        bad = invalid.columns[invalid.any()].tolist()
        raise ValueError(f"Non-numeric values in: {bad}")

    # Missing values may be imputed by the predictor,
    # but every feature must have a defined training median.
    medians = metadata["imputation_values"]
    if any(col not in medians for col in required):
        raise ValueError("Missing imputation metadata")

    print("\nAll 26 required feature columns are present.")
    print("Numeric validation passed.")
    print(
        "\nWARNING: Schema validation does not establish "
        "ReLink metric-definition compatibility."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()

    validate_metrics(args.csv_path)
