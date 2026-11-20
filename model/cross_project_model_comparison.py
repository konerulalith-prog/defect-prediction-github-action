
from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from preprocess_relink import prepare_dataset


# --------------------------------------------------
# 1. Experiment settings
# --------------------------------------------------

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "cross_project_model_comparison.csv"
)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = prepare_dataset()

feature_columns = [
    col for col in df.columns
    if col not in ["isDefective", "project"]
]

projects = sorted(df["project"].unique())

print("=" * 85)
print("CROSS-PROJECT MODEL COMPARISON — RELINK")
print("=" * 85)

print(f"Total samples: {len(df)}")
print(f"Features: {len(feature_columns)}")
print(f"Projects: {projects}")


# --------------------------------------------------
# 3. Configure models using previous experiments
# --------------------------------------------------

def create_models():

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        verbosity=-1,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    ensemble = VotingClassifier(
        estimators=[
            ("xgboost", clone(xgb)),
            ("lightgbm", clone(lgbm)),
        ],
        voting="soft",
        weights=[1, 1],
        n_jobs=1,
    )

    return {
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "XGBoost + LightGBM Average": ensemble,
    }


# --------------------------------------------------
# 4. Leave-one-project-out evaluation
# --------------------------------------------------

results = []

for test_project in projects:

    train_df = df[df["project"] != test_project].copy()
    test_df = df[df["project"] == test_project].copy()

    X_train = train_df[feature_columns].astype(float)
    y_train = train_df["isDefective"].astype(int)

    X_test = test_df[feature_columns].astype(float)
    y_test = test_df["isDefective"].astype(int)

    print("\n" + "=" * 85)
    print(f"TEST PROJECT: {test_project}")
    print("=" * 85)

    print(f"Training projects: {sorted(train_df['project'].unique())}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training defect rate: {y_train.mean():.3f}")
    print(f"Testing defect rate: {y_test.mean():.3f}")

    for name, model in create_models().items():

        print(f"\nEvaluating {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "Test Project": test_project,
            "Model": name,
            "Training Samples": len(X_train),
            "Testing Samples": len(X_test),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# --------------------------------------------------
# 5. Summarize results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 110)
print("COMPLETE CROSS-PROJECT RESULTS")
print("=" * 110)

print(results_df.round(4).to_string(index=False))


# --------------------------------------------------
# 6. Average performance across projects
# --------------------------------------------------

summary = (
    results_df.groupby("Model")[
        ["Accuracy", "Precision", "Recall", "F1"]
    ]
    .mean()
    .reset_index()
    .sort_values("F1", ascending=False)
)

print("\n" + "=" * 85)
print("AVERAGE PERFORMANCE ACROSS ALL THREE PROJECTS")
print("=" * 85)

print(summary.round(4).to_string(index=False))


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nCross-project experiment completed.")
