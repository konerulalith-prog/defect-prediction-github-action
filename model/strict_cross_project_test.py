
from pathlib import Path

import pandas as pd

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

from preprocess_relink import load_relink_arff


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "ReLink"

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "strict_cross_project_results.csv"
)


# --------------------------------------------------
# 1. Load raw datasets without project-level imputation
# --------------------------------------------------

def load_raw_dataset():

    datasets = []

    for file in sorted(DATA_FOLDER.glob("*.arff")):

        df = load_relink_arff(file)

        for column in df.columns:
            if column != "isDefective":
                df[column] = pd.to_numeric(
                    df[column], errors="coerce"
                )

        df["isDefective"] = df["isDefective"].map({
            "FALSE": 0,
            "TRUE": 1,
        })

        df["project"] = file.stem

        datasets.append(df)

    return pd.concat(datasets, ignore_index=True)


# --------------------------------------------------
# 2. Configure models
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
            ("xgboost", xgb),
            ("lightgbm", lgbm),
        ],
        voting="soft",
        weights=[1, 1],
        n_jobs=1,
    )

    return {
        "XGBoost": xgb,
        
    }


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

df = load_raw_dataset()

feature_columns = [
    col for col in df.columns
    if col not in ["isDefective", "project"]
]

projects = sorted(df["project"].unique())

print("=" * 85)
print("STRICT CROSS-PROJECT EVALUATION")
print("=" * 85)

print("Total samples:", len(df))
print("Features:", len(feature_columns))
print("Projects:", projects)


# --------------------------------------------------
# 4. Leave-one-project-out evaluation
# --------------------------------------------------

results = []

for test_project in projects:

    train_df = df[df["project"] != test_project].copy()
    test_df = df[df["project"] == test_project].copy()

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    y_train = train_df["isDefective"].astype(int)
    y_test = test_df["isDefective"].astype(int)

    # Remove invalid AvgLine values in both sets.
    X_train.loc[X_train["AvgLine"] > 1000, "AvgLine"] = pd.NA
    X_test.loc[X_test["AvgLine"] > 1000, "AvgLine"] = pd.NA

    # Fit imputation using training data only.
    avgline_median = X_train["AvgLine"].median()

    X_train["AvgLine"] = X_train["AvgLine"].fillna(avgline_median)
    X_test["AvgLine"] = X_test["AvgLine"].fillna(avgline_median)

    # Handle any remaining missing feature values.
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    print("\n" + "=" * 85)
    print("TEST PROJECT:", test_project)
    print("=" * 85)

    print("Training projects:", sorted(train_df["project"].unique()))
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training AvgLine median:", avgline_median)
    print("Remaining missing values:", X_train.isnull().sum().sum() +
          X_test.isnull().sum().sum())

    for name, model in create_models().items():

        print("\nEvaluating:", name)

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

summary = (
    results_df.groupby("Model")[
        ["Accuracy", "Precision", "Recall", "F1"]
    ]
    .mean()
    .reset_index()
    .sort_values("F1", ascending=False)
)

print("\n" + "=" * 100)
print("STRICT CROSS-PROJECT RESULTS")
print("=" * 100)

print(results_df.round(4).to_string(index=False))

print("\nAVERAGE PERFORMANCE ACROSS PROJECTS")
print(summary.round(4).to_string(index=False))


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nExperiment completed.")
