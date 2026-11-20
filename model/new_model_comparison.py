
from pathlib import Path

import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from preprocess_relink import prepare_dataset


# --------------------------------------------------
# 1. Reproducible experiment settings
# --------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "new_model_comparison_results.csv"


# --------------------------------------------------
# 2. Load the prepared ReLink dataset
# --------------------------------------------------

df = prepare_dataset()

feature_columns = [
    column
    for column in df.columns
    if column not in ["isDefective", "project"]
]

X = df[feature_columns].astype(float)
y = df["isDefective"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print("=" * 75)
print("NEW MODEL FAMILY COMPARISON — RELINK DATASET")
print("=" * 75)
print(f"Total samples: {len(df)}")
print(f"Features: {len(feature_columns)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Training defect rate: {y_train.mean():.3f}")
print(f"Test defect rate: {y_test.mean():.3f}")
print("Positive class: defective (1)")
print("Decision threshold: 0.50")
print("Cross-validation: 5-fold stratified CV on training data")


# --------------------------------------------------
# 3. Define additional model families
# --------------------------------------------------

models = {
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,
        random_state=RANDOM_STATE,
    ),

    "Gaussian Naive Bayes": GaussianNB(),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),

    "CatBoost": CatBoostClassifier(
        iterations=200,
        depth=5,
        learning_rate=0.05,
        loss_function="Logloss",
        verbose=False,
        allow_writing_files=False,
        random_seed=RANDOM_STATE,
        thread_count=1,
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        verbosity=-1,
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
}


# --------------------------------------------------
# 4. Cross-validate, train and evaluate each model
# --------------------------------------------------

results = []

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # Cross-validation uses training data only.
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=1,
    )

    # Fit on the full training split.
    model.fit(X_train, y_train)

    # Evaluate once on the held-out test split.
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

    result = {
        "Model": name,
        "CV F1 Mean": cv_scores.mean(),
        "CV F1 Std": cv_scores.std(),
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    results.append(result)

    print(f"  CV F1:          {cv_scores.mean():.4f}")
    print(f"  Test accuracy:  {accuracy:.4f}")
    print(f"  Test precision: {precision:.4f}")
    print(f"  Test recall:    {recall:.4f}")
    print(f"  Test F1:        {f1:.4f}")


# --------------------------------------------------
# 5. Display and save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

# Rank by cross-validation F1, not test-set F1.
results_df = results_df.sort_values(
    by="CV F1 Mean",
    ascending=False,
).reset_index(drop=True)

print("\n" + "=" * 100)
print("FINAL RESULTS — RANKED BY CROSS-VALIDATION F1")
print("=" * 100)

print(
    results_df.round(4).to_string(index=False)
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nExperiment completed.")
