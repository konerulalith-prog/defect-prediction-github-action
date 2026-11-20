
from pathlib import Path

import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
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


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "model_averaging_results.csv"


# 1. Load the ReLink dataset
df = prepare_dataset()

feature_columns = [
    col for col in df.columns
    if col not in ["isDefective", "project"]
]

X = df[feature_columns].astype(float)
y = df["isDefective"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# 2. Configure the two models
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

# Average the two models' predicted probabilities equally.
ensemble = VotingClassifier(
    estimators=[
        ("xgboost", xgb),
        ("lightgbm", lgbm),
    ],
    voting="soft",
    weights=[1, 1],
    n_jobs=1,
)

models = {
    "XGBoost": xgb,
    "LightGBM": lgbm,
    "XGBoost + LightGBM Average": ensemble,
}


# 3. Evaluate models
results = []

print("=" * 75)
print("XGBOOST + LIGHTGBM MODEL AVERAGING")
print("=" * 75)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {len(feature_columns)}")
print("Decision threshold: 0.50")
print("Positive class: defective (1)")
print("Cross-validation: 5-fold, training data only")

for name, model in models.items():
    print(f"\nEvaluating {name}...")

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="f1",
        cv=cv,
        n_jobs=1,
    )

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
    })

    print(f"  CV F1:          {cv_scores.mean():.4f}")
    print(f"  Test accuracy:  {accuracy:.4f}")
    print(f"  Test precision: {precision:.4f}")
    print(f"  Test recall:    {recall:.4f}")
    print(f"  Test F1:        {f1:.4f}")
    print(f"  False negatives: {fn}")


# 4. Display and save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "CV F1 Mean", ascending=False
)

print("\n" + "=" * 100)
print("RESULTS — RANKED BY CROSS-VALIDATION F1")
print("=" * 100)
print(results_df.round(4).to_string(index=False))

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nResults saved to: {OUTPUT_FILE}")
