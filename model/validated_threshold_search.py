
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.base import clone

from preprocess_relink import prepare_dataset


# --------------------------------------------------
# 1. Load data and preserve the held-out test set
# --------------------------------------------------

df = prepare_dataset()

feature_columns = [
    col for col in df.columns
    if col not in ["isDefective", "project"]
]

X = df[feature_columns]
y = df["isDefective"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# 2. Candidate Gradient Boosting configurations
# --------------------------------------------------

candidates = {
    "GB baseline": {
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_leaf": 1,
        "n_estimators": 100,
    },
    "GB current best": {
        "learning_rate": 0.1,
        "max_depth": 4,
        "min_samples_leaf": 2,
        "n_estimators": 150,
    },
    "GB simpler": {
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 3,
        "n_estimators": 100,
    },
    "GB deeper": {
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 3,
        "n_estimators": 200,
    },
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

thresholds = [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70,
]

candidate_results = []


# --------------------------------------------------
# 3. Generate out-of-fold training probabilities
# --------------------------------------------------

for name, params in candidates.items():

    model = GradientBoostingClassifier(
        **params,
        random_state=42,
    )

    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    for threshold in thresholds:

        oof_predictions = (
            oof_probabilities >= threshold
        ).astype(int)

        candidate_results.append({
            "Model": name,
            "Parameters": params,
            "Threshold": threshold,
            "CV Precision": precision_score(
                y_train,
                oof_predictions,
                zero_division=0,
            ),
            "CV Recall": recall_score(
                y_train,
                oof_predictions,
                zero_division=0,
            ),
            "CV F1": f1_score(
                y_train,
                oof_predictions,
                zero_division=0,
            ),
        })


# --------------------------------------------------
# 4. Select model and threshold using training CV
# --------------------------------------------------

results_df = pd.DataFrame(candidate_results)

# F1 is the primary selection metric.
best = results_df.loc[
    results_df["CV F1"].idxmax()
]

best_name = best["Model"]
best_params = best["Parameters"]
best_threshold = float(best["Threshold"])

print("=" * 75)
print("TRAINING CROSS-VALIDATION RESULTS")
print("=" * 75)

print(
    results_df.sort_values(
        "CV F1",
        ascending=False,
    ).head(10)[
        ["Model", "Threshold", "CV Precision",
         "CV Recall", "CV F1"]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nSelected model:", best_name)
print("Selected parameters:", best_params)
print("Selected threshold:", best_threshold)
print("Selected CV F1:", round(float(best["CV F1"]), 4))


# --------------------------------------------------
# 5. Fit selected model on all training data
#    and evaluate test set once
# --------------------------------------------------

final_model = GradientBoostingClassifier(
    **best_params,
    random_state=42,
)

final_model.fit(X_train, y_train)

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n" + "=" * 75)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 75)

print("Test samples:", len(y_test))
print("Files flagged:", int(test_predictions.sum()))
print(
    "Accuracy :",
    round(accuracy_score(y_test, test_predictions), 4),
)
print(
    "Precision:",
    round(precision_score(
        y_test, test_predictions, zero_division=0
    ), 4),
)
print(
    "Recall   :",
    round(recall_score(
        y_test, test_predictions, zero_division=0
    ), 4),
)
print(
    "F1       :",
    round(f1_score(
        y_test, test_predictions, zero_division=0
    ), 4),
)