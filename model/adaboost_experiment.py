import pandas as pd

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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

from preprocess_relink import prepare_dataset


# Load dataset
df = prepare_dataset()

feature_columns = [
    col for col in df.columns
    if col not in ["isDefective", "project"]
]

X = df[feature_columns]
y = df["isDefective"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# Test different AdaBoost configurations
models = {
    "AdaBoost_Default": AdaBoostClassifier(
        n_estimators=50,
        learning_rate=1.0,
        random_state=42,
    ),

    "AdaBoost_Conservative": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    ),

    "AdaBoost_Stronger": AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.5,
        random_state=42,
    ),

    "AdaBoost_Stumps": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=2,
            random_state=42,
        ),
        n_estimators=100,
        learning_rate=0.5,
        random_state=42,
    ),

    "AdaBoost_RF": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=3,
            random_state=42,
        ),
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    ),
}

results = []

print("=" * 65)
print("ADABOOST EXPERIMENT - RELINK DEFECT PREDICTION")
print("=" * 65)

for name, model in models.items():

    print(f"\nTesting: {name}")

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
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
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    print(f"CV F1 Mean: {cv_scores.mean():.4f}")
    print(f"CV F1 Std:  {cv_scores.std():.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Test F1:   {f1:.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

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


# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="CV F1 Mean",
    ascending=False,
)

results_df.to_csv(
    "data/adaboost_results.csv",
    index=False,
)

print("\n" + "=" * 65)
print("FINAL ADABOOST COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nResults saved to data/adaboost_results.csv")