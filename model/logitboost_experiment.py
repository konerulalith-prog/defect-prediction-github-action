
import pandas as pd

from logitboost import LogitBoost
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from preprocess_relink import prepare_dataset

# Load the same prepared ReLink dataset used by previous experiments.
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

# Start with the package's default LogitBoost configuration.
model = LogitBoost()

# Five-fold cross-validation on training data only.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("=" * 60)
print("LOGITBOOST - RELINK DEFECT PREDICTION")
print("=" * 60)

try:
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=1,
        error_score="raise",
    )

    print(f"CV F1 scores: {cv_scores.round(4)}")
    print(f"CV F1 mean:   {cv_scores.mean():.4f}")
    print(f"CV F1 std:    {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nHeld-out test results")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, predictions, zero_division=0):.4f}")

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(f"\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")

    results = pd.DataFrame([{
        "Model": "LogitBoost",
        "CV F1 Mean": cv_scores.mean(),
        "CV F1 Std": cv_scores.std(),
        "Test Accuracy": accuracy_score(y_test, predictions),
        "Test Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Test Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Test F1": f1_score(y_test, predictions, zero_division=0),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }])

    output_path = "data/logitboost_results.csv"
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

except Exception as error:
    print("\nLogitBoost experiment failed.")
    print(f"Error: {type(error).__name__}: {error}")
    print(
        "\nSend me the complete error output. "
        "The installed package may need compatibility adjustments."
    )
 