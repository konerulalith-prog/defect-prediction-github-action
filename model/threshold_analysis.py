import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from preprocess_relink import prepare_dataset

# Load data
df = prepare_dataset()

feature_columns = [
    col for col in df.columns
    if col not in ["isDefective", "project"]
]

X = df[feature_columns]
y = df["isDefective"]

# Keep the same split as our previous experiments
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Current best Gradient Boosting parameters
model = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=4,
    min_samples_leaf=2,
    n_estimators=150,
    random_state=42,
)

model.fit(X_train, y_train)

# Probability that each test sample is defective
probabilities = model.predict_proba(X_test)[:, 1]

results = []

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    predictions = (probabilities >= threshold).astype(int)

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Flagged Files": int(predictions.sum()),
    })

results_df = pd.DataFrame(results)

print("\nGRADIENT BOOSTING: THRESHOLD ANALYSIS")
print("=" * 85)

print(results_df.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}"
))