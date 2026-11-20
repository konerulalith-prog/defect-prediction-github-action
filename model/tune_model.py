import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from preprocess_relink import prepare_dataset


# Load dataset
df = prepare_dataset()

# Use the 26 code-metric features
feature_columns = [
    column
    for column in df.columns
    if column not in ["isDefective", "project"]
]

X = df[feature_columns]
y = df["isDefective"]


# Same split used in our previous model comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# Hyperparameters to test
param_grid = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4],
}


model = GradientBoostingClassifier(
    random_state=42
)


# Search for the best combination using F1 score
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)


grid_search.fit(X_train, y_train)


# Best model
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)


print("=" * 60)
print("TUNED GRADIENT BOOSTING")
print("=" * 60)

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1:")
print(round(grid_search.best_score_, 4))

print("\nTest-set performance:")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1 Score :", round(f1_score(y_test, y_pred), 4))