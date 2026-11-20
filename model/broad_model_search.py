import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from preprocess_relink import prepare_dataset


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = prepare_dataset()

feature_columns = [
    column
    for column in df.columns
    if column not in ["isDefective", "project"]
]

X = df[feature_columns]
y = df["isDefective"]


# ---------------------------------------------------------
# Fixed final test set
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ---------------------------------------------------------
# Models and parameter grids
# ---------------------------------------------------------

models = {

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
    ),

    "Extra Trees": (
        ExtraTreesClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(
            random_state=42
        ),
        {
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "min_samples_leaf": [1, 2, 4],
        },
    ),

    "HistGradientBoosting": (
        HistGradientBoostingClassifier(
            random_state=42
        ),
        {
            "max_iter": [100, 200, 300],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_leaf_nodes": [7, 15, 31],
            "l2_regularization": [0, 0.1, 1.0],
        },
    ),

    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                class_weight="balanced"
            )),
        ]),
        {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto", 0.01, 0.1],
            "model__kernel": ["rbf"],
        },
    ),

    "KNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]),
        {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
    ),
}


# ---------------------------------------------------------
# Run model search
# ---------------------------------------------------------

results = []


for name, (model, param_grid) in models.items():

    print("\n" + "=" * 70)
    print("Searching:", name)
    print("=" * 70)

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    print("Best parameters:")
    print(search.best_params_)

    print("\nCV F1:", round(search.best_score_, 4))

    print("\nTest performance:")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1       :", round(f1, 4))

    results.append({
        "Model": name,
        "CV F1": search.best_score_,
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1": f1,
        "Best Parameters": search.best_params_,
    })


# ---------------------------------------------------------
# Final comparison
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Test F1",
    ascending=False
)

print("\n\n")
print("=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)

print(
    results_df[
        [
            "Model",
            "CV F1",
            "Test Accuracy",
            "Test Precision",
            "Test Recall",
            "Test F1",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nBest model by test F1:")
print(results_df.iloc[0]["Model"])

print("\nBest parameters:")
print(results_df.iloc[0]["Best Parameters"])