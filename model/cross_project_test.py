from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess_relink import load_relink_arff


projects = ["Apache", "Safe", "Zxing"]


def prepare_project(project):
    df = load_relink_arff(f"data/ReLink/{project}.arff")

    feature_columns = [
        column for column in df.columns
        if column != "isDefective"
    ]

    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["isDefective"] = df["isDefective"].map({
        "FALSE": 0,
        "TRUE": 1
    })

    df.loc[df["AvgLine"] > 1000, "AvgLine"] = pd.NA
    df["AvgLine"] = df["AvgLine"].fillna(df["AvgLine"].median())

    return df


import pandas as pd


for test_project in projects:

    train_projects = [
        project for project in projects
        if project != test_project
    ]

    train_data = pd.concat(
        [prepare_project(project) for project in train_projects],
        ignore_index=True
    )

    test_data = prepare_project(test_project)

    feature_columns = [
        column for column in train_data.columns
        if column != "isDefective"
    ]

    X_train = train_data[feature_columns]
    y_train = train_data["isDefective"]

    X_test = test_data[feature_columns]
    y_test = test_data["isDefective"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print("Test Project:", test_project)
    print("Training Projects:", ", ".join(train_projects))
    print("=" * 60)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nAccuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("F1 Score :", round(f1_score(y_test, y_pred, zero_division=0), 4))