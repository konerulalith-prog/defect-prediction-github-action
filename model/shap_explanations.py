
import pandas as pd
import shap

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from preprocess_relink import prepare_dataset


# 1. Load and prepare the dataset
df = prepare_dataset()

feature_columns = [
    col for col in df.columns
    if col not in ["isDefective", "project"]
]

X = df[feature_columns]
y = df["isDefective"]


# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# 3. Train our current prototype model
model = GradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=3,
    n_estimators=200,
    random_state=42,
)

model.fit(X_train, y_train)


# 4. Select one test file to explain
sample = X_test.iloc[[0]]

probability = model.predict_proba(sample)[0, 1]
threshold = 0.30
prediction = int(probability >= threshold)

print("=" * 60)
print("DEFECT PREDICTION WITH SHAP EXPLANATION")
print("=" * 60)

print("Predicted defect probability:", round(probability, 4))
print("Decision threshold:", threshold)
print(
    "Prediction:",
    "Potentially defective" if prediction else "Lower predicted risk"
)


# 5. Explain the prediction using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# Binary classification: explain the defective-class output.
if isinstance(shap_values, list):
    values = shap_values[1][0]
else:
    values = shap_values[0]

explanation = pd.DataFrame({
    "Feature": feature_columns,
    "Feature Value": sample.iloc[0].values,
    "SHAP Value": values,
})

explanation["Absolute SHAP"] = explanation["SHAP Value"].abs()

explanation = explanation.sort_values(
    "Absolute SHAP",
    ascending=False,
)

print("\nTop 10 influential features:")
print(
    explanation[
        ["Feature", "Feature Value", "SHAP Value"]
    ].head(10).to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nInterpretation:")
print("Positive SHAP values push the model output toward the")
print("defective class; negative values push it away.")