import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "titanic_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = [
    "pclass", "sex", "age", "sibsp", "parch", "fare",
    "embarked", "adult_male", "deck", "alone"
]
target = "survived"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked", "adult_male", "deck", "alone"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

def evaluate_model(name, pipeline, Xtr, ytr, Xte, yte):
    pipeline.fit(Xtr, ytr)
    pred = pipeline.predict(Xte)
    prob = pipeline.predict_proba(Xte)[:, 1]
    cm = confusion_matrix(yte, pred)

    result = {
        "model": name,
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "roc_auc": roc_auc_score(yte, prob),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    return result, pipeline

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, oob_score=True),
}

results = []
trained_models = {}

for name, model in models.items():
    pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    result, fitted = evaluate_model(name, pipe, X_train, y_train, X_test, y_test)
    results.append(result)
    trained_models[name] = fitted
    print(f"{name}: accuracy={result['accuracy']:.4f}, precision={result['precision']:.4f}, recall={result['recall']:.4f}, f1={result['f1']:.4f}, roc_auc={result['roc_auc']:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

pd.DataFrame([{
    "model": name,
    "oob_score": trained_models[name].named_steps["model"].oob_score_
} for name in ["Random Forest"]]).to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_oob.csv"), index=False
)

cm_rows = []
for row in results:
    cm_rows.append({
        "model": row["model"],
        "TN": row["tn"],
        "FP": row["fp"],
        "FN": row["fn"],
        "TP": row["tp"],
    })
pd.DataFrame(cm_rows).to_csv(os.path.join(OUTPUT_DIR, "confusion_matrices.csv"), index=False)

balanced_model = ImbPipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])
balanced_result, balanced_fitted = evaluate_model(
    "Logistic Regression - Balanced", balanced_model,
    X_train, y_train, X_test, y_test
)

smote_model = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
smote_result, smote_fitted = evaluate_model(
    "Logistic Regression - SMOTE", smote_model,
    X_train, y_train, X_test, y_test
)

imbalance_df = pd.DataFrame([
    {"experiment": "Baseline", **{k: balanced_result.get(k) for k in []}},
    {"experiment": "Balanced class weights", "accuracy": balanced_result["accuracy"], "precision": balanced_result["precision"], "recall": balanced_result["recall"], "f1": balanced_result["f1"], "roc_auc": balanced_result["roc_auc"]},
    {"experiment": "SMOTE", "accuracy": smote_result["accuracy"], "precision": smote_result["precision"], "recall": smote_result["recall"], "f1": smote_result["f1"], "roc_auc": smote_result["roc_auc"]},
])

baseline_result = next(r for r in results if r["model"] == "Logistic Regression")
imbalance_df.loc[0, ["accuracy", "precision", "recall", "f1", "roc_auc"]] = [
    baseline_result["accuracy"], baseline_result["precision"], baseline_result["recall"], baseline_result["f1"], baseline_result["roc_auc"]
]
imbalance_df.to_csv(os.path.join(OUTPUT_DIR, "class_imbalance_comparison.csv"), index=False)

rf_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)),
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
}

grid = GridSearchCV(
    rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

grid_pred = grid.predict(X_test)
grid_prob = grid.predict_proba(X_test)[:, 1]
grid_model = grid.best_estimator_

grid_metrics = pd.DataFrame([{
    "best_params": str(grid.best_params_),
    "accuracy": accuracy_score(y_test, grid_pred),
    "precision": precision_score(y_test, grid_pred, zero_division=0),
    "recall": recall_score(y_test, grid_pred, zero_division=0),
    "f1": f1_score(y_test, grid_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, grid_prob),
    "oob_score": grid_model.named_steps["model"].oob_score_,
}])
grid_metrics.to_csv(os.path.join(OUTPUT_DIR, "random_forest_gridsearch.csv"), index=False)

grid_model_path = os.path.join(MODEL_DIR, "titanic_model.joblib")
joblib.dump(grid_model, grid_model_path)

fare_features = ["pclass", "age", "sibsp", "parch", "survived"]
fare_X = df[fare_features]
fare_y = df["fare"]

fare_X_train, fare_X_test, fare_y_train, fare_y_test = train_test_split(
    fare_X, fare_y, test_size=0.20, random_state=42
)

fare_model = LinearRegression()
fare_model.fit(fare_X_train, fare_y_train)
fare_pred = fare_model.predict(fare_X_test)

mae = mean_absolute_error(fare_y_test, fare_pred)
rmse = np.sqrt(mean_squared_error(fare_y_test, fare_pred))
r2 = r2_score(fare_y_test, fare_pred)
n = len(fare_y_test)
p = fare_X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

pd.DataFrame([{
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted_R2": adjusted_r2,
}]).to_csv(os.path.join(OUTPUT_DIR, "fare_regression_metrics.csv"), index=False)

residuals = fare_y_test - fare_pred
plt.figure(figsize=(8, 5))
plt.scatter(fare_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Fare Regression Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fare_regression_residuals.png"))
plt.close()

print("=" * 60)
print("MODULE 2 MODELING COMPLETE")
print("=" * 60)
print(results_df.to_string(index=False))
print("Random Forest GridSearch best parameters:")
print(grid.best_params_)
print("Fare regression:")
print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, Adjusted R2={adjusted_r2:.4f}")
print(f"Saved model: {grid_model_path}")
