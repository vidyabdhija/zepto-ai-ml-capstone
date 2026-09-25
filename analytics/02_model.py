import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "titanic_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

target = "survived"

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "adult_male",
    "deck",
    "alone",
]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

categorical_features = [
    "sex",
    "embarked",
    "adult_male",
    "deck",
    "alone",
]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = []
trained_models = {}

for model_name, model in models.items():

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    results.append(result)
    trained_models[model_name] = pipeline

    print(f"\n{model_name}")
    print("-" * len(model_name))
    print(f"Accuracy : {result["accuracy"]:.4f}")
    print(f"Precision: {result["precision"]:.4f}")
    print(f"Recall   : {result["recall"]:.4f}")
    print(f"F1       : {result["f1"]:.4f}")
    print(f"ROC-AUC  : {result["roc_auc"]:.4f}")

results_df = pd.DataFrame(results)

results_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
results_df.to_csv(results_path, index=False)

best_model_name = results_df.sort_values(
    "roc_auc",
    ascending=False,
).iloc[0]["model"]

best_model = trained_models[best_model_name]

model_path = os.path.join(MODEL_DIR, "titanic_model.joblib")
joblib.dump(best_model, model_path)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)
print(f"Best model by ROC-AUC: {best_model_name}")
print(f"Comparison saved to: {results_path}")
print(f"Model saved to: {model_path}")
