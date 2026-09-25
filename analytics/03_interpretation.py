import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "analytics", "models", "titanic_model.joblib")
OUTPUT_DIR = os.path.join(BASE_DIR, "analytics", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

if hasattr(model, "feature_importances_"):
    importance_values = model.feature_importances_
    importance_column = "feature_importance"

elif hasattr(model, "coef_"):
    importance_values = model.coef_[0]
    importance_column = "coefficient"

else:
    raise ValueError(
        f"Model {type(model).__name__} does not support "
        "feature importance or coefficients."
    )

importance = pd.DataFrame({
    "feature": feature_names,
    importance_column: importance_values,
})

importance["abs_importance"] = abs(importance[importance_column])

if importance_column == "coefficient":
    importance["direction"] = importance[importance_column].apply(
        lambda x: "positive" if x > 0 else "negative"
    )
else:
    importance["direction"] = "not_applicable"

importance = importance.sort_values(
    "abs_importance",
    ascending=False,
)

output_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv",
)

importance.to_csv(output_path, index=False)

print("INTERPRETATION: PASS")
print(f"Saved to: {output_path}")
print()
print(f"Model: {type(model).__name__}")
print()
print("Top 10 features by importance:")
print(
    importance[
        ["feature", importance_column, "direction"]
    ].head(10).to_string(index=False)
)
