import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "titanic_model.joblib")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
coefficients = model.coef_[0]

importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "abs_coefficient": abs(coefficients),
})

importance["direction"] = importance["coefficient"].apply(
    lambda x: "positive" if x > 0 else "negative"
)

importance = importance.sort_values(
    "abs_coefficient",
    ascending=False,
)

output_path = os.path.join(
    OUTPUT_DIR,
    "feature_coefficients.csv",
)

importance.to_csv(output_path, index=False)

print("INTERPRETATION: PASS")
print(f"Saved to: {output_path}")
print()
print("Top 10 features by absolute coefficient:")
print(
    importance[
        ["feature", "coefficient", "direction"]
    ].head(10).to_string(index=False)
)
