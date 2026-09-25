import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "titanic_model.joblib")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Titanic Survival Prediction API",
    version="1.0.0",
)


class PassengerRequest(BaseModel):
    pclass: int = Field(..., ge=1, le=3)
    sex: str
    age: float = Field(..., ge=0)
    sibsp: int = Field(..., ge=0)
    parch: int = Field(..., ge=0)
    fare: float = Field(..., ge=0)
    embarked: str
    adult_male: bool
    deck: str
    alone: bool


@app.get("/")
def root():
    return {
        "service": "Titanic Survival Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "service": "Titanic Survival Prediction API",
        "status": "healthy",
        "model": "loaded",
    }


@app.post("/predict")
def predict(request: PassengerRequest):
    data = pd.DataFrame([request.model_dump()])

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0, 1])

    return {
        "prediction": prediction,
        "prediction_label": (
            "survived" if prediction == 1 else "did_not_survive"
        ),
        "survival_probability": round(probability, 4),
    }
