from fastapi.testclient import TestClient

from analytics.api import app as analytics_app
from support_assistant.main import app as support_app


analytics_client = TestClient(analytics_app)
support_client = TestClient(support_app)


def test_analytics_health():
    response = analytics_client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "loaded"


def test_analytics_prediction():
    payload = {
        "pclass": 1,
        "sex": "female",
        "age": 30,
        "sibsp": 0,
        "parch": 0,
        "fare": 100,
        "embarked": "C",
        "adult_male": False,
        "deck": "C",
        "alone": True,
    }

    response = analytics_client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in [
        "survived",
        "did_not_survive",
    ]

    assert 0 <= data["survival_probability"] <= 1


def test_analytics_validation():
    payload = {
        "pclass": 5,
        "sex": "female",
        "age": 30,
        "sibsp": 0,
        "parch": 0,
        "fare": 100,
        "embarked": "C",
        "adult_male": False,
        "deck": "C",
        "alone": True,
    }

    response = analytics_client.post("/predict", json=payload)

    assert response.status_code == 422


def test_support_health():
    response = support_client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["mock_llm"] is True


def test_support_question():
    response = support_client.post(
        "/ask",
        json={"query": "What is the refund policy?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert data["answer"]

    assert "sources" in data
    assert len(data["sources"]) > 0

    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1
