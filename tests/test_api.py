from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction(client: TestClient) -> None:
    response = client.post("/predict", json={"features": [0] * 2})

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert body["prediction"] in (0, 1)


def test_predict_rejects_wrong_feature_count(client: TestClient) -> None:
    response = client.post("/predict", json={"features": []})

    assert response.status_code == 422


def test_predict_rejects_invalid_feature_type(client: TestClient) -> None:
    response = client.post("/predict", json={"features": ["abc"] * 2})

    assert response.status_code == 422
