"""Integration tests for FastAPI recommendation serving endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.graphrec.api.app import app, initialize_app_state


@pytest.fixture(scope="module")
def client():
    initialize_app_state("data", max_ratings=5000)
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "two_tower" in data["available_models"]
    assert data["num_users"] > 0
    assert data["num_movies"] > 0
    assert data["graph_nodes"] > 0


def test_recommend_endpoint_with_explanations(client):
    response = client.get("/recommend?user_id=1&k=5&model=two_tower&explain=true")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["count"] == 5
    assert len(data["recommendations"]) == 5

    rec = data["recommendations"][0]
    assert "title" in rec
    assert "affinity_score" in rec
    assert "explanation" in rec
    assert rec["explanation"] is not None


def test_movies_search_endpoint(client):
    response = client.get("/movies/search?q=Toy")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert any("Toy" in item["title"] for item in data)


def test_invalid_model_error(client):
    response = client.get("/recommend?user_id=1&k=5&model=non_existent_model")
    assert response.status_code == 400
