"""
tests/test_server.py
────────────────────
Unit and integration tests for the Starlette ASGI server and REST endpoints.
"""

from unittest.mock import patch
import pytest
from starlette.testclient import TestClient

from src.server import app


@pytest.fixture
def client():
    """Create a Starlette TestClient instance."""
    return TestClient(app)


def test_index_route(client):
    """Verify that root endpoint serves the frontend HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Zomato" in response.text
    assert "Curated Restaurant Discovery" in response.text


def test_locations_endpoint(client):
    """Verify that /api/locations returns a list of Bangalore areas."""
    response = client.get("/api/locations")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert isinstance(data["locations"], list)
    assert len(data["locations"]) > 0
    assert "Indiranagar" in data["locations"] or "Koramangala" in data["locations"]


def test_presets_endpoint(client):
    """Verify that /api/presets returns curated discovery presets."""
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["presets"]) >= 4
    preset = data["presets"][0]
    assert "location" in preset
    assert "cuisine" in preset
    assert "budget" in preset


def test_health_endpoint(client):
    """Verify that /api/health returns system operational status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["city"] == "Bangalore"
    assert data["dataset_nodes"] > 1000


def test_book_table_endpoint(client):
    """Verify instant table booking confirmation generation."""
    payload = {
        "restaurant": "Milano Pizzeria & Trattoria",
        "guests": 4,
        "time": "8:30 PM",
        "date": "Tonight",
    }
    response = client.post("/api/book", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "confirmed"
    assert "confirmation_id" in data
    assert data["guests"] == 4
    assert data["restaurant"] == "Milano Pizzeria & Trattoria"


def test_recommend_endpoint_validation_error(client):
    """Verify that empty location returns 400 bad request."""
    response = client.post("/api/recommend", json={"location": ""})
    assert response.status_code == 400
    assert "Location is required" in response.json()["error"]


@patch("src.server.call_llm")
def test_recommend_endpoint_mocked_llm(mock_call_llm, client):
    """Verify recommendation pipeline execution with mocked LLM response."""
    mock_call_llm.return_value = """
    {
      "summary": "Great picks found in Indiranagar matching your taste preferences.",
      "recommendations": [
        {
          "rank": 1,
          "name": "Milano Pizzeria",
          "cuisine": "Italian",
          "rating": 4.5,
          "budget_tier": "medium",
          "location": "Indiranagar",
          "explanation": "Authentic wood-fired crusts and great atmosphere."
        }
      ]
    }
    """
    payload = {
        "location": "Indiranagar",
        "cuisine": "Italian",
        "budget": "medium",
        "min_rating": 3.8,
        "notes": "Cozy dinner for two",
        "top_n": 5,
    }
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["name"] == "Milano Pizzeria"
    assert "vector_trace" in data["metadata"]
