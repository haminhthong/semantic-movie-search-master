"""Unit tests cho FastAPI Web Application endpoints."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api import app, get_service

client = TestClient(app)


def test_health_endpoint():
    """Kiểm tra endpoint GET /health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_search_endpoint_validation_error():
    """Kiểm tra validation khi gửi request với query rỗng."""
    response = client.post("/search", json={"query": "", "top_n": 10})
    assert response.status_code == 422  # Unprocessable Entity


def test_search_endpoint_success_mocked():
    """Kiểm tra endpoint POST /search với mock SearchService."""
    mock_service = MagicMock()
    mock_service.search.return_value = {
        "movies": [
            {
                "movie_id": 157336,
                "title": "Interstellar",
                "genres": "Adventure, Drama, Science Fiction",
                "release_date": "2014-11-05",
                "release_year": 2014,
                "vote_average": 8.4,
                "popularity": 145.2,
                "poster_path": "/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
                "document": {"id": "doc1", "text": "Interstellar plot..."},
                "relevance_score": 0.05,
                "final_score": 1.0,
            }
        ],
        "route": "EASY",
        "hyde": None,
        "latency_ms": 42.5,
    }

    app.dependency_overrides[get_service] = lambda: mock_service

    try:
        response = client.post(
            "/search",
            json={
                "query": "astronauts traveling through wormhole",
                "top_n": 5,
                "genre": "Science Fiction",
                "year": "2014",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "EASY"
        assert len(data["movies"]) == 1
        assert data["movies"][0]["title"] == "Interstellar"
        assert data["latency_ms"] == 42.5
    finally:
        app.dependency_overrides.clear()
