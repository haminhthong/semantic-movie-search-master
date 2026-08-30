"""Unit tests cho bộ điều phối tìm kiếm MovieSearch và bộ lọc build_filter."""

from unittest.mock import MagicMock, patch

from retrieval.search import MovieSearch, build_filter


def test_build_filter_genre_only():
    """Kiểm tra tạo filter khi chỉ có thể loại."""
    q_filter = build_filter(genre="Action")
    assert q_filter is not None
    assert len(q_filter.must) == 1


def test_build_filter_year_only():
    """Kiểm tra tạo filter khi chỉ có năm."""
    q_filter = build_filter(year="2014")
    assert q_filter is not None
    assert len(q_filter.must) == 1


def test_build_filter_empty():
    """Kiểm tra tạo filter khi không có điều kiện (All genre, empty year)."""
    assert build_filter(genre="All", year="") is None
    assert build_filter(genre="", year="") is None


@patch("retrieval.search.hybrid_search")
@patch("retrieval.search.QueryEncoder")
def test_movie_search_easy_route(mock_encoder_cls, mock_hybrid_search):
    """Kiểm tra tuyến EASY khi kết quả top 1 có độ tự tin cao."""
    # Mock encoder
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = ("clean query", [0.1] * 384, ([1], [1.0]))
    mock_encoder.dense_model = MagicMock()
    mock_encoder_cls.return_value = mock_encoder

    # Mock hybrid_search
    mock_hybrid_search.return_value = (
        [{"id": "doc1", "score": 0.9, "payload": {"movie_id": 1, "title": "Interstellar"}}],
        [{"id": "doc1", "score": 10.0, "payload": {"movie_id": 1, "title": "Interstellar"}}],
    )

    search_engine = MovieSearch()
    response = search_engine.search("space wormhole", top_n=5)

    assert response["route"] == "EASY"
    assert response["hyde"] is None
    assert len(response["movies"]) > 0
    assert response["movies"][0]["title"] == "Interstellar"
