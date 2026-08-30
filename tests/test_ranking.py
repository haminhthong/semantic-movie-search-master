"""Unit tests cho module ranking: RRF fusion, mapping payload và normalize score."""

from retrieval.ranking import normalize_scores, reciprocal_rank_fusion, to_movies


def test_reciprocal_rank_fusion():
    """Kiểm tra tính điểm RRF và gộp hai danh sách Dense & Sparse."""
    dense = [
        {"id": "doc1", "score": 0.9, "payload": {"movie_id": 1, "title": "Movie A"}},
        {"id": "doc2", "score": 0.8, "payload": {"movie_id": 2, "title": "Movie B"}},
    ]
    sparse = [
        {"id": "doc2", "score": 5.0, "payload": {"movie_id": 2, "title": "Movie B"}},
        {"id": "doc3", "score": 4.0, "payload": {"movie_id": 3, "title": "Movie C"}},
    ]

    fused = reciprocal_rank_fusion(dense, sparse, limit=10)
    assert len(fused) == 3

    # doc2 xuất hiện ở rank 2 nhánh dense (1/(60+2)) và rank 1 nhánh sparse (1/(60+1)), nên sẽ đứng đầu
    fused_ids = [item["id"] for item in fused]
    assert fused_ids[0] == "doc2"


def test_to_movies():
    """Kiểm tra chuyển đổi danh sách payload tài liệu Qdrant thành ứng viên phim."""
    docs = [
        {
            "id": "doc1",
            "score": 0.03,
            "payload": {
                "movie_id": 101,
                "title": "Inception",
                "genres": "Action, Sci-Fi",
                "release_year": 2010,
                "vote_average": 8.8,
                "document_text": "Overview text...",
            },
        },
        {
            "id": "doc1_dup",
            "score": 0.02,
            "payload": {
                "movie_id": 101,  # Trùng movie_id
                "title": "Inception Duplicate",
            },
        },
    ]
    movies = to_movies(docs)
    assert len(movies) == 1
    assert movies[0]["movie_id"] == 101
    assert movies[0]["title"] == "Inception"
    assert movies[0]["release_year"] == 2010


def test_normalize_scores():
    """Kiểm tra chuẩn hóa điểm Min-Max scaling về khoảng [0.0, 1.0]."""
    movies = [
        {"movie_id": 1, "relevance_score": 10.0},
        {"movie_id": 2, "relevance_score": 5.0},
        {"movie_id": 3, "relevance_score": 0.0},
    ]

    normalized = normalize_scores(movies, top_n=3)
    assert len(normalized) == 3
    assert normalized[0]["final_score"] == 1.0
    assert normalized[1]["final_score"] == 0.5
    assert normalized[2]["final_score"] == 0.0


def test_normalize_scores_single_item():
    """Kiểm tra chuẩn hóa khi chỉ có 1 phần tử (high == low)."""
    movies = [{"movie_id": 1, "relevance_score": 5.0}]
    normalized = normalize_scores(movies, top_n=10)
    assert len(normalized) == 1
    assert normalized[0]["final_score"] == 1.0
