"""Gộp thứ hạng và chuyển kết quả Qdrant thành ứng viên phim."""

from collections import defaultdict

from .config import settings


def reciprocal_rank_fusion(dense: list[dict], sparse: list[dict], limit: int | None = None):
    """Gộp hai danh sách bằng RRF và loại trùng theo point id."""
    scores = defaultdict(float)
    documents = {}
    for results in (dense, sparse):
        for rank, item in enumerate(results, start=1):
            document_id = item["id"]
            scores[document_id] += 1.0 / (settings.rrf_k + rank)
            documents.setdefault(document_id, item)
    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [
        {**documents[document_id], "score": scores[document_id]}
        for document_id in ranked_ids[: limit or settings.candidate_k]
    ]


def to_movies(documents: list[dict]) -> list[dict]:
    """Tạo ứng viên phim từ payload và giữ nguyên thứ hạng."""
    movies, seen = [], set()
    for document in documents:
        payload = document.get("payload", {})
        movie_id = payload.get("movie_id")
        if movie_id is None or movie_id in seen:
            continue
        seen.add(movie_id)
        movies.append({
            "movie_id": movie_id,
            "title": payload.get("title", "Không rõ tên"),
            "genres": payload.get("genres", ""),
            "release_date": payload.get("release_date", ""),
            "release_year": payload.get("release_year", 0),
            "vote_average": payload.get("vote_average", 0.0),
            "popularity": payload.get("popularity", 0.0),
            "poster_path": payload.get("poster_path", ""),
            "document": {"id": document["id"], "text": payload.get("document_text", "")},
            "relevance_score": float(document["score"]),
        })
    return movies


def normalize_scores(movies: list[dict], top_n: int) -> list[dict]:
    """Chuẩn hóa điểm về 0-1 mà không trộn rating/popularity."""
    if not movies or top_n <= 0:
        return []
    scores = [movie["relevance_score"] for movie in movies]
    low, high = min(scores), max(scores)
    for movie in movies:
        movie["final_score"] = 1.0 if high == low else round((movie["relevance_score"] - low) / (high - low), 4)
    return sorted(movies, key=lambda movie: movie["relevance_score"], reverse=True)[:top_n]
