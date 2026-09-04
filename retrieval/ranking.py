"""Module gộp thứ hạng (Rank Fusion) và chuyển đổi dữ liệu Qdrant thành ứng viên phim.

Áp dụng thuật toán Reciprocal Rank Fusion (RRF) để kết hợp danh sách xếp hạng
từ hai nhánh tìm kiếm khác nhau (Dense Vector & Sparse BM25) mà không bị ảnh hưởng
bởi sự chênh lệch thang điểm gốc giữa hai mô hình.
"""

from collections import defaultdict
from typing import Any

from .config import settings


def reciprocal_rank_fusion(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Gộp hai danh sách xếp hạng từ Dense và Sparse retrieval bằng thuật toán Reciprocal Rank Fusion (RRF).

    Công thức RRF cho mỗi tài liệu d:
        RRF_Score(d) = sum(1 / (k + rank_i(d)))
    Trong đó:
        - k: Hệ số làm mượt (settings.rrf_k, mặc định = 60).
        - rank_i(d): Thứ hạng (1-indexed) của tài liệu d trong danh sách xếp hạng thứ i.

    Args:
        dense: Danh sách kết quả trả về từ nhánh Dense Search (đã xếp hạng theo cosine similarity).
        sparse: Danh sách kết quả trả về từ nhánh Sparse Search (đã xếp hạng theo BM25 score).
        limit: Số lượng tài liệu top đầu cần lấy. Mặc định lấy theo settings.candidate_k (20).

    Returns:
        Danh sách tài liệu đã được gộp hạng và gán lại trường "score" theo điểm RRF mới.
    """
    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, dict[str, Any]] = {}

    for results in (dense, sparse):
        for rank, item in enumerate(results, start=1):
            document_id: str = str(item["id"])
            # Cộng dồn điểm nghịch đảo thứ hạng RRF
            scores[document_id] += 1.0 / (settings.rrf_k + rank)
            documents.setdefault(document_id, item)

    # Sắp xếp các tài liệu theo điểm RRF tổng hợp giảm dần
    ranked_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
    max_candidates = limit or settings.candidate_k

    return [{**documents[doc_id], "score": scores[doc_id]} for doc_id in ranked_ids[:max_candidates]]


def to_movies(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chuyển đổi danh sách payload tài liệu Qdrant thô thành cấu trúc dữ liệu ứng viên phim chuẩn.

    Hỗ trợ khử trùng lặp theo `movie_id` nếu trong tập hợp tài liệu có nhiều điểm trùng phim.

    Args:
        documents: Danh sách tài liệu Qdrant (chứa id, score, payload).

    Returns:
        List[Dict[str, Any]]: Danh sách các dict phim chứa metadata chi tiết (title, release_year, relevance_score,...).
    """
    movies: list[dict[str, Any]] = []
    seen: set[Any] = set()

    for document in documents:
        payload: dict[str, Any] = document.get("payload", {})
        movie_id = payload.get("movie_id")

        # Bỏ qua tài liệu thiếu movie_id hoặc đã xuất hiện trước đó
        if movie_id is None or movie_id in seen:
            continue
        seen.add(movie_id)

        movies.append(
            {
                "movie_id": movie_id,
                "title": payload.get("title", "Không rõ tên"),
                "genres": payload.get("genres", ""),
                "release_date": payload.get("release_date", ""),
                "release_year": payload.get("release_year", 0),
                "vote_average": payload.get("vote_average", 0.0),
                "popularity": payload.get("popularity", 0.0),
                "poster_path": payload.get("poster_path", ""),
                "document": {
                    "id": document["id"],
                    "text": payload.get("document_text", ""),
                },
                "relevance_score": float(document.get("score", 0.0)),
            }
        )

    return movies


def normalize_scores(movies: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    """Chuẩn hóa điểm tương quan (relevance_score) về khoảng [0.0, 1.0] bằng Min-Max Scaling.

    Lưu ý: Chỉ chuẩn hóa điểm tương quan ngữ nghĩa (relevance score), tuyệt đối không
    trộn hay nhân với điểm TMDB rating/popularity để đảm bảo kết quả phản ánh chính xác
    độ tương quan với truy vấn của người dùng.

    Args:
        movies: Danh sách ứng viên phim đã được xếp hạng.
        top_n: Số lượng phim tối đa muốn trả về cho người dùng.

    Returns:
        Danh sách top_n phim đã được bổ sung trường "final_score" [0.0 - 1.0].
    """
    if not movies or top_n <= 0:
        return []

    scores = [movie["relevance_score"] for movie in movies]
    low, high = min(scores), max(scores)

    for movie in movies:
        if high == low:
            movie["final_score"] = 1.0
        else:
            # Chuẩn hóa Min-Max và làm tròn 4 chữ số thập phân
            scaled = (movie["relevance_score"] - low) / (high - low)
            movie["final_score"] = round(scaled, 4)

    # Sắp xếp giảm dần theo relevance_score và cắt lấy top_n
    sorted_movies = sorted(movies, key=lambda m: m["relevance_score"], reverse=True)
    return sorted_movies[:top_n]
