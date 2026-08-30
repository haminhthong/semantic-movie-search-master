"""Module điều phối (Orchestration) toàn bộ pipeline tìm kiếm phim thích ứng.

Thực hiện luồng xử lý thông minh (Adaptive Routing):
1. Mã hóa truy vấn -> Hybrid Search (Dense + Sparse) -> Reciprocal Rank Fusion (RRF).
2. Kiểm tra độ tự tin kết quả (Confidence Score Router):
   - Tuyến EASY: Điểm top 1 cao và khoảng cách với top 2 lớn (Confidence Gap) -> Trả về kết quả ngay lập tức để tiết kiệm chi phí/latency.
   - Tuyến HARD: Kết quả chưa đủ độ tự tin -> Kích hoạt HyDE mở rộng truy vấn qua LLM + Chạy Cross-Encoder Rerank để xếp hạng lại ứng viên.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .hyde import HyDEProcessor
from .query import QueryEncoder
from .ranking import normalize_scores, reciprocal_rank_fusion, to_movies
from .rerank import CrossEncoderReranker
from .store import hybrid_search


def parse_year(value: str) -> Tuple[int, int]:
    """Phân tích cú pháp chuỗi năm phát hành (ví dụ: "2014" hoặc "2000-2020" hoặc "2000 to 2020").

    Args:
        value: Chuỗi nhập vào từ giao diện hoặc API đại diện cho năm/khoảng năm.

    Returns:
        Tuple[int, int]: Cặp (năm_bắt_đầu, năm_kết_thúc).

    Raises:
        ValueError: Nếu định dạng chuỗi không hợp lệ hoặc năm ngoài khoảng 1888-2100.
    """
    match = re.fullmatch(r"\s*(\d{4})(?:\s*(?:-|to)\s*(\d{4}))?\s*", value, re.IGNORECASE)
    if not match:
        raise ValueError("Định dạng năm không hợp lệ. Vui lòng nhập năm dạng YYYY (vd: 2014) hoặc YYYY-YYYY (vd: 2010-2020).")

    start, end = int(match.group(1)), int(match.group(2) or match.group(1))

    if start > end:
        raise ValueError("Năm bắt đầu không được lớn hơn năm kết thúc.")

    if not 1888 <= start <= end <= 2100:
        raise ValueError("Khoảng năm phải nằm trong từ năm 1888 đến 2100.")

    return start, end


def build_filter(genre: str = "", year: str = "") -> Optional[Any]:
    """Tạo bộ lọc điều kiện (Qdrant Filter) theo thể loại và năm phát hành.

    Args:
        genre: Tên thể loại phim (ví dụ: "Action", "Sci-Fi"). Nếu là "All" hoặc rỗng sẽ bỏ qua.
        year: Chuỗi lọc năm (ví dụ: "2014" hoặc "2010-2020").

    Returns:
        Optional[models.Filter]: Đối tượng filter của Qdrant hoặc None nếu không có điều kiện nào.
    """
    from qdrant_client import models

    conditions: List[Any] = []

    # Lọc thể loại nếu có
    if genre and genre != "All":
        conditions.append(models.FieldCondition(key="genres", match=models.MatchText(text=genre)))

    # Lọc khoảng năm phát hành
    if year.strip():
        start, end = parse_year(year)
        conditions.append(models.FieldCondition(key="release_year", range=models.Range(gte=start, lte=end)))

    return models.Filter(must=conditions) if conditions else None



class MovieSearch:
    """Động cơ tìm kiếm phim ngữ nghĩa thích ứng (Adaptive Semantic Movie Search Engine)."""

    def __init__(self) -> None:
        """Khởi tạo và duy trì các mô hình trong bộ nhớ ứng dụng (QueryEncoder, HyDEProcessor)."""
        self.encoder = QueryEncoder()
        # Dùng chung mô hình Dense Encoder cho HyDE để tối ưu RAM
        self.hyde = HyDEProcessor(settings.groq_api_key, self.encoder.dense_model)
        self.reranker: Optional[CrossEncoderReranker] = None

    def _candidates(
        self,
        dense_vector: List[float],
        sparse_vector: Tuple[List[int], List[float]],
        query_filter: Optional[Any],
    ) -> List[Dict[str, Any]]:

        """Thực hiện hybrid search trên Qdrant và gộp kết quả bằng RRF."""
        dense, sparse = hybrid_search(dense_vector, sparse_vector, query_filter)
        return to_movies(reciprocal_rank_fusion(dense, sparse))

    def _get_reranker(self) -> CrossEncoderReranker:
        """Nạp lười (Lazy loading) mô hình CrossEncoderReranker khi có truy vấn HARD đầu tiên."""
        if self.reranker is None:
            self.reranker = CrossEncoderReranker()
        return self.reranker

    def search(
        self,
        query: str,
        top_n: int = 10,
        genre: str = "",
        year: str = "",
    ) -> Dict[str, Any]:
        """Thực hiện quy trình tìm kiếm đầy đủ với bộ điều tuyến thích ứng (Adaptive Router).

        Args:
            query: Mô hình hay câu miêu tả cốt truyện phim từ người dùng.
            top_n: Số lượng phim kết quả mong muốn trả về. Mặc định 10.
            genre: Thể loại phim muốn lọc. Mặc định "" (tất cả).
            year: Khoảng năm sản xuất muốn lọc (dạng "2014" hoặc "2010-2020").

        Returns:
            Dict chứa:
            - movies (List[Dict]): Danh sách phim kèm điểm tương quan đã chuẩn hóa.
            - route (str): Phân tuyến xử lý ("EASY" hoặc "HARD").
            - hyde (Optional[str]): Nội dung tóm tắt giả định sinh bởi HyDE (nếu thuộc tuyến HARD).

        Raises:
            ValueError: Nếu tham số top_n <= 0 hoặc truy vấn rỗng.
        """
        if top_n <= 0:
            raise ValueError("Số lượng phim top_n phải lớn hơn 0.")

        # 1. Làm sạch truy vấn & mã hóa vector
        clean_query, dense_vector, sparse_vector = self.encoder.encode(query)
        query_filter = build_filter(genre, year)

        # 2. Truy hồi vòng 1 (Hybrid Search + RRF Fusion)
        candidates = self._candidates(dense_vector, sparse_vector, query_filter)

        # 3. Đánh giá độ tự tin (Confidence Router)
        top_score = candidates[0]["relevance_score"] if candidates else 0.0
        runner_up = candidates[1]["relevance_score"] if len(candidates) > 1 else top_score

        # Điểm RRF tối thiểu và khoảng cách giữa top 1 & top 2 đủ lớn -> Tuyến EASY
        is_easy_route = (
            top_score >= settings.minimum_score
            and (top_score - runner_up) >= settings.confidence_gap
        )

        if is_easy_route:
            return {
                "movies": normalize_scores(candidates, top_n),
                "route": "EASY",
                "hyde": None,
            }

        # 4. Tuyến HARD: Mở rộng truy vấn HyDE + Cross-Encoder Reranking
        hyde_vector, hypothetical = self.hyde.expand(clean_query)

        # Nếu sinh được văn bản HyDE khác với câu gốc, chạy lại truy hồi lai với hyde_vector
        if hypothetical != clean_query:
            candidates = self._candidates(hyde_vector, sparse_vector, query_filter)

        # Rerank danh sách ứng viên thông qua Cross-Encoder
        candidates = self._get_reranker().rerank(clean_query, candidates)

        return {
            "movies": normalize_scores(candidates, top_n),
            "route": "HARD",
            "hyde": hypothetical if hypothetical != clean_query else None,
        }

