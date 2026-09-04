"""Module xếp hạng lại (Reranking) danh sách phim ứng viên bằng mô hình Cross-Encoder.

Khác với bi-encoder (mã hóa query và document độc lập), mô hình Cross-Encoder
nhận đồng thời cặp [query, document_text] làm đầu vào và cho phép cơ chế tự chú ý (self-attention)
đánh giá chính xác mức độ tương quan ngữ nghĩa trực tiếp giữa câu hỏi và văn bản phim.
"""

import logging
import time
from typing import Any

from .config import RERANK_MODEL

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Đánh giá lại độ tương quan của danh sách ứng viên thông qua Cross-Encoder transformer model."""

    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        """Khởi tạo mô hình Cross-Encoder và tự động chọn phần cứng GPU (cuda) nếu sẵn sàng.

        Args:
            model_name: Tên mô hình reranker từ HuggingFace / SentenceTransformers.
        """
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Đang nạp mô hình Reranker %s trên thiết bị %s", model_name, device.upper())
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, movies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Tính điểm độ tương quan thực sự (relevance_score) cho từng phim và sắp xếp giảm dần.

        Args:
            query: Câu truy vấn đã làm sạch của người dùng.
            movies: Danh sách ứng viên phim thu được từ bước truy hồi và gộp hạng RRF.

        Returns:
            Danh sách các dict phim đã được gán lại trường `relevance_score` và sắp xếp theo thứ tự ưu tiên mới.
        """
        if not movies:
            return []

        # Chuẩn bị cặp dữ liệu [query, document_text] cho mô hình Cross-Encoder
        pairs = []
        for movie in movies:
            text = movie.get("document", {}).get("text", "")
            pairs.append([query, text])

        started = time.perf_counter()
        try:
            # Chạy dự đoán điểm tương quan trực tiếp
            scores = self.model.predict(pairs)
        except Exception:
            logger.exception("Tiến trình Reranker gặp sự cố; giữ nguyên thứ tự điểm từ bước truy hồi RRF")
            for movie in movies:
                movie["relevance_score"] = float(movie.get("relevance_score", 0.0))
            return movies

        # Gán lại điểm relevance_score từ kết quả predict của Cross-Encoder
        for movie, score in zip(movies, scores):
            movie["relevance_score"] = float(score)

        elapsed = time.perf_counter() - started
        logger.info("Hoàn tất Rerank cho %d phim trong %.4f giây", len(movies), elapsed)

        # Sắp xếp danh sách phim giảm dần theo điểm tương quan mới
        return sorted(movies, key=lambda item: item["relevance_score"], reverse=True)
