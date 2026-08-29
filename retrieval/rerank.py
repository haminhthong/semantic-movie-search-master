"""Xếp hạng lại ứng viên bằng cross-encoder."""

import logging
import time
from typing import Any

import torch
from sentence_transformers import CrossEncoder

from .config import RERANK_MODEL

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Đang nạp reranker %s trên %s", model_name, device.upper())
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, movies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Gán relevance_score và sắp xếp giảm dần."""
        if not movies:
            return []
        pairs = []
        for movie in movies:
            text = movie.get("document", {}).get("text", "")
            pairs.append([query, text])

        started = time.perf_counter()
        try:
            scores = self.model.predict(pairs)
        except Exception:
            logger.exception("Reranker gặp lỗi; giữ thứ tự từ bước truy hồi")
            for movie in movies:
                movie["relevance_score"] = float(movie.get("retrieval_score", 0.0))
            return movies

        for movie, score in zip(movies, scores):
            movie["relevance_score"] = float(score)
        logger.info("Rerank %d phim sau %.4f giây", len(movies), time.perf_counter() - started)
        return sorted(movies, key=lambda item: item["relevance_score"], reverse=True)
