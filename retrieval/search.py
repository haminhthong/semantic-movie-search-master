"""Điều phối pipeline tìm kiếm phim."""

import re

from qdrant_client import models

from .config import settings
from .hyde import HyDEProcessor
from .query import QueryEncoder
from .ranking import normalize_scores, reciprocal_rank_fusion, to_movies
from .rerank import CrossEncoderReranker
from .store import hybrid_search


def parse_year(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d{4})(?:\s*(?:-|to)\s*(\d{4}))?\s*", value, re.IGNORECASE)
    if not match:
        raise ValueError("Năm phải có dạng YYYY hoặc YYYY-YYYY.")
    start, end = int(match.group(1)), int(match.group(2) or match.group(1))
    if not 1888 <= start <= end <= 2100:
        raise ValueError("Khoảng năm không hợp lệ.")
    return start, end


def build_filter(genre: str = "", year: str = ""):
    conditions = []
    if genre and genre != "All":
        conditions.append(models.FieldCondition(key="genres", match=models.MatchText(text=genre)))
    if year.strip():
        start, end = parse_year(year)
        conditions.append(models.FieldCondition(key="release_year", range=models.Range(gte=start, lte=end)))
    return models.Filter(must=conditions) if conditions else None


class MovieSearch:
    """Giữ model trong bộ nhớ và thực hiện một truy vấn hoàn chỉnh."""

    def __init__(self):
        self.encoder = QueryEncoder()
        self.hyde = HyDEProcessor(settings.groq_api_key, self.encoder.dense_model)
        self.reranker = None

    def _candidates(self, dense_vector, sparse_vector, query_filter):
        dense, sparse = hybrid_search(dense_vector, sparse_vector, query_filter)
        return to_movies(reciprocal_rank_fusion(dense, sparse))

    def _get_reranker(self):
        if self.reranker is None:
            self.reranker = CrossEncoderReranker()
        return self.reranker

    def search(self, query: str, top_n: int = 10, genre: str = "", year: str = "") -> dict:
        if top_n <= 0:
            raise ValueError("top_n phải lớn hơn 0.")
        clean_query, dense_vector, sparse_vector = self.encoder.encode(query)
        query_filter = build_filter(genre, year)
        candidates = self._candidates(dense_vector, sparse_vector, query_filter)

        top_score = candidates[0]["relevance_score"] if candidates else 0.0
        runner_up = candidates[1]["relevance_score"] if len(candidates) > 1 else top_score
        easy = top_score >= settings.minimum_score and top_score - runner_up >= settings.confidence_gap
        if easy:
            return {"movies": normalize_scores(candidates, top_n), "route": "EASY", "hyde": None}

        hyde_vector, hypothetical = self.hyde.expand(clean_query)
        if hypothetical != clean_query:
            candidates = self._candidates(hyde_vector, sparse_vector, query_filter)
        candidates = self._get_reranker().rerank(clean_query, candidates)
        return {"movies": normalize_scores(candidates, top_n), "route": "HARD", "hyde": hypothetical}
