"""Orchestrate hybrid retrieval, optional HyDE, reranking and filtering."""
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from qdrant_client import models
from .aggregate import DocumentAggregator
from .final_scorer import FinalScorer
from .hybrid import hybrid_retrieval
from .hyde import HyDEProcessor
from .query import QueryEncoder
from .rerank import CrossEncoderReranker
from .rrf import RankFusion

def parse_year_filter(value: str) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d{4})(?:\s*(?:-|to)\s*(\d{4}))?\s*", value, re.I)
    if not match:
        raise ValueError("Year must be YYYY or YYYY-YYYY.")
    start, end = int(match.group(1)), int(match.group(2) or match.group(1))
    if not (1888 <= start <= end <= 2100):
        raise ValueError("Year range must be between 1888 and 2100.")
    return start, end

class AdaptiveSearchPipeline:
    def __init__(self, ci_threshold: float = 0.01, min_score_threshold: float = 0.03):
        self.query_encoder = QueryEncoder()
        self.rank_fusion = RankFusion(rrf_k=60)
        self.aggregator = DocumentAggregator(top_n_movies=20)
        self.reranker = CrossEncoderReranker()
        self.final_scorer = FinalScorer()
        self.hyde = HyDEProcessor(api_key=os.getenv("GROQ_API_KEY"), encoder=self.query_encoder.dense_model)
        self.ci_threshold, self.min_score_threshold = ci_threshold, min_score_threshold

    def _build_qdrant_filter(self, user_filters: Optional[dict]):
        if not user_filters:
            return None
        conditions = []
        genre = user_filters.get("genre")
        if genre and genre != "All":
            conditions.append(models.FieldCondition(key="genres", match=models.MatchText(text=genre)))
        year = str(user_filters.get("year", "")).strip()
        if year:
            start, end = parse_year_filter(year)
            conditions.append(models.FieldCondition(key="release_year", range=models.Range(gte=start, lte=end)))
        return models.Filter(must=conditions) if conditions else None

    def search(self, raw_query: str, top_n: int = 10, user_filters: Optional[dict] = None) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
        query_filter = self._build_qdrant_filter(user_filters)
        clean_query, dense, sparse = self.query_encoder.encode(raw_query)
        dense_hits, sparse_hits = hybrid_retrieval(dense, sparse, query_filter=query_filter)
        first = self.aggregator.aggregate(self.rank_fusion.fuse(dense_hits, sparse_hits))
        top1 = first[0]["max_score"] if first else 0.0
        top2 = first[1]["max_score"] if len(first) > 1 else top1
        confidence = top1 - top2 if top1 >= self.min_score_threshold else 0.0
        if confidence >= self.ci_threshold:
            route, hypothetical, candidates = "EASY", None, first
            for movie in candidates:
                movie["ce_score"] = movie["movie_score"]
        else:
            route = "HARD"
            hyde_dense, hypothetical, _ = self.hyde.get_hyde_vector(clean_query)
            dense_hits, sparse_hits = hybrid_retrieval(hyde_dense, sparse, query_filter=query_filter)
            second = self.aggregator.aggregate(self.rank_fusion.fuse(dense_hits, sparse_hits))
            candidates = self.reranker.rerank(clean_query, second[:20])
        return self.final_scorer.score_and_filter(candidates, top_n), route, hypothetical
