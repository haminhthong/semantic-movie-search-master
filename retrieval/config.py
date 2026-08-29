"""Cấu hình dùng chung cho indexing và tìm kiếm."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "movies_hybrid_collection"
DENSE_MODEL = "all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    groq_api_key: str | None = os.getenv("GROQ_API_KEY") or None
    retrieval_k: int = 100
    candidate_k: int = 20
    rrf_k: int = 60
    confidence_gap: float = 0.01
    minimum_score: float = 0.03

    def require_qdrant(self) -> None:
        if not self.qdrant_url:
            raise RuntimeError("Thiếu biến môi trường QDRANT_URL.")


settings = Settings()
