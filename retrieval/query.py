"""Làm sạch và mã hóa truy vấn cho hai nhánh truy hồi."""

import logging
import re

from .config import DENSE_MODEL, SPARSE_MODEL

logger = logging.getLogger(__name__)


class QueryEncoder:
    def __init__(
        self,
        dense_model_name: str = DENSE_MODEL,
        sparse_model_name: str = SPARSE_MODEL,
    ):
        from fastembed import SparseTextEmbedding
        from sentence_transformers import SentenceTransformer

        logger.info("Đang nạp mô hình dense %s", dense_model_name)
        self.dense_model = SentenceTransformer(dense_model_name)
        logger.info("Đang nạp mô hình sparse %s", sparse_model_name)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)

    @staticmethod
    def clean_query(query: str) -> str:
        """Bỏ HTML và dấu câu, đồng thời giữ lại chữ cái Unicode."""
        if not isinstance(query, str):
            return ""
        cleaned = re.sub(r"<[^>]+>", " ", query).lower()
        cleaned = re.sub(r"[^\w\s]", " ", cleaned, flags=re.UNICODE)
        return re.sub(r"\s+", " ", cleaned).strip()

    def encode(self, raw_query: str) -> tuple[str, list[float], tuple[list[int], list[float]]]:
        """Trả truy vấn sạch, vector dense và vector sparse."""
        clean_query = self.clean_query(raw_query)
        if not clean_query:
            raise ValueError("Truy vấn không được để trống.")
        dense_vector = self.dense_model.encode(
            clean_query,
            normalize_embeddings=True,
        ).tolist()
        sparse_result = next(iter(self.sparse_model.embed([clean_query])))
        sparse_vector = (
            sparse_result.indices.tolist(),
            sparse_result.values.tolist(),
        )
        return clean_query, dense_vector, sparse_vector
