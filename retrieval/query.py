"""Module xử lý làm sạch và chuyển đổi truy vấn văn bản thành định dạng vector.

Hỗ trợ mã hóa song song hai định dạng vector:
- Dense Vector: Biểu diễn ngữ nghĩa thông qua mô hình Transformer (SentenceTransformers).
- Sparse Vector: Biểu diễn từ khóa dựa trên trọng số BM25 (FastEmbed).
"""

import logging
import re

from .config import DENSE_MODEL, SPARSE_MODEL

logger = logging.getLogger(__name__)


class QueryEncoder:
    """Quản lý các mô hình nhúng (embedding) và thực hiện mã hóa truy vấn tìm kiếm."""

    def __init__(
        self,
        dense_model_name: str = DENSE_MODEL,
        sparse_model_name: str = SPARSE_MODEL,
    ) -> None:
        """Khởi tạo và tải các mô hình dense và sparse embedding vào bộ nhớ.

        Args:
            dense_model_name: Tên mô hình SentenceTransformer cho dense vector.
            sparse_model_name: Tên mô hình FastEmbed cho sparse vector (BM25).
        """
        from fastembed import SparseTextEmbedding
        from sentence_transformers import SentenceTransformer

        logger.info("Đang nạp mô hình dense embedding: %s", dense_model_name)
        self.dense_model = SentenceTransformer(dense_model_name)

        logger.info("Đang nạp mô hình sparse embedding (BM25): %s", sparse_model_name)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)

    @staticmethod
    def clean_query(query: str) -> str:
        """Làm sạch chuỗi truy vấn đầu vào: loại bỏ thẻ HTML, dấu câu và chuẩn hóa khoảng trắng.

        Giữ lại các ký tự từ (bao gồm cả Unicode tiếng Việt/các ngôn ngữ khác).

        Args:
            query: Chuỗi văn bản truy vấn thô.

        Returns:
            Chuỗi văn bản đã qua làm sạch và chuyển thành chữ thường.
        """
        if not isinstance(query, str):
            return ""
        # Loại bỏ các thẻ HTML nếu có
        cleaned = re.sub(r"<[^>]+>", " ", query).lower()
        # Loại bỏ các đường dẫn URL nếu có
        cleaned = re.sub(r"(?:https?://|www\.)\S+", " ", cleaned, flags=re.IGNORECASE)
        # Loại bỏ ký tự đặc biệt/dấu câu nhưng giữ nguyên ký tự Unicode chữ và số
        cleaned = re.sub(r"[^\w\s]", " ", cleaned, flags=re.UNICODE)
        # Chuẩn hóa khoảng trắng thừa
        return re.sub(r"\s+", " ", cleaned).strip()

    def encode(self, raw_query: str) -> tuple[str, list[float], tuple[list[int], list[float]]]:
        """Mã hóa câu truy vấn thô thành vector dense và vector sparse.

        Args:
            raw_query: Truy vấn văn bản từ người dùng.

        Returns:
            Tuple gồm:
            - clean_query (str): Truy vấn đã làm sạch.
            - dense_vector (list[float]): Danh sách các giá trị float biểu diễn dense embedding.
            - sparse_vector (Tuple[list[int], list[float]]): Cặp danh sách chỉ số (indices) và giá trị (values) của BM25 sparse vector.

        Raises:
            ValueError: Nếu truy vấn sau khi làm sạch bị rỗng.
        """
        clean_query = self.clean_query(raw_query)
        if not clean_query:
            raise ValueError("Truy vấn tìm kiếm không được để trống hoặc chỉ chứa ký tự đặc biệt.")

        # Tạo dense vector và chuẩn hóa độ dài vector (L2 norm = 1)
        dense_vector: list[float] = self.dense_model.encode(
            clean_query,
            normalize_embeddings=True,
        ).tolist()

        # Tạo sparse BM25 vector thông qua FastEmbed
        sparse_result = next(iter(self.sparse_model.embed([clean_query])))
        sparse_vector: tuple[list[int], list[float]] = (
            sparse_result.indices.tolist(),
            sparse_result.values.tolist(),
        )

        return clean_query, dense_vector, sparse_vector
