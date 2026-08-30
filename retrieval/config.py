"""Cấu hình dùng chung cho quá trình tạo chỉ mục (indexing) và tìm kiếm (retrieval).

Module này định nghĩa các thông số toàn cục của hệ thống như tên tập hợp Qdrant,
các mô hình embedding (Dense/Sparse/Cross-Encoder) và các ngưỡng quyết định
cho Adaptive Router.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Tải các biến môi trường từ tệp .env
load_dotenv()

# Tên bộ sưu tập lưu trữ vector trong cơ sở dữ liệu Qdrant
COLLECTION_NAME: str = "movies_hybrid_collection"

# Mô hình Dense Embedding dùng để mã hóa ý nghĩa ngữ cảnh (Sentence-Transformers)
DENSE_MODEL: str = "all-MiniLM-L6-v2"

# Mô hình Sparse Embedding dùng để trích xuất từ khóa tần suất BM25 (FastEmbed)
SPARSE_MODEL: str = "Qdrant/bm25"

# Mô hình Cross-Encoder dùng để xếp hạng lại (reranking) độ tương quan chi tiết giữa query và document
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Mô hình ngôn ngữ lớn (LLM) trên Groq dùng cho kỹ thuật HyDE (Hypothetical Document Embeddings)
GROQ_MODEL: str = "llama-3.1-8b-instant"


@dataclass(frozen=True)
class Settings:
    """Quản lý thông số cấu hình hệ thống và môi trường.

    Attributes:
        qdrant_url: Đường dẫn URL đến Qdrant Cloud hoặc dịch vụ Qdrant local.
        qdrant_api_key: Khóa API xác thực Qdrant (nếu có).
        groq_api_key: Khóa API Groq phục vụ mở rộng truy vấn HyDE qua LLM.
        retrieval_k: Số lượng ứng viên tối đa lấy về ở bước truy hồi dense/sparse.
        candidate_k: Số lượng ứng viên hàng đầu sau khi gộp bằng RRF.
        rrf_k: Hệ số làm mượt trong công thức gộp hạng Reciprocal Rank Fusion (mặc định 60).
        confidence_gap: Khoảng cách điểm RRF giữa top 1 và top 2 để quyết định truy vấn dễ (EASY).
        minimum_score: Điểm RRF tối thiểu của top 1 để coi là truy vấn tin cậy.
    """

    qdrant_url: str = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY") or None
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY") or None
    retrieval_k: int = 100
    candidate_k: int = 20
    rrf_k: int = 60
    confidence_gap: float = 0.01
    minimum_score: float = 0.03

    def require_qdrant(self) -> None:
        """Kiểm tra sự tồn tại của cấu hình kết nối Qdrant URL.

        Raises:
            RuntimeError: Nếu biến môi trường QDRANT_URL bị rỗng hoặc chưa thiết lập.
        """
        if not self.qdrant_url:
            raise RuntimeError("Thiếu biến môi trường QDRANT_URL. Vui lòng cấu hình trong tệp .env.")


# Đối tượng cấu hình dùng chung toàn hệ thống
settings: Settings = Settings()

