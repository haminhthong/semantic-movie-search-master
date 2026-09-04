"""Module thực hiện kỹ thuật HyDE (Hypothetical Document Embeddings) để mở rộng truy vấn.

Kỹ thuật HyDE chuyển đổi một truy vấn tìm kiếm ngắn hoặc mập mờ từ người dùng
thành một đoạn tóm tắt cốt truyện giả định (hypothetical document) thông qua LLM.
Đoạn cốt truyện giả định này sau đó được mã hóa thành Dense Vector, giúp cải thiện
đáng kể khả năng khớp ngữ nghĩa với nội dung phim thực tế trong Vector DB.
"""

import logging
import re
from typing import Any

from .config import DENSE_MODEL, GROQ_MODEL

logger = logging.getLogger(__name__)


class HyDEProcessor:
    """Quản lý tiến trình sinh tài liệu giả định và chuyển đổi thành vector mở rộng."""

    def __init__(
        self,
        api_key: str | None = None,
        encoder: Any | None = None,
        embedding_model_name: str = DENSE_MODEL,
    ) -> None:
        """Khởi tạo HyDE processor với Groq API client và Dense Encoder.

        Tái sử dụng mô hình `SentenceTransformer` đã nạp từ trước để tránh lãng phí RAM.

        Args:
            api_key: Khóa API Groq. Nếu không có, HyDE sẽ tự động fallback về câu truy vấn gốc.
            encoder: Mô hình SentenceTransformer dùng chung.
            embedding_model_name: Tên mô hình embedding nếu cần tự khởi tạo.
        """
        if encoder:
            self.encoder = encoder
        else:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(embedding_model_name)

        self.client: Any | None = None
        if api_key:
            try:
                from groq import Groq

                self.client = Groq(api_key=api_key)
            except ImportError:
                logger.warning("Chưa cài đặt thư viện 'groq'; HyDE sẽ tự động fallback về truy vấn gốc.")

        self.model_name = GROQ_MODEL
        self.cache: dict[str, str] = {}

        if self.client is None and not api_key:
            logger.warning(
                "Không tìm thấy GROQ_API_KEY; tiến trình HyDE sẽ tự động fallback về truy vấn gốc."
            )

    def _generate_hypothetical_document(self, query: str) -> str:
        """Sử dụng Groq LLM để sinh một tóm tắt cốt truyện phim giả định khoảng 3 câu.

        Args:
            query: Truy vấn tìm kiếm từ người dùng.

        Returns:
            Chuỗi văn bản cốt truyện giả định (hoặc truy vấn gốc nếu gặp sự cố/bộ nhớ cache).
        """
        if query in self.cache:
            return self.cache[query]
        if self.client is None:
            return query

        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write concise movie plot premises. Return only a three-sentence "
                            "premise, without a title or introductory text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Write a hypothetical movie plot matching this request: {query}",
                    },
                ],
                model=self.model_name,
                temperature=0.3,
                max_tokens=150,
            )
            content = completion.choices[0].message.content or ""
            document = re.sub(r"\s+", " ", content).strip()

            # Nếu kết quả quá ngắn, hủy kết quả và quay về truy vấn gốc
            if len(document) < 20:
                return query

            self.cache[query] = document
            return document

        except Exception:
            logger.exception("Lỗi khi sinh tài liệu HyDE từ Groq LLM; tự động fallback về truy vấn gốc.")
            return query

    def expand(self, query: str) -> tuple[list[float], str]:
        """Tạo cốt truyện giả định và mã hóa thành vector dense embedding.

        Args:
            query: Câu truy vấn đã làm sạch của người dùng.

        Returns:
            Tuple chứa:
            - hyde_vector (list[float]): Vector dense đại diện cho văn bản giả định.
            - hypothetical_doc (str): Nội dung văn bản cốt truyện giả định vừa được sinh.
        """
        document = self._generate_hypothetical_document(query)
        vector = self.encoder.encode(document, normalize_embeddings=True).tolist()
        return vector, document
