"""Mở rộng truy vấn khó bằng một cốt truyện giả định."""

import logging
import re

from groq import Groq
from sentence_transformers import SentenceTransformer

from .config import DENSE_MODEL, GROQ_MODEL

logger = logging.getLogger(__name__)


class HyDEProcessor:
    def __init__(
        self,
        api_key: str | None = None,
        encoder=None,
        embedding_model_name: str = DENSE_MODEL,
    ):
        self.encoder = encoder or SentenceTransformer(embedding_model_name)
        self.client = Groq(api_key=api_key) if api_key else None
        self.model_name = GROQ_MODEL
        self.cache: dict[str, str] = {}
        if self.client is None:
            logger.warning("Không có GROQ_API_KEY; HyDE sẽ dùng lại truy vấn gốc.")

    def _generate_hypothetical_document(self, query: str) -> str:
        """Sinh một premise ngắn; trả truy vấn gốc khi Groq không dùng được."""
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
            if len(document) < 20:
                return query
            self.cache[query] = document
            return document
        except Exception:
            logger.exception("Không thể sinh tài liệu HyDE; chuyển sang truy vấn gốc")
            return query

    def expand(self, query: str):
        """Trả vector dense và tài liệu giả định."""
        document = self._generate_hypothetical_document(query)
        vector = self.encoder.encode(document, normalize_embeddings=True).tolist()
        return vector, document
