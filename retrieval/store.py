"""Truy vấn dense và sparse trên Qdrant."""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from qdrant_client import QdrantClient, models

from .config import COLLECTION_NAME, settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    settings.require_qdrant()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=30.0,
    )


def _query(vector, vector_name: str, limit: int, query_filter=None) -> list[dict]:
    response = get_client().query_points(
        collection_name=COLLECTION_NAME,
        using=vector_name,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [
        {"id": str(point.id), "score": float(point.score), "payload": dict(point.payload or {})}
        for point in response.points
    ]


def dense_search(vector, query_filter=None, limit: int | None = None):
    return _query(vector, "dense", limit or settings.retrieval_k, query_filter)


def sparse_search(vector, query_filter=None, limit: int | None = None):
    sparse = models.SparseVector(indices=vector[0], values=vector[1])
    return _query(sparse, "sparse", limit or settings.retrieval_k, query_filter)


def hybrid_search(dense_vector, sparse_vector, query_filter=None, limit: int | None = None):
    """Chạy hai nhánh song song; tiếp tục nếu chỉ một nhánh gặp lỗi."""
    limit = limit or settings.retrieval_k
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            "dense": pool.submit(dense_search, dense_vector, query_filter, limit),
            "sparse": pool.submit(sparse_search, sparse_vector, query_filter, limit),
        }
        results, errors = {}, []
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:
                logger.exception("Nhánh %s gặp lỗi", name)
                results[name] = []
                errors.append(exc)
    if len(errors) == 2:
        raise RuntimeError("Không thể truy vấn Qdrant.") from errors[0]
    return results["dense"], results["sparse"]
