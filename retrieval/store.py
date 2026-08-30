"""Module tương tác trực tiếp với cơ sở dữ liệu Vector Qdrant.

Cung cấp các hàm thực hiện tìm kiếm vector tương đồng (vector search) cho nhánh
Dense Vector, nhánh Sparse Vector (BM25) và hàm Hybrid Search thực thi song song
bằng đa luồng (ThreadPoolExecutor) để tối ưu hóa hiệu năng.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import COLLECTION_NAME, settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> Any:
    """Tạo hoặc lấy ra kết nối đơn thể (singleton) tới Qdrant Client.

    Sử dụng `@lru_cache(maxsize=1)` để tái sử dụng connection pool trong suốt tiến trình.

    Returns:
        QdrantClient: Đối tượng client tương tác với dịch vụ Qdrant.
    """
    from qdrant_client import QdrantClient

    settings.require_qdrant()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=30.0,
    )



def _query(
    vector: Any,
    vector_name: str,
    limit: int,
    query_filter: Optional[Any] = None,
) -> List[Dict[str, Any]]:

    """Thực hiện truy vấn lấy các điểm dữ liệu (points) tương đồng nhất từ Qdrant.

    Args:
        vector: Vector dense (list float) hoặc vector sparse (models.SparseVector).
        vector_name: Tên không gian vector ("dense" hoặc "sparse").
        limit: Số lượng điểm tối đa cần trả về.
        query_filter: Bộ lọc điều kiện Qdrant Filter (nếu có).

    Returns:
        List[Dict[str, Any]]: Danh sách các từ điển chứa point id, score tương đồng và payload metadata.
    """
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
        {
            "id": str(point.id),
            "score": float(point.score),
            "payload": dict(point.payload or {}),
        }
        for point in response.points
    ]


def dense_search(
    vector: List[float],
    query_filter: Optional[Any] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Tìm kiếm tương đồng theo vector Dense (ngữ nghĩa).

    Args:
        vector: Vector dense biểu diễn câu truy vấn.
        query_filter: Bộ lọc Qdrant Filter (ví dụ: lọc năm, thể loại).
        limit: Số kết quả tối đa. Mặc định lấy theo settings.retrieval_k.

    Returns:
        Danh sách kết quả tìm kiếm từ nhánh Dense.
    """
    return _query(vector, "dense", limit or settings.retrieval_k, query_filter)


def sparse_search(
    vector: Tuple[List[int], List[float]],
    query_filter: Optional[Any] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Tìm kiếm tương đồng theo vector Sparse (BM25 từ khóa).

    Args:
        vector: Tuple gồm (indices, values) biểu diễn sparse BM25.
        query_filter: Bộ lọc Qdrant Filter.
        limit: Số kết quả tối đa. Mặc định lấy theo settings.retrieval_k.

    Returns:
        Danh sách kết quả tìm kiếm từ nhánh Sparse.
    """
    from qdrant_client import models

    sparse = models.SparseVector(indices=vector[0], values=vector[1])
    return _query(sparse, "sparse", limit or settings.retrieval_k, query_filter)



def hybrid_search(
    dense_vector: List[float],
    sparse_vector: Tuple[List[int], List[float]],
    query_filter: Optional[models.Filter] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Thực thi tìm kiếm song song cả hai nhánh Dense và Sparse thông qua ThreadPoolExecutor.

    Nếu một trong hai nhánh gặp lỗi hệ thống, nhánh còn lại vẫn sẽ trả kết quả dự phòng (graceful degradation).
    Chỉ ném ra lỗi RuntimeError nếu cả hai nhánh đều không hoạt động.

    Args:
        dense_vector: Vector dense của truy vấn.
        sparse_vector: Tuple sparse BM25 của truy vấn.
        query_filter: Bộ lọc kết hợp theo năm / thể loại phim.
        limit: Số lượng kết quả ứng viên trả về cho mỗi nhánh.

    Returns:
        Tuple chứa danh sách kết quả (dense_results, sparse_results).

    Raises:
        RuntimeError: Khi cả hai nhánh truy vấn đều gặp ngoại lệ.
    """
    search_limit = limit or settings.retrieval_k
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            "dense": pool.submit(dense_search, dense_vector, query_filter, search_limit),
            "sparse": pool.submit(sparse_search, sparse_vector, query_filter, search_limit),
        }
        results: Dict[str, List[Dict[str, Any]]] = {}
        errors: List[Exception] = []

        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:
                logger.exception("Nhánh truy hồi %s gặp lỗi", name)
                results[name] = []
                errors.append(exc)

    if len(errors) == 2:
        raise RuntimeError("Không thể truy vấn cơ sở dữ liệu Qdrant ở cả 2 nhánh Dense và Sparse.") from errors[0]

    return results["dense"], results["sparse"]

