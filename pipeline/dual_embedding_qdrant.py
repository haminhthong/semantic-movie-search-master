"""Module tạo vector kép (Dual Embedding) và index dữ liệu vào Vector DB Qdrant.

Sử dụng đồng thời:
1. Mô hình Dense Embedding (`all-MiniLM-L6-v2`) để mã hóa ngữ nghĩa văn bản.
2. Mô hình Sparse Embedding (`Qdrant/bm25`) để mã hóa từ khóa theo thuật toán BM25.
Dữ liệu vector và payload metadata được đưa vào Qdrant collection với cơ chế thử lại (retry mechanism).
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from retrieval.config import COLLECTION_NAME, DENSE_MODEL, SPARSE_MODEL, settings

logger = logging.getLogger(__name__)

# Thư mục dữ liệu mặc định
INPUT_FILE: Path = Path(__file__).resolve().parent / "data" / "movies_clean.csv"
BATCH_SIZE: int = 32
MAX_RETRIES: int = 3


def normalize_document(value: Any) -> str:
    """Chuẩn hóa văn bản trước khi đưa đi embedding (chuyển chữ thường, xóa khoảng trắng thừa).

    Args:
        value: Giá trị văn bản đầu vào.

    Returns:
        Chuỗi văn bản đã qua chuẩn hóa.
    """
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().split())


def safe_int(value: Any, default: int = 0) -> int:
    """Chuyển đổi số từ CSV sang kiểu int an toàn, xử lý NaN/rỗng mà không gây ngoại lệ.

    Args:
        value: Giá trị số thô từ pandas CSV.
        default: Giá trị mặc định nếu chuyển đổi thất bại.

    Returns:
        Số nguyên int.
    """
    numeric = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(numeric) else int(numeric)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Chuyển đổi số từ CSV sang kiểu float an toàn, xử lý NaN/rỗng mà không gây ngoại lệ.

    Args:
        value: Giá trị số thô từ pandas CSV.
        default: Giá trị mặc định nếu chuyển đổi thất bại.

    Returns:
        Số thực float.
    """
    numeric = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(numeric) else float(numeric)


def safe_text(value: Any) -> str:
    """Chuyển giá trị CSV sang chuỗi văn bản an toàn mà không để lộ chuỗi literal "nan".

    Args:
        value: Giá trị từ CSV.

    Returns:
        Chuỗi văn bản hoặc "" nếu là NaN/rỗng.
    """
    return "" if pd.isna(value) else str(value)


def _ensure_collection(client: QdrantClient, dense_size: int) -> None:
    """Tạo bộ sưu tập (collection) Qdrant và thiết lập các payload index hỗ trợ lọc điều kiện (Filter).

    Args:
        client: Đối tượng QdrantClient kết nối tới database.
        dense_size: Số chiều (dimensions) của mô hình Dense Embedding (ví dụ: 384 cho MiniLM).
    """
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": models.VectorParams(size=dense_size, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

    # Đăng ký chỉ mục tìm kiếm (Payload Index) hỗ trợ Filter
    indexes = {
        "movie_id": models.PayloadSchemaType.INTEGER,
        "title": models.PayloadSchemaType.TEXT,
        "genres": models.PayloadSchemaType.TEXT,
        "release_date": models.PayloadSchemaType.TEXT,
        "release_year": models.PayloadSchemaType.INTEGER,
    }
    for field_name, field_schema in indexes.items():
        client.create_payload_index(COLLECTION_NAME, field_name, field_schema)


def process_dual_embedding(input_file: Path = INPUT_FILE) -> None:
    """Đọc dữ liệu phim đã làm sạch, mã hóa vector dense/sparse theo từng lô (batch) và upsert vào Qdrant.

    Args:
        input_file: Đường dẫn tệp CSV dữ liệu đã làm sạch.

    Raises:
        FileNotFoundError: Nếu không tìm thấy tệp CSV đầu vào.
        ValueError: Nếu tệp CSV thiếu thông tin bắt buộc hoặc rỗng.
    """
    settings.require_qdrant()
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu đã làm sạch: {input_file}")

    dataframe = pd.read_csv(input_file)
    required = {"movie_id", "combined_text"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {', '.join(sorted(missing))}")

    dataframe = dataframe[dataframe["combined_text"].map(normalize_document).str.len() > 0].copy()
    if dataframe.empty:
        raise ValueError("Không có tài liệu hợp lệ nào để tạo chỉ mục.")

    # Khởi tạo kết nối Qdrant và các mô hình Embedding
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60.0,
    )
    dense_model = SentenceTransformer(DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    _ensure_collection(client, dense_model.get_sentence_embedding_dimension())

    # Duyệt qua dữ liệu theo từng lô (Batching)
    for offset in tqdm(range(0, len(dataframe), BATCH_SIZE), desc="Đang index dữ liệu phim"):
        batch = dataframe.iloc[offset : offset + BATCH_SIZE]
        documents = [normalize_document(val) for val in batch["combined_text"]]

        # Tạo Dense & Sparse vectors cho cả lô
        dense_vectors = dense_model.encode(
            documents,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
        sparse_vectors = list(sparse_model.embed(documents))

        points = []
        for position, (_, row) in enumerate(batch.iterrows()):
            movie_id = safe_int(row["movie_id"])
            if movie_id <= 0:
                logger.warning("Bỏ qua bản ghi có movie_id không hợp lệ: %r", row["movie_id"])
                continue

            payload = {
                "movie_id": movie_id,
                "document_text": safe_text(row["combined_text"]),
                "title": safe_text(row.get("title", "")),
                "genres": safe_text(row.get("genres", "")),
                "release_date": safe_text(row.get("release_date", "")),
                "release_year": safe_int(row.get("release_year", 0)),
                "vote_average": safe_float(row.get("vote_average", 0)),
                "popularity": safe_float(row.get("popularity", 0)),
                "poster_path": safe_text(row.get("poster_path", "")),
            }
            sparse = sparse_vectors[position]

            # Tạo Qdrant PointStruct với UUID cố định dựa trên movie_id
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"movie_{movie_id}")),
                    vector={
                        "dense": dense_vectors[position],
                        "sparse": models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )

        if not points:
            continue

        # Thử lại upsert nếu gặp lỗi kết nối mạng chập chờn
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
                break
            except Exception:
                if attempt == MAX_RETRIES:
                    raise
                logger.exception("Upsert lần %d/%d thất bại; đang chờ thử lại...", attempt, MAX_RETRIES)
                time.sleep(2**attempt)

    logger.info("Hoàn tất Indexing! Đã đưa %d bộ phim vào collection %s", len(dataframe), COLLECTION_NAME)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    process_dual_embedding()
