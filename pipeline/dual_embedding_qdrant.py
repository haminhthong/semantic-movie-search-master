"""Tạo vector dense/sparse và index một tài liệu cho mỗi phim vào Qdrant."""

import logging
import time
import uuid
from pathlib import Path

import pandas as pd
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from retrieval.config import COLLECTION_NAME, DENSE_MODEL, SPARSE_MODEL, settings

logger = logging.getLogger(__name__)
INPUT_FILE = Path(__file__).resolve().parent / "data" / "movies_clean.csv"
BATCH_SIZE = 32
MAX_RETRIES = 3


def normalize_document(value) -> str:
    """Chuẩn hóa tài liệu trước khi embedding."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().split())


def safe_int(value, default: int = 0) -> int:
    """Chuyển số từ CSV sang int và xử lý NaN/rỗng an toàn."""
    numeric = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(numeric) else int(numeric)


def safe_float(value, default: float = 0.0) -> float:
    """Chuyển số từ CSV sang float và xử lý NaN/rỗng an toàn."""
    numeric = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(numeric) else float(numeric)


def safe_text(value) -> str:
    """Chuyển giá trị CSV sang chuỗi mà không để lộ literal ``nan``."""
    return "" if pd.isna(value) else str(value)


def _ensure_collection(client: QdrantClient, dense_size: int) -> None:
    """Tạo collection khi chưa có và bảo đảm payload index cần thiết tồn tại."""
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(size=dense_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

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
    """Đọc dữ liệu sạch, tạo vector theo lô và upsert vào Qdrant."""
    settings.require_qdrant()
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu đầu vào: {input_file}")

    dataframe = pd.read_csv(input_file)
    required = {"movie_id", "combined_text"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {', '.join(sorted(missing))}")
    dataframe = dataframe[dataframe["combined_text"].map(normalize_document).str.len() > 0].copy()
    if dataframe.empty:
        raise ValueError("Không có tài liệu hợp lệ để index.")

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60.0,
    )
    dense_model = SentenceTransformer(DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    _ensure_collection(client, dense_model.get_sentence_embedding_dimension())

    for offset in tqdm(range(0, len(dataframe), BATCH_SIZE), desc="Đang index"):
        batch = dataframe.iloc[offset : offset + BATCH_SIZE]
        documents = [normalize_document(value) for value in batch["combined_text"]]
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

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
                break
            except Exception:
                if attempt == MAX_RETRIES:
                    raise
                logger.exception("Upsert lần %d/%d thất bại; sẽ thử lại", attempt, MAX_RETRIES)
                time.sleep(2**attempt)

    logger.info("Đã index %d phim vào collection %s", len(dataframe), COLLECTION_NAME)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    process_dual_embedding()
