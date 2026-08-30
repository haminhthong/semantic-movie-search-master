"""FastAPI RESTful API Backend cho dịch vụ MovieScout AI.

Cung cấp các endpoints:
- GET `/health`: Kiểm tra trạng thái hoạt động của dịch vụ (Health Check).
- POST `/search`: Tìm kiếm phim ngữ nghĩa thích ứng (Adaptive Semantic Search).
"""

from functools import lru_cache
from typing import Annotated, Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from retrieval.service import SearchService

# Khởi tạo ứng dụng FastAPI với thông tin mô tả chi tiết cho OpenAPI / Swagger UI
app = FastAPI(
    title="MovieScout AI API",
    description=(
        "API Tìm kiếm phim ngữ nghĩa lai (Hybrid Semantic Search) kết hợp "
        "BM25, Dense Vector, RRF, Adaptive Routing, HyDE và Cross-Encoder Reranking."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# === Pydantic Request Models ===

class SearchRequest(BaseModel):
    """Mô hình dữ liệu đầu vào cho yêu cầu tìm kiếm phim."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Mô tả nội dung cốt truyện, thể loại hoặc từ khóa phim.",
        example="A team of astronauts travels through a wormhole in space to save humanity",
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Số lượng bộ phim kết quả tối đa cần trả về (1-50).",
        example=10,
    )
    genre: str = Field(
        default="",
        description="Tên thể loại phim muốn lọc (ví dụ: 'Action', 'Sci-Fi'). Để rỗng nếu tìm tất cả.",
        example="Science Fiction",
    )
    year: str = Field(
        default="",
        description="Năm hoặc khoảng năm phát hành (ví dụ: '2014' hoặc '2010-2020').",
        example="2010-2020",
    )


# === Pydantic Response Models ===

class MovieDocumentItem(BaseModel):
    id: str
    text: str


class MovieItemResponse(BaseModel):
    """Mô hình dữ liệu của một bộ phim kết quả."""

    movie_id: int = Field(..., description="ID duy nhất của phim từ TMDB")
    title: str = Field(..., description="Tiêu đề bộ phim")
    genres: str = Field("", description="Danh sách các thể loại phim")
    release_date: str = Field("", description="Ngày phát hành (YYYY-MM-DD)")
    release_year: int = Field(0, description="Năm phát hành")
    vote_average: float = Field(0.0, description="Điểm đánh giá trung bình trên TMDB (0-10)")
    popularity: float = Field(0.0, description="Chỉ số độ phổ biến TMDB")
    poster_path: str = Field("", description="Đường dẫn ảnh poster TMDB")
    document: Dict[str, Any] = Field(..., description="Tài liệu văn bản đã mã hóa")
    relevance_score: float = Field(..., description="Điểm tương quan thô từ RRF / Cross-Encoder")
    final_score: float = Field(..., description="Điểm tương quan đã chuẩn hóa Min-Max [0.0 - 1.0]")


class SearchResponse(BaseModel):
    """Mô hình dữ liệu trả về cho API tìm kiếm."""

    movies: List[MovieItemResponse] = Field(..., description="Danh sách kết quả phim xếp hạng")
    route: str = Field(..., description="Tuyến xử lý Adaptive Router ('EASY' hoặc 'HARD')")
    hyde: Optional[str] = Field(None, description="Đoạn văn bản cốt truyện giả định sinh bởi HyDE (nếu có)")
    latency_ms: float = Field(..., description="Thời gian xử lý phản hồi (tính bằng miligiây)")


class HealthResponse(BaseModel):
    """Mô hình dữ liệu kiểm tra sức khỏe hệ thống."""

    status: str = Field("ok", description="Trạng thái hệ thống")
    service: str = Field("MovieScout AI Search Engine", description="Tên dịch vụ")


# === Dependency Injection ===

@lru_cache(maxsize=1)
def get_service() -> SearchService:
    """Tạo hoặc trả về đối tượng SearchService đơn thể (Singleton)."""
    return SearchService()


# === API Endpoints ===

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Kiểm tra trạng thái hoạt động dịch vụ",
)
def health() -> HealthResponse:
    """Endpoint kiểm tra sức khỏe của API service."""
    return HealthResponse(status="ok", service="MovieScout AI Search Engine")


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search Engine"],
    summary="Tìm kiếm phim ngữ nghĩa thích ứng",
    responses={
        422: {"description": "Tham số truy vấn đầu vào không hợp lệ"},
        503: {"description": "Dịch vụ Qdrant hoặc AI models không sẵn sàng"},
    },
)
def search(
    request: SearchRequest,
    service: Annotated[SearchService, Depends(get_service)],
) -> SearchResponse:
    """Thực hiện quy trình tìm kiếm phim kết hợp Dense + Sparse retrieval, RRF fusion, Adaptive Routing, HyDE và Reranking.

    - **query**: Mô tả cốt truyện phim.
    - **top_n**: Số phim kết quả mong muốn.
    - **genre**: Lọc theo thể loại (tùy chọn).
    - **year**: Lọc theo năm phát hành (tùy chọn).
    """
    try:
        raw_result = service.search(
            query=request.query,
            top_n=request.top_n,
            genre=request.genre,
            year=request.year,
        )
        return SearchResponse(**raw_result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

