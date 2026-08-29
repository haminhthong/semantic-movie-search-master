"""HTTP API cho dịch vụ tìm kiếm phim."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from retrieval.service import SearchService

app = FastAPI(title="MovieScout API", version="0.1.0")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_n: int = Field(default=10, ge=1, le=50)
    genre: str = ""
    year: str = ""


@lru_cache(maxsize=1)
def get_service() -> SearchService:
    return SearchService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(request: SearchRequest, service: Annotated[SearchService, Depends(get_service)]):
    try:
        return service.search(request.query, request.top_n, request.genre, request.year)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
