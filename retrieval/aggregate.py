"""Convert fused document hits into movie candidates."""
from typing import Any, Dict, List

class DocumentAggregator:
    def __init__(self, top_n_movies: int = 20):
        self.top_n_movies = top_n_movies

    def aggregate(self, ranked_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        movies, seen = [], set()
        for document in ranked_documents:
            payload = document.get("payload", {})
            movie_id = payload.get("movie_id")
            if movie_id is None or movie_id in seen:
                continue
            seen.add(movie_id)
            score = float(document.get("rrf_score", 0.0))
            text = payload.get("document_text", payload.get("chunk_text", ""))
            movies.append({
                "movie_id": movie_id, "title": payload.get("title", "Unknown"),
                "genres": payload.get("genres", ""), "release_date": payload.get("release_date", ""),
                "release_year": payload.get("release_year", 0), "vote_average": payload.get("vote_average", 0.0),
                "popularity": payload.get("popularity", 0.0), "poster_path": payload.get("poster_path", ""),
                "document": {"id": document.get("chunk_id"), "text": text},
                "max_score": score, "movie_score": score,
            })
        return movies[:self.top_n_movies]

ChunkAggregator = DocumentAggregator
