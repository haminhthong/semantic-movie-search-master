"""Thu thập metadata phim từ TMDB và ghi ra tệp CSV thô."""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OUTPUT_FILE = Path(__file__).resolve().parent / "data" / "movies_raw.csv"
BASE_URL = "https://api.themoviedb.org/3"
REQUEST_DELAY = 0.05
REQUEST_TIMEOUT = 15


def _request_json(path: str, params: dict) -> dict:
    """Gọi TMDB và trả JSON; lỗi HTTP được chuyển thành ngoại lệ rõ ràng."""
    if not TMDB_API_KEY:
        raise RuntimeError("Thiếu biến môi trường TMDB_API_KEY.")
    response = requests.get(
        f"{BASE_URL}/{path.lstrip('/')}",
        params={"api_key": TMDB_API_KEY, **params},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def fetch_genre_mapping() -> dict[int, str]:
    """Lấy bảng ánh xạ mã thể loại sang tên tiếng Anh."""
    data = _request_json("genre/movie/list", {"language": "en-US"})
    return {
        int(item["id"]): str(item["name"])
        for item in data.get("genres", [])
        if item.get("id") is not None and item.get("name")
    }


def fetch_movie_details(movie_id: int) -> dict:
    """Lấy đạo diễn, năm diễn viên đầu tiên và từ khóa của một phim."""
    try:
        data = _request_json(
            f"movie/{movie_id}",
            {"append_to_response": "credits,keywords"},
        )
    except requests.RequestException:
        logger.exception("Không thể lấy thông tin chi tiết của phim %s", movie_id)
        return {"director": "", "cast": [], "keywords": []}
    finally:
        time.sleep(REQUEST_DELAY)

    crew = data.get("credits", {}).get("crew", [])
    director = next(
        (member.get("name", "") for member in crew if member.get("job") == "Director"),
        "",
    )
    cast = [member.get("name", "") for member in data.get("credits", {}).get("cast", [])[:5]]
    keywords = [item.get("name", "") for item in data.get("keywords", {}).get("keywords", [])]
    return {
        "director": director,
        "cast": [name for name in cast if name],
        "keywords": [keyword for keyword in keywords if keyword],
    }


def _release_year(release_date: str, fallback: int) -> int:
    """Lấy năm từ ngày phát hành và dùng năm truy vấn khi ngày bị thiếu."""
    try:
        return int(str(release_date)[:4])
    except (TypeError, ValueError):
        return fallback


def fetch_tmdb_movies_by_year(
    start_year: int = 2023,
    end_year: int = 2024,
    max_pages_per_year: int = 10,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """Thu thập phim theo năm, làm sạch tối thiểu và ghi kết quả ra CSV."""
    current_year = pd.Timestamp.utcnow().year + 1
    if not 1888 <= start_year <= end_year <= current_year:
        raise ValueError(f"Khoảng năm phải nằm trong 1888–{current_year}.")
    if max_pages_per_year <= 0:
        raise ValueError("max_pages_per_year phải lớn hơn 0.")

    genre_mapping = fetch_genre_mapping()
    records: list[dict] = []
    for year in range(start_year, end_year + 1):
        for page in range(1, max_pages_per_year + 1):
            try:
                data = _request_json(
                    "discover/movie",
                    {
                        "language": "en-US",
                        "page": page,
                        "sort_by": "popularity.desc",
                        "primary_release_year": year,
                        "vote_count.gte": 100,
                        "vote_average.gte": 6.5,
                    },
                )
            except requests.RequestException:
                logger.exception("Không thể tải năm %d, trang %d", year, page)
                break

            movies = data.get("results", [])
            if not movies:
                break
            for movie in movies:
                movie_id = movie.get("id")
                if movie_id is None:
                    continue
                details = fetch_movie_details(int(movie_id))
                genres = [genre_mapping[item] for item in movie.get("genre_ids", []) if item in genre_mapping]
                release_date = str(movie.get("release_date", ""))
                records.append({
                    "movie_id": int(movie_id),
                    "title": movie.get("title", ""),
                    "overview": movie.get("overview", ""),
                    "release_date": release_date,
                    "release_year": _release_year(release_date, year),
                    "vote_average": movie.get("vote_average", 0.0),
                    "popularity": movie.get("popularity", 0.0),
                    "genres": ", ".join(genres),
                    "director": details["director"],
                    "cast": ", ".join(details["cast"]),
                    "keywords": ", ".join(details["keywords"]),
                    "original_language": movie.get("original_language", ""),
                    "poster_path": movie.get("poster_path") or "",
                })
            logger.info("Đã tải năm %d, trang %d; tổng cộng %d phim", year, page, len(records))
            time.sleep(REQUEST_DELAY)

    dataframe = pd.DataFrame(records)
    if dataframe.empty:
        logger.warning("TMDB không trả về phim nào.")
        return dataframe
    dataframe["overview"] = dataframe["overview"].fillna("").astype(str)
    dataframe = dataframe[dataframe["overview"].str.strip().str.len() > 0]
    dataframe = dataframe.drop_duplicates(subset=["movie_id"], keep="first")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_file, index=False, encoding="utf-8")
    logger.info("Đã ghi %d phim vào %s", len(dataframe), output_file)
    return dataframe


def main() -> None:
    """Chạy ingestion với phạm vi dữ liệu mặc định của dự án."""
    fetch_tmdb_movies_by_year(start_year=1990, end_year=2026, max_pages_per_year=25)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
