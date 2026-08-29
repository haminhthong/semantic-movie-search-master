"""Làm sạch dữ liệu TMDB và tạo một tài liệu tìm kiếm cho mỗi phim."""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "movies_raw.csv"
OUTPUT_FILE = BASE_DIR / "data" / "movies_clean.csv"
TEXT_FIELDS = ("title", "overview", "director", "cast", "keywords", "genres")
REQUIRED_COLUMNS = {"movie_id", "title", "overview", "release_date", *TEXT_FIELDS}
DOCUMENT_FIELDS = (
    ("title", "Title"),
    ("director", "Director"),
    ("cast", "Cast"),
    ("genres", "Genres"),
    ("keywords", "Keywords"),
    ("overview", "Overview"),
)


def clean_text(value) -> str:
    """Loại HTML, URL và khoảng trắng dư khỏi một giá trị văn bản."""
    if not isinstance(value, str) or pd.isna(value):
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def create_combined_text(row: pd.Series) -> str:
    """Ghép metadata và overview theo thứ tự ổn định để đưa đi embedding."""
    parts = [f"{label}: {row.get(field, '')}" for field, label in DOCUMENT_FIELDS if row.get(field, "")]
    return ". ".join(parts)


def _release_years(df: pd.DataFrame) -> pd.Series:
    """Ưu tiên năm trong release_date và chỉ dùng release_year cũ làm dự phòng."""
    from_date = pd.to_numeric(df["release_date"].astype(str).str[:4], errors="coerce")
    if "release_year" in df.columns:
        fallback = pd.to_numeric(df["release_year"], errors="coerce")
        from_date = from_date.fillna(fallback)
    return from_date.fillna(0).astype(int)


def process_documents(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """Đọc CSV thô, làm sạch, loại bản ghi lỗi và ghi CSV dùng để index."""
    input_file = Path(input_file)
    output_file = Path(output_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu đầu vào: {input_file}")

    dataframe = pd.read_csv(input_file)
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {', '.join(sorted(missing))}")

    before_count = len(dataframe)
    for field in TEXT_FIELDS:
        dataframe[field] = dataframe[field].map(clean_text)
    dataframe["release_year"] = _release_years(dataframe)
    dataframe["combined_text"] = dataframe.apply(create_combined_text, axis=1)
    dataframe = dataframe[
        (dataframe["overview"].str.len() > 0)
        & (dataframe["combined_text"].str.len() > 0)
    ]
    dataframe = dataframe.drop_duplicates(subset=["movie_id"], keep="first").copy()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_file, index=False, encoding="utf-8")
    logger.info(
        "Đã ghi %d/%d phim sau khi làm sạch vào %s",
        len(dataframe),
        before_count,
        output_file,
    )
    return dataframe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    process_documents()
