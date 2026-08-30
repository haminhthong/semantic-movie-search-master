"""Module tiền xử lý và làm sạch dữ liệu phim TMDB.

Thực hiện quy trình làm sạch văn bản (xóa thẻ HTML, URL, khoảng trắng thừa) và ghép các trường metadata
(tên phim, đạo diễn, diễn viên, thể loại, từ khóa, tóm tắt) thành một văn bản hợp nhất (`combined_text`)
đạt chuẩn để đưa vào các mô hình Dense và Sparse embedding.
"""

import logging
import re
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Thư mục chứa dữ liệu mặc định
BASE_DIR: Path = Path(__file__).resolve().parent
INPUT_FILE: Path = BASE_DIR / "data" / "movies_raw.csv"
OUTPUT_FILE: Path = BASE_DIR / "data" / "movies_clean.csv"

# Danh sách các trường văn bản cần làm sạch
TEXT_FIELDS: Tuple[str, ...] = ("title", "overview", "director", "cast", "keywords", "genres")
REQUIRED_COLUMNS: set[str] = {"movie_id", "title", "overview", "release_date", *TEXT_FIELDS}

# Cấu trúc các trường ghép văn bản tài liệu tìm kiếm
DOCUMENT_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("title", "Title"),
    ("director", "Director"),
    ("cast", "Cast"),
    ("genres", "Genres"),
    ("keywords", "Keywords"),
    ("overview", "Overview"),
)


def clean_text(value: Any) -> str:
    """Loại bỏ thẻ HTML, đường dẫn URL và chuẩn hóa khoảng trắng thừa khỏi một giá trị văn bản thô.

    Args:
        value: Ký tự hoặc chuỗi văn bản đầu vào từ dữ liệu CSV.

    Returns:
        Chuỗi văn bản đã qua làm sạch. Trả về "" nếu dữ liệu bị rỗng/NaN.
    """
    if not isinstance(value, str) or pd.isna(value):
        return ""
    # Xóa thẻ HTML
    text = re.sub(r"<[^>]+>", " ", value)
    # Xóa đường dẫn liên kết URL
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text, flags=re.IGNORECASE)
    # Chuẩn hóa khoảng trắng thừa
    return re.sub(r"\s+", " ", text).strip()


def create_combined_text(row: pd.Series) -> str:
    """Ghép nối các thông tin metadata và overview của phim theo thứ tự cố định để tạo tài liệu embedding.

    Cấu trúc văn bản đầu ra:
        "Title: ... Director: ... Cast: ... Genres: ... Keywords: ... Overview: ..."

    Args:
        row: Một dòng dữ liệu (pandas Series) của bản ghi phim.

    Returns:
        Chuỗi văn bản `combined_text` hoàn chỉnh dùng làm searchable document.
    """
    parts = [
        f"{label}: {row.get(field, '')}"
        for field, label in DOCUMENT_FIELDS
        if row.get(field, "")
    ]
    return ". ".join(parts)


def _release_years(df: pd.DataFrame) -> pd.Series:
    """Trích xuất năm phát hành kiểu số nguyên từ chuỗi ngày `release_date`.

    Ưu tiên lấy 4 ký tự đầu của `release_date` (YYYY), nếu thiếu sẽ lấy từ cột dự phòng `release_year`.

    Args:
        df: DataFrame dữ liệu thô.

    Returns:
        Series kiểu int đại diện cho năm phát hành.
    """
    from_date = pd.to_numeric(df["release_date"].astype(str).str[:4], errors="coerce")
    if "release_year" in df.columns:
        fallback = pd.to_numeric(df["release_year"], errors="coerce")
        from_date = from_date.fillna(fallback)
    return from_date.fillna(0).astype(int)


def process_documents(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """Đọc dữ liệu CSV phim thô, tiến hành làm sạch, tạo tài liệu tìm kiếm và ghi kết quả ra CSV sạch.

    Args:
        input_file: Đường dẫn tệp CSV dữ liệu thô đầu vào.
        output_file: Đường dẫn tệp CSV lưu dữ liệu đã làm sạch.

    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu sạch sẵn sàng để tạo chỉ mục.

    Raises:
        FileNotFoundError: Nếu tệp input_file không tồn tại.
        ValueError: Nếu tệp dữ liệu thiếu các cột cấu trúc bắt buộc.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu đầu vào: {input_file}")

    dataframe = pd.read_csv(input_file)
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {', '.join(sorted(missing))}")

    before_count = len(dataframe)

    # Làm sạch từng trường văn bản
    for field in TEXT_FIELDS:
        dataframe[field] = dataframe[field].map(clean_text)

    # Chuẩn hóa năm và tạo tài liệu combined_text
    dataframe["release_year"] = _release_years(dataframe)
    dataframe["combined_text"] = dataframe.apply(create_combined_text, axis=1)

    # Lọc bỏ bản ghi không có phần tóm tắt overview hoặc combined_text rỗng
    dataframe = dataframe[
        (dataframe["overview"].str.len() > 0)
        & (dataframe["combined_text"].str.len() > 0)
    ]

    # Loại bỏ bản ghi trùng lặp theo movie_id
    dataframe = dataframe.drop_duplicates(subset=["movie_id"], keep="first").copy()

    # Ghi tệp kết quả CSV sạch
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_file, index=False, encoding="utf-8")

    logger.info(
        "Làm sạch hoàn tất! Đã ghi %d/%d bản ghi hợp lệ vào %s",
        len(dataframe),
        before_count,
        output_file,
    )
    return dataframe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    process_documents()

