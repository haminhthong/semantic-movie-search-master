"""Lớp dịch vụ tầng trung gian (Service Layer) dùng chung cho FastAPI Backend và Streamlit UI.

Cung cấp cơ chế bộ nhớ đệm LRU Cache để lưu trữ tạm thời các kết quả truy vấn phổ biến,
đồng thời đo lường chính xác thời gian phản hồi (latency_ms) và ghi nhận nhật ký (logging).
"""

import copy
import logging
import time
from collections import OrderedDict
from typing import Any

from .search import MovieSearch

logger = logging.getLogger(__name__)


class SearchService:
    """Quản lý các yêu cầu tìm kiếm, bộ nhớ đệm LRU Cache và ghi log hiệu năng hệ thống."""

    def __init__(self) -> None:
        """Khởi tạo SearchEngine đơn thể và bảng bộ nhớ đệm OrderedDict với kích thước tối đa 128."""
        self.engine = MovieSearch()
        self.cache: OrderedDict[tuple[str, int, str, str], dict[str, Any]] = OrderedDict()

    def _cached_search(
        self,
        query: str,
        top_n: int,
        genre: str,
        year: str,
    ) -> dict[str, Any]:
        """Truy xuất kết quả từ bộ nhớ đệm hoặc gọi MovieSearch engine nếu chưa cache."""
        key = (query, top_n, genre, year)
        if key not in self.cache:
            self.cache[key] = self.engine.search(query, top_n, genre, year)
            # Giới hạn kích thước cache tối đa 128 truy vấn gần nhất
            if len(self.cache) > 128:
                self.cache.popitem(last=False)
        else:
            # Đưa phần tử vừa truy cập xuống cuối danh sách (LRU)
            self.cache.move_to_end(key)
        return self.cache[key]

    def search(
        self,
        query: str,
        top_n: int = 10,
        genre: str = "",
        year: str = "",
    ) -> dict[str, Any]:
        """Thực hiện yêu cầu tìm kiếm, ghi nhận độ trễ (ms) và trả kết quả sao chép an toàn (deepcopy).

        Args:
            query: Câu truy vấn từ người dùng.
            top_n: Số lượng phim trả về.
            genre: Thể loại lọc.
            year: Chuỗi khoảng năm lọc.

        Returns:
            Dict chứa danh sách kết quả phim, tuyến xử lý ("EASY"/"HARD"), HyDE text và latency_ms.
        """
        started = time.perf_counter()

        # Thực hiện tìm kiếm có đệm và tạo bản sao sâu để tránh thay đổi tham chiếu cache
        result = copy.deepcopy(self._cached_search(query.strip(), top_n, genre.strip(), year.strip()))

        # Tính toán độ trễ phản hồi (ms)
        elapsed_ms = (time.perf_counter() - started) * 1000
        result["latency_ms"] = round(elapsed_ms, 2)

        logger.info(
            "Yêu cầu tìm kiếm thành công | Route: %s | Tìm thấy: %d phim | Độ trễ: %.2f ms",
            result["route"],
            len(result["movies"]),
            result["latency_ms"],
        )
        return result
