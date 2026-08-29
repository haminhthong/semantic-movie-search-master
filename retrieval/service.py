"""Lớp dịch vụ dùng chung cho API và giao diện."""

import copy
import logging
import time
from collections import OrderedDict

from .search import MovieSearch

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self):
        self.engine = MovieSearch()
        self.cache = OrderedDict()

    def _cached_search(self, query: str, top_n: int, genre: str, year: str):
        key = (query, top_n, genre, year)
        if key not in self.cache:
            self.cache[key] = self.engine.search(query, top_n, genre, year)
            if len(self.cache) > 128:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]

    def search(self, query: str, top_n: int = 10, genre: str = "", year: str = "") -> dict:
        started = time.perf_counter()
        result = copy.deepcopy(self._cached_search(query.strip(), top_n, genre.strip(), year.strip()))
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "search route=%s results=%d latency_ms=%.2f",
            result["route"],
            len(result["movies"]),
            result["latency_ms"],
        )
        return result
