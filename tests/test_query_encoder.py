from retrieval.query import QueryEncoder

def test_clean_query_preserves_vietnamese_without_loading_models():
    encoder = QueryEncoder.__new__(QueryEncoder)
    assert encoder.clean_query("Phim về vũ trụ!") == "phim về vũ trụ"

