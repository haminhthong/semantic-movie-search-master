from retrieval.aggregate import DocumentAggregator

def test_document_aggregator_deduplicates_movies():
    hits = [
        {"chunk_id": "a", "rrf_score": .2, "payload": {"movie_id": 1, "title": "A", "document_text": "x"}},
        {"chunk_id": "b", "rrf_score": .1, "payload": {"movie_id": 1, "title": "A"}},
    ]
    result = DocumentAggregator().aggregate(hits)
    assert len(result) == 1
    assert result[0]["document"]["text"] == "x"

