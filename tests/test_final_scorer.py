from retrieval.final_scorer import FinalScorer

def test_rating_does_not_override_relevance():
    movies = [
        {"title": "Relevant", "ce_score": 1.0, "vote_average": 1},
        {"title": "Popular", "ce_score": 0.0, "vote_average": 10},
    ]
    assert FinalScorer().score_and_filter(movies)[0]["title"] == "Relevant"

