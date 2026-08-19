from retrieval.rrf import RankFusion

def test_rrf_rewards_documents_seen_by_both_retrievers():
    dense = {"chunk_ids": ["a", "b"], "payloads": [{}, {}]}
    sparse = {"chunk_ids": ["b", "c"], "payloads": [{}, {}]}
    assert RankFusion().fuse(dense, sparse)[0]["chunk_id"] == "b"

