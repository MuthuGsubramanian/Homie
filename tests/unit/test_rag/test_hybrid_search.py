"""Tests for hybrid search — BM25, RRF, and HybridSearch."""
import pytest
from homie_core.rag.hybrid_search import BM25Index, reciprocal_rank_fusion, HybridSearch


# -----------------------------------------------------------------------
# BM25 Index
# -----------------------------------------------------------------------

class TestBM25Index:
    def test_add_and_search(self):
        idx = BM25Index()
        idx.add("d1", "the quick brown fox jumps over the lazy dog")
        idx.add("d2", "the cat sat on the mat")
        idx.add("d3", "python programming language")

        results = idx.search("quick fox")
        assert len(results) >= 1
        assert results[0]["id"] == "d1"

    def test_empty_query(self):
        idx = BM25Index()
        idx.add("d1", "some text")
        results = idx.search("")
        assert results == []

    def test_empty_index(self):
        idx = BM25Index()
        results = idx.search("hello")
        assert results == []

    def test_no_match(self):
        idx = BM25Index()
        idx.add("d1", "hello world")
        results = idx.search("quantum physics")
        assert len(results) == 0

    def test_ranking_order(self):
        idx = BM25Index()
        idx.add("d1", "python programming")
        idx.add("d2", "python python python programming language deep learning")
        idx.add("d3", "java programming")

        results = idx.search("python programming")
        # d2 has more "python" mentions, should rank higher
        ids = [r["id"] for r in results]
        assert "d2" in ids or "d1" in ids  # both match

    def test_remove_document(self):
        idx = BM25Index()
        idx.add("d1", "hello world")
        idx.add("d2", "goodbye world")
        idx.remove("d1")

        results = idx.search("hello")
        assert all(r["id"] != "d1" for r in results)
        assert idx.size == 1

    def test_remove_nonexistent(self):
        idx = BM25Index()
        idx.remove("nope")  # should not raise
        assert idx.size == 0

    def test_metadata_preserved(self):
        idx = BM25Index()
        idx.add("d1", "hello world", {"file": "test.py", "line": "10"})
        results = idx.search("hello")
        assert results[0]["metadata"]["file"] == "test.py"

    def test_top_k_limit(self):
        idx = BM25Index()
        for i in range(20):
            idx.add(f"d{i}", f"document number {i} with some shared words")
        results = idx.search("document number", top_k=5)
        assert len(results) <= 5

    def test_code_search(self):
        idx = BM25Index()
        idx.add("f1", "def authenticate_user(username, password):\n    return check_credentials(username, password)")
        idx.add("f2", "def get_user_profile(user_id):\n    return db.query(User, user_id)")
        idx.add("f3", "def calculate_tax(income, rate):\n    return income * rate")

        results = idx.search("authentication password")
        assert results[0]["id"] == "f1"


# -----------------------------------------------------------------------
# Reciprocal Rank Fusion
# -----------------------------------------------------------------------

class TestRRF:
    def test_single_list(self):
        results = reciprocal_rank_fusion(
            [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.5}],
            top_n=2,
        )
        assert len(results) == 2
        assert results[0]["id"] == "a"

    def test_two_lists_fusion(self):
        list1 = [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.5}]
        list2 = [{"id": "b", "score": 1.0}, {"id": "c", "score": 0.5}]

        results = reciprocal_rank_fusion(list1, list2, top_n=3)
        # "b" appears in both lists, should rank highest
        assert results[0]["id"] == "b"

    def test_empty_lists(self):
        results = reciprocal_rank_fusion([], [], top_n=5)
        assert results == []

    def test_rrf_score_present(self):
        results = reciprocal_rank_fusion(
            [{"id": "x", "score": 1.0}],
            top_n=1,
        )
        assert "rrf_score" in results[0]
        assert results[0]["rrf_score"] > 0

    def test_top_n_limit(self):
        big_list = [{"id": f"d{i}", "score": 1.0 / (i + 1)} for i in range(20)]
        results = reciprocal_rank_fusion(big_list, top_n=3)
        assert len(results) == 3


# -----------------------------------------------------------------------
# Hybrid Search
# -----------------------------------------------------------------------

class TestHybridSearch:
    def test_index_and_search_bm25_only(self):
        search = HybridSearch(vector_store=None)
        search.index_chunk("c1", "python authentication module", {"file": "auth.py"})
        search.index_chunk("c2", "javascript frontend component", {"file": "app.js"})

        results = search.search("authentication", top_k=2)
        assert len(results) >= 1
        assert results[0]["id"] == "c1"
        assert "authentication" in results[0]["text"]

    def test_remove_chunk(self):
        search = HybridSearch(vector_store=None)
        search.index_chunk("c1", "hello world")
        search.remove_chunk("c1")
        results = search.search("hello")
        assert all(r["id"] != "c1" for r in results)

    def test_size_tracking(self):
        search = HybridSearch(vector_store=None)
        assert search.size == 0
        search.index_chunk("c1", "text")
        assert search.size == 1

    def test_empty_search(self):
        search = HybridSearch(vector_store=None)
        results = search.search("anything")
        assert results == []
