import pytest
from homie_core.storage.vectors import VectorStore


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "chroma")
    s.initialize()
    return s


def test_add_and_query_episode(store):
    store.add_episode("ep1", "User debugged auth module for 2 hours", {"mood": "frustrated", "tags": '["work"]'})
    results = store.query_episodes("authentication debugging", n=1)
    assert len(results) == 1
    assert "auth" in results[0]["text"]


def test_add_and_query_file_chunk(store):
    store.add_file_chunk("f1", "def hello(): return 'world'", {"file_path": "src/main.py", "chunk_index": "0"})
    results = store.query_files("hello function", n=1)
    assert len(results) == 1
    assert "hello" in results[0]["text"]


def test_delete_by_id(store):
    store.add_episode("ep_del", "temporary episode", {})
    store.delete_episodes(["ep_del"])
    results = store.query_episodes("temporary", n=1)
    assert len(results) == 0 or "temporary" not in results[0].get("text", "")
