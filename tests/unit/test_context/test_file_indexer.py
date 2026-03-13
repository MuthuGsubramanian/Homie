import pytest
from pathlib import Path
from unittest.mock import MagicMock
from homie_core.context.file_indexer import FileIndexer


@pytest.fixture
def indexer():
    return FileIndexer(vector_store=None, chunk_size=100)


def test_chunk_text(indexer):
    text = "\n".join([f"line {i}" for i in range(20)])
    chunks = indexer._chunk_text(text)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_index_file(indexer, tmp_path):
    f = tmp_path / "test.py"
    f.write_text("def hello():\n    return 'world'\n")
    result = indexer._index_file(f)
    assert result is True


def test_skip_unchanged_file(indexer, tmp_path):
    f = tmp_path / "test.py"
    f.write_text("content")
    indexer._index_file(f)
    result = indexer._index_file(f)
    assert result is False  # already indexed, no change


def test_index_directory(indexer, tmp_path):
    (tmp_path / "a.py").write_text("code a")
    (tmp_path / "b.py").write_text("code b")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01")  # not a text file
    count = indexer.index_directory(tmp_path)
    assert count == 2


def test_queue_file(indexer, tmp_path):
    f = tmp_path / "test.py"
    f.write_text("content")
    indexer.queue_file(f)
    assert len(indexer._queue) == 1


def test_with_vector_store(tmp_path):
    mock_vs = MagicMock()
    indexer = FileIndexer(vector_store=mock_vs, chunk_size=100)
    f = tmp_path / "test.py"
    f.write_text("hello world")
    indexer._index_file(f)
    mock_vs.add_file_chunk.assert_called()
