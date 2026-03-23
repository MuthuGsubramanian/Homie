"""Tests for LocalEmbedder — sentence-transformers or TF-IDF fallback."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from homie_core.ml.embedder import LocalEmbedder, _TfidfFallback, _cosine_similarity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORPUS = [
    "the cat sat on the mat",
    "the dog barked at the cat",
    "a quick brown fox jumped",
    "the rain in spain falls mainly",
    "machine learning is fascinating",
    "deep learning models are powerful",
]


@pytest.fixture
def embedder():
    """Create an embedder forced to TF-IDF fallback."""
    e = LocalEmbedder(model_name="test-model")
    e._backend = "tfidf"
    e._sbert_model = None
    e._trained = False
    return e


@pytest.fixture
def trained_embedder(embedder):
    embedder.train(CORPUS)
    return embedder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLocalEmbedderInit:
    def test_model_type(self, embedder):
        assert embedder.model_type == "embedder"

    def test_name(self, embedder):
        assert embedder.name == "test-model"


class TestLocalEmbedderTrain:
    def test_train_returns_metrics(self, embedder):
        metrics = embedder.train(CORPUS)
        assert "vocab_size" in metrics
        assert metrics["n_texts"] == len(CORPUS)

    def test_is_trained_after_fit(self, embedder):
        embedder.train(CORPUS)
        assert embedder.is_trained is True


class TestLocalEmbedderEmbed:
    def test_embed_returns_vectors(self, trained_embedder):
        vecs = trained_embedder.embed(["hello world"])
        assert isinstance(vecs, list)
        assert isinstance(vecs[0], list)
        assert all(isinstance(v, float) for v in vecs[0])

    def test_embed_batch(self, trained_embedder):
        vecs = trained_embedder.embed(["hello", "world", "foo"])
        assert len(vecs) == 3

    def test_embed_vectors_normalized(self, trained_embedder):
        vecs = trained_embedder.embed(["the cat sat"])
        import math
        norm = math.sqrt(sum(v * v for v in vecs[0]))
        assert abs(norm - 1.0) < 0.01 or norm == 0.0  # zero vector for OOV is ok

    def test_embed_before_train_raises(self, embedder):
        with pytest.raises(RuntimeError, match="not been fitted"):
            embedder.embed(["hello"])


class TestLocalEmbedderSimilarity:
    def test_similar_texts_high_score(self, trained_embedder):
        score = trained_embedder.similarity("the cat sat on the mat", "the cat on a mat")
        assert score > 0.5

    def test_dissimilar_texts_lower_score(self, trained_embedder):
        score_sim = trained_embedder.similarity("the cat sat", "the cat on the mat")
        score_dis = trained_embedder.similarity("the cat sat", "machine learning models")
        assert score_sim > score_dis


class TestLocalEmbedderPersistence:
    def test_save_and_load(self, trained_embedder, tmp_path):
        path = tmp_path / "embedder.json"
        trained_embedder.save(path)
        assert path.exists()

        loaded = LocalEmbedder(model_name="empty")
        loaded._backend = "tfidf"
        loaded._sbert_model = None
        loaded.load(path)
        assert loaded.is_trained

    def test_loaded_model_embeds(self, trained_embedder, tmp_path):
        path = tmp_path / "embedder.json"
        trained_embedder.save(path)

        loaded = LocalEmbedder(model_name="empty")
        loaded._backend = "tfidf"
        loaded._sbert_model = None
        loaded.load(path)
        vecs = loaded.embed(["the cat sat"])
        assert len(vecs) == 1

    def test_load_nonexistent_raises(self, embedder, tmp_path):
        with pytest.raises(FileNotFoundError):
            embedder.load(tmp_path / "nope.json")


class TestTfidfFallback:
    def test_fit_and_transform(self):
        tfidf = _TfidfFallback()
        tfidf.fit(["hello world", "foo bar"])
        vecs = tfidf.transform(["hello"])
        assert len(vecs) == 1
        assert len(vecs[0]) == len(tfidf.vocab)

    def test_roundtrip(self):
        tfidf = _TfidfFallback()
        tfidf.fit(["alpha beta", "gamma delta"])
        data = tfidf.to_dict()
        restored = _TfidfFallback.from_dict(data)
        assert restored.transform(["alpha"]) == tfidf.transform(["alpha"])


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert abs(_cosine_similarity([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        assert abs(_cosine_similarity([1, 0], [0, 1])) < 1e-9
