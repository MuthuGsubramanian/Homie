"""Local embedding model — wraps sentence-transformers or falls back to TF-IDF vectors."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from homie_core.ml.base import LocalModel

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer

    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ===================================================================
# Pure-Python TF-IDF fallback embedder
# ===================================================================

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _TfidfFallback:
    """Minimal TF-IDF vectorizer in pure Python."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        doc_freq: Counter = Counter()
        n_docs = len(texts)
        all_tokens: set[str] = set()

        for text in texts:
            tokens = set(_tokenize(text))
            doc_freq.update(tokens)
            all_tokens.update(tokens)

        # build vocab (sorted for determinism)
        self.vocab = {tok: idx for idx, tok in enumerate(sorted(all_tokens))}
        self.idf = {
            tok: math.log((n_docs + 1) / (count + 1)) + 1.0
            for tok, count in doc_freq.items()
        }
        self._fitted = True

    def transform(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("TF-IDF fallback has not been fitted.")
        dim = len(self.vocab)
        vectors: list[list[float]] = []
        for text in texts:
            tokens = _tokenize(text)
            tf: Counter = Counter(tokens)
            total = max(len(tokens), 1)
            vec = [0.0] * dim
            for tok, count in tf.items():
                if tok in self.vocab:
                    idx = self.vocab[tok]
                    vec[idx] = (count / total) * self.idf.get(tok, 1.0)
            # L2-normalise
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors

    def to_dict(self) -> dict:
        return {"vocab": self.vocab, "idf": self.idf}

    @classmethod
    def from_dict(cls, data: dict) -> "_TfidfFallback":
        obj = cls()
        obj.vocab = data["vocab"]
        obj.idf = data["idf"]
        obj._fitted = True
        return obj


# ===================================================================
# Helpers
# ===================================================================

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = _dot(a, b)
    norm_a = math.sqrt(_dot(a, a)) or 1.0
    norm_b = math.sqrt(_dot(b, b)) or 1.0
    return dot / (norm_a * norm_b)


# ===================================================================
# Public API
# ===================================================================

class LocalEmbedder(LocalModel):
    """Local embedding model.

    When *sentence-transformers* is available, it delegates to a pre-trained
    model (default ``all-MiniLM-L6-v2``).  Otherwise, it uses a pure-Python
    TF-IDF fallback that must be fitted on a corpus before use.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(name=model_name, model_type="embedder")
        self.model_name = model_name
        self._backend: str = "sbert" if _HAS_SBERT else "tfidf"

        # sbert state
        self._sbert_model: Any = None

        # fallback state
        self._tfidf: _TfidfFallback | None = None

        if self._backend == "sbert":
            self._sbert_model = SentenceTransformer(model_name)
            self._trained = True  # pre-trained model

    # ------------------------------------------------------------------
    # Training (only needed for the TF-IDF fallback)
    # ------------------------------------------------------------------

    def train(self, X: list, y: list | None = None) -> dict:  # type: ignore[override]
        """Fit the fallback TF-IDF vectorizer on *X* (list of texts).

        *y* is ignored — embedders are unsupervised.  When using the
        sentence-transformers backend this is a no-op.
        """
        if self._backend == "sbert":
            self._trained = True
            return {"backend": "sbert", "status": "pre-trained", "n_texts": len(X)}

        self._tfidf = _TfidfFallback()
        self._tfidf.fit(X)
        self._trained = True
        metrics = {
            "backend": "tfidf",
            "vocab_size": len(self._tfidf.vocab),
            "n_texts": len(X),
        }
        self._metrics = metrics
        return metrics

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return dense vectors for each text."""
        if self._backend == "sbert":
            embeddings = self._sbert_model.encode(texts, show_progress_bar=False)
            if _HAS_NUMPY:
                return [vec.tolist() for vec in embeddings]
            return [list(vec) for vec in embeddings]

        # TF-IDF fallback
        if self._tfidf is None or not self._tfidf._fitted:
            raise RuntimeError("TF-IDF fallback has not been fitted. Call train() first.")
        return self._tfidf.transform(texts)

    def predict(self, X: list) -> list:
        """Alias for :meth:`embed` to satisfy the base interface."""
        return self.embed(X)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Return cosine similarity between two texts (range -1 .. 1)."""
        vecs = self.embed([text_a, text_b])
        return _cosine_similarity(vecs[0], vecs[1])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "backend": self._backend,
            "model_name": self.model_name,
            "name": self.name,
            "metrics": self._metrics,
            "trained": self._trained,
        }
        if self._tfidf is not None:
            data["tfidf"] = self._tfidf.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.model_name = data.get("model_name", self.model_name)
        self.name = data.get("name", self.name)
        self._backend = data.get("backend", self._backend)
        self._metrics = data.get("metrics", {})
        self._trained = data.get("trained", False)
        if "tfidf" in data:
            self._tfidf = _TfidfFallback.from_dict(data["tfidf"])
            self._trained = True
        if self._backend == "sbert" and _HAS_SBERT:
            self._sbert_model = SentenceTransformer(self.model_name)
            self._trained = True
