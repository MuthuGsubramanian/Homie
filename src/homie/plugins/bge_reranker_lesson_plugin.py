"""BGE Reranker Lesson plugin for Homie.

Companion lesson to Homie's existing BGE reranker plugin. Instead of adding
new retrieval machinery, this plugin teaches â€” it is a runnable, interactive
walkthrough of WHY cross-encoder rerankers (BAAI/bge-reranker-v2-m3) beat
cosine similarity alone in RAG pipelines.

The lesson demonstrates three things with live, local computation:

1. Bi-encoder retrieval (what ChromaDB does): query and documents are
   embedded INDEPENDENTLY, then compared with cosine similarity. Fast, but
   the model never sees the query and document together, so it cannot model
   word-level interactions (negation, paraphrase, 'fan' the device vs 'fan'
   the person).
2. Where cosine fails: a distractor document that shares many surface words
   with the query outranks the truly relevant paraphrase.
3. Cross-encoder reranking: the query and each candidate are fed JOINTLY
   through one transformer, letting attention connect query tokens directly
   to document tokens. Slow per pair â€” which is exactly why it is used as a
   second stage over the top-k, not over the whole corpus.

Offline-first: with no extra dependencies the lesson runs in 'toy mode'
(pure-Python TF cosine + a clearly-labelled simulated interaction scorer).
If ``sentence-transformers`` is installed AND the operator has opted in via
``plugins.reranker_lesson.allow_model_load: true`` in homie.config.yaml,
the real BAAI/bge-reranker-v2-m3 cross-encoder is used instead. The plugin
never downloads anything unless that flag is set.

Usage (from Homie's plugin host or a REPL)::

    from homie.plugins.bge_reranker_lesson_plugin import register
    plugin = register(cfg)          # cfg optional
    print(plugin.run_lesson())      # full lesson transcript
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # Optional; only used when the operator opts in to real model loading.
    from homie.config import HomieConfig, cfg_get
except Exception:  # noqa: BLE001 â€” lesson must run standalone too.
    HomieConfig = None  # type: ignore[assignment]

    def cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:  # type: ignore[misc]
        return default


PLUGIN_NAME = "bge_reranker_lesson"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"

# A query engineered so that lexical/embedding overlap misleads retrieval.
LESSON_QUERY = "How do I stop my laptop fan from running loudly all the time?"

# doc, is_relevant, teaching_note
LESSON_CORPUS: List[Tuple[str, bool, str]] = [
    (
        "The laptop fan runs loudly all the time when the stop button in the "
        "fan control app is greyed out â€” this is a known UI bug with no fix.",
        False,
        "Distractor: shares almost every keyword with the query but answers a "
        "different question (a UI bug report, not a remedy).",
    ),
    (
        "To quiet a noisy notebook cooler, clean dust from the vents, reapply "
        "thermal paste, and cap the CPU's maximum clock in power settings.",
        True,
        "True answer: a paraphrase â€” 'quiet a noisy notebook cooler' shares "
        "nearly zero surface words with the query.",
    ),
    (
        "Ceiling fans that run loudly can be fixed by tightening the mounting "
        "bracket and balancing the blades.",
        False,
        "Wrong 'fan' entirely â€” tests whether the scorer models word sense in "
        "context.",
    ),
    (
        "Stop running your laptop on soft surfaces like beds; blocked intake "
        "vents make thermal throttling worse over time.",
        False,
        "Related topic, partial keyword overlap, but does not answer the "
        "question asked.",
    ),
]

# Tiny synonym map used ONLY by the toy interaction scorer so the offline
# demo can illustrate what real cross-attention learns at scale.
_TOY_SYNONYMS: Dict[str, str] = {
    "quiet": "stop-loud",
    "noisy": "loud",
    "loudly": "loud",
    "notebook": "laptop",
    "cooler": "fan",
    "ceiling": "NOT-laptop",
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _tf_cosine(query: str, doc: str) -> float:
    """Pure-Python term-frequency cosine â€” a stand-in for bi-encoder scoring.

    Real bi-encoders embed meaning, not just terms, but they share the same
    structural blindness demonstrated here: query and document are encoded
    independently, so token-level interactions are invisible to the scorer.
    """
    q, d = Counter(_tokenize(query)), Counter(_tokenize(doc))
    dot = sum(q[t] * d[t] for t in q)
    norm = math.sqrt(sum(v * v for v in q.values())) * math.sqrt(
        sum(v * v for v in d.values())
    )
    return dot / norm if norm else 0.0


def _toy_interaction_score(query: str, doc: str) -> float:
    """Simulated cross-encoder score (offline teaching aid â€” NOT a model).

    Mimics two behaviours a real cross-encoder learns from data: (1) synonym
    and paraphrase matching via joint attention, and (2) penalising sense
    mismatches ('ceiling fan' vs 'laptop fan'). The synonym table is
    hand-written for this corpus; bge-reranker-v2-m3 learns the general
    version of this from hundreds of millions of query-passage pairs.
    """
    canon = lambda ts: [_TOY_SYNONYMS.get(t, t) for t in ts]  # noqa: E731
    q_terms = set(canon(_tokenize(query)))
    d_terms = set(canon(_tokenize(doc)))
    if "NOT-laptop" in d_terms and "laptop" in q_terms:
        return 0.05  # sense mismatch: joint encoding catches this
    overlap = len(q_terms & d_terms) / max(len(q_terms), 1)
    answers_how = any(t in d_terms for t in ("clean", "reapply", "cap", "stop-loud"))
    return round(min(1.0, overlap + (0.35 if answers_how else 0.0)), 3)


@dataclass
class ScoredDoc:
    """One candidate document with both stage scores attached."""

    text: str
    relevant: bool
    note: str
    cosine: float = 0.0
    rerank: float = 0.0


@dataclass
class BGERerankerLessonPlugin:
    """Interactive lesson: why rerankers beat cosine similarity alone.

    Attributes:
        cfg: Optional HomieConfig; read for the ``allow_model_load`` opt-in.
        use_real_model: True once bge-reranker-v2-m3 is actually loaded.
    """

    cfg: Optional["HomieConfig"] = None
    use_real_model: bool = False
    _cross_encoder: Any = field(default=None, repr=False)
    _active: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def activate(self) -> Dict[str, Any]:
        """Activate the plugin; loads the real reranker only if opted in."""
        allow_load = bool(
            cfg_get(self.cfg, "plugins", "reranker_lesson", "allow_model_load", default=False)
        )
        if allow_load:
            self._try_load_cross_encoder()
        self._active = True
        mode = "real-model" if self.use_real_model else "toy (offline)"
        logging.info("%s activated in %s mode", PLUGIN_NAME, mode)
        return {"name": PLUGIN_NAME, "mode": mode, "model": RERANKER_MODEL_ID if self.use_real_model else None}

    def deactivate(self) -> None:
        """Release the model (if loaded) and mark the plugin inactive."""
        self._cross_encoder = None
        self.use_real_model = False
        self._active = False
        logging.info("%s deactivated", PLUGIN_NAME)

    def _try_load_cross_encoder(self) -> None:
        """Best-effort load of the real cross-encoder. Never raises."""
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._cross_encoder = CrossEncoder(RERANKER_MODEL_ID, max_length=512)
            self.use_real_model = True
        except Exception as exc:  # noqa: BLE001 â€” fall back to toy mode
            logging.warning(
                "Could not load %s (%s); lesson runs in toy mode", RERANKER_MODEL_ID, exc
            )

    # ------------------------------------------------------------------ #
    # Scoring stages
    # ------------------------------------------------------------------ #
    def score_corpus(self, query: str = LESSON_QUERY) -> List[ScoredDoc]:
        """Run both retrieval stages over the lesson corpus."""
        docs = [ScoredDoc(text=t, relevant=r, note=n) for t, r, n in LESSON_CORPUS]
        for d in docs:
            d.cosine = round(_tf_cosine(query, d.text), 3)
        if self.use_real_model and self._cross_encoder is not None:
            pairs = [(query, d.text) for d in docs]
            for d, s in zip(docs, self._cross_encoder.predict(pairs)):
                d.rerank = round(float(s), 3)
        else:
            for d in docs:
                d.rerank = _toy_interaction_score(query, d.text)
        return docs

    # ------------------------------------------------------------------ #
    # The lesson itself
    # ------------------------------------------------------------------ #
    def run_lesson(self, query: str = LESSON_QUERY) -> str:
        """Return the full lesson transcript as a printable string."""
        if not self._active:
            self.activate()
        docs = self.score_corpus(query)
        by_cosine = sorted(docs, key=lambda d: d.cosine, reverse=True)
        by_rerank = sorted(docs, key=lambda d: d.rerank, reverse=True)
        scorer = (
            f"real {RERANKER_MODEL_ID}" if self.use_real_model else "simulated interaction scorer (toy mode)"
        )

        lines: List[str] = []
        w = lines.append
        w("LESSON: Why rerankers beat cosine similarity alone")
        w("=" * 54)
        w(f"Query: {query}")
        w("")
        w("STAGE 1 â€” Bi-encoder retrieval (cosine similarity)")
        w("Query and documents are embedded independently; the scorer only")
        w("ever compares two fixed vectors. Ranking:")
        for i, d in enumerate(by_cosine, 1):
            tag = "RELEVANT" if d.relevant else "not relevant"
            w(f"  {i}. cosine={d.cosine:<6} [{tag}] {d.text[:72]}...")
        top = by_cosine[0]
        if not top.relevant:
            w("")
            w("  >> FAILURE: the top hit is a keyword-stuffed distractor.")
            w(f"     Why: {top.note}")
        w("")
        w(f"STAGE 2 â€” Cross-encoder reranking ({scorer})")
        w("Query and document are fed JOINTLY through one model; attention")
        w("connects each query token to each document token, so paraphrase,")
        w("negation, and word sense become visible to the scorer. Ranking:")
        for i, d in enumerate(by_rerank, 1):
            tag = "RELEVANT" if d.relevant else "not relevant"
            w(f"  {i}. score={d.rerank:<6} [{tag}] {d.text[:72]}...")
        w("")
        w("KEY TAKEAWAYS")
        w("  1. Bi-encoders compress a whole document into one vector BEFORE")
        w("     seeing the query â€” interactions are lost by construction.")
        w("  2. Cross-encoders score (query, doc) pairs jointly â€” accurate,")
        w("     but O(corpus) transformer passes, so they cannot index.")
        w("  3. Production pattern (and Homie's RAG): ChromaDB cosine top-k")
        w("     (cheap, recall-oriented) -> bge-reranker-v2-m3 over those k")
        w("     (expensive, precision-oriented). Two stages, best of both.")
        if not self.use_real_model:
            w("")
            w("  NOTE: stage 2 used a hand-built simulation so this lesson runs")
            w("  fully offline. Set plugins.reranker_lesson.allow_model_load:")
            w("  true in homie.config.yaml (requires sentence-transformers) to")
            w("  rerun it with the real cross-encoder.")
        return "\n".join(lines)


def register(cfg: Optional["HomieConfig"] = None) -> BGERerankerLessonPlugin:
    """Homie plugin entry point: build and activate the lesson plugin."""
    plugin = BGERerankerLessonPlugin(cfg=cfg)
    plugin.activate()
    return plugin


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(register().run_lesson())

__all__ = ["BGERerankerLessonPlugin", "register", "ScoredDoc", "PLUGIN_NAME"]
