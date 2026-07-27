"""Skill Router plugin for HOMIE.

Routes free-form user requests to the most relevant skill (plugin) in a large
local skill library, entirely on-device.  Inspired by the "torafirma-skill-router"
project (local-first routing for large agent skill libraries).

Why HOMIE needs this
--------------------
As the plugin library grows past a handful of entries, stuffing every skill
description into the planner prompt wastes context and confuses small local
models.  This plugin keeps a lightweight local index of skill manifests and
narrows any request down to the top-K candidate skills *before* the LLM planner
runs, so the planner prompt only carries the relevant tools.

Routing strategy (all local, layered, graceful degradation):

1. Lexical TF-IDF scoring        - always available, zero extra dependencies.
2. Embedding similarity          - used automatically when ``fastembed`` is
   installed (small ONNX model, CPU-only, no network after the first model
   download the user performed when installing it).
3. LLM tie-break via local Ollama - opt-in (``skill_router.llm_tiebreak``),
   consulted only when the top candidates are too close to call.

This plugin makes no network calls itself; the optional embedding backend and
Ollama are local services the user has already opted into.

Config (homie.config.yaml)::

    skill_router:
      manifest_dir: ~/.homie/skills      # directory of *.json skill manifests
      top_k: 3
      min_score: 0.05
      llm_tiebreak: false

Manifest file format (one JSON object per ``*.json`` file)::

    {
      "name": "gpu_status",
      "description": "Report GPU utilisation and temperature on a target machine",
      "keywords": ["gpu", "nvidia", "vram", "temperature"],
      "examples": ["is my gpu busy", "check vram on msi"]
    }

Usage::

    from homie.plugins.skill_router import activate, deactivate

    router = activate(cfg)
    for match in router.route("how hot is the gpu on the msi box"):
        print(match.skill.name, round(match.score, 3), match.method)
    deactivate()
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from homie.config import HomieConfig

try:  # Prefer HOMIE's config helper, but keep the plugin usable standalone.
    from homie.config import cfg_get
except Exception:  # noqa: BLE001

    def cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:  # type: ignore[misc]
        cur = getattr(cfg, "raw", cfg)
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur


_TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.05
TIE_MARGIN = 0.10


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class SkillManifest:
    """Static, declarative description of one routable skill."""

    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    handler: Optional[Callable[..., Any]] = None
    source: Optional[Path] = None

    def document(self) -> str:
        """Flatten the manifest into one searchable text blob."""
        parts = [self.name.replace("_", " "), self.description]
        parts.extend(self.keywords)
        parts.extend(self.examples)
        return " ".join(p for p in parts if p)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], source: Optional[Path] = None) -> "SkillManifest":
        name = str(data.get("name", "")).strip()
        if not name:
            raise ValueError(f"Skill manifest missing 'name' ({source})")
        return cls(
            name=name,
            description=str(data.get("description", "")),
            keywords=[str(k) for k in data.get("keywords", [])],
            examples=[str(e) for e in data.get("examples", [])],
            source=source,
        )


@dataclass
class RouteResult:
    """One ranked routing candidate."""

    skill: SkillManifest
    score: float
    method: str  # "lexical" | "embedding" | "llm"


class SkillRouter:
    """Local-first router mapping natural-language requests to skills.

    Lexical scoring is always on; embedding scoring joins in when ``fastembed``
    is importable; a local-LLM tie-break can be enabled via config for the rare
    ambiguous case.
    """

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        llm_tiebreak: bool = False,
        cfg: Optional["HomieConfig"] = None,
    ) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self.llm_tiebreak = llm_tiebreak
        self.cfg = cfg
        self._skills: Dict[str, SkillManifest] = {}
        self._embedder: Any = None
        self._embedder_unavailable = False
        self._embed_cache: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------ #
    # Registry
    # ------------------------------------------------------------------ #
    def register(self, manifest: SkillManifest) -> None:
        """Add or replace a skill in the routing index."""
        self._skills[manifest.name] = manifest
        self._embed_cache.pop(manifest.name, None)
        logging.debug("SkillRouter registered skill %s", manifest.name)

    def unregister(self, name: str) -> None:
        self._skills.pop(name, None)
        self._embed_cache.pop(name, None)

    def load_directory(self, directory: Path) -> int:
        """Load every ``*.json`` manifest in *directory*. Returns count loaded."""
        loaded = 0
        if not directory.is_dir():
            logging.debug("SkillRouter manifest dir %s missing, skipping", directory)
            return loaded
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.register(SkillManifest.from_dict(data, source=path))
                loaded += 1
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                logging.warning("SkillRouter skipped bad manifest %s: %s", path, exc)
        return loaded

    @property
    def skills(self) -> List[SkillManifest]:
        return list(self._skills.values())

    # ------------------------------------------------------------------ #
    # Scoring backends
    # ------------------------------------------------------------------ #
    def _lexical_scores(self, query: str) -> Dict[str, float]:
        """TF-IDF-weighted token overlap, normalised to [0, 1]."""
        query_tokens = set(_tokenize(query))
        if not query_tokens or not self._skills:
            return {}

        docs = {name: set(_tokenize(m.document())) for name, m in self._skills.items()}
        n_docs = len(docs)
        idf: Dict[str, float] = {}
        for token in query_tokens:
            hits = sum(1 for tokens in docs.values() if token in tokens)
            idf[token] = math.log((n_docs + 1) / (hits + 1)) + 1.0

        total_weight = sum(idf.values())
        scores: Dict[str, float] = {}
        for name, tokens in docs.items():
            matched = sum(idf[t] for t in query_tokens if t in tokens)
            scores[name] = matched / total_weight if total_weight else 0.0
        return scores

    def _get_embedder(self) -> Any:
        """Lazily create a fastembed embedder; remember if unavailable."""
        if self._embedder is not None or self._embedder_unavailable:
            return self._embedder
        try:
            from fastembed import TextEmbedding  # local ONNX, CPU-only

            self._embedder = TextEmbedding()
            logging.info("SkillRouter embedding backend enabled (fastembed)")
        except Exception as exc:  # noqa: BLE001 - ImportError or model load issue
            self._embedder_unavailable = True
            logging.debug("SkillRouter embeddings unavailable: %s", exc)
        return self._embedder

    def _embed(self, key: str, text: str) -> Optional[List[float]]:
        embedder = self._get_embedder()
        if embedder is None:
            return None
        if key not in self._embed_cache:
            vector = next(iter(embedder.embed([text])))
            self._embed_cache[key] = [float(v) for v in vector]
        return self._embed_cache[key]

    def _embedding_scores(self, query: str) -> Dict[str, float]:
        if self._get_embedder() is None:
            return {}
        query_vec = self._embed(f"__query__:{query}", query)
        self._embed_cache.pop(f"__query__:{query}", None)  # do not cache queries
        if query_vec is None:
            return {}
        scores: Dict[str, float] = {}
        for name, manifest in self._skills.items():
            doc_vec = self._embed(name, manifest.document())
            if doc_vec is not None:
                # cosine in [-1, 1] -> clamp to [0, 1]
                scores[name] = max(0.0, _cosine(query_vec, doc_vec))
        return scores

    def _llm_pick(self, query: str, candidates: List[RouteResult]) -> Optional[str]:
        """Ask the local Ollama model to break a near-tie. Returns a skill name."""
        if self.cfg is None:
            return None
        try:
            from homie.llm_ollama import LLMError, ollama_generate
        except Exception:  # noqa: BLE001
            return None

        menu = "\n".join(
            f"- {r.skill.name}: {r.skill.description}" for r in candidates
        )
        prompt = (
            "You route user requests to skills. Return ONLY the single best "
            f"skill name from this list, nothing else.\n\nSkills:\n{menu}\n\n"
            f"User request: {query}"
        )
        try:
            answer = ollama_generate(self.cfg, prompt).strip().splitlines()[0].strip()
        except (LLMError, IndexError) as exc:
            logging.debug("SkillRouter LLM tie-break failed: %s", exc)
            return None
        return answer if answer in self._skills else None

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def route(self, query: str, top_k: Optional[int] = None) -> List[RouteResult]:
        """Return the top-K skills for *query*, best first.

        Blends lexical and (when available) embedding scores, then optionally
        lets the local LLM re-rank a near-tie between the top two candidates.
        """
        k = top_k or self.top_k
        lexical = self._lexical_scores(query)
        embedding = self._embedding_scores(query)

        results: List[RouteResult] = []
        for name, manifest in self._skills.items():
            lex = lexical.get(name, 0.0)
            if embedding:
                emb = embedding.get(name, 0.0)
                score = 0.5 * lex + 0.5 * emb
                method = "embedding"
            else:
                score = lex
                method = "lexical"
            if score >= self.min_score:
                results.append(RouteResult(skill=manifest, score=score, method=method))

        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:k]

        if (
            self.llm_tiebreak
            and len(results) >= 2
            and results[0].score - results[1].score < TIE_MARGIN
        ):
            winner = self._llm_pick(query, results[:2])
            if winner and winner != results[0].skill.name:
                results[0], results[1] = results[1], results[0]
                results[0].method = "llm"

        return results

    def planner_context(self, query: str) -> str:
        """Compact tool menu for the top candidates, ready to embed in a planner prompt."""
        lines = [
            f"- {r.skill.name}: {r.skill.description}" for r in self.route(query)
        ]
        return "\n".join(lines) if lines else "(no matching skills)"


_ROUTER: Optional[SkillRouter] = None


def activate(cfg: Optional["HomieConfig"] = None) -> SkillRouter:
    """Create the module-level router, loading manifests from config."""
    global _ROUTER
    top_k = cfg_get(cfg, "skill_router", "top_k", default=DEFAULT_TOP_K) if cfg else DEFAULT_TOP_K
    min_score = (
        cfg_get(cfg, "skill_router", "min_score", default=DEFAULT_MIN_SCORE)
        if cfg
        else DEFAULT_MIN_SCORE
    )
    llm_tiebreak = (
        bool(cfg_get(cfg, "skill_router", "llm_tiebreak", default=False)) if cfg else False
    )

    _ROUTER = SkillRouter(
        top_k=int(top_k), min_score=float(min_score), llm_tiebreak=llm_tiebreak, cfg=cfg
    )

    manifest_dir = cfg_get(cfg, "skill_router", "manifest_dir", default=None) if cfg else None
    if manifest_dir:
        count = _ROUTER.load_directory(Path(str(manifest_dir)).expanduser())
        logging.info("SkillRouter active with %d skill(s)", count)
    else:
        logging.info("SkillRouter active (no manifest_dir configured)")
    return _ROUTER


def deactivate() -> None:
    """Tear down the module-level router."""
    global _ROUTER
    _ROUTER = None
    logging.info("SkillRouter deactivated")


def register(cfg: Optional["HomieConfig"] = None) -> SkillRouter:
    """Entry point for HOMIE plugin loaders that expect ``register()``."""
    return activate(cfg)


def get_router() -> Optional[SkillRouter]:
    """Return the active router, if any."""
    return _ROUTER


__all__ = [
    "SkillManifest",
    "RouteResult",
    "SkillRouter",
    "activate",
    "deactivate",
    "register",
    "get_router",
]


if __name__ == "__main__":  # pragma: no cover - tiny local smoke test
    logging.basicConfig(level=logging.INFO)
    demo = SkillRouter()
    demo.register(
        SkillManifest(
            name="gpu_status",
            description="Report GPU utilisation and temperature on a target machine",
            keywords=["gpu", "nvidia", "vram", "temperature"],
            examples=["is my gpu busy", "check vram on msi"],
        )
    )
    demo.register(
        SkillManifest(
            name="disk_cleanup",
            description="Free disk space by clearing caches and old temp files",
            keywords=["disk", "cleanup", "space", "cache"],
            examples=["my disk is full", "clean up space on lenovo"],
        )
    )
    for result in demo.route("how hot is the gpu running"):
        print(f"{result.skill.name:<14} {result.score:.3f} ({result.method})")
