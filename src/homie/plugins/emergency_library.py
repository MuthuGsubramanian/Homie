"""Emergency Library plugin for HOMIE.

Bundles a small, fully offline survival/emergency knowledge base and exposes
keyword-based retrieval plus optional local LLM synthesis (via the already
configured Ollama backend) so HOMIE keeps answering basic survival questions
-- water purification, bleeding control, hypothermia, shelter, signaling --
even with no internet access and no vector database installed.

Design notes:
- Retrieval never depends on chromadb or a downloaded embedding model. Those
  require a network call on first use, which is unacceptable for a plugin
  whose entire purpose is to work when nothing else does. Instead it uses a
  small stdlib keyword-overlap scorer over a bundled, hardcoded knowledge
  base.
- LLM synthesis via homie.llm_ollama.ollama_generate is optional and used
  only to phrase retrieved facts conversationally; if the local Ollama
  server is unreachable the plugin still returns the raw matched entries.
- No outbound network calls are made by this plugin under any configuration.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from homie.config import HomieConfig, cfg_get
from homie.llm_ollama import LLMError, ollama_generate

logger = logging.getLogger(__name__)


# Bundled offline survival/emergency reference entries: (topic, keywords, text).
# Kept short, self-contained, and non-exhaustive -- this is a quick-reference
# aid, not a substitute for professional medical or emergency training.
_EMERGENCY_ENTRIES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    (
        "severe_bleeding",
        ("bleed", "bleeding", "blood", "wound", "cut", "hemorrhage"),
        "Apply firm, direct pressure to the wound with a clean cloth. Do not "
        "remove an embedded object. Elevate the injury above heart level if "
        "possible. If bleeding soaks through, add more cloth on top rather "
        "than removing the first layer. Seek medical help immediately.",
    ),
    (
        "cpr_basics",
        ("cpr", "not breathing", "no pulse", "cardiac", "resuscitation"),
        "If someone is unresponsive and not breathing normally: call for "
        "emergency help, then push hard and fast in the center of the chest "
        "at roughly 100-120 compressions per minute, about 5cm deep, letting "
        "the chest fully recoil between compressions. Continue until help "
        "arrives or the person responds.",
    ),
    (
        "hypothermia",
        ("hypothermia", "cold", "freezing", "shivering"),
        "Move the person somewhere dry and sheltered from wind. Remove wet "
        "clothing and insulate with dry layers or blankets, covering the "
        "head. Give warm (not hot) sweet drinks only if they are fully "
        "alert. Do not rub or massage the limbs or apply direct heat to "
        "the skin.",
    ),
    (
        "heatstroke",
        ("heatstroke", "heat exhaustion", "overheating", "dehydration"),
        "Move to shade or a cool area immediately. Remove excess clothing "
        "and cool the skin with water, fanning, or damp cloths at the neck, "
        "armpits, and groin. Give small sips of water only if the person is "
        "conscious and alert. Confusion or collapse means this is a medical "
        "emergency.",
    ),
    (
        "water_purification",
        ("water", "purify", "drinking water", "boil water", "clean water"),
        "Boiling is the most reliable field method: bring water to a full "
        "rolling boil for at least 1 minute (3 minutes above 2000m "
        "altitude). If boiling is not possible, a 0.1-1 micron filter or "
        "unscented liquid chlorine bleach (about 2 drops per liter, rest 30 "
        "minutes) can help. Avoid stagnant or visibly contaminated sources "
        "even after treatment.",
    ),
    (
        "fire_starting",
        ("fire", "spark", "tinder", "kindling", "ignite"),
        "Gather tinder (dry grass, bark shavings), kindling (pencil-thick "
        "dry twigs), and fuel wood before striking a spark. Build a small "
        "teepee or lean-to structure over the tinder bundle so flame can "
        "climb into the kindling. Protect the site from wind and never "
        "leave an open flame unattended.",
    ),
    (
        "emergency_shelter",
        ("shelter", "exposure", "camp", "lean-to", "stay warm"),
        "Prioritize insulation from the ground before height of structure: "
        "a bed of dry leaves, branches, or debris under you prevents more "
        "heat loss than most roofing. Keep the shelter just large enough "
        "for your body to retain warmth, and close off openings against "
        "wind.",
    ),
    (
        "signaling_for_rescue",
        ("rescue", "signal", "lost", "stranded", "sos", "flare"),
        "Use three of anything (three fires, three whistle blasts, three "
        "flashes) as the universal distress signal. Ground-to-air signals "
        "work best in open areas: a large X or SOS made from contrasting "
        "material, rocks, or trampled snow is visible from aircraft. Stay "
        "near your shelter once you have signaled so rescuers can find you.",
    ),
    (
        "basic_triage",
        ("triage", "multiple injured", "who first", "mass casualty"),
        "With multiple injured people and limited help, check responsiveness "
        "and breathing first. Treat life-threatening bleeding and airway "
        "obstruction before anything else. People who are walking and "
        "talking can generally wait longer than those who are unresponsive "
        "or struggling to breathe.",
    ),
    (
        "food_safety",
        ("spoiled food", "food poisoning", "eat safely", "foraging"),
        "When unsure about stored food, discard anything with off smell, "
        "unusual color, or swollen packaging. Never eat wild plants or "
        "mushrooms you cannot positively identify -- misidentification is a "
        "leading cause of poisoning in survival situations.",
    ),
)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_entry(query_tokens: Sequence[str], keywords: Tuple[str, ...]) -> int:
    """Count how many entry keywords appear (as substrings) in the query."""
    joined = " ".join(query_tokens)
    return sum(1 for kw in keywords if kw in joined)


@dataclass
class EmergencyLibraryPlugin:
    """Offline-first emergency and survival knowledge base for HOMIE.

    Retrieval is pure stdlib keyword matching over a bundled knowledge base
    so it keeps working with no installed dependencies, no model downloads,
    and no network access -- the plugin's entire reason to exist is to help
    when those things are unavailable. Answer synthesis via the local LLM
    is a convenience layer only, never a requirement.
    """

    cfg: HomieConfig
    use_llm_synthesis: bool = True
    min_match_score: int = 1
    _active: bool = field(default=False, repr=False)

    def activate(self) -> None:
        """Load plugin configuration. No network or disk I/O is required."""
        self.use_llm_synthesis = bool(
            cfg_get(self.cfg, "plugins", "emergency_library", "use_llm_synthesis", default=True)
        )
        self.min_match_score = int(
            cfg_get(self.cfg, "plugins", "emergency_library", "min_match_score", default=1)
        )
        self._active = True
        logger.info(
            "EmergencyLibraryPlugin activated | entries=%d llm_synthesis=%s",
            len(_EMERGENCY_ENTRIES),
            self.use_llm_synthesis,
        )

    def deactivate(self) -> None:
        """Release plugin state. Nothing persistent is held open."""
        self._active = False
        logger.info("EmergencyLibraryPlugin deactivated")

    def search(self, user_prompt: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """Return up to ``top_k`` (topic, text) entries matching the prompt."""
        query_tokens = _tokenize(user_prompt)
        scored = [
            (score, topic, text)
            for topic, keywords, text in _EMERGENCY_ENTRIES
            if (score := _score_entry(query_tokens, keywords)) >= self.min_match_score
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(topic, text) for _, topic, text in scored[:top_k]]

    def answer(self, user_prompt: str) -> str:
        """Answer an emergency/survival question, offline-first.

        Always grounds the answer in the bundled knowledge base. If LLM
        synthesis is enabled and the local Ollama server responds, the
        matched entries are phrased conversationally; otherwise the raw
        entries are returned verbatim so the plugin degrades gracefully
        instead of failing.
        """
        matches = self.search(user_prompt)
        if not matches:
            return (
                "No matching entry in the offline emergency library for "
                f"'{user_prompt}'. This library only covers a small set of "
                "common survival topics (bleeding, CPR, hypothermia, "
                "heatstroke, water purification, fire, shelter, signaling, "
                "triage, food safety)."
            )

        raw_answer = "\n\n".join(f"[{topic}] {text}" for topic, text in matches)

        if not self.use_llm_synthesis:
            return raw_answer

        synthesis_prompt = (
            "You are HOMIE's offline emergency assistant. Using ONLY the "
            "reference notes below, answer the user's question clearly and "
            "calmly. Do not invent facts beyond what is given.\n\n"
            f"Reference notes:\n{raw_answer}\n\n"
            f"User question: {user_prompt}"
        )
        try:
            synthesized = ollama_generate(self.cfg, synthesis_prompt)
        except LLMError as exc:
            logger.warning("LLM synthesis unavailable, returning raw entries: %s", exc)
            return raw_answer

        return synthesized.strip() or raw_answer

    @property
    def topics(self) -> List[str]:
        """List the topics covered by the bundled knowledge base."""
        return [topic for topic, _, _ in _EMERGENCY_ENTRIES]


def register(cfg: HomieConfig) -> EmergencyLibraryPlugin:
    """Build, activate, and return an EmergencyLibraryPlugin instance.

    Entry point for HOMIE's plugin loader.
    """
    plugin = EmergencyLibraryPlugin(cfg=cfg)
    plugin.activate()
    return plugin


__all__ = ["EmergencyLibraryPlugin", "register"]
