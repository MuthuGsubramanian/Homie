"""Cross-Session User Model — unified profile from all memory sources.

Aggregates facts from SemanticMemory, session patterns from EpisodicMemory,
and behavioral signals into a queryable user profile. The profile is
materialized on-demand and cached briefly to avoid redundant computation.

This replaces scattered fact lookups with a coherent user understanding
that the cognitive architecture can inject into prompts.
"""
from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from homie_core.memory.semantic import SemanticMemory
from homie_core.memory.episodic import EpisodicMemory


# Cache TTL in seconds — user model doesn't change mid-conversation
_CACHE_TTL = 120.0


@dataclass
class UserProfile:
    """Structured user profile synthesized from all memory sources."""

    # Identity
    name: str = ""
    role: str = ""
    location: str = ""

    # Preferences & skills
    preferences: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    tools_and_tech: list[str] = field(default_factory=list)

    # Behavioral patterns
    common_topics: list[str] = field(default_factory=list)
    preferred_hours: list[int] = field(default_factory=list)
    avg_session_turns: float = 0.0
    dominant_mood: str = "neutral"

    # Interaction style
    verbosity_preference: str = "moderate"  # terse / moderate / detailed
    asks_follow_ups: bool = False

    # Raw facts for prompt injection
    all_facts: list[str] = field(default_factory=list)

    def to_context_block(self, max_chars: int = 800) -> str:
        """Render as a structured block for prompt injection."""
        lines: list[str] = []

        if self.name:
            identity = f"User: {self.name}"
            if self.role:
                identity += f" ({self.role})"
            if self.location:
                identity += f", {self.location}"
            lines.append(identity)

        if self.skills:
            lines.append(f"Skills: {', '.join(self.skills[:5])}")

        if self.tools_and_tech:
            lines.append(f"Tech: {', '.join(self.tools_and_tech[:5])}")

        if self.preferences:
            lines.append(f"Preferences: {', '.join(self.preferences[:4])}")

        if self.common_topics:
            lines.append(f"Frequent topics: {', '.join(self.common_topics[:5])}")

        if self.dominant_mood != "neutral":
            lines.append(f"Usual mood: {self.dominant_mood}")

        if self.verbosity_preference != "moderate":
            lines.append(f"Prefers {self.verbosity_preference} responses")

        result = "\n".join(lines)
        if len(result) > max_chars:
            result = result[:max_chars - 3] + "..."
        return result

    def is_empty(self) -> bool:
        return not self.name and not self.all_facts


# ------------------------------------------------------------------
# Fact categorization patterns
# ------------------------------------------------------------------

_NAME_PATTERN = re.compile(
    r"user(?:'s name is| is called| goes by)\s+(\w+)", re.IGNORECASE
)
_ROLE_PATTERN = re.compile(
    r"user\s+is\s+(?:a|an)\s+(.{3,50}?)(?:\.|$)", re.IGNORECASE
)
_LOCATION_PATTERN = re.compile(
    r"user\s+(?:lives? in|is (?:from|based in|located in))\s+(.{3,40}?)(?:\.|$)",
    re.IGNORECASE,
)
_SKILL_PATTERN = re.compile(
    r"user\s+(?:knows?|speaks?|studies|learned|practices?|has expertise in)\s+(.{3,50}?)(?:\.|$)",
    re.IGNORECASE,
)
_TECH_PATTERN = re.compile(
    r"user\s+(?:uses?|works? with|prefers?|switched to)\s+(.{3,50}?)(?:\.|$)",
    re.IGNORECASE,
)
_PREFERENCE_PATTERN = re.compile(
    r"user\s+(?:prefers?|likes?|loves?|enjoys?|hates?|dislikes?|avoids?)\s+(.{3,60}?)(?:\.|$)",
    re.IGNORECASE,
)


def _categorize_fact(fact_text: str) -> tuple[str, str]:
    """Categorize a fact into (category, extracted_value)."""
    for pattern, category in [
        (_NAME_PATTERN, "name"),
        (_ROLE_PATTERN, "role"),
        (_LOCATION_PATTERN, "location"),
        (_SKILL_PATTERN, "skill"),
        (_TECH_PATTERN, "tech"),
        (_PREFERENCE_PATTERN, "preference"),
    ]:
        m = pattern.search(fact_text)
        if m:
            return category, m.group(1).strip()
    return "other", fact_text


class UserModelSynthesizer:
    """Builds a unified user profile from all memory sources.

    Materializes the profile on-demand and caches it briefly.
    Only queries memory stores — never writes.
    """

    def __init__(
        self,
        semantic_memory: Optional[SemanticMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
    ):
        self._sm = semantic_memory
        self._em = episodic_memory
        self._cache: Optional[UserProfile] = None
        self._cache_time: float = 0.0

    def get_profile(self, force_refresh: bool = False) -> UserProfile:
        """Get the current user profile, using cache if fresh."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._cache is not None
            and (now - self._cache_time) < _CACHE_TTL
        ):
            return self._cache

        profile = self._build_profile()
        self._cache = profile
        self._cache_time = now
        return profile

    def get_relevant_context(self, query: str, max_chars: int = 600) -> str:
        """Get user context relevant to a specific query.

        Returns a formatted string for prompt injection, filtered
        by relevance to the query.
        """
        profile = self.get_profile()
        if profile.is_empty():
            return ""

        # For now, return the full context block (already concise)
        # Future: TF-IDF filter facts by query relevance
        return profile.to_context_block(max_chars=max_chars)

    def _build_profile(self) -> UserProfile:
        """Synthesize profile from semantic + episodic memory."""
        profile = UserProfile()

        # 1. Extract structured info from semantic facts
        if self._sm:
            try:
                facts = self._sm.get_facts(min_confidence=0.3)
                profile.all_facts = [f["fact"] for f in facts]
                self._extract_from_facts(profile, facts)
            except Exception:
                pass

        # 2. Extract patterns from episodic memory
        if self._em:
            try:
                self._extract_from_episodes(profile)
            except Exception:
                pass

        # 3. Infer interaction preferences from facts
        self._infer_preferences(profile)

        return profile

    def _extract_from_facts(
        self, profile: UserProfile, facts: list[dict]
    ) -> None:
        """Categorize facts into profile fields."""
        for fact in facts:
            category, value = _categorize_fact(fact["fact"])

            if category == "name" and not profile.name:
                profile.name = value
            elif category == "role" and not profile.role:
                profile.role = value
            elif category == "location" and not profile.location:
                profile.location = value
            elif category == "skill":
                profile.skills.append(value)
            elif category == "tech":
                profile.tools_and_tech.append(value)
            elif category == "preference":
                profile.preferences.append(value)

        # Deduplicate
        profile.skills = list(dict.fromkeys(profile.skills))
        profile.tools_and_tech = list(dict.fromkeys(profile.tools_and_tech))
        profile.preferences = list(dict.fromkeys(profile.preferences))

    def _extract_from_episodes(self, profile: UserProfile) -> None:
        """Extract behavioral patterns from episodic memory."""
        try:
            # Get recent episodes for pattern analysis
            episodes = self._em.recall("", n=50)
        except Exception:
            return

        if not episodes:
            return

        # Topic frequency
        topic_counts: Counter = Counter()
        mood_counts: Counter = Counter()
        turn_counts: list[int] = []

        for ep in episodes:
            summary = ep.get("summary", "")
            mood = ep.get("mood", "neutral")

            mood_counts[mood] += 1

            # Extract turn count from summary like "5-turn coding session"
            turn_match = re.search(r"(\d+)-turn", summary)
            if turn_match:
                turn_counts.append(int(turn_match.group(1)))

            # Extract topics from context tags
            tags = ep.get("context_tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    if tag and len(tag) > 2:
                        topic_counts[tag] += 1

        # Populate profile
        profile.common_topics = [
            topic for topic, _ in topic_counts.most_common(10)
        ]

        if mood_counts:
            profile.dominant_mood = mood_counts.most_common(1)[0][0]

        if turn_counts:
            profile.avg_session_turns = sum(turn_counts) / len(turn_counts)

    def _infer_preferences(self, profile: UserProfile) -> None:
        """Infer interaction preferences from accumulated signals."""
        # Verbosity: if user has many technical skills, they probably
        # prefer concise responses
        if len(profile.skills) > 3 or len(profile.tools_and_tech) > 3:
            profile.verbosity_preference = "terse"
        elif not profile.skills and not profile.tools_and_tech:
            profile.verbosity_preference = "detailed"

    def invalidate_cache(self) -> None:
        """Force next get_profile() to rebuild."""
        self._cache = None
        self._cache_time = 0.0
