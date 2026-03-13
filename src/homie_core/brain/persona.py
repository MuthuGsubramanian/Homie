"""Specialist Persona System — context-aware personality switching.

Selects a specialist persona based on the user's current activity,
injecting domain-specific response guidance into the cognitive prompt.
Each persona adjusts tone, preferred output format, and domain hints.

Zero performance cost — pure string selection based on existing
SituationalAwareness data that's already computed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Persona:
    """A specialist personality the assistant can adopt."""

    name: str
    activity_types: tuple[str, ...]  # which activities trigger this persona
    tone: str                         # e.g. "precise and technical"
    guidance: str                     # injected into [GUIDANCE] block
    greeting_style: str               # how to open responses


# ------------------------------------------------------------------
# Built-in personas
# ------------------------------------------------------------------

_PERSONAS: list[Persona] = [
    Persona(
        name="engineer",
        activity_types=("coding", "debugging", "devops"),
        tone="precise and technical",
        guidance=(
            "User is coding. Prefer code examples over explanations. "
            "Use exact function/class names. Suggest edge cases and tests. "
            "If showing code, include only the relevant diff — no boilerplate. "
            "Mention performance implications when relevant."
        ),
        greeting_style="direct",
    ),
    Persona(
        name="writer",
        activity_types=("writing", "email", "documentation"),
        tone="concise and stylistic",
        guidance=(
            "User is writing. Match their register and tone. "
            "Offer structural suggestions, not just word swaps. "
            "Keep feedback actionable and specific. "
            "Avoid over-explaining — respect their craft."
        ),
        greeting_style="conversational",
    ),
    Persona(
        name="researcher",
        activity_types=("research", "browsing", "reading"),
        tone="structured and source-aware",
        guidance=(
            "User is researching. Provide structured information with clear "
            "headings and bullet points. Cite sources when possible. "
            "Distinguish established facts from speculation. "
            "Offer follow-up angles they might not have considered."
        ),
        greeting_style="informative",
    ),
    Persona(
        name="designer",
        activity_types=("design", "ui", "graphics"),
        tone="visual and intuitive",
        guidance=(
            "User is doing design work. Think visually — describe layouts, "
            "spacing, color relationships. Reference common design patterns. "
            "Offer alternatives rather than single answers. "
            "Consider accessibility implications."
        ),
        greeting_style="creative",
    ),
    Persona(
        name="planner",
        activity_types=("planning", "project_management", "meetings"),
        tone="organized and decisive",
        guidance=(
            "User is planning or managing. Break things into phases with "
            "clear deliverables. Identify risks and dependencies. "
            "Be decisive — recommend a path rather than listing options. "
            "Use time-aware language (deadlines, milestones)."
        ),
        greeting_style="professional",
    ),
    Persona(
        name="learner",
        activity_types=("learning", "tutorial", "studying"),
        tone="patient and pedagogical",
        guidance=(
            "User is learning. Build on what they already know — use "
            "analogies to familiar concepts. Check understanding before "
            "advancing. Provide small exercises when helpful. "
            "Celebrate progress without being patronizing."
        ),
        greeting_style="encouraging",
    ),
    Persona(
        name="general",
        activity_types=("general", "unknown", "idle", "media", "social"),
        tone="friendly and adaptable",
        guidance=(
            "Be conversational and helpful. Adapt depth to the question — "
            "short questions get short answers, detailed questions get "
            "thorough responses."
        ),
        greeting_style="casual",
    ),
]

# Index by activity type for O(1) lookup
_ACTIVITY_TO_PERSONA: dict[str, Persona] = {}
for _p in _PERSONAS:
    for _act in _p.activity_types:
        _ACTIVITY_TO_PERSONA[_act] = _p

_DEFAULT_PERSONA = _PERSONAS[-1]  # "general"


# ------------------------------------------------------------------
# Custom persona registry
# ------------------------------------------------------------------

_custom_personas: list[Persona] = []


def register_persona(persona: Persona) -> None:
    """Register a custom persona. Custom personas take priority over built-ins."""
    _custom_personas.append(persona)
    for act in persona.activity_types:
        _ACTIVITY_TO_PERSONA[act] = persona


# ------------------------------------------------------------------
# Selection
# ------------------------------------------------------------------

def select_persona(activity_type: str, activity_confidence: float = 0.0) -> Persona:
    """Select the best persona for the current activity.

    Falls back to 'general' if activity confidence is too low
    or no matching persona exists.
    """
    if activity_confidence < 0.3:
        return _DEFAULT_PERSONA
    return _ACTIVITY_TO_PERSONA.get(activity_type, _DEFAULT_PERSONA)


def get_persona_guidance(
    activity_type: str,
    activity_confidence: float = 0.0,
) -> str:
    """Get response guidance string for the current activity context.

    This is the primary API used by CognitiveArchitecture._generate_response_guidance().
    Returns an empty string if the general persona is selected (no special guidance needed).
    """
    persona = select_persona(activity_type, activity_confidence)
    if persona.name == "general":
        return ""
    return f"[Persona: {persona.name}] {persona.guidance}"


def list_personas() -> list[dict]:
    """Return all registered personas for debugging/transparency."""
    all_personas = _PERSONAS + _custom_personas
    return [
        {
            "name": p.name,
            "activity_types": list(p.activity_types),
            "tone": p.tone,
        }
        for p in all_personas
    ]
