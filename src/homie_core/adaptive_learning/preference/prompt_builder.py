"""Generates the preference prompt layer prepended to system prompt."""

from .profile import PreferenceProfile


def _verbosity_label(v: float) -> str:
    if v < 0.25:
        return "very concise (keep responses short and direct)"
    if v < 0.45:
        return "concise (prefer brief responses)"
    if v > 0.75:
        return "detailed and thorough (provide comprehensive responses)"
    return ""


def _formality_label(f: float) -> str:
    if f < 0.3:
        return "casual and conversational"
    if f > 0.7:
        return "professional and formal"
    return ""


def _depth_label(d: float) -> str:
    if d < 0.3:
        return "keep explanations simple, avoid jargon"
    if d > 0.7:
        return "expert level, skip basic explanations"
    return ""


def _format_label(fmt: str) -> str:
    labels = {
        "bullets": "prefer bullet points over prose",
        "code_first": "lead with code examples, explain after",
        "prose": "use flowing prose",
    }
    return labels.get(fmt, "")


def _style_label(style: str) -> str:
    labels = {
        "bottom_up": "explain from specifics to general",
        "example_first": "start with examples, then explain the concept",
    }
    return labels.get(style, "")


def build_preference_prompt(
    profile: PreferenceProfile,
    min_confidence: float = 0.1,
) -> str:
    """Build a preference prompt layer from a profile.

    Returns empty string if confidence is too low.
    """
    if profile.confidence < min_confidence:
        return ""

    lines = []
    for label_fn, value in [
        (_verbosity_label, profile.verbosity),
        (_formality_label, profile.formality),
        (_depth_label, profile.technical_depth),
    ]:
        label = label_fn(value)
        if label:
            lines.append(f"- {label}")

    fmt = _format_label(profile.format_preference)
    if fmt:
        lines.append(f"- {fmt}")

    style = _style_label(profile.explanation_style)
    if style:
        lines.append(f"- {style}")

    if not lines:
        return ""

    return "[Learned preferences for this context]\n" + "\n".join(lines)
