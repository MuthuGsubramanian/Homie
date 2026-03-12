"""Markdown-based skill system for the Homie AI assistant.

Allows users to add custom behaviors by dropping SKILL.md files into a
skills directory (~/.homie/skills by default). Each skill file uses YAML
frontmatter to declare metadata and contains the full skill prompt as its
body content.

Inspired by the hermes-agent SKILL.md format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Skill:
    """A single loaded skill definition."""

    name: str
    description: str
    category: str
    content: str
    required_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    file_path: str = ""
    enabled: bool = True


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Parse YAML-like frontmatter delimited by ``---`` markers.

    Handles scalar ``key: value`` pairs and inline lists in
    ``[a, b, c]`` format.  This is intentionally minimal — it covers
    only the fields used by skill files and avoids pulling in a full
    YAML library.

    Returns:
        A tuple of (metadata dict, remaining body content).
    """
    match = re.match(r"\A---[ \t]*\r?\n(.*?\r?\n)---[ \t]*\r?\n", text, re.DOTALL)
    if not match:
        return {}, text

    raw_front = match.group(1)
    body = text[match.end():]

    meta: dict[str, object] = {}
    for line in raw_front.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        colon_idx = line.find(":")
        if colon_idx == -1:
            continue

        key = line[:colon_idx].strip()
        value_str = line[colon_idx + 1:].strip()

        # Inline list: [a, b, c]
        list_match = re.match(r"^\[(.*)]\s*$", value_str)
        if list_match:
            items = [item.strip() for item in list_match.group(1).split(",")]
            meta[key] = [item for item in items if item]
            continue

        # Boolean
        if value_str.lower() in ("true", "yes"):
            meta[key] = True
            continue
        if value_str.lower() in ("false", "no"):
            meta[key] = False
            continue

        # Plain string (strip optional surrounding quotes)
        if len(value_str) >= 2 and value_str[0] == value_str[-1] and value_str[0] in ('"', "'"):
            value_str = value_str[1:-1]

        meta[key] = value_str

    return meta, body


def _skill_from_frontmatter(meta: dict[str, object], body: str, file_path: str) -> Skill:
    """Construct a ``Skill`` from parsed frontmatter and body text."""
    required_tools = meta.get("required_tools", [])
    if isinstance(required_tools, str):
        required_tools = [required_tools]

    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]

    enabled = meta.get("enabled", True)
    if not isinstance(enabled, bool):
        enabled = str(enabled).lower() in ("true", "yes", "1")

    return Skill(
        name=str(meta.get("name", Path(file_path).stem)),
        description=str(meta.get("description", "")),
        category=str(meta.get("category", "general")),
        content=body.strip(),
        required_tools=list(required_tools),
        tags=list(tags),
        file_path=file_path,
        enabled=bool(enabled),
    )


class SkillLoader:
    """Discover, parse, and manage markdown-based skill files.

    Skills are stored as ``SKILL.md`` files (case-insensitive match on the
    ``.md`` extension) anywhere under *skills_dir*.  Each file must begin
    with YAML frontmatter between ``---`` markers.

    Example skill file::

        ---
        name: arxiv-search
        description: Search and summarize arXiv papers
        category: research
        required_tools: [web_search, file_read]
        tags: [research, papers]
        enabled: true
        ---

        When asked to find academic papers, use the web_search tool
        to query arxiv.org ...
    """

    def __init__(self, skills_dir: str | Path = "~/.homie/skills") -> None:
        self._skills_dir = Path(skills_dir).expanduser().resolve()
        self._skills: dict[str, Skill] = {}

    @property
    def skills_dir(self) -> Path:
        """Return the resolved skills directory path."""
        return self._skills_dir

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def scan(self) -> list[Skill]:
        """Discover and parse all ``SKILL.md`` files under the skills directory.

        Files are matched recursively by the pattern ``*SKILL.md``
        (case-insensitive on Windows).  Previously loaded skills are
        replaced on each scan so the index always reflects the current
        state of the filesystem.

        Returns:
            A list of all discovered :class:`Skill` instances (including
            disabled ones).
        """
        self._skills.clear()

        if not self._skills_dir.is_dir():
            return []

        for path in sorted(self._skills_dir.rglob("*SKILL.md")):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue

            meta, body = _parse_frontmatter(text)
            skill = _skill_from_frontmatter(meta, body, str(path))
            self._skills[skill.name] = skill

        return list(self._skills.values())

    def get_skill(self, name: str) -> Optional[Skill]:
        """Return a skill by exact name, or ``None`` if not found."""
        return self._skills.get(name)

    def get_skills_by_category(self, category: str) -> list[Skill]:
        """Return all enabled skills matching *category*."""
        return [
            s for s in self._skills.values()
            if s.category == category and s.enabled
        ]

    def get_skills_for_tools(self, available_tools: list[str]) -> list[Skill]:
        """Return enabled skills whose required tools are all available.

        A skill with an empty ``required_tools`` list is always included.
        """
        tool_set = set(available_tools)
        return [
            s for s in self._skills.values()
            if s.enabled and set(s.required_tools) <= tool_set
        ]

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def build_skills_index(self) -> str:
        """Generate a formatted index of all enabled skills.

        The output is suitable for inclusion in a system prompt so the
        model knows which skills are available.  Skills are grouped by
        category with a markdown header per category.

        Returns:
            A multi-line string.  Empty string if no skills are loaded.
        """
        by_category: dict[str, list[Skill]] = {}
        for skill in self._skills.values():
            if not skill.enabled:
                continue
            by_category.setdefault(skill.category, []).append(skill)

        if not by_category:
            return ""

        lines: list[str] = []
        for category in sorted(by_category):
            lines.append(f"### {category.title()}")
            for skill in sorted(by_category[category], key=lambda s: s.name):
                lines.append(f"- **{skill.name}**: {skill.description}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def build_skill_prompt(self, name: str) -> Optional[str]:
        """Return the full skill content for injection into a conversation.

        Returns ``None`` if the skill does not exist or is disabled.
        """
        skill = self._skills.get(name)
        if skill is None or not skill.enabled:
            return None
        return skill.content
