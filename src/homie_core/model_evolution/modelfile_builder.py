"""ModelfileBuilder — assembles Ollama Modelfile from learned layers."""

import hashlib
from pathlib import Path
from typing import Optional


class ModelfileBuilder:
    """Builds an Ollama Modelfile from layered components."""

    def __init__(self, base_model: str = "lfm2", user_name: str = "Master") -> None:
        self._base_model = base_model
        self._user_name = user_name
        self._preferences: Optional[dict] = None
        self._knowledge: list[str] = []
        self._instructions: list[str] = []
        self._customizations: list[str] = []
        self._parameters: dict[str, object] = {}

    def set_preferences(
        self,
        verbosity: str = "",
        formality: str = "",
        depth: str = "",
        format_pref: str = "",
    ) -> None:
        """Set the learned preferences layer."""
        self._preferences = {
            "verbosity": verbosity,
            "formality": formality,
            "depth": depth,
            "format": format_pref,
        }

    def set_knowledge(self, facts: list[str]) -> None:
        """Set the knowledge context layer."""
        self._knowledge = facts

    def set_instructions(self, instructions: list[str]) -> None:
        """Set the instructions layer."""
        self._instructions = instructions

    def set_customizations(self, customizations: list[str]) -> None:
        """Set the active customizations layer."""
        self._customizations = customizations

    def set_parameters(self, **params) -> None:
        """Set Modelfile PARAMETER directives."""
        self._parameters.update(params)

    def build(self) -> str:
        """Build the complete Modelfile content."""
        lines = [f"FROM {self._base_model}"]

        # Build SYSTEM prompt from layers
        system_parts = []

        # Base personality (always present)
        system_parts.append(
            f"[Base Personality]\n"
            f"You are Homie, {self._user_name}'s personal AI assistant. "
            f"You are local, private, and evolving. Be helpful, direct, and concise."
        )

        # Learned preferences
        if self._preferences:
            prefs = []
            if self._preferences.get("verbosity"):
                prefs.append(f"- Response style: {self._preferences['verbosity']}")
            if self._preferences.get("formality"):
                prefs.append(f"- Tone: {self._preferences['formality']}")
            if self._preferences.get("depth"):
                prefs.append(f"- Technical depth: {self._preferences['depth']}")
            if self._preferences.get("format"):
                prefs.append(f"- Format: prefer {self._preferences['format']}")
            if prefs:
                system_parts.append("[Learned Preferences]\n" + "\n".join(prefs))

        # Knowledge context
        if self._knowledge:
            facts_text = "\n".join(f"- {f}" for f in self._knowledge)
            system_parts.append(f"[Knowledge Context]\n{facts_text}")

        # Instructions
        if self._instructions:
            inst_text = "\n".join(f"- {i}" for i in self._instructions)
            system_parts.append(f"[Instructions]\n{inst_text}")

        # Active customizations
        if self._customizations:
            cust_text = "\n".join(f"- {c}" for c in self._customizations)
            system_parts.append(f"[Active Customizations]\n{cust_text}")

        system_prompt = "\n\n".join(system_parts)
        lines.append(f'SYSTEM """\n{system_prompt}\n"""')

        # Parameters
        for key, value in sorted(self._parameters.items()):
            lines.append(f"PARAMETER {key} {value}")

        return "\n".join(lines) + "\n"

    def write(self, path: Path | str) -> None:
        """Write the Modelfile to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build())

    def content_hash(self) -> str:
        """Hash the Modelfile content for change detection."""
        return hashlib.sha256(self.build().encode()).hexdigest()[:16]
