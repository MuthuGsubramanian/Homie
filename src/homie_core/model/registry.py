from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ModelEntry:
    name: str
    path: str
    format: str
    params: str
    repo_id: str = ""
    quant: str = ""
    active: bool = False


class ModelRegistry:
    def __init__(self, models_dir: Path | str):
        self.models_dir = Path(models_dir)
        self._registry_path = self.models_dir / "registry.json"
        self._entries: dict[str, ModelEntry] = {}

    def initialize(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if self._registry_path.exists():
            data = json.loads(self._registry_path.read_text())
            self._entries = {k: ModelEntry(**v) for k, v in data.items()}

    def _save(self) -> None:
        data = {k: asdict(v) for k, v in self._entries.items()}
        self._registry_path.write_text(json.dumps(data, indent=2))

    def register(self, name: str, path: Path | str, format: str, params: str, repo_id: str = "", quant: str = "") -> ModelEntry:
        entry = ModelEntry(name=name, path=str(path), format=format, params=params, repo_id=repo_id, quant=quant)
        self._entries[name] = entry
        self._save()
        return entry

    def get(self, name: str) -> Optional[ModelEntry]:
        return self._entries.get(name)

    def list_models(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def remove(self, name: str) -> None:
        self._entries.pop(name, None)
        self._save()

    def set_active(self, name: str) -> None:
        for entry in self._entries.values():
            entry.active = False
        if name in self._entries:
            self._entries[name].active = True
        self._save()

    def get_active(self) -> Optional[ModelEntry]:
        for entry in self._entries.values():
            if entry.active:
                return entry
        return None
