# Model Fine-Tuning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a model evolution pipeline that creates, validates, and pushes custom Ollama models from Homie's learned state — preferences, knowledge, customizations, and optimal parameters.

**Architecture:** EvolutionEngine coordinates: ModelfileBuilder assembles layered system prompt from sub-project data, OllamaManager wraps CLI commands, Validator runs benchmark + shadow tests, MilestoneTracker triggers rebuilds, DataCurator collects training data for future adapter training.

**Tech Stack:** Python 3.11+, Ollama CLI (`ollama create/push`), `keyring` (API key storage), existing sub-project APIs (PreferenceEngine, GraphQuery, CustomizationManager, OptimizationProfiler).

**Spec:** `docs/superpowers/specs/2026-03-23-model-fine-tuning-pipeline-design.md`

---

## Chunk 1: Ollama Manager & Modelfile Builder

### Task 1: Ollama Manager

**Files:**
- Create: `src/homie_core/model_evolution/__init__.py`
- Create: `src/homie_core/model_evolution/ollama_manager.py`
- Test: `tests/unit/model_evolution/__init__.py`
- Test: `tests/unit/model_evolution/test_ollama_manager.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/homie_core/model_evolution
mkdir -p tests/unit/model_evolution
```

- [ ] **Step 2: Write failing test**

```python
# tests/unit/model_evolution/__init__.py
```

```python
# tests/unit/model_evolution/test_ollama_manager.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.model_evolution.ollama_manager import OllamaManager


class TestOllamaManager:
    def test_pull_success(self):
        mgr = OllamaManager()
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="success")
            assert mgr.pull("lfm2") is True
            mock_run.assert_called_once()

    def test_pull_failure(self):
        mgr = OllamaManager()
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")
            assert mgr.pull("nonexistent") is False

    def test_create_with_modelfile(self, tmp_path):
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM lfm2\nSYSTEM You are Homie.")
        mgr = OllamaManager()
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="created")
            assert mgr.create("MSG-88/Homie", modelfile) is True

    def test_push_with_api_key(self):
        mgr = OllamaManager(api_key="test-key-123")
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="pushed")
            assert mgr.push("MSG-88/Homie") is True
            # Verify OLLAMA_API_KEY was in the environment
            call_kwargs = mock_run.call_args[1]
            assert "env" in call_kwargs
            assert call_kwargs["env"]["OLLAMA_API_KEY"] == "test-key-123"

    def test_list_models(self):
        mgr = OllamaManager()
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="NAME\nlfm2:latest\nglm-4.7-flash:latest\n")
            models = mgr.list_models()
            assert len(models) >= 2

    def test_show_model(self):
        mgr = OllamaManager()
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Model: lfm2\nParameters: 7B")
            info = mgr.show("lfm2")
            assert isinstance(info, str)
            assert "lfm2" in info

    def test_remove_model(self):
        mgr = OllamaManager()
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert mgr.remove("old-model") is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/unit/model_evolution/test_ollama_manager.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Write implementation**

```python
# src/homie_core/model_evolution/__init__.py
"""Homie Model Evolution — create, validate, and push custom Ollama models."""
```

```python
# src/homie_core/model_evolution/ollama_manager.py
"""OllamaManager — wrapper around Ollama CLI commands."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_OLLAMA_CMD = "ollama"
_TIMEOUT = 300  # 5 minutes for pull/push


class OllamaManager:
    """Manages Ollama model operations via CLI."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key

    def _run(self, args: list[str], timeout: int = _TIMEOUT, env_extra: Optional[dict] = None) -> subprocess.CompletedProcess:
        """Run an ollama command."""
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            [_OLLAMA_CMD] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

    def pull(self, model: str) -> bool:
        """Pull a model from the registry."""
        try:
            result = self._run(["pull", model])
            if result.returncode == 0:
                logger.info("Pulled model: %s", model)
                return True
            logger.warning("Failed to pull %s: %s", model, result.stderr)
            return False
        except Exception as exc:
            logger.error("Pull failed: %s", exc)
            return False

    def create(self, name: str, modelfile: Path | str) -> bool:
        """Create a model from a Modelfile."""
        try:
            result = self._run(["create", name, "-f", str(modelfile)])
            if result.returncode == 0:
                logger.info("Created model: %s", name)
                return True
            logger.warning("Failed to create %s: %s", name, result.stderr)
            return False
        except Exception as exc:
            logger.error("Create failed: %s", exc)
            return False

    def push(self, name: str) -> bool:
        """Push a model to the registry."""
        env_extra = {}
        if self._api_key:
            env_extra["OLLAMA_API_KEY"] = self._api_key
        try:
            result = self._run(["push", name], env_extra=env_extra)
            if result.returncode == 0:
                logger.info("Pushed model: %s", name)
                return True
            logger.warning("Failed to push %s: %s", name, result.stderr)
            return False
        except Exception as exc:
            logger.error("Push failed: %s", exc)
            return False

    def list_models(self) -> list[str]:
        """List installed models."""
        try:
            result = self._run(["list"], timeout=30)
            if result.returncode != 0:
                return []
            lines = result.stdout.strip().split("\n")
            # Skip header line
            return [line.split()[0] for line in lines[1:] if line.strip()]
        except Exception:
            return []

    def show(self, name: str) -> str:
        """Show model info."""
        try:
            result = self._run(["show", name], timeout=30)
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    def remove(self, name: str) -> bool:
        """Remove a model."""
        try:
            result = self._run(["rm", name], timeout=60)
            return result.returncode == 0
        except Exception:
            return False
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/model_evolution/test_ollama_manager.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/model_evolution/ tests/unit/model_evolution/
git commit -m "feat(model-evolution): add OllamaManager CLI wrapper"
```

---

### Task 2: Modelfile Builder

**Files:**
- Create: `src/homie_core/model_evolution/modelfile_builder.py`
- Test: `tests/unit/model_evolution/test_modelfile_builder.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/model_evolution/test_modelfile_builder.py
import pytest
from homie_core.model_evolution.modelfile_builder import ModelfileBuilder


class TestModelfileBuilder:
    def test_builds_basic_modelfile(self):
        builder = ModelfileBuilder(base_model="lfm2")
        content = builder.build()
        assert "FROM lfm2" in content
        assert "SYSTEM" in content

    def test_includes_base_personality(self):
        builder = ModelfileBuilder(base_model="lfm2", user_name="Master")
        content = builder.build()
        assert "Homie" in content
        assert "Master" in content

    def test_includes_preferences_layer(self):
        builder = ModelfileBuilder(base_model="lfm2")
        builder.set_preferences(verbosity="concise", formality="casual", depth="expert", format_pref="bullets")
        content = builder.build()
        assert "concise" in content.lower() or "brief" in content.lower()
        assert "bullet" in content.lower()

    def test_includes_knowledge_layer(self):
        builder = ModelfileBuilder(base_model="lfm2")
        builder.set_knowledge(["Works on Homie AI project", "Uses Python and ChromaDB"])
        content = builder.build()
        assert "Homie AI" in content
        assert "ChromaDB" in content

    def test_includes_instructions_layer(self):
        builder = ModelfileBuilder(base_model="lfm2")
        builder.set_instructions(["Show code diffs first", "Morning greeting includes git summary"])
        content = builder.build()
        assert "diff" in content.lower()

    def test_includes_customizations_layer(self):
        builder = ModelfileBuilder(base_model="lfm2")
        builder.set_customizations(["/standup: show git + calendar", "Morning briefing with project status"])
        content = builder.build()
        assert "standup" in content.lower()

    def test_includes_parameters(self):
        builder = ModelfileBuilder(base_model="lfm2")
        builder.set_parameters(temperature=0.5, num_ctx=32768)
        content = builder.build()
        assert "PARAMETER temperature 0.5" in content
        assert "PARAMETER num_ctx 32768" in content

    def test_write_to_file(self, tmp_path):
        builder = ModelfileBuilder(base_model="lfm2")
        path = tmp_path / "Modelfile"
        builder.write(path)
        assert path.exists()
        assert "FROM lfm2" in path.read_text()

    def test_content_hash(self):
        builder = ModelfileBuilder(base_model="lfm2")
        h1 = builder.content_hash()
        builder.set_knowledge(["New fact"])
        h2 = builder.content_hash()
        assert h1 != h2  # hash changes with content

    def test_same_content_same_hash(self):
        b1 = ModelfileBuilder(base_model="lfm2", user_name="Test")
        b2 = ModelfileBuilder(base_model="lfm2", user_name="Test")
        assert b1.content_hash() == b2.content_hash()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/model_evolution/test_modelfile_builder.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/model_evolution/modelfile_builder.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/model_evolution/test_modelfile_builder.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/model_evolution/modelfile_builder.py tests/unit/model_evolution/test_modelfile_builder.py
git commit -m "feat(model-evolution): add ModelfileBuilder with layered system prompt"
```

---

## Chunk 2: Model Registry, Milestone Tracker & Data Curator

### Task 3: Model Registry

**Files:**
- Create: `src/homie_core/model_evolution/model_registry.py`
- Test: `tests/unit/model_evolution/test_model_registry.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/model_evolution/test_model_registry.py
import pytest
from unittest.mock import MagicMock
from homie_core.model_evolution.model_registry import ModelRegistry, ModelVersion


class TestModelVersion:
    def test_creation(self):
        v = ModelVersion(
            version_id="homie-v1",
            base_model="lfm2",
            ollama_name="MSG-88/Homie",
            modelfile_hash="abc123",
            status="active",
        )
        assert v.version_id == "homie-v1"
        assert v.is_active is True

    def test_to_dict(self):
        v = ModelVersion(version_id="v1", base_model="lfm2", ollama_name="test", modelfile_hash="x")
        d = v.to_dict()
        assert d["version_id"] == "v1"


class TestModelRegistry:
    def test_register_version(self):
        storage = MagicMock()
        reg = ModelRegistry(storage=storage)
        version = reg.register("lfm2", "MSG-88/Homie", "hash123", changelog="Initial version")
        assert version.version_id.startswith("homie-v")
        storage.save_model_version.assert_called()

    def test_get_active_version(self):
        storage = MagicMock()
        storage.get_active_model_version.return_value = {"version_id": "homie-v1", "status": "active", "base_model": "lfm2", "ollama_name": "test", "modelfile_hash": "x", "metrics": "{}", "changelog": "init"}
        reg = ModelRegistry(storage=storage)
        active = reg.get_active()
        assert active is not None
        assert active.version_id == "homie-v1"

    def test_promote_version(self):
        storage = MagicMock()
        reg = ModelRegistry(storage=storage)
        reg.promote("homie-v2")
        storage.update_model_version_status.assert_called_with("homie-v2", "active")

    def test_rollback(self):
        storage = MagicMock()
        storage.get_previous_model_version.return_value = {"version_id": "homie-v1", "status": "archived", "base_model": "lfm2", "ollama_name": "test", "modelfile_hash": "x", "metrics": "{}", "changelog": ""}
        reg = ModelRegistry(storage=storage)
        prev = reg.rollback("homie-v2")
        assert prev is not None
        storage.update_model_version_status.assert_any_call("homie-v2", "rolled_back")

    def test_list_versions(self):
        storage = MagicMock()
        storage.list_model_versions.return_value = [
            {"version_id": "v1", "status": "archived", "base_model": "lfm2", "ollama_name": "t", "modelfile_hash": "x", "metrics": "{}", "changelog": ""},
            {"version_id": "v2", "status": "active", "base_model": "lfm2", "ollama_name": "t", "modelfile_hash": "y", "metrics": "{}", "changelog": ""},
        ]
        reg = ModelRegistry(storage=storage)
        versions = reg.list_versions()
        assert len(versions) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/model_evolution/test_model_registry.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/model_evolution/model_registry.py
"""Model registry — tracks all Homie model versions."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelVersion:
    """A versioned Homie model."""

    version_id: str
    base_model: str
    ollama_name: str
    modelfile_hash: str
    status: str = "created"  # created, shadow_testing, active, archived, rolled_back
    metrics: dict = field(default_factory=dict)
    changelog: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "base_model": self.base_model,
            "ollama_name": self.ollama_name,
            "modelfile_hash": self.modelfile_hash,
            "status": self.status,
            "metrics": json.dumps(self.metrics) if isinstance(self.metrics, dict) else self.metrics,
            "changelog": self.changelog,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        metrics = data.get("metrics", "{}")
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        return cls(
            version_id=data["version_id"],
            base_model=data["base_model"],
            ollama_name=data["ollama_name"],
            modelfile_hash=data["modelfile_hash"],
            status=data.get("status", "created"),
            metrics=metrics,
            changelog=data.get("changelog", ""),
            created_at=data.get("created_at", time.time()),
        )


class ModelRegistry:
    """Manages model version lifecycle."""

    def __init__(self, storage) -> None:
        self._storage = storage
        self._version_counter = 0

    def register(self, base_model: str, ollama_name: str, modelfile_hash: str, changelog: str = "") -> ModelVersion:
        """Register a new model version."""
        self._version_counter += 1
        version = ModelVersion(
            version_id=f"homie-v{self._version_counter}",
            base_model=base_model,
            ollama_name=ollama_name,
            modelfile_hash=modelfile_hash,
            changelog=changelog,
        )
        self._storage.save_model_version(version.version_id, version.to_dict())
        return version

    def get_active(self) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        data = self._storage.get_active_model_version()
        return ModelVersion.from_dict(data) if data else None

    def promote(self, version_id: str) -> None:
        """Promote a version to active, archiving the current active."""
        current = self.get_active()
        if current and current.version_id != version_id:
            self._storage.update_model_version_status(current.version_id, "archived")
        self._storage.update_model_version_status(version_id, "active")

    def rollback(self, current_version_id: str) -> Optional[ModelVersion]:
        """Rollback current version and restore previous."""
        self._storage.update_model_version_status(current_version_id, "rolled_back")
        prev_data = self._storage.get_previous_model_version()
        if prev_data:
            prev = ModelVersion.from_dict(prev_data)
            self._storage.update_model_version_status(prev.version_id, "active")
            return prev
        return None

    def list_versions(self) -> list[ModelVersion]:
        """List all versions."""
        rows = self._storage.list_model_versions()
        return [ModelVersion.from_dict(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/model_evolution/test_model_registry.py -v`
Expected: All 6 tests PASS (including 1 ModelVersion test)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/model_evolution/model_registry.py tests/unit/model_evolution/test_model_registry.py
git commit -m "feat(model-evolution): add ModelRegistry with version lifecycle"
```

---

### Task 4: Milestone Tracker

**Files:**
- Create: `src/homie_core/model_evolution/milestone_tracker.py`
- Test: `tests/unit/model_evolution/test_milestone_tracker.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/model_evolution/test_milestone_tracker.py
import pytest
from homie_core.model_evolution.milestone_tracker import MilestoneTracker


class TestMilestoneTracker:
    def test_no_milestone_initially(self):
        tracker = MilestoneTracker(min_facts=50, min_prefs=10, min_customs=3)
        assert tracker.should_rebuild() is False

    def test_facts_milestone(self):
        tracker = MilestoneTracker(min_facts=5, min_prefs=100, min_customs=100)
        for _ in range(5):
            tracker.record_new_fact()
        assert tracker.should_rebuild() is True

    def test_preference_milestone(self):
        tracker = MilestoneTracker(min_facts=100, min_prefs=3, min_customs=100)
        for _ in range(3):
            tracker.record_preference_change()
        assert tracker.should_rebuild() is True

    def test_customization_milestone(self):
        tracker = MilestoneTracker(min_facts=100, min_prefs=100, min_customs=2)
        tracker.record_new_customization()
        tracker.record_new_customization()
        assert tracker.should_rebuild() is True

    def test_reset_after_rebuild(self):
        tracker = MilestoneTracker(min_facts=2, min_prefs=100, min_customs=100)
        tracker.record_new_fact()
        tracker.record_new_fact()
        assert tracker.should_rebuild() is True
        tracker.reset()
        assert tracker.should_rebuild() is False

    def test_manual_trigger(self):
        tracker = MilestoneTracker(min_facts=100, min_prefs=100, min_customs=100)
        tracker.trigger_manual()
        assert tracker.should_rebuild() is True

    def test_get_summary(self):
        tracker = MilestoneTracker(min_facts=50, min_prefs=10, min_customs=3)
        tracker.record_new_fact()
        tracker.record_new_fact()
        summary = tracker.get_summary()
        assert summary["new_facts"] == 2
        assert summary["thresholds"]["min_facts"] == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/model_evolution/test_milestone_tracker.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/model_evolution/milestone_tracker.py
"""Milestone tracker — determines when to rebuild the Homie model."""

import threading


class MilestoneTracker:
    """Tracks changes since last model push, triggers rebuild at thresholds."""

    def __init__(
        self,
        min_facts: int = 50,
        min_prefs: int = 10,
        min_customs: int = 3,
    ) -> None:
        self._min_facts = min_facts
        self._min_prefs = min_prefs
        self._min_customs = min_customs
        self._lock = threading.Lock()
        self._new_facts = 0
        self._pref_changes = 0
        self._new_customs = 0
        self._manual = False

    def record_new_fact(self) -> None:
        with self._lock:
            self._new_facts += 1

    def record_preference_change(self) -> None:
        with self._lock:
            self._pref_changes += 1

    def record_new_customization(self) -> None:
        with self._lock:
            self._new_customs += 1

    def trigger_manual(self) -> None:
        with self._lock:
            self._manual = True

    def should_rebuild(self) -> bool:
        """Check if any milestone threshold has been crossed."""
        with self._lock:
            if self._manual:
                return True
            return (
                self._new_facts >= self._min_facts
                or self._pref_changes >= self._min_prefs
                or self._new_customs >= self._min_customs
            )

    def reset(self) -> None:
        """Reset counters after a successful rebuild."""
        with self._lock:
            self._new_facts = 0
            self._pref_changes = 0
            self._new_customs = 0
            self._manual = False

    def get_summary(self) -> dict:
        with self._lock:
            return {
                "new_facts": self._new_facts,
                "preference_changes": self._pref_changes,
                "new_customizations": self._new_customs,
                "manual_triggered": self._manual,
                "thresholds": {
                    "min_facts": self._min_facts,
                    "min_prefs": self._min_prefs,
                    "min_customs": self._min_customs,
                },
            }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/model_evolution/test_milestone_tracker.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/model_evolution/milestone_tracker.py tests/unit/model_evolution/test_milestone_tracker.py
git commit -m "feat(model-evolution): add MilestoneTracker for rebuild triggers"
```

---

### Task 5: Data Curator

**Files:**
- Create: `src/homie_core/model_evolution/data_curator.py`
- Test: `tests/unit/model_evolution/test_data_curator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/model_evolution/test_data_curator.py
import pytest
from unittest.mock import MagicMock
from homie_core.model_evolution.data_curator import DataCurator


class TestDataCurator:
    def test_collect_sft_example(self):
        storage = MagicMock()
        curator = DataCurator(storage=storage)
        curator.collect_sft(
            system_prompt="You are Homie.",
            user_message="What is Python?",
            response="A programming language.",
            quality_score=0.8,
        )
        storage.save_training_example.assert_called_once()

    def test_collect_dpo_pair(self):
        storage = MagicMock()
        curator = DataCurator(storage=storage)
        curator.collect_dpo(
            user_message="Explain decorators",
            chosen="Good response here",
            rejected="Bad response here",
        )
        storage.save_training_example.assert_called_once()
        call_data = storage.save_training_example.call_args[1]
        assert call_data["example_type"] == "dpo"

    def test_export_sft_jsonl(self, tmp_path):
        storage = MagicMock()
        storage.get_training_examples.return_value = [
            {"example_type": "sft", "data": '{"instruction": "sys", "input": "hi", "output": "hello"}', "quality_score": 0.9},
        ]
        curator = DataCurator(storage=storage)
        path = tmp_path / "sft.jsonl"
        count = curator.export_sft(path)
        assert count == 1
        assert path.exists()

    def test_export_dpo_jsonl(self, tmp_path):
        storage = MagicMock()
        storage.get_training_examples.return_value = [
            {"example_type": "dpo", "data": '{"prompt": "q", "chosen": "good", "rejected": "bad"}', "quality_score": 0.0},
        ]
        curator = DataCurator(storage=storage)
        path = tmp_path / "dpo.jsonl"
        count = curator.export_dpo(path)
        assert count == 1

    def test_get_stats(self):
        storage = MagicMock()
        storage.count_training_examples.return_value = {"sft": 100, "dpo": 25}
        curator = DataCurator(storage=storage)
        stats = curator.get_stats()
        assert stats["sft"] == 100
        assert stats["dpo"] == 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/model_evolution/test_data_curator.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/model_evolution/data_curator.py
"""Data curator — collects and exports SFT/DPO training data."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DataCurator:
    """Curates training data from Homie's interactions."""

    def __init__(self, storage) -> None:
        self._storage = storage

    def collect_sft(
        self,
        system_prompt: str,
        user_message: str,
        response: str,
        quality_score: float,
    ) -> None:
        """Collect an SFT training example."""
        data = {
            "instruction": system_prompt,
            "input": user_message,
            "output": response,
        }
        self._storage.save_training_example(
            example_type="sft",
            data=json.dumps(data),
            quality_score=quality_score,
        )

    def collect_dpo(
        self,
        user_message: str,
        chosen: str,
        rejected: str,
    ) -> None:
        """Collect a DPO preference pair."""
        data = {
            "prompt": user_message,
            "chosen": chosen,
            "rejected": rejected,
        }
        self._storage.save_training_example(
            example_type="dpo",
            data=json.dumps(data),
            quality_score=0.0,
        )

    def export_sft(self, output_path: Path | str, min_quality: float = 0.5) -> int:
        """Export SFT examples as JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        examples = self._storage.get_training_examples(example_type="sft")
        count = 0
        with open(output_path, "w") as f:
            for ex in examples:
                if ex.get("quality_score", 0) >= min_quality:
                    f.write(ex["data"] + "\n")
                    count += 1
        logger.info("Exported %d SFT examples to %s", count, output_path)
        return count

    def export_dpo(self, output_path: Path | str) -> int:
        """Export DPO pairs as JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        examples = self._storage.get_training_examples(example_type="dpo")
        count = 0
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(ex["data"] + "\n")
                count += 1
        logger.info("Exported %d DPO pairs to %s", count, output_path)
        return count

    def get_stats(self) -> dict:
        """Get training data statistics."""
        return self._storage.count_training_examples()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/model_evolution/test_data_curator.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/model_evolution/data_curator.py tests/unit/model_evolution/test_data_curator.py
git commit -m "feat(model-evolution): add DataCurator for SFT/DPO training data"
```

---

## Chunk 3: Validator, Evolution Engine & Integration

### Task 6: Validator

**Files:**
- Create: `src/homie_core/model_evolution/validator.py`
- Test: `tests/unit/model_evolution/test_validator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/model_evolution/test_validator.py
import pytest
from unittest.mock import MagicMock
from homie_core.model_evolution.validator import ModelValidator, BenchmarkResult


class TestBenchmarkResult:
    def test_passes_above_threshold(self):
        r = BenchmarkResult(scores={"style": 0.8, "knowledge": 0.9, "instructions": 0.7}, threshold=0.7)
        assert r.passed is True
        assert r.average_score > 0.7

    def test_fails_below_threshold(self):
        r = BenchmarkResult(scores={"style": 0.5, "knowledge": 0.4}, threshold=0.7)
        assert r.passed is False


class TestModelValidator:
    def test_run_benchmark(self):
        inference_fn = MagicMock(return_value="Python is a programming language.")
        validator = ModelValidator(inference_fn=inference_fn, benchmark_threshold=0.5)
        result = validator.run_benchmark()
        assert isinstance(result, BenchmarkResult)
        assert len(result.scores) > 0

    def test_benchmark_calls_inference(self):
        inference_fn = MagicMock(return_value="response")
        validator = ModelValidator(inference_fn=inference_fn)
        validator.run_benchmark()
        assert inference_fn.call_count >= 3  # at least a few benchmark prompts

    def test_shadow_test_result(self):
        validator = ModelValidator(inference_fn=MagicMock(return_value="ok"))
        validator.record_shadow_result(old_score=0.6, new_score=0.8)
        validator.record_shadow_result(old_score=0.7, new_score=0.9)
        assert validator.shadow_test_passed(min_win_rate=0.5) is True

    def test_shadow_test_fails_when_old_better(self):
        validator = ModelValidator(inference_fn=MagicMock(return_value="ok"))
        for _ in range(5):
            validator.record_shadow_result(old_score=0.9, new_score=0.3)
        assert validator.shadow_test_passed(min_win_rate=0.6) is False

    def test_shadow_test_needs_enough_samples(self):
        validator = ModelValidator(inference_fn=MagicMock(return_value="ok"), shadow_min_queries=10)
        validator.record_shadow_result(old_score=0.5, new_score=0.9)
        assert validator.shadow_test_passed() is False  # not enough samples
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/model_evolution/test_validator.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/model_evolution/validator.py
"""Model validator — benchmark suite and A/B shadow testing."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Fixed benchmark prompts
_BENCHMARK_PROMPTS = [
    {"category": "style", "prompt": "Explain what a Python decorator is.", "expected_traits": ["concise", "code"]},
    {"category": "knowledge", "prompt": "What projects am I working on?", "expected_traits": ["project", "working"]},
    {"category": "instructions", "prompt": "List three benefits of testing.", "expected_traits": ["1", "2", "3"]},
    {"category": "personality", "prompt": "Good morning!", "expected_traits": ["homie", "morning", "hello", "hey"]},
    {"category": "format", "prompt": "Give me a bullet point summary of Python features.", "expected_traits": ["-", "*", "•"]},
]


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    scores: dict[str, float]
    threshold: float = 0.7

    @property
    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    @property
    def passed(self) -> bool:
        return self.average_score >= self.threshold


class ModelValidator:
    """Validates model quality via benchmarks and shadow testing."""

    def __init__(
        self,
        inference_fn: Callable[[str], str],
        benchmark_threshold: float = 0.7,
        shadow_min_queries: int = 50,
    ) -> None:
        self._infer = inference_fn
        self._benchmark_threshold = benchmark_threshold
        self._shadow_min = shadow_min_queries
        self._shadow_results: list[tuple[float, float]] = []  # (old_score, new_score)

    def run_benchmark(self) -> BenchmarkResult:
        """Run the benchmark suite against the model."""
        scores = {}
        for bench in _BENCHMARK_PROMPTS:
            try:
                response = self._infer(bench["prompt"])
                score = self._score_response(response, bench["expected_traits"])
                scores[bench["category"]] = score
            except Exception as exc:
                logger.warning("Benchmark failed for %s: %s", bench["category"], exc)
                scores[bench["category"]] = 0.0

        result = BenchmarkResult(scores=scores, threshold=self._benchmark_threshold)
        logger.info("Benchmark: avg=%.2f passed=%s", result.average_score, result.passed)
        return result

    def _score_response(self, response: str, expected_traits: list[str]) -> float:
        """Score a response based on expected traits."""
        if not response:
            return 0.0
        response_lower = response.lower()
        matches = sum(1 for t in expected_traits if t.lower() in response_lower)
        # Base score: 0.5 for any response, up to 1.0 for matching traits
        trait_score = matches / max(len(expected_traits), 1)
        return min(1.0, 0.5 + 0.5 * trait_score)

    def record_shadow_result(self, old_score: float, new_score: float) -> None:
        """Record a shadow test comparison."""
        self._shadow_results.append((old_score, new_score))

    def shadow_test_passed(self, min_win_rate: float = 0.6) -> bool:
        """Check if shadow testing has passed."""
        if len(self._shadow_results) < self._shadow_min:
            return False
        wins = sum(1 for old, new in self._shadow_results if new >= old)
        win_rate = wins / len(self._shadow_results)
        return win_rate >= min_win_rate

    def reset_shadow(self) -> None:
        """Reset shadow test results."""
        self._shadow_results.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/model_evolution/test_validator.py -v`
Expected: All 7 tests PASS (2 BenchmarkResult + 5 Validator)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/model_evolution/validator.py tests/unit/model_evolution/test_validator.py
git commit -m "feat(model-evolution): add ModelValidator with benchmark and shadow testing"
```

---

### Task 7: Evolution Engine

**Files:**
- Create: `src/homie_core/model_evolution/evolution_engine.py`
- Test: `tests/unit/model_evolution/test_evolution_engine.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/model_evolution/test_evolution_engine.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.model_evolution.evolution_engine import EvolutionEngine


class TestEvolutionEngine:
    def _make_engine(self, **overrides):
        defaults = {
            "storage": MagicMock(),
            "ollama_manager": MagicMock(),
            "preference_engine": MagicMock(),
            "knowledge_query": MagicMock(),
            "customization_manager": MagicMock(),
            "profiler": MagicMock(),
            "inference_fn": MagicMock(return_value="test response"),
            "base_model": "lfm2",
            "registry_name": "MSG-88/Homie",
            "user_name": "Master",
            "modelfile_dir": "/tmp/homie",
        }
        defaults.update(overrides)
        return EvolutionEngine(**defaults)

    def test_build_modelfile(self):
        pref = MagicMock()
        pref.get_active_profile.return_value = MagicMock(verbosity=0.2, formality=0.3, technical_depth=0.8, format_preference="bullets")
        km = MagicMock()
        km.list_customizations.return_value = [{"request_text": "/standup command", "status": "active"}]
        engine = self._make_engine(preference_engine=pref, customization_manager=km)
        builder = engine.build_modelfile()
        content = builder.build()
        assert "FROM lfm2" in content
        assert "Master" in content

    def test_check_milestone_returns_false_initially(self):
        engine = self._make_engine()
        assert engine.should_evolve() is False

    def test_manual_trigger(self):
        engine = self._make_engine()
        engine.trigger_evolution()
        assert engine.should_evolve() is True

    def test_evolve_creates_model(self):
        ollama = MagicMock()
        ollama.create.return_value = True
        storage = MagicMock()
        storage.get_active_model_version.return_value = None
        storage.get_previous_model_version.return_value = None
        engine = self._make_engine(ollama_manager=ollama, storage=storage)
        engine.trigger_evolution()
        result = engine.evolve()
        assert result["status"] in ("created", "benchmark_passed", "promoted")
        ollama.create.assert_called()

    def test_evolve_skips_if_no_milestone(self):
        engine = self._make_engine()
        result = engine.evolve()
        assert result["status"] == "no_changes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/model_evolution/test_evolution_engine.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/model_evolution/evolution_engine.py
"""Evolution engine — coordinates the full model evolution pipeline."""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from .milestone_tracker import MilestoneTracker
from .model_registry import ModelRegistry
from .modelfile_builder import ModelfileBuilder
from .ollama_manager import OllamaManager
from .validator import ModelValidator

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """Coordinates model evolution: build → validate → push."""

    def __init__(
        self,
        storage,
        ollama_manager: OllamaManager,
        preference_engine,
        knowledge_query,
        customization_manager,
        profiler,
        inference_fn: Callable[[str], str],
        base_model: str = "lfm2",
        registry_name: str = "MSG-88/Homie",
        user_name: str = "Master",
        modelfile_dir: str | Path = "",
        benchmark_threshold: float = 0.7,
        min_facts: int = 50,
        min_prefs: int = 10,
        min_customs: int = 3,
    ) -> None:
        self._storage = storage
        self._ollama = ollama_manager
        self._pref = preference_engine
        self._kg = knowledge_query
        self._customs = customization_manager
        self._profiler = profiler
        self._infer = inference_fn
        self._base_model = base_model
        self._registry_name = registry_name
        self._user_name = user_name
        self._modelfile_dir = Path(modelfile_dir) if modelfile_dir else Path.home() / ".homie" / "model_evolution"

        self._registry = ModelRegistry(storage=storage)
        self._milestone = MilestoneTracker(min_facts=min_facts, min_prefs=min_prefs, min_customs=min_customs)
        self._validator = ModelValidator(inference_fn=inference_fn, benchmark_threshold=benchmark_threshold)
        self._last_hash: Optional[str] = None

    def should_evolve(self) -> bool:
        """Check if model should be rebuilt."""
        return self._milestone.should_rebuild()

    def trigger_evolution(self) -> None:
        """Manually trigger model evolution."""
        self._milestone.trigger_manual()

    def record_new_fact(self) -> None:
        self._milestone.record_new_fact()

    def record_preference_change(self) -> None:
        self._milestone.record_preference_change()

    def record_new_customization(self) -> None:
        self._milestone.record_new_customization()

    def build_modelfile(self) -> ModelfileBuilder:
        """Build the Modelfile from current learned state."""
        builder = ModelfileBuilder(base_model=self._base_model, user_name=self._user_name)

        # Preferences layer
        try:
            profile = self._pref.get_active_profile()
            verb = "concise" if profile.verbosity < 0.4 else "detailed" if profile.verbosity > 0.7 else ""
            form = "casual" if profile.formality < 0.4 else "formal" if profile.formality > 0.7 else ""
            depth = "expert" if profile.technical_depth > 0.7 else "simple" if profile.technical_depth < 0.3 else ""
            builder.set_preferences(verbosity=verb, formality=form, depth=depth, format_pref=profile.format_preference)
        except Exception:
            pass

        # Customizations layer
        try:
            customs = self._customs.list_customizations()
            active = [c["request_text"] for c in customs if c.get("status") == "active"]
            if active:
                builder.set_customizations(active)
        except Exception:
            pass

        # Parameters from profiler
        try:
            profile = self._profiler.get_profile("general")
            if profile and profile.sample_count > 5:
                builder.set_parameters(temperature=round(profile.temperature, 2))
        except Exception:
            pass

        return builder

    def evolve(self) -> dict[str, Any]:
        """Run the evolution pipeline. Returns status dict."""
        if not self.should_evolve():
            return {"status": "no_changes"}

        # Build Modelfile
        builder = self.build_modelfile()
        new_hash = builder.content_hash()

        # Check if content actually changed
        if new_hash == self._last_hash:
            self._milestone.reset()
            return {"status": "no_changes", "reason": "modelfile unchanged"}

        # Write Modelfile
        self._modelfile_dir.mkdir(parents=True, exist_ok=True)
        modelfile_path = self._modelfile_dir / "Modelfile"
        builder.write(modelfile_path)

        # Create model via Ollama
        if not self._ollama.create(self._registry_name, modelfile_path):
            return {"status": "create_failed"}

        # Register version
        version = self._registry.register(
            self._base_model, self._registry_name, new_hash,
            changelog=f"Milestone rebuild (hash: {new_hash})",
        )

        # Run benchmark
        benchmark = self._validator.run_benchmark()
        if not benchmark.passed:
            return {
                "status": "benchmark_failed",
                "version": version.version_id,
                "scores": benchmark.scores,
            }

        # Promote (skip shadow test for now — can be wired later)
        self._registry.promote(version.version_id)
        self._last_hash = new_hash
        self._milestone.reset()

        logger.info("Model evolved to %s (hash: %s)", version.version_id, new_hash)
        return {
            "status": "promoted",
            "version": version.version_id,
            "benchmark_scores": benchmark.scores,
        }
```

- [ ] **Step 4: Update __init__.py**

```python
# src/homie_core/model_evolution/__init__.py
"""Homie Model Evolution — create, validate, and push custom Ollama models."""
from .evolution_engine import EvolutionEngine
from .modelfile_builder import ModelfileBuilder
from .ollama_manager import OllamaManager

__all__ = ["EvolutionEngine", "ModelfileBuilder", "OllamaManager"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/model_evolution/test_evolution_engine.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/model_evolution/ tests/unit/model_evolution/test_evolution_engine.py
git commit -m "feat(model-evolution): add EvolutionEngine coordinator"
```

---

### Task 8: Storage, Config & Integration Test

**Files:**
- Modify: `src/homie_core/adaptive_learning/storage.py`
- Modify: `src/homie_core/config.py`
- Modify: `homie.config.yaml`
- Test: `tests/unit/model_evolution/test_storage.py`
- Test: `tests/unit/model_evolution/test_config.py`
- Test: `tests/integration/test_model_evolution_lifecycle.py`

- [ ] **Step 1: Write storage test**

```python
# tests/unit/model_evolution/test_storage.py
import pytest
from homie_core.adaptive_learning.storage import LearningStorage


class TestModelEvolutionStorage:
    def test_save_and_get_model_version(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.save_model_version("v1", {"version_id": "v1", "base_model": "lfm2", "status": "active", "ollama_name": "test", "modelfile_hash": "x", "metrics": "{}", "changelog": ""})
        result = store.get_active_model_version()
        assert result is not None
        assert result["version_id"] == "v1"

    def test_update_model_version_status(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.save_model_version("v1", {"version_id": "v1", "base_model": "lfm2", "status": "active", "ollama_name": "t", "modelfile_hash": "x", "metrics": "{}", "changelog": ""})
        store.update_model_version_status("v1", "archived")
        result = store.get_active_model_version()
        assert result is None

    def test_save_and_get_training_example(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.save_training_example(example_type="sft", data='{"input": "hi"}', quality_score=0.9)
        examples = store.get_training_examples(example_type="sft")
        assert len(examples) == 1

    def test_count_training_examples(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.save_training_example(example_type="sft", data='{}', quality_score=0.8)
        store.save_training_example(example_type="dpo", data='{}', quality_score=0.0)
        counts = store.count_training_examples()
        assert counts["sft"] == 1
        assert counts["dpo"] == 1
```

- [ ] **Step 2: Add tables and methods to storage.py**

Read `src/homie_core/adaptive_learning/storage.py`. In `initialize()`, add:

```sql
CREATE TABLE IF NOT EXISTS model_versions (
    version_id TEXT PRIMARY KEY,
    base_model TEXT NOT NULL,
    ollama_name TEXT NOT NULL,
    modelfile_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    metrics TEXT NOT NULL DEFAULT '{}',
    changelog TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS training_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    example_type TEXT NOT NULL,
    data TEXT NOT NULL,
    quality_score REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_training_type ON training_data(example_type);
```

Add methods:

```python
def save_model_version(self, version_id: str, data: dict) -> None:
    if self._conn is None:
        return
    with self._lock:
        self._conn.execute(
            """INSERT OR REPLACE INTO model_versions
               (version_id, base_model, ollama_name, modelfile_hash, status, metrics, changelog, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (version_id, data.get("base_model", ""), data.get("ollama_name", ""),
             data.get("modelfile_hash", ""), data.get("status", "created"),
             data.get("metrics", "{}"), data.get("changelog", ""), time.time()),
        )
        self._conn.commit()

def get_active_model_version(self) -> Optional[dict]:
    if self._conn is None:
        return None
    row = self._conn.execute("SELECT * FROM model_versions WHERE status = 'active' ORDER BY created_at DESC LIMIT 1").fetchone()
    return dict(row) if row else None

def update_model_version_status(self, version_id: str, status: str) -> None:
    if self._conn is None:
        return
    with self._lock:
        self._conn.execute("UPDATE model_versions SET status = ? WHERE version_id = ?", (status, version_id))
        self._conn.commit()

def get_previous_model_version(self) -> Optional[dict]:
    if self._conn is None:
        return None
    row = self._conn.execute("SELECT * FROM model_versions WHERE status = 'archived' ORDER BY created_at DESC LIMIT 1").fetchone()
    return dict(row) if row else None

def list_model_versions(self) -> list[dict]:
    if self._conn is None:
        return []
    rows = self._conn.execute("SELECT * FROM model_versions ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]

def save_training_example(self, example_type: str, data: str, quality_score: float = 0.0) -> None:
    if self._conn is None:
        return
    with self._lock:
        self._conn.execute(
            "INSERT INTO training_data (example_type, data, quality_score, created_at) VALUES (?, ?, ?, ?)",
            (example_type, data, quality_score, time.time()),
        )
        self._conn.commit()

def get_training_examples(self, example_type: Optional[str] = None, limit: int = 1000) -> list[dict]:
    if self._conn is None:
        return []
    if example_type:
        rows = self._conn.execute("SELECT * FROM training_data WHERE example_type = ? ORDER BY created_at DESC LIMIT ?", (example_type, limit)).fetchall()
    else:
        rows = self._conn.execute("SELECT * FROM training_data ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]

def count_training_examples(self) -> dict[str, int]:
    if self._conn is None:
        return {"sft": 0, "dpo": 0}
    result = {}
    for etype in ("sft", "dpo"):
        row = self._conn.execute("SELECT COUNT(*) as c FROM training_data WHERE example_type = ?", (etype,)).fetchone()
        result[etype] = row["c"] if row else 0
    return result
```

- [ ] **Step 3: Write config test**

```python
# tests/unit/model_evolution/test_config.py
import pytest
from homie_core.config import ModelEvolutionConfig


class TestModelEvolutionConfig:
    def test_defaults(self):
        cfg = ModelEvolutionConfig()
        assert cfg.enabled is True
        assert cfg.ollama_registry_name == "MSG-88/Homie"
        assert cfg.ollama_base_model == "lfm2"
        assert cfg.milestones_min_facts == 50
        assert cfg.validation_benchmark_min_score == 0.7
```

- [ ] **Step 4: Add config class**

Read `src/homie_core/config.py`. Add before HomieConfig:

```python
class ModelEvolutionConfig(BaseModel):
    enabled: bool = True
    ollama_registry_name: str = "MSG-88/Homie"
    ollama_base_model: str = "lfm2"
    milestones_min_facts: int = 50
    milestones_min_prefs: int = 10
    milestones_min_customs: int = 3
    validation_benchmark_min_score: float = 0.7
    validation_shadow_queries: int = 50
    validation_shadow_max_hours: int = 24
    validation_promotion_threshold: float = 0.6
    sft_collection: bool = True
    dpo_collection: bool = True
```

Add to HomieConfig: `model_evolution: ModelEvolutionConfig = Field(default_factory=ModelEvolutionConfig)`

Add `model_evolution:` section to `homie.config.yaml`.

- [ ] **Step 5: Write integration test**

```python
# tests/integration/test_model_evolution_lifecycle.py
"""Integration test: model evolution lifecycle."""
import pytest
from unittest.mock import MagicMock
from homie_core.model_evolution.evolution_engine import EvolutionEngine
from homie_core.model_evolution.modelfile_builder import ModelfileBuilder
from homie_core.model_evolution.data_curator import DataCurator
from homie_core.adaptive_learning.storage import LearningStorage


class TestModelEvolutionLifecycle:
    def test_modelfile_builds_from_preferences(self):
        builder = ModelfileBuilder(base_model="lfm2", user_name="Master")
        builder.set_preferences(verbosity="concise", depth="expert", format_pref="bullets")
        builder.set_knowledge(["Works on Homie AI", "Uses Python"])
        builder.set_customizations(["/standup: git + calendar"])
        builder.set_parameters(temperature=0.5, num_ctx=32768)
        content = builder.build()
        assert "FROM lfm2" in content
        assert "Master" in content
        assert "concise" in content.lower()
        assert "Homie AI" in content
        assert "standup" in content.lower()
        assert "PARAMETER temperature 0.5" in content

    def test_evolution_with_milestone_trigger(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        engine = EvolutionEngine(
            storage=storage,
            ollama_manager=MagicMock(create=MagicMock(return_value=True)),
            preference_engine=MagicMock(get_active_profile=MagicMock(return_value=MagicMock(verbosity=0.3, formality=0.5, technical_depth=0.8, format_preference="bullets"))),
            knowledge_query=MagicMock(),
            customization_manager=MagicMock(list_customizations=MagicMock(return_value=[])),
            profiler=MagicMock(get_profile=MagicMock(return_value=None)),
            inference_fn=MagicMock(return_value="Homie here! Python is great for coding."),
            base_model="lfm2",
            registry_name="MSG-88/Homie",
            modelfile_dir=str(tmp_path / "modelfiles"),
            min_facts=1,
        )
        engine.record_new_fact()
        result = engine.evolve()
        assert result["status"] in ("promoted", "benchmark_passed", "created")

    def test_training_data_curation(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        curator = DataCurator(storage=storage)
        curator.collect_sft("sys", "What is Python?", "A language.", 0.9)
        curator.collect_dpo("Explain X", "Good explanation", "Bad explanation")
        stats = curator.get_stats()
        assert stats["sft"] == 1
        assert stats["dpo"] == 1
        # Export
        sft_count = curator.export_sft(tmp_path / "sft.jsonl")
        assert sft_count == 1
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest tests/unit/model_evolution/ tests/integration/test_model_evolution_lifecycle.py -v`
Expected: All tests PASS

- [ ] **Step 7: Run full regression**

Run: `python -m pytest tests/unit/self_healing/ tests/unit/adaptive_learning/ tests/unit/knowledge_evolution/ tests/unit/self_optimizer/ tests/unit/model_evolution/ tests/integration/ -q --tb=short`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/homie_core/adaptive_learning/storage.py src/homie_core/config.py homie.config.yaml tests/unit/model_evolution/test_storage.py tests/unit/model_evolution/test_config.py tests/integration/test_model_evolution_lifecycle.py
git commit -m "feat(model-evolution): add storage tables, config, and lifecycle tests"
```
