# Always-Active Intelligence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform Homie into an always-active intelligent assistant with background observation, task graph inference, proactive retrieval, adaptive interruptions, overlay UI, voice/hotkey activation, morning briefing, end-of-day digest, and enterprise policy support.

**Architecture:** A single-process daemon with three threads (main/observer/scheduler). The observer subscribes to OS events (zero-polling), the task graph algorithm clusters observations into user tasks, and the overlay popup provides instant access via Alt+8 or wake word. Enterprise policy is a YAML file that overrides personal config.

**Tech Stack:** Python stdlib threading, pynput (hotkey), tkinter (overlay), APScheduler (scheduling), numpy (interruption model), existing homie_core modules.

---

### Task 1: Task Graph Data Structure & Algorithm

**Files:**
- Create: `src/homie_core/intelligence/task_graph.py`
- Create: `src/homie_core/intelligence/__init__.py`
- Test: `tests/unit/test_intelligence/test_task_graph.py`
- Create: `tests/unit/test_intelligence/__init__.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_intelligence/__init__.py` (empty file).

Create `tests/unit/test_intelligence/test_task_graph.py`:

```python
from datetime import datetime, timezone, timedelta

from homie_core.intelligence.task_graph import TaskGraph, TaskNode


def _ts(minutes_ago: int) -> str:
    dt = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.isoformat()


def test_new_observation_creates_task():
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(0))
    tasks = tg.get_tasks()
    assert len(tasks) == 1
    assert tasks[0].state == "active"
    assert "Code.exe" in tasks[0].apps


def test_related_observation_joins_existing_task():
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(2))
    tg.observe(process="Code.exe", title="config.py - Homie", timestamp=_ts(1))
    tasks = tg.get_tasks()
    assert len(tasks) == 1
    assert len(tasks[0].windows) == 2


def test_unrelated_observation_creates_new_task():
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(10))
    tg.observe(process="spotify.exe", title="Spotify - Now Playing", timestamp=_ts(3))
    tasks = tg.get_tasks()
    assert len(tasks) == 2


def test_task_pauses_after_inactivity():
    tg = TaskGraph(boundary_minutes=5)
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(20))
    tg.observe(process="chrome.exe", title="Google", timestamp=_ts(10))
    # 10 minutes gap > 5 min boundary => first task should be paused
    tasks = tg.get_tasks()
    code_task = [t for t in tasks if "Code.exe" in t.apps][0]
    assert code_task.state == "paused"


def test_task_resumes_when_revisited():
    tg = TaskGraph(boundary_minutes=5)
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(20))
    tg.observe(process="chrome.exe", title="Google", timestamp=_ts(10))
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(0))
    tasks = tg.get_tasks()
    code_task = [t for t in tasks if "Code.exe" in t.apps][0]
    assert code_task.state == "active"


def test_stuck_detection():
    tg = TaskGraph(boundary_minutes=5, stuck_minutes=15)
    # Same task with lots of switches in 15+ minutes
    for i in range(20, 0, -1):
        proc = "Code.exe" if i % 2 == 0 else "chrome.exe"
        tg.observe(process=proc, title="engine.py - Homie", timestamp=_ts(i))
    tasks = tg.get_tasks()
    # The task has many rapid switches over 20 minutes — should be stuck
    assert any(t.state == "stuck" for t in tasks)


def test_serialize_and_restore():
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(5))
    data = tg.serialize()
    tg2 = TaskGraph.deserialize(data)
    assert len(tg2.get_tasks()) == 1
    assert tg2.get_tasks()[0].apps == tg.get_tasks()[0].apps


def test_get_incomplete_tasks():
    tg = TaskGraph(boundary_minutes=5)
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(20))
    tg.observe(process="chrome.exe", title="Google", timestamp=_ts(10))
    incomplete = tg.get_incomplete_tasks()
    assert len(incomplete) >= 1


def test_summary_generation():
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(5))
    tg.observe(process="Code.exe", title="config.py - Homie", timestamp=_ts(3))
    summary = tg.summarize()
    assert "Code.exe" in summary
    assert isinstance(summary, str)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_intelligence/test_task_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'homie_core.intelligence'`

**Step 3: Write the implementation**

Create `src/homie_core/intelligence/__init__.py` (empty file).

Create `src/homie_core/intelligence/task_graph.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any


@dataclass
class TaskNode:
    """A cluster of related user activity."""
    id: str
    apps: set[str] = field(default_factory=set)
    windows: list[dict] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    state: str = "active"  # active, paused, stuck, completed
    switch_count: int = 0

    def duration_minutes(self) -> float:
        try:
            start = datetime.fromisoformat(self.first_seen)
            end = datetime.fromisoformat(self.last_seen)
            return (end - start).total_seconds() / 60
        except (ValueError, TypeError):
            return 0.0


class TaskGraph:
    """Infers user tasks from observation streams.

    Groups observations by semantic similarity (same project in window title)
    and temporal proximity. Detects task state transitions:
    active -> paused (inactivity > boundary_minutes)
    active -> stuck (duration > stuck_minutes AND high switch rate)
    paused -> active (revisited)
    """

    def __init__(self, boundary_minutes: int = 5, stuck_minutes: int = 15,
                 stuck_switch_rate: float = 0.5):
        self._tasks: list[TaskNode] = []
        self._next_id = 0
        self._boundary = timedelta(minutes=boundary_minutes)
        self._stuck_minutes = stuck_minutes
        self._stuck_switch_rate = stuck_switch_rate
        self._last_observation: dict | None = None

    def observe(self, process: str, title: str, timestamp: str) -> None:
        obs = {"process": process, "title": title, "timestamp": timestamp}
        ts = datetime.fromisoformat(timestamp)

        # Try to find an existing task this belongs to
        matched_task = self._find_matching_task(process, title, ts)

        if matched_task:
            matched_task.windows.append(obs)
            matched_task.last_seen = timestamp
            matched_task.apps.add(process)
            if matched_task.state == "paused":
                matched_task.state = "active"
            # Count switches
            if self._last_observation and self._last_observation["process"] != process:
                matched_task.switch_count += 1
        else:
            # Pause old active tasks if time gap is large
            for t in self._tasks:
                if t.state == "active" and t.last_seen:
                    last = datetime.fromisoformat(t.last_seen)
                    if ts - last > self._boundary:
                        t.state = "paused"

            task = TaskNode(
                id=f"task_{self._next_id}",
                apps={process},
                windows=[obs],
                first_seen=timestamp,
                last_seen=timestamp,
            )
            self._next_id += 1
            self._tasks.append(task)

        # Check for stuck state
        self._check_stuck()
        self._last_observation = obs

    def _find_matching_task(self, process: str, title: str, ts: datetime) -> TaskNode | None:
        project = self._extract_project(title)
        for task in reversed(self._tasks):
            if task.state == "completed":
                continue
            # Check temporal proximity to the last observation in this task
            if task.last_seen:
                last = datetime.fromisoformat(task.last_seen)
                if ts - last > self._boundary and task.state != "paused":
                    continue
            # Check semantic match: same process or same project substring
            if process in task.apps:
                return task
            if project:
                task_project = self._extract_project_from_task(task)
                if task_project and project.lower() == task_project.lower():
                    return task
            # If paused task matches process, resume it
            if task.state == "paused" and process in task.apps:
                return task
        return None

    def _extract_project(self, title: str) -> str:
        """Extract project/file name from window title."""
        parts = title.split(" - ")
        if len(parts) >= 2:
            return parts[-1].strip()
        return ""

    def _extract_project_from_task(self, task: TaskNode) -> str:
        for w in reversed(task.windows):
            proj = self._extract_project(w["title"])
            if proj:
                return proj
        return ""

    def _check_stuck(self) -> None:
        for task in self._tasks:
            if task.state != "active":
                continue
            duration = task.duration_minutes()
            if duration >= self._stuck_minutes:
                rate = task.switch_count / max(1, duration)
                if rate >= self._stuck_switch_rate:
                    task.state = "stuck"

    def get_tasks(self) -> list[TaskNode]:
        return list(self._tasks)

    def get_incomplete_tasks(self) -> list[TaskNode]:
        return [t for t in self._tasks if t.state in ("active", "paused", "stuck")]

    def summarize(self) -> str:
        lines = []
        for t in self._tasks:
            apps = ", ".join(sorted(t.apps))
            proj = self._extract_project_from_task(t)
            dur = t.duration_minutes()
            label = f"{proj} ({apps})" if proj else apps
            lines.append(f"- [{t.state}] {label}: {dur:.0f}min, {len(t.windows)} observations")
        return "\n".join(lines) if lines else "No tasks recorded."

    def serialize(self) -> dict:
        tasks = []
        for t in self._tasks:
            tasks.append({
                "id": t.id, "apps": sorted(t.apps), "windows": t.windows,
                "first_seen": t.first_seen, "last_seen": t.last_seen,
                "state": t.state, "switch_count": t.switch_count,
            })
        return {"tasks": tasks, "next_id": self._next_id}

    @classmethod
    def deserialize(cls, data: dict) -> TaskGraph:
        tg = cls()
        tg._next_id = data.get("next_id", 0)
        for td in data.get("tasks", []):
            task = TaskNode(
                id=td["id"], apps=set(td["apps"]), windows=td["windows"],
                first_seen=td["first_seen"], last_seen=td["last_seen"],
                state=td["state"], switch_count=td["switch_count"],
            )
            tg._tasks.append(task)
        return tg
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_intelligence/test_task_graph.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/homie_core/intelligence/ tests/unit/test_intelligence/
git commit -m "feat: add task graph inference algorithm"
```

---

### Task 2: Proactive Retrieval Engine

**Files:**
- Create: `src/homie_core/intelligence/proactive_retrieval.py`
- Test: `tests/unit/test_intelligence/test_proactive_retrieval.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_intelligence/test_proactive_retrieval.py`:

```python
from unittest.mock import MagicMock

from homie_core.intelligence.proactive_retrieval import ProactiveRetrieval


def test_stage_context_on_change():
    sm = MagicMock()
    sm.get_facts.return_value = [{"fact": "User prefers dark mode", "confidence": 0.8}]
    em = MagicMock()
    em.recall.return_value = [{"summary": "Worked on engine.py yesterday"}]

    pr = ProactiveRetrieval(semantic_memory=sm, episodic_memory=em)
    pr.on_context_change(process="Code.exe", title="engine.py - Homie")
    staged = pr.get_staged_context()

    assert len(staged["facts"]) == 1
    assert len(staged["episodes"]) == 1


def test_staged_context_is_cleared_on_consume():
    sm = MagicMock()
    sm.get_facts.return_value = [{"fact": "test"}]
    em = MagicMock()
    em.recall.return_value = []

    pr = ProactiveRetrieval(semantic_memory=sm, episodic_memory=em)
    pr.on_context_change(process="Code.exe", title="engine.py")
    pr.consume_staged_context()
    staged = pr.get_staged_context()
    assert staged["facts"] == []
    assert staged["episodes"] == []


def test_no_query_when_same_context():
    sm = MagicMock()
    sm.get_facts.return_value = []
    em = MagicMock()
    em.recall.return_value = []

    pr = ProactiveRetrieval(semantic_memory=sm, episodic_memory=em)
    pr.on_context_change(process="Code.exe", title="engine.py")
    pr.on_context_change(process="Code.exe", title="engine.py")
    # Only called once since context didn't change
    assert sm.get_facts.call_count == 1


def test_builds_query_from_title():
    sm = MagicMock()
    sm.get_facts.return_value = []
    em = MagicMock()
    em.recall.return_value = []

    pr = ProactiveRetrieval(semantic_memory=sm, episodic_memory=em)
    pr.on_context_change(process="Code.exe", title="config.py - Homie")
    # Should query with title info
    em.recall.assert_called_once()
    query_arg = em.recall.call_args[0][0]
    assert "config.py" in query_arg or "Homie" in query_arg
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_intelligence/test_proactive_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/homie_core/intelligence/proactive_retrieval.py`:

```python
from __future__ import annotations

import threading
from typing import Any, Optional

from homie_core.memory.episodic import EpisodicMemory
from homie_core.memory.semantic import SemanticMemory


class ProactiveRetrieval:
    """Silently pre-fetches relevant context on every context change.

    When the user switches windows/apps, this queries semantic and episodic
    memory and stages the results. If the user triggers Homie, the context
    is immediately available. If not, it's overwritten on the next change.
    """

    def __init__(
        self,
        semantic_memory: Optional[SemanticMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
    ):
        self._sm = semantic_memory
        self._em = episodic_memory
        self._staged: dict[str, list] = {"facts": [], "episodes": []}
        self._last_key: str = ""
        self._lock = threading.Lock()

    def on_context_change(self, process: str, title: str) -> None:
        key = f"{process}::{title}"
        if key == self._last_key:
            return
        self._last_key = key

        query = self._build_query(process, title)
        facts: list[dict] = []
        episodes: list[dict] = []

        if self._sm:
            try:
                facts = self._sm.get_facts(min_confidence=0.5)
            except Exception:
                facts = []

        if self._em:
            try:
                episodes = self._em.recall(query, n=3)
            except Exception:
                episodes = []

        with self._lock:
            self._staged = {"facts": facts, "episodes": episodes}

    def get_staged_context(self) -> dict[str, list]:
        with self._lock:
            return dict(self._staged)

    def consume_staged_context(self) -> dict[str, list]:
        with self._lock:
            result = dict(self._staged)
            self._staged = {"facts": [], "episodes": []}
            return result

    def _build_query(self, process: str, title: str) -> str:
        parts = title.split(" - ")
        if len(parts) >= 2:
            return f"{parts[0].strip()} {parts[-1].strip()}"
        return title
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_intelligence/test_proactive_retrieval.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/homie_core/intelligence/proactive_retrieval.py tests/unit/test_intelligence/test_proactive_retrieval.py
git commit -m "feat: add proactive retrieval engine"
```

---

### Task 3: Adaptive Interruption Model

**Files:**
- Create: `src/homie_core/intelligence/interruption_model.py`
- Test: `tests/unit/test_intelligence/test_interruption_model.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_intelligence/test_interruption_model.py`:

```python
from homie_core.intelligence.interruption_model import InterruptionModel


def test_initial_prediction_is_conservative():
    model = InterruptionModel()
    prob = model.predict(
        minutes_in_task=5,
        switch_freq_10min=2,
        minutes_since_interaction=10,
        category="health",
    )
    # With no training data, should return base rate (0.5)
    assert 0.3 <= prob <= 0.7


def test_should_interrupt_above_threshold():
    model = InterruptionModel(threshold=0.7)
    # Train with many acceptances
    for _ in range(20):
        model.record_feedback(
            accepted=True,
            minutes_in_task=30,
            switch_freq_10min=5,
            minutes_since_interaction=60,
            category="health",
        )
    prob = model.predict(
        minutes_in_task=30,
        switch_freq_10min=5,
        minutes_since_interaction=60,
        category="health",
    )
    assert model.should_interrupt(
        minutes_in_task=30,
        switch_freq_10min=5,
        minutes_since_interaction=60,
        category="health",
    )


def test_should_not_interrupt_after_dismissals():
    model = InterruptionModel(threshold=0.7)
    for _ in range(20):
        model.record_feedback(
            accepted=False,
            minutes_in_task=5,
            switch_freq_10min=1,
            minutes_since_interaction=2,
            category="health",
        )
    assert not model.should_interrupt(
        minutes_in_task=5,
        switch_freq_10min=1,
        minutes_since_interaction=2,
        category="health",
    )


def test_serialize_and_restore():
    model = InterruptionModel()
    for _ in range(5):
        model.record_feedback(True, 10, 3, 20, "health")
    data = model.serialize()
    model2 = InterruptionModel.deserialize(data)
    p1 = model.predict(10, 3, 20, "health")
    p2 = model2.predict(10, 3, 20, "health")
    assert abs(p1 - p2) < 0.01


def test_category_encoding():
    model = InterruptionModel()
    # Different categories should produce different predictions after training
    for _ in range(15):
        model.record_feedback(True, 10, 3, 20, "health")
        model.record_feedback(False, 10, 3, 20, "calendar")
    p_health = model.predict(10, 3, 20, "health")
    p_calendar = model.predict(10, 3, 20, "calendar")
    assert p_health > p_calendar
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_intelligence/test_interruption_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/homie_core/intelligence/interruption_model.py`:

```python
from __future__ import annotations

import math
from typing import Any


# Category -> index mapping for feature encoding
_CATEGORIES = ["health", "calendar", "suggestion", "task", "reminder", "other"]


def _category_features(category: str) -> list[float]:
    """One-hot encode the category."""
    vec = [0.0] * len(_CATEGORIES)
    idx = _CATEGORIES.index(category) if category in _CATEGORIES else len(_CATEGORIES) - 1
    vec[idx] = 1.0
    return vec


def _build_features(minutes_in_task: float, switch_freq_10min: float,
                    minutes_since_interaction: float, category: str) -> list[float]:
    """Build normalized feature vector."""
    return [
        min(minutes_in_task / 120.0, 1.0),
        min(switch_freq_10min / 20.0, 1.0),
        min(minutes_since_interaction / 120.0, 1.0),
    ] + _category_features(category)


def _sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class InterruptionModel:
    """Logistic regression model that predicts whether the user will accept
    an interruption, trained online via SGD.

    Features: minutes_in_task, switch_frequency_10min,
    minutes_since_interaction, category (one-hot).

    No external dependencies — pure Python math.
    """

    def __init__(self, threshold: float = 0.7, learning_rate: float = 0.1):
        n_features = 3 + len(_CATEGORIES)
        self._weights = [0.0] * n_features
        self._bias = 0.0
        self._threshold = threshold
        self._lr = learning_rate
        self._n_samples = 0

    def predict(self, minutes_in_task: float, switch_freq_10min: float,
                minutes_since_interaction: float, category: str) -> float:
        features = _build_features(minutes_in_task, switch_freq_10min,
                                   minutes_since_interaction, category)
        z = self._bias + sum(w * x for w, x in zip(self._weights, features))
        return _sigmoid(z)

    def should_interrupt(self, minutes_in_task: float, switch_freq_10min: float,
                         minutes_since_interaction: float, category: str) -> bool:
        return self.predict(minutes_in_task, switch_freq_10min,
                           minutes_since_interaction, category) >= self._threshold

    def record_feedback(self, accepted: bool, minutes_in_task: float,
                        switch_freq_10min: float, minutes_since_interaction: float,
                        category: str) -> None:
        """Online SGD update."""
        features = _build_features(minutes_in_task, switch_freq_10min,
                                   minutes_since_interaction, category)
        y = 1.0 if accepted else 0.0
        p = self.predict(minutes_in_task, switch_freq_10min,
                         minutes_since_interaction, category)
        error = y - p

        for i in range(len(self._weights)):
            self._weights[i] += self._lr * error * features[i]
        self._bias += self._lr * error
        self._n_samples += 1

    def serialize(self) -> dict:
        return {
            "weights": list(self._weights),
            "bias": self._bias,
            "threshold": self._threshold,
            "lr": self._lr,
            "n_samples": self._n_samples,
        }

    @classmethod
    def deserialize(cls, data: dict) -> InterruptionModel:
        model = cls(threshold=data["threshold"], learning_rate=data["lr"])
        model._weights = list(data["weights"])
        model._bias = data["bias"]
        model._n_samples = data["n_samples"]
        return model
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_intelligence/test_interruption_model.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/homie_core/intelligence/interruption_model.py tests/unit/test_intelligence/test_interruption_model.py
git commit -m "feat: add adaptive interruption model (logistic regression)"
```

---

### Task 4: Session Tracker & Cross-Session Continuity

**Files:**
- Create: `src/homie_core/intelligence/session_tracker.py`
- Test: `tests/unit/test_intelligence/test_session_tracker.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_intelligence/test_session_tracker.py`:

```python
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

from homie_core.intelligence.session_tracker import SessionTracker
from homie_core.intelligence.task_graph import TaskGraph


def _ts(minutes_ago: int) -> str:
    dt = datetime(2026, 3, 10, 18, 0, 0, tzinfo=timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.isoformat()


def test_save_and_load_session(tmp_path):
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(10))
    tg.observe(process="chrome.exe", title="Stack Overflow", timestamp=_ts(5))

    tracker = SessionTracker(storage_dir=tmp_path)
    tracker.save_session(tg, apps_used={"Code.exe": 1200.0, "chrome.exe": 300.0})

    loaded = tracker.load_last_session()
    assert loaded is not None
    assert loaded["task_graph"] is not None
    assert "Code.exe" in loaded["apps_used"]


def test_generate_resumption_summary(tmp_path):
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(10))

    tracker = SessionTracker(storage_dir=tmp_path)
    tracker.save_session(tg, apps_used={"Code.exe": 3600.0})

    summary = tracker.get_resumption_summary()
    assert summary is not None
    assert "Code.exe" in summary


def test_no_session_returns_none(tmp_path):
    tracker = SessionTracker(storage_dir=tmp_path)
    assert tracker.load_last_session() is None
    assert tracker.get_resumption_summary() is None


def test_generate_end_of_day_digest(tmp_path):
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(120))
    tg.observe(process="chrome.exe", title="Google", timestamp=_ts(60))
    tg.observe(process="Code.exe", title="config.py - Homie", timestamp=_ts(30))

    tracker = SessionTracker(storage_dir=tmp_path)
    digest = tracker.generate_digest(
        tg,
        apps_used={"Code.exe": 5400.0, "chrome.exe": 1800.0},
        switch_count=15,
    )
    assert "Code.exe" in digest
    assert isinstance(digest, str)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_intelligence/test_session_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/homie_core/intelligence/session_tracker.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from homie_core.intelligence.task_graph import TaskGraph
from homie_core.utils import utc_now


class SessionTracker:
    """Persists task graph and session metadata across sessions.

    Saves to a JSON file in the storage directory. Provides resumption
    summaries and end-of-day digests.
    """

    def __init__(self, storage_dir: Path | str):
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._session_file = self._dir / "last_session.json"

    def save_session(self, task_graph: TaskGraph,
                     apps_used: dict[str, float] | None = None) -> None:
        data = {
            "task_graph": task_graph.serialize(),
            "apps_used": apps_used or {},
            "saved_at": utc_now().isoformat(),
        }
        self._session_file.write_text(json.dumps(data, indent=2))

    def load_last_session(self) -> Optional[dict[str, Any]]:
        if not self._session_file.exists():
            return None
        try:
            data = json.loads(self._session_file.read_text())
            data["task_graph"] = TaskGraph.deserialize(data["task_graph"])
            return data
        except (json.JSONDecodeError, KeyError):
            return None

    def get_resumption_summary(self) -> Optional[str]:
        session = self.load_last_session()
        if not session:
            return None

        tg: TaskGraph = session["task_graph"]
        apps = session.get("apps_used", {})
        incomplete = tg.get_incomplete_tasks()

        lines = []
        if incomplete:
            lines.append("You left off with these tasks:")
            for t in incomplete:
                proj = tg._extract_project_from_task(t)
                app_list = ", ".join(sorted(t.apps))
                label = proj if proj else app_list
                lines.append(f"  - [{t.state}] {label} ({len(t.windows)} activities)")

        if apps:
            top = sorted(apps.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("\nApps used yesterday:")
            for app, secs in top:
                mins = secs / 60
                lines.append(f"  - {app}: {mins:.0f} min")

        return "\n".join(lines) if lines else None

    def generate_digest(self, task_graph: TaskGraph,
                        apps_used: dict[str, float] | None = None,
                        switch_count: int = 0) -> str:
        apps = apps_used or {}
        total_hours = sum(apps.values()) / 3600 if apps else 0

        lines = ["End-of-day summary:"]
        lines.append(f"  Total active time: {total_hours:.1f} hours")
        lines.append(f"  Context switches: {switch_count}")

        if apps:
            lines.append("  Top apps:")
            top = sorted(apps.items(), key=lambda x: x[1], reverse=True)[:5]
            for app, secs in top:
                lines.append(f"    - {app}: {secs / 60:.0f} min")

        tasks = task_graph.get_tasks()
        if tasks:
            lines.append(f"  Tasks tracked: {len(tasks)}")
            stuck = [t for t in tasks if t.state == "stuck"]
            if stuck:
                lines.append(f"  Stuck tasks: {len(stuck)}")

        incomplete = task_graph.get_incomplete_tasks()
        if incomplete:
            lines.append(f"  Incomplete tasks: {len(incomplete)}")

        return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_intelligence/test_session_tracker.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/homie_core/intelligence/session_tracker.py tests/unit/test_intelligence/test_session_tracker.py
git commit -m "feat: add session tracker with cross-session continuity"
```

---

### Task 5: Morning Briefing & End-of-Day Digest Generator

**Files:**
- Create: `src/homie_core/intelligence/briefing.py`
- Test: `tests/unit/test_intelligence/test_briefing.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_intelligence/test_briefing.py`:

```python
from datetime import datetime, timezone, timedelta
from pathlib import Path

from homie_core.intelligence.briefing import BriefingGenerator
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.intelligence.session_tracker import SessionTracker


def _ts(minutes_ago: int) -> str:
    dt = datetime(2026, 3, 10, 18, 0, 0, tzinfo=timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.isoformat()


def test_morning_briefing_with_previous_session(tmp_path):
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(120))
    tg.observe(process="chrome.exe", title="Google", timestamp=_ts(60))

    tracker = SessionTracker(storage_dir=tmp_path)
    tracker.save_session(tg, apps_used={"Code.exe": 5400.0, "chrome.exe": 1200.0})

    gen = BriefingGenerator(session_tracker=tracker, user_name="Master")
    briefing = gen.morning_briefing()

    assert "Master" in briefing or "Good" in briefing
    assert "Code.exe" in briefing


def test_morning_briefing_without_previous_session(tmp_path):
    tracker = SessionTracker(storage_dir=tmp_path)
    gen = BriefingGenerator(session_tracker=tracker, user_name="Master")
    briefing = gen.morning_briefing()
    assert "Master" in briefing or "Good" in briefing
    assert isinstance(briefing, str)


def test_end_of_day_digest(tmp_path):
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(120))

    tracker = SessionTracker(storage_dir=tmp_path)
    gen = BriefingGenerator(session_tracker=tracker, user_name="Master")
    digest = gen.end_of_day_digest(
        task_graph=tg,
        apps_used={"Code.exe": 7200.0},
        switch_count=10,
    )
    assert "Code.exe" in digest
    assert isinstance(digest, str)


def test_greeting_varies_by_hour():
    from homie_core.intelligence.briefing import _greeting
    assert "morning" in _greeting(8).lower()
    assert "afternoon" in _greeting(14).lower()
    assert "evening" in _greeting(20).lower()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_intelligence/test_briefing.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/homie_core/intelligence/briefing.py`:

```python
from __future__ import annotations

from typing import Optional

from homie_core.intelligence.session_tracker import SessionTracker
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.utils import utc_now


def _greeting(hour: int) -> str:
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"


class BriefingGenerator:
    """Generates morning briefings and end-of-day digests."""

    def __init__(self, session_tracker: SessionTracker,
                 user_name: str = ""):
        self._tracker = session_tracker
        self._user_name = user_name

    def morning_briefing(self) -> str:
        now = utc_now()
        name_part = f", {self._user_name}" if self._user_name else ""
        greeting = f"{_greeting(now.hour)}{name_part}!"

        lines = [greeting, ""]

        resumption = self._tracker.get_resumption_summary()
        if resumption:
            lines.append(resumption)
        else:
            lines.append("No previous session found. Starting fresh!")

        lines.append("")
        lines.append(f"Today is {now.strftime('%A, %B %d')}. Ready when you are.")

        return "\n".join(lines)

    def end_of_day_digest(self, task_graph: TaskGraph,
                          apps_used: dict[str, float] | None = None,
                          switch_count: int = 0) -> str:
        digest = self._tracker.generate_digest(
            task_graph, apps_used=apps_used, switch_count=switch_count,
        )

        # Save session for tomorrow's morning briefing
        self._tracker.save_session(task_graph, apps_used=apps_used)

        name_part = f", {self._user_name}" if self._user_name else ""
        return f"Wrapping up{name_part}.\n\n{digest}"
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_intelligence/test_briefing.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/homie_core/intelligence/briefing.py tests/unit/test_intelligence/test_briefing.py
git commit -m "feat: add morning briefing and end-of-day digest"
```

---

### Task 6: Enterprise Policy System

**Files:**
- Create: `src/homie_core/enterprise.py`
- Test: `tests/unit/test_enterprise.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_enterprise.py`:

```python
import yaml
from pathlib import Path

from homie_core.enterprise import EnterprisePolicy, load_enterprise_policy, apply_policy
from homie_core.config import HomieConfig


def test_load_policy_from_file(tmp_path):
    policy_data = {
        "org_name": "Acme Corp",
        "model_policy": {
            "allowed_backends": ["cloud"],
            "endpoint": "https://models.acme.internal/v1",
            "api_key_env": "ACME_AI_KEY",
            "allowed_models": ["llama-3.1-70b"],
        },
        "plugins": {
            "disabled": ["browser_plugin"],
            "required": ["git_plugin"],
        },
        "privacy": {
            "data_retention_days": 90,
            "disable_observers": ["browsing", "social"],
            "audit_log": True,
        },
    }
    policy_file = tmp_path / "homie.enterprise.yaml"
    policy_file.write_text(yaml.dump(policy_data))

    policy = load_enterprise_policy(tmp_path)
    assert policy is not None
    assert policy.org_name == "Acme Corp"
    assert "cloud" in policy.model_policy.allowed_backends
    assert policy.privacy.audit_log is True


def test_no_policy_returns_none(tmp_path):
    policy = load_enterprise_policy(tmp_path)
    assert policy is None


def test_apply_policy_overrides_config(tmp_path):
    policy_data = {
        "org_name": "TestCorp",
        "model_policy": {
            "allowed_backends": ["cloud"],
            "endpoint": "https://internal.api/v1",
        },
        "privacy": {
            "data_retention_days": 90,
            "disable_observers": ["browsing"],
        },
    }
    policy_file = tmp_path / "homie.enterprise.yaml"
    policy_file.write_text(yaml.dump(policy_data))
    policy = load_enterprise_policy(tmp_path)

    cfg = HomieConfig()
    cfg.llm.backend = "gguf"
    cfg.privacy.data_retention_days = 30

    applied = apply_policy(cfg, policy)
    # Enterprise overrides personal config
    assert applied.llm.api_base_url == "https://internal.api/v1"
    assert applied.privacy.data_retention_days == 90
    assert applied.privacy.observers["browsing"] is False


def test_plugin_restrictions():
    from homie_core.enterprise import EnterprisePolicy, PluginPolicy

    policy = EnterprisePolicy(
        org_name="Test",
        plugins=PluginPolicy(disabled=["browser_plugin"], required=["git_plugin"]),
    )
    assert policy.is_plugin_disabled("browser_plugin")
    assert not policy.is_plugin_disabled("git_plugin")
    assert policy.is_plugin_required("git_plugin")


def test_model_allowed():
    from homie_core.enterprise import EnterprisePolicy, ModelPolicy

    policy = EnterprisePolicy(
        org_name="Test",
        model_policy=ModelPolicy(
            allowed_backends=["cloud"],
            allowed_models=["llama-3.1-70b"],
        ),
    )
    assert policy.is_backend_allowed("cloud")
    assert not policy.is_backend_allowed("gguf")
    assert policy.is_model_allowed("llama-3.1-70b")
    assert not policy.is_model_allowed("gpt-4o")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_enterprise.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/homie_core/enterprise.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from homie_core.config import HomieConfig


@dataclass
class ModelPolicy:
    allowed_backends: list[str] = field(default_factory=list)
    endpoint: str = ""
    api_key_env: str = ""
    allowed_models: list[str] = field(default_factory=list)


@dataclass
class PluginPolicy:
    disabled: list[str] = field(default_factory=list)
    required: list[str] = field(default_factory=list)


@dataclass
class PrivacyPolicy:
    data_retention_days: int = 0
    disable_observers: list[str] = field(default_factory=list)
    audit_log: bool = False


@dataclass
class EnterprisePolicy:
    org_name: str = ""
    model_policy: ModelPolicy = field(default_factory=ModelPolicy)
    plugins: PluginPolicy = field(default_factory=PluginPolicy)
    privacy: PrivacyPolicy = field(default_factory=PrivacyPolicy)
    policy_url: str = ""
    org_user_id: str = ""

    def is_plugin_disabled(self, name: str) -> bool:
        return name in self.plugins.disabled

    def is_plugin_required(self, name: str) -> bool:
        return name in self.plugins.required

    def is_backend_allowed(self, backend: str) -> bool:
        if not self.model_policy.allowed_backends:
            return True
        return backend in self.model_policy.allowed_backends

    def is_model_allowed(self, model: str) -> bool:
        if not self.model_policy.allowed_models:
            return True
        return model in self.model_policy.allowed_models


def load_enterprise_policy(config_dir: Path | str) -> Optional[EnterprisePolicy]:
    path = Path(config_dir) / "homie.enterprise.yaml"
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return None

    mp = data.get("model_policy", {})
    pp = data.get("plugins", {})
    pr = data.get("privacy", {})

    return EnterprisePolicy(
        org_name=data.get("org_name", ""),
        model_policy=ModelPolicy(
            allowed_backends=mp.get("allowed_backends", []),
            endpoint=mp.get("endpoint", ""),
            api_key_env=mp.get("api_key_env", ""),
            allowed_models=mp.get("allowed_models", []),
        ),
        plugins=PluginPolicy(
            disabled=pp.get("disabled", []),
            required=pp.get("required", []),
        ),
        privacy=PrivacyPolicy(
            data_retention_days=pr.get("data_retention_days", 0),
            disable_observers=pr.get("disable_observers", []),
            audit_log=pr.get("audit_log", False),
        ),
        policy_url=data.get("policy_url", ""),
    )


def apply_policy(cfg: HomieConfig, policy: EnterprisePolicy) -> HomieConfig:
    """Merge enterprise policy over personal config. Enterprise wins."""
    if policy.model_policy.endpoint:
        cfg.llm.api_base_url = policy.model_policy.endpoint

    if policy.model_policy.api_key_env:
        key = os.environ.get(policy.model_policy.api_key_env, "")
        if key:
            cfg.llm.api_key = key

    if policy.model_policy.allowed_backends:
        if cfg.llm.backend not in policy.model_policy.allowed_backends:
            cfg.llm.backend = policy.model_policy.allowed_backends[0]

    if policy.privacy.data_retention_days:
        cfg.privacy.data_retention_days = policy.privacy.data_retention_days

    for obs in policy.privacy.disable_observers:
        if obs in cfg.privacy.observers:
            cfg.privacy.observers[obs] = False

    return cfg
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_enterprise.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/homie_core/enterprise.py tests/unit/test_enterprise.py
git commit -m "feat: add enterprise policy system"
```

---

### Task 7: Audit Logger

**Files:**
- Create: `src/homie_core/intelligence/audit_log.py`
- Test: `tests/unit/test_intelligence/test_audit_log.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_intelligence/test_audit_log.py`:

```python
import json
from pathlib import Path

from homie_core.intelligence.audit_log import AuditLogger


def test_log_query(tmp_path):
    logger = AuditLogger(log_dir=tmp_path)
    logger.log_query(prompt="What is the weather?", response="It's sunny.", model="gpt-4o")

    log_file = list(tmp_path.glob("audit_*.jsonl"))
    assert len(log_file) == 1

    lines = log_file[0].read_text().strip().split("\n")
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["prompt"] == "What is the weather?"
    assert entry["response"] == "It's sunny."
    assert entry["model"] == "gpt-4o"
    assert "timestamp" in entry


def test_multiple_entries_append(tmp_path):
    logger = AuditLogger(log_dir=tmp_path)
    logger.log_query("q1", "r1", "m1")
    logger.log_query("q2", "r2", "m2")

    log_file = list(tmp_path.glob("audit_*.jsonl"))
    lines = log_file[0].read_text().strip().split("\n")
    assert len(lines) == 2


def test_disabled_logger_does_nothing(tmp_path):
    logger = AuditLogger(log_dir=tmp_path, enabled=False)
    logger.log_query("q1", "r1", "m1")

    log_files = list(tmp_path.glob("audit_*.jsonl"))
    assert len(log_files) == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_intelligence/test_audit_log.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/homie_core/intelligence/audit_log.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from homie_core.utils import utc_now


class AuditLogger:
    """Append-only JSON Lines log of all LLM queries and responses.

    Used by enterprise policy when audit_log is enabled.
    Each day gets its own log file: audit_YYYY-MM-DD.jsonl
    """

    def __init__(self, log_dir: Path | str, enabled: bool = True):
        self._dir = Path(log_dir)
        self._enabled = enabled
        if enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def log_query(self, prompt: str, response: str, model: str) -> None:
        if not self._enabled:
            return
        now = utc_now()
        filename = f"audit_{now.strftime('%Y-%m-%d')}.jsonl"
        entry = {
            "timestamp": now.isoformat(),
            "prompt": prompt,
            "response": response,
            "model": model,
        }
        with open(self._dir / filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_intelligence/test_audit_log.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/homie_core/intelligence/audit_log.py tests/unit/test_intelligence/test_audit_log.py
git commit -m "feat: add audit logger for enterprise compliance"
```

---

### Task 8: Global Hotkey Handler (Alt+8)

**Files:**
- Create: `src/homie_app/hotkey.py`
- Test: `tests/unit/test_hotkey.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_hotkey.py`:

```python
from unittest.mock import MagicMock, patch

from homie_app.hotkey import HotkeyListener


def test_hotkey_listener_init():
    callback = MagicMock()
    listener = HotkeyListener(hotkey="alt+8", callback=callback)
    assert listener._hotkey == "alt+8"
    assert listener._callback is callback
    assert not listener._running


def test_hotkey_listener_start_stop():
    callback = MagicMock()
    listener = HotkeyListener(hotkey="alt+8", callback=callback)
    # Mock pynput to avoid actually registering hotkeys
    with patch("homie_app.hotkey.keyboard") as mock_kb:
        mock_listener = MagicMock()
        mock_kb.GlobalHotKeys.return_value = mock_listener
        listener.start()
        assert listener._running
        mock_listener.start.assert_called_once()

        listener.stop()
        assert not listener._running
        mock_listener.stop.assert_called_once()


def test_hotkey_triggers_callback():
    callback = MagicMock()
    listener = HotkeyListener(hotkey="alt+8", callback=callback)
    # Simulate the hotkey being pressed by calling the internal handler
    listener._on_activate()
    callback.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_hotkey.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/homie_app/hotkey.py`:

```python
from __future__ import annotations

import threading
from typing import Callable, Optional

try:
    from pynput import keyboard
except ImportError:
    keyboard = None  # type: ignore[assignment]


# Map friendly hotkey strings to pynput format
_HOTKEY_MAP = {
    "alt+8": "<alt>+8",
    "alt+h": "<alt>+h",
    "ctrl+space": "<ctrl>+<space>",
    "f9": "<f9>",
}


class HotkeyListener:
    """Registers a global hotkey and calls a callback when pressed.

    Uses pynput for cross-platform hotkey listening.
    Falls back gracefully if pynput is not installed.
    """

    def __init__(self, hotkey: str = "alt+8", callback: Optional[Callable] = None):
        self._hotkey = hotkey
        self._callback = callback
        self._running = False
        self._listener = None
        self._thread: Optional[threading.Thread] = None

    def _on_activate(self) -> None:
        if self._callback:
            self._callback()

    def start(self) -> None:
        if keyboard is None:
            return
        pynput_hotkey = _HOTKEY_MAP.get(self._hotkey, self._hotkey)
        self._listener = keyboard.GlobalHotKeys({
            pynput_hotkey: self._on_activate,
        })
        self._listener.start()
        self._running = True

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_hotkey.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/homie_app/hotkey.py tests/unit/test_hotkey.py
git commit -m "feat: add global hotkey listener (Alt+8)"
```

---

### Task 9: Overlay Popup UI

**Files:**
- Create: `src/homie_app/overlay.py`
- Test: `tests/unit/test_overlay.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_overlay.py`:

```python
from unittest.mock import MagicMock, patch

from homie_app.overlay import OverlayPopup


def test_overlay_init():
    callback = MagicMock()
    overlay = OverlayPopup(on_submit=callback)
    assert overlay._on_submit is callback
    assert not overlay._visible


def test_overlay_submit_calls_callback():
    callback = MagicMock(return_value="Hello back!")
    overlay = OverlayPopup(on_submit=callback)
    # Simulate text submission without actual tkinter
    result = overlay._handle_submit("Hello")
    callback.assert_called_once_with("Hello")
    assert result == "Hello back!"


def test_overlay_toggle():
    overlay = OverlayPopup(on_submit=MagicMock())
    assert not overlay._visible
    overlay._visible = True
    assert overlay._visible
    overlay._visible = False
    assert not overlay._visible


def test_overlay_empty_submit_ignored():
    callback = MagicMock()
    overlay = OverlayPopup(on_submit=callback)
    result = overlay._handle_submit("")
    callback.assert_not_called()
    assert result is None

    result = overlay._handle_submit("   ")
    callback.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/homie_app/overlay.py`:

```python
from __future__ import annotations

import threading
from typing import Callable, Optional


class OverlayPopup:
    """Lightweight overlay popup for quick Homie interactions.

    Uses tkinter for the UI (stdlib, no extra deps). Shows a frameless
    window with a text input and response area. Dismissed with Escape
    or click-outside.

    The actual tkinter window is created lazily on first show() call
    to avoid import issues in headless environments.
    """

    def __init__(self, on_submit: Optional[Callable[[str], str]] = None):
        self._on_submit = on_submit
        self._visible = False
        self._root = None
        self._thread: Optional[threading.Thread] = None

    def _handle_submit(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        if self._on_submit:
            return self._on_submit(text.strip())
        return None

    def show(self) -> None:
        if self._visible:
            return
        self._visible = True
        self._thread = threading.Thread(target=self._create_window, daemon=True)
        self._thread.start()

    def hide(self) -> None:
        self._visible = False
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass
            self._root = None

    def toggle(self) -> None:
        if self._visible:
            self.hide()
        else:
            self.show()

    def _create_window(self) -> None:
        try:
            import tkinter as tk
        except ImportError:
            self._visible = False
            return

        self._root = tk.Tk()
        root = self._root
        root.title("Homie")
        root.overrideredirect(True)
        root.attributes("-topmost", True)

        # Center on screen
        width, height = 600, 200
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 3
        root.geometry(f"{width}x{height}+{x}+{y}")

        root.configure(bg="#1e1e1e")

        # Input field
        input_frame = tk.Frame(root, bg="#1e1e1e")
        input_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        label = tk.Label(input_frame, text="Homie>", fg="#61afef", bg="#1e1e1e",
                         font=("Consolas", 12))
        label.pack(side=tk.LEFT)

        entry = tk.Entry(input_frame, bg="#2d2d2d", fg="white", insertbackground="white",
                         font=("Consolas", 12), relief=tk.FLAT, bd=5)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        entry.focus_set()

        # Response area
        response = tk.Label(root, text="", fg="#abb2bf", bg="#1e1e1e",
                            font=("Consolas", 11), wraplength=560, justify=tk.LEFT,
                            anchor=tk.NW)
        response.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

        def on_enter(event=None):
            text = entry.get()
            result = self._handle_submit(text)
            if result:
                response.config(text=result)
                entry.delete(0, tk.END)

        def on_escape(event=None):
            self.hide()

        entry.bind("<Return>", on_enter)
        root.bind("<Escape>", on_escape)
        root.bind("<FocusOut>", lambda e: None)  # Could hide on focus out

        root.mainloop()
        self._visible = False
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_overlay.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/homie_app/overlay.py tests/unit/test_overlay.py
git commit -m "feat: add overlay popup UI for quick interactions"
```

---

### Task 10: Event-Driven Observer Thread

**Files:**
- Create: `src/homie_core/intelligence/observer_loop.py`
- Test: `tests/unit/test_intelligence/test_observer_loop.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_intelligence/test_observer_loop.py`:

```python
import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from homie_core.intelligence.observer_loop import ObserverLoop
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.context.screen_monitor import WindowInfo
from homie_core.memory.working import WorkingMemory


def test_observer_loop_init():
    wm = WorkingMemory()
    tg = TaskGraph()
    loop = ObserverLoop(working_memory=wm, task_graph=tg)
    assert not loop.is_running


def test_observer_processes_window_event():
    wm = WorkingMemory()
    tg = TaskGraph()
    pr_callback = MagicMock()
    loop = ObserverLoop(working_memory=wm, task_graph=tg,
                        on_context_change=pr_callback)

    window = WindowInfo(
        title="engine.py - Homie",
        process_name="Code.exe",
        pid=1234,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    loop._handle_window_change(window)

    assert wm.get("active_window") == "engine.py - Homie"
    assert len(tg.get_tasks()) == 1
    pr_callback.assert_called_once_with("Code.exe", "engine.py - Homie")


def test_observer_debounces_same_window():
    wm = WorkingMemory()
    tg = TaskGraph()
    pr_callback = MagicMock()
    loop = ObserverLoop(working_memory=wm, task_graph=tg,
                        on_context_change=pr_callback)

    window = WindowInfo(title="engine.py", process_name="Code.exe", pid=1,
                        timestamp=datetime.now(timezone.utc).isoformat())
    loop._handle_window_change(window)
    loop._handle_window_change(window)

    # Should only process once
    assert pr_callback.call_count == 1


def test_observer_cpu_budget():
    loop = ObserverLoop(working_memory=WorkingMemory(), task_graph=TaskGraph(),
                        cpu_budget=0.05)
    assert loop._cpu_budget == 0.05
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_intelligence/test_observer_loop.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/homie_core/intelligence/observer_loop.py`:

```python
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from homie_core.context.screen_monitor import ScreenMonitor, WindowInfo
from homie_core.context.app_tracker import AppTracker
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.memory.working import WorkingMemory
from homie_core.utils import utc_now


class ObserverLoop:
    """Event-driven observer thread that watches OS state changes.

    Uses a low-frequency poll (1s default) with debouncing to approximate
    event-driven behavior. Only processes actual changes — if the active
    window hasn't changed, no work is done.

    Performance budget: stays under cpu_budget (default 5%) averaged
    over 60 seconds. If exceeded, increases poll interval.
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        task_graph: TaskGraph,
        app_tracker: Optional[AppTracker] = None,
        screen_monitor: Optional[ScreenMonitor] = None,
        on_context_change: Optional[Callable[[str, str], None]] = None,
        poll_interval: float = 1.0,
        cpu_budget: float = 0.05,
    ):
        self._wm = working_memory
        self._tg = task_graph
        self._apps = app_tracker or AppTracker()
        self._screen = screen_monitor or ScreenMonitor()
        self._on_context_change = on_context_change
        self._poll_interval = poll_interval
        self._cpu_budget = cpu_budget
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_window: Optional[WindowInfo] = None

    def _handle_window_change(self, window: WindowInfo) -> None:
        """Process a window focus change."""
        # Debounce: skip if same window
        if (self._last_window and
            self._last_window.title == window.title and
            self._last_window.process_name == window.process_name):
            return
        self._last_window = window

        # Update working memory
        self._wm.update("active_window", window.title)
        self._wm.update("active_process", window.process_name)

        # Feed app tracker
        self._apps.track(window.process_name)

        # Feed task graph
        self._tg.observe(
            process=window.process_name,
            title=window.title,
            timestamp=window.timestamp or utc_now().isoformat(),
        )

        # Notify proactive retrieval
        if self._on_context_change:
            self._on_context_change(window.process_name, window.title)

    def _loop(self) -> None:
        while self._running:
            start = time.monotonic()
            try:
                window = self._screen.get_active_window()
                if self._screen.has_changed(window):
                    self._handle_window_change(window)

                # Update deep work / switch stats
                self._wm.update("is_deep_work", self._apps.is_deep_work())
                self._wm.update("switch_count_30m", self._apps.get_switch_count(30))
            except Exception:
                pass

            elapsed = time.monotonic() - start
            # Adaptive throttle: if we're using too much CPU, slow down
            if elapsed > self._poll_interval * self._cpu_budget:
                self._poll_interval = min(self._poll_interval * 1.5, 10.0)

            time.sleep(self._poll_interval)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="observer")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def get_app_tracker(self) -> AppTracker:
        return self._apps
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_intelligence/test_observer_loop.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/homie_core/intelligence/observer_loop.py tests/unit/test_intelligence/test_observer_loop.py
git commit -m "feat: add event-driven observer loop with CPU budgeting"
```

---

### Task 11: Daemon — Tie Everything Together

**Files:**
- Create: `src/homie_app/daemon.py`
- Modify: `src/homie_app/cli.py:8-58` (add `homie daemon` command)
- Test: `tests/unit/test_daemon.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_daemon.py`:

```python
from unittest.mock import MagicMock, patch
from pathlib import Path

from homie_app.daemon import HomieDaemon


def test_daemon_init():
    daemon = HomieDaemon.__new__(HomieDaemon)
    daemon._config = MagicMock()
    daemon._running = False
    assert not daemon._running


def test_daemon_components_created(tmp_path):
    with patch("homie_app.daemon.load_config") as mock_cfg:
        cfg = MagicMock()
        cfg.storage.path = str(tmp_path)
        cfg.storage.log_dir = "logs"
        cfg.user_name = "Test"
        cfg.voice.enabled = False
        cfg.llm.backend = "cloud"
        cfg.llm.api_key = "test"
        cfg.llm.api_base_url = "http://localhost"
        mock_cfg.return_value = cfg

        with patch("homie_app.daemon.load_enterprise_policy", return_value=None):
            daemon = HomieDaemon(config_path=None)

    assert daemon._task_graph is not None
    assert daemon._observer is not None
    assert daemon._briefing is not None


def test_daemon_on_hotkey_triggers_overlay(tmp_path):
    with patch("homie_app.daemon.load_config") as mock_cfg:
        cfg = MagicMock()
        cfg.storage.path = str(tmp_path)
        cfg.storage.log_dir = "logs"
        cfg.user_name = "Test"
        cfg.voice.enabled = False
        cfg.llm.backend = "cloud"
        cfg.llm.api_key = "test"
        cfg.llm.api_base_url = "http://localhost"
        mock_cfg.return_value = cfg

        with patch("homie_app.daemon.load_enterprise_policy", return_value=None):
            daemon = HomieDaemon(config_path=None)

    daemon._overlay = MagicMock()
    daemon._on_hotkey()
    daemon._overlay.toggle.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_daemon.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/homie_app/daemon.py`:

```python
from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Optional

from homie_core.config import HomieConfig, load_config
from homie_core.enterprise import load_enterprise_policy, apply_policy
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.intelligence.proactive_retrieval import ProactiveRetrieval
from homie_core.intelligence.interruption_model import InterruptionModel
from homie_core.intelligence.session_tracker import SessionTracker
from homie_core.intelligence.briefing import BriefingGenerator
from homie_core.intelligence.observer_loop import ObserverLoop
from homie_core.intelligence.audit_log import AuditLogger
from homie_core.memory.working import WorkingMemory
from homie_app.hotkey import HotkeyListener
from homie_app.overlay import OverlayPopup


class HomieDaemon:
    """Always-active background daemon.

    Three threads:
    1. Main — hotkey listener + overlay UI
    2. Observer — watches OS events, feeds task graph
    3. Scheduler — morning briefing, end-of-day, memory consolidation
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config = load_config(config_path)
        self._running = False

        # Apply enterprise policy if present
        storage = Path(self._config.storage.path)
        policy = load_enterprise_policy(storage)
        if policy:
            self._config = apply_policy(self._config, policy)
            self._audit = AuditLogger(
                log_dir=storage / self._config.storage.log_dir,
                enabled=policy.privacy.audit_log,
            )
        else:
            self._audit = AuditLogger(
                log_dir=storage / self._config.storage.log_dir,
                enabled=False,
            )

        # Core components
        self._working_memory = WorkingMemory()
        self._task_graph = TaskGraph()
        self._interruption_model = InterruptionModel()

        # Session tracker
        self._session_tracker = SessionTracker(storage_dir=storage / "sessions")

        # Proactive retrieval (memories injected later when model loads)
        self._retrieval = ProactiveRetrieval()

        # Briefing generator
        self._briefing = BriefingGenerator(
            session_tracker=self._session_tracker,
            user_name=self._config.user_name,
        )

        # Observer loop
        self._observer = ObserverLoop(
            working_memory=self._working_memory,
            task_graph=self._task_graph,
            on_context_change=self._retrieval.on_context_change,
        )

        # UI components (created but not started)
        self._overlay = OverlayPopup(on_submit=self._on_user_query)
        self._hotkey = HotkeyListener(hotkey="alt+8", callback=self._on_hotkey)

        # Model engine (lazy loaded)
        self._engine = None

    def _on_hotkey(self) -> None:
        self._overlay.toggle()

    def _on_user_query(self, text: str) -> str:
        if not self._engine:
            self._load_engine()
        if not self._engine:
            return "Model not available. Run 'homie init' to set up."

        # Include staged context from proactive retrieval
        staged = self._retrieval.consume_staged_context()
        context_parts = []
        if staged.get("facts"):
            facts = [f.get("fact", str(f)) for f in staged["facts"][:5]]
            context_parts.append("Relevant facts:\n- " + "\n- ".join(facts))
        if staged.get("episodes"):
            eps = [e.get("summary", str(e)) for e in staged["episodes"][:3]]
            context_parts.append("Related past sessions:\n- " + "\n- ".join(eps))

        active = self._working_memory.get("active_window", "")
        if active:
            context_parts.append(f"User is currently in: {active}")

        context = "\n\n".join(context_parts)
        prompt = f"You are Homie, a helpful AI assistant. Be concise.\n\n{context}\n\nUser: {text}\nAssistant:"

        try:
            response = self._engine.generate(prompt, max_tokens=2048)
            self._audit.log_query(prompt=text, response=response,
                                  model=self._config.llm.model_path)
            return response
        except Exception as e:
            return f"Error: {e}"

    def _load_engine(self) -> None:
        try:
            from homie_core.model.engine import ModelEngine
            from homie_core.model.registry import ModelRegistry, ModelEntry

            engine = ModelEngine()
            registry = ModelRegistry(
                Path(self._config.storage.path) / self._config.storage.models_dir
            )
            registry.initialize()
            entry = registry.get_active()

            if not entry and self._config.llm.model_path:
                entry = ModelEntry(
                    name=self._config.llm.model_path,
                    path=self._config.llm.model_path,
                    format=self._config.llm.backend,
                    params="cloud" if self._config.llm.backend == "cloud" else "unknown",
                )

            if entry:
                kwargs = {}
                if entry.format == "cloud":
                    kwargs["api_key"] = self._config.llm.api_key
                    kwargs["base_url"] = self._config.llm.api_base_url or "https://api.openai.com/v1"
                else:
                    kwargs["n_ctx"] = self._config.llm.context_length
                    kwargs["n_gpu_layers"] = self._config.llm.gpu_layers
                engine.load(entry, **kwargs)
                self._engine = engine
        except Exception:
            self._engine = None

    def start(self) -> None:
        self._running = True
        print("Homie daemon starting...")

        # Show morning briefing if there's a previous session
        briefing = self._briefing.morning_briefing()
        print(f"\n{briefing}\n")

        # Start observer thread
        self._observer.start()
        print("  Observer: running")

        # Start hotkey listener
        self._hotkey.start()
        print("  Hotkey (Alt+8): active")

        print("\nHomie is running in the background. Press Alt+8 or say 'hey homie' to activate.")
        print("Press Ctrl+C to stop.\n")

        # Main thread waits
        try:
            signal.signal(signal.SIGINT, lambda *_: self.stop())
            while self._running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        print("\nHomie daemon stopping...")
        self._running = False

        # Save session for tomorrow
        apps = self._observer.get_app_tracker().get_usage()
        switch_count = self._observer.get_app_tracker().get_switch_count(minutes=1440)

        digest = self._briefing.end_of_day_digest(
            self._task_graph, apps_used=apps, switch_count=switch_count,
        )
        print(f"\n{digest}")

        self._observer.stop()
        self._hotkey.stop()
        self._overlay.hide()

        if self._engine:
            self._engine.unload()

        print("Goodbye!")
        sys.exit(0)
```

**Step 4: Add `homie daemon` command to CLI**

In `src/homie_app/cli.py`, add after the `chat` subparser (before `return parser`):

Add this subparser:
```python
    # homie daemon
    daemon_parser = subparsers.add_parser("daemon", help="Run as always-active background daemon")
    daemon_parser.add_argument("--config", type=str, help="Path to config file")
```

Add this handler function before `main()`:
```python
def cmd_daemon(args, config=None):
    from homie_app.daemon import HomieDaemon
    daemon = HomieDaemon(config_path=getattr(args, 'config', None))
    daemon.start()
```

In the `commands` dict inside `main()`, add:
```python
        "daemon": cmd_daemon,
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_daemon.py -v`
Expected: All 3 tests PASS

**Step 6: Run all tests**

Run: `pytest tests/ -q`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/homie_app/daemon.py src/homie_app/cli.py tests/unit/test_daemon.py
git commit -m "feat: add always-active daemon with hotkey, overlay, and observer"
```

---

### Task 12: Config Updates — Hotkey and Daemon Settings

**Files:**
- Modify: `src/homie_core/config.py:23-29` (VoiceConfig — change hotkey default)
- Test: verify existing config tests still pass

**Step 1: Update VoiceConfig hotkey default**

In `src/homie_core/config.py`, line 29, change:
```python
    hotkey: str = "F9"
```
to:
```python
    hotkey: str = "alt+8"
```

**Step 2: Run all tests**

Run: `pytest tests/ -q`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/homie_core/config.py
git commit -m "chore: change default hotkey to Alt+8"
```

---

### Task 13: Full Integration Test

**Step 1: Run complete test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (should be 217 + ~41 new = ~258 tests)

**Step 2: Verify imports**

Run:
```bash
python -c "from homie_core.intelligence.task_graph import TaskGraph; print('TaskGraph OK')"
python -c "from homie_core.intelligence.proactive_retrieval import ProactiveRetrieval; print('ProactiveRetrieval OK')"
python -c "from homie_core.intelligence.interruption_model import InterruptionModel; print('InterruptionModel OK')"
python -c "from homie_core.intelligence.session_tracker import SessionTracker; print('SessionTracker OK')"
python -c "from homie_core.intelligence.briefing import BriefingGenerator; print('BriefingGenerator OK')"
python -c "from homie_core.enterprise import load_enterprise_policy; print('Enterprise OK')"
python -c "from homie_core.intelligence.audit_log import AuditLogger; print('AuditLogger OK')"
python -c "from homie_core.intelligence.observer_loop import ObserverLoop; print('ObserverLoop OK')"
python -c "from homie_app.hotkey import HotkeyListener; print('HotkeyListener OK')"
python -c "from homie_app.overlay import OverlayPopup; print('OverlayPopup OK')"
python -c "from homie_app.daemon import HomieDaemon; print('Daemon OK')"
```
Expected: All print OK

**Step 3: Verify CLI help**

Run: `python -m homie_app.cli --help`
Expected: Shows `daemon` command in output

**Step 4: Commit if any fixes needed**

```bash
git add -A
git commit -m "test: verify full integration of always-active intelligence"
```
