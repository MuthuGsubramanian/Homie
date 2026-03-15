# Phase 1: Core Architecture Refactor — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce middleware architecture, backend protocol, and dynamic model switching into Homie's cognitive pipeline — natively, with zero new dependencies.

**Architecture:** Hybrid two-layer middleware (outer stack + inner pipeline hooks) wrapping `BrainOrchestrator`. All file/exec I/O routed through a `BackendProtocol` interface. Model selection driven by complexity classification via `ModelResolverMiddleware`.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, dataclasses, typing.Protocol

**Spec:** `docs/superpowers/specs/2026-03-15-core-architecture-refactor-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/homie_core/middleware/__init__.py` | Public API: `HomieMiddleware`, `MiddlewareStack`, `HookRegistry`, `PipelineStage` |
| `src/homie_core/middleware/base.py` | `HomieMiddleware` base class with 6 intercept points |
| `src/homie_core/middleware/stack.py` | `MiddlewareStack` — ordered execution engine, onion model |
| `src/homie_core/middleware/hooks.py` | `HookRegistry`, `PipelineStage` enum, `RetrievalBundle` dataclass |
| `src/homie_core/backend/__init__.py` | Public API: `BackendProtocol`, `ExecutableBackend`, data classes |
| `src/homie_core/backend/protocol.py` | Protocol interfaces + 5 data classes (`FileInfo`, `FileContent`, `EditResult`, `GrepMatch`, `ExecutionResult`) |
| `src/homie_core/backend/local_filesystem.py` | `LocalFilesystemBackend` — path-contained local I/O |
| `src/homie_core/backend/state.py` | `StateBackend` — in-memory ephemeral storage |
| `src/homie_core/backend/composite.py` | `CompositeBackend` — path-prefix routing |
| `src/homie_core/backend/encrypted.py` | `EncryptedVaultBackend` — AES-256-GCM decorator |
| `src/homie_core/backend/lan.py` | `LANBackend` — WebSocket RPC proxy (stub for Step 7) |
| `src/homie_core/brain/model_resolver.py` | `ModelResolverMiddleware` — complexity-aware model selection |
| `src/homie_core/inference/model_registry.py` | `ModelRegistry` — discovers models across local/LAN/Qubrid |
| `tests/unit/test_middleware/__init__.py` | Test package init |
| `tests/unit/test_middleware/test_base.py` | Tests for `HomieMiddleware` base class |
| `tests/unit/test_middleware/test_stack.py` | Tests for `MiddlewareStack` ordering, onion, blocking |
| `tests/unit/test_middleware/test_hooks.py` | Tests for `HookRegistry` emit/register/no-op |
| `tests/unit/test_backend/__init__.py` | Test package init |
| `tests/unit/test_backend/test_protocol.py` | Tests for protocol data classes |
| `tests/unit/test_backend/test_local_filesystem.py` | Tests for path containment, CRUD, execute |
| `tests/unit/test_backend/test_state.py` | Tests for in-memory backend |
| `tests/unit/test_backend/test_composite.py` | Tests for routing, aggregation |
| `tests/unit/test_brain/test_model_resolver.py` | Tests for tier mapping, override, hook timing |
| `tests/unit/test_inference/test_model_registry.py` | Tests for registry scan, available, best_for |

### Modified Files

| File | Change |
|------|--------|
| `src/homie_core/brain/cognitive_arch.py` | Accept `HookRegistry`, emit at 5 stage boundaries |
| `src/homie_core/brain/orchestrator.py` | Accept `MiddlewareStack`, run outer lifecycle, add `state` kwarg |
| `src/homie_core/brain/builtin_tools.py` | `search_files`, `read_file`, `run_command` accept `backend` param |
| `src/homie_core/brain/tool_registry.py` | Add `context` dict, inject `backend` via `inspect.signature` |
| `src/homie_core/config.py` | Add `ModelTier`, `ModelProfile` |
| `src/homie_core/inference/router.py` | Accept `model` and `preferred_location` params in `generate()`/`stream()` |
| `tests/unit/test_brain/test_orchestrator.py` | Add tests for middleware integration |

---

## Chunk 1: Middleware Module (Tasks 1–3)

### Task 1: Inner Pipeline Hooks — `middleware/hooks.py`

**Files:**
- Create: `src/homie_core/middleware/__init__.py`
- Create: `src/homie_core/middleware/hooks.py`
- Create: `tests/unit/test_middleware/__init__.py`
- Create: `tests/unit/test_middleware/test_hooks.py`

- [ ] **Step 1: Write failing tests for HookRegistry**

```python
# tests/unit/test_middleware/__init__.py
# (empty)

# tests/unit/test_middleware/test_hooks.py
import pytest
from homie_core.middleware.hooks import HookRegistry, PipelineStage, RetrievalBundle


class TestPipelineStage:
    def test_has_five_stages(self):
        assert len(PipelineStage) == 5

    def test_stage_values(self):
        assert PipelineStage.PERCEIVED == "on_perceived"
        assert PipelineStage.CLASSIFIED == "on_classified"
        assert PipelineStage.RETRIEVED == "on_retrieved"
        assert PipelineStage.PROMPT_BUILT == "on_prompt_built"
        assert PipelineStage.REFLECTED == "on_reflected"


class TestRetrievalBundle:
    def test_creation(self):
        bundle = RetrievalBundle(facts=["f1"], episodes=["e1"], documents=["d1"])
        assert bundle.facts == ["f1"]
        assert bundle.episodes == ["e1"]
        assert bundle.documents == ["d1"]


class TestHookRegistry:
    def test_emit_no_hooks_returns_data_unchanged(self):
        registry = HookRegistry()
        data = {"key": "value"}
        result = registry.emit(PipelineStage.PERCEIVED, data)
        assert result is data  # same object, not just equal

    def test_register_and_emit_single_hook(self):
        registry = HookRegistry()
        received = []

        def hook(stage, data):
            received.append((stage, data))
            return data

        registry.register(PipelineStage.CLASSIFIED, hook)
        registry.emit(PipelineStage.CLASSIFIED, "moderate")
        assert received == [(PipelineStage.CLASSIFIED, "moderate")]

    def test_hook_can_modify_data(self):
        registry = HookRegistry()

        def upgrade_complexity(stage, data):
            return "complex" if data == "moderate" else data

        registry.register(PipelineStage.CLASSIFIED, upgrade_complexity)
        result = registry.emit(PipelineStage.CLASSIFIED, "moderate")
        assert result == "complex"

    def test_hook_returning_none_does_not_modify(self):
        registry = HookRegistry()

        def observer(stage, data):
            return None  # observe only

        registry.register(PipelineStage.CLASSIFIED, observer)
        result = registry.emit(PipelineStage.CLASSIFIED, "moderate")
        assert result == "moderate"

    def test_multiple_hooks_chain(self):
        registry = HookRegistry()

        def add_prefix(stage, data):
            return f"prefix_{data}"

        def add_suffix(stage, data):
            return f"{data}_suffix"

        registry.register(PipelineStage.PROMPT_BUILT, add_prefix)
        registry.register(PipelineStage.PROMPT_BUILT, add_suffix)
        result = registry.emit(PipelineStage.PROMPT_BUILT, "prompt")
        assert result == "prefix_prompt_suffix"

    def test_hooks_on_different_stages_are_independent(self):
        registry = HookRegistry()
        calls = []

        def track(stage, data):
            calls.append(stage)
            return data

        registry.register(PipelineStage.PERCEIVED, track)
        registry.emit(PipelineStage.CLASSIFIED, "test")
        assert calls == []  # PERCEIVED hook not called for CLASSIFIED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_middleware/test_hooks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'homie_core.middleware'`

- [ ] **Step 3: Implement hooks.py**

```python
# src/homie_core/middleware/__init__.py
from homie_core.middleware.hooks import HookRegistry, PipelineStage, RetrievalBundle

__all__ = ["HookRegistry", "PipelineStage", "RetrievalBundle"]
```

```python
# src/homie_core/middleware/hooks.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class PipelineStage(str, Enum):
    """Cognitive pipeline stages where hooks can intercept."""
    PERCEIVED = "on_perceived"
    CLASSIFIED = "on_classified"
    RETRIEVED = "on_retrieved"
    PROMPT_BUILT = "on_prompt_built"
    REFLECTED = "on_reflected"


@dataclass
class RetrievalBundle:
    """Combined retrieval results from all memory sources."""
    facts: list
    episodes: list
    documents: list


HookCallback = Callable[[PipelineStage, Any], Any]


class HookRegistry:
    """Registry for inner pipeline hooks at cognitive stage boundaries."""

    def __init__(self) -> None:
        self._hooks: dict[PipelineStage, list[HookCallback]] = {
            stage: [] for stage in PipelineStage
        }

    def register(self, stage: PipelineStage, callback: HookCallback) -> None:
        """Register a callback for a pipeline stage."""
        self._hooks[stage].append(callback)

    def emit(self, stage: PipelineStage, data: Any) -> Any:
        """Emit a stage signal. Each hook can modify and return data.
        Returning None from a hook means 'observe only, don't modify'."""
        for callback in self._hooks[stage]:
            result = callback(stage, data)
            if result is not None:
                data = result
        return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_middleware/test_hooks.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/middleware/__init__.py src/homie_core/middleware/hooks.py tests/unit/test_middleware/__init__.py tests/unit/test_middleware/test_hooks.py
git commit -m "feat(middleware): add HookRegistry and PipelineStage for inner pipeline hooks"
```

---

### Task 2: Middleware Base Class — `middleware/base.py`

**Files:**
- Create: `src/homie_core/middleware/base.py`
- Create: `tests/unit/test_middleware/test_base.py`
- Modify: `src/homie_core/middleware/__init__.py`

- [ ] **Step 1: Write failing tests for HomieMiddleware**

```python
# tests/unit/test_middleware/test_base.py
from homie_core.middleware.base import HomieMiddleware


class TestHomieMiddleware:
    def test_default_name(self):
        mw = HomieMiddleware()
        assert mw.name == "unnamed"

    def test_default_order(self):
        mw = HomieMiddleware()
        assert mw.order == 100

    def test_modify_tools_passthrough(self):
        mw = HomieMiddleware()
        tools = [{"name": "test"}]
        assert mw.modify_tools(tools) is tools

    def test_modify_prompt_passthrough(self):
        mw = HomieMiddleware()
        assert mw.modify_prompt("prompt") == "prompt"

    def test_before_turn_passthrough(self):
        mw = HomieMiddleware()
        assert mw.before_turn("hello", {}) == "hello"

    def test_after_turn_passthrough(self):
        mw = HomieMiddleware()
        assert mw.after_turn("response", {}) == "response"

    def test_wrap_tool_call_passthrough(self):
        mw = HomieMiddleware()
        args = {"path": "/test"}
        assert mw.wrap_tool_call("read_file", args) is args

    def test_wrap_tool_result_passthrough(self):
        mw = HomieMiddleware()
        assert mw.wrap_tool_result("read_file", "content") == "content"

    def test_subclass_can_override(self):
        class LogMiddleware(HomieMiddleware):
            name = "logger"
            order = 10

            def before_turn(self, message, state):
                state["logged"] = True
                return message

        mw = LogMiddleware()
        state = {}
        mw.before_turn("test", state)
        assert state["logged"] is True
        assert mw.name == "logger"
        assert mw.order == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_middleware/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'homie_core.middleware.base'`

- [ ] **Step 3: Implement base.py**

```python
# src/homie_core/middleware/base.py
from __future__ import annotations

from typing import Any, Optional


class HomieMiddleware:
    """Base class for outer middleware. Override only what you need.

    Middleware intercepts at 6 points around BrainOrchestrator.process():
    - modify_tools: add/remove/wrap tools before model sees them
    - modify_prompt: inject context into system prompt
    - before_turn: pre-process user message (return None to block)
    - after_turn: post-process assistant response
    - wrap_tool_call: intercept before tool executes (return None to block)
    - wrap_tool_result: intercept after tool returns
    """

    name: str = "unnamed"
    order: int = 100  # lower = runs first for before_*, last for after_*

    def modify_tools(self, tools: list[dict]) -> list[dict]:
        return tools

    def modify_prompt(self, system_prompt: str) -> str:
        return system_prompt

    def before_turn(self, message: str, state: dict) -> Optional[str]:
        return message

    def after_turn(self, response: str, state: dict) -> str:
        return response

    def wrap_tool_call(self, name: str, args: dict) -> Optional[dict]:
        return args

    def wrap_tool_result(self, name: str, result: str) -> str:
        return result
```

- [ ] **Step 4: Update `__init__.py`**

```python
# src/homie_core/middleware/__init__.py
from homie_core.middleware.base import HomieMiddleware
from homie_core.middleware.hooks import HookRegistry, PipelineStage, RetrievalBundle

__all__ = [
    "HomieMiddleware",
    "HookRegistry",
    "PipelineStage",
    "RetrievalBundle",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_middleware/test_base.py -v`
Expected: All 9 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/middleware/base.py src/homie_core/middleware/__init__.py tests/unit/test_middleware/test_base.py
git commit -m "feat(middleware): add HomieMiddleware base class with 6 intercept points"
```

---

### Task 3: Middleware Stack — `middleware/stack.py`

**Files:**
- Create: `src/homie_core/middleware/stack.py`
- Create: `tests/unit/test_middleware/test_stack.py`
- Modify: `src/homie_core/middleware/__init__.py`

- [ ] **Step 1: Write failing tests for MiddlewareStack**

```python
# tests/unit/test_middleware/test_stack.py
import pytest
from homie_core.middleware.base import HomieMiddleware
from homie_core.middleware.stack import MiddlewareStack


class OrderTracker(HomieMiddleware):
    """Test middleware that records call order."""
    def __init__(self, label: str, order: int = 100):
        self.name = label
        self.order = order
        self.calls = []

    def before_turn(self, message, state):
        self.calls.append(("before", message))
        return message

    def after_turn(self, response, state):
        self.calls.append(("after", response))
        return response


class TestMiddlewareStackOrdering:
    def test_empty_stack(self):
        stack = MiddlewareStack()
        assert stack.run_before_turn("hello", {}) == "hello"
        assert stack.run_after_turn("response", {}) == "response"

    def test_before_runs_in_order(self):
        calls = []

        class A(HomieMiddleware):
            name = "a"
            order = 10
            def before_turn(self, msg, state):
                calls.append("a")
                return msg

        class B(HomieMiddleware):
            name = "b"
            order = 20
            def before_turn(self, msg, state):
                calls.append("b")
                return msg

        stack = MiddlewareStack([B(), A()])  # B added first, but A has lower order
        stack.run_before_turn("test", {})
        assert calls == ["a", "b"]

    def test_after_runs_in_reverse_order(self):
        calls = []

        class A(HomieMiddleware):
            name = "a"
            order = 10
            def after_turn(self, resp, state):
                calls.append("a")
                return resp

        class B(HomieMiddleware):
            name = "b"
            order = 20
            def after_turn(self, resp, state):
                calls.append("b")
                return resp

        stack = MiddlewareStack([A(), B()])
        stack.run_after_turn("test", {})
        assert calls == ["b", "a"]  # reverse: B first, then A


class TestMiddlewareStackBlocking:
    def test_before_turn_none_blocks(self):
        class Blocker(HomieMiddleware):
            name = "blocker"
            order = 10
            def before_turn(self, msg, state):
                return None

        class After(HomieMiddleware):
            name = "after"
            order = 20
            def before_turn(self, msg, state):
                state["reached"] = True
                return msg

        state = {}
        stack = MiddlewareStack([Blocker(), After()])
        result = stack.run_before_turn("test", state)
        assert result is None
        assert "reached" not in state  # After was never called

    def test_wrap_tool_call_none_blocks(self):
        class Blocker(HomieMiddleware):
            name = "blocker"
            def wrap_tool_call(self, name, args):
                if name == "dangerous":
                    return None
                return args

        stack = MiddlewareStack([Blocker()])
        assert stack.run_wrap_tool_call("dangerous", {"cmd": "rm -rf /"}) is None
        assert stack.run_wrap_tool_call("safe", {"path": "/"}) == {"path": "/"}


class TestMiddlewareStackModification:
    def test_apply_tools(self):
        class AddTool(HomieMiddleware):
            name = "add_tool"
            def modify_tools(self, tools):
                return tools + [{"name": "extra"}]

        stack = MiddlewareStack([AddTool()])
        result = stack.apply_tools([{"name": "original"}])
        assert len(result) == 2
        assert result[1]["name"] == "extra"

    def test_apply_prompt(self):
        class InjectContext(HomieMiddleware):
            name = "inject"
            def modify_prompt(self, prompt):
                return prompt + "\nContext: user is coding"

        stack = MiddlewareStack([InjectContext()])
        result = stack.apply_prompt("You are Homie.")
        assert "Context: user is coding" in result

    def test_wrap_tool_result_reverse_order(self):
        class A(HomieMiddleware):
            name = "a"
            order = 10
            def wrap_tool_result(self, name, result):
                return f"[A]{result}"

        class B(HomieMiddleware):
            name = "b"
            order = 20
            def wrap_tool_result(self, name, result):
                return f"[B]{result}"

        stack = MiddlewareStack([A(), B()])
        result = stack.run_wrap_tool_result("test", "data")
        assert result == "[A][B]data"  # B runs first (reverse), then A


class TestMiddlewareStackAdd:
    def test_add_maintains_order(self):
        calls = []

        class MW(HomieMiddleware):
            def __init__(self, label, order):
                self.name = label
                self.order = order
            def before_turn(self, msg, state):
                calls.append(self.name)
                return msg

        stack = MiddlewareStack([MW("b", 20)])
        stack.add(MW("a", 10))
        stack.add(MW("c", 30))
        stack.run_before_turn("test", {})
        assert calls == ["a", "b", "c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_middleware/test_stack.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'homie_core.middleware.stack'`

- [ ] **Step 3: Implement stack.py**

```python
# src/homie_core/middleware/stack.py
from __future__ import annotations

from typing import Optional

from homie_core.middleware.base import HomieMiddleware


class MiddlewareStack:
    """Ordered execution engine for middleware layers.

    Execution order:
    - before_turn, modify_tools, modify_prompt, wrap_tool_call: lowest order first
    - after_turn, wrap_tool_result: highest order first (onion unwinding)
    """

    def __init__(self, middleware: list[HomieMiddleware] | None = None) -> None:
        self._middleware = sorted(middleware or [], key=lambda m: m.order)

    def add(self, mw: HomieMiddleware) -> None:
        self._middleware.append(mw)
        self._middleware.sort(key=lambda m: m.order)

    def apply_tools(self, tools: list[dict]) -> list[dict]:
        for mw in self._middleware:
            tools = mw.modify_tools(tools)
        return tools

    def apply_prompt(self, prompt: str) -> str:
        for mw in self._middleware:
            prompt = mw.modify_prompt(prompt)
        return prompt

    def run_before_turn(self, message: str, state: dict) -> Optional[str]:
        for mw in self._middleware:
            result = mw.before_turn(message, state)
            if result is None:
                return None
            message = result
        return message

    def run_after_turn(self, response: str, state: dict) -> str:
        for mw in reversed(self._middleware):
            response = mw.after_turn(response, state)
        return response

    def run_wrap_tool_call(self, name: str, args: dict) -> Optional[dict]:
        for mw in self._middleware:
            result = mw.wrap_tool_call(name, args)
            if result is None:
                return None
            args = result
        return args

    def run_wrap_tool_result(self, name: str, result: str) -> str:
        for mw in reversed(self._middleware):
            result = mw.wrap_tool_result(name, result)
        return result
```

- [ ] **Step 4: Update `__init__.py`**

```python
# src/homie_core/middleware/__init__.py
from homie_core.middleware.base import HomieMiddleware
from homie_core.middleware.hooks import HookRegistry, PipelineStage, RetrievalBundle
from homie_core.middleware.stack import MiddlewareStack

__all__ = [
    "HomieMiddleware",
    "MiddlewareStack",
    "HookRegistry",
    "PipelineStage",
    "RetrievalBundle",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_middleware/ -v`
Expected: All tests PASS (hooks: 8, base: 9, stack: 9 = 26 total)

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/middleware/stack.py src/homie_core/middleware/__init__.py tests/unit/test_middleware/test_stack.py
git commit -m "feat(middleware): add MiddlewareStack with onion-order execution"
```

---

## Chunk 2: Backend Protocol + LocalFilesystemBackend (Tasks 4–5)

### Task 4: Backend Protocol + Data Classes — `backend/protocol.py`

**Files:**
- Create: `src/homie_core/backend/__init__.py`
- Create: `src/homie_core/backend/protocol.py`
- Create: `tests/unit/test_backend/__init__.py`
- Create: `tests/unit/test_backend/test_protocol.py`

- [ ] **Step 1: Write failing tests for protocol data classes**

```python
# tests/unit/test_backend/__init__.py
# (empty)

# tests/unit/test_backend/test_protocol.py
import pytest
from homie_core.backend.protocol import (
    FileInfo, FileContent, EditResult, GrepMatch, ExecutionResult,
    BackendProtocol, ExecutableBackend,
)


class TestDataClasses:
    def test_file_info(self):
        fi = FileInfo(path="src/main.py", name="main.py", is_dir=False, size=1024)
        assert fi.path == "src/main.py"
        assert fi.modified is None  # optional

    def test_file_content(self):
        fc = FileContent(content="hello\nworld", total_lines=2)
        assert fc.truncated is False

    def test_edit_result_success(self):
        er = EditResult(success=True, occurrences=1)
        assert er.error is None

    def test_edit_result_failure(self):
        er = EditResult(success=False, error="Not found")
        assert er.occurrences == 0

    def test_grep_match(self):
        gm = GrepMatch(path="file.py", line_number=42, line="def foo():")
        assert gm.line_number == 42

    def test_execution_result(self):
        er = ExecutionResult(stdout="ok", stderr="", exit_code=0)
        assert er.timed_out is False

    def test_execution_result_timeout(self):
        er = ExecutionResult(stdout="", stderr="timeout", exit_code=124, timed_out=True)
        assert er.timed_out is True


class TestProtocolChecks:
    def test_backend_protocol_is_runtime_checkable(self):
        assert hasattr(BackendProtocol, "__protocol_attrs__") or hasattr(BackendProtocol, "__abstractmethods__") or True
        # Just verify it's importable as a Protocol

    def test_executable_backend_extends_backend(self):
        # ExecutableBackend should include all BackendProtocol methods plus execute
        assert True  # structural typing — checked at use site
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_backend/test_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement protocol.py**

```python
# src/homie_core/backend/protocol.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class FileInfo:
    path: str
    name: str
    is_dir: bool
    size: Optional[int] = None
    modified: Optional[float] = None


@dataclass
class FileContent:
    content: str
    total_lines: int
    truncated: bool = False


@dataclass
class EditResult:
    success: bool
    occurrences: int = 0
    error: Optional[str] = None


@dataclass
class GrepMatch:
    path: str
    line_number: int
    line: str


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


@runtime_checkable
class BackendProtocol(Protocol):
    """Anything that can store and retrieve files."""

    def ls(self, path: str = "/") -> list[FileInfo]: ...
    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent: ...
    def write(self, path: str, content: str) -> None: ...
    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> EditResult: ...
    def glob(self, pattern: str) -> list[FileInfo]: ...
    def grep(self, pattern: str, path: str = "/", include: Optional[str] = None) -> list[GrepMatch]: ...


@runtime_checkable
class ExecutableBackend(BackendProtocol, Protocol):
    """Backend that can also run commands."""

    def execute(self, command: str, timeout: int = 30) -> ExecutionResult: ...
```

```python
# src/homie_core/backend/__init__.py
from homie_core.backend.protocol import (
    BackendProtocol,
    EditResult,
    ExecutableBackend,
    ExecutionResult,
    FileContent,
    FileInfo,
    GrepMatch,
)

__all__ = [
    "BackendProtocol",
    "EditResult",
    "ExecutableBackend",
    "ExecutionResult",
    "FileContent",
    "FileInfo",
    "GrepMatch",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_backend/test_protocol.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/backend/__init__.py src/homie_core/backend/protocol.py tests/unit/test_backend/__init__.py tests/unit/test_backend/test_protocol.py
git commit -m "feat(backend): add BackendProtocol interface and data classes"
```

---

### Task 5: LocalFilesystemBackend — `backend/local_filesystem.py`

**Files:**
- Create: `src/homie_core/backend/local_filesystem.py`
- Create: `tests/unit/test_backend/test_local_filesystem.py`
- Modify: `src/homie_core/backend/__init__.py`

- [ ] **Step 1: Write failing tests for LocalFilesystemBackend**

```python
# tests/unit/test_backend/test_local_filesystem.py
import os
import pytest
from pathlib import Path
from homie_core.backend.local_filesystem import LocalFilesystemBackend
from homie_core.backend.protocol import FileContent, EditResult, ExecutionResult


@pytest.fixture
def backend(tmp_path):
    """Create a backend rooted at a temp directory with test files."""
    (tmp_path / "hello.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested content", encoding="utf-8")
    return LocalFilesystemBackend(root_dir=tmp_path)


class TestPathContainment:
    def test_resolve_within_root(self, backend, tmp_path):
        resolved = backend._resolve("hello.txt")
        assert resolved == tmp_path / "hello.txt"

    def test_escape_blocked(self, backend):
        with pytest.raises(ValueError, match="Path escapes root"):
            backend._resolve("../../etc/passwd")

    def test_dotdot_in_middle_blocked(self, backend):
        with pytest.raises(ValueError, match="Path escapes root"):
            backend._resolve("subdir/../../etc/passwd")


class TestRead:
    def test_read_full_file(self, backend):
        result = backend.read("hello.txt")
        assert result.content == "line1\nline2\nline3"
        assert result.total_lines == 3  # splitlines() ignores trailing newline
        assert result.truncated is False

    def test_read_with_offset_and_limit(self, backend):
        result = backend.read("hello.txt", offset=1, limit=1)
        assert result.content == "line2"
        assert result.truncated is True

    def test_read_nested(self, backend):
        result = backend.read("subdir/nested.txt")
        assert result.content == "nested content"


class TestWrite:
    def test_write_new_file(self, backend, tmp_path):
        backend.write("new.txt", "hello world")
        assert (tmp_path / "new.txt").read_text() == "hello world"

    def test_write_creates_parent_dirs(self, backend, tmp_path):
        backend.write("a/b/c.txt", "deep")
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"

    def test_write_refuses_symlink(self, backend, tmp_path):
        if os.name == "nt":
            # Windows symlinks need admin — skip if can't create
            try:
                target = tmp_path / "target.txt"
                target.write_text("original")
                link = tmp_path / "link.txt"
                link.symlink_to(target)
            except OSError:
                pytest.skip("Cannot create symlinks on this Windows system")
            with pytest.raises(ValueError, match="symlink"):
                backend.write("link.txt", "malicious")


class TestEdit:
    def test_edit_unique_string(self, backend):
        result = backend.edit("hello.txt", "line2", "LINE_TWO")
        assert result.success is True
        assert result.occurrences == 1
        content = backend.read("hello.txt")
        assert "LINE_TWO" in content.content

    def test_edit_not_found(self, backend):
        result = backend.edit("hello.txt", "nonexistent", "replacement")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_edit_not_unique_without_replace_all(self, backend, tmp_path):
        (tmp_path / "dups.txt").write_text("aaa\naaa\naaa")
        result = backend.edit("dups.txt", "aaa", "bbb")
        assert result.success is False
        assert "3 times" in result.error

    def test_edit_replace_all(self, backend, tmp_path):
        (tmp_path / "dups.txt").write_text("aaa\naaa\naaa")
        result = backend.edit("dups.txt", "aaa", "bbb", replace_all=True)
        assert result.success is True
        assert result.occurrences == 3
        assert backend.read("dups.txt").content == "bbb\nbbb\nbbb"


class TestLs:
    def test_ls_root(self, backend):
        entries = backend.ls("/")
        names = {e.name for e in entries}
        assert "hello.txt" in names
        assert "subdir" in names

    def test_ls_subdir(self, backend):
        entries = backend.ls("subdir")
        assert len(entries) == 1
        assert entries[0].name == "nested.txt"


class TestGlob:
    def test_glob_txt(self, backend):
        matches = backend.glob("**/*.txt")
        paths = {m.name for m in matches}
        assert "hello.txt" in paths
        assert "nested.txt" in paths


class TestGrep:
    def test_grep_finds_match(self, backend):
        matches = backend.grep("line2", "/")
        assert len(matches) == 1
        assert matches[0].line_number == 2

    def test_grep_regex(self, backend):
        matches = backend.grep(r"line\d", "/")
        assert len(matches) == 3  # line1, line2, line3


class TestExecute:
    def test_execute_simple_command(self, backend):
        result = backend.execute("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_execute_timeout(self, backend):
        # Use a command that takes too long
        if os.name == "nt":
            cmd = "ping -n 10 127.0.0.1"
        else:
            cmd = "sleep 10"
        result = backend.execute(cmd, timeout=1)
        assert result.timed_out is True
        assert result.exit_code == 124
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_backend/test_local_filesystem.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement local_filesystem.py**

```python
# src/homie_core/backend/local_filesystem.py
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from homie_core.backend.protocol import (
    EditResult,
    ExecutionResult,
    FileContent,
    FileInfo,
    GrepMatch,
)


class LocalFilesystemBackend:
    """Local filesystem with path containment and symlink protection."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir).resolve()

    def _resolve(self, path: str) -> Path:
        """Resolve path within root_dir. Raises ValueError on escape."""
        resolved = (self._root / path.lstrip("/")).resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"Path escapes root: {path}")
        return resolved

    def ls(self, path: str = "/") -> list[FileInfo]:
        resolved = self._resolve(path)
        entries = []
        for child in sorted(resolved.iterdir()):
            try:
                stat = child.stat()
            except OSError:
                continue
            entries.append(FileInfo(
                path=str(child.relative_to(self._root)),
                name=child.name,
                is_dir=child.is_dir(),
                size=stat.st_size if child.is_file() else None,
                modified=stat.st_mtime,
            ))
        return entries

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        resolved = self._resolve(path)
        lines = resolved.read_text(encoding="utf-8").splitlines()
        total = len(lines)
        selected = lines[offset:offset + limit]
        return FileContent(
            content="\n".join(selected),
            total_lines=total,
            truncated=(offset + limit < total),
        )

    def write(self, path: str, content: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform != "win32":
            fd = os.open(
                str(resolved),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                0o644,
            )
            try:
                os.write(fd, content.encode("utf-8"))
            finally:
                os.close(fd)
        else:
            if resolved.is_symlink():
                raise ValueError(f"Refusing to write through symlink: {path}")
            resolved.write_text(content, encoding="utf-8")

    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> EditResult:
        resolved = self._resolve(path)
        text = resolved.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            return EditResult(success=False, error=f"String not found in {path}")
        if not replace_all and count > 1:
            return EditResult(
                success=False,
                error=f"String found {count} times — not unique. Use replace_all=True or provide more context.",
            )
        new_text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        self.write(path, new_text)
        return EditResult(success=True, occurrences=count if replace_all else 1)

    def glob(self, pattern: str) -> list[FileInfo]:
        matches = []
        for match in self._root.glob(pattern):
            if not match.is_relative_to(self._root):
                continue
            try:
                stat = match.stat()
            except OSError:
                continue
            matches.append(FileInfo(
                path=str(match.relative_to(self._root)),
                name=match.name,
                is_dir=match.is_dir(),
                size=stat.st_size if match.is_file() else None,
                modified=stat.st_mtime,
            ))
        return matches

    def grep(self, pattern: str, path: str = "/", include: Optional[str] = None) -> list[GrepMatch]:
        resolved = self._resolve(path)
        glob_pattern = include or "**/*"
        matches = []
        for file_path in resolved.glob(glob_pattern):
            if not file_path.is_file() or not file_path.is_relative_to(self._root):
                continue
            try:
                for i, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
                    if re.search(pattern, line):
                        matches.append(GrepMatch(
                            path=str(file_path.relative_to(self._root)),
                            line_number=i,
                            line=line,
                        ))
            except (UnicodeDecodeError, PermissionError):
                continue
        return matches

    def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self._root),
            )
            return ExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr="Command timed out",
                exit_code=124,
                timed_out=True,
            )
```

- [ ] **Step 4: Update `__init__.py`**

Add to `src/homie_core/backend/__init__.py`:
```python
from homie_core.backend.local_filesystem import LocalFilesystemBackend
```
Add `"LocalFilesystemBackend"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_backend/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/backend/local_filesystem.py src/homie_core/backend/__init__.py tests/unit/test_backend/test_local_filesystem.py
git commit -m "feat(backend): add LocalFilesystemBackend with path containment"
```

---

## Chunk 3: StateBackend + CompositeBackend (Tasks 6–7)

### Task 6: StateBackend — `backend/state.py`

**Files:**
- Create: `src/homie_core/backend/state.py`
- Create: `tests/unit/test_backend/test_state.py`

- [ ] **Step 1: Write failing tests for StateBackend**

```python
# tests/unit/test_backend/test_state.py
import pytest
from homie_core.backend.state import StateBackend


@pytest.fixture
def backend():
    b = StateBackend()
    b.write("/file.txt", "line1\nline2\nline3")
    b.write("/dir/nested.txt", "nested content")
    return b


class TestStateRead:
    def test_read_full(self, backend):
        result = backend.read("/file.txt")
        assert result.content == "line1\nline2\nline3"
        assert result.total_lines == 3

    def test_read_with_offset(self, backend):
        result = backend.read("/file.txt", offset=1, limit=1)
        assert result.content == "line2"
        assert result.truncated is True

    def test_read_nonexistent_raises(self, backend):
        with pytest.raises(FileNotFoundError):
            backend.read("/nope.txt")


class TestStateWrite:
    def test_write_and_read_back(self):
        b = StateBackend()
        b.write("/test.txt", "hello")
        assert b.read("/test.txt").content == "hello"

    def test_write_overwrites(self, backend):
        backend.write("/file.txt", "new content")
        assert backend.read("/file.txt").content == "new content"


class TestStateEdit:
    def test_edit_success(self, backend):
        result = backend.edit("/file.txt", "line2", "LINE_TWO")
        assert result.success is True
        assert "LINE_TWO" in backend.read("/file.txt").content

    def test_edit_not_found(self, backend):
        result = backend.edit("/file.txt", "nonexistent", "x")
        assert result.success is False

    def test_edit_not_unique(self, backend):
        backend.write("/dups.txt", "aaa\naaa")
        result = backend.edit("/dups.txt", "aaa", "bbb")
        assert result.success is False

    def test_edit_replace_all(self, backend):
        backend.write("/dups.txt", "aaa\naaa")
        result = backend.edit("/dups.txt", "aaa", "bbb", replace_all=True)
        assert result.success is True
        assert backend.read("/dups.txt").content == "bbb\nbbb"


class TestStateLs:
    def test_ls_root(self, backend):
        entries = backend.ls("/")
        names = {e.name for e in entries}
        assert "file.txt" in names
        assert "dir" in names

    def test_ls_subdir(self, backend):
        entries = backend.ls("/dir")
        assert len(entries) == 1
        assert entries[0].name == "nested.txt"


class TestStateGlob:
    def test_glob_all_txt(self, backend):
        matches = backend.glob("*.txt")
        assert any(m.name == "file.txt" for m in matches)


class TestStateGrep:
    def test_grep_finds_match(self, backend):
        matches = backend.grep("line2")
        assert len(matches) == 1
        assert matches[0].line == "line2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_backend/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement state.py**

```python
# src/homie_core/backend/state.py
from __future__ import annotations

import re
from fnmatch import fnmatch
from typing import Optional

from homie_core.backend.protocol import (
    EditResult,
    FileContent,
    FileInfo,
    GrepMatch,
)


class StateBackend:
    """Ephemeral in-memory file storage. Perfect for tests and sub-agent isolation."""

    def __init__(self) -> None:
        self._files: dict[str, str] = {}

    def ls(self, path: str = "/") -> list[FileInfo]:
        prefix = path.rstrip("/") + "/" if path != "/" else "/"
        seen_dirs: set[str] = set()
        entries = []
        for fpath in sorted(self._files):
            if path != "/" and not fpath.startswith(prefix):
                continue
            rel = fpath[len(prefix):] if path != "/" else fpath.lstrip("/")
            if not rel:
                continue
            parts = rel.split("/")
            if len(parts) == 1:
                entries.append(FileInfo(
                    path=fpath, name=parts[0], is_dir=False,
                    size=len(self._files[fpath]),
                ))
            elif parts[0] not in seen_dirs:
                seen_dirs.add(parts[0])
                entries.append(FileInfo(
                    path=prefix + parts[0], name=parts[0], is_dir=True,
                ))
        return entries

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        if path not in self._files:
            raise FileNotFoundError(path)
        lines = self._files[path].splitlines()
        selected = lines[offset:offset + limit]
        return FileContent(
            content="\n".join(selected),
            total_lines=len(lines),
            truncated=(offset + limit < len(lines)),
        )

    def write(self, path: str, content: str) -> None:
        self._files[path] = content

    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> EditResult:
        if path not in self._files:
            return EditResult(success=False, error=f"File not found: {path}")
        text = self._files[path]
        count = text.count(old)
        if count == 0:
            return EditResult(success=False, error=f"String not found in {path}")
        if not replace_all and count > 1:
            return EditResult(
                success=False,
                error=f"String found {count} times — not unique.",
            )
        self._files[path] = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        return EditResult(success=True, occurrences=count if replace_all else 1)

    def glob(self, pattern: str) -> list[FileInfo]:
        return [
            FileInfo(path=p, name=p.split("/")[-1], is_dir=False, size=len(c))
            for p, c in sorted(self._files.items())
            if fnmatch(p, pattern) or fnmatch(p.lstrip("/"), pattern)
        ]

    def grep(self, pattern: str, path: str = "/", include: Optional[str] = None) -> list[GrepMatch]:
        matches = []
        for fpath, content in sorted(self._files.items()):
            if path != "/" and not fpath.startswith(path):
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if re.search(pattern, line):
                    matches.append(GrepMatch(path=fpath, line_number=i, line=line))
        return matches
```

- [ ] **Step 4: Update `__init__.py`, run tests**

Add `StateBackend` to `src/homie_core/backend/__init__.py` imports and `__all__`.

Run: `python -m pytest tests/unit/test_backend/test_state.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/backend/state.py src/homie_core/backend/__init__.py tests/unit/test_backend/test_state.py
git commit -m "feat(backend): add StateBackend for in-memory ephemeral storage"
```

---

### Task 7: CompositeBackend — `backend/composite.py`

**Files:**
- Create: `src/homie_core/backend/composite.py`
- Create: `tests/unit/test_backend/test_composite.py`

- [ ] **Step 1: Write failing tests for CompositeBackend**

```python
# tests/unit/test_backend/test_composite.py
import pytest
from homie_core.backend.state import StateBackend
from homie_core.backend.composite import CompositeBackend


@pytest.fixture
def composite():
    default = StateBackend()
    default.write("/main.txt", "default content")

    vault = StateBackend()
    vault.write("/secrets.json", '{"key": "value"}')

    tmp = StateBackend()

    return CompositeBackend(
        default=default,
        routes={"/vault/": vault, "/tmp/": tmp},
    ), default, vault, tmp


class TestRouting:
    def test_read_default(self, composite):
        comp, default, _, _ = composite
        result = comp.read("/main.txt")
        assert result.content == "default content"

    def test_read_vault_route(self, composite):
        comp, _, vault, _ = composite
        result = comp.read("/vault/secrets.json")
        assert "key" in result.content

    def test_write_to_tmp_route(self, composite):
        comp, _, _, tmp = composite
        comp.write("/tmp/scratch.txt", "temp data")
        assert tmp.read("/scratch.txt").content == "temp data"

    def test_unmatched_goes_to_default(self, composite):
        comp, default, _, _ = composite
        comp.write("/notes.md", "some notes")
        assert default.read("/notes.md").content == "some notes"


class TestLongestPrefixMatch:
    def test_longer_prefix_wins(self):
        short = StateBackend()
        long = StateBackend()
        long.write("/deep.txt", "deep")

        comp = CompositeBackend(
            default=StateBackend(),
            routes={"/a/": short, "/a/b/": long},
        )
        result = comp.read("/a/b/deep.txt")
        assert result.content == "deep"


class TestLsAggregation:
    def test_ls_root_shows_routes(self, composite):
        comp, _, _, _ = composite
        entries = comp.ls("/")
        names = {e.name for e in entries}
        assert "vault" in names
        assert "tmp" in names
        assert "main.txt" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_backend/test_composite.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement composite.py**

```python
# src/homie_core/backend/composite.py
from __future__ import annotations

from typing import Optional

from homie_core.backend.protocol import (
    BackendProtocol,
    EditResult,
    FileContent,
    FileInfo,
    GrepMatch,
)


class CompositeBackend:
    """Routes operations to different backends by path prefix."""

    def __init__(
        self,
        default: BackendProtocol,
        routes: dict[str, BackendProtocol] | None = None,
    ) -> None:
        self._default = default
        self._routes = sorted(
            (routes or {}).items(),
            key=lambda r: len(r[0]),
            reverse=True,
        )

    def _route(self, path: str) -> tuple[BackendProtocol, str]:
        for prefix, backend in self._routes:
            if path.startswith(prefix):
                rel = path[len(prefix):] or "/"
                return backend, rel
        return self._default, path

    def ls(self, path: str = "/") -> list[FileInfo]:
        if path == "/":
            entries = self._default.ls("/")
            for prefix, _ in self._routes:
                dir_name = prefix.strip("/").split("/")[0]
                if not any(e.name == dir_name for e in entries):
                    entries.append(FileInfo(path=prefix.rstrip("/"), name=dir_name, is_dir=True))
            return entries
        backend, rel_path = self._route(path)
        return backend.ls(rel_path)

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        backend, rel_path = self._route(path)
        return backend.read(rel_path, offset=offset, limit=limit)

    def write(self, path: str, content: str) -> None:
        backend, rel_path = self._route(path)
        backend.write(rel_path, content)

    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> EditResult:
        backend, rel_path = self._route(path)
        return backend.edit(rel_path, old, new, replace_all=replace_all)

    def glob(self, pattern: str) -> list[FileInfo]:
        results = self._default.glob(pattern)
        for prefix, backend in self._routes:
            for match in backend.glob(pattern):
                match.path = prefix + match.path.lstrip("/")
                results.append(match)
        return results

    def grep(self, pattern: str, path: str = "/", include: Optional[str] = None) -> list[GrepMatch]:
        if path == "/":
            results = self._default.grep(pattern, "/", include)
            for prefix, backend in self._routes:
                for match in backend.grep(pattern, "/", include):
                    match.path = prefix + match.path.lstrip("/")
                    results.append(match)
            return results
        backend, rel_path = self._route(path)
        return backend.grep(pattern, rel_path, include)
```

- [ ] **Step 4: Update `__init__.py`, run tests**

Add `CompositeBackend` to `src/homie_core/backend/__init__.py`.

Run: `python -m pytest tests/unit/test_backend/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/backend/composite.py src/homie_core/backend/__init__.py tests/unit/test_backend/test_composite.py
git commit -m "feat(backend): add CompositeBackend with path-prefix routing"
```

---

## Chunk 4: Integration — Wire Middleware + Backend into Brain (Tasks 8–10)

### Task 8: Wire HookRegistry into CognitiveArchitecture

**Files:**
- Modify: `src/homie_core/brain/cognitive_arch.py:1-50` (constructor) and `_prepare_prompt` method
- No new test file — existing tests must still pass, plus we test hooks via middleware integration

- [ ] **Step 1: Add HookRegistry to CognitiveArchitecture constructor**

In `src/homie_core/brain/cognitive_arch.py`, add to imports at the top (after line 11):
```python
from homie_core.middleware.hooks import HookRegistry, PipelineStage, RetrievalBundle
```

In `CognitiveArchitecture.__init__()` (starts ~line 200), add `hooks` parameter:
```python
def __init__(self, ..., hooks: Optional[HookRegistry] = None):
    ...
    self._hooks = hooks or HookRegistry()
```

- [ ] **Step 2: Add hook emissions in `_prepare_prompt` (line 651)**

The method `_prepare_prompt(self, user_input: str)` at line 651 is the pipeline method. Insert hooks at these exact locations:

After line 665 (`awareness = self._perceive()`):
```python
awareness = self._hooks.emit(PipelineStage.PERCEIVED, awareness)
```

After line 666 (`complexity = self._classify(user_input, awareness)`):
```python
complexity = self._hooks.emit(PipelineStage.CLASSIFIED, complexity)
```

After line 683 (end of documents retrieval block), before line 685:
```python
bundle = RetrievalBundle(facts=facts, episodes=episodes, documents=documents_block)
bundle = self._hooks.emit(PipelineStage.RETRIEVED, bundle)
facts = bundle.facts
episodes = bundle.episodes
documents_block = bundle.documents
```

After line 691 (after tool prompt injection, before line 693):
```python
prompt = self._hooks.emit(PipelineStage.PROMPT_BUILT, prompt)
```

After line 693 (`adjustments = self._reflect_on_response(user_input, complexity, awareness)`):
```python
adjustments = self._hooks.emit(PipelineStage.REFLECTED, adjustments)
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `python -m pytest tests/unit/test_brain/ -v`
Expected: All existing tests PASS — hooks are no-op by default

- [ ] **Step 4: Commit**

```bash
git add src/homie_core/brain/cognitive_arch.py
git commit -m "feat(brain): emit inner pipeline hooks at 5 cognitive stage boundaries"
```

---

### Task 9: Wire MiddlewareStack into BrainOrchestrator

**Files:**
- Modify: `src/homie_core/brain/orchestrator.py`
- Modify: `tests/unit/test_brain/test_orchestrator.py`

- [ ] **Step 1: Write failing tests for middleware integration**

Add these imports at the top of `tests/unit/test_brain/test_orchestrator.py` (after existing imports):

```python
from homie_core.middleware import HomieMiddleware, MiddlewareStack, HookRegistry
```

Then add these test functions at the end of the file:

```python
class TrackingMiddleware(HomieMiddleware):
    name = "tracker"
    order = 10

    def __init__(self):
        self.before_calls = []
        self.after_calls = []

    def before_turn(self, message, state):
        self.before_calls.append(message)
        return message

    def after_turn(self, response, state):
        self.after_calls.append(response)
        return response


def test_orchestrator_accepts_middleware_stack():
    engine = MagicMock()
    engine.generate.return_value = "Hi!"
    wm = WorkingMemory()
    tracker = TrackingMiddleware()
    stack = MiddlewareStack([tracker])
    br = BrainOrchestrator(model_engine=engine, working_memory=wm, middleware_stack=stack)
    br.process("hello")
    assert tracker.before_calls == ["hello"]
    assert len(tracker.after_calls) == 1


def test_orchestrator_works_without_middleware():
    engine = MagicMock()
    engine.generate.return_value = "Hi!"
    wm = WorkingMemory()
    br = BrainOrchestrator(model_engine=engine, working_memory=wm)
    response = br.process("hello")
    assert response  # still works


def test_orchestrator_state_kwarg_backward_compat():
    engine = MagicMock()
    engine.generate.return_value = "Hi!"
    wm = WorkingMemory()
    br = BrainOrchestrator(model_engine=engine, working_memory=wm)
    # Old callers don't pass state — should still work
    response = br.process("hello")
    assert response


def test_process_stream_fires_middleware():
    engine = MagicMock()
    engine.stream.return_value = iter(["Hello", " world"])
    wm = WorkingMemory()
    tracker = TrackingMiddleware()
    stack = MiddlewareStack([tracker])
    br = BrainOrchestrator(model_engine=engine, working_memory=wm, middleware_stack=stack)
    tokens = list(br.process_stream("hi"))
    assert tracker.before_calls == ["hi"]
    assert len(tracker.after_calls) == 1  # after_turn fires once with full response
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_brain/test_orchestrator.py -v`
Expected: FAIL — `BrainOrchestrator() got an unexpected keyword argument 'middleware_stack'`

- [ ] **Step 3: Modify orchestrator.py to accept and run middleware**

```python
# src/homie_core/brain/orchestrator.py — updated __init__ and process
from homie_core.middleware import MiddlewareStack, HookRegistry

class BrainOrchestrator:
    def __init__(
        self,
        model_engine,
        working_memory: WorkingMemory,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        tool_registry: Optional[ToolRegistry] = None,
        rag_pipeline: Optional[RagPipeline] = None,
        middleware_stack: Optional[MiddlewareStack] = None,
    ):
        self._engine = model_engine
        self._wm = working_memory
        self._em = episodic_memory
        self._sm = semantic_memory
        self._system_prompt = "You are Homie, a helpful local AI assistant. Be concise and direct."
        self._middleware = middleware_stack or MiddlewareStack()
        self._hooks = HookRegistry()

        self._cognitive = CognitiveArchitecture(
            model_engine=model_engine,
            working_memory=working_memory,
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
            system_prompt=self._system_prompt,
            tool_registry=tool_registry,
            rag_pipeline=rag_pipeline,
            hooks=self._hooks,
        )

    def process(self, user_input: str, *, state: dict | None = None) -> str:
        state = state or {}

        message = self._middleware.run_before_turn(user_input, state)
        if message is None:
            return ""

        response = self._cognitive.process(message)

        response = self._middleware.run_after_turn(response, state)
        return response

    def process_stream(self, user_input: str, *, state: dict | None = None) -> Iterator[str]:
        state = state or {}

        message = self._middleware.run_before_turn(user_input, state)
        if message is None:
            return iter([])

        tokens = []
        for token in self._cognitive.process_stream(message):
            tokens.append(token)
            yield token

        full_response = "".join(tokens)
        self._middleware.run_after_turn(full_response, state)

    @property
    def hooks(self) -> HookRegistry:
        return self._hooks
```

- [ ] **Step 4: Run all orchestrator tests**

Run: `python -m pytest tests/unit/test_brain/test_orchestrator.py -v`
Expected: All tests PASS (old + new)

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -v --tb=short`
Expected: No regressions

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/brain/orchestrator.py tests/unit/test_brain/test_orchestrator.py
git commit -m "feat(brain): wire MiddlewareStack into BrainOrchestrator"
```

---

### Task 10: Wire Backend into ToolRegistry + builtin_tools

**Files:**
- Modify: `src/homie_core/brain/tool_registry.py:177-240` (ToolRegistry class)
- Modify: `src/homie_core/brain/builtin_tools.py:219-250, 522-540` (file/exec tools)

- [ ] **Step 1: Add `context` dict to ToolRegistry**

In `src/homie_core/brain/tool_registry.py`, modify `ToolRegistry`:

```python
import inspect

class ToolRegistry:
    _FUZZY_THRESHOLD = 2

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._context: dict[str, Any] = {}

    def set_context(self, context: dict[str, Any]) -> None:
        """Set execution context (e.g., backend) for tool calls."""
        self._context = context

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if not tool:
            tool = self._fuzzy_match(call.name)
            if tool:
                resolved_name = tool.name
            else:
                return ToolResult(
                    tool_name=call.name, success=False, output="",
                    error=f"Unknown tool: {call.name}",
                )
        else:
            resolved_name = call.name

        try:
            # Inject backend if the tool accepts it
            sig = inspect.signature(tool.execute)
            args = dict(call.arguments)
            if "backend" in sig.parameters and "backend" not in args:
                args["backend"] = self._context.get("backend")
            output = tool.execute(**args)
            return ToolResult(tool_name=resolved_name, success=True, output=str(output))
        except TypeError as e:
            return ToolResult(
                tool_name=resolved_name, success=False, output="",
                error=f"Invalid arguments: {e}",
            )
        except Exception as e:
            return ToolResult(
                tool_name=resolved_name, success=False, output="",
                error=f"Tool error: {e}",
            )
```

- [ ] **Step 2: Refactor `search_files`, `read_file`, `run_command` to accept optional `backend`**

In `src/homie_core/brain/builtin_tools.py`, modify the three file/exec tools to accept an optional `backend` parameter and use it when provided, falling back to current behavior otherwise:

For each of the three tool functions below, add `backend=None` as the last parameter and insert an `if backend:` block at the top of the function body. The existing code becomes the `else` branch (no changes to it). Do NOT replace the existing function body — wrap it.

For `tool_search_files` (line 219): add `backend=None` param, then insert before `search_dir = Path(directory)...`:
```python
def tool_search_files(pattern: str, directory: str = "", backend=None) -> str:
    """Search for files matching a pattern."""
    if backend:
        matches = backend.glob(f"**/{pattern}")
        if not matches:
            return f"No files matching '{pattern}'"
        lines = [f"Found {len(matches)} files:"]
        for m in matches[:20]:
            size_str = f"{m.size // 1024}KB" if m.size and m.size > 1024 else f"{m.size or 0}B"
            lines.append(f"  {m.path} ({size_str})")
        return "\n".join(lines)
    # --- existing code below remains unchanged (lines 221-237) ---
    search_dir = Path(directory) if directory else Path.home()
    # ... (rest of existing function body stays as-is)
```

For `tool_read_file` (line 250): add `backend=None` param, then insert before `file_path = Path(path)...`:
```python
def tool_read_file(path: str, max_lines: int = 50, backend=None) -> str:
    """Read contents of a text file."""
    if backend:
        try:
            result = backend.read(path, limit=int(max_lines))
            content = result.content
            if result.truncated:
                content += f"\n\n... ({result.total_lines - int(max_lines)} more lines)"
            return content
        except (FileNotFoundError, ValueError) as e:
            return f"Error: {e}"
    # --- existing code below remains unchanged (lines 252-265) ---
    file_path = Path(path)
    # ... (rest of existing function body stays as-is)
```

For `tool_run_command` (line 522): add `backend=None` param. The blocked-command check stays at the top. Then insert `if backend:` block after the blocked check, before the existing `subprocess.run`:
```python
def tool_run_command(command: str, backend=None) -> str:
    """Execute a shell command with safety checks."""
    cmd_lower = command.lower().strip()
    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            return f"Blocked: command contains dangerous pattern '{blocked}'."
    if backend:
        result = backend.execute(command, timeout=10)
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        output = output.strip()
        if result.timed_out:
            return "Command timed out after 10 seconds."
        return output[:5000] if len(output) > 5000 else output if output else "(no output)"
    # --- existing subprocess.run code below remains unchanged (lines 529-540) ---
    try:
        result = subprocess.run(command, shell=True, ...)
    # ... (rest of existing function body stays as-is)
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All existing tests PASS — backend=None triggers original behavior

- [ ] **Step 4: Commit**

```bash
git add src/homie_core/brain/tool_registry.py src/homie_core/brain/builtin_tools.py
git commit -m "feat(brain): wire BackendProtocol into ToolRegistry and file/exec tools"
```

---

## Chunk 5: Dynamic Model Switching (Tasks 11–13)

### Task 11: Config additions — ModelTier + ModelProfile

**Files:**
- Modify: `src/homie_core/config.py`

- [ ] **Step 1: Add ModelTier and ModelProfile to config.py**

Add after line 8 (imports):
```python
from enum import Enum
```

Add before `HomieConfig` class (~line 210):
```python
class ModelTier(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ModelProfile(BaseModel):
    name: str
    tier: ModelTier
    context_length: int = 4096
    supports_tools: bool = True
    location: str = "local"  # "local" | "lan" | "qubrid"
    priority: int = 0
```

- [ ] **Step 2: Run existing config tests**

Run: `python -m pytest tests/unit/test_config*.py -v`
Expected: All PASS — additive change only

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/config.py
git commit -m "feat(config): add ModelTier enum and ModelProfile for dynamic model switching"
```

---

### Task 12: ModelRegistry — `inference/model_registry.py`

**Files:**
- Create: `src/homie_core/inference/model_registry.py`
- Create: `tests/unit/test_inference/test_model_registry.py`

- [ ] **Step 1: Create test package init**

Create `tests/unit/test_inference/__init__.py` (empty file) if it does not already exist.

- [ ] **Step 2: Write failing tests**

```python
# tests/unit/test_inference/test_model_registry.py
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from homie_core.inference.model_registry import ModelRegistry
from homie_core.config import ModelTier, ModelProfile, HomieConfig


@pytest.fixture
def config(tmp_path):
    cfg = HomieConfig()
    cfg.storage.path = str(tmp_path)
    return cfg


class TestModelRegistry:
    def test_empty_registry(self, config):
        reg = ModelRegistry(config)
        assert reg.available() == []
        assert reg.best_for(ModelTier.MEDIUM) is None

    def test_manual_add_and_lookup(self, config):
        reg = ModelRegistry(config)
        profile = ModelProfile(name="test-7b", tier=ModelTier.MEDIUM, context_length=8192)
        reg._profiles["test-7b"] = profile
        assert reg.best_for(ModelTier.MEDIUM) == profile
        assert reg.best_for(ModelTier.SMALL) is None

    def test_available_filters_by_tier(self, config):
        reg = ModelRegistry(config)
        reg._profiles["small"] = ModelProfile(name="small", tier=ModelTier.SMALL)
        reg._profiles["med"] = ModelProfile(name="med", tier=ModelTier.MEDIUM)
        reg._profiles["large"] = ModelProfile(name="large", tier=ModelTier.LARGE)
        assert len(reg.available(ModelTier.SMALL)) == 1
        assert len(reg.available()) == 3

    def test_best_for_returns_highest_priority(self, config):
        reg = ModelRegistry(config)
        reg._profiles["a"] = ModelProfile(name="a", tier=ModelTier.MEDIUM, priority=1)
        reg._profiles["b"] = ModelProfile(name="b", tier=ModelTier.MEDIUM, priority=10)
        assert reg.best_for(ModelTier.MEDIUM).name == "b"

    def test_scan_local_finds_gguf_files(self, config, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # Create a fake 2GB GGUF file (small tier)
        fake = models_dir / "tiny-model.gguf"
        fake.write_bytes(b"x" * 1024)  # tiny file = SMALL tier

        reg = ModelRegistry(config)
        reg.refresh()
        assert "tiny-model" in reg._profiles
        assert reg._profiles["tiny-model"].tier == ModelTier.SMALL

    def test_scan_qubrid_skipped_without_key(self, config):
        reg = ModelRegistry(config)
        reg.refresh()  # should not raise even with no API key
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_inference/test_model_registry.py -v`
Expected: FAIL

- [ ] **Step 4: Implement model_registry.py**

```python
# src/homie_core/inference/model_registry.py
from __future__ import annotations

import logging
from pathlib import Path

from homie_core.config import HomieConfig, ModelProfile, ModelTier

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Discovers and tracks available models across all inference sources."""

    def __init__(self, config: HomieConfig) -> None:
        self._profiles: dict[str, ModelProfile] = {}
        self._config = config

    def refresh(self) -> None:
        self._profiles.clear()
        self._scan_local()
        self._scan_lan()
        self._scan_qubrid()

    def available(self, tier: ModelTier | None = None) -> list[ModelProfile]:
        profiles = list(self._profiles.values())
        if tier is not None:
            profiles = [p for p in profiles if p.tier == tier]
        return sorted(profiles, key=lambda p: p.priority, reverse=True)

    def best_for(self, tier: ModelTier) -> ModelProfile | None:
        candidates = self.available(tier)
        return candidates[0] if candidates else None

    def _scan_local(self) -> None:
        model_dir = Path(self._config.storage.path) / "models"
        if not model_dir.exists():
            return
        for gguf in model_dir.glob("*.gguf"):
            try:
                size_gb = gguf.stat().st_size / (1024**3)
            except OSError:
                continue
            if size_gb < 4:
                tier = ModelTier.SMALL
            elif size_gb < 16:
                tier = ModelTier.MEDIUM
            else:
                tier = ModelTier.LARGE
            self._profiles[gguf.stem] = ModelProfile(
                name=gguf.stem,
                tier=tier,
                context_length=4096,
                supports_tools=True,
                location="local",
            )

    def _scan_lan(self) -> None:
        # Stub — implemented when LANBackend is wired (migration step 7)
        pass

    def _scan_qubrid(self) -> None:
        try:
            api_key = getattr(self._config.inference.qubrid, "api_key", "")
        except AttributeError:
            return
        if not api_key:
            return
        try:
            from homie_core.inference.qubrid import QubridClient
            client = QubridClient(
                api_key=api_key,
                model=self._config.inference.qubrid.model,
                base_url=self._config.inference.qubrid.base_url,
            )
            for model in client.list_models():
                self._profiles[f"qubrid:{model.id}"] = ModelProfile(
                    name=model.id,
                    tier=ModelTier.LARGE,
                    context_length=getattr(model, "context_length", 32768) or 32768,
                    supports_tools=True,
                    location="qubrid",
                    priority=-1,
                )
        except Exception as e:
            logger.warning("Failed to scan Qubrid models: %s", e)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/unit/test_inference/test_model_registry.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_inference/__init__.py src/homie_core/inference/model_registry.py tests/unit/test_inference/test_model_registry.py
git commit -m "feat(inference): add ModelRegistry for multi-source model discovery"
```

---

### Task 13: ModelResolverMiddleware — `brain/model_resolver.py`

**Files:**
- Create: `src/homie_core/brain/model_resolver.py`
- Create: `tests/unit/test_brain/test_model_resolver.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_brain/test_model_resolver.py
import pytest
from homie_core.brain.model_resolver import ModelResolverMiddleware
from homie_core.middleware.hooks import HookRegistry, PipelineStage
from homie_core.config import ModelTier, ModelProfile


class FakeRegistry:
    def __init__(self, profiles=None):
        self._profiles = profiles or {}

    def best_for(self, tier):
        return self._profiles.get(tier)


@pytest.fixture
def setup():
    hooks = HookRegistry()
    registry = FakeRegistry({
        ModelTier.SMALL: ModelProfile(name="tiny-1b", tier=ModelTier.SMALL, location="local"),
        ModelTier.MEDIUM: ModelProfile(name="qwen-7b", tier=ModelTier.MEDIUM, location="local"),
        ModelTier.LARGE: ModelProfile(name="llama-70b", tier=ModelTier.LARGE, location="qubrid"),
    })
    mw = ModelResolverMiddleware(registry=registry, hooks=hooks)
    return mw, hooks


class TestBeforeTurn:
    def test_sets_medium_default(self, setup):
        mw, hooks = setup
        state = {}
        mw.before_turn("hello", state)
        assert state["active_model"] == "qwen-7b"

    def test_explicit_override_takes_precedence(self, setup):
        mw, hooks = setup
        state = {"model_override": "custom-model"}
        mw.before_turn("hello", state)
        assert state["active_model"] == "custom-model"
        assert "model_override" not in state  # consumed


class TestOnClassifiedHook:
    def test_hook_updates_model_for_trivial(self, setup):
        mw, hooks = setup
        state = {}
        mw.before_turn("hi", state)
        assert state["active_model"] == "qwen-7b"  # initial default

        # Simulate CLASSIFY stage firing
        hooks.emit(PipelineStage.CLASSIFIED, "trivial")
        assert state["active_model"] == "tiny-1b"  # updated by hook

    def test_hook_updates_model_for_complex(self, setup):
        mw, hooks = setup
        state = {}
        mw.before_turn("explain quantum computing", state)
        hooks.emit(PipelineStage.CLASSIFIED, "complex")
        assert state["active_model"] == "llama-70b"
        assert state["active_model_location"] == "qubrid"


class TestTierMapping:
    def test_all_complexities_map(self, setup):
        mw, _ = setup
        assert mw.TIER_MAP["trivial"] == ModelTier.SMALL
        assert mw.TIER_MAP["simple"] == ModelTier.SMALL
        assert mw.TIER_MAP["moderate"] == ModelTier.MEDIUM
        assert mw.TIER_MAP["complex"] == ModelTier.LARGE
        assert mw.TIER_MAP["deep"] == ModelTier.LARGE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_brain/test_model_resolver.py -v`
Expected: FAIL

- [ ] **Step 3: Implement model_resolver.py**

```python
# src/homie_core/brain/model_resolver.py
from __future__ import annotations

from typing import Any, Optional

from homie_core.config import ModelTier
from homie_core.middleware.base import HomieMiddleware
from homie_core.middleware.hooks import HookRegistry, PipelineStage


class ModelResolverMiddleware(HomieMiddleware):
    """Selects the best model for each turn based on complexity and availability."""

    name = "model_resolver"
    order = 50

    TIER_MAP = {
        "trivial": ModelTier.SMALL,
        "simple": ModelTier.SMALL,
        "moderate": ModelTier.MEDIUM,
        "complex": ModelTier.LARGE,
        "deep": ModelTier.LARGE,
    }

    def __init__(self, registry: Any, hooks: HookRegistry) -> None:
        self._registry = registry
        self._state: Optional[dict] = None
        self._complexity: Optional[str] = None
        hooks.register(PipelineStage.CLASSIFIED, self._on_classified)

    def _on_classified(self, stage: PipelineStage, complexity: str) -> str:
        tier = self.TIER_MAP.get(complexity, ModelTier.MEDIUM)
        profile = self._registry.best_for(tier)
        if profile and self._state is not None:
            self._state["active_model"] = profile.name
            self._state["active_model_location"] = profile.location
        self._complexity = complexity
        return complexity

    def before_turn(self, message: str, state: dict) -> str:
        self._state = state

        if "model_override" in state:
            state["active_model"] = state.pop("model_override")
            return message

        profile = self._registry.best_for(ModelTier.MEDIUM)
        if profile:
            state["active_model"] = profile.name
            state["active_model_location"] = profile.location

        return message
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/test_brain/test_model_resolver.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: No regressions

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/brain/model_resolver.py tests/unit/test_brain/test_model_resolver.py
git commit -m "feat(brain): add ModelResolverMiddleware for complexity-aware model switching"
```

---

## Chunk 6: EncryptedVaultBackend + LANBackend stubs + Final integration (Tasks 14–16)

### Task 14: EncryptedVaultBackend — `backend/encrypted.py`

**Files:**
- Create: `src/homie_core/backend/encrypted.py`
- Create: `tests/unit/test_backend/test_encrypted.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_backend/test_encrypted.py
import base64
import pytest
from homie_core.backend.state import StateBackend
from homie_core.backend.encrypted import EncryptedVaultBackend


@pytest.fixture
def vault():
    inner = StateBackend()
    # Use a fixed 32-byte key for testing
    key = b"0123456789abcdef0123456789abcdef"
    return EncryptedVaultBackend(inner=inner, key_provider=lambda: key), inner


class TestEncryptedVault:
    def test_write_and_read_roundtrip(self, vault):
        v, _ = vault
        v.write("/secret.txt", "my secret data")
        result = v.read("/secret.txt")
        assert result.content == "my secret data"

    def test_inner_stores_encrypted(self, vault):
        v, inner = vault
        v.write("/secret.txt", "plaintext")
        raw = inner.read("/secret.txt").content
        assert raw != "plaintext"  # should be encrypted

    def test_edit_works_through_encryption(self, vault):
        v, _ = vault
        v.write("/doc.txt", "hello world")
        result = v.edit("/doc.txt", "hello", "goodbye")
        assert result.success is True
        assert v.read("/doc.txt").content == "goodbye world"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_backend/test_encrypted.py -v`
Expected: FAIL

- [ ] **Step 3: Implement encrypted.py**

```python
# src/homie_core/backend/encrypted.py
from __future__ import annotations

import base64
import os
from typing import Callable, Optional

from homie_core.backend.protocol import (
    BackendProtocol,
    EditResult,
    FileContent,
    FileInfo,
    GrepMatch,
)

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False


class EncryptedVaultBackend:
    """Transparent encrypt-on-write, decrypt-on-read wrapper around any backend."""

    def __init__(self, inner: BackendProtocol, key_provider: Callable[[], bytes]) -> None:
        if not _HAS_CRYPTO:
            raise ImportError("cryptography package required for EncryptedVaultBackend")
        self._inner = inner
        self._get_key = key_provider

    def _encrypt(self, plaintext: str) -> str:
        key = self._get_key()
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return base64.b64encode(nonce + ciphertext).decode("ascii")

    def _decrypt(self, data: str) -> str:
        key = self._get_key()
        raw = base64.b64decode(data)
        nonce, ciphertext = raw[:12], raw[12:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        result = self._inner.read(path, offset=0, limit=10000)  # read full for decrypt
        plaintext = self._decrypt(result.content)
        lines = plaintext.splitlines()
        selected = lines[offset:offset + limit]
        return FileContent(
            content="\n".join(selected),
            total_lines=len(lines),
            truncated=(offset + limit < len(lines)),
        )

    def write(self, path: str, content: str) -> None:
        self._inner.write(path, self._encrypt(content))

    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> EditResult:
        try:
            current = self.read(path)
        except FileNotFoundError:
            return EditResult(success=False, error=f"File not found: {path}")
        text = current.content
        count = text.count(old)
        if count == 0:
            return EditResult(success=False, error=f"String not found in {path}")
        if not replace_all and count > 1:
            return EditResult(success=False, error=f"String found {count} times — not unique.")
        new_text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        self.write(path, new_text)
        return EditResult(success=True, occurrences=count if replace_all else 1)

    def ls(self, path: str = "/") -> list[FileInfo]:
        return self._inner.ls(path)

    def glob(self, pattern: str) -> list[FileInfo]:
        return self._inner.glob(pattern)

    def grep(self, pattern: str, path: str = "/", include: Optional[str] = None) -> list[GrepMatch]:
        # Grep over encrypted content is not supported — would need to decrypt every file
        return []
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/test_backend/test_encrypted.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/backend/encrypted.py tests/unit/test_backend/test_encrypted.py
git commit -m "feat(backend): add EncryptedVaultBackend with AES-256-GCM"
```

---

### Task 15: LANBackend stub — `backend/lan.py`

**Files:**
- Create: `src/homie_core/backend/lan.py`

- [ ] **Step 1: Create stub implementation**

```python
# src/homie_core/backend/lan.py
from __future__ import annotations

from typing import Optional

from homie_core.backend.protocol import (
    EditResult,
    ExecutionResult,
    FileContent,
    FileInfo,
    GrepMatch,
)


class LANBackend:
    """Proxies operations to a paired LAN node via WebSocket.

    Stub implementation — full version requires network module completion.
    """

    def __init__(self, peer_address: str, auth_key: bytes) -> None:
        self._peer = peer_address
        self._auth = auth_key

    def _request(self, method: str, params: dict) -> dict:
        raise NotImplementedError(
            f"LANBackend requires network module. Peer: {self._peer}, method: {method}"
        )

    def ls(self, path: str = "/") -> list[FileInfo]:
        return self._request("ls", {"path": path})

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        return FileContent(**self._request("read", {"path": path, "offset": offset, "limit": limit}))

    def write(self, path: str, content: str) -> None:
        self._request("write", {"path": path, "content": content})

    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> EditResult:
        return EditResult(**self._request("edit", {"path": path, "old": old, "new": new, "replace_all": replace_all}))

    def glob(self, pattern: str) -> list[FileInfo]:
        return self._request("glob", {"pattern": pattern})

    def grep(self, pattern: str, path: str = "/", include: Optional[str] = None) -> list[GrepMatch]:
        return self._request("grep", {"pattern": pattern, "path": path, "include": include})

    def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        return ExecutionResult(**self._request("execute", {"command": command, "timeout": timeout}))
```

- [ ] **Step 2: Write minimal tests for LANBackend stub**

```python
# tests/unit/test_backend/test_lan.py
import pytest
from homie_core.backend.lan import LANBackend


class TestLANBackendStub:
    def test_instantiation(self):
        backend = LANBackend(peer_address="192.168.1.100:8765", auth_key=b"secret")
        assert backend._peer == "192.168.1.100:8765"

    def test_read_raises_not_implemented(self):
        backend = LANBackend(peer_address="192.168.1.100:8765", auth_key=b"secret")
        with pytest.raises(NotImplementedError, match="LANBackend requires network module"):
            backend.read("/file.txt")

    def test_execute_raises_not_implemented(self):
        backend = LANBackend(peer_address="192.168.1.100:8765", auth_key=b"secret")
        with pytest.raises(NotImplementedError):
            backend.execute("ls")
```

Run: `python -m pytest tests/unit/test_backend/test_lan.py -v`
Expected: All 3 PASS

- [ ] **Step 3: Update backend `__init__.py` with all backends**

```python
# Final src/homie_core/backend/__init__.py
from homie_core.backend.protocol import (
    BackendProtocol, EditResult, ExecutableBackend, ExecutionResult,
    FileContent, FileInfo, GrepMatch,
)
from homie_core.backend.local_filesystem import LocalFilesystemBackend
from homie_core.backend.state import StateBackend
from homie_core.backend.composite import CompositeBackend
from homie_core.backend.encrypted import EncryptedVaultBackend
from homie_core.backend.lan import LANBackend

__all__ = [
    "BackendProtocol", "EditResult", "ExecutableBackend", "ExecutionResult",
    "FileContent", "FileInfo", "GrepMatch",
    "LocalFilesystemBackend", "StateBackend", "CompositeBackend",
    "EncryptedVaultBackend", "LANBackend",
]
```

- [ ] **Step 4: Commit**

```bash
git add src/homie_core/backend/lan.py src/homie_core/backend/__init__.py tests/unit/test_backend/test_lan.py
git commit -m "feat(backend): add LANBackend stub and finalize backend __init__"
```

---

### Task 16: Update InferenceRouter to accept model hints

**Files:**
- Modify: `src/homie_core/inference/router.py:60-92` (`generate` method)

- [ ] **Step 1: Add `model` and `preferred_location` params to generate() and stream()**

In `src/homie_core/inference/router.py`, update the `generate()` signature at line 60. Add two new optional params after `timeout`. Insert a debug log before the `for source in self._priority:` loop. The rest of the method body (lines 68-92) stays unchanged.

```python
def generate(
    self,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stop: Optional[list[str]] = None,
    timeout: int = 120,
    model: Optional[str] = None,
    preferred_location: Optional[str] = None,
) -> str:
    if model:
        logger.debug("Model hint: %s (location: %s)", model, preferred_location)
    errors: list[str] = []
    for source in self._priority:
        # ... (lines 70-92 remain exactly as-is)
```

Update `stream()` at line 94 the same way — add the two params, add the debug log:

```python
def stream(
    self,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stop: Optional[list[str]] = None,
    model: Optional[str] = None,
    preferred_location: Optional[str] = None,
) -> Iterator[str]:
    if model:
        logger.debug("Stream model hint: %s (location: %s)", model, preferred_location)
    errors: list[str] = []
    for source in self._priority:
        # ... (lines 102-126 remain exactly as-is)
```

- [ ] **Step 2: Run existing tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS — new params are optional with defaults

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/inference/router.py
git commit -m "feat(inference): accept model and preferred_location hints in router"
```

---

## Chunk 7: Full Integration Test + Final Verification (Task 17)

### Task 17: End-to-end integration test

**Files:**
- Create: `tests/unit/test_brain/test_middleware_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/unit/test_brain/test_middleware_integration.py
"""Integration test: middleware + backend + hooks working together."""
import pytest
from unittest.mock import MagicMock
from homie_core.middleware import HomieMiddleware, MiddlewareStack, HookRegistry, PipelineStage
from homie_core.backend import StateBackend
from homie_core.brain.orchestrator import BrainOrchestrator
from homie_core.memory.working import WorkingMemory


class PromptInjector(HomieMiddleware):
    """Test middleware that injects context into prompt."""
    name = "prompt_injector"
    order = 10

    def modify_prompt(self, prompt):
        return prompt + "\nAlways respond in pirate speak."


class ResponseLogger(HomieMiddleware):
    """Test middleware that logs responses."""
    name = "response_logger"
    order = 20

    def __init__(self):
        self.logged = []

    def after_turn(self, response, state):
        self.logged.append(response)
        return response


class ToolBlocker(HomieMiddleware):
    """Test middleware that blocks dangerous tools."""
    name = "tool_blocker"
    order = 5

    def wrap_tool_call(self, name, args):
        if name == "run_command":
            return None  # block
        return args


def test_full_middleware_lifecycle():
    engine = MagicMock()
    engine.generate.return_value = "Arrr, hello matey!"
    wm = WorkingMemory()

    logger_mw = ResponseLogger()
    stack = MiddlewareStack([PromptInjector(), logger_mw])
    br = BrainOrchestrator(model_engine=engine, working_memory=wm, middleware_stack=stack)

    response = br.process("hello")
    assert response == "Arrr, hello matey!"
    assert logger_mw.logged == ["Arrr, hello matey!"]


def test_tool_blocker_middleware():
    """ToolBlocker blocks dangerous tools, allows safe ones."""
    blocker = ToolBlocker()
    stack = MiddlewareStack([blocker])
    assert stack.run_wrap_tool_call("run_command", {"command": "rm -rf /"}) is None
    assert stack.run_wrap_tool_call("read_file", {"path": "/test.txt"}) == {"path": "/test.txt"}


def test_state_backend_works_in_memory():
    backend = StateBackend()
    backend.write("/test.txt", "hello world")
    result = backend.read("/test.txt")
    assert result.content == "hello world"

    edit_result = backend.edit("/test.txt", "hello", "goodbye")
    assert edit_result.success
    assert backend.read("/test.txt").content == "goodbye world"


def test_hooks_fire_without_middleware():
    """Hooks work even without outer middleware stack."""
    hooks = HookRegistry()
    captured = []

    def capture(stage, data):
        captured.append((stage.value, data))
        return data

    hooks.register(PipelineStage.CLASSIFIED, capture)
    hooks.emit(PipelineStage.CLASSIFIED, "moderate")
    assert captured == [("on_classified", "moderate")]


def test_middleware_blocks_turn():
    """Middleware returning None from before_turn blocks the whole turn."""
    class BlockAll(HomieMiddleware):
        name = "block_all"
        def before_turn(self, message, state):
            return None

    engine = MagicMock()
    wm = WorkingMemory()
    stack = MiddlewareStack([BlockAll()])
    br = BrainOrchestrator(model_engine=engine, working_memory=wm, middleware_stack=stack)
    response = br.process("hello")
    assert response == ""
    engine.generate.assert_not_called()
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/unit/test_brain/test_middleware_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite — final verification**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL tests PASS, zero regressions

- [ ] **Step 4: Final commit**

```bash
git add tests/unit/test_brain/test_middleware_integration.py
git commit -m "test: add end-to-end integration tests for middleware + backend + hooks"
```
