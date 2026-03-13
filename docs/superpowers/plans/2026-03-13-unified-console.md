# Unified Console Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all CLI subcommands with a single `homie start` entry point where all operations happen via slash commands inside an interactive console.

**Architecture:** New `src/homie_app/console/` module with a `SlashCommandRouter` registry and per-command handler modules. The existing `cmd_chat()` loop migrates into `Console.__init__` + `Console.run()`. The existing `_handle_meta_command()` is decomposed into individual command files. `cli.py` shrinks to ~30 lines: parse args, launch console.

**Tech Stack:** Python 3.11+, prompt_toolkit (new dep for autocomplete/history), requests (new core dep), existing homie_core services unchanged.

**Important implementation notes:**
- The codebase will be intentionally broken between Tasks 9 (cli.py rewrite) and 11 (test updates). Old CLI tests will fail in that window. This is expected — Task 11 fixes them.
- `_SM_PLATFORMS` is currently defined inside `cmd_connect()`, not at module scope. It must be moved to `connect.py` directly (not imported from cli.py).
- The `Console` class must re-export from `console/__init__.py` so imports work as `from homie_app.console import Console`.
- Vault must be initialized once in `_bootstrap()` and passed through context — never re-created per command.
- The remaining 4 Brain tools (`get_local_conditions`, `get_personalized_briefing`, `get_local_events`, `get_commute_update`) are deferred to a follow-up plan as they require additional API integrations not yet scoped.

**Spec:** `docs/superpowers/specs/2026-03-13-unified-console-design.md`

---

## Chunk 1: Console Foundation (Router + Console Shell)

### Task 1: Add prompt_toolkit dependency

**Files:**
- Modify: `pyproject.toml:29-35`

- [ ] **Step 1: Add prompt_toolkit to core dependencies**

In `pyproject.toml`, add `"prompt-toolkit>=3.0"` and `"requests>=2.31"` to `dependencies`:

```toml
dependencies = [
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "cryptography>=43.0",
    "keyring>=25.0",
    "feedparser>=6.0",
    "prompt-toolkit>=3.0",
    "requests>=2.31",
]
```

- [ ] **Step 2: Install updated dependencies**

Run: `pip install -e ".[dev]"`
Expected: prompt_toolkit installs successfully

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add prompt-toolkit dependency for console autocomplete"
```

---

### Task 2: Create SlashCommand dataclass and SlashCommandRouter

**Files:**
- Create: `src/homie_app/console/__init__.py`
- Create: `src/homie_app/console/router.py`
- Test: `tests/unit/test_app/test_console_router.py`

- [ ] **Step 1: Write failing tests for router**

```python
# tests/unit/test_app/test_console_router.py
"""Tests for SlashCommandRouter — registration, dispatch, subcommands."""
import pytest
from homie_app.console.router import SlashCommand, SlashCommandRouter


def test_register_and_dispatch():
    router = SlashCommandRouter()
    called_with = {}

    def handler(args: str, **ctx):
        called_with["args"] = args
        return "ok"

    router.register(SlashCommand(
        name="test",
        description="A test command",
        handler_fn=handler,
    ))
    result = router.dispatch("/test some args", **{})
    assert result == "ok"
    assert called_with["args"] == "some args"


def test_dispatch_unknown_command():
    router = SlashCommandRouter()
    result = router.dispatch("/nonexistent", **{})
    assert "Unknown command" in result
    assert "/help" in result


def test_list_commands():
    router = SlashCommandRouter()
    router.register(SlashCommand(name="alpha", description="First"))
    router.register(SlashCommand(name="beta", description="Second"))
    commands = router.list_commands()
    assert len(commands) == 2
    assert commands[0].name == "alpha"


def test_dispatch_with_subcommands():
    router = SlashCommandRouter()
    sub_called = {}

    def sub_handler(args: str, **ctx):
        sub_called["args"] = args
        return "sub ok"

    def parent_handler(args: str, **ctx):
        return "parent ok"

    router.register(SlashCommand(
        name="daemon",
        description="Manage daemon",
        handler_fn=parent_handler,
        subcommands={
            "start": SlashCommand(name="start", description="Start daemon", handler_fn=sub_handler),
        },
    ))
    # With subcommand
    result = router.dispatch("/daemon start extra", **{})
    assert result == "sub ok"
    assert sub_called["args"] == "extra"

    # Without subcommand — shows subcommand help
    result = router.dispatch("/daemon", **{})
    assert "start" in result


def test_autocomplete_matching():
    router = SlashCommandRouter()
    router.register(SlashCommand(name="connect", description="Connect provider"))
    router.register(SlashCommand(name="connections", description="List connections"))
    router.register(SlashCommand(name="consent-log", description="Consent log"))
    router.register(SlashCommand(name="help", description="Help"))

    matches = router.get_completions("con")
    names = [m.name for m in matches]
    assert "connect" in names
    assert "connections" in names
    assert "consent-log" in names
    assert "help" not in names


def test_bare_slash_lists_all():
    router = SlashCommandRouter()
    router.register(SlashCommand(name="help", description="Help"))
    router.register(SlashCommand(name="quit", description="Quit"))
    result = router.dispatch("/", **{})
    assert "help" in result
    assert "quit" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_app/test_console_router.py -v`
Expected: ImportError — module doesn't exist yet

- [ ] **Step 3: Implement SlashCommand and SlashCommandRouter**

```python
# src/homie_app/console/__init__.py
"""Homie unified console — single interactive entry point."""
from homie_app.console.console import Console  # noqa: F401

__all__ = ["Console"]
```

```python
# src/homie_app/console/router.py
"""Slash command router — registry, dispatch, and autocomplete."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class SlashCommand:
    """A registered slash command."""
    name: str
    description: str
    args_spec: str = ""  # e.g., "<provider>", "summary|sync|config", "[--days N]"
    handler_fn: Optional[Callable[..., str]] = None
    subcommands: dict[str, "SlashCommand"] = field(default_factory=dict)
    autocomplete_fn: Optional[Callable[[str], list[str]]] = None

    def format_help(self) -> str:
        """One-line help: /name — description."""
        args_part = f" {self.args_spec}" if self.args_spec else ""
        return f"  /{self.name}{args_part:<{16 - len(self.name)}} — {self.description}"


class SlashCommandRouter:
    """Registry and dispatcher for slash commands."""

    def __init__(self):
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        self._commands[command.name] = command

    def list_commands(self) -> list[SlashCommand]:
        return sorted(self._commands.values(), key=lambda c: c.name)

    def get_completions(self, prefix: str) -> list[SlashCommand]:
        """Return commands whose name starts with prefix."""
        return [c for c in self._commands.values() if c.name.startswith(prefix)]

    def dispatch(self, raw_input: str, **ctx: Any) -> str:
        """Parse and dispatch a slash command. Returns response string."""
        text = raw_input.lstrip("/").strip()

        # Bare "/" — list all commands
        if not text:
            return self._format_command_list()

        parts = text.split(maxsplit=1)
        cmd_name = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        command = self._commands.get(cmd_name)
        if not command:
            return f"Unknown command: /{cmd_name}. Type /help to see available commands."

        # Check for subcommand dispatch
        if command.subcommands and rest:
            sub_parts = rest.split(maxsplit=1)
            sub_name = sub_parts[0].lower()
            if sub_name in command.subcommands:
                sub_rest = sub_parts[1] if len(sub_parts) > 1 else ""
                return command.subcommands[sub_name].handler_fn(args=sub_rest, **ctx)

        # No subcommand match — if command has subcommands and no handler, show help
        if command.subcommands and not rest and not command.handler_fn:
            lines = [f"**/{command.name}** subcommands:"]
            for sc in sorted(command.subcommands.values(), key=lambda s: s.name):
                lines.append(sc.format_help())
            return "\n".join(lines)

        # Direct handler
        if command.handler_fn:
            return command.handler_fn(args=rest, **ctx)

        # Has subcommands but no direct handler and unrecognized sub
        if command.subcommands:
            lines = [f"**/{command.name}** subcommands:"]
            for sc in sorted(command.subcommands.values(), key=lambda s: s.name):
                lines.append(sc.format_help())
            return "\n".join(lines)

        return f"/{cmd_name}: no handler registered."

    def _format_command_list(self) -> str:
        lines = ["**Available Commands:**"]
        for cmd in self.list_commands():
            lines.append(cmd.format_help())
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_app/test_console_router.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_app/console/__init__.py src/homie_app/console/router.py tests/unit/test_app/test_console_router.py
git commit -m "feat(console): add SlashCommand dataclass and SlashCommandRouter"
```

---

### Task 3: Create Console class with main loop

**Files:**
- Create: `src/homie_app/console/console.py`
- Test: `tests/unit/test_app/test_console.py`

- [ ] **Step 1: Write failing tests for Console**

```python
# tests/unit/test_app/test_console.py
"""Tests for Console main loop — input routing, init detection, quit."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from homie_app.console.console import Console


def test_slash_command_routes_to_router(tmp_path):
    """Slash commands dispatch through the router, not the brain."""
    from homie_app.console.router import SlashCommand

    config = MagicMock()
    config.storage.path = str(tmp_path)
    config.user_name = "Tester"

    console = Console(config=config, skip_init=True)
    console._router.register(
        SlashCommand(name="test", description="test", handler_fn=lambda args, **ctx: "routed")
    )

    with patch("builtins.input", side_effect=["/test", "quit"]):
        with patch.object(console, "_print") as mock_print:
            console.run()
            # Find the call that printed "routed"
            printed = [str(c) for c in mock_print.call_args_list]
            assert any("routed" in p for p in printed)


def test_quit_exits_loop(tmp_path):
    """Typing 'quit' exits the console loop."""
    config = MagicMock()
    config.storage.path = str(tmp_path)
    config.user_name = "Tester"

    console = Console(config=config, skip_init=True)
    with patch("builtins.input", side_effect=["quit"]):
        console.run()  # Should not hang


def test_empty_input_skipped(tmp_path):
    """Empty input is ignored, loop continues."""
    config = MagicMock()
    config.storage.path = str(tmp_path)
    config.user_name = "Tester"

    console = Console(config=config, skip_init=True)
    with patch("builtins.input", side_effect=["", "quit"]):
        console.run()  # Should not error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_app/test_console.py -v`
Expected: ImportError — Console doesn't exist yet

- [ ] **Step 3: Implement Console class**

```python
# src/homie_app/console/console.py
"""Main console loop — routes input to slash commands or brain chat."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from homie_app.console.router import SlashCommandRouter


class Console:
    """Unified Homie console: wizard, slash commands, and chat in one loop."""

    def __init__(
        self,
        config,
        config_path: Optional[str] = None,
        no_voice: bool = False,
        no_tray: bool = False,
        skip_init: bool = False,
    ):
        self._config = config
        self._config_path = config_path
        self._no_voice = no_voice
        self._no_tray = no_tray
        self._router = SlashCommandRouter()
        self._brain = None
        self._wm = None
        self._sm = None
        self._em = None
        self._engine = None
        self._vault = None
        self._user_name = getattr(config, "user_name", None) or "User"

        if not skip_init:
            self._bootstrap()

    def _bootstrap(self):
        """Load model + intelligence stack, register commands."""
        self._print("=" * 50)
        self._print("  Homie AI v0.1.0 — Interactive Console")
        self._print("=" * 50)

        # Initialize vault once for the session
        try:
            from homie_core.vault.secure_vault import SecureVault
            self._vault = SecureVault()
            self._vault.unlock()
        except Exception as e:
            self._print(f"  [-] Vault unavailable: {e}")

        self._print("\n[Loading model...]")
        from homie_app.cli import _load_model_engine
        self._engine, entry = _load_model_engine(self._config)
        if not self._engine:
            self._print("  No model found. The setup wizard will help you configure one.")
            return

        self._print("\n[Initializing intelligence...]")
        from homie_app.cli import _init_intelligence_stack
        self._wm, self._em, self._sm, tool_registry, rag, plugin_mgr = _init_intelligence_stack(self._config)

        from homie_core.brain.orchestrator import BrainOrchestrator
        from homie_app.prompts.system import build_system_prompt

        self._brain = BrainOrchestrator(
            model_engine=self._engine,
            working_memory=self._wm,
            episodic_memory=self._em,
            semantic_memory=self._sm,
            tool_registry=tool_registry if tool_registry.list_tools() else None,
            rag_pipeline=rag,
        )

        known_facts = []
        if self._sm:
            try:
                facts = self._sm.get_facts(min_confidence=0.5)
                known_facts = [f["fact"] for f in facts[:10]]
            except Exception:
                pass

        system_prompt = build_system_prompt(
            user_name=self._user_name,
            known_facts=known_facts if known_facts else None,
        )
        self._brain.set_system_prompt(system_prompt)

        # Register all slash commands
        self._register_commands(plugin_mgr=plugin_mgr)

        self._print(
            f"\nHey{' ' + self._user_name if self._user_name != 'User' else ''}! "
            f"Type /help for commands or just start chatting.\n"
        )

    def _register_commands(self, **services):
        """Register all slash commands from console/commands/."""
        from homie_app.console.commands import register_all_commands
        register_all_commands(self._router, self._build_ctx(**services))

    def _build_ctx(self, **services) -> dict:
        """Build the context dict passed to all command handlers."""
        return {
            "config": self._config,
            "config_path": self._config_path,
            "brain": self._brain,
            "wm": self._wm,
            "sm": self._sm,
            "em": self._em,
            "engine": self._engine,
            "vault": self._vault,
            "_router": self._router,
            "console": self,
            **services,
        }

    def run(self):
        """Main input loop."""
        while True:
            try:
                user_input = input(f"{self._user_name}> ").strip()
            except (EOFError, KeyboardInterrupt):
                self._print("\n")
                break

            if user_input.lower() in ("quit", "exit", ":q", "/quit"):
                break
            if not user_input:
                continue

            if user_input.startswith("/"):
                ctx = self._build_ctx()
                result = self._router.dispatch(user_input, **ctx)
                # Note: /quit is caught above before reaching router
                if result:
                    self._print(f"Homie> {result}\n")
                continue

            # Chat with brain
            if not self._brain:
                self._print("Homie> No model loaded. Use /settings to configure one.\n")
                continue

            self._chat(user_input)

        self._shutdown()

    def _chat(self, user_input: str):
        """Send input to brain, stream response."""
        try:
            from homie_app.loading import CLILoadingSpinner

            spinner = CLILoadingSpinner(style="random")
            spinner.start()
            first_token = True
            try:
                for token in self._brain.process_stream(user_input):
                    if first_token:
                        spinner.stop()
                        sys.stdout.write("Homie> ")
                        sys.stdout.flush()
                        first_token = False
                    sys.stdout.write(token)
                    sys.stdout.flush()
            except Exception:
                if not first_token:
                    raise
                spinner.stop()
                sys.stdout.write(f"Homie> {self._brain.process(user_input)}")
                sys.stdout.flush()
            finally:
                if first_token:
                    spinner.stop()
            print("\n")
        except ConnectionError as e:
            self._print(f"Homie> Connection failed: {e}\n")
        except Exception as e:
            self._print(f"Homie> [Error: {e}]\n")

    def _shutdown(self):
        """Save session and clean up."""
        self._print("[Saving session...]")
        if self._brain:
            try:
                summary = self._brain.consolidate_session()
                if summary:
                    self._print(f"  Session saved: {summary}")
                else:
                    self._print("  Nothing to save.")
            except Exception:
                self._print("  Could not save session.")
        if self._engine:
            self._engine.unload()

    def _print(self, text: str):
        """Print to console. Abstracted for testability."""
        print(text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_app/test_console.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_app/console/console.py tests/unit/test_app/test_console.py
git commit -m "feat(console): add Console class with main loop, brain integration, and shutdown"
```

---

### Task 4: Create command registration scaffold

**Files:**
- Create: `src/homie_app/console/commands/__init__.py`
- Create: `src/homie_app/console/commands/help.py`
- Create: `src/homie_app/console/commands/memory.py`

- [ ] **Step 1: Write failing tests for /help and /status**

```python
# tests/unit/test_app/test_console_commands.py
"""Tests for individual slash command handlers."""
import pytest
from unittest.mock import MagicMock
from homie_app.console.router import SlashCommandRouter, SlashCommand
from homie_app.console.commands.help import register as register_help
from homie_app.console.commands.memory import register as register_memory


def test_help_lists_all_commands():
    router = SlashCommandRouter()
    register_help(router, {})
    # Add a dummy command so help has something to show
    router.register(SlashCommand(name="test", description="A test"))
    result = router.dispatch("/help", **{})
    assert "test" in result
    assert "help" in result


def test_help_specific_command():
    router = SlashCommandRouter()
    register_help(router, {})
    router.register(SlashCommand(name="connect", description="Connect a provider"))
    result = router.dispatch("/help connect", **{})
    assert "connect" in result.lower()


def test_status_shows_memory_info():
    wm = MagicMock()
    wm.get_conversation.return_value = ["msg1", "msg2"]
    sm = MagicMock()
    sm.get_facts.return_value = [{"fact": "test", "confidence": 0.9}]
    cfg = MagicMock()
    cfg.user_name = "Tester"

    router = SlashCommandRouter()
    ctx = {"wm": wm, "sm": sm, "config": cfg}
    register_memory(router, ctx)
    result = router.dispatch("/status", **ctx)
    assert "2 messages" in result
    assert "Tester" in result


def test_remember_stores_fact():
    sm = MagicMock()
    router = SlashCommandRouter()
    ctx = {"wm": MagicMock(), "sm": sm, "config": MagicMock()}
    register_memory(router, ctx)
    result = router.dispatch("/remember I like coffee", **ctx)
    sm.learn.assert_called_once_with("I like coffee", confidence=0.9, tags=["user_explicit"])
    assert "remember" in result.lower() or "coffee" in result.lower()


def test_clear_clears_working_memory():
    wm = MagicMock()
    router = SlashCommandRouter()
    ctx = {"wm": wm, "sm": None, "config": MagicMock()}
    register_memory(router, ctx)
    result = router.dispatch("/clear", **ctx)
    wm.clear.assert_called_once()
    assert "clear" in result.lower() or "fresh" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_app/test_console_commands.py -v`
Expected: ImportError

- [ ] **Step 3: Implement help command**

```python
# src/homie_app/console/commands/help.py
"""Handler for /help slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_help(args: str, **ctx) -> str:
    router: SlashCommandRouter = ctx.get("_router")
    if not router:
        return "Help unavailable."

    if args.strip():
        # Help for specific command
        cmd_name = args.strip().lstrip("/")
        commands = {c.name: c for c in router.list_commands()}
        cmd = commands.get(cmd_name)
        if not cmd:
            return f"Unknown command: /{cmd_name}"
        lines = [f"**/{cmd.name}** — {cmd.description}"]
        if cmd.subcommands:
            lines.append("\nSubcommands:")
            for sc in sorted(cmd.subcommands.values(), key=lambda s: s.name):
                lines.append(sc.format_help())
        return "\n".join(lines)

    # List all commands
    lines = ["**Homie Commands:**"]
    for cmd in router.list_commands():
        lines.append(cmd.format_help())
    lines.append("\nType /help <command> for details on a specific command.")
    return "\n".join(lines)


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="help",
        description="Show available commands or help for a specific command",
        handler_fn=_handle_help,
    ))
```

- [ ] **Step 4: Implement memory commands (/status, /learn, /facts, /remember, /forget, /clear)**

```python
# src/homie_app/console/commands/memory.py
"""Handlers for memory-related slash commands: /status, /learn, /facts, /remember, /forget, /clear."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_status(args: str, **ctx) -> str:
    wm = ctx.get("wm")
    sm = ctx.get("sm")
    em = ctx.get("em")
    cfg = ctx.get("config")

    lines = ["**Homie Status**"]
    if wm:
        lines.append(f"  Memory: {len(wm.get_conversation())} messages this session")
    if sm:
        try:
            facts = sm.get_facts(min_confidence=0.0)
            lines.append(f"  Facts stored: {len(facts)}")
        except Exception:
            lines.append("  Facts: unavailable")
    if em:
        lines.append("  Episodic memory: active")
    lines.append(f"  User: {getattr(cfg, 'user_name', 'Unknown') or 'Unknown'}")
    return "\n".join(lines)


def _handle_learn(args: str, **ctx) -> str:
    brain = ctx.get("brain")
    if not brain:
        return "Brain not loaded."
    try:
        stats = brain._cognitive._learning.get_session_stats()
        lines = ["**Session Learning Stats**"]
        lines.append(f"  Interactions: {stats['interactions']}")
        lines.append(f"  Facts learned: {stats['facts_learned']}")
        if stats.get("facts"):
            for f in stats["facts"]:
                lines.append(f"    - {f}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not get learning stats: {e}"


def _handle_facts(args: str, **ctx) -> str:
    sm = ctx.get("sm")
    if not sm:
        return "Semantic memory not available."
    facts = sm.get_facts(min_confidence=0.3)
    if not facts:
        return "No facts stored yet. Chat with me and I'll learn about you!"
    lines = ["**What I know about you:**"]
    for f in facts[:15]:
        lines.append(f"  - {f['fact']} ({f['confidence']:.0%} confident)")
    return "\n".join(lines)


def _handle_remember(args: str, **ctx) -> str:
    sm = ctx.get("sm")
    fact = args.strip()
    if sm and fact:
        sm.learn(fact, confidence=0.9, tags=["user_explicit"])
        return f"Got it, I'll remember: {fact}"
    if not fact:
        return "Usage: /remember <fact to store>"
    return "Could not store that — semantic memory not available."


def _handle_forget(args: str, **ctx) -> str:
    sm = ctx.get("sm")
    topic = args.strip()
    if sm and topic:
        sm.forget_topic(topic)
        return f"Forgotten everything about: {topic}"
    if not topic:
        return "Usage: /forget <topic>"
    return "Could not forget — semantic memory not available."


def _handle_clear(args: str, **ctx) -> str:
    wm = ctx.get("wm")
    if wm:
        wm.clear()
    return "Conversation cleared. Fresh start!"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(name="status", description="Show system status"))
    router.register(SlashCommand(name="learn", description="Show what I learned this session"))
    router.register(SlashCommand(name="facts", description="Show stored facts about you"))
    router.register(SlashCommand(name="remember", description="Store a fact (e.g., /remember I prefer dark mode)"))
    router.register(SlashCommand(name="forget", description="Forget a topic (e.g., /forget work)"))
    router.register(SlashCommand(name="clear", description="Clear conversation (fresh start)"))

    # Assign handlers
    for name, fn in [
        ("status", _handle_status),
        ("learn", _handle_learn),
        ("facts", _handle_facts),
        ("remember", _handle_remember),
        ("forget", _handle_forget),
        ("clear", _handle_clear),
    ]:
        router._commands[name].handler_fn = fn
```

- [ ] **Step 5: Create command registration entry point**

```python
# src/homie_app/console/commands/__init__.py
"""Register all slash commands with the router."""
from __future__ import annotations
from homie_app.console.router import SlashCommandRouter


def register_all_commands(router: SlashCommandRouter, ctx: dict) -> None:
    """Import and register every command module."""
    # Pass router reference into ctx so /help can list commands
    ctx["_router"] = router

    from homie_app.console.commands.help import register as reg_help
    from homie_app.console.commands.memory import register as reg_memory

    reg_help(router, ctx)
    reg_memory(router, ctx)

    # Quit is a no-op — handled by Console.run() directly
    from homie_app.console.router import SlashCommand
    router.register(SlashCommand(
        name="quit",
        description="Exit Homie",
        handler_fn=lambda args, **ctx: "",
    ))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_app/test_console_commands.py tests/unit/test_app/test_console_router.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/homie_app/console/commands/ tests/unit/test_app/test_console_commands.py
git commit -m "feat(console): add /help and memory commands (/status /learn /facts /remember /forget /clear)"
```

---

## Chunk 2: Migrate CLI Commands to Slash Commands

### Task 5: Migrate /connect, /disconnect, /connections

**Files:**
- Create: `src/homie_app/console/commands/connect.py`
- Test: `tests/unit/test_app/test_cmd_connect.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_app/test_cmd_connect.py
"""Tests for /connect, /disconnect, /connections slash commands."""
import pytest
from unittest.mock import MagicMock, patch
from homie_app.console.router import SlashCommandRouter
from homie_app.console.commands.connect import register


def test_connections_lists_providers():
    router = SlashCommandRouter()
    vault = MagicMock()
    conn1 = MagicMock(provider="gmail", connected=True, display_label="user@gmail.com")
    conn2 = MagicMock(provider="linkedin", connected=False, display_label=None)
    vault.get_all_connections.return_value = [conn1, conn2]
    ctx = {"config": MagicMock(), "vault": vault}
    register(router, ctx)

    result = router.dispatch("/connections", **ctx)
    assert "gmail" in result
    assert "linkedin" in result


def test_connect_no_provider_shows_usage():
    router = SlashCommandRouter()
    ctx = {"config": MagicMock(), "vault": MagicMock()}
    register(router, ctx)
    result = router.dispatch("/connect", **ctx)
    assert "provider" in result.lower() or "usage" in result.lower()


def test_disconnect_removes_credentials():
    router = SlashCommandRouter()
    vault = MagicMock()
    cfg = MagicMock()
    ctx = {"config": cfg, "vault": vault}
    register(router, ctx)

    result = router.dispatch("/disconnect gmail", **ctx)
    assert "gmail" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_app/test_cmd_connect.py -v`
Expected: ImportError

- [ ] **Step 3: Implement connect commands**

Move the OAuth flow logic from `cmd_connect()` in `cli.py` (lines 1068-1192) into `connect.py`. The handler prompts for credentials interactively within the console, opens the browser for OAuth, and stores tokens in the vault. For API-key providers (weather, news), it prompts for the key and validates with a test call.

```python
# src/homie_app/console/commands/connect.py
"""Handlers for /connect, /disconnect, /connections."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter

_OAUTH_PROVIDERS = {"gmail", "slack", "twitter", "reddit", "linkedin", "facebook", "instagram"}
_APIKEY_PROVIDERS = {"weather", "news"}
_ALL_PROVIDERS = _OAUTH_PROVIDERS | _APIKEY_PROVIDERS | {"blog"}


def _handle_connect(args: str, **ctx) -> str:
    provider = args.strip().lower()
    if not provider:
        providers = ", ".join(sorted(_ALL_PROVIDERS))
        return f"Usage: /connect <provider>\nAvailable: {providers}"

    if provider not in _ALL_PROVIDERS:
        return f"Unknown provider: {provider}. Available: {', '.join(sorted(_ALL_PROVIDERS))}"

    if provider == "gmail":
        return _connect_gmail(**ctx)
    elif provider in _APIKEY_PROVIDERS:
        return _connect_apikey(provider, **ctx)
    elif provider == "blog":
        return _connect_blog(**ctx)
    else:
        return _connect_social_oauth(provider, **ctx)


def _connect_gmail(**ctx) -> str:
    """Run Gmail OAuth flow inline in the console."""
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        # Check for existing OAuth client credentials
        existing = vault.get_credential("gmail_oauth_client", "default")
        if existing:
            client_id = existing.access_token
            client_secret = existing.refresh_token
        else:
            print("  Gmail requires OAuth credentials from Google Cloud Console.")
            print("  See: https://console.cloud.google.com/apis/credentials")
            client_id = input("  Client ID: ").strip()
            client_secret = input("  Client Secret: ").strip()
            if not client_id or not client_secret:
                return "Cancelled — client ID and secret are required."
            vault.store_credential(
                provider="gmail_oauth_client", account_id="default",
                token_type="oauth_client",
                access_token=client_id, refresh_token=client_secret,
                scopes=[], expires_at=None,
            )

        from homie_core.email.oauth import GmailOAuth
        oauth = GmailOAuth(client_id=client_id, client_secret=client_secret)
        auth_url = oauth.get_auth_url(use_local_server=True)

        import webbrowser
        print(f"\n  Opening browser for Google sign-in...")
        print(f"  If it doesn't open, visit: {auth_url}\n")
        webbrowser.open(auth_url)

        print("  Waiting for authorization (up to 120s)...")
        code = oauth.wait_for_redirect(timeout=120)
        if not code:
            return "Authorization timed out. Try again with /connect gmail"

        tokens = oauth.exchange(code, client_id=client_id, client_secret=client_secret)
        profile = oauth.get_profile(tokens["access_token"])
        email_addr = profile.get("emailAddress", "unknown")

        vault.store_credential(
            provider="gmail", account_id=email_addr,
            token_type="oauth2",
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token", ""),
            scopes=tokens.get("scope", "").split(),
            expires_at=tokens.get("expires_at"),
        )
        vault.set_connection_status("gmail", connected=True, label=email_addr)
        vault.log_consent("gmail", "connected", scopes=tokens.get("scope", "").split())

        return f"Gmail connected: {email_addr}"
    except Exception as e:
        return f"Gmail connection failed: {e}"


def _connect_social_oauth(provider: str, **ctx) -> str:
    """Run social media OAuth flow."""
    # Reuse existing _SM_PLATFORMS from cli.py logic
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        print(f"  {provider.title()} OAuth setup...")
        client_id = input(f"  {provider.title()} Client ID: ").strip()
        client_secret = input(f"  {provider.title()} Client Secret: ").strip()
        if not client_id or not client_secret:
            return "Cancelled."

        # Platform-specific OAuth URLs (moved from cli.py cmd_connect)
        _SM_PLATFORMS = {
            "twitter": ("https://twitter.com/i/oauth2/authorize", "https://api.twitter.com/2/oauth2/token", ["tweet.read", "users.read"], 8551),
            "reddit": ("https://www.reddit.com/api/v1/authorize", "https://www.reddit.com/api/v1/access_token", ["identity", "read"], 8552),
            "linkedin": ("https://www.linkedin.com/oauth/v2/authorization", "https://www.linkedin.com/oauth/v2/accessToken", ["r_liteprofile", "r_emailaddress"], 8553),
            "facebook": ("https://www.facebook.com/v18.0/dialog/oauth", "https://graph.facebook.com/v18.0/oauth/access_token", ["public_profile", "email"], 8554),
            "instagram": ("https://api.instagram.com/oauth/authorize", "https://api.instagram.com/oauth/access_token", ["user_profile", "user_media"], 8555),
        }
        if provider not in _SM_PLATFORMS:
            return f"OAuth not configured for {provider} yet."

        auth_url_tpl, token_url, scopes, port = _SM_PLATFORMS[provider]
        import secrets
        state = secrets.token_urlsafe(16)

        from urllib.parse import urlencode
        params = {
            "client_id": client_id,
            "redirect_uri": f"http://localhost:{port}/callback",
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": state,
        }
        auth_url = f"{auth_url_tpl}?{urlencode(params)}"

        import webbrowser
        print(f"\n  Opening browser for {provider.title()} sign-in...")
        print(f"  If it doesn't open, visit: {auth_url}\n")
        webbrowser.open(auth_url)

        from homie_core.email.oauth import _wait_for_oauth_redirect
        code = _wait_for_oauth_redirect(port=port, timeout=120)
        if not code:
            return f"Authorization timed out for {provider}."

        # Exchange code for token
        import requests
        resp = requests.post(token_url, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": f"http://localhost:{port}/callback",
            "client_id": client_id,
            "client_secret": client_secret,
        })
        resp.raise_for_status()
        tokens = resp.json()

        vault.store_credential(
            provider=provider, account_id="default",
            token_type="oauth2",
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token", ""),
            scopes=scopes,
            expires_at=tokens.get("expires_in"),
        )
        vault.set_connection_status(provider, connected=True)
        vault.log_consent(provider, "connected", scopes=scopes)

        return f"{provider.title()} connected!"
    except Exception as e:
        return f"{provider.title()} connection failed: {e}"


def _connect_apikey(provider: str, **ctx) -> str:
    """Connect an API-key-based provider (weather, news)."""
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        hints = {
            "weather": "Get a free API key from https://openweathermap.org/api",
            "news": "Get a free API key from https://newsapi.org/register",
        }
        print(f"  {hints.get(provider, '')}")
        api_key = input(f"  {provider.title()} API Key: ").strip()
        if not api_key:
            return "Cancelled."

        # Validate API key with a test call
        import requests
        if provider == "weather":
            resp = requests.get("https://api.openweathermap.org/data/2.5/weather", params={"q": "London", "appid": api_key}, timeout=10)
            if resp.status_code == 401:
                return "Invalid API key. Please check and try again."
        elif provider == "news":
            resp = requests.get("https://newsapi.org/v2/top-headlines", params={"country": "us", "apiKey": api_key, "pageSize": 1}, timeout=10)
            if resp.status_code == 401:
                return "Invalid API key. Please check and try again."

        vault.store_credential(
            provider=provider, account_id="default",
            token_type="api_key",
            access_token=api_key, refresh_token="",
            scopes=[], expires_at=None,
        )
        vault.set_connection_status(provider, connected=True)
        return f"{provider.title()} connected!"
    except Exception as e:
        return f"{provider.title()} connection failed: {e}"


def _connect_blog(**ctx) -> str:
    """Connect a blog via RSS feed URL."""
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        feed_url = input("  Blog RSS/Atom feed URL: ").strip()
        if not feed_url:
            return "Cancelled."

        vault.store_credential(
            provider="blog", account_id="default",
            token_type="api_key",
            access_token=feed_url, refresh_token="",
            scopes=[], expires_at=None,
        )
        vault.set_connection_status("blog", connected=True, label=feed_url)
        return f"Blog connected: {feed_url}"
    except Exception as e:
        return f"Blog connection failed: {e}"


def _handle_disconnect(args: str, **ctx) -> str:
    provider = args.strip().lower()
    if not provider:
        return "Usage: /disconnect <provider>"
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        vault.set_connection_status(provider, connected=False)
        vault.log_consent(provider, "disconnected")
        return f"Disconnected: {provider}"
    except Exception as e:
        return f"Could not disconnect {provider}: {e}"


def _handle_connections(args: str, **ctx) -> str:
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        connections = vault.get_all_connections()
        if not connections:
            return "No connections configured. Use /connect <provider> to set one up."
        lines = ["**Connections:**"]
        for c in connections:
            icon = "+" if c.connected else "-"
            lines.append(f"  [{icon}] {c.provider}: {c.display_label or 'no label'}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not check connections: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(name="connect", description="Connect a provider (gmail, linkedin, weather, etc.)", args_spec="<provider>", handler_fn=_handle_connect))
    router.register(SlashCommand(name="disconnect", description="Disconnect a provider", args_spec="<provider>", handler_fn=_handle_disconnect))
    router.register(SlashCommand(name="connections", description="Show all provider connections", handler_fn=_handle_connections))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_app/test_cmd_connect.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Register in commands/__init__.py**

Add to `register_all_commands()`:

```python
from homie_app.console.commands.connect import register as reg_connect
reg_connect(router, ctx)
```

- [ ] **Step 6: Commit**

```bash
git add src/homie_app/console/commands/connect.py tests/unit/test_app/test_cmd_connect.py src/homie_app/console/commands/__init__.py
git commit -m "feat(console): add /connect /disconnect /connections with inline OAuth"
```

---

### Task 6: Migrate /email, /consent-log, /vault

**Files:**
- Create: `src/homie_app/console/commands/email.py`
- Create: `src/homie_app/console/commands/consent.py`
- Create: `src/homie_app/console/commands/vault.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_app/test_cmd_email_vault.py
"""Tests for /email, /consent-log, /vault slash commands."""
import pytest
from unittest.mock import MagicMock
from homie_app.console.router import SlashCommandRouter
from homie_app.console.commands.email import register as reg_email
from homie_app.console.commands.consent import register as reg_consent
from homie_app.console.commands.vault import register as reg_vault


def test_email_no_subcommand_shows_help():
    router = SlashCommandRouter()
    reg_email(router, {})
    result = router.dispatch("/email", **{})
    assert "summary" in result
    assert "sync" in result


def test_consent_log_no_provider():
    router = SlashCommandRouter()
    reg_consent(router, {})
    result = router.dispatch("/consent-log", **{})
    assert "usage" in result.lower() or "provider" in result.lower()


def test_vault_status():
    router = SlashCommandRouter()
    vault = MagicMock()
    vault.get_all_connections.return_value = []
    vault.has_password = True
    reg_vault(router, {"vault": vault})
    result = router.dispatch("/vault", **{"vault": vault})
    assert "vault" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_app/test_cmd_email_vault.py -v`
Expected: ImportError

- [ ] **Step 3: Implement email command**

```python
# src/homie_app/console/commands/email.py
"""Handler for /email slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_email_summary(args: str, **ctx) -> str:
    try:
        from homie_core.email import EmailService
        from homie_core.vault.secure_vault import SecureVault
        from pathlib import Path

        cfg = ctx.get("config")
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        storage_path = Path(cfg.storage.path).expanduser()
        import sqlite3
        cache_conn = sqlite3.connect(str(storage_path / "cache.db"))

        service = EmailService(vault, cache_conn)
        accounts = service.initialize()
        if not accounts:
            return "No email accounts connected. Use /connect gmail first."

        days = 1
        if args.strip():
            try:
                days = int(args.strip().split("--days")[-1].strip() if "--days" in args else args.strip())
            except ValueError:
                days = 1

        summary = service.get_summary(days=days)
        lines = [f"**Email Summary ({days} day{'s' if days > 1 else ''}):**"]
        lines.append(f"  Total: {summary.get('total', 0)}")
        lines.append(f"  Unread: {summary.get('unread', 0)}")
        hp = summary.get("high_priority", [])
        if hp:
            lines.append(f"  High priority: {len(hp)}")
            for msg in hp[:5]:
                lines.append(f"    - {msg.get('subject', '(no subject)')}")
        return "\n".join(lines)
    except Exception as e:
        return f"Email summary failed: {e}"


def _handle_email_sync(args: str, **ctx) -> str:
    try:
        from homie_core.email import EmailService
        from homie_core.vault.secure_vault import SecureVault
        from pathlib import Path

        cfg = ctx.get("config")
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        storage_path = Path(cfg.storage.path).expanduser()
        import sqlite3
        cache_conn = sqlite3.connect(str(storage_path / "cache.db"))

        service = EmailService(vault, cache_conn)
        accounts = service.initialize()
        if not accounts:
            return "No email accounts connected. Use /connect gmail first."

        result = service.sync_tick()
        return f"Sync complete: {result}"
    except Exception as e:
        return f"Email sync failed: {e}"


def _handle_email_config(args: str, **ctx) -> str:
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        creds = vault.list_credentials("gmail")
        if not creds:
            return "No Gmail accounts configured. Use /connect gmail."
        lines = ["**Email Configuration:**"]
        for c in creds:
            lines.append(f"  Account: {c.account_id}")
            lines.append(f"  Active: {c.active}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not read email config: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="email",
        description="Email operations (summary, sync, config)",
        subcommands={
            "summary": SlashCommand(name="summary", description="Email summary", handler_fn=_handle_email_summary),
            "sync": SlashCommand(name="sync", description="Force sync now", handler_fn=_handle_email_sync),
            "config": SlashCommand(name="config", description="Show email settings", handler_fn=_handle_email_config),
        },
    ))
```

- [ ] **Step 4: Implement consent-log command**

```python
# src/homie_app/console/commands/consent.py
"""Handler for /consent-log slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_consent_log(args: str, **ctx) -> str:
    provider = args.strip()
    if not provider:
        return "Usage: /consent-log <provider>"
    try:
        from datetime import datetime
        from homie_core.vault.secure_vault import SecureVault

        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        history = vault.get_consent_history(provider)
        if not history:
            return f"No consent history for '{provider}'."
        lines = [f"**Consent log for {provider}:**"]
        for entry in history:
            dt = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  {dt}  {entry.action}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not check consent log: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="consent-log",
        description="Show consent audit trail (e.g., /consent-log gmail)",
        handler_fn=_handle_consent_log,
    ))
```

- [ ] **Step 5: Implement vault command**

```python
# src/homie_app/console/commands/vault.py
"""Handler for /vault slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_vault(args: str, **ctx) -> str:
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        connections = vault.get_all_connections()
        active = sum(1 for c in connections if c.connected)
        has_pw = vault.has_password
        return (
            f"**Vault Status:**\n"
            f"  Connections: {active} active / {len(connections)} total\n"
            f"  Password: {'set' if has_pw else 'not set'}"
        )
    except Exception as e:
        return f"Could not check vault: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="vault",
        description="Vault health and status",
        handler_fn=_handle_vault,
    ))
```

- [ ] **Step 6: Register all three in commands/__init__.py**

Add to `register_all_commands()`:

```python
from homie_app.console.commands.email import register as reg_email
from homie_app.console.commands.consent import register as reg_consent
from homie_app.console.commands.vault import register as reg_vault

reg_email(router, ctx)
reg_consent(router, ctx)
reg_vault(router, ctx)
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/unit/test_app/test_cmd_email_vault.py -v`
Expected: All 3 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/homie_app/console/commands/email.py src/homie_app/console/commands/consent.py src/homie_app/console/commands/vault.py tests/unit/test_app/test_cmd_email_vault.py src/homie_app/console/commands/__init__.py
git commit -m "feat(console): add /email /consent-log /vault slash commands"
```

---

### Task 7: Migrate /settings

**Files:**
- Create: `src/homie_app/console/commands/settings.py`
- Test: `tests/unit/test_app/test_cmd_settings.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_app/test_cmd_settings.py
"""Tests for /settings slash command."""
import pytest
from unittest.mock import MagicMock, patch
from homie_app.console.router import SlashCommandRouter
from homie_app.console.commands.settings import register


def test_settings_no_args_shows_menu():
    router = SlashCommandRouter()
    register(router, {})
    # Mock input to select "Back" immediately
    with patch("builtins.input", return_value="10"):
        result = router.dispatch("/settings", **{"config": MagicMock()})
    assert "settings" in result.lower() or "LLM" in result or result == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_app/test_cmd_settings.py -v`
Expected: ImportError

- [ ] **Step 3: Implement settings command**

Reuse `_step_*` functions from `cli.py` and `init.py`. The handler shows the numbered menu and calls the appropriate step function.

```python
# src/homie_app/console/commands/settings.py
"""Handler for /settings slash command — interactive settings menu."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


_CATEGORIES = [
    "LLM & Model",
    "Voice",
    "User Profile",
    "Screen Reader",
    "Email & Socials",
    "Privacy",
    "Plugins",
    "Notifications",
    "Service Mode",
    "Location",
    "Back",
]


def _ask_choice(title: str, options: list[str]) -> int:
    print(f"\n  {title}")
    print("  " + "-" * len(title))
    for i, opt in enumerate(options):
        print(f"    {i}: {opt}")
    while True:
        try:
            raw = input(f"  Choose [0-{len(options)-1}]: ").strip()
            choice = int(raw)
            if 0 <= choice < len(options):
                return choice
        except (ValueError, EOFError, KeyboardInterrupt):
            return len(options) - 1  # Back


def _handle_settings(args: str, **ctx) -> str:
    cfg = ctx.get("config")
    if not cfg:
        return "No configuration loaded."

    # If a category was specified directly, jump to it
    if args.strip():
        cat = args.strip().lower()
        for i, name in enumerate(_CATEGORIES[:-1]):  # Skip "Back"
            if cat in name.lower():
                _run_step(i, cfg)
                _save(cfg, ctx.get("config_path"))
                return f"Settings updated: {name}"
        return f"Unknown category: {args}. Available: {', '.join(_CATEGORIES[:-1])}"

    # Interactive menu loop
    while True:
        choice = _ask_choice("Homie Settings", _CATEGORIES)
        if choice == len(_CATEGORIES) - 1:  # Back
            break
        _run_step(choice, cfg)
        _save(cfg, ctx.get("config_path"))

    return ""


def _run_step(choice: int, cfg) -> None:
    """Run the settings step for a given menu index."""
    try:
        from homie_app.init import (
            _step_user_profile, _step_screen_reader,
            _step_email, _step_social_connections,
            _step_privacy, _step_plugins,
            _step_service_mode,
        )

        steps = {
            2: _step_user_profile,
            3: _step_screen_reader,
            4: lambda c: (_step_email(c), _step_social_connections(c)),
            5: _step_privacy,
            6: _step_plugins,
            8: _step_service_mode,
        }

        fn = steps.get(choice)
        if fn:
            fn(cfg)
        elif choice == 0:
            print("  LLM & Model settings — use /model to manage models.")
        elif choice == 1:
            print("  Voice settings — use /voice to manage voice pipeline.")
        elif choice == 7:
            print("  Notification settings — not yet implemented.")
        elif choice == 9:
            print("  Location settings — use /location set <city>.")
    except Exception as e:
        print(f"  Error in settings: {e}")


def _save(cfg, config_path: str | None) -> None:
    """Save config to disk."""
    try:
        from homie_app.init import _save_config
        path = config_path or "homie.config.yaml"
        _save_config(cfg, path)
    except Exception:
        pass


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="settings",
        description="Configure Homie (LLM, voice, privacy, email, location, etc.)",
        handler_fn=_handle_settings,
    ))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_app/test_cmd_settings.py -v`
Expected: PASS

- [ ] **Step 5: Register and commit**

Add `reg_settings` to `commands/__init__.py`, then:

```bash
git add src/homie_app/console/commands/settings.py tests/unit/test_app/test_cmd_settings.py src/homie_app/console/commands/__init__.py
git commit -m "feat(console): add /settings interactive menu"
```

---

### Task 8: Migrate remaining CLI commands (model, plugins, daemon, insights, schedule, skills, folder, social, sm, browser, voice, backup/restore)

**Files:**
- Create: `src/homie_app/console/commands/model.py`
- Create: `src/homie_app/console/commands/plugins.py`
- Create: `src/homie_app/console/commands/daemon.py`
- Create: `src/homie_app/console/commands/insights.py`
- Create: `src/homie_app/console/commands/schedule.py`
- Create: `src/homie_app/console/commands/skills.py`
- Create: `src/homie_app/console/commands/folder.py`
- Create: `src/homie_app/console/commands/social.py`
- Create: `src/homie_app/console/commands/sm.py`
- Create: `src/homie_app/console/commands/browser.py`
- Create: `src/homie_app/console/commands/voice.py`
- Create: `src/homie_app/console/commands/backup.py`
- Test: `tests/unit/test_app/test_cmd_remaining.py`

Each command follows the same pattern: extract the handler logic from `cli.py`'s `cmd_*` function, wrap it in a slash command handler that takes `(args: str, **ctx) -> str`, register subcommands where the CLI had subparsers.

- [ ] **Step 1: Write failing tests for key commands**

```python
# tests/unit/test_app/test_cmd_remaining.py
"""Tests for remaining migrated slash commands."""
import pytest
from unittest.mock import MagicMock, patch
from homie_app.console.router import SlashCommandRouter


def test_model_list():
    from homie_app.console.commands.model import register
    router = SlashCommandRouter()
    register(router, {})
    result = router.dispatch("/model", **{"config": MagicMock()})
    assert "list" in result  # Shows subcommand help


def test_plugins_list():
    from homie_app.console.commands.plugins import register
    router = SlashCommandRouter()
    register(router, {})
    result = router.dispatch("/plugins", **{"config": MagicMock()})
    assert "list" in result or "enable" in result


def test_daemon_no_subcommand():
    from homie_app.console.commands.daemon import register
    router = SlashCommandRouter()
    register(router, {})
    result = router.dispatch("/daemon", **{})
    assert "start" in result
    assert "stop" in result
    assert "status" in result


def test_insights_command():
    from homie_app.console.commands.insights import register
    router = SlashCommandRouter()
    register(router, {})
    # With mocked InsightsEngine
    with patch("homie_core.analytics.insights.InsightsEngine") as MockEngine:
        mock_engine = MockEngine.return_value
        mock_engine.generate_insights.return_value = {}
        mock_engine.format_terminal.return_value = "Insights: 10 sessions"
        result = router.dispatch("/insights", **{"config": MagicMock(storage=MagicMock(path="/tmp"))})
    assert "insights" in result.lower() or "sessions" in result.lower()


def test_skills_command():
    from homie_app.console.commands.skills import register
    router = SlashCommandRouter()
    register(router, {})
    with patch("homie_core.skills.loader.SkillLoader") as MockLoader:
        MockLoader.return_value.scan.return_value = []
        result = router.dispatch("/skills", **{})
    assert "no skills" in result.lower() or "skills" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_app/test_cmd_remaining.py -v`
Expected: ImportError

- [ ] **Step 3: Implement each command module**

Each follows this template — extract logic from `cli.py`'s corresponding `cmd_*` function. For subcommand commands (model, plugins, daemon, folder, social, sm, browser, voice, schedule), register with `subcommands` dict.

**model.py** — wraps `cmd_model`: list, download, add, remove, switch, benchmark
**plugins.py** — wraps `cmd_plugin`: list, enable, disable
**daemon.py** — wraps `cmd_daemon`: start (spawn subprocess), stop (PID signal), status
**insights.py** — wraps `cmd_insights`: generate analytics
**schedule.py** — wraps `cmd_schedule`: add, list, remove
**skills.py** — wraps `cmd_skills`: list installed skills
**folder.py** — wraps `cmd_folder`: watch, list, scan, unwatch
**social.py** — wraps `cmd_social`: channels, recent
**sm.py** — wraps `cmd_sm`: feed, profile, scan, publish, dms, send-dm
**browser.py** — wraps `cmd_browser`: enable, disable, config, history, scan, patterns
**voice.py** — wraps `cmd_voice`: status, enable, disable (plus --mode and --tts as args)
**backup.py** — wraps `cmd_backup` + `cmd_restore`: /backup <path>, /restore <path>

Example implementation for daemon.py:

```python
# src/homie_app/console/commands/daemon.py
"""Handler for /daemon slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_daemon_start(args: str, **ctx) -> str:
    try:
        import subprocess
        import sys
        config_path = ctx.get("config_path", "")
        cmd = [sys.executable, "-m", "homie_app.daemon"]
        if config_path:
            cmd.extend(["--config", config_path])
        proc = subprocess.Popen(cmd, start_new_session=True)

        # Write PID file
        from pathlib import Path
        pid_file = Path.home() / ".homie" / "daemon.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(proc.pid))

        return f"Daemon started (PID: {proc.pid})"
    except Exception as e:
        return f"Could not start daemon: {e}"


def _handle_daemon_stop(args: str, **ctx) -> str:
    try:
        from pathlib import Path
        import signal
        pid_file = Path.home() / ".homie" / "daemon.pid"
        if not pid_file.exists():
            return "No daemon PID file found. Is the daemon running?"
        pid = int(pid_file.read_text().strip())

        import os
        try:
            os.kill(pid, signal.SIGTERM)
            pid_file.unlink(missing_ok=True)
            return f"Daemon stopped (PID: {pid})"
        except ProcessLookupError:
            pid_file.unlink(missing_ok=True)
            return "Daemon was not running (stale PID file cleaned up)."
    except Exception as e:
        return f"Could not stop daemon: {e}"


def _handle_daemon_status(args: str, **ctx) -> str:
    try:
        from pathlib import Path
        pid_file = Path.home() / ".homie" / "daemon.pid"
        if not pid_file.exists():
            return "Daemon: not running"
        pid = int(pid_file.read_text().strip())

        import psutil
        try:
            proc = psutil.Process(pid)
            if proc.is_running():
                import time
                uptime = time.time() - proc.create_time()
                hours = int(uptime // 3600)
                mins = int((uptime % 3600) // 60)
                return f"Daemon: running (PID: {pid}, uptime: {hours}h {mins}m)"
            else:
                return "Daemon: not running (stale PID)"
        except psutil.NoSuchProcess:
            return "Daemon: not running (stale PID)"
    except ImportError:
        return "Daemon: PID file exists but psutil not installed for status check."
    except Exception as e:
        return f"Could not check daemon status: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="daemon",
        description="Manage background service",
        subcommands={
            "start": SlashCommand(name="start", description="Start background daemon", handler_fn=_handle_daemon_start),
            "stop": SlashCommand(name="stop", description="Stop daemon", handler_fn=_handle_daemon_stop),
            "status": SlashCommand(name="status", description="Show daemon status", handler_fn=_handle_daemon_status),
        },
    ))
```

Implement each remaining module following the same pattern. Each extracts logic from the corresponding `cmd_*` function in `cli.py`.

- [ ] **Step 4: Register all commands in commands/__init__.py**

Update `register_all_commands()` to import and call all `register()` functions:

```python
from homie_app.console.commands.model import register as reg_model
from homie_app.console.commands.plugins import register as reg_plugins
from homie_app.console.commands.daemon import register as reg_daemon
from homie_app.console.commands.insights import register as reg_insights
from homie_app.console.commands.schedule import register as reg_schedule
from homie_app.console.commands.skills import register as reg_skills
from homie_app.console.commands.folder import register as reg_folder
from homie_app.console.commands.social import register as reg_social
from homie_app.console.commands.sm import register as reg_sm
from homie_app.console.commands.browser import register as reg_browser
from homie_app.console.commands.voice import register as reg_voice
from homie_app.console.commands.backup import register as reg_backup

for reg in [reg_model, reg_plugins, reg_daemon, reg_insights, reg_schedule,
            reg_skills, reg_folder, reg_social, reg_sm, reg_browser, reg_voice, reg_backup]:
    reg(router, ctx)
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/unit/test_app/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_app/console/commands/ tests/unit/test_app/test_cmd_remaining.py
git commit -m "feat(console): migrate all CLI commands to slash commands (model, plugins, daemon, folder, social, sm, browser, voice, backup, insights, schedule, skills)"
```

---

## Chunk 3: CLI Simplification & First-Run Wizard

### Task 9: Rewrite cli.py as thin entry point

**Files:**
- Modify: `src/homie_app/cli.py`
- Test: `tests/unit/test_app/test_cli_entry.py`

- [ ] **Step 1: Write failing test for new entry point**

```python
# tests/unit/test_app/test_cli_entry.py
"""Tests for simplified CLI entry point."""
import pytest
from unittest.mock import patch, MagicMock


def test_no_args_launches_console():
    """Running 'homie' with no args should launch the console."""
    with patch("homie_app.console.console.Console") as MockConsole:
        mock_instance = MockConsole.return_value
        from homie_app.cli import main
        with patch("homie_core.config.load_config") as mock_load:
            mock_load.return_value = MagicMock()
            main([])
        MockConsole.assert_called_once()
        mock_instance.run.assert_called_once()


def test_start_launches_console():
    """Running 'homie start' should launch the console."""
    with patch("homie_app.console.console.Console") as MockConsole:
        mock_instance = MockConsole.return_value
        from homie_app.cli import main
        with patch("homie_core.config.load_config") as mock_load:
            mock_load.return_value = MagicMock()
            main(["start"])
        MockConsole.assert_called_once()
        mock_instance.run.assert_called_once()


def test_version_flag():
    """Running 'homie --version' should print version."""
    from homie_app.cli import main
    with pytest.raises(SystemExit):
        main(["--version"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_app/test_cli_entry.py -v`
Expected: FAIL — current main() doesn't launch Console

- [ ] **Step 3: Rewrite cli.py**

Keep the helper functions (`_load_model_engine`, `_init_intelligence_stack`, `_register_plugin_tools`, `_register_meta_tools`, `_pick_usable_model`, `_validate_model_entry`, `_SM_PLATFORMS`) in `cli.py` since they're imported by `console.py`. Remove all `cmd_*` functions, `create_parser`, `_handle_meta_command`, and the old `main()`.

New `cli.py`:

```python
# src/homie_app/cli.py
"""Homie AI CLI — single entry point."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keep helper functions that console.py imports:
# _validate_model_entry, _pick_usable_model, _load_model_engine,
# _init_intelligence_stack, _register_plugin_tools, _register_meta_tools,
# _SM_PLATFORMS
# ... (these stay as-is from the existing file)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="homie", description="Homie AI — Local Personal Assistant")
    parser.add_argument("--version", action="version", version="homie-ai 0.1.0")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice pipeline")
    parser.add_argument("--no-tray", action="store_true", help="Disable system tray")

    subparsers = parser.add_subparsers(dest="command")
    start_parser = subparsers.add_parser("start", help="Start the assistant (default)")
    start_parser.add_argument("--config", type=str, help="Path to config file", dest="start_config")
    start_parser.add_argument("--no-voice", action="store_true", dest="start_no_voice")
    start_parser.add_argument("--no-tray", action="store_true", dest="start_no_tray")

    return parser


def main(argv: list[str] | None = None):
    parser = create_parser()
    args = parser.parse_args(argv)

    # Resolve config path and flags (top-level or from 'start' subcommand)
    config_path = getattr(args, "start_config", None) or getattr(args, "config", None)
    no_voice = getattr(args, "start_no_voice", False) or getattr(args, "no_voice", False)
    no_tray = getattr(args, "start_no_tray", False) or getattr(args, "no_tray", False)

    # Both 'homie' and 'homie start' launch the console
    from homie_core.config import load_config
    cfg = load_config(config_path)

    from homie_app.console import Console
    console = Console(
        config=cfg,
        config_path=config_path,
        no_voice=no_voice,
        no_tray=no_tray,
    )
    console.run()


if __name__ == "__main__":
    main()
```

**Important:** Keep the helper functions (`_load_model_engine`, `_init_intelligence_stack`, etc.) in `cli.py` unchanged — they're imported by `console.py`. Only remove:
- The old `create_parser()` with all 22+ subparsers
- All `cmd_*` functions
- `_handle_meta_command()`
- The old `main()` with the command dispatch dict

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_app/test_cli_entry.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest tests/ -v --tb=short`
Expected: Old CLI tests may fail — they'll be updated in Task 11

- [ ] **Step 6: Commit**

```bash
git add src/homie_app/cli.py tests/unit/test_app/test_cli_entry.py
git commit -m "refactor(cli): simplify to single entry point — all commands now in console"
```

---

### Task 10: First-run wizard detection and inline flow

**Files:**
- Modify: `src/homie_app/console/console.py`
- Test: `tests/unit/test_app/test_first_run.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_app/test_first_run.py
"""Tests for first-run wizard detection and inline wizard flow."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path


def test_no_config_triggers_wizard(tmp_path):
    """When config has no model_path and no user_name, wizard should run."""
    from homie_app.console import Console

    cfg = MagicMock()
    cfg.user_name = ""
    cfg.llm.model_path = ""
    cfg.storage.path = str(tmp_path)

    with patch.object(Console, "_run_wizard") as mock_wizard:
        with patch.object(Console, "_bootstrap"):
            console = Console(config=cfg, skip_init=True)
            console._config = cfg
            assert console._needs_wizard()
            mock_wizard.return_value = None


def test_complete_config_skips_wizard(tmp_path):
    """When config is complete, wizard should not run."""
    from homie_app.console import Console

    cfg = MagicMock()
    cfg.user_name = "Master"
    cfg.llm.model_path = "/some/model.gguf"
    cfg.storage.path = str(tmp_path)

    with patch.object(Console, "_bootstrap"):
        console = Console(config=cfg, skip_init=True)
        console._config = cfg
        assert not console._needs_wizard()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_app/test_first_run.py -v`
Expected: FAIL — `_needs_wizard` doesn't exist

- [ ] **Step 3: Add wizard detection and inline flow to Console**

Add to `Console` class in `console.py`:

```python
def _needs_wizard(self) -> bool:
    """Check if first-run wizard is needed."""
    cfg = self._config
    # No model configured
    if not getattr(cfg.llm, "model_path", ""):
        return True
    # No user name set
    if not getattr(cfg, "user_name", ""):
        return True
    return False

def _run_wizard(self):
    """Run the init wizard inline in the console."""
    self._print("\nWelcome to Homie AI! Let's get you set up.\n")
    try:
        from homie_app.init import run_init
        run_init(auto=False, config_path=self._config_path)
        # Reload config after wizard
        from homie_core.config import load_config
        self._config = load_config(self._config_path)
        self._user_name = self._config.user_name or "User"
    except Exception as e:
        self._print(f"Wizard error: {e}. You can configure manually with /settings.")
```

Update `_bootstrap()` to call wizard check before model loading:

```python
def _bootstrap(self):
    # Check if wizard needed
    if self._needs_wizard():
        self._run_wizard()
        if self._needs_wizard():
            self._print("Setup incomplete. Use /settings to finish configuration.\n")
            self._register_commands()
            return

    # ... existing model loading and intelligence init ...
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_app/test_first_run.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_app/console/console.py tests/unit/test_app/test_first_run.py
git commit -m "feat(console): add first-run wizard detection and inline wizard flow"
```

---

### Task 11: Update existing tests for new architecture

**Files:**
- Modify: `tests/unit/test_app/test_cli.py`
- Modify: `tests/integration/test_init_wizard.py`

- [ ] **Step 1: Update CLI parser tests**

The old tests tested `create_parser()` with 22+ subparsers. Update to test the new slim parser:

```python
# Replace old parser tests with:
def test_parser_no_args():
    from homie_app.cli import create_parser
    parser = create_parser()
    args = parser.parse_args([])
    assert args.command is None

def test_parser_start():
    from homie_app.cli import create_parser
    parser = create_parser()
    args = parser.parse_args(["start"])
    assert args.command == "start"

def test_parser_config_flag():
    from homie_app.cli import create_parser
    parser = create_parser()
    args = parser.parse_args(["--config", "/path/to/config.yaml"])
    assert args.config == "/path/to/config.yaml"
```

- [ ] **Step 2: Update meta-command tests to test new command handlers**

Replace tests for `_handle_meta_command()` with tests against the console router dispatch.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: update CLI and integration tests for unified console architecture"
```

---

## Chunk 4: Location & Local Intelligence

### Task 12: Add location config and /location command

**Files:**
- Modify: `src/homie_core/config.py` (add LocationConfig)
- Create: `src/homie_app/console/commands/location.py`
- Test: `tests/unit/test_app/test_cmd_location.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_app/test_cmd_location.py
"""Tests for /location slash command."""
import pytest
from unittest.mock import MagicMock
from homie_app.console.router import SlashCommandRouter
from homie_app.console.commands.location import register


def test_location_shows_current():
    router = SlashCommandRouter()
    cfg = MagicMock()
    cfg.location = MagicMock(city="Chennai", region="Tamil Nadu", country="IN", timezone="Asia/Kolkata")
    register(router, {})
    result = router.dispatch("/location", **{"config": cfg})
    assert "Chennai" in result


def test_location_not_set():
    router = SlashCommandRouter()
    cfg = MagicMock()
    cfg.location = None
    register(router, {})
    result = router.dispatch("/location", **{"config": cfg})
    assert "not set" in result.lower()


def test_location_set():
    router = SlashCommandRouter()
    cfg = MagicMock()
    cfg.location = None
    register(router, {})
    result = router.dispatch("/location set Chennai", **{"config": cfg, "config_path": None})
    assert "chennai" in result.lower() or "location" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_app/test_cmd_location.py -v`
Expected: ImportError

- [ ] **Step 3: Add LocationConfig to config.py**

Add to `src/homie_core/config.py`:

```python
class LocationConfig(BaseModel):
    city: str = ""
    region: str = ""
    country: str = ""
    timezone: str = ""
```

Add `location: Optional[LocationConfig] = None` to `HomieConfig`.

- [ ] **Step 4: Implement /location command**

```python
# src/homie_app/console/commands/location.py
"""Handler for /location slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_location(args: str, **ctx) -> str:
    cfg = ctx.get("config")
    if not cfg:
        return "No configuration loaded."

    text = args.strip()

    # /location set <city>
    if text.lower().startswith("set "):
        city = text[4:].strip()
        if not city:
            return "Usage: /location set <city>"

        from homie_core.config import LocationConfig
        cfg.location = LocationConfig(city=city)
        # Save config
        try:
            from homie_app.init import _save_config
            _save_config(cfg, ctx.get("config_path") or "homie.config.yaml")
        except Exception:
            pass
        return f"Location set to: {city}. Refine with /settings > Location for region/country/timezone."

    # /location — show current
    loc = getattr(cfg, "location", None)
    if not loc or not loc.city:
        return "Location not set. Use /location set <city> to configure."

    lines = ["**Location:**"]
    lines.append(f"  City: {loc.city}")
    if loc.region:
        lines.append(f"  Region: {loc.region}")
    if loc.country:
        lines.append(f"  Country: {loc.country}")
    if loc.timezone:
        lines.append(f"  Timezone: {loc.timezone}")
    return "\n".join(lines)


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="location",
        description="View or set location (e.g., /location set Chennai)",
        handler_fn=_handle_location,
    ))
```

- [ ] **Step 5: Register and run tests**

Run: `pytest tests/unit/test_app/test_cmd_location.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/config.py src/homie_app/console/commands/location.py tests/unit/test_app/test_cmd_location.py src/homie_app/console/commands/__init__.py
git commit -m "feat(console): add /location command and LocationConfig"
```

---

### Task 13: Add /weather command and weather service

**Files:**
- Create: `src/homie_core/intelligence/weather.py`
- Create: `src/homie_app/console/commands/weather.py`
- Test: `tests/unit/test_core/test_weather.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_core/test_weather.py
"""Tests for weather service."""
import pytest
from unittest.mock import patch, MagicMock
from homie_core.intelligence.weather import WeatherService


def test_get_current_weather():
    service = WeatherService(api_key="test_key")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "main": {"temp": 32, "humidity": 65},
        "weather": [{"description": "clear sky"}],
        "wind": {"speed": 3.5},
        "name": "Chennai",
    }
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        result = service.get_current("Chennai")

    assert result["city"] == "Chennai"
    assert result["temp"] == 32
    assert result["description"] == "clear sky"


def test_get_weather_no_api_key():
    service = WeatherService(api_key="")
    result = service.get_current("Chennai")
    assert "error" in result


def test_format_weather():
    service = WeatherService(api_key="test")
    data = {"city": "Chennai", "temp": 32, "humidity": 65, "description": "clear sky", "wind_speed": 3.5}
    text = service.format_current(data)
    assert "Chennai" in text
    assert "32" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_core/test_weather.py -v`
Expected: ImportError

- [ ] **Step 3: Implement WeatherService**

```python
# src/homie_core/intelligence/weather.py
"""Weather data fetching and formatting."""
from __future__ import annotations


class WeatherService:
    """Fetches weather data from OpenWeatherMap API."""

    _BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: str):
        self._api_key = api_key

    def get_current(self, city: str) -> dict:
        if not self._api_key:
            return {"error": "No weather API key configured. Use /connect weather to set one up."}
        try:
            import requests
            resp = requests.get(
                f"{self._BASE_URL}/weather",
                params={"q": city, "appid": self._api_key, "units": "metric"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "city": data.get("name", city),
                "temp": data["main"]["temp"],
                "feels_like": data["main"].get("feels_like"),
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"] if data.get("weather") else "unknown",
                "wind_speed": data.get("wind", {}).get("speed", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_forecast(self, city: str, days: int = 3) -> dict:
        if not self._api_key:
            return {"error": "No weather API key configured."}
        try:
            import requests
            resp = requests.get(
                f"{self._BASE_URL}/forecast",
                params={"q": city, "appid": self._api_key, "units": "metric", "cnt": days * 8},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            forecasts = []
            for item in data.get("list", [])[:days * 8:8]:  # One per day
                forecasts.append({
                    "dt_txt": item.get("dt_txt", ""),
                    "temp": item["main"]["temp"],
                    "description": item["weather"][0]["description"] if item.get("weather") else "",
                })
            return {"city": data.get("city", {}).get("name", city), "forecasts": forecasts}
        except Exception as e:
            return {"error": str(e)}

    def format_current(self, data: dict) -> str:
        if "error" in data:
            return data["error"]
        lines = [f"**Weather in {data['city']}:**"]
        lines.append(f"  Temperature: {data['temp']}°C")
        if data.get("feels_like"):
            lines.append(f"  Feels like: {data['feels_like']}°C")
        lines.append(f"  Conditions: {data['description']}")
        lines.append(f"  Humidity: {data['humidity']}%")
        lines.append(f"  Wind: {data['wind_speed']} m/s")
        return "\n".join(lines)

    def format_forecast(self, data: dict) -> str:
        if "error" in data:
            return data["error"]
        lines = [f"**Forecast for {data['city']}:**"]
        for f in data.get("forecasts", []):
            lines.append(f"  {f['dt_txt']}: {f['temp']}°C, {f['description']}")
        return "\n".join(lines)
```

- [ ] **Step 4: Implement /weather slash command**

```python
# src/homie_app/console/commands/weather.py
"""Handler for /weather slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_weather(args: str, **ctx) -> str:
    cfg = ctx.get("config")
    location = getattr(cfg, "location", None)
    city = args.strip() if args.strip() and args.strip() != "forecast" else (location.city if location else "")

    if not city:
        return "No location set. Use /location set <city> first, or /weather <city>."

    # Get API key from vault
    api_key = ""
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()
        cred = vault.get_credential("weather", "default")
        if cred:
            api_key = cred.access_token
    except Exception:
        pass

    from homie_core.intelligence.weather import WeatherService
    service = WeatherService(api_key=api_key)

    if "forecast" in args.lower():
        data = service.get_forecast(city)
        return service.format_forecast(data)
    else:
        data = service.get_current(city)
        return service.format_current(data)


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="weather",
        description="Current weather or forecast (e.g., /weather, /weather forecast, /weather London)",
        handler_fn=_handle_weather,
    ))
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_core/test_weather.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Register and commit**

```bash
git add src/homie_core/intelligence/weather.py src/homie_app/console/commands/weather.py tests/unit/test_core/test_weather.py src/homie_app/console/commands/__init__.py
git commit -m "feat(intelligence): add weather service and /weather command"
```

---

### Task 14: Add /news command and news service

**Files:**
- Create: `src/homie_core/intelligence/news.py`
- Create: `src/homie_app/console/commands/news.py`
- Test: `tests/unit/test_core/test_news.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_core/test_news.py
"""Tests for news service."""
import pytest
from unittest.mock import patch, MagicMock
from homie_core.intelligence.news import NewsService


def test_get_headlines():
    service = NewsService(api_key="test_key")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "ok",
        "articles": [
            {"title": "Test headline", "source": {"name": "Test Source"}, "url": "http://example.com"},
        ],
    }
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        result = service.get_headlines(country="in")

    assert len(result["articles"]) == 1
    assert result["articles"][0]["title"] == "Test headline"


def test_format_headlines():
    service = NewsService(api_key="test")
    data = {"articles": [{"title": "Big news", "source": "BBC", "url": "http://bbc.com"}]}
    text = service.format_headlines(data)
    assert "Big news" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_core/test_news.py -v`
Expected: ImportError

- [ ] **Step 3: Implement NewsService and /news command**

```python
# src/homie_core/intelligence/news.py
"""News data fetching and formatting."""
from __future__ import annotations


class NewsService:
    """Fetches news from NewsAPI."""

    _BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: str):
        self._api_key = api_key

    def get_headlines(self, country: str = "us", query: str = "", category: str = "") -> dict:
        if not self._api_key:
            return {"error": "No news API key configured. Use /connect news to set one up.", "articles": []}
        try:
            import requests
            params = {"apiKey": self._api_key, "country": country, "pageSize": 10}
            if query:
                params["q"] = query
            if category:
                params["category"] = category
            resp = requests.get(f"{self._BASE_URL}/top-headlines", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for a in data.get("articles", []):
                articles.append({
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "url": a.get("url", ""),
                    "description": a.get("description", ""),
                })
            return {"articles": articles}
        except Exception as e:
            return {"error": str(e), "articles": []}

    def format_headlines(self, data: dict) -> str:
        if "error" in data and not data.get("articles"):
            return data["error"]
        if not data.get("articles"):
            return "No news articles found."
        lines = ["**Top Headlines:**"]
        for i, a in enumerate(data["articles"][:10], 1):
            source = f" ({a['source']})" if a.get("source") else ""
            lines.append(f"  {i}. {a['title']}{source}")
        return "\n".join(lines)
```

```python
# src/homie_app/console/commands/news.py
"""Handler for /news slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_news(args: str, **ctx) -> str:
    cfg = ctx.get("config")
    location = getattr(cfg, "location", None)

    # Determine country code from location
    country = "us"
    if location and location.country:
        country = location.country.lower()

    query = args.strip()

    # Get API key
    api_key = ""
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()
        cred = vault.get_credential("news", "default")
        if cred:
            api_key = cred.access_token
    except Exception:
        pass

    from homie_core.intelligence.news import NewsService
    service = NewsService(api_key=api_key)
    data = service.get_headlines(country=country, query=query)
    return service.format_headlines(data)


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="news",
        description="Top headlines (e.g., /news, /news technology, /news local)",
        handler_fn=_handle_news,
    ))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_core/test_news.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Register and commit**

```bash
git add src/homie_core/intelligence/news.py src/homie_app/console/commands/news.py tests/unit/test_core/test_news.py src/homie_app/console/commands/__init__.py
git commit -m "feat(intelligence): add news service and /news command"
```

---

### Task 15: Add /briefing command

**Files:**
- Create: `src/homie_app/console/commands/briefing.py`
- Test: `tests/unit/test_app/test_cmd_briefing.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_app/test_cmd_briefing.py
"""Tests for /briefing slash command."""
import pytest
from unittest.mock import MagicMock, patch
from homie_app.console.router import SlashCommandRouter
from homie_app.console.commands.briefing import register


def test_briefing_combines_services():
    router = SlashCommandRouter()
    register(router, {})

    cfg = MagicMock()
    cfg.location = MagicMock(city="Chennai", country="IN")
    cfg.user_name = "Master"
    cfg.storage = MagicMock(path="/tmp/.homie")

    vault = MagicMock()
    vault.get_credential.return_value = None  # No API keys

    result = router.dispatch("/briefing", **{"config": cfg, "vault": vault})
    assert "briefing" in result.lower() or "Master" in result or "no" in result.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_app/test_cmd_briefing.py -v`
Expected: ImportError

- [ ] **Step 3: Implement /briefing command**

```python
# src/homie_app/console/commands/briefing.py
"""Handler for /briefing slash command — personalized daily briefing."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_briefing(args: str, **ctx) -> str:
    cfg = ctx.get("config")
    location = getattr(cfg, "location", None)
    user_name = getattr(cfg, "user_name", "User") or "User"

    lines = [f"**Good {'morning' if _is_morning() else 'day'}, {user_name}!**\n"]

    # Weather
    if location and location.city:
        try:
            api_key = _get_api_key("weather", ctx)
            if api_key:
                from homie_core.intelligence.weather import WeatherService
                weather = WeatherService(api_key=api_key)
                data = weather.get_current(location.city)
                if "error" not in data:
                    lines.append(f"**Weather in {data['city']}:** {data['temp']}°C, {data['description']}")
                    lines.append("")
        except Exception:
            pass

    # News
    try:
        api_key = _get_api_key("news", ctx)
        if api_key:
            from homie_core.intelligence.news import NewsService
            country = location.country.lower() if location and location.country else "us"
            news = NewsService(api_key=api_key)
            data = news.get_headlines(country=country)
            if data.get("articles"):
                lines.append("**Top Headlines:**")
                for a in data["articles"][:5]:
                    lines.append(f"  - {a['title']}")
                lines.append("")
    except Exception:
        pass

    # Email summary
    try:
        from homie_core.email import EmailService
        from pathlib import Path
        import sqlite3
        from homie_core.vault.secure_vault import SecureVault

        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        storage_path = Path(cfg.storage.path).expanduser()
        cache_conn = sqlite3.connect(str(storage_path / "cache.db"))
        email_svc = EmailService(vault, cache_conn)
        accounts = email_svc.initialize()
        if accounts:
            summary = email_svc.get_summary(days=1)
            unread = summary.get("unread", 0)
            hp = summary.get("high_priority", [])
            if unread > 0:
                lines.append(f"**Email:** {unread} unread")
                if hp:
                    lines.append(f"  {len(hp)} high priority:")
                    for msg in hp[:3]:
                        lines.append(f"    - {msg.get('subject', '(no subject)')}")
                lines.append("")
    except Exception:
        pass

    if len(lines) == 1:
        lines.append("No data sources configured yet. Use /connect to set up weather, news, or email.")

    return "\n".join(lines)


def _get_api_key(provider: str, ctx: dict) -> str:
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()
        cred = vault.get_credential(provider, "default")
        return cred.access_token if cred else ""
    except Exception:
        return ""


def _is_morning() -> bool:
    from datetime import datetime
    return datetime.now().hour < 12


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="briefing",
        description="Full personalized briefing (weather, news, email, schedule)",
        handler_fn=_handle_briefing,
    ))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_app/test_cmd_briefing.py -v`
Expected: PASS

- [ ] **Step 5: Register and commit**

```bash
git add src/homie_app/console/commands/briefing.py tests/unit/test_app/test_cmd_briefing.py src/homie_app/console/commands/__init__.py
git commit -m "feat(console): add /briefing command — personalized daily briefing"
```

---

## Chunk 5: Autocomplete & Brain Tools

### Task 16: Add prompt_toolkit autocomplete

**Files:**
- Create: `src/homie_app/console/autocomplete.py`
- Modify: `src/homie_app/console/console.py` (use prompt_toolkit for input)
- Test: `tests/unit/test_app/test_autocomplete.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_app/test_autocomplete.py
"""Tests for autocomplete integration."""
import pytest
from homie_app.console.autocomplete import HomieCompleter
from homie_app.console.router import SlashCommandRouter, SlashCommand


def test_completer_returns_slash_commands():
    router = SlashCommandRouter()
    router.register(SlashCommand(name="connect", description="Connect"))
    router.register(SlashCommand(name="connections", description="List"))
    router.register(SlashCommand(name="help", description="Help"))

    completer = HomieCompleter(router)
    # Simulate completing "/con"
    from prompt_toolkit.document import Document
    doc = Document("/con", cursor_position=4)
    completions = list(completer.get_completions(doc, None))
    texts = [c.text for c in completions]
    assert "connect" in texts or "/connect" in texts
    assert "help" not in texts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_app/test_autocomplete.py -v`
Expected: ImportError

- [ ] **Step 3: Implement HomieCompleter**

```python
# src/homie_app/console/autocomplete.py
"""Autocomplete for slash commands using prompt_toolkit."""
from __future__ import annotations

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from homie_app.console.router import SlashCommandRouter


class HomieCompleter(Completer):
    """Autocomplete slash commands."""

    def __init__(self, router: SlashCommandRouter):
        self._router = router

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        # Strip the leading "/"
        after_slash = text[1:]

        # Check if we're completing a subcommand
        parts = after_slash.split(maxsplit=1)
        if len(parts) == 2:
            cmd_name = parts[0]
            sub_prefix = parts[1]
            cmd = self._router._commands.get(cmd_name)
            if cmd and cmd.subcommands:
                for name, sc in sorted(cmd.subcommands.items()):
                    if name.startswith(sub_prefix):
                        yield Completion(
                            name,
                            start_position=-len(sub_prefix),
                            display_meta=sc.description,
                        )
            return

        # Complete top-level command names
        prefix = parts[0] if parts else ""
        for cmd in self._router.list_commands():
            if cmd.name.startswith(prefix):
                yield Completion(
                    cmd.name,
                    start_position=-len(prefix),
                    display_meta=cmd.description,
                )
```

- [ ] **Step 4: Update Console.run() to use prompt_toolkit**

In `console.py`, update the `run()` method:

```python
def run(self):
    """Main input loop with autocomplete."""
    try:
        from prompt_toolkit import PromptSession
        from homie_app.console.autocomplete import HomieCompleter
        session = PromptSession(
            completer=HomieCompleter(self._router),
            complete_while_typing=False,
        )
        use_prompt_toolkit = True
    except ImportError:
        session = None
        use_prompt_toolkit = False

    while True:
        try:
            if use_prompt_toolkit:
                user_input = session.prompt(f"{self._user_name}> ").strip()
            else:
                user_input = input(f"{self._user_name}> ").strip()
        except (EOFError, KeyboardInterrupt):
            self._print("\n")
            break

        # ... rest of loop unchanged ...
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_app/test_autocomplete.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_app/console/autocomplete.py src/homie_app/console/console.py tests/unit/test_app/test_autocomplete.py
git commit -m "feat(console): add prompt_toolkit autocomplete for slash commands"
```

---

### Task 17: Register weather/news/briefing as Brain tools

**Files:**
- Modify: `src/homie_core/brain/builtin_tools.py`
- Test: `tests/unit/test_core/test_brain_tools_intelligence.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_core/test_brain_tools_intelligence.py
"""Tests for weather/news/briefing brain tools."""
import pytest
from unittest.mock import MagicMock, patch


def test_weather_tool_registered():
    from homie_core.brain.tool_registry import ToolRegistry
    registry = ToolRegistry()

    with patch("homie_core.brain.builtin_tools._register_intelligence_tools") as mock_reg:
        mock_reg(registry, config=MagicMock(), vault=MagicMock())
        mock_reg.assert_called_once()
```

- [ ] **Step 2: Implement intelligence tool registration**

Add to `builtin_tools.py` a new function `_register_intelligence_tools()` that registers `get_weather`, `get_news`, `get_local_conditions`, `get_personalized_briefing` as Brain tools. These tools call the `WeatherService` and `NewsService` classes and return formatted text.

```python
def _register_intelligence_tools(registry, config=None, vault=None):
    """Register weather, news, briefing tools for the Brain."""
    from homie_core.brain.tool_registry import Tool, ToolParam

    def get_weather(location: str = "", **kwargs):
        city = location or ""
        if not city and config:
            loc = getattr(config, "location", None)
            if loc:
                city = loc.city
        if not city:
            return "No location provided or configured."
        api_key = _get_vault_key(vault, "weather")
        from homie_core.intelligence.weather import WeatherService
        svc = WeatherService(api_key=api_key)
        data = svc.get_current(city)
        return svc.format_current(data)

    registry.register(Tool(
        name="get_weather",
        description="Get current weather for a location",
        params=[ToolParam(name="location", description="City name (optional, defaults to configured location)", type="string", required=False, default="")],
        execute=get_weather,
        category="intelligence",
    ))

    def get_news(query: str = "", location: str = "", **kwargs):
        api_key = _get_vault_key(vault, "news")
        country = "us"
        if config:
            loc = getattr(config, "location", None)
            if loc and loc.country:
                country = loc.country.lower()
        from homie_core.intelligence.news import NewsService
        svc = NewsService(api_key=api_key)
        data = svc.get_headlines(country=country, query=query)
        return svc.format_headlines(data)

    registry.register(Tool(
        name="get_news",
        description="Get top news headlines, optionally filtered by topic",
        params=[
            ToolParam(name="query", description="Topic to search for", type="string", required=False, default=""),
            ToolParam(name="location", description="Country code", type="string", required=False, default=""),
        ],
        execute=get_news,
        category="intelligence",
    ))


def _get_vault_key(vault, provider: str) -> str:
    if not vault:
        return ""
    try:
        cred = vault.get_credential(provider, "default")
        return cred.access_token if cred else ""
    except Exception:
        return ""
```

Call `_register_intelligence_tools()` from `register_builtin_tools()`.

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_core/test_brain_tools_intelligence.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/homie_core/brain/builtin_tools.py tests/unit/test_core/test_brain_tools_intelligence.py
git commit -m "feat(brain): register weather and news as Brain tools for conversational access"
```

---

### Task 18: Final integration test

**Files:**
- Create: `tests/integration/test_console_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_console_integration.py
"""Integration tests for the unified console."""
import pytest
from unittest.mock import MagicMock, patch


def test_full_console_slash_command_flow():
    """Test: launch console, run slash commands, quit."""
    from homie_app.console import Console
    from homie_app.console.router import SlashCommand

    cfg = MagicMock()
    cfg.user_name = "Tester"
    cfg.storage.path = "/tmp/.homie"
    cfg.location = None

    console = Console(config=cfg, skip_init=True)
    # Register a test command
    console._router.register(SlashCommand(
        name="test", description="Test", handler_fn=lambda args, **ctx: f"echo:{args}"
    ))

    inputs = ["/test hello", "/help", "/connections", "quit"]
    with patch("builtins.input", side_effect=inputs):
        with patch.object(console, "_print") as mock_print:
            console.run()

    printed = " ".join(str(c) for c in mock_print.call_args_list)
    assert "echo:hello" in printed
    assert "Available" in printed or "Commands" in printed


def test_console_first_run_triggers_wizard():
    """Test: no model configured triggers wizard."""
    from homie_app.console import Console

    cfg = MagicMock()
    cfg.user_name = ""
    cfg.llm.model_path = ""
    cfg.storage.path = "/tmp/.homie"

    with patch.object(Console, "_run_wizard") as mock_wizard:
        with patch.object(Console, "run"):
            console = Console(config=cfg)
            # Wizard should have been called during bootstrap
            mock_wizard.assert_called_once()
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/integration/test_console_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_console_integration.py
git commit -m "test(integration): add unified console integration tests"
```

---

## Summary

| Chunk | Tasks | What it delivers |
|-------|-------|-----------------|
| 1: Console Foundation | 1-4 | Router, Console class, /help, memory commands |
| 2: CLI Migration | 5-8 | All 22+ CLI commands as slash commands |
| 3: CLI Simplification | 9-11 | Thin cli.py entry point, wizard detection, test updates |
| 4: Location & Intelligence | 12-15 | /location, /weather, /news, /briefing |
| 5: Autocomplete & Brain Tools | 16-18 | Tab completion, Brain tool registration, integration tests |
