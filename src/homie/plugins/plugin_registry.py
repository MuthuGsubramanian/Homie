"""CowAgent-inspired plugin registry for HOMIE.

CowAgent (https://cowagent.ai) is an open-source "super assistant" built
around a manifest-driven plugin system: each capability ships as a folder
with a manifest declaring what it needs (files, shell, network) and the host
decides whether to load it. This module brings that model to HOMIE while
keeping HOMIE's local-first, safety-first stance.

Design
------
* Plugins live under ``~/.homie/plugins/<name>/`` with a ``plugin.yaml``
  manifest and an entry module exposing ``register(cfg)`` (preferred) or a
  module-level ``activate()`` / ``deactivate()`` pair.
* Manifests declare ``permissions`` drawn from an ordered tier list.
  ``network`` is DENIED unless ``plugins.allow_network: true`` is set in
  ``homie.config.yaml``; nothing in this module opens a socket.
* A static CowAgent-vs-HOMIE parity matrix is bundled so ``:parity`` can
  show where HOMIE is ahead, behind, or deliberately different. It is a
  hand-maintained snapshot, not a live comparison.
* Everything is reversible: ``deactivate()`` unloads every plugin and drops
  any meta commands this module registered.

Example manifest (``~/.homie/plugins/disk_report/plugin.yaml``)::

    name: disk_report
    version: 0.1.0
    entry: plugin.py
    description: Summarise disk usage of the local machine
    permissions: [read_local, exec_local]

Config keys (all optional)::

    plugins:
      enabled: true
      dir: ~/.homie/plugins
      allow_network: false
      max_permission_tier: ssh      # read_local|write_local|exec_local|ssh|network
      denylist: []

Usage::

    from homie.plugins.plugin_registry import register
    registry = register(cfg, meta_handlers=META_HANDLERS)
    registry.activate()
    ...
    registry.deactivate()

Meta commands (when ``meta_handlers`` is supplied)::

    :plugins            list discovered plugins and their state
    :plugins reload     rescan the plugins directory
    :parity             show the CowAgent / HOMIE parity matrix
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

from homie.config import HomieConfig, cfg_get

PLUGIN_NAME = "plugin_registry"
PLUGIN_VERSION = "0.1.0"
MANIFEST_NAME = "plugin.yaml"
DEFAULT_PLUGIN_DIR = Path.home() / ".homie" / "plugins"
MODULE_PREFIX = "homie_plugin_"

# Ordered least -> most sensitive. A plugin is loadable only if every
# permission it requests is at or below plugins.max_permission_tier.
PERMISSION_TIERS: Dict[str, int] = {
    "read_local": 0,   # read files under the user's home
    "write_local": 1,  # write files under ~/.homie
    "exec_local": 2,   # spawn local subprocesses
    "ssh": 3,          # dispatch through Orchestrator (already safety-gated)
    "network": 4,      # any outbound socket; opt-in via plugins.allow_network
}

MetaHandler = Callable[[str], None]

# Hand-maintained snapshot (2026-10) of how CowAgent's advertised feature set
# lines up with HOMIE. Columns: feature, cowagent, homie, status.
PARITY_MATRIX: List[Tuple[str, str, str, str]] = [
    ("Plugin system", "manifest plugins + marketplace", "this registry; no marketplace by design", "parity (local-only)"),
    ("LLM backend", "cloud APIs with optional local", "llama.cpp / Ollama only", "differentiation"),
    ("RAG over local files", "built-in", "ChromaDB", "parity"),
    ("Voice in / out", "built-in", "Whisper + Piper", "parity"),
    ("Multi-machine orchestration", "not advertised", "SSH + signed node API", "ahead"),
    ("Action safety gating", "prompt-level", "validate_plan + risk_class + autonomy_level", "ahead"),
    ("Scheduled automations", "built-in", "APScheduler cron jobs", "parity"),
    ("Browser automation", "built-in", "none", "behind (candidate plugin)"),
    ("Chat-channel bridges", "WeChat / Telegram / Slack", "none", "behind (deliberate; opt-in only)"),
    ("Cross-session memory", "built-in", "preferences + knowledge_index tables", "partial"),
    ("Audit ledger", "not advertised", "SQLite ledger with rollback flag", "ahead"),
]


class PluginError(Exception):
    """Raised when a plugin manifest or module cannot be used."""


@dataclass
class PluginManifest:
    """Parsed ``plugin.yaml``."""

    name: str
    version: str
    entry: str
    description: str = ""
    permissions: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], folder: Path) -> "PluginManifest":
        name = str(data.get("name") or folder.name).strip()
        if not name.replace("_", "").replace("-", "").isalnum():
            raise PluginError(f"invalid plugin name '{name}'")
        perms = [str(p) for p in (data.get("permissions") or [])]
        unknown = sorted(set(perms) - set(PERMISSION_TIERS))
        if unknown:
            raise PluginError(f"unknown permissions {unknown}")
        return cls(
            name=name,
            version=str(data.get("version") or "0.0.0"),
            entry=str(data.get("entry") or "plugin.py"),
            description=str(data.get("description") or ""),
            permissions=perms,
        )

    @property
    def max_tier(self) -> int:
        return max((PERMISSION_TIERS[p] for p in self.permissions), default=-1)


@dataclass
class LoadedPlugin:
    """Runtime record for one discovered plugin."""

    manifest: PluginManifest
    folder: Path
    module: Optional[ModuleType] = None
    instance: Any = None
    state: str = "discovered"  # discovered | denied | loaded | active | failed
    error: Optional[str] = None

    @property
    def name(self) -> str:
        return self.manifest.name


class PluginRegistry:
    """Discover, permission-check, load and activate manifest plugins."""

    def __init__(
        self,
        cfg: HomieConfig,
        plugin_dir: Optional[Path] = None,
        meta_handlers: Optional[Dict[str, MetaHandler]] = None,
    ) -> None:
        self.cfg = cfg
        raw_dir = plugin_dir or cfg_get(cfg, "plugins", "dir", default=DEFAULT_PLUGIN_DIR)
        self.plugin_dir = Path(raw_dir).expanduser()
        self.enabled = bool(cfg_get(cfg, "plugins", "enabled", default=True))
        self.allow_network = bool(cfg_get(cfg, "plugins", "allow_network", default=False))
        tier_name = str(cfg_get(cfg, "plugins", "max_permission_tier", default="ssh"))
        self.max_tier = PERMISSION_TIERS.get(tier_name, PERMISSION_TIERS["ssh"])
        self.denylist: Set[str] = set(cfg_get(cfg, "plugins", "denylist", default=[]) or [])
        self.plugins: Dict[str, LoadedPlugin] = {}
        self._meta_handlers = meta_handlers
        self._registered_meta: Set[str] = set()
        self._active = False

    # ---- discovery -------------------------------------------------

    def discover(self) -> List[LoadedPlugin]:
        """Scan the plugin directory and parse every manifest found."""
        self.plugins.clear()
        if not self.plugin_dir.is_dir():
            logging.debug("Plugin dir %s missing; nothing to discover", self.plugin_dir)
            return []
        for folder in sorted(p for p in self.plugin_dir.iterdir() if p.is_dir()):
            manifest_path = folder / MANIFEST_NAME
            if not manifest_path.is_file():
                continue
            try:
                data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
                manifest = PluginManifest.from_dict(data, folder)
            except (yaml.YAMLError, PluginError, OSError) as exc:
                logging.warning("Skipping plugin at %s: %s", folder, exc)
                continue
            record = LoadedPlugin(manifest=manifest, folder=folder)
            ok, reason = self._permitted(manifest)
            if not ok:
                record.state = "denied"
                record.error = reason
            self.plugins[manifest.name] = record
        return list(self.plugins.values())

    def _permitted(self, manifest: PluginManifest) -> Tuple[bool, str]:
        if manifest.name in self.denylist:
            return False, "listed in plugins.denylist"
        if "network" in manifest.permissions and not self.allow_network:
            return False, "requests network; set plugins.allow_network: true to opt in"
        if manifest.max_tier > self.max_tier:
            return False, f"requests tier above plugins.max_permission_tier ({manifest.permissions})"
        return True, ""

    # ---- loading ---------------------------------------------------

    def _entry_path(self, record: LoadedPlugin) -> Path:
        entry = (record.folder / record.manifest.entry).resolve()
        if record.folder.resolve() not in entry.parents:
            raise PluginError("entry must live inside the plugin folder")
        if not entry.is_file():
            raise PluginError(f"entry file {entry.name} not found")
        return entry

    def load(self, name: str) -> LoadedPlugin:
        """Import a plugin's entry module and call its ``register(cfg)``."""
        record = self.plugins[name]
        if record.state == "denied":
            return record
        mod_name = f"{MODULE_PREFIX}{name}"
        try:
            entry = self._entry_path(record)
            spec = importlib.util.spec_from_file_location(mod_name, entry)
            if spec is None or spec.loader is None:
                raise PluginError("could not build import spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            record.module = module
            reg = getattr(module, "register", None)
            record.instance = reg(self.cfg) if callable(reg) else module
            if not callable(getattr(record.instance, "activate", None)):
                raise PluginError("plugin exposes neither register() nor activate()")
            record.state = "loaded"
            record.error = None
        except Exception as exc:  # noqa: BLE001 - third-party plugin code
            logging.exception("Failed to load plugin %s", name)
            record.state = "failed"
            record.error = str(exc)
            sys.modules.pop(mod_name, None)
        return record

    # ---- lifecycle -------------------------------------------------

    def activate(self) -> None:
        """Discover, load and activate every permitted plugin."""
        if self._active:
            return
        self._register_meta()
        if not self.enabled:
            logging.info("plugins.enabled is false; registry active with no plugins")
            self._active = True
            return
        self.discover()
        for name in list(self.plugins):
            record = self.load(name)
            if record.state != "loaded":
                continue
            try:
                record.instance.activate()
                record.state = "active"
            except Exception as exc:  # noqa: BLE001
                logging.exception("Plugin %s failed to activate", name)
                record.state = "failed"
                record.error = str(exc)
        self._active = True
        logging.info("Plugin registry active: %s", "; ".join(self.status_lines()) or "no plugins")

    def deactivate(self) -> None:
        """Deactivate and unload every plugin, then drop our meta commands."""
        for name, record in self.plugins.items():
            if record.state == "active":
                deact = getattr(record.instance, "deactivate", None)
                try:
                    if callable(deact):
                        deact()
                except Exception:  # noqa: BLE001
                    logging.exception("Plugin %s failed to deactivate", name)
            if record.state != "denied":
                record.state = "discovered"
            record.module = None
            record.instance = None
            sys.modules.pop(f"{MODULE_PREFIX}{name}", None)
        self._unregister_meta()
        self._active = False

    def reload(self) -> None:
        """Full deactivate / rescan / activate cycle."""
        self.deactivate()
        self.activate()

    # ---- reporting -------------------------------------------------

    def status_lines(self) -> List[str]:
        lines: List[str] = []
        for record in self.plugins.values():
            m = record.manifest
            perms = ",".join(m.permissions) or "none"
            line = f"{m.name} v{m.version} [{record.state}] perms={perms}"
            if record.error:
                line += f" ({record.error})"
            lines.append(line)
        return lines

    def parity_lines(self) -> List[str]:
        width = max(len(row[0]) for row in PARITY_MATRIX)
        lines = [f"{'feature'.ljust(width)} | cowagent -> homie | status"]
        for feature, cow, homie, status in PARITY_MATRIX:
            lines.append(f"{feature.ljust(width)} | {cow} -> {homie} | {status}")
        return lines

    # ---- meta commands ---------------------------------------------

    def handle_meta(self, cmd: str) -> bool:
        """Handle ``:plugins`` / ``:parity``. Return True if consumed."""
        parts = cmd.strip().lower().split()
        if not parts:
            return False
        if parts[0] == ":plugins":
            if len(parts) > 1 and parts[1] == "reload":
                self.reload()
            print(f"Plugins ({self.plugin_dir}):")
            for line in self.status_lines() or ["none"]:
                print("  " + line)
            return True
        if parts[0] == ":parity":
            for line in self.parity_lines():
                print(line)
            return True
        return False

    def _register_meta(self) -> None:
        if self._meta_handlers is None:
            return
        for key in (":plugins", ":parity"):
            if key not in self._meta_handlers:
                self._meta_handlers[key] = self.handle_meta
                self._registered_meta.add(key)

    def _unregister_meta(self) -> None:
        if self._meta_handlers is None:
            return
        for key in self._registered_meta:
            self._meta_handlers.pop(key, None)
        self._registered_meta.clear()


_REGISTRY: Optional[PluginRegistry] = None


def register(
    cfg: HomieConfig,
    meta_handlers: Optional[Dict[str, MetaHandler]] = None,
    plugin_dir: Optional[Path] = None,
) -> PluginRegistry:
    """Create (or return) the process-wide registry. Does not activate it."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = PluginRegistry(cfg, plugin_dir=plugin_dir, meta_handlers=meta_handlers)
    return _REGISTRY


def activate(cfg: Optional[HomieConfig] = None, **kwargs: Any) -> PluginRegistry:
    """Convenience: register (if needed) and activate."""
    if _REGISTRY is None:
        if cfg is None:
            raise PluginError("activate() needs a HomieConfig on first call")
        register(cfg, **kwargs)
    assert _REGISTRY is not None
    _REGISTRY.activate()
    return _REGISTRY


def deactivate() -> None:
    """Deactivate and forget the process-wide registry."""
    global _REGISTRY
    if _REGISTRY is not None:
        _REGISTRY.deactivate()
        _REGISTRY = None


__all__ = [
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PERMISSION_TIERS",
    "PARITY_MATRIX",
    "PluginError",
    "PluginManifest",
    "LoadedPlugin",
    "PluginRegistry",
    "register",
    "activate",
    "deactivate",
]
