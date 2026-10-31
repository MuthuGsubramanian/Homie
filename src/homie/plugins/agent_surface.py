"""Agent Surface plugin for HOMIE.

Inspired by the Show HN discussion "What should the GUI for AI agents look
like?" (https://marbleos.com/demo), this plugin turns HOMIE's internal agent
state into a small set of GUI-ready *surface primitives* instead of raw JSON
dumps:

* **Timeline** -- a chronological "what did the agent do and why" feed built
  from the run ledger (the single most requested element in agent-GUI
  discussions: visible provenance).
* **Plan card** -- the currently proposed plan (action / target / command /
  reason) shaped as a card with explicit approve / reject affordances derived
  from ``risk_class`` and ``autonomy_level``.
* **Approval tray** -- pending suggestions awaiting a human decision, so
  approvals never hide inside a chat scrollback.
* **Status strip** -- compact per-machine health for an always-visible
  header or tray widget.

The plugin is deliberately frontend-agnostic: every builder returns plain,
JSON-serialisable view models that any local UI (the FastAPI dashboard, a
TUI, a mobile shell) can render. It performs **no network calls** of its
own; if the existing local dashboard ``FastAPI`` app is passed to
``activate()`` it mounts read-only routes onto it, and otherwise it is a
pure in-process view-model layer. UIs that want live updates can register a
callback via ``subscribe()`` and forward events over their own transport
(SSE, websocket, Qt signal, ...).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from homie.config import HomieConfig, cfg_get

try:  # FastAPI ships with the controller extras; keep the plugin importable without it.
    from fastapi import FastAPI
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]


PLUGIN_NAME = "agent_surface"

EventCallback = Callable[[Dict[str, Any]], None]

# Risk classes that should always render a blocking confirmation affordance,
# regardless of the configured autonomy level.
_ALWAYS_CONFIRM_RISK = {"medium", "high"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class AgentSurfacePlugin:
    """Builds GUI-ready view models from HOMIE's agent state.

    Parameters
    ----------
    cfg:
        Loaded :class:`~homie.config.HomieConfig`.
    storage:
        Optional :class:`~homie.controller.storage.Storage` adapter. When
        omitted, history-backed surfaces (timeline, approval tray, status
        strip) degrade to empty lists instead of raising.
    """

    def __init__(self, cfg: HomieConfig, storage: Optional[Any] = None) -> None:
        self.cfg = cfg
        self.storage = storage
        self._active = False
        self._subscribers: List[EventCallback] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self, app: Optional["FastAPI"] = None) -> None:
        """Enable the plugin and optionally mount routes on the local dashboard."""
        if self._active:
            return
        self._active = True
        if app is not None and FastAPI is not None:
            self._mount_routes(app)
        logging.info("[%s] activated (routes=%s)", PLUGIN_NAME, app is not None)

    def deactivate(self) -> None:
        """Disable the plugin. Mounted routes start returning an inactive marker."""
        self._active = False
        self._subscribers.clear()
        logging.info("[%s] deactivated", PLUGIN_NAME)

    @property
    def active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Live-update hooks (transport-agnostic)
    # ------------------------------------------------------------------
    def subscribe(self, callback: EventCallback) -> None:
        """Register a callback invoked with each surface event dict."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit(self, kind: str, payload: Dict[str, Any]) -> None:
        """Push an event (e.g. plan proposed, run finished) to all subscribers.

        Call this from the orchestrator/dispatch path; failures in one
        subscriber never break the others.
        """
        if not self._active:
            return
        event = {"ts": _utc_now(), "kind": kind, "payload": payload}
        for callback in list(self._subscribers):
            try:
                callback(event)
            except Exception:  # noqa: BLE001
                logging.exception("[%s] subscriber failed for event %s", PLUGIN_NAME, kind)

    # ------------------------------------------------------------------
    # Surface builders (all return JSON-serialisable view models)
    # ------------------------------------------------------------------
    def surface_state(self, pending_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Full snapshot a UI needs to render one frame of the agent surface."""
        return {
            "plugin": PLUGIN_NAME,
            "generated_at": _utc_now(),
            "active": self._active,
            "plan_card": self.plan_card(pending_plan) if pending_plan else None,
            "timeline": self.timeline(),
            "approval_tray": self.approval_tray(),
            "status_strip": self.status_strip(),
        }

    def plan_card(self, plan_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Shape a planner output (see ``ai_planner.PlanModel``) into a card.

        The card carries explicit ``affordances`` so the frontend never has
        to re-derive policy: a plan either auto-runs, or renders approve /
        reject buttons.
        """
        risk_class = str(plan_dict.get("risk_class", "medium")).lower()
        autonomy_level = str(plan_dict.get("autonomy_level", "suggest")).lower()
        needs_approval = autonomy_level != "auto" or risk_class in _ALWAYS_CONFIRM_RISK
        return {
            "type": "plan_card",
            "title": f"{plan_dict.get('action', '?')} on {plan_dict.get('target', '?')}",
            "action": plan_dict.get("action"),
            "target": plan_dict.get("target"),
            "command": plan_dict.get("command"),
            "reason": plan_dict.get("reason") or "(no reason given)",
            "risk_class": risk_class,
            "autonomy_level": autonomy_level,
            "affordances": (
                ["approve", "reject", "edit_command"] if needs_approval else ["cancel"]
            ),
            "needs_approval": needs_approval,
        }

    def timeline(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Chronological activity feed from the ledger: what, why, outcome."""
        entries: List[Dict[str, Any]] = []
        for row in self._safe_rows("recent_ledger"):
            entries.append(
                {
                    "type": "timeline_entry",
                    "ts": row.get("ts"),
                    "actor": row.get("actor", "homie"),
                    "what": row.get("what"),
                    "why": row.get("why"),
                    "outcome": row.get("outcome"),
                    "risk_class": row.get("risk_class"),
                    "autonomy_level": row.get("autonomy_level"),
                    "rollback_available": bool(row.get("rollback_available")),
                }
            )
            if len(entries) >= limit:
                break
        return entries

    def approval_tray(self) -> List[Dict[str, Any]]:
        """Suggestions still awaiting a human decision."""
        tray: List[Dict[str, Any]] = []
        for row in self._safe_rows("recent_suggestions"):
            if str(row.get("status", "")).lower() not in {"", "pending", "offered"}:
                continue
            tray.append(
                {
                    "type": "approval_item",
                    "id": row.get("id"),
                    "summary": row.get("summary"),
                    "intent_tag": row.get("intent_tag"),
                    "confidence": row.get("confidence"),
                    "autonomy_level_offered": row.get("autonomy_level_offered"),
                    "affordances": ["approve", "reject", "defer"],
                }
            )
        return tray

    def status_strip(self) -> List[Dict[str, Any]]:
        """Compact per-machine health for an always-visible header widget."""
        strip: List[Dict[str, Any]] = []
        for machine in self._safe_rows("list_machines"):
            strip.append(
                {
                    "type": "machine_chip",
                    "name": machine.get("display_name") or machine.get("ip"),
                    "ip": machine.get("ip"),
                    "os_type": machine.get("os_type"),
                    "status": machine.get("status") or "unknown",
                    "last_seen_ts": machine.get("last_seen_ts"),
                    "blocked": bool(machine.get("blocked")),
                }
            )
        return strip

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _safe_rows(self, accessor: str) -> List[Dict[str, Any]]:
        """Call an optional storage accessor, degrading to [] when unavailable."""
        if self.storage is None:
            return []
        fn = getattr(self.storage, accessor, None)
        if fn is None:
            return []
        try:
            return [dict(row) for row in fn()]
        except Exception:  # noqa: BLE001
            logging.exception("[%s] storage accessor %s failed", PLUGIN_NAME, accessor)
            return []

    def _mount_routes(self, app: "FastAPI") -> None:
        """Attach read-only surface routes to the existing local dashboard app."""
        prefix = cfg_get(self.cfg, "plugins", PLUGIN_NAME, "route_prefix", default="/api/surface")

        @app.get(f"{prefix}/state")
        def surface_state() -> Dict[str, Any]:  # pragma: no cover - thin wrapper
            if not self._active:
                return {"plugin": PLUGIN_NAME, "active": False}
            return self.surface_state()

        @app.get(f"{prefix}/timeline")
        def surface_timeline() -> List[Dict[str, Any]]:  # pragma: no cover
            return self.timeline() if self._active else []

        logging.info("[%s] mounted routes under %s", PLUGIN_NAME, prefix)


def register(
    cfg: HomieConfig,
    storage: Optional[Any] = None,
    app: Optional["FastAPI"] = None,
) -> AgentSurfacePlugin:
    """Create and activate the plugin (convenience entry point for loaders)."""
    plugin = AgentSurfacePlugin(cfg, storage=storage)
    plugin.activate(app=app)
    return plugin


__all__ = ["AgentSurfacePlugin", "register", "PLUGIN_NAME"]
