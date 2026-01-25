from __future__ import annotations

import re
from typing import Dict, Tuple

from homie.config import HomieConfig, cfg_get


DEFAULT_BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/(?:\s|$)",
    r"\bmkfs(\.\w+)?\b",
    r"\bfdisk\b",
    r"\bwipefs\b",
    r"\bshutdown\b",
    r"\breboot\b",
]


def validate_plan(plan: Dict, cfg: HomieConfig) -> Tuple[bool, str]:
    """
    Validate safety rules for a planned action.
    Returns (is_safe, reason_if_not).
    """
    allowed_actions = cfg_get(
        cfg,
        "orchestrator",
        "allowed_actions",
        default=["run_command", "check_status", "copy_file"],
    )
    targets = cfg_get(cfg, "ssh", "targets", default={}) or {}
    safety_cfg = cfg_get(cfg, "safety", default={}) or {}

    action = plan.get("action")
    target = plan.get("target")
    command = (plan.get("command") or "").strip()
    args = plan.get("args") or {}
    reason = (plan.get("reason") or "").strip()

    if action not in allowed_actions:
        return False, f"Action '{action}' not in allowed_actions {allowed_actions}"

    if target != "all" and target not in targets:
        return False, f"Unknown target '{target}'. Known targets: {list(targets.keys())}"

    if safety_cfg.get("require_reason", True) and not reason:
        return False, "Missing 'reason' field"

    if action == "run_command":
        if not command:
            return False, "command is required for run_command"
        max_len = int(safety_cfg.get("max_command_len", 300))
        if len(command) > max_len:
            return False, f"command length {len(command)} exceeds max {max_len}"

        blocked_substrings = safety_cfg.get("blocked_substrings", [])
        for bad in blocked_substrings:
            if bad in command:
                return False, f"command blocked due to substring '{bad}'"

        for pattern in DEFAULT_BLOCKED_PATTERNS:
            if re.search(pattern, command):
                return False, f"command blocked by pattern '{pattern}'"

    if action == "copy_file":
        src = args.get("src") or args.get("source")
        dest = args.get("dest") or args.get("destination") or args.get("target")
        if not src or not dest:
            return False, "copy_file requires args.src and args.dest"

    return True, ""


__all__ = ["validate_plan", "DEFAULT_BLOCKED_PATTERNS"]
