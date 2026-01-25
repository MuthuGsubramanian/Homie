from __future__ import annotations

import re
from typing import Dict, Tuple

from homie.safety import DEFAULT_BLOCKED_PATTERNS


def is_command_safe(command: str, policy: Dict) -> Tuple[bool, str]:
    max_len = int(policy.get("max_command_len", 300))
    if len(command) > max_len:
        return False, f"command too long ({len(command)}/{max_len})"

    for bad in policy.get("blocked_substrings", []):
        if bad in command:
            return False, f"blocked substring '{bad}'"

    for pattern in DEFAULT_BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return False, f"blocked pattern '{pattern}'"

    return True, ""


__all__ = ["is_command_safe"]
