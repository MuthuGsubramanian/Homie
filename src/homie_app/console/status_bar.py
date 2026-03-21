from __future__ import annotations

from datetime import datetime

from rich.rule import Rule


def print_status_bar(
    rc,
    model_name: str = "",
    memory_count: int = 0,
    project: str = "",
) -> None:
    now = datetime.now().strftime("%H:%M")
    parts: list[str] = []
    if model_name:
        parts.append(f"[homie.brand]{model_name}[/]")
    if memory_count:
        parts.append(f"[homie.memory]{memory_count} facts[/]")
    if project:
        parts.append(f"[homie.tool]{project}[/]")
    parts.append(f"[homie.dim]{now}[/]")
    rc.print(Rule(" · ".join(parts), style="homie.dim", align="left"))
