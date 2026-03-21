from __future__ import annotations

from rich.console import Console as RichConsole
from rich.theme import Theme

HOMIE_THEME = Theme({
    "homie.system":    "bold cyan",
    "homie.user":      "bold white",
    "homie.assistant": "green",
    "homie.tool":      "yellow",
    "homie.tool.ok":   "bold green",
    "homie.tool.err":  "bold red",
    "homie.memory":    "magenta",
    "homie.dim":       "dim white",
    "homie.error":     "bold red",
    "homie.warn":      "yellow",
    "homie.stage":     "bold blue",
    "homie.brand":     "bold cyan",
})

rc = RichConsole(theme=HOMIE_THEME, highlight=False, soft_wrap=True)
