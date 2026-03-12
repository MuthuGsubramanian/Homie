"""AI tool wrappers for browser history."""
from __future__ import annotations
import json
from homie_core.brain.tool_registry import Tool, ToolParam, ToolRegistry

_MAX_OUTPUT = 2000

def _truncate(text: str) -> str:
    return text[:_MAX_OUTPUT] + "..." if len(text) > _MAX_OUTPUT else text

def register_browser_tools(registry: ToolRegistry, browser_service) -> None:

    def tool_browser_history(limit: str = "50", domain: str = "", since: str = "") -> str:
        try:
            num_limit = int(limit)
        except (ValueError, TypeError):
            num_limit = 50
        since_ts = None
        if since:
            try:
                since_ts = float(since)
            except (ValueError, TypeError):
                pass
        results = browser_service.get_history(limit=num_limit, domain=domain or None, since=since_ts)
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="browser_history",
        description="Get browsing history entries, optionally filtered by domain or time.",
        params=[
            ToolParam(name="limit", description="Max entries", type="string", required=False, default="50"),
            ToolParam(name="domain", description="Filter by domain", type="string", required=False, default=""),
            ToolParam(name="since", description="Unix timestamp to start from", type="string", required=False, default=""),
        ],
        execute=tool_browser_history,
        category="browser",
    ))

    def tool_browser_patterns() -> str:
        results = browser_service.get_patterns()
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="browser_patterns",
        description="Analyze browsing patterns — top domains, peak hours, categories.",
        params=[],
        execute=tool_browser_patterns,
        category="browser",
    ))

    def tool_browser_scan() -> str:
        results = browser_service.scan()
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="browser_scan",
        description="Full browser history scan and analysis.",
        params=[],
        execute=tool_browser_scan,
        category="browser",
    ))

    def tool_browser_config(enabled: str = "", browsers: str = "",
                            exclude_domains: str = "", retention_days: str = "") -> str:
        if not any([enabled, browsers, exclude_domains, retention_days]):
            return json.dumps(browser_service.get_config())
        kwargs = {}
        if enabled:
            kwargs["enabled"] = enabled.lower() == "true"
        if browsers:
            kwargs["browsers"] = [b.strip() for b in browsers.split(",")]
        if exclude_domains:
            kwargs["exclude_domains"] = [d.strip() for d in exclude_domains.split(",")]
        if retention_days:
            try:
                kwargs["retention_days"] = int(retention_days)
            except (ValueError, TypeError):
                pass
        result = browser_service.configure(**kwargs)
        return json.dumps(result)

    registry.register(Tool(
        name="browser_config",
        description="View or update browser history settings (enabled, browsers, exclude_domains, retention_days).",
        params=[
            ToolParam(name="enabled", description="'true' or 'false'", type="string", required=False, default=""),
            ToolParam(name="browsers", description="Comma-separated: chrome,firefox,edge", type="string", required=False, default=""),
            ToolParam(name="exclude_domains", description="Comma-separated domains to exclude", type="string", required=False, default=""),
            ToolParam(name="retention_days", description="Days to keep history", type="string", required=False, default=""),
        ],
        execute=tool_browser_config,
        category="browser",
    ))
