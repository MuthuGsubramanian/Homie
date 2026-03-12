"""AI tool wrappers for social media profile integrations."""
from __future__ import annotations
import json
from homie_core.brain.tool_registry import Tool, ToolParam, ToolRegistry

_MAX_OUTPUT = 2000

def _truncate(text: str) -> str:
    return text[:_MAX_OUTPUT] + "..." if len(text) > _MAX_OUTPUT else text

def register_social_media_tools(registry: ToolRegistry, social_media_service) -> None:
    """Register all social media tools with the tool registry."""

    def tool_sm_feed(platform: str = "all", limit: str = "20") -> str:
        try:
            num_limit = int(limit)
        except (ValueError, TypeError):
            num_limit = 20
        results = social_media_service.get_feed(platform=platform, limit=num_limit)
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="sm_feed",
        description="Get recent feed posts from social media platforms (Twitter, Reddit, LinkedIn, etc.).",
        params=[
            ToolParam(name="platform", description="Platform name or 'all'", type="string", required=False, default="all"),
            ToolParam(name="limit", description="Maximum posts to return", type="string", required=False, default="20"),
        ],
        execute=tool_sm_feed,
        category="social_media",
    ))

    def tool_sm_profile(platform: str = "", username: str = "") -> str:
        if not platform:
            return json.dumps({"error": "platform is required"})
        results = social_media_service.get_profile(platform, username=username or None)
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="sm_profile",
        description="Get profile info from a social media platform.",
        params=[
            ToolParam(name="platform", description="Platform name (twitter, reddit, etc.)", type="string"),
            ToolParam(name="username", description="Username to look up (own profile if empty)", type="string", required=False, default=""),
        ],
        execute=tool_sm_profile,
        category="social_media",
    ))

    def tool_sm_scan_profiles() -> str:
        results = social_media_service.scan_profiles()
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="sm_scan_profiles",
        description="Full scan of all connected social media profiles — analyzes topics, tone, audience, and posting patterns.",
        params=[],
        execute=tool_sm_scan_profiles,
        category="social_media",
    ))

    def tool_sm_publish(platform: str = "", content: str = "", media_urls: str = "") -> str:
        if not platform or not content:
            return json.dumps({"error": "platform and content are required"})
        urls = [u.strip() for u in media_urls.split(",") if u.strip()] if media_urls else None
        results = social_media_service.publish(platform, content, media_urls=urls)
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="sm_publish",
        description="Publish a post to a social media platform.",
        params=[
            ToolParam(name="platform", description="Platform name", type="string"),
            ToolParam(name="content", description="Post content/text", type="string"),
            ToolParam(name="media_urls", description="Comma-separated media URLs (optional)", type="string", required=False, default=""),
        ],
        execute=tool_sm_publish,
        category="social_media",
    ))

    def tool_sm_conversations(platform: str = "", limit: str = "20") -> str:
        if not platform:
            return json.dumps({"error": "platform is required"})
        try:
            num_limit = int(limit)
        except (ValueError, TypeError):
            num_limit = 20
        results = social_media_service.get_conversations(platform, limit=num_limit)
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="sm_conversations",
        description="List DM conversations on a social media platform.",
        params=[
            ToolParam(name="platform", description="Platform name", type="string"),
            ToolParam(name="limit", description="Maximum conversations", type="string", required=False, default="20"),
        ],
        execute=tool_sm_conversations,
        category="social_media",
    ))

    def tool_sm_dms(platform: str = "", conversation_id: str = "", limit: str = "20") -> str:
        if not platform or not conversation_id:
            return json.dumps({"error": "platform and conversation_id are required"})
        try:
            num_limit = int(limit)
        except (ValueError, TypeError):
            num_limit = 20
        results = social_media_service.get_dms(platform, conversation_id, limit=num_limit)
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="sm_dms",
        description="Get messages in a DM conversation on a social media platform.",
        params=[
            ToolParam(name="platform", description="Platform name", type="string"),
            ToolParam(name="conversation_id", description="Conversation ID", type="string"),
            ToolParam(name="limit", description="Maximum messages", type="string", required=False, default="20"),
        ],
        execute=tool_sm_dms,
        category="social_media",
    ))

    def tool_sm_send_dm(platform: str = "", recipient: str = "", text: str = "") -> str:
        if not platform or not recipient or not text:
            return json.dumps({"error": "platform, recipient, and text are required"})
        results = social_media_service.send_dm(platform, recipient, text)
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="sm_send_dm",
        description="Send a direct message on a social media platform.",
        params=[
            ToolParam(name="platform", description="Platform name", type="string"),
            ToolParam(name="recipient", description="Recipient username or ID", type="string"),
            ToolParam(name="text", description="Message text", type="string"),
        ],
        execute=tool_sm_send_dm,
        category="social_media",
    ))

    def tool_sm_search(query: str = "", platform: str = "all", limit: str = "10") -> str:
        if not query:
            return json.dumps({"error": "query is required"})
        try:
            num_limit = int(limit)
        except (ValueError, TypeError):
            num_limit = 10
        results = social_media_service.search(query, platform=platform, limit=num_limit)
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="sm_search",
        description="Search posts across social media platforms.",
        params=[
            ToolParam(name="query", description="Search query", type="string"),
            ToolParam(name="platform", description="Platform name or 'all'", type="string", required=False, default="all"),
            ToolParam(name="limit", description="Maximum results", type="string", required=False, default="10"),
        ],
        execute=tool_sm_search,
        category="social_media",
    ))

    def tool_sm_notifications(platform: str = "all", limit: str = "20") -> str:
        try:
            num_limit = int(limit)
        except (ValueError, TypeError):
            num_limit = 20
        results = social_media_service.get_notifications(platform=platform, limit=num_limit)
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="sm_notifications",
        description="Get recent notifications (likes, comments, mentions) from social media platforms.",
        params=[
            ToolParam(name="platform", description="Platform name or 'all'", type="string", required=False, default="all"),
            ToolParam(name="limit", description="Maximum notifications", type="string", required=False, default="20"),
        ],
        execute=tool_sm_notifications,
        category="social_media",
    ))
