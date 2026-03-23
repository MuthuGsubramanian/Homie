"""AI tool wrappers for email integration.

Registers 9 tools with the ToolRegistry so the Brain can interact
with email via the agentic loop.
"""
from __future__ import annotations

import json

from homie_core.brain.tool_registry import Tool, ToolParam, ToolRegistry

_MAX_OUTPUT = 2000


def _truncate(text: str) -> str:
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT] + "..."
    return text


def register_email_tools(registry: ToolRegistry, email_service) -> None:
    """Register all email tools with the tool registry."""

    def tool_email_search(query: str, account: str = "all", max_results: str = "10") -> str:
        try:
            limit = int(max_results)
        except (ValueError, TypeError):
            limit = 10
        results = email_service.search(query, account=account, max_results=limit)
        return _truncate(json.dumps([m.to_dict() if hasattr(m, "to_dict") else m for m in results]))

    registry.register(Tool(
        name="email_search",
        description="Search emails using Gmail query syntax (e.g. 'from:alice subject:meeting').",
        params=[
            ToolParam(name="query", description="Gmail search query", type="string"),
            ToolParam(name="account", description="Account email or 'all'", type="string", required=False, default="all"),
            ToolParam(name="max_results", description="Maximum results", type="string", required=False, default="10"),
        ],
        execute=tool_email_search,
        category="email",
    ))

    def tool_email_read(message_id: str) -> str:
        result = email_service.read_message(message_id)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_read",
        description="Read the full body of a specific email by its message ID.",
        params=[
            ToolParam(name="message_id", description="Email message ID", type="string"),
        ],
        execute=tool_email_read,
        category="email",
    ))

    def tool_email_thread(thread_id: str) -> str:
        result = email_service.get_thread(thread_id)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_thread",
        description="Get all messages in a conversation thread.",
        params=[
            ToolParam(name="thread_id", description="Thread/conversation ID", type="string"),
        ],
        execute=tool_email_thread,
        category="email",
    ))

    def tool_email_draft(
        to: str, subject: str, body: str,
        reply_to: str = "", cc: str = "", bcc: str = "",
        account: str = "",
    ) -> str:
        draft_id = email_service.create_draft(
            to=to, subject=subject, body=body,
            reply_to=reply_to or None,
            cc=[c.strip() for c in cc.split(",") if c.strip()] if cc else None,
            bcc=[b.strip() for b in bcc.split(",") if b.strip()] if bcc else None,
            account=account or None,
        )
        return json.dumps({"draft_id": draft_id, "status": "Draft created. User must review and send manually."})

    registry.register(Tool(
        name="email_draft",
        description="Create a draft email (never sends automatically). User reviews and sends manually.",
        params=[
            ToolParam(name="to", description="Recipient email", type="string"),
            ToolParam(name="subject", description="Email subject", type="string"),
            ToolParam(name="body", description="Email body text", type="string"),
            ToolParam(name="reply_to", description="Message ID to reply to", type="string", required=False, default=""),
            ToolParam(name="cc", description="CC recipients (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="bcc", description="BCC recipients (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="account", description="Send from this account", type="string", required=False, default=""),
        ],
        execute=tool_email_draft,
        category="email",
    ))

    def tool_email_labels(account: str = "") -> str:
        labels = email_service.list_labels(account=account or None)
        return _truncate(json.dumps(labels))

    registry.register(Tool(
        name="email_labels",
        description="List all email labels/folders.",
        params=[
            ToolParam(name="account", description="Account email (optional)", type="string", required=False, default=""),
        ],
        execute=tool_email_labels,
        category="email",
    ))

    def tool_email_summary(days: str = "1") -> str:
        try:
            num_days = int(days)
        except (ValueError, TypeError):
            num_days = 1
        result = email_service.get_summary(days=num_days)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_summary",
        description="Get email summary: unread count, high-priority items, action items.",
        params=[
            ToolParam(name="days", description="Number of days to summarize", type="string", required=False, default="1"),
        ],
        execute=tool_email_summary,
        category="email",
    ))

    def tool_email_unread(account: str = "all") -> str:
        result = email_service.get_unread(account=account)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_unread",
        description="List unread emails grouped by priority (high, medium, low).",
        params=[
            ToolParam(name="account", description="Account email or 'all'", type="string", required=False, default="all"),
        ],
        execute=tool_email_unread,
        category="email",
    ))

    def tool_email_archive(message_id: str) -> str:
        email_service.archive_message(message_id)
        return json.dumps({"status": "archived", "message_id": message_id})

    registry.register(Tool(
        name="email_archive",
        description="Archive a message (remove from inbox).",
        params=[
            ToolParam(name="message_id", description="Email message ID", type="string"),
        ],
        execute=tool_email_archive,
        category="email",
    ))

    def tool_email_mark_read(message_id: str) -> str:
        email_service.mark_read(message_id)
        return json.dumps({"status": "marked_read", "message_id": message_id})

    registry.register(Tool(
        name="email_mark_read",
        description="Mark a message as read.",
        params=[
            ToolParam(name="message_id", description="Email message ID", type="string"),
        ],
        execute=tool_email_mark_read,
        category="email",
    ))

    # ── LLM-powered email intelligence tools ─────────────────────────────

    def tool_email_analyze(message_id: str) -> str:
        """Deep-analyze a single email using the local LLM."""
        result = email_service.analyze_email(message_id)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_analyze",
        description=(
            "Deep-analyze a single email using AI: spam detection, intent analysis, "
            "action detection, and content summary. Returns structured analysis."
        ),
        params=[
            ToolParam(name="message_id", description="Email message ID to analyze", type="string"),
        ],
        execute=tool_email_analyze,
        category="email",
    ))

    def tool_email_deep_analyze(message_id: str) -> str:
        """Deep contextual analysis — deadlines, actions, impact, draft reply."""
        result = email_service.deep_analyze_email(message_id)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_deep_analyze",
        description=(
            "Deep contextual email analysis: extracts deadlines, required actions, "
            "business/personal impact, and drafts a suggested response. "
            "Use when the user asks 'what should I do about this email' or similar."
        ),
        params=[
            ToolParam(name="message_id", description="Email message ID", type="string"),
        ],
        execute=tool_email_deep_analyze,
        category="email",
    ))

    def tool_email_triage(account: str = "all", max_emails: str = "15") -> str:
        """Batch-triage unread emails using the local LLM."""
        try:
            limit = int(max_emails)
        except (ValueError, TypeError):
            limit = 15
        result = email_service.triage(account=account, max_emails=limit)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_triage",
        description=(
            "Batch-triage unread emails using AI. Analyzes up to 15 emails at once, "
            "classifying spam vs important, detecting action items, and summarizing content. "
            "Returns structured triage results for each email."
        ),
        params=[
            ToolParam(name="account", description="Account email or 'all'", type="string", required=False, default="all"),
            ToolParam(name="max_emails", description="Max emails to triage (up to 15)", type="string", required=False, default="15"),
        ],
        execute=tool_email_triage,
        category="email",
    ))

    def tool_email_digest(days: str = "1") -> str:
        """Generate an intelligent email digest using the local LLM."""
        try:
            num_days = int(days)
        except (ValueError, TypeError):
            num_days = 1
        result = email_service.get_intelligent_digest(days=num_days)
        return result if isinstance(result, str) else json.dumps(result)

    registry.register(Tool(
        name="email_digest",
        description=(
            "Generate a natural-language email digest using AI. Summarizes recent emails "
            "into action items, important updates, and noise count. Much richer than email_summary."
        ),
        params=[
            ToolParam(name="days", description="Number of days to cover", type="string", required=False, default="1"),
        ],
        execute=tool_email_digest,
        category="email",
    ))

    # ── Thread tools ─────────────────────────────────────────────────

    def tool_email_inbox_threads(account: str = "all", start: str = "0", max_results: str = "20") -> str:
        threads = email_service.list_inbox_threads(
            account=None if account == "all" else account,
            start=int(start), max_results=int(max_results),
        )
        return _truncate(json.dumps([
            {"id": t.id, "subject": t.subject, "participants": t.participants,
             "message_count": t.message_count, "snippet": t.snippet}
            for t in threads
        ]))

    registry.register(Tool(
        name="email_inbox_threads",
        description="List inbox conversation threads with pagination.",
        params=[
            ToolParam(name="account", description="Account or 'all'", type="string", required=False, default="all"),
            ToolParam(name="start", description="Start offset", type="string", required=False, default="0"),
            ToolParam(name="max_results", description="Max threads", type="string", required=False, default="20"),
        ],
        execute=tool_email_inbox_threads, category="email",
    ))

    def tool_email_thread_full(thread_id: str) -> str:
        thread = email_service.get_thread_messages(thread_id)
        if not thread:
            return json.dumps({"error": "Thread not found"})
        return _truncate(json.dumps({
            "id": thread.id, "subject": thread.subject,
            "messages": [m.to_dict() for m in thread.messages],
        }))

    registry.register(Tool(
        name="email_thread_full",
        description="Fetch complete conversation thread with all messages.",
        params=[ToolParam(name="thread_id", description="Thread ID", type="string")],
        execute=tool_email_thread_full, category="email",
    ))

    def tool_email_unread_counts(account: str = "all") -> str:
        return json.dumps(email_service.get_unread_counts(
            account=None if account == "all" else account,
        ))

    registry.register(Tool(
        name="email_unread_counts",
        description="Get unread email counts by category (inbox, spam, starred).",
        params=[ToolParam(name="account", description="Account or 'all'", type="string", required=False, default="all")],
        execute=tool_email_unread_counts, category="email",
    ))

    # ── Draft tools ──────────────────────────────────────────────────

    def tool_email_list_drafts(account: str = "") -> str:
        drafts = email_service.list_drafts(account=account or None)
        return _truncate(json.dumps([
            {"id": d.id, "subject": d.message.subject, "to": d.message.recipients}
            for d in drafts
        ]))

    registry.register(Tool(
        name="email_list_drafts", description="List all email drafts.",
        params=[ToolParam(name="account", description="Account (optional)", type="string", required=False, default="")],
        execute=tool_email_list_drafts, category="email",
    ))

    def tool_email_get_draft(draft_id: str) -> str:
        draft = email_service.get_draft(draft_id)
        if not draft:
            return json.dumps({"error": "Draft not found"})
        return _truncate(json.dumps({
            "id": draft.id, "subject": draft.message.subject,
            "to": draft.message.recipients, "body": draft.message.body,
        }))

    registry.register(Tool(
        name="email_get_draft", description="Read a specific draft by ID.",
        params=[ToolParam(name="draft_id", description="Draft ID", type="string")],
        execute=tool_email_get_draft, category="email",
    ))

    def tool_email_update_draft(draft_id: str, to: str = "", subject: str = "", body: str = "", cc: str = "", bcc: str = "") -> str:
        result = email_service.update_draft(draft_id, to, subject, body)
        return json.dumps({"draft_id": result, "status": "updated"})

    registry.register(Tool(
        name="email_update_draft", description="Update an existing draft.",
        params=[
            ToolParam(name="draft_id", description="Draft ID", type="string"),
            ToolParam(name="to", description="Recipient", type="string", required=False, default=""),
            ToolParam(name="subject", description="Subject", type="string", required=False, default=""),
            ToolParam(name="body", description="Body", type="string", required=False, default=""),
            ToolParam(name="cc", description="CC (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="bcc", description="BCC (comma-separated)", type="string", required=False, default=""),
        ],
        execute=tool_email_update_draft, category="email",
    ))

    def tool_email_delete_draft(draft_id: str) -> str:
        email_service.delete_draft(draft_id)
        return json.dumps({"status": "deleted", "draft_id": draft_id})

    registry.register(Tool(
        name="email_delete_draft", description="Delete a draft permanently.",
        params=[ToolParam(name="draft_id", description="Draft ID", type="string")],
        execute=tool_email_delete_draft, category="email",
    ))

    # ── Send tools (HITL gated) ──────────────────────────────────────

    def tool_email_send(to: str, subject: str, body: str, cc: str = "", bcc: str = "", attachments: str = "", account: str = "") -> str:
        try:
            att_list = [a.strip() for a in attachments.split(",") if a.strip()] if attachments else None
            cc_list = [c.strip() for c in cc.split(",") if c.strip()] if cc else None
            bcc_list = [b.strip() for b in bcc.split(",") if b.strip()] if bcc else None
            msg_id = email_service.send_email(to=to, subject=subject, body=body, cc=cc_list, bcc=bcc_list, attachments=att_list, account=account or None)
            return json.dumps({"message_id": msg_id, "status": "sent"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    registry.register(Tool(
        name="email_send", description="Send an email directly. Requires user approval.",
        params=[
            ToolParam(name="to", description="Recipient", type="string"),
            ToolParam(name="subject", description="Subject", type="string"),
            ToolParam(name="body", description="Body", type="string"),
            ToolParam(name="cc", description="CC (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="bcc", description="BCC (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="attachments", description="File paths (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="account", description="Send from account", type="string", required=False, default=""),
        ],
        execute=tool_email_send, category="email",
    ))

    def tool_email_send_draft(draft_id: str) -> str:
        try:
            msg_id = email_service.send_draft(draft_id)
            return json.dumps({"message_id": msg_id, "status": "sent"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    registry.register(Tool(
        name="email_send_draft", description="Send an existing draft. Requires user approval.",
        params=[ToolParam(name="draft_id", description="Draft ID", type="string")],
        execute=tool_email_send_draft, category="email",
    ))

    def tool_email_reply(message_id: str, body: str, send: str = "false") -> str:
        try:
            should_send = send.lower() == "true"
            result_id = email_service.reply(message_id, body, send=should_send)
            status = "sent" if should_send else "draft_created"
            return json.dumps({"id": result_id, "status": status})
        except Exception as e:
            return json.dumps({"error": str(e)})

    registry.register(Tool(
        name="email_reply", description="Reply to an email. Creates draft by default; set send=true to send (requires approval).",
        params=[
            ToolParam(name="message_id", description="Message ID to reply to", type="string"),
            ToolParam(name="body", description="Reply body", type="string"),
            ToolParam(name="send", description="'true' to send, 'false' for draft", type="string", required=False, default="false"),
        ],
        execute=tool_email_reply, category="email",
    ))

    def tool_email_reply_all(message_id: str, body: str, send: str = "false") -> str:
        try:
            should_send = send.lower() == "true"
            result_id = email_service.reply_all(message_id, body, send=should_send)
            status = "sent" if should_send else "draft_created"
            return json.dumps({"id": result_id, "status": status})
        except Exception as e:
            return json.dumps({"error": str(e)})

    registry.register(Tool(
        name="email_reply_all", description="Reply-all to an email. Creates draft by default; set send=true to send.",
        params=[
            ToolParam(name="message_id", description="Message ID to reply to", type="string"),
            ToolParam(name="body", description="Reply body", type="string"),
            ToolParam(name="send", description="'true' to send, 'false' for draft", type="string", required=False, default="false"),
        ],
        execute=tool_email_reply_all, category="email",
    ))

    def tool_email_forward(message_id: str, to: str, body: str = "", send: str = "false") -> str:
        try:
            should_send = send.lower() == "true"
            result_id = email_service.forward(message_id, to, body, send=should_send)
            status = "sent" if should_send else "draft_created"
            return json.dumps({"id": result_id, "status": status})
        except Exception as e:
            return json.dumps({"error": str(e)})

    registry.register(Tool(
        name="email_forward", description="Forward an email. Creates draft by default; set send=true to send.",
        params=[
            ToolParam(name="message_id", description="Message ID to forward", type="string"),
            ToolParam(name="to", description="Forward to", type="string"),
            ToolParam(name="body", description="Additional message", type="string", required=False, default=""),
            ToolParam(name="send", description="'true' to send, 'false' for draft", type="string", required=False, default="false"),
        ],
        execute=tool_email_forward, category="email",
    ))

    # ── Attachment tools ─────────────────────────────────────────────

    def tool_email_attachments(message_id: str) -> str:
        attachments = email_service.get_attachments(message_id)
        return _truncate(json.dumps([
            {"id": a.id, "filename": a.filename, "mime_type": a.mime_type, "size": a.size}
            for a in attachments
        ]))

    registry.register(Tool(
        name="email_attachments", description="List attachments for an email message.",
        params=[ToolParam(name="message_id", description="Message ID", type="string")],
        execute=tool_email_attachments, category="email",
    ))

    def tool_email_download_attachment(message_id: str, attachment_id: str) -> str:
        path = email_service.download_attachment(message_id, attachment_id)
        if path:
            return json.dumps({"status": "downloaded", "path": path})
        return json.dumps({"error": "Download failed or path rejected"})

    registry.register(Tool(
        name="email_download_attachment", description="Download an email attachment to local storage.",
        params=[
            ToolParam(name="message_id", description="Message ID", type="string"),
            ToolParam(name="attachment_id", description="Attachment ID", type="string"),
        ],
        execute=tool_email_download_attachment, category="email",
    ))

    # ── Knowledge / Insight tools ────────────────────────────────────

    def tool_email_contact_insights(email_or_name: str) -> str:
        result = email_service.get_contact_insights(email_or_name)
        if not result:
            return json.dumps({"error": "Contact not found"})
        return _truncate(json.dumps({
            "email": result.email, "name": result.name,
            "organization": result.organization, "relationship": result.relationship,
            "email_count": result.email_count, "topics": result.topics,
            "pending_actions": result.pending_actions,
        }))

    registry.register(Tool(
        name="email_contact_insights", description="Get relationship history, email frequency, topics, and pending actions for a contact.",
        params=[ToolParam(name="email_or_name", description="Email address or contact name", type="string")],
        execute=tool_email_contact_insights, category="email",
    ))

    def tool_email_topic_summary(topic: str) -> str:
        return json.dumps({"summary": email_service.get_topic_summary(topic)})

    registry.register(Tool(
        name="email_topic_summary", description="Get cross-thread summary of a topic or project from email context.",
        params=[ToolParam(name="topic", description="Topic or project name", type="string")],
        execute=tool_email_topic_summary, category="email",
    ))

    def tool_email_pending_actions() -> str:
        actions = email_service.get_pending_actions()
        return _truncate(json.dumps([
            {"id": a.id, "description": a.description, "assignee": a.assignee,
             "deadline": a.deadline, "urgency": a.urgency}
            for a in actions
        ]))

    registry.register(Tool(
        name="email_pending_actions", description="List all pending action items extracted from emails, with deadlines and urgency.",
        params=[], execute=tool_email_pending_actions, category="email",
    ))

    def tool_email_briefing(days: str = "1") -> str:
        try:
            num_days = int(days)
        except (ValueError, TypeError):
            num_days = 1
        return email_service.get_email_insights(days=num_days)

    registry.register(Tool(
        name="email_briefing", description="Generate an AI-powered daily/weekly email intelligence briefing.",
        params=[ToolParam(name="days", description="Days to cover", type="string", required=False, default="1")],
        execute=tool_email_briefing, category="email",
    ))
