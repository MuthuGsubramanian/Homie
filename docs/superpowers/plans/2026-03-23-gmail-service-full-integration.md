# Gmail Service Full Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill all gaps in Homie's Gmail integration to match the full Google Apps Script Gmail Service API, wire email knowledge extraction into the knowledge graph, and expose insights to the user.

**Architecture:** Extend existing `EmailProvider` ABC → `GmailProvider` → `EmailService` → brain tools pipeline. Add `EmailKnowledgeExtractor` as a new class that supersedes `EmailIndexer`, feeding extracted entities/relationships into the existing `KnowledgeGraph`. All new methods are multi-account aware.

**Tech Stack:** Python 3.12, google-api-python-client, Pydantic, SQLite, pytest, existing homie_core knowledge graph and LLM infrastructure.

**Spec:** `docs/superpowers/specs/2026-03-23-gmail-service-full-integration-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/homie_core/email/models.py` | Data models — extend EmailThread, add EmailDraft, EmailAttachment, ContactInsight, ActionItem |
| `src/homie_core/email/provider.py` | ABC — add ~20 new abstract methods |
| `src/homie_core/email/gmail_provider.py` | Gmail API implementation of all new methods |
| `src/homie_core/email/__init__.py` | EmailService facade — ~25 new multi-account methods |
| `src/homie_core/email/tools.py` | Brain tool registrations — ~16 new tools |
| `src/homie_core/email/knowledge_extractor.py` | **New** — LLM + heuristic email knowledge extraction |
| `src/homie_core/email/sync_engine.py` | Wire knowledge extractor into sync pipeline |
| `src/homie_core/config.py` | Add EmailConfig Pydantic model |
| `src/homie_core/knowledge/models.py` | Extend entity_type and relation allowed values |
| `tests/unit/test_email/test_models.py` | Tests for new models |
| `tests/unit/test_email/test_gmail_provider.py` | Tests for new provider methods |
| `tests/unit/test_email/test_email_service.py` | Tests for new facade methods |
| `tests/unit/test_email/test_tools.py` | Tests for new brain tools |
| `tests/unit/test_email/test_knowledge_extractor.py` | Tests for extraction pipeline |

---

### Task 1: Extend Data Models

**Files:**
- Modify: `src/homie_core/email/models.py:80-91` (EmailThread)
- Modify: `src/homie_core/email/models.py` (append new dataclasses)
- Test: `tests/unit/test_email/test_models.py`

- [ ] **Step 1: Write failing tests for new models**

```python
# tests/unit/test_email/test_models.py
"""Tests for new email data models."""
from __future__ import annotations

from homie_core.email.models import (
    EmailAttachment,
    EmailDraft,
    EmailMessage,
    EmailThread,
    ActionItem,
    ContactInsight,
)


class TestEmailThreadExtension:
    def test_thread_has_messages_field(self):
        thread = EmailThread(
            id="t1", account_id="user@gmail.com", subject="Test",
            participants=["alice@x.com"], message_count=2,
            last_message_date=1710288000.0, snippet="Hello",
        )
        assert thread.messages == []

    def test_thread_with_messages(self):
        msg = EmailMessage(
            id="msg1", thread_id="t1", account_id="user@gmail.com",
            provider="gmail", subject="Test", sender="alice@x.com",
            recipients=["user@gmail.com"], snippet="Hello",
        )
        thread = EmailThread(
            id="t1", account_id="user@gmail.com", subject="Test",
            participants=["alice@x.com"], message_count=1,
            last_message_date=1710288000.0, snippet="Hello",
            messages=[msg],
        )
        assert len(thread.messages) == 1
        assert thread.messages[0].id == "msg1"


class TestEmailDraft:
    def test_draft_creation(self):
        msg = EmailMessage(
            id="msg1", thread_id="t1", account_id="user@gmail.com",
            provider="gmail", subject="Draft", sender="user@gmail.com",
            recipients=["bob@x.com"], snippet="",
        )
        draft = EmailDraft(id="d1", message=msg, updated_at=1710288000.0)
        assert draft.id == "d1"
        assert draft.message.subject == "Draft"


class TestEmailAttachment:
    def test_attachment_metadata(self):
        att = EmailAttachment(
            id="att1", message_id="msg1", filename="report.pdf",
            mime_type="application/pdf", size=1024,
        )
        assert att.data is None
        assert att.size == 1024

    def test_attachment_with_data(self):
        att = EmailAttachment(
            id="att1", message_id="msg1", filename="report.pdf",
            mime_type="application/pdf", size=5, data=b"hello",
        )
        assert att.data == b"hello"


class TestActionItem:
    def test_action_item_defaults(self):
        item = ActionItem(
            id="a1", message_id="msg1", thread_id="t1",
            description="Review PR", assignee="user@gmail.com",
            deadline=None, urgency="medium", status="pending",
            extracted_at=1710288000.0,
        )
        assert item.status == "pending"
        assert item.deadline is None


class TestContactInsight:
    def test_contact_insight(self):
        ci = ContactInsight(
            email="alice@x.com", name="Alice", organization="X Corp",
            relationship="colleague", email_count=42,
            last_contact=1710288000.0, topics=["Project Alpha"],
            pending_actions=["Review PR #123"],
        )
        assert ci.email_count == 42
        assert len(ci.topics) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_email/test_models.py -v`
Expected: ImportError for EmailDraft, EmailAttachment, ActionItem, ContactInsight; AttributeError for messages field on EmailThread

- [ ] **Step 3: Add messages field to EmailThread**

In `src/homie_core/email/models.py`, add `messages` field to existing `EmailThread` dataclass at line 90 (after `labels` field):

```python
@dataclass
class EmailThread:
    """A conversation thread grouping multiple messages."""
    id: str
    account_id: str
    subject: str
    participants: list[str]
    message_count: int
    last_message_date: float
    snippet: str
    labels: list[str] = field(default_factory=list)
    messages: list[EmailMessage] = field(default_factory=list)
```

- [ ] **Step 4: Add new dataclasses at end of models.py**

Append to `src/homie_core/email/models.py` after `SyncResult` class:

```python
@dataclass
class EmailDraft:
    """A draft email with metadata."""
    id: str
    message: EmailMessage
    updated_at: float


@dataclass
class EmailAttachment:
    """An email attachment with optional downloaded data."""
    id: str
    message_id: str
    filename: str
    mime_type: str
    size: int  # bytes
    data: bytes | None = None  # populated on download


@dataclass
class ContactInsight:
    """Aggregated contact intelligence from email patterns."""
    email: str
    name: str
    organization: str
    relationship: str  # "colleague", "client", "vendor", etc.
    email_count: int
    last_contact: float
    topics: list[str]
    pending_actions: list[str]


@dataclass
class ActionItem:
    """An action item extracted from an email."""
    id: str
    message_id: str
    thread_id: str
    description: str
    assignee: str
    deadline: float | None
    urgency: str  # "high", "medium", "low"
    status: str  # "pending", "done", "expired"
    extracted_at: float
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_email/test_models.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/email/models.py tests/unit/test_email/test_models.py
git commit -m "feat(email): add EmailDraft, EmailAttachment, ContactInsight, ActionItem models; extend EmailThread with messages field"
```

---

### Task 2: Add EmailConfig to Config System

**Files:**
- Modify: `src/homie_core/config.py:375-395` (HomieConfig)
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/unit/test_config.py
def test_email_config_defaults():
    from homie_core.config import HomieConfig
    cfg = HomieConfig()
    assert cfg.email.auto_download_attachments is True
    assert cfg.email.max_attachment_size_mb == 25
    assert cfg.email.knowledge_extraction is True
    assert cfg.email.extraction_batch_size == 20
    assert cfg.email.send_requires_confirmation is True
    assert cfg.email.insight_refresh_interval == 3600
    assert cfg.email.auto_download_categories == ["bill", "order", "work"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py::test_email_config_defaults -v`
Expected: AttributeError — HomieConfig has no `email` field

- [ ] **Step 3: Add EmailConfig model**

In `src/homie_core/config.py`, add before `HomieConfig` class (around line 373):

```python
class EmailConfig(BaseModel):
    auto_download_attachments: bool = True
    auto_download_categories: list[str] = ["bill", "order", "work"]
    max_attachment_size_mb: int = 25
    knowledge_extraction: bool = True
    extraction_batch_size: int = 20
    insight_refresh_interval: int = 3600
    send_requires_confirmation: bool = True
```

Then add to `HomieConfig`:

```python
    email: EmailConfig = Field(default_factory=EmailConfig)
```

(Add after the `model_evolution` line in `HomieConfig`)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py::test_email_config_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/config.py tests/unit/test_config.py
git commit -m "feat(config): add EmailConfig with attachment, knowledge extraction, and send confirmation settings"
```

---

### Task 3: Extend Knowledge Graph Schema

**Files:**
- Modify: `src/homie_core/knowledge/models.py:15-32` (Entity docstring), `src/homie_core/knowledge/models.py:36-51` (Relationship docstring)

- [ ] **Step 1: Update Entity entity_type docstring**

In `src/homie_core/knowledge/models.py`, update lines 17-24:

```python
@dataclass
class Entity:
    """A node in the knowledge graph.

    entity_type must be one of:
        person, project, concept, tool, document, task, event,
        location, snippet, goal, organization
    """

    name: str
    entity_type: str  # person, project, concept, tool, document, task, event, location, snippet, goal, organization
```

- [ ] **Step 2: Update Relationship relation docstring**

Update lines 36-45:

```python
@dataclass
class Relationship:
    """A directed edge in the knowledge graph.

    relation must be one of:
        authored, works_on, mentions, depends_on, contains, related_to,
        uses, supports, child_of, has_fact, works_with, reports_to,
        client_of, contacted
    """

    subject_id: str
    relation: str  # authored, works_on, mentions, depends_on, contains, related_to, uses, supports, child_of, has_fact, works_with, reports_to, client_of, contacted
```

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/knowledge/models.py
git commit -m "feat(knowledge): add organization entity type; add works_with, reports_to, client_of, contacted relations"
```

---

### Task 4: Extend EmailProvider ABC + GmailProvider Stubs (all new abstract methods)

> **IMPORTANT:** All new abstract methods are added to the ABC and stub implementations are added to GmailProvider in the SAME commit. This prevents breaking existing tests (GmailProvider becomes un-instantiable if abstract methods are added without implementations).

**Files:**
- Modify: `src/homie_core/email/provider.py:10-14` (imports), `src/homie_core/email/provider.py:91` (end of class)
- Modify: `src/homie_core/email/gmail_provider.py` (add `raise NotImplementedError` stubs for all new methods)

- [ ] **Step 1: Add thread imports and abstract methods**

Add `EmailThread` to the imports in `src/homie_core/email/provider.py`:

```python
from homie_core.email.models import (
    EmailMessage,
    EmailThread,
    HistoryChange,
    Label,
)
```

Then append these abstract methods after `create_label` (line 91):

```python
    # ── Thread operations ────────────────────────────────────────────

    @abstractmethod
    def get_thread(self, thread_id: str) -> EmailThread:
        """Fetch full thread with all messages."""

    @abstractmethod
    def list_threads(self, query: str, max_results: int = 20) -> list[EmailThread]:
        """Search/list threads."""

    @abstractmethod
    def get_inbox_threads(self, start: int = 0, max_results: int = 20) -> list[EmailThread]:
        """Get threads in INBOX."""

    @abstractmethod
    def get_starred_threads(self, start: int = 0, max_results: int = 20) -> list[EmailThread]:
        """Get starred threads."""

    @abstractmethod
    def get_spam_threads(self, start: int = 0, max_results: int = 20) -> list[EmailThread]:
        """Get spam threads."""

    @abstractmethod
    def get_trash_threads(self, start: int = 0, max_results: int = 20) -> list[EmailThread]:
        """Get trashed threads."""

    @abstractmethod
    def get_unread_count(self, label: str = "INBOX") -> int:
        """Get unread message count for a label."""

    @abstractmethod
    def archive_thread(self, thread_id: str) -> None:
        """Archive entire thread (remove INBOX label)."""

    @abstractmethod
    def trash_thread(self, thread_id: str) -> None:
        """Move entire thread to trash."""

    @abstractmethod
    def apply_label_to_thread(self, thread_id: str, label_id: str) -> None:
        """Apply label to all messages in thread."""

    @abstractmethod
    def mark_thread_read(self, thread_id: str) -> None:
        """Mark entire thread as read."""

    @abstractmethod
    def mark_thread_unread(self, thread_id: str) -> None:
        """Mark entire thread as unread."""
```

- [ ] **Step 2: DO NOT commit yet — continue to step 3 (add all remaining ABC methods)**

### (continued) Add all remaining ABC abstract methods in same task

**Files:**
- Modify: `src/homie_core/email/provider.py` (imports + end of class)

- [ ] **Step 1: Add remaining imports and abstract methods**

Add `EmailDraft, EmailAttachment` to imports:

```python
from homie_core.email.models import (
    EmailAttachment,
    EmailDraft,
    EmailMessage,
    EmailThread,
    HistoryChange,
    Label,
)
```

Append after thread methods:

```python
    # ── Draft management ─────────────────────────────────────────────

    @abstractmethod
    def list_drafts(self, max_results: int = 20) -> list[EmailDraft]:
        """List all drafts."""

    @abstractmethod
    def get_draft(self, draft_id: str) -> EmailDraft:
        """Get a single draft with message content."""

    @abstractmethod
    def update_draft(self, draft_id: str, to: str, subject: str, body: str,
                     cc: list[str] | None = None, bcc: list[str] | None = None) -> str:
        """Update an existing draft. Returns draft ID."""

    @abstractmethod
    def delete_draft(self, draft_id: str) -> None:
        """Permanently delete a draft."""

    # ── Send / Reply / Forward ───────────────────────────────────────

    @abstractmethod
    def send_email(self, to: str, subject: str, body: str,
                   cc: list[str] | None = None, bcc: list[str] | None = None,
                   attachments: list[str] | None = None,
                   reply_to_message_id: str | None = None) -> str:
        """Send email directly. Returns message ID."""

    @abstractmethod
    def send_draft(self, draft_id: str) -> str:
        """Send an existing draft. Returns message ID."""

    @abstractmethod
    def reply(self, message_id: str, body: str, send: bool = False) -> str:
        """Reply to a message. Draft by default. Returns draft/message ID."""

    @abstractmethod
    def reply_all(self, message_id: str, body: str, send: bool = False) -> str:
        """Reply-all. Draft by default. Returns draft/message ID."""

    @abstractmethod
    def forward(self, message_id: str, to: str, body: str, send: bool = False) -> str:
        """Forward a message. Draft by default. Returns draft/message ID."""

    # ── Attachments ──────────────────────────────────────────────────

    @abstractmethod
    def get_attachments(self, message_id: str) -> list[EmailAttachment]:
        """List attachment metadata for a message."""

    @abstractmethod
    def download_attachment(self, message_id: str, attachment_id: str,
                            save_path: str) -> str:
        """Download attachment to local storage. Returns file path."""

    # ── Misc ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_aliases(self) -> list[str]:
        """List send-as aliases."""

    @abstractmethod
    def star(self, message_id: str) -> None:
        """Star a message."""

    @abstractmethod
    def unstar(self, message_id: str) -> None:
        """Unstar a message."""

    @abstractmethod
    def mark_unread(self, message_id: str) -> None:
        """Mark a message as unread."""

    @abstractmethod
    def move_to_inbox(self, message_id: str) -> None:
        """Move message back to inbox (undo archive)."""

    @abstractmethod
    def delete_label(self, label_id: str) -> None:
        """Delete a user label."""

    @abstractmethod
    def update_label(self, label_id: str, new_name: str) -> Label:
        """Rename a label. Returns updated Label."""

    @abstractmethod
    def untrash(self, message_id: str) -> None:
        """Restore a message from trash."""
```

- [ ] **Step 2: Add NotImplementedError stubs to GmailProvider for all new methods**

In `src/homie_core/email/gmail_provider.py`, add stubs for every new abstract method after `create_label`. Each stub follows this pattern:

```python
    def get_thread(self, thread_id: str):
        raise NotImplementedError

    def list_threads(self, query: str, max_results: int = 20):
        raise NotImplementedError

    # ... (one stub per new abstract method — threads, drafts, send, attachments, misc)
    # Total: ~20 stubs. Each is just `raise NotImplementedError`.
```

This keeps `GmailProvider` instantiable and all existing tests passing. Tasks 6-10 replace these stubs with real implementations.

- [ ] **Step 3: Run existing tests to verify nothing breaks**

Run: `pytest tests/unit/test_email/test_gmail_provider.py -v`
Expected: All existing tests PASS (stubs are never called by existing code)

- [ ] **Step 4: Commit**

```bash
git add src/homie_core/email/provider.py src/homie_core/email/gmail_provider.py
git commit -m "feat(email): add all new abstract methods to EmailProvider ABC with GmailProvider stubs"
```

---

### Task 6: Implement GmailProvider — Thread Methods

**Files:**
- Modify: `src/homie_core/email/gmail_provider.py:13` (imports), append after `create_label` method
- Test: `tests/unit/test_email/test_gmail_provider.py`

- [ ] **Step 1: Write failing tests for thread methods**

Append to `tests/unit/test_email/test_gmail_provider.py`:

```python
from homie_core.email.models import EmailThread


class TestGmailProviderThreads:
    def test_get_thread(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().threads().get().execute.return_value = {
            "id": "t1",
            "messages": [
                _make_gmail_message(msg_id="msg1", thread_id="t1"),
                _make_gmail_message(msg_id="msg2", thread_id="t1", subject="Re: Test"),
            ],
        }

        thread = provider.get_thread("t1")
        assert thread.id == "t1"
        assert len(thread.messages) == 2
        assert thread.messages[0].id == "msg1"

    def test_get_inbox_threads(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().threads().list().execute.return_value = {
            "threads": [{"id": "t1"}, {"id": "t2"}],
        }
        service.users().threads().get().execute.side_effect = [
            {
                "id": "t1",
                "messages": [_make_gmail_message(msg_id="msg1", thread_id="t1")],
            },
            {
                "id": "t2",
                "messages": [_make_gmail_message(msg_id="msg2", thread_id="t2")],
            },
        ]

        threads = provider.get_inbox_threads(max_results=10)
        assert len(threads) == 2

    def test_get_unread_count(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().labels().get().execute.return_value = {
            "messagesUnread": 5,
        }

        count = provider.get_unread_count("INBOX")
        assert count == 5

    def test_archive_thread(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        provider.archive_thread("t1")
        service.users().threads().modify.assert_called()

    def test_trash_thread(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        provider.trash_thread("t1")
        service.users().threads().trash.assert_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_email/test_gmail_provider.py::TestGmailProviderThreads -v`
Expected: AttributeError — GmailProvider has no thread methods

- [ ] **Step 3: Implement thread methods in GmailProvider**

Update import in `src/homie_core/email/gmail_provider.py` line 13:

```python
from homie_core.email.models import EmailMessage, EmailThread, HistoryChange, Label
```

Append after `create_label` method (before `# --- Internal helpers ---` comment at line 286):

```python
    # ── Thread operations ────────────────────────────────────────────

    def get_thread(self, thread_id: str) -> EmailThread:
        self._check_token_freshness()
        raw = (
            self._service.users()
            .threads()
            .get(userId="me", id=thread_id, format="full")
            .execute()
        )
        return self._parse_thread(raw)

    def list_threads(self, query: str, max_results: int = 20) -> list[EmailThread]:
        self._check_token_freshness()
        response = (
            self._service.users()
            .threads()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        threads = []
        for t in response.get("threads", []):
            raw = (
                self._service.users()
                .threads()
                .get(userId="me", id=t["id"], format="full")
                .execute()
            )
            threads.append(self._parse_thread(raw))
        return threads

    def get_inbox_threads(self, start: int = 0, max_results: int = 20) -> list[EmailThread]:
        return self.list_threads("in:inbox", max_results=max_results)

    def get_starred_threads(self, start: int = 0, max_results: int = 20) -> list[EmailThread]:
        return self.list_threads("is:starred", max_results=max_results)

    def get_spam_threads(self, start: int = 0, max_results: int = 20) -> list[EmailThread]:
        return self.list_threads("in:spam", max_results=max_results)

    def get_trash_threads(self, start: int = 0, max_results: int = 20) -> list[EmailThread]:
        return self.list_threads("in:trash", max_results=max_results)

    def get_unread_count(self, label: str = "INBOX") -> int:
        self._check_token_freshness()
        result = (
            self._service.users()
            .labels()
            .get(userId="me", id=label)
            .execute()
        )
        return result.get("messagesUnread", 0)

    def archive_thread(self, thread_id: str) -> None:
        self._check_token_freshness()
        self._service.users().threads().modify(
            userId="me", id=thread_id,
            body={"removeLabelIds": ["INBOX"]},
        ).execute()

    def trash_thread(self, thread_id: str) -> None:
        self._check_token_freshness()
        self._service.users().threads().trash(userId="me", id=thread_id).execute()

    def apply_label_to_thread(self, thread_id: str, label_id: str) -> None:
        self._check_token_freshness()
        self._service.users().threads().modify(
            userId="me", id=thread_id,
            body={"addLabelIds": [label_id]},
        ).execute()

    def mark_thread_read(self, thread_id: str) -> None:
        self._check_token_freshness()
        self._service.users().threads().modify(
            userId="me", id=thread_id,
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()

    def mark_thread_unread(self, thread_id: str) -> None:
        self._check_token_freshness()
        self._service.users().threads().modify(
            userId="me", id=thread_id,
            body={"addLabelIds": ["UNREAD"]},
        ).execute()
```

Add `_parse_thread` helper in the internal helpers section:

```python
    def _parse_thread(self, raw: dict) -> EmailThread:
        """Parse a Gmail API thread dict into EmailThread."""
        messages = [self._parse_message(m) for m in raw.get("messages", [])]
        participants = list({m.sender for m in messages} | {r for m in messages for r in m.recipients})
        last_msg = messages[-1] if messages else None
        return EmailThread(
            id=raw["id"],
            account_id=self._account_id,
            subject=messages[0].subject if messages else "",
            participants=participants,
            message_count=len(messages),
            last_message_date=last_msg.date if last_msg else 0.0,
            snippet=last_msg.snippet if last_msg else "",
            labels=list({lbl for m in messages for lbl in m.labels}),
            messages=messages,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_email/test_gmail_provider.py::TestGmailProviderThreads -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/gmail_provider.py tests/unit/test_email/test_gmail_provider.py
git commit -m "feat(email): implement thread operations in GmailProvider"
```

---

### Task 7: Implement GmailProvider — Draft Methods

**Files:**
- Modify: `src/homie_core/email/gmail_provider.py` (imports + append methods)
- Test: `tests/unit/test_email/test_gmail_provider.py`

- [ ] **Step 1: Write failing tests**

```python
from homie_core.email.models import EmailDraft


class TestGmailProviderDrafts:
    def test_list_drafts(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().drafts().list().execute.return_value = {
            "drafts": [
                {"id": "d1", "message": _make_gmail_message(msg_id="msg1")},
            ],
        }
        service.users().drafts().get().execute.return_value = {
            "id": "d1",
            "message": _make_gmail_message(msg_id="msg1"),
        }

        drafts = provider.list_drafts()
        assert len(drafts) == 1
        assert drafts[0].id == "d1"

    def test_get_draft(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().drafts().get().execute.return_value = {
            "id": "d1",
            "message": _make_gmail_message(msg_id="msg1", subject="My Draft"),
        }

        draft = provider.get_draft("d1")
        assert draft.id == "d1"
        assert draft.message.subject == "My Draft"

    def test_delete_draft(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        provider.delete_draft("d1")
        service.users().drafts().delete.assert_called()

    def test_update_draft(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().drafts().update().execute.return_value = {"id": "d1"}

        result = provider.update_draft("d1", to="bob@x.com", subject="Updated", body="New body")
        assert result == "d1"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_email/test_gmail_provider.py::TestGmailProviderDrafts -v`

- [ ] **Step 3: Implement draft methods**

Update import to add `EmailDraft`:

```python
from homie_core.email.models import EmailDraft, EmailMessage, EmailThread, HistoryChange, Label
```

Append after thread methods:

```python
    # ── Draft management ─────────────────────────────────────────────

    def list_drafts(self, max_results: int = 20) -> list[EmailDraft]:
        self._check_token_freshness()
        response = (
            self._service.users()
            .drafts()
            .list(userId="me", maxResults=max_results)
            .execute()
        )
        drafts = []
        for d in response.get("drafts", []):
            raw = (
                self._service.users()
                .drafts()
                .get(userId="me", id=d["id"], format="full")
                .execute()
            )
            drafts.append(self._parse_draft(raw))
        return drafts

    def get_draft(self, draft_id: str) -> EmailDraft:
        self._check_token_freshness()
        raw = (
            self._service.users()
            .drafts()
            .get(userId="me", id=draft_id, format="full")
            .execute()
        )
        return self._parse_draft(raw)

    def update_draft(self, draft_id: str, to: str, subject: str, body: str,
                     cc: list[str] | None = None, bcc: list[str] | None = None) -> str:
        self._check_token_freshness()
        mime_msg = email.mime.text.MIMEText(body)
        mime_msg["To"] = to
        mime_msg["Subject"] = subject
        if cc:
            mime_msg["Cc"] = ", ".join(cc)
        if bcc:
            mime_msg["Bcc"] = ", ".join(bcc)
        encoded = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        result = (
            self._service.users()
            .drafts()
            .update(userId="me", id=draft_id, body={"message": {"raw": encoded}})
            .execute()
        )
        return result["id"]

    def delete_draft(self, draft_id: str) -> None:
        self._check_token_freshness()
        self._service.users().drafts().delete(userId="me", id=draft_id).execute()
```

Add `_parse_draft` to internal helpers:

```python
    def _parse_draft(self, raw: dict) -> EmailDraft:
        """Parse a Gmail API draft dict into EmailDraft."""
        msg_raw = raw.get("message", {})
        msg = self._parse_message(msg_raw)
        return EmailDraft(id=raw["id"], message=msg, updated_at=time.time())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_email/test_gmail_provider.py::TestGmailProviderDrafts -v`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/gmail_provider.py tests/unit/test_email/test_gmail_provider.py
git commit -m "feat(email): implement draft management in GmailProvider"
```

---

### Task 8: Implement GmailProvider — Send / Reply / Forward

**Files:**
- Modify: `src/homie_core/email/gmail_provider.py`
- Test: `tests/unit/test_email/test_gmail_provider.py`

- [ ] **Step 1: Write failing tests**

```python
class TestGmailProviderSend:
    def test_send_email(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().messages().send().execute.return_value = {"id": "sent1"}

        msg_id = provider.send_email(to="bob@x.com", subject="Hello", body="Hi Bob")
        assert msg_id == "sent1"

    def test_send_draft(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().drafts().send().execute.return_value = {"id": "sent2"}

        msg_id = provider.send_draft("d1")
        assert msg_id == "sent2"

    def test_reply_creates_draft_by_default(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        # Mock fetching original message for headers
        service.users().messages().get().execute.return_value = _make_gmail_message(
            msg_id="orig1", thread_id="t1",
        )
        service.users().drafts().create().execute.return_value = {"id": "draft_reply"}

        result = provider.reply("orig1", body="Thanks!", send=False)
        assert result == "draft_reply"

    def test_reply_sends_when_flag_true(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().messages().get().execute.return_value = _make_gmail_message(
            msg_id="orig1", thread_id="t1",
        )
        service.users().messages().send().execute.return_value = {"id": "sent_reply"}

        result = provider.reply("orig1", body="Thanks!", send=True)
        assert result == "sent_reply"

    def test_forward_creates_draft(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().messages().get().execute.return_value = _make_gmail_message(
            msg_id="orig1", thread_id="t1",
        )
        service.users().drafts().create().execute.return_value = {"id": "draft_fwd"}

        result = provider.forward("orig1", to="charlie@x.com", body="FYI", send=False)
        assert result == "draft_fwd"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_email/test_gmail_provider.py::TestGmailProviderSend -v`

- [ ] **Step 3: Implement send/reply/forward methods**

Add new imports at top of `gmail_provider.py`:

```python
import email.mime.multipart
import email.mime.base
import mimetypes
import os
```

Append after draft methods:

```python
    # ── Send / Reply / Forward ───────────────────────────────────────

    def send_email(self, to: str, subject: str, body: str,
                   cc: list[str] | None = None, bcc: list[str] | None = None,
                   attachments: list[str] | None = None,
                   reply_to_message_id: str | None = None) -> str:
        self._check_token_freshness()
        if attachments:
            mime_msg = self._build_mime_with_attachments(
                to, subject, body, cc, bcc, attachments, reply_to_message_id,
            )
        else:
            mime_msg = email.mime.text.MIMEText(body)
            mime_msg["To"] = to
            mime_msg["Subject"] = subject
            if cc:
                mime_msg["Cc"] = ", ".join(cc)
            if bcc:
                mime_msg["Bcc"] = ", ".join(bcc)
            if reply_to_message_id:
                mime_msg["In-Reply-To"] = reply_to_message_id
                mime_msg["References"] = reply_to_message_id

        encoded = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        result = (
            self._service.users()
            .messages()
            .send(userId="me", body={"raw": encoded})
            .execute()
        )
        return result["id"]

    def send_draft(self, draft_id: str) -> str:
        self._check_token_freshness()
        result = (
            self._service.users()
            .drafts()
            .send(userId="me", body={"id": draft_id})
            .execute()
        )
        return result["id"]

    def reply(self, message_id: str, body: str, send: bool = False) -> str:
        self._check_token_freshness()
        original = self._fetch_and_parse(message_id)
        raw_msg = (
            self._service.users()
            .messages()
            .get(userId="me", id=message_id, format="metadata",
                 metadataHeaders=["Message-ID", "Subject", "From"])
            .execute()
        )
        headers = raw_msg.get("payload", {}).get("headers", [])
        message_id_header = _header(headers, "Message-ID")

        mime_msg = email.mime.text.MIMEText(body)
        mime_msg["To"] = original.sender
        mime_msg["Subject"] = f"Re: {original.subject}" if not original.subject.startswith("Re:") else original.subject
        mime_msg["In-Reply-To"] = message_id_header
        mime_msg["References"] = message_id_header

        encoded = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        msg_body: dict[str, Any] = {"raw": encoded, "threadId": original.thread_id}

        if send:
            result = self._service.users().messages().send(userId="me", body=msg_body).execute()
        else:
            result = self._service.users().drafts().create(
                userId="me", body={"message": msg_body},
            ).execute()
        return result["id"]

    def reply_all(self, message_id: str, body: str, send: bool = False) -> str:
        self._check_token_freshness()
        original = self._fetch_and_parse(message_id)
        raw_msg = (
            self._service.users()
            .messages()
            .get(userId="me", id=message_id, format="metadata",
                 metadataHeaders=["Message-ID", "Subject", "From", "To", "Cc"])
            .execute()
        )
        headers = raw_msg.get("payload", {}).get("headers", [])
        message_id_header = _header(headers, "Message-ID")

        # Build recipient list: original sender + all To/Cc minus self
        all_recipients = set()
        all_recipients.add(original.sender)
        for r in original.recipients:
            all_recipients.add(r)
        all_recipients.discard(self._account_id)

        mime_msg = email.mime.text.MIMEText(body)
        mime_msg["To"] = ", ".join(all_recipients)
        mime_msg["Subject"] = f"Re: {original.subject}" if not original.subject.startswith("Re:") else original.subject
        mime_msg["In-Reply-To"] = message_id_header
        mime_msg["References"] = message_id_header

        encoded = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        msg_body: dict[str, Any] = {"raw": encoded, "threadId": original.thread_id}

        if send:
            result = self._service.users().messages().send(userId="me", body=msg_body).execute()
        else:
            result = self._service.users().drafts().create(
                userId="me", body={"message": msg_body},
            ).execute()
        return result["id"]

    def forward(self, message_id: str, to: str, body: str, send: bool = False) -> str:
        self._check_token_freshness()
        original = self._fetch_and_parse(message_id)
        original_body = self.fetch_message_body(message_id)

        forward_body = f"{body}\n\n---------- Forwarded message ----------\nFrom: {original.sender}\nSubject: {original.subject}\n\n{original_body}"

        mime_msg = email.mime.text.MIMEText(forward_body)
        mime_msg["To"] = to
        mime_msg["Subject"] = f"Fwd: {original.subject}" if not original.subject.startswith("Fwd:") else original.subject

        encoded = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        msg_body: dict[str, Any] = {"raw": encoded}

        if send:
            result = self._service.users().messages().send(userId="me", body=msg_body).execute()
        else:
            result = self._service.users().drafts().create(
                userId="me", body={"message": msg_body},
            ).execute()
        return result["id"]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_email/test_gmail_provider.py::TestGmailProviderSend -v`
Expected: All 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/gmail_provider.py tests/unit/test_email/test_gmail_provider.py
git commit -m "feat(email): implement send, reply, reply_all, forward in GmailProvider"
```

---

### Task 9: Implement GmailProvider — Attachments

**Files:**
- Modify: `src/homie_core/email/gmail_provider.py`
- Test: `tests/unit/test_email/test_gmail_provider.py`

- [ ] **Step 1: Write failing tests**

```python
from homie_core.email.models import EmailAttachment


class TestGmailProviderAttachments:
    def test_get_attachments(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().messages().get().execute.return_value = {
            "id": "msg1", "threadId": "t1", "snippet": "",
            "labelIds": ["INBOX"], "internalDate": "1710288000000",
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Test"},
                    {"name": "From", "value": "alice@x.com"},
                    {"name": "To", "value": "user@gmail.com"},
                ],
                "parts": [
                    {"filename": "report.pdf", "mimeType": "application/pdf",
                     "body": {"attachmentId": "att1", "size": 1024}},
                    {"filename": "photo.jpg", "mimeType": "image/jpeg",
                     "body": {"attachmentId": "att2", "size": 2048}},
                ],
            },
        }

        attachments = provider.get_attachments("msg1")
        assert len(attachments) == 2
        assert attachments[0].filename == "report.pdf"
        assert attachments[0].size == 1024

    def test_download_attachment(self, tmp_path):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().messages().attachments().get().execute.return_value = {
            "data": base64.urlsafe_b64encode(b"file content").decode(),
        }

        save_dir = str(tmp_path / "attachments")
        path = provider.download_attachment("msg1", "att1", save_dir)
        assert os.path.exists(path)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_email/test_gmail_provider.py::TestGmailProviderAttachments -v`

- [ ] **Step 3: Implement attachment methods**

Add `EmailAttachment` to imports and append:

```python
    # ── Attachments ──────────────────────────────────────────────────

    def get_attachments(self, message_id: str) -> list[EmailAttachment]:
        self._check_token_freshness()
        raw = (
            self._service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        attachments = []
        for part in raw.get("payload", {}).get("parts", []):
            filename = part.get("filename", "")
            if filename:
                body = part.get("body", {})
                attachments.append(EmailAttachment(
                    id=body.get("attachmentId", ""),
                    message_id=message_id,
                    filename=filename,
                    mime_type=part.get("mimeType", "application/octet-stream"),
                    size=body.get("size", 0),
                ))
        return attachments

    def download_attachment(self, message_id: str, attachment_id: str,
                            save_path: str) -> str:
        self._check_token_freshness()
        result = (
            self._service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message_id, id=attachment_id)
            .execute()
        )
        data = base64.urlsafe_b64decode(result["data"])
        os.makedirs(save_path, exist_ok=True)
        # Use attachment_id as fallback filename
        file_path = os.path.join(save_path, attachment_id)
        with open(file_path, "wb") as f:
            f.write(data)
        return file_path

    def _build_mime_with_attachments(
        self, to: str, subject: str, body: str,
        cc: list[str] | None, bcc: list[str] | None,
        file_paths: list[str],
        reply_to_message_id: str | None = None,
    ) -> email.mime.multipart.MIMEMultipart:
        """Build multipart MIME message with file attachments."""
        msg = email.mime.multipart.MIMEMultipart()
        msg["To"] = to
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = ", ".join(cc)
        if bcc:
            msg["Bcc"] = ", ".join(bcc)
        if reply_to_message_id:
            msg["In-Reply-To"] = reply_to_message_id
            msg["References"] = reply_to_message_id

        msg.attach(email.mime.text.MIMEText(body))

        for path in file_paths:
            content_type, _ = mimetypes.guess_type(path)
            if content_type is None:
                content_type = "application/octet-stream"
            maintype, subtype = content_type.split("/", 1)
            with open(path, "rb") as f:
                attachment = email.mime.base.MIMEBase(maintype, subtype)
                attachment.set_payload(f.read())
            import email.encoders
            email.encoders.encode_base64(attachment)
            attachment.add_header(
                "Content-Disposition", "attachment",
                filename=os.path.basename(path),
            )
            msg.attach(attachment)
        return msg
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_email/test_gmail_provider.py::TestGmailProviderAttachments -v`
Expected: All 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/gmail_provider.py tests/unit/test_email/test_gmail_provider.py
git commit -m "feat(email): implement attachment operations in GmailProvider"
```

---

### Task 10: Implement GmailProvider — Misc Methods (star, unstar, aliases, etc.)

**Files:**
- Modify: `src/homie_core/email/gmail_provider.py`
- Test: `tests/unit/test_email/test_gmail_provider.py`

- [ ] **Step 1: Write failing tests**

```python
class TestGmailProviderMisc:
    def test_star(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        provider.star("msg1")
        service.users().messages().modify.assert_called()

    def test_unstar(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        provider.unstar("msg1")
        service.users().messages().modify.assert_called()

    def test_mark_unread(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        provider.mark_unread("msg1")
        service.users().messages().modify.assert_called()

    def test_move_to_inbox(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        provider.move_to_inbox("msg1")
        service.users().messages().modify.assert_called()

    def test_untrash(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        provider.untrash("msg1")
        service.users().messages().untrash.assert_called()

    def test_get_aliases(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().settings().sendAs().list().execute.return_value = {
            "sendAs": [
                {"sendAsEmail": "user@gmail.com"},
                {"sendAsEmail": "alias@gmail.com"},
            ],
        }
        aliases = provider.get_aliases()
        assert len(aliases) == 2

    def test_delete_label(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        provider.delete_label("Label_1")
        service.users().labels().delete.assert_called()

    def test_update_label(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().labels().update().execute.return_value = {
            "id": "Label_1", "name": "NewName", "type": "user",
        }
        label = provider.update_label("Label_1", "NewName")
        assert label.name == "NewName"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_email/test_gmail_provider.py::TestGmailProviderMisc -v`

- [ ] **Step 3: Implement misc methods**

```python
    # ── Misc ─────────────────────────────────────────────────────────

    def get_aliases(self) -> list[str]:
        self._check_token_freshness()
        result = (
            self._service.users()
            .settings()
            .sendAs()
            .list(userId="me")
            .execute()
        )
        return [a["sendAsEmail"] for a in result.get("sendAs", [])]

    def star(self, message_id: str) -> None:
        self._check_token_freshness()
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"addLabelIds": ["STARRED"]},
        ).execute()

    def unstar(self, message_id: str) -> None:
        self._check_token_freshness()
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"removeLabelIds": ["STARRED"]},
        ).execute()

    def mark_unread(self, message_id: str) -> None:
        self._check_token_freshness()
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"addLabelIds": ["UNREAD"]},
        ).execute()

    def move_to_inbox(self, message_id: str) -> None:
        self._check_token_freshness()
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"addLabelIds": ["INBOX"]},
        ).execute()

    def delete_label(self, label_id: str) -> None:
        self._check_token_freshness()
        self._service.users().labels().delete(userId="me", id=label_id).execute()

    def update_label(self, label_id: str, new_name: str) -> Label:
        self._check_token_freshness()
        result = self._service.users().labels().update(
            userId="me", id=label_id,
            body={"name": new_name},
        ).execute()
        return Label(id=result["id"], name=result["name"], type=result.get("type", "user"))

    def untrash(self, message_id: str) -> None:
        self._check_token_freshness()
        self._service.users().messages().untrash(userId="me", id=message_id).execute()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_email/test_gmail_provider.py::TestGmailProviderMisc -v`
Expected: All 8 PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/gmail_provider.py tests/unit/test_email/test_gmail_provider.py
git commit -m "feat(email): implement star, unstar, aliases, label management, untrash in GmailProvider"
```

---

### Task 11: Extend EmailService Facade — Thread & Draft Methods

**Files:**
- Modify: `src/homie_core/email/__init__.py`
- Test: `tests/unit/test_email/test_email_service.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_email/test_email_service.py
"""Tests for new EmailService facade methods."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

from homie_core.email import EmailService
from homie_core.email.models import EmailDraft, EmailMessage, EmailThread


def _make_service():
    vault = MagicMock()
    vault.list_credentials.return_value = []
    conn = sqlite3.connect(":memory:")
    svc = EmailService(vault=vault, cache_conn=conn)
    return svc


def _make_service_with_provider():
    svc = _make_service()
    provider = MagicMock()
    svc._providers["user@gmail.com"] = provider
    return svc, provider


class TestEmailServiceThreads:
    def test_get_thread_messages(self):
        svc, provider = _make_service_with_provider()
        thread = EmailThread(
            id="t1", account_id="user@gmail.com", subject="Test",
            participants=[], message_count=1, last_message_date=0.0,
            snippet="",
        )
        provider.get_thread.return_value = thread

        result = svc.get_thread_messages("t1")
        assert result.id == "t1"

    def test_list_inbox_threads(self):
        svc, provider = _make_service_with_provider()
        provider.get_inbox_threads.return_value = []

        result = svc.list_inbox_threads()
        assert result == []

    def test_get_unread_counts(self):
        svc, provider = _make_service_with_provider()
        provider.get_unread_count.side_effect = lambda label: {"INBOX": 5, "SPAM": 2, "STARRED": 1}.get(label, 0)

        counts = svc.get_unread_counts()
        assert counts["inbox"] == 5
        assert counts["spam"] == 2


class TestEmailServiceDrafts:
    def test_list_drafts(self):
        svc, provider = _make_service_with_provider()
        provider.list_drafts.return_value = []
        assert svc.list_drafts() == []

    def test_delete_draft(self):
        svc, provider = _make_service_with_provider()
        svc.delete_draft("d1")
        provider.delete_draft.assert_called_with("d1")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_email/test_email_service.py -v`

- [ ] **Step 3: Implement facade methods**

Append to `EmailService` class in `src/homie_core/email/__init__.py`:

```python
    # ── Thread operations ────────────────────────────────────────────

    def get_thread_messages(self, thread_id: str, account: str | None = None) -> EmailThread | None:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                return provider.get_thread(thread_id)
            except Exception:
                continue
        return None

    def list_inbox_threads(self, account: str | None = None, start: int = 0,
                           max_results: int = 20) -> list:
        threads = []
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                threads.extend(provider.get_inbox_threads(start, max_results))
            except Exception:
                pass
        return threads[:max_results]

    def list_starred_threads(self, account: str | None = None, start: int = 0,
                             max_results: int = 20) -> list:
        threads = []
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                threads.extend(provider.get_starred_threads(start, max_results))
            except Exception:
                pass
        return threads[:max_results]

    def list_spam_threads(self, account: str | None = None, start: int = 0,
                          max_results: int = 20) -> list:
        threads = []
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                threads.extend(provider.get_spam_threads(start, max_results))
            except Exception:
                pass
        return threads[:max_results]

    def get_unread_counts(self, account: str | None = None) -> dict:
        totals = {"inbox": 0, "spam": 0, "starred": 0, "total": 0}
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                inbox = provider.get_unread_count("INBOX")
                spam = provider.get_unread_count("SPAM")
                starred = provider.get_unread_count("STARRED")
                totals["inbox"] += inbox
                totals["spam"] += spam
                totals["starred"] += starred
                totals["total"] += inbox + spam
            except Exception:
                pass
        return totals

    # ── Draft management ─────────────────────────────────────────────

    def list_drafts(self, account: str | None = None) -> list:
        drafts = []
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                drafts.extend(provider.list_drafts())
            except Exception:
                pass
        return drafts

    def get_draft(self, draft_id: str, account: str | None = None):
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                return provider.get_draft(draft_id)
            except Exception:
                continue
        return None

    def update_draft(self, draft_id: str, to: str, subject: str, body: str,
                     account: str | None = None) -> str:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.update_draft(draft_id, to, subject, body)
        return ""

    def delete_draft(self, draft_id: str, account: str | None = None) -> None:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                provider.delete_draft(draft_id)
                return
            except Exception:
                continue
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_email/test_email_service.py -v`
Expected: All 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/__init__.py tests/unit/test_email/test_email_service.py
git commit -m "feat(email): add thread and draft facade methods to EmailService"
```

---

### Task 12: Extend EmailService Facade — Send, Attachments, Insights

**Files:**
- Modify: `src/homie_core/email/__init__.py`
- Test: `tests/unit/test_email/test_email_service.py`

- [ ] **Step 1: Write failing tests**

```python
class TestEmailServiceSend:
    def test_send_email(self):
        svc, provider = _make_service_with_provider()
        provider.send_email.return_value = "sent1"
        result = svc.send_email(to="bob@x.com", subject="Hi", body="Hello")
        assert result == "sent1"

    def test_reply_default_draft(self):
        svc, provider = _make_service_with_provider()
        provider.reply.return_value = "draft1"
        result = svc.reply("msg1", body="Thanks")
        provider.reply.assert_called_with("msg1", "Thanks", False)

    def test_forward(self):
        svc, provider = _make_service_with_provider()
        provider.forward.return_value = "fwd1"
        result = svc.forward("msg1", to="charlie@x.com", body="FYI")
        assert result == "fwd1"


class TestEmailServiceAttachments:
    def test_get_attachments(self):
        svc, provider = _make_service_with_provider()
        provider.get_attachments.return_value = []
        result = svc.get_attachments("msg1")
        assert result == []

    def test_download_attachment_path_traversal_rejected(self):
        svc, provider = _make_service_with_provider()
        result = svc.download_attachment("msg1", "att1", "../../etc/passwd")
        assert result == "" or "error" in str(result).lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_email/test_email_service.py::TestEmailServiceSend -v`

- [ ] **Step 3: Implement send, attachment, and insight facade methods**

Append to `EmailService`:

```python
    # ── Send / Reply / Forward ───────────────────────────────────────

    def send_email(self, to: str, subject: str, body: str,
                   cc: list[str] | None = None, bcc: list[str] | None = None,
                   attachments: list[str] | None = None,
                   account: str | None = None) -> str:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.send_email(to, subject, body, cc, bcc, attachments)
        return ""

    def send_draft(self, draft_id: str, account: str | None = None) -> str:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.send_draft(draft_id)
        return ""

    def reply(self, message_id: str, body: str, send: bool = False,
              account: str | None = None) -> str:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.reply(message_id, body, send)
        return ""

    def reply_all(self, message_id: str, body: str, send: bool = False,
                  account: str | None = None) -> str:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.reply_all(message_id, body, send)
        return ""

    def forward(self, message_id: str, to: str, body: str, send: bool = False,
                account: str | None = None) -> str:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.forward(message_id, to, body, send)
        return ""

    # ── Attachments ──────────────────────────────────────────────────

    def get_attachments(self, message_id: str, account: str | None = None) -> list:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                return provider.get_attachments(message_id)
            except Exception:
                continue
        return []

    def download_attachment(self, message_id: str, attachment_id: str,
                            save_path: str | None = None,
                            account: str | None = None) -> str:
        from pathlib import Path
        base = Path.home() / ".homie" / "attachments"
        if save_path:
            resolved = Path(save_path).resolve()
            if not str(resolved).startswith(str(base.resolve())):
                return ""  # path traversal rejected
            target = str(resolved)
        else:
            target = str(base / message_id)

        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                return provider.download_attachment(message_id, attachment_id, target)
            except Exception:
                continue
        return ""

    # ── Knowledge & Insights ─────────────────────────────────────────

    def get_contact_insights(self, email_or_name: str):
        row = self._conn.execute(
            "SELECT email, name, organization, relationship, email_count, "
            "last_contact, topics, entity_id FROM email_contacts "
            "WHERE email=? OR name LIKE ?",
            (email_or_name, f"%{email_or_name}%"),
        ).fetchone()
        if not row:
            return None
        from homie_core.email.models import ContactInsight
        actions = self._conn.execute(
            "SELECT description FROM email_action_items WHERE assignee=? AND status='pending'",
            (row[0],),
        ).fetchall()
        return ContactInsight(
            email=row[0], name=row[1] or "", organization=row[2] or "",
            relationship=row[3] or "", email_count=row[4] or 0,
            last_contact=row[5] or 0.0,
            topics=json.loads(row[6] or "[]"),
            pending_actions=[a[0] for a in actions],
        )

    def get_pending_actions(self) -> list:
        from homie_core.email.models import ActionItem
        rows = self._conn.execute(
            "SELECT id, message_id, thread_id, description, assignee, deadline, "
            "urgency, status, extracted_at FROM email_action_items "
            "WHERE status='pending' ORDER BY deadline ASC NULLS LAST",
        ).fetchall()
        return [
            ActionItem(id=r[0], message_id=r[1], thread_id=r[2] or "",
                       description=r[3], assignee=r[4] or "", deadline=r[5],
                       urgency=r[6] or "medium", status=r[7], extracted_at=r[8] or 0.0)
            for r in rows
        ]

    def get_topic_summary(self, topic: str) -> str:
        row = self._conn.execute(
            "SELECT thread_ids, message_count FROM email_topics WHERE name LIKE ?",
            (f"%{topic}%",),
        ).fetchone()
        if not row:
            return f"No email threads found for topic '{topic}'"
        thread_ids = json.loads(row[0] or "[]")
        return f"Topic '{topic}': {row[1]} messages across {len(thread_ids)} threads"

    def get_email_insights(self, days: int = 1) -> str:
        """Generate intelligence briefing combining digest + pending actions."""
        # Use existing get_intelligent_digest if available (defined in EmailService)
        try:
            digest = self.get_intelligent_digest(days=days)
            if isinstance(digest, str):
                return digest
            return json.dumps(digest)
        except AttributeError:
            # Fallback: structured summary
            summary = self.get_summary(days=days)
            actions = self.get_pending_actions()
            result = {**summary, "pending_actions": [
                {"description": a.description, "urgency": a.urgency, "deadline": a.deadline}
                for a in actions[:10]
            ]}
            return json.dumps(result)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_email/test_email_service.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/__init__.py tests/unit/test_email/test_email_service.py
git commit -m "feat(email): add send, attachment, and insight facade methods to EmailService"
```

---

### Task 13: Create EmailKnowledgeExtractor

**Files:**
- Create: `src/homie_core/email/knowledge_extractor.py`
- Test: `tests/unit/test_email/test_knowledge_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_email/test_knowledge_extractor.py
"""Tests for EmailKnowledgeExtractor."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

from homie_core.email.models import EmailMessage


def _make_msg(msg_id="msg1", sender="alice@corp.com", subject="Q1 Review",
              recipients=None, snippet="Please review the Q1 report"):
    return EmailMessage(
        id=msg_id, thread_id="t1", account_id="user@gmail.com",
        provider="gmail", subject=subject, sender=sender,
        recipients=recipients or ["user@gmail.com"],
        snippet=snippet, body=snippet,
    )


class TestHeuristicExtraction:
    def test_extract_contacts(self):
        from homie_core.email.knowledge_extractor import EmailKnowledgeExtractor
        conn = sqlite3.connect(":memory:")
        extractor = EmailKnowledgeExtractor(cache_conn=conn, graph=None)
        msg = _make_msg()
        result = extractor._extract_heuristic(msg)
        assert any(e["name"] == "alice" or "alice" in e.get("email", "")
                    for e in result.get("entities", []))

    def test_extract_organization_from_domain(self):
        from homie_core.email.knowledge_extractor import EmailKnowledgeExtractor
        conn = sqlite3.connect(":memory:")
        extractor = EmailKnowledgeExtractor(cache_conn=conn, graph=None)
        msg = _make_msg(sender="bob@acmecorp.com")
        result = extractor._extract_heuristic(msg)
        orgs = [e for e in result.get("entities", []) if e.get("type") == "organization"]
        assert len(orgs) >= 1

    def test_extract_deadline_pattern(self):
        from homie_core.email.knowledge_extractor import EmailKnowledgeExtractor
        conn = sqlite3.connect(":memory:")
        extractor = EmailKnowledgeExtractor(cache_conn=conn, graph=None)
        msg = _make_msg(snippet="Please submit by Friday March 28")
        result = extractor._extract_heuristic(msg)
        assert len(result.get("action_items", [])) >= 1 or len(result.get("events", [])) >= 1


class TestProcessMessage:
    def test_process_stores_contact(self):
        from homie_core.email.knowledge_extractor import EmailKnowledgeExtractor
        conn = sqlite3.connect(":memory:")
        extractor = EmailKnowledgeExtractor(cache_conn=conn, graph=None)
        extractor._ensure_tables()
        msg = _make_msg()
        extractor.process_message(msg)
        row = conn.execute("SELECT email FROM email_contacts WHERE email LIKE '%alice%'").fetchone()
        assert row is not None


class TestBatchExtract:
    def test_batch_processes_multiple(self):
        from homie_core.email.knowledge_extractor import EmailKnowledgeExtractor
        conn = sqlite3.connect(":memory:")
        extractor = EmailKnowledgeExtractor(cache_conn=conn, graph=None)
        extractor._ensure_tables()
        msgs = [_make_msg(msg_id=f"msg{i}", sender=f"person{i}@x.com") for i in range(5)]
        extractor.batch_extract(msgs)
        count = conn.execute("SELECT COUNT(*) FROM email_contacts").fetchone()[0]
        assert count >= 5
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_email/test_knowledge_extractor.py -v`
Expected: ImportError

- [ ] **Step 3: Implement EmailKnowledgeExtractor**

```python
# src/homie_core/email/knowledge_extractor.py
"""Email knowledge extraction — heuristic + LLM-powered entity/relationship extraction.

Supersedes homie_core/knowledge/email_indexer.py.
Feeds extracted entities and relationships into the knowledge graph.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from typing import Any, Optional

from homie_core.email.models import EmailMessage

_log = logging.getLogger(__name__)

# Common free email providers (not useful as organization names)
_FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "icloud.com", "mail.com", "protonmail.com", "zoho.com", "yandex.com",
}

# Deadline patterns
_DEADLINE_PATTERNS = [
    re.compile(r"(?:by|before|due|deadline)\s+(\w+\s+\d{1,2}(?:,?\s*\d{4})?)", re.IGNORECASE),
    re.compile(r"(?:by|before|due)\s+(tomorrow|today|end of (?:day|week|month))", re.IGNORECASE),
    re.compile(r"(?:by|before|due)\s+((?:mon|tues|wednes|thurs|fri|satur|sun)day)", re.IGNORECASE),
]

# Action request patterns
_ACTION_PATTERNS = [
    re.compile(r"(?:please|could you|can you|kindly)\s+(.{10,80}?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:action required|action needed)[:\s]+(.{10,80}?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:need you to|asking you to)\s+(.{10,80}?)(?:\.|$)", re.IGNORECASE),
]

_LLM_EXTRACT_PROMPT = """\
Extract structured knowledge from this email. Return ONLY a JSON object.

From: {sender}
To: {recipients}
Subject: {subject}
Body:
{body}

Return:
{{"entities": [{{"name": "<name>", "type": "<person|organization|project>", "email": "<if person>"}}],
"relationships": [{{"from": "<entity name>", "to": "<entity name>", "relation": "<works_with|reports_to|client_of|contacted>"}}],
"action_items": [{{"description": "<what needs to be done>", "assignee": "<who>", "deadline": "<date or null>", "urgency": "<high|medium|low>"}}],
"events": [{{"description": "<event>", "date": "<date or null>"}}],
"topics": ["<topic1>", "<topic2>"]}}"""


def _extract_email_address(sender: str) -> str:
    match = re.search(r"[\w.+-]+@[\w.-]+", sender)
    return match.group(0).lower() if match else sender.lower()


def _extract_name(sender: str) -> str:
    name = sender.split("<")[0].strip().strip('"')
    if not name or "@" in name:
        email = _extract_email_address(sender)
        name = email.split("@")[0]
    return name


def _extract_domain(email_addr: str) -> str:
    parts = email_addr.split("@")
    return parts[1].lower() if len(parts) == 2 else ""


class EmailKnowledgeExtractor:
    """Extracts entities, relationships, actions, and topics from emails."""

    def __init__(
        self,
        cache_conn: sqlite3.Connection,
        graph=None,
        model_engine=None,
    ):
        self._conn = cache_conn
        self._graph = graph
        self._model = model_engine
        self._ensure_tables()

    @property
    def has_llm(self) -> bool:
        return self._model is not None and hasattr(self._model, "is_loaded") and self._model.is_loaded

    def _ensure_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS email_threads (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                subject TEXT,
                participants TEXT,
                message_count INTEGER DEFAULT 0,
                last_message_date REAL,
                snippet TEXT,
                labels TEXT,
                fetched_at REAL
            );
            CREATE TABLE IF NOT EXISTS email_drafts (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                to_addr TEXT,
                subject TEXT,
                body TEXT,
                cc TEXT,
                bcc TEXT,
                reply_to_message_id TEXT,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS email_attachments (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                account_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                mime_type TEXT,
                size INTEGER DEFAULT 0,
                local_path TEXT,
                downloaded_at REAL
            );
            CREATE TABLE IF NOT EXISTS email_contacts (
                email TEXT PRIMARY KEY,
                name TEXT,
                organization TEXT,
                relationship TEXT,
                email_count INTEGER DEFAULT 0,
                last_contact REAL,
                topics TEXT,
                entity_id TEXT,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS email_action_items (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                thread_id TEXT,
                account_id TEXT NOT NULL,
                description TEXT NOT NULL,
                assignee TEXT,
                deadline REAL,
                urgency TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'pending',
                extracted_at REAL,
                completed_at REAL
            );
            CREATE TABLE IF NOT EXISTS email_topics (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                thread_ids TEXT,
                entity_ids TEXT,
                message_count INTEGER DEFAULT 0,
                last_activity REAL,
                updated_at REAL
            );
        """)

    def process_message(self, msg: EmailMessage) -> None:
        """Extract knowledge from a single message and store."""
        if self.has_llm:
            result = self._extract_llm(msg)
            if not result:
                result = self._extract_heuristic(msg)
        else:
            result = self._extract_heuristic(msg)

        self._store_extraction(msg, result)

    def batch_extract(self, messages: list[EmailMessage], batch_size: int = 20) -> None:
        """Batch-extract knowledge from multiple messages."""
        for msg in messages:
            self.process_message(msg)

    def build_contact_graph(self) -> None:
        """Build contact relationships in the knowledge graph from stored contacts."""
        if not self._graph:
            return
        from homie_core.knowledge.models import Entity, Relationship

        rows = self._conn.execute(
            "SELECT email, name, organization, relationship FROM email_contacts"
        ).fetchall()

        for row in rows:
            email_addr, name, org, rel = row
            person = Entity(
                name=name or email_addr,
                entity_type="person",
                attributes={"email": email_addr, "organization": org or ""},
                source="email_sync",
            )
            entity_id = self._graph.merge_entity(person)
            self._conn.execute(
                "UPDATE email_contacts SET entity_id=? WHERE email=?",
                (entity_id, email_addr),
            )

            if org and org.lower() not in _FREE_EMAIL_DOMAINS:
                org_entity = Entity(
                    name=org, entity_type="organization",
                    attributes={}, source="email_sync",
                )
                org_id = self._graph.merge_entity(org_entity)
                self._graph.add_relationship(Relationship(
                    subject_id=entity_id,
                    relation="works_with",
                    object_id=org_id,
                    source="email_sync",
                ))

        self._conn.commit()

    def build_topic_clusters(self) -> None:
        """Group threads by topic and store."""
        # Topics are already populated per-message via _store_extraction
        pass

    def _extract_heuristic(self, msg: EmailMessage) -> dict[str, Any]:
        """Fast heuristic extraction — no LLM needed."""
        entities = []
        action_items = []
        events = []
        topics = []

        # Extract sender as person entity
        email_addr = _extract_email_address(msg.sender)
        name = _extract_name(msg.sender)
        entities.append({"name": name, "type": "person", "email": email_addr})

        # Extract organization from domain
        domain = _extract_domain(email_addr)
        if domain and domain not in _FREE_EMAIL_DOMAINS:
            org_name = domain.split(".")[0].title()
            entities.append({"name": org_name, "type": "organization", "domain": domain})

        # Extract recipients as person entities
        for r in msg.recipients:
            r_email = _extract_email_address(r)
            r_name = _extract_name(r)
            entities.append({"name": r_name, "type": "person", "email": r_email})

        # Deadline detection
        text = f"{msg.subject} {msg.body or msg.snippet or ''}"
        for pattern in _DEADLINE_PATTERNS:
            match = pattern.search(text)
            if match:
                action_items.append({
                    "description": f"Deadline: {match.group(0).strip()}",
                    "assignee": "",
                    "deadline": None,
                    "urgency": "high",
                })
                break

        # Action item detection
        for pattern in _ACTION_PATTERNS:
            match = pattern.search(text)
            if match:
                action_items.append({
                    "description": match.group(1).strip(),
                    "assignee": "",
                    "deadline": None,
                    "urgency": "medium",
                })
                break

        # Simple topic extraction from subject
        subject_clean = re.sub(r"^(Re:|Fwd:|FW:)\s*", "", msg.subject, flags=re.IGNORECASE).strip()
        if subject_clean:
            topics.append(subject_clean)

        return {
            "entities": entities,
            "relationships": [],
            "action_items": action_items,
            "events": events,
            "topics": topics,
        }

    def _extract_llm(self, msg: EmailMessage) -> dict[str, Any] | None:
        """LLM-powered deep extraction."""
        if not self.has_llm:
            return None

        from homie_core.email.classifier import _email_text, _parse_llm_json

        prompt = _LLM_EXTRACT_PROMPT.format(
            sender=msg.sender,
            recipients=", ".join(msg.recipients[:5]),
            subject=msg.subject,
            body=_email_text(msg),
        )

        try:
            raw = self._model.generate(prompt, max_tokens=512, temperature=0.1, timeout=30)
            result = _parse_llm_json(raw)
            if isinstance(result, dict):
                return result
        except Exception as e:
            _log.debug("LLM extraction failed: %s", e)

        return None

    def _store_extraction(self, msg: EmailMessage, result: dict[str, Any]) -> None:
        """Store extracted knowledge in cache tables and knowledge graph."""
        now = time.time()

        # Store contacts
        for entity in result.get("entities", []):
            if entity.get("type") == "person" and entity.get("email"):
                email_addr = entity["email"]
                self._conn.execute(
                    """INSERT INTO email_contacts (email, name, organization, email_count, last_contact, topics, updated_at)
                       VALUES (?, ?, ?, 1, ?, '[]', ?)
                       ON CONFLICT(email) DO UPDATE SET
                           email_count = email_count + 1,
                           last_contact = ?,
                           updated_at = ?""",
                    (email_addr, entity.get("name", ""), "", now, now, now, now),
                )

        # Store action items
        for action in result.get("action_items", []):
            self._conn.execute(
                """INSERT OR IGNORE INTO email_action_items
                   (id, message_id, thread_id, account_id, description, assignee,
                    deadline, urgency, status, extracted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (str(uuid.uuid4()), msg.id, msg.thread_id, msg.account_id,
                 action.get("description", ""), action.get("assignee", ""),
                 action.get("deadline"), action.get("urgency", "medium"), now),
            )

        # Store/update topics
        for topic in result.get("topics", []):
            existing = self._conn.execute(
                "SELECT id, thread_ids, message_count FROM email_topics WHERE name=?",
                (topic,),
            ).fetchone()
            if existing:
                thread_ids = json.loads(existing[1] or "[]")
                if msg.thread_id not in thread_ids:
                    thread_ids.append(msg.thread_id)
                self._conn.execute(
                    "UPDATE email_topics SET thread_ids=?, message_count=?, last_activity=?, updated_at=? WHERE id=?",
                    (json.dumps(thread_ids), existing[2] + 1, now, now, existing[0]),
                )
            else:
                self._conn.execute(
                    "INSERT INTO email_topics (id, name, thread_ids, entity_ids, message_count, last_activity, updated_at) VALUES (?, ?, ?, '[]', 1, ?, ?)",
                    (str(uuid.uuid4()), topic, json.dumps([msg.thread_id]), now, now),
                )

        # Feed into knowledge graph
        if self._graph:
            from homie_core.knowledge.models import Entity, Relationship
            for entity in result.get("entities", []):
                etype = entity.get("type", "person")
                if etype not in ("person", "organization", "project"):
                    continue
                e = Entity(
                    name=entity.get("name", ""),
                    entity_type=etype,
                    attributes={k: v for k, v in entity.items() if k not in ("name", "type")},
                    source="email_sync",
                )
                self._graph.merge_entity(e)

        self._conn.commit()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_email/test_knowledge_extractor.py -v`
Expected: All 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/knowledge_extractor.py tests/unit/test_email/test_knowledge_extractor.py
git commit -m "feat(email): add EmailKnowledgeExtractor with heuristic + LLM extraction"
```

---

### Task 14: Wire Knowledge Extractor into Sync Pipeline

**Files:**
- Modify: `src/homie_core/email/sync_engine.py:31-50` (constructor), `src/homie_core/email/sync_engine.py:53-78` (initial_sync), `src/homie_core/email/sync_engine.py:142-210` (_classify_and_organize)
- Modify: `src/homie_core/email/__init__.py:36-91` (initialize method)

- [ ] **Step 1: Update SyncEngine constructor to accept knowledge_extractor**

In `src/homie_core/email/sync_engine.py`, update `__init__`:

```python
    def __init__(
        self,
        provider: EmailProvider,
        classifier: EmailClassifier,
        cache_conn: sqlite3.Connection,
        account_id: str,
        organizer=None,
        vault=None,
        working_memory=None,
        knowledge_extractor=None,
    ):
        self._provider = provider
        self._classifier = classifier
        self._conn = cache_conn
        self._account_id = account_id
        self._organizer = organizer
        self._vault = vault
        self._working_memory = working_memory
        self._llm_enabled = classifier.has_llm
        self._knowledge_extractor = knowledge_extractor
```

- [ ] **Step 2: Wire into _classify_and_organize**

At end of `_classify_and_organize` method, after `self._store_message(msg)` (line 209), add:

```python
        # Knowledge extraction
        if self._knowledge_extractor:
            try:
                self._knowledge_extractor.process_message(msg)
            except Exception:
                pass
```

- [ ] **Step 3: Wire batch extraction into initial_sync**

In `initial_sync`, after the message loop and before eviction (after line 63), add:

```python
            # Batch knowledge extraction
            if self._knowledge_extractor:
                try:
                    self._knowledge_extractor.batch_extract(messages)
                    self._knowledge_extractor.build_contact_graph()
                    self._knowledge_extractor.build_topic_clusters()
                except Exception:
                    pass
```

- [ ] **Step 4: Update EmailService.initialize() to create and pass extractor**

In `src/homie_core/email/__init__.py`, in the `initialize` method, after creating the organizer (around line 69), add:

```python
                from homie_core.email.knowledge_extractor import EmailKnowledgeExtractor
                knowledge_extractor = EmailKnowledgeExtractor(
                    cache_conn=self._conn,
                    graph=getattr(self, '_knowledge_graph', None),
                    model_engine=self._model_engine,
                )
```

And pass it to SyncEngine:

```python
                engine = SyncEngine(
                    provider=provider,
                    classifier=classifier,
                    cache_conn=self._conn,
                    account_id=account_id,
                    organizer=organizer,
                    vault=self._vault,
                    working_memory=self._working_memory,
                    knowledge_extractor=knowledge_extractor,
                )
```

- [ ] **Step 5: Run existing tests to verify nothing breaks**

Run: `pytest tests/unit/test_email/ -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/email/sync_engine.py src/homie_core/email/__init__.py
git commit -m "feat(email): wire knowledge extractor into sync pipeline"
```

---

### Task 15: Register New Brain Tools

**Files:**
- Modify: `src/homie_core/email/tools.py`
- Test: `tests/unit/test_email/test_tools.py`

- [ ] **Step 1: Write failing tests for new tools**

```python
# tests/unit/test_email/test_tools.py
"""Tests for new email brain tools."""
from __future__ import annotations

from unittest.mock import MagicMock

from homie_core.email.tools import register_email_tools


def _make_registry_and_service():
    registry = MagicMock()
    service = MagicMock()
    return registry, service


class TestNewToolRegistration:
    def test_registers_all_new_tools(self):
        registry, service = _make_registry_and_service()
        register_email_tools(registry, service)

        registered_names = {call.args[0].name for call in registry.register.call_args_list}
        expected_new = {
            "email_inbox_threads", "email_thread_full", "email_unread_counts",
            "email_list_drafts", "email_get_draft", "email_update_draft", "email_delete_draft",
            "email_send", "email_send_draft", "email_reply", "email_reply_all", "email_forward",
            "email_attachments", "email_download_attachment",
            "email_contact_insights", "email_topic_summary",
            "email_pending_actions", "email_briefing",
        }
        for name in expected_new:
            assert name in registered_names, f"Missing tool: {name}"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_email/test_tools.py -v`

- [ ] **Step 3: Append new tool registrations to tools.py**

Append to `register_email_tools` function in `src/homie_core/email/tools.py`:

```python
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
        execute=tool_email_inbox_threads,
        category="email",
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
        execute=tool_email_thread_full,
        category="email",
    ))

    def tool_email_unread_counts(account: str = "all") -> str:
        return json.dumps(email_service.get_unread_counts(
            account=None if account == "all" else account,
        ))

    registry.register(Tool(
        name="email_unread_counts",
        description="Get unread email counts by category (inbox, spam, starred).",
        params=[ToolParam(name="account", description="Account or 'all'", type="string", required=False, default="all")],
        execute=tool_email_unread_counts,
        category="email",
    ))

    # ── Draft tools ──────────────────────────────────────────────────

    def tool_email_list_drafts(account: str = "") -> str:
        drafts = email_service.list_drafts(account=account or None)
        return _truncate(json.dumps([
            {"id": d.id, "subject": d.message.subject, "to": d.message.recipients}
            for d in drafts
        ]))

    registry.register(Tool(
        name="email_list_drafts",
        description="List all email drafts.",
        params=[ToolParam(name="account", description="Account (optional)", type="string", required=False, default="")],
        execute=tool_email_list_drafts,
        category="email",
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
        name="email_get_draft",
        description="Read a specific draft by ID.",
        params=[ToolParam(name="draft_id", description="Draft ID", type="string")],
        execute=tool_email_get_draft,
        category="email",
    ))

    def tool_email_update_draft(draft_id: str, to: str = "", subject: str = "", body: str = "",
                                 cc: str = "", bcc: str = "") -> str:
        result = email_service.update_draft(draft_id, to, subject, body)
        return json.dumps({"draft_id": result, "status": "updated"})

    registry.register(Tool(
        name="email_update_draft",
        description="Update an existing draft.",
        params=[
            ToolParam(name="draft_id", description="Draft ID", type="string"),
            ToolParam(name="to", description="Recipient", type="string", required=False, default=""),
            ToolParam(name="subject", description="Subject", type="string", required=False, default=""),
            ToolParam(name="body", description="Body", type="string", required=False, default=""),
            ToolParam(name="cc", description="CC (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="bcc", description="BCC (comma-separated)", type="string", required=False, default=""),
        ],
        execute=tool_email_update_draft,
        category="email",
    ))

    def tool_email_delete_draft(draft_id: str) -> str:
        email_service.delete_draft(draft_id)
        return json.dumps({"status": "deleted", "draft_id": draft_id})

    registry.register(Tool(
        name="email_delete_draft",
        description="Delete a draft permanently.",
        params=[ToolParam(name="draft_id", description="Draft ID", type="string")],
        execute=tool_email_delete_draft,
        category="email",
    ))

    # ── Send tools (HITL gated) ──────────────────────────────────────

    def tool_email_send(to: str, subject: str, body: str, cc: str = "",
                        bcc: str = "", attachments: str = "", account: str = "") -> str:
        try:
            att_list = [a.strip() for a in attachments.split(",") if a.strip()] if attachments else None
            cc_list = [c.strip() for c in cc.split(",") if c.strip()] if cc else None
            bcc_list = [b.strip() for b in bcc.split(",") if b.strip()] if bcc else None
            msg_id = email_service.send_email(
                to=to, subject=subject, body=body,
                cc=cc_list, bcc=bcc_list, attachments=att_list,
                account=account or None,
            )
            return json.dumps({"message_id": msg_id, "status": "sent"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    registry.register(Tool(
        name="email_send",
        description="Send an email directly. Requires user approval.",
        params=[
            ToolParam(name="to", description="Recipient", type="string"),
            ToolParam(name="subject", description="Subject", type="string"),
            ToolParam(name="body", description="Body", type="string"),
            ToolParam(name="cc", description="CC (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="bcc", description="BCC (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="attachments", description="File paths (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="account", description="Send from account", type="string", required=False, default=""),
        ],
        execute=tool_email_send,
        category="email",
    ))

    def tool_email_send_draft(draft_id: str) -> str:
        try:
            msg_id = email_service.send_draft(draft_id)
            return json.dumps({"message_id": msg_id, "status": "sent"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    registry.register(Tool(
        name="email_send_draft",
        description="Send an existing draft. Requires user approval.",
        params=[ToolParam(name="draft_id", description="Draft ID", type="string")],
        execute=tool_email_send_draft,
        category="email",
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
        name="email_reply",
        description="Reply to an email. Creates draft by default; set send=true to send (requires approval).",
        params=[
            ToolParam(name="message_id", description="Message ID to reply to", type="string"),
            ToolParam(name="body", description="Reply body", type="string"),
            ToolParam(name="send", description="'true' to send, 'false' for draft", type="string", required=False, default="false"),
        ],
        execute=tool_email_reply,
        category="email",
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
        name="email_reply_all",
        description="Reply-all to an email. Creates draft by default; set send=true to send (requires approval).",
        params=[
            ToolParam(name="message_id", description="Message ID to reply to", type="string"),
            ToolParam(name="body", description="Reply body", type="string"),
            ToolParam(name="send", description="'true' to send, 'false' for draft", type="string", required=False, default="false"),
        ],
        execute=tool_email_reply_all,
        category="email",
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
        name="email_forward",
        description="Forward an email. Creates draft by default; set send=true to send (requires approval).",
        params=[
            ToolParam(name="message_id", description="Message ID to forward", type="string"),
            ToolParam(name="to", description="Forward to", type="string"),
            ToolParam(name="body", description="Additional message", type="string", required=False, default=""),
            ToolParam(name="send", description="'true' to send, 'false' for draft", type="string", required=False, default="false"),
        ],
        execute=tool_email_forward,
        category="email",
    ))

    # ── Attachment tools ─────────────────────────────────────────────

    def tool_email_attachments(message_id: str) -> str:
        attachments = email_service.get_attachments(message_id)
        return _truncate(json.dumps([
            {"id": a.id, "filename": a.filename, "mime_type": a.mime_type, "size": a.size}
            for a in attachments
        ]))

    registry.register(Tool(
        name="email_attachments",
        description="List attachments for an email message.",
        params=[ToolParam(name="message_id", description="Message ID", type="string")],
        execute=tool_email_attachments,
        category="email",
    ))

    def tool_email_download_attachment(message_id: str, attachment_id: str) -> str:
        path = email_service.download_attachment(message_id, attachment_id)
        if path:
            return json.dumps({"status": "downloaded", "path": path})
        return json.dumps({"error": "Download failed or path rejected"})

    registry.register(Tool(
        name="email_download_attachment",
        description="Download an email attachment to local storage.",
        params=[
            ToolParam(name="message_id", description="Message ID", type="string"),
            ToolParam(name="attachment_id", description="Attachment ID", type="string"),
        ],
        execute=tool_email_download_attachment,
        category="email",
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
        name="email_contact_insights",
        description="Get relationship history, email frequency, topics, and pending actions for a contact.",
        params=[ToolParam(name="email_or_name", description="Email address or contact name", type="string")],
        execute=tool_email_contact_insights,
        category="email",
    ))

    def tool_email_topic_summary(topic: str) -> str:
        return json.dumps({"summary": email_service.get_topic_summary(topic)})

    registry.register(Tool(
        name="email_topic_summary",
        description="Get cross-thread summary of a topic or project from email context.",
        params=[ToolParam(name="topic", description="Topic or project name", type="string")],
        execute=tool_email_topic_summary,
        category="email",
    ))

    def tool_email_pending_actions() -> str:
        actions = email_service.get_pending_actions()
        return _truncate(json.dumps([
            {"id": a.id, "description": a.description, "assignee": a.assignee,
             "deadline": a.deadline, "urgency": a.urgency}
            for a in actions
        ]))

    registry.register(Tool(
        name="email_pending_actions",
        description="List all pending action items extracted from emails, with deadlines and urgency.",
        params=[],
        execute=tool_email_pending_actions,
        category="email",
    ))

    def tool_email_briefing(days: str = "1") -> str:
        try:
            num_days = int(days)
        except (ValueError, TypeError):
            num_days = 1
        return email_service.get_email_insights(days=num_days)

    registry.register(Tool(
        name="email_briefing",
        description="Generate an AI-powered daily/weekly email intelligence briefing.",
        params=[ToolParam(name="days", description="Days to cover", type="string", required=False, default="1")],
        execute=tool_email_briefing,
        category="email",
    ))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_email/test_tools.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/tools.py tests/unit/test_email/test_tools.py
git commit -m "feat(email): register 17 new brain tools for threads, drafts, send, attachments, and insights"
```

---

### Task 16: Wire HITL Gate for Send Tools

**Files:**
- Modify: Wherever `HITLMiddleware` is instantiated (likely in `src/homie_app/` or brain wiring code)

- [ ] **Step 1: Find where HITLMiddleware is constructed**

Run: `grep -r "HITLMiddleware(" src/ --include="*.py"`

- [ ] **Step 2: Add send tools to the gated set**

Wherever `HITLMiddleware` is constructed, ensure the `gated_tools` parameter includes email send tools:

```python
HITLMiddleware(gated_tools={
    "run_command", "write_file",
    "email_send", "email_send_draft", "email_reply", "email_reply_all", "email_forward",
})
```

If the construction uses a default set, extend it rather than replacing.

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "feat(email): wire send tools into HITL gate for user approval"
```

---

### Task 17: Deprecate EmailIndexer

**Files:**
- Modify: `src/homie_core/knowledge/email_indexer.py`

- [ ] **Step 1: Add deprecation redirect**

Replace the body of `index_recent` in `src/homie_core/knowledge/email_indexer.py`:

```python
    def index_recent(self, days: int = 30) -> dict:
        """Index emails from the last *days* days into the knowledge graph.

        .. deprecated:: Use EmailKnowledgeExtractor.batch_extract() instead.

        Returns stats dict:
            {"indexed": int, "entities_created": int, "relationships_created": int}
        """
        import warnings
        warnings.warn(
            "EmailIndexer.index_recent() is deprecated. "
            "Use homie_core.email.knowledge_extractor.EmailKnowledgeExtractor instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        stats: dict[str, int] = {
            "indexed": 0,
            "entities_created": 0,
            "relationships_created": 0,
        }

        if self._graph is None:
            return stats

        try:
            db = sqlite3.connect(self._cache_path)
            db.row_factory = sqlite3.Row
        except Exception:
            return stats

        try:
            cursor = db.execute(
                "SELECT * FROM emails WHERE date >= datetime('now', ?) ORDER BY date DESC LIMIT 500",
                (f"-{days} days",),
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            db.close()
            return stats

        for row in rows:
            self._index_email(dict(row), stats)

        db.close()
        return stats
```

- [ ] **Step 2: Commit**

```bash
git add src/homie_core/knowledge/email_indexer.py
git commit -m "chore(email): deprecate EmailIndexer in favor of EmailKnowledgeExtractor"
```

---

### Task 18: Run Full Test Suite & Final Verification

- [ ] **Step 1: Run all email tests**

Run: `pytest tests/unit/test_email/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run full project tests to check no regressions**

Run: `pytest tests/ -v --timeout=60`
Expected: No new failures

- [ ] **Step 3: Final commit if any fixups needed**

```bash
git add -A
git commit -m "test(email): fix any test issues from full integration"
```
