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

_FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "icloud.com", "mail.com", "protonmail.com", "zoho.com", "yandex.com",
}

_DEADLINE_PATTERNS = [
    re.compile(r"(?:by|before|due|deadline)\s+(\w+\s+\d{1,2}(?:,?\s*\d{4})?)", re.IGNORECASE),
    re.compile(r"(?:by|before|due)\s+(tomorrow|today|end of (?:day|week|month))", re.IGNORECASE),
    re.compile(r"(?:by|before|due)\s+((?:mon|tues|wednes|thurs|fri|satur|sun)day)", re.IGNORECASE),
]

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

    def __init__(self, cache_conn: sqlite3.Connection, graph=None, model_engine=None):
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
                id TEXT PRIMARY KEY, account_id TEXT NOT NULL, subject TEXT,
                participants TEXT, message_count INTEGER DEFAULT 0,
                last_message_date REAL, snippet TEXT, labels TEXT, fetched_at REAL
            );
            CREATE TABLE IF NOT EXISTS email_drafts (
                id TEXT PRIMARY KEY, account_id TEXT NOT NULL, to_addr TEXT,
                subject TEXT, body TEXT, cc TEXT, bcc TEXT,
                reply_to_message_id TEXT, updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS email_attachments (
                id TEXT PRIMARY KEY, message_id TEXT NOT NULL, account_id TEXT NOT NULL,
                filename TEXT NOT NULL, mime_type TEXT, size INTEGER DEFAULT 0,
                local_path TEXT, downloaded_at REAL
            );
            CREATE TABLE IF NOT EXISTS email_contacts (
                email TEXT PRIMARY KEY, name TEXT, organization TEXT,
                relationship TEXT, email_count INTEGER DEFAULT 0,
                last_contact REAL, topics TEXT, entity_id TEXT, updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS email_action_items (
                id TEXT PRIMARY KEY, message_id TEXT NOT NULL, thread_id TEXT,
                account_id TEXT NOT NULL, description TEXT NOT NULL,
                assignee TEXT, deadline REAL, urgency TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'pending', extracted_at REAL, completed_at REAL
            );
            CREATE TABLE IF NOT EXISTS email_topics (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, thread_ids TEXT,
                entity_ids TEXT, message_count INTEGER DEFAULT 0,
                last_activity REAL, updated_at REAL
            );
        """)

    def process_message(self, msg: EmailMessage) -> None:
        if self.has_llm:
            result = self._extract_llm(msg)
            if not result:
                result = self._extract_heuristic(msg)
        else:
            result = self._extract_heuristic(msg)
        self._store_extraction(msg, result)

    def batch_extract(self, messages: list[EmailMessage], batch_size: int = 20) -> None:
        for msg in messages:
            self.process_message(msg)

    def build_contact_graph(self) -> None:
        if not self._graph:
            return
        from homie_core.knowledge.models import Entity, Relationship
        rows = self._conn.execute(
            "SELECT email, name, organization, relationship FROM email_contacts"
        ).fetchall()
        for row in rows:
            email_addr, name, org, rel = row
            person = Entity(
                name=name or email_addr, entity_type="person",
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
                    subject_id=entity_id, relation="works_with",
                    object_id=org_id, source="email_sync",
                ))
        self._conn.commit()

    def build_topic_clusters(self) -> None:
        pass

    def _extract_heuristic(self, msg: EmailMessage) -> dict[str, Any]:
        entities = []
        action_items = []
        events = []
        topics = []

        email_addr = _extract_email_address(msg.sender)
        name = _extract_name(msg.sender)
        entities.append({"name": name, "type": "person", "email": email_addr})

        domain = _extract_domain(email_addr)
        if domain and domain not in _FREE_EMAIL_DOMAINS:
            org_name = domain.split(".")[0].title()
            entities.append({"name": org_name, "type": "organization", "domain": domain})

        for r in msg.recipients:
            r_email = _extract_email_address(r)
            r_name = _extract_name(r)
            entities.append({"name": r_name, "type": "person", "email": r_email})

        text = f"{msg.subject} {msg.body or msg.snippet or ''}"
        for pattern in _DEADLINE_PATTERNS:
            match = pattern.search(text)
            if match:
                action_items.append({
                    "description": f"Deadline: {match.group(0).strip()}",
                    "assignee": "", "deadline": None, "urgency": "high",
                })
                break

        for pattern in _ACTION_PATTERNS:
            match = pattern.search(text)
            if match:
                action_items.append({
                    "description": match.group(1).strip(),
                    "assignee": "", "deadline": None, "urgency": "medium",
                })
                break

        subject_clean = re.sub(r"^(Re:|Fwd:|FW:)\s*", "", msg.subject, flags=re.IGNORECASE).strip()
        if subject_clean:
            topics.append(subject_clean)

        return {
            "entities": entities, "relationships": [],
            "action_items": action_items, "events": events, "topics": topics,
        }

    def _extract_llm(self, msg: EmailMessage) -> dict[str, Any] | None:
        if not self.has_llm:
            return None
        from homie_core.email.classifier import _email_text, _parse_llm_json
        prompt = _LLM_EXTRACT_PROMPT.format(
            sender=msg.sender, recipients=", ".join(msg.recipients[:5]),
            subject=msg.subject, body=_email_text(msg),
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
        now = time.time()
        for entity in result.get("entities", []):
            if entity.get("type") == "person" and entity.get("email"):
                email_addr = entity["email"]
                self._conn.execute(
                    """INSERT INTO email_contacts (email, name, organization, email_count, last_contact, topics, updated_at)
                       VALUES (?, ?, ?, 1, ?, '[]', ?)
                       ON CONFLICT(email) DO UPDATE SET
                           email_count = email_count + 1, last_contact = ?, updated_at = ?""",
                    (email_addr, entity.get("name", ""), "", now, now, now, now),
                )
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
        if self._graph:
            from homie_core.knowledge.models import Entity
            for entity in result.get("entities", []):
                etype = entity.get("type", "person")
                if etype not in ("person", "organization", "project"):
                    continue
                e = Entity(
                    name=entity.get("name", ""), entity_type=etype,
                    attributes={k: v for k, v in entity.items() if k not in ("name", "type")},
                    source="email_sync",
                )
                self._graph.merge_entity(e)
        self._conn.commit()
