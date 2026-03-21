"""Data models for the Knowledge Graph.

Entities represent nodes in the graph (people, projects, concepts, etc.)
Relationships represent directed edges between entities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class Entity:
    """A node in the knowledge graph.

    entity_type must be one of:
        person, project, concept, tool, document, task, event,
        location, snippet, goal
    """

    name: str
    entity_type: str  # person, project, concept, tool, document, task, event, location, snippet, goal
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attributes: dict = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ""  # "extraction", "user", "inference"
    created_at: str = ""
    updated_at: str = ""
    last_accessed: str = ""


@dataclass
class Relationship:
    """A directed edge in the knowledge graph.

    relation must be one of:
        authored, works_on, mentions, depends_on, contains, related_to,
        uses, supports, child_of, has_fact
    """

    subject_id: str
    relation: str  # authored, works_on, mentions, depends_on, contains, related_to, uses, supports, child_of, has_fact
    object_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence: float = 1.0
    source: str = ""
    source_chunk_id: str = ""
    created_at: str = ""
    updated_at: str = ""
