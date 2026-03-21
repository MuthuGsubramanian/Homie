"""Knowledge Graph package for Homie AI.

Provides a SQLite-backed triple store for entity and relationship tracking,
with pattern-based and spaCy-powered entity extraction.
"""
from __future__ import annotations

from homie_core.knowledge.models import Entity, Relationship
from homie_core.knowledge.graph import KnowledgeGraph
from homie_core.knowledge.extractor import EntityExtractor

__all__ = ["Entity", "Relationship", "KnowledgeGraph", "EntityExtractor"]
