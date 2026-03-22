"""Contradiction detector — finds and resolves conflicting facts."""

import logging
import time
from collections import defaultdict
from typing import Any

from ..graph.store import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# Predicates that are typically single-valued (one object at a time)
_SINGLE_VALUED_PREDICATES = {
    "works_at", "lives_in", "located_in", "primary_language",
    "current_role", "reports_to", "managed_by",
}


class ContradictionDetector:
    """Detects and resolves contradictory relationships in the knowledge graph."""

    def __init__(self, graph_store: KnowledgeGraphStore) -> None:
        self._graph = graph_store

    def detect(self) -> list[dict[str, Any]]:
        """Find contradictions — same subject + single-valued predicate with multiple current objects."""
        if self._graph._conn is None:
            return []

        contradictions = []
        current_rels = self._graph._conn.execute(
            "SELECT * FROM kg_relationships WHERE valid_until IS NULL"
        ).fetchall()

        # Group by (subject_id, relation)
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in current_rels:
            r = dict(r)
            groups[(r["subject_id"], r["relation"])].append(r)

        for (subj, rel), rels in groups.items():
            if rel not in _SINGLE_VALUED_PREDICATES:
                continue
            if len(rels) <= 1:
                continue
            # Multiple current values for single-valued predicate = contradiction
            contradictions.append({
                "subject_id": subj,
                "relation": rel,
                "conflicting_relationships": rels,
            })

        return contradictions

    def resolve_all(self) -> list[dict]:
        """Detect and resolve all contradictions. Returns list of resolutions."""
        contradictions = self.detect()
        resolutions = []

        for contradiction in contradictions:
            rels = contradiction["conflicting_relationships"]
            # Keep highest confidence, supersede others
            rels_sorted = sorted(rels, key=lambda r: (r["confidence"], r["created_at"]), reverse=True)
            winner = rels_sorted[0]
            now = time.time()

            for loser in rels_sorted[1:]:
                self._graph.update_relationship_valid_until(loser["id"], now)
                resolutions.append({
                    "subject_id": contradiction["subject_id"],
                    "relation": contradiction["relation"],
                    "kept": winner["object_id"],
                    "superseded": loser["object_id"],
                    "reason": "higher_confidence",
                })
                logger.info(
                    "Resolved contradiction: %s %s — kept %s, superseded %s",
                    contradiction["subject_id"], contradiction["relation"],
                    winner["object_id"], loser["object_id"],
                )

        return resolutions
