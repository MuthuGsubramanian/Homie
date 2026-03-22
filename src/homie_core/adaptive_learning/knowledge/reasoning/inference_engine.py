"""Inference engine — derives new relationships from existing graph connections."""

import logging
from typing import Any

from ..graph.store import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# Transitive inference rules: (rel1, rel2) → inferred_relation, confidence_multiplier
_INFERENCE_RULES = [
    (("works_on", "uses"), "works_with", 0.7),
    (("depends_on", "depends_on"), "indirectly_depends_on", 0.5),
    (("member_of", "located_in"), "located_in", 0.4),
    (("uses", "uses"), "indirectly_uses", 0.5),
    (("works_on", "depends_on"), "works_with", 0.6),
]


class InferenceEngine:
    """Derives new relationships from existing graph connections."""

    def __init__(
        self,
        graph_store: KnowledgeGraphStore,
        max_hops: int = 2,
    ) -> None:
        self._graph = graph_store
        self._max_hops = max_hops

    def run_inference(self) -> list[dict[str, Any]]:
        """Run inference rules and return newly created relationships."""
        if self._graph._conn is None:
            return []

        inferred = []

        # Get all current relationships
        all_rels = self._graph._conn.execute(
            "SELECT * FROM kg_relationships WHERE valid_until IS NULL AND source != 'inference'"
        ).fetchall()
        all_rels = [dict(r) for r in all_rels]

        # Build adjacency: {subject_id: [(relation, object_id, confidence)]}
        adj: dict[str, list[tuple[str, str, float]]] = {}
        for r in all_rels:
            adj.setdefault(r["subject_id"], []).append((r["relation"], r["object_id"], r["confidence"]))

        # Apply rules (1-hop transitive)
        for (rel1, rel2), inferred_rel, conf_mult in _INFERENCE_RULES:
            for r in all_rels:
                if r["relation"] != rel1:
                    continue
                # Look for second hop from r's object
                for next_rel, next_obj, next_conf in adj.get(r["object_id"], []):
                    if next_rel != rel2:
                        continue
                    if next_obj == r["subject_id"]:
                        continue  # skip self-loops

                    # Check if inference already exists
                    existing = self._graph.find_current_relationships(r["subject_id"], inferred_rel)
                    if any(e["object_id"] == next_obj for e in existing):
                        continue

                    confidence = min(r["confidence"], next_conf) * conf_mult
                    rid = self._graph.add_relationship(
                        r["subject_id"],
                        inferred_rel,
                        next_obj,
                        confidence=confidence,
                        source="inference",
                    )
                    inferred.append({
                        "id": rid,
                        "subject_id": r["subject_id"],
                        "relation": inferred_rel,
                        "object_id": next_obj,
                        "confidence": confidence,
                        "source": "inference",
                    })

        logger.info("Inference produced %d new relationships", len(inferred))
        return inferred
