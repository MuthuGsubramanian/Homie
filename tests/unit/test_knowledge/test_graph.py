"""Tests for the KnowledgeGraph SQLite triple store."""
from __future__ import annotations

import time
import pytest
from pathlib import Path

from homie_core.knowledge.models import Entity, Relationship
from homie_core.knowledge.graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def graph(tmp_path):
    """A fresh in-memory-like KnowledgeGraph backed by a temp SQLite file."""
    return KnowledgeGraph(tmp_path / "kg.db")


def _person(name: str, **kwargs) -> Entity:
    return Entity(name=name, entity_type="person", **kwargs)


def _project(name: str, **kwargs) -> Entity:
    return Entity(name=name, entity_type="project", **kwargs)


def _tool(name: str, **kwargs) -> Entity:
    return Entity(name=name, entity_type="tool", **kwargs)


def _rel(subject_id: str, relation: str, object_id: str, **kwargs) -> Relationship:
    return Relationship(subject_id=subject_id, relation=relation, object_id=object_id, **kwargs)


# ---------------------------------------------------------------------------
# Entity operations
# ---------------------------------------------------------------------------

class TestAddAndGetEntity:
    def test_roundtrip(self, graph):
        alice = _person("Alice", source="user")
        eid = graph.add_entity(alice)
        fetched = graph.get_entity(eid)
        assert fetched is not None
        assert fetched.name == "Alice"
        assert fetched.entity_type == "person"
        assert fetched.source == "user"

    def test_returns_entity_id(self, graph):
        alice = _person("Alice")
        eid = graph.add_entity(alice)
        assert eid == alice.id

    def test_get_nonexistent_returns_none(self, graph):
        assert graph.get_entity("no-such-id") is None

    def test_attributes_preserved(self, graph):
        e = Entity(name="GPT-4", entity_type="tool", attributes={"vendor": "OpenAI", "version": "4"})
        eid = graph.add_entity(e)
        fetched = graph.get_entity(eid)
        assert fetched.attributes["vendor"] == "OpenAI"
        assert fetched.attributes["version"] == "4"

    def test_confidence_preserved(self, graph):
        e = _person("Bob", confidence=0.75)
        eid = graph.add_entity(e)
        fetched = graph.get_entity(eid)
        assert abs(fetched.confidence - 0.75) < 1e-6


class TestMergeEntity:
    def test_merge_inserts_new(self, graph):
        e = _person("Charlie")
        eid = graph.merge_entity(e)
        assert graph.get_entity(eid) is not None

    def test_merge_updates_existing_by_name_and_type(self, graph):
        e1 = _person("Diana", confidence=0.5)
        graph.merge_entity(e1)

        e2 = _person("Diana", confidence=0.9, source="user")
        eid2 = graph.merge_entity(e2)

        # Should only have one Diana
        dianas = graph.find_entities(name="Diana", entity_type="person")
        assert len(dianas) == 1
        assert abs(dianas[0].confidence - 0.9) < 1e-6
        assert dianas[0].source == "user"

    def test_merge_returns_existing_id(self, graph):
        e1 = _person("Eve")
        eid1 = graph.merge_entity(e1)
        e2 = _person("Eve")
        eid2 = graph.merge_entity(e2)
        assert eid1 == eid2

    def test_merge_different_types_creates_separate(self, graph):
        e1 = Entity(name="Python", entity_type="tool")
        e2 = Entity(name="Python", entity_type="concept")
        graph.merge_entity(e1)
        graph.merge_entity(e2)
        tools = graph.find_entities(name="Python", entity_type="tool")
        concepts = graph.find_entities(name="Python", entity_type="concept")
        assert len(tools) == 1
        assert len(concepts) == 1


class TestFindEntities:
    def test_find_by_name(self, graph):
        graph.add_entity(_person("Frank"))
        graph.add_entity(_project("FrankProject"))
        results = graph.find_entities(name="Frank")
        names = [e.name for e in results]
        assert "Frank" in names

    def test_find_by_type(self, graph):
        graph.add_entity(_person("Grace"))
        graph.add_entity(_project("Homie"))
        persons = graph.find_entities(entity_type="person")
        assert all(e.entity_type == "person" for e in persons)
        assert any(e.name == "Grace" for e in persons)

    def test_find_by_name_and_type(self, graph):
        graph.add_entity(_person("Heidi"))
        graph.add_entity(_project("Heidi's Project"))
        results = graph.find_entities(name="Heidi", entity_type="person")
        assert len(results) == 1
        assert results[0].name == "Heidi"

    def test_find_empty_returns_all_up_to_limit(self, graph):
        for i in range(5):
            graph.add_entity(_person(f"Person{i}"))
        all_results = graph.find_entities()
        assert len(all_results) <= 50
        assert len(all_results) >= 5

    def test_find_limit(self, graph):
        for i in range(10):
            graph.add_entity(_person(f"P{i}"))
        results = graph.find_entities(entity_type="person", limit=3)
        assert len(results) <= 3

    def test_find_name_case_insensitive(self, graph):
        graph.add_entity(_person("Ivan"))
        results = graph.find_entities(name="ivan")
        assert any(e.name == "Ivan" for e in results)


# ---------------------------------------------------------------------------
# Relationship operations
# ---------------------------------------------------------------------------

class TestAddAndGetRelationship:
    def test_add_and_retrieve(self, graph):
        alice = graph.add_entity(_person("Alice"))
        proj = graph.add_entity(_project("Homie"))
        rel = _rel(alice, "works_on", proj)
        graph.add_relationship(rel)

        rels = graph.get_relationships(alice)
        assert len(rels) == 1
        assert rels[0].relation == "works_on"
        assert rels[0].object_id == proj

    def test_direction_outgoing(self, graph):
        a = graph.add_entity(_person("A"))
        b = graph.add_entity(_person("B"))
        graph.add_relationship(_rel(a, "mentions", b))

        outgoing = graph.get_relationships(a, direction="outgoing")
        incoming = graph.get_relationships(a, direction="incoming")
        assert len(outgoing) == 1
        assert len(incoming) == 0

    def test_direction_incoming(self, graph):
        a = graph.add_entity(_person("A"))
        b = graph.add_entity(_person("B"))
        graph.add_relationship(_rel(a, "mentions", b))

        incoming = graph.get_relationships(b, direction="incoming")
        assert len(incoming) == 1
        assert incoming[0].subject_id == a

    def test_direction_both(self, graph):
        a = graph.add_entity(_person("A"))
        b = graph.add_entity(_person("B"))
        graph.add_relationship(_rel(a, "mentions", b))
        graph.add_relationship(_rel(b, "supports", a))

        both = graph.get_relationships(a, direction="both")
        assert len(both) == 2

    def test_filter_by_relation(self, graph):
        a = graph.add_entity(_person("A"))
        b = graph.add_entity(_project("P"))
        c = graph.add_entity(_tool("T"))
        graph.add_relationship(_rel(a, "works_on", b))
        graph.add_relationship(_rel(a, "uses", c))

        works_on = graph.get_relationships(a, relation="works_on")
        assert all(r.relation == "works_on" for r in works_on)
        assert len(works_on) == 1

    def test_returns_id(self, graph):
        a = graph.add_entity(_person("A"))
        b = graph.add_entity(_person("B"))
        rel = _rel(a, "mentions", b)
        rid = graph.add_relationship(rel)
        assert rid == rel.id


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------

class TestNeighbors:
    def test_direct_neighbors_1_hop(self, graph):
        alice = graph.add_entity(_person("Alice"))
        proj = graph.add_entity(_project("Homie"))
        graph.add_relationship(_rel(alice, "works_on", proj))

        neighbors = graph.neighbors(alice, max_hops=1)
        ids = [e.id for e in neighbors]
        assert proj in ids

    def test_2_hop_traversal(self, graph):
        alice = graph.add_entity(_person("Alice"))
        proj = graph.add_entity(_project("Homie"))
        tool = graph.add_entity(_tool("Python"))
        graph.add_relationship(_rel(alice, "works_on", proj))
        graph.add_relationship(_rel(proj, "uses", tool))

        neighbors_2 = graph.neighbors(alice, max_hops=2)
        ids = [e.id for e in neighbors_2]
        assert proj in ids
        assert tool in ids

    def test_self_not_in_neighbors(self, graph):
        alice = graph.add_entity(_person("Alice"))
        proj = graph.add_entity(_project("P"))
        graph.add_relationship(_rel(alice, "works_on", proj))

        neighbors = graph.neighbors(alice, max_hops=1)
        ids = [e.id for e in neighbors]
        assert alice not in ids

    def test_1_hop_does_not_include_2_hop(self, graph):
        alice = graph.add_entity(_person("Alice"))
        proj = graph.add_entity(_project("Homie"))
        tool = graph.add_entity(_tool("Python"))
        graph.add_relationship(_rel(alice, "works_on", proj))
        graph.add_relationship(_rel(proj, "uses", tool))

        neighbors_1 = graph.neighbors(alice, max_hops=1)
        ids = [e.id for e in neighbors_1]
        assert tool not in ids

    def test_no_relationships_returns_empty(self, graph):
        alice = graph.add_entity(_person("Alice"))
        assert graph.neighbors(alice, max_hops=2) == []

    def test_nonexistent_entity_returns_empty(self, graph):
        assert graph.neighbors("no-such-id", max_hops=2) == []


# ---------------------------------------------------------------------------
# Query support
# ---------------------------------------------------------------------------

class TestEntitiesMentionedIn:
    def test_finds_entity_by_name_in_text(self, graph):
        graph.add_entity(_person("Alice"))
        graph.add_entity(_project("Homie"))
        results = graph.entities_mentioned_in("Alice is working on the Homie project today")
        names = [e.name for e in results]
        assert "Alice" in names
        assert "Homie" in names

    def test_case_insensitive_match(self, graph):
        graph.add_entity(_person("Bob"))
        results = graph.entities_mentioned_in("bob sent an email")
        names = [e.name for e in results]
        assert "Bob" in names

    def test_no_match_returns_empty(self, graph):
        graph.add_entity(_person("Alice"))
        results = graph.entities_mentioned_in("nothing relevant here")
        assert results == []

    def test_empty_text_returns_empty(self, graph):
        graph.add_entity(_person("Alice"))
        assert graph.entities_mentioned_in("") == []


class TestContextForEntity:
    def test_generates_readable_summary(self, graph):
        alice = graph.add_entity(_person("Alice"))
        proj = graph.add_entity(_project("Homie"))
        graph.add_relationship(_rel(alice, "works_on", proj))

        ctx = graph.context_for_entity(alice)
        assert "Alice" in ctx
        assert "person" in ctx
        assert "works_on" in ctx
        assert "Homie" in ctx

    def test_entity_no_relationships(self, graph):
        bob = graph.add_entity(_person("Bob"))
        ctx = graph.context_for_entity(bob)
        assert "Bob" in ctx
        assert "person" in ctx

    def test_nonexistent_entity_returns_empty(self, graph):
        ctx = graph.context_for_entity("no-such-id")
        assert ctx == ""


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty_graph(self, graph):
        s = graph.stats()
        assert s["entity_count"] == 0
        assert s["relationship_count"] == 0

    def test_counts_after_adds(self, graph):
        alice = graph.add_entity(_person("Alice"))
        proj = graph.add_entity(_project("P"))
        graph.add_relationship(_rel(alice, "works_on", proj))
        s = graph.stats()
        assert s["entity_count"] == 2
        assert s["relationship_count"] == 1

    def test_entities_by_type(self, graph):
        graph.add_entity(_person("Alice"))
        graph.add_entity(_person("Bob"))
        graph.add_entity(_project("Homie"))
        s = graph.stats()
        assert s["entities_by_type"]["person"] == 2
        assert s["entities_by_type"]["project"] == 1

    def test_relationships_by_type(self, graph):
        a = graph.add_entity(_person("A"))
        b = graph.add_entity(_project("B"))
        c = graph.add_entity(_tool("C"))
        graph.add_relationship(_rel(a, "works_on", b))
        graph.add_relationship(_rel(a, "uses", c))
        graph.add_relationship(_rel(b, "uses", c))
        s = graph.stats()
        assert s["relationships_by_type"]["uses"] == 2
        assert s["relationships_by_type"]["works_on"] == 1


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

class TestDecayAndPrune:
    def test_decay_reduces_confidence_of_old_entity(self, graph):
        """Entities with old updated_at should have confidence reduced."""
        old_entity = Entity(
            name="OldEntity",
            entity_type="concept",
            confidence=1.0,
            updated_at="2020-01-01T00:00:00",
        )
        eid = graph.add_entity(old_entity)
        graph.decay_scores(half_life_days=30)
        fetched = graph.get_entity(eid)
        # Very old entity should have decayed significantly
        assert fetched.confidence < 0.5

    def test_decay_leaves_recent_entity_high(self, graph):
        """Recently updated entities should retain high confidence."""
        from datetime import datetime, timezone
        now_str = datetime.now(timezone.utc).isoformat()
        recent = Entity(
            name="RecentEntity",
            entity_type="concept",
            confidence=1.0,
            updated_at=now_str,
        )
        eid = graph.add_entity(recent)
        graph.decay_scores(half_life_days=30)
        fetched = graph.get_entity(eid)
        # Less than 1 day old, should be very close to 1.0
        assert fetched.confidence > 0.95

    def test_prune_removes_low_confidence(self, graph):
        e_low = Entity(name="LowConf", entity_type="concept", confidence=0.05)
        e_high = Entity(name="HighConf", entity_type="concept", confidence=0.9)
        eid_low = graph.add_entity(e_low)
        eid_high = graph.add_entity(e_high)

        graph.prune(min_confidence=0.1)

        assert graph.get_entity(eid_low) is None
        assert graph.get_entity(eid_high) is not None

    def test_prune_removes_low_confidence_relationships(self, graph):
        a = graph.add_entity(_person("A"))
        b = graph.add_entity(_project("B"))
        rel_low = Relationship(subject_id=a, relation="mentions", object_id=b, confidence=0.05)
        rel_high = Relationship(subject_id=a, relation="works_on", object_id=b, confidence=0.9)
        graph.add_relationship(rel_low)
        graph.add_relationship(rel_high)

        graph.prune(min_confidence=0.1)

        rels = graph.get_relationships(a)
        assert all(r.confidence >= 0.1 for r in rels)
        assert len(rels) == 1
