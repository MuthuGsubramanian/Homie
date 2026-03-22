import time
import pytest
from homie_core.knowledge.models import Relationship, Entity


class TestTemporalRelationship:
    def test_relationship_has_temporal_fields(self):
        rel = Relationship(
            subject_id="e1",
            relation="works_at",
            object_id="e2",
            valid_from=time.time(),
            valid_until=None,
        )
        assert rel.valid_from > 0
        assert rel.valid_until is None

    def test_relationship_is_current(self):
        rel = Relationship(
            subject_id="e1",
            relation="works_at",
            object_id="e2",
            valid_from=time.time() - 100,
            valid_until=None,
        )
        assert rel.is_current is True

    def test_relationship_is_superseded(self):
        rel = Relationship(
            subject_id="e1",
            relation="works_at",
            object_id="e2",
            valid_from=time.time() - 200,
            valid_until=time.time() - 100,
        )
        assert rel.is_current is False

    def test_entity_has_aliases(self):
        ent = Entity(
            name="Python",
            entity_type="technology",
            aliases=["Python3", "CPython"],
        )
        assert "Python3" in ent.aliases
        assert len(ent.aliases) == 2

    def test_entity_default_aliases_empty(self):
        ent = Entity(name="Go", entity_type="technology")
        assert ent.aliases == []
