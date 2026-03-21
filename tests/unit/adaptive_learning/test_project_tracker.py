# tests/unit/adaptive_learning/test_project_tracker.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.knowledge.project_tracker import ProjectTracker


class TestProjectTracker:
    def test_register_project(self):
        storage = MagicMock()
        tracker = ProjectTracker(storage=storage)
        tracker.register_project("homie", "/path/to/homie", branch="main")
        projects = tracker.list_projects()
        assert "homie" in projects

    def test_add_knowledge_triple(self):
        storage = MagicMock()
        tracker = ProjectTracker(storage=storage)
        tracker.add_knowledge("homie", "uses", "ChromaDB")
        knowledge = tracker.get_knowledge("homie")
        assert any(k["predicate"] == "uses" and k["object"] == "ChromaDB" for k in knowledge)

    def test_update_recent_activity(self):
        storage = MagicMock()
        tracker = ProjectTracker(storage=storage)
        tracker.register_project("homie", "/path")
        tracker.update_activity("homie", "Added self-healing runtime")
        project = tracker.get_project("homie")
        assert "self-healing" in project["recent_activity"][-1].lower()

    def test_unknown_project_returns_none(self):
        storage = MagicMock()
        tracker = ProjectTracker(storage=storage)
        assert tracker.get_project("nonexistent") is None

    def test_get_active_project(self):
        storage = MagicMock()
        tracker = ProjectTracker(storage=storage)
        tracker.register_project("homie", "/path")
        tracker.set_active("homie")
        assert tracker.active_project == "homie"
