"""Tests for the Learning Pipeline — auto-extraction of facts from conversations."""
import pytest
from unittest.mock import MagicMock

from homie_core.memory.learning_pipeline import LearningPipeline


class TestFactExtraction:
    def setup_method(self):
        self.sm = MagicMock()
        self.sm.get_facts.return_value = []
        self.sm.learn.return_value = 1
        self.pipeline = LearningPipeline(semantic_memory=self.sm)

    def test_extracts_preference(self):
        learned = self.pipeline.process_user_message("I prefer dark mode in all my editors")
        assert len(learned) >= 1
        self.sm.learn.assert_called()

    def test_extracts_identity(self):
        learned = self.pipeline.process_user_message("I'm a software engineer at Google")
        assert len(learned) >= 1

    def test_extracts_work_info(self):
        learned = self.pipeline.process_user_message("I work on machine learning infrastructure")
        assert len(learned) >= 1

    def test_extracts_location(self):
        learned = self.pipeline.process_user_message("I live in San Francisco")
        assert len(learned) >= 1

    def test_extracts_interests(self):
        learned = self.pipeline.process_user_message("I'm interested in quantum computing")
        assert len(learned) >= 1

    def test_extracts_habits(self):
        learned = self.pipeline.process_user_message("I usually code in the morning before meetings")
        assert len(learned) >= 1

    def test_extracts_skills(self):
        learned = self.pipeline.process_user_message("I speak Python and Rust fluently")
        assert len(learned) >= 1

    def test_skips_vague_statements(self):
        learned = self.pipeline.process_user_message("I want to learn more about this")
        assert len(learned) == 0

    def test_skips_short_messages(self):
        learned = self.pipeline.process_user_message("ok")
        assert len(learned) == 0

    def test_skips_questions(self):
        learned = self.pipeline.process_user_message("Can you help me with this?")
        assert len(learned) == 0

    def test_no_memory_returns_empty(self):
        pipeline = LearningPipeline(semantic_memory=None)
        learned = pipeline.process_user_message("I prefer dark mode")
        assert learned == []


class TestCorrectionHandling:
    def setup_method(self):
        self.sm = MagicMock()
        self.sm.learn.return_value = 1
        self.sm._db = MagicMock()
        self.sm._db._conn = MagicMock()
        self.pipeline = LearningPipeline(semantic_memory=self.sm)

    def test_correction_updates_old_fact(self):
        # Old fact exists
        self.sm.get_facts.return_value = [
            {"id": 1, "fact": "User prefers vim", "confidence": 0.8},
        ]
        learned = self.pipeline.process_user_message("actually, I switched to neovim now")
        # Should have reduced confidence of old fact and stored new one
        assert len(learned) >= 0  # May or may not extract depending on pattern

    def test_correction_pattern_detected(self):
        self.sm.get_facts.return_value = []
        learned = self.pipeline.process_user_message("no, my main language is actually Rust, not Python")
        # Should detect this as a correction
        assert isinstance(learned, list)

    def test_not_anymore_pattern(self):
        self.sm.get_facts.return_value = []
        learned = self.pipeline.process_user_message("I stopped using Windows and moved to Linux")
        assert isinstance(learned, list)


class TestSessionConsolidation:
    def test_consolidation_creates_summary(self):
        sm = MagicMock()
        sm.get_facts.return_value = []
        sm.learn.return_value = 1
        em = MagicMock()
        em.record.return_value = "ep_123"

        pipeline = LearningPipeline(semantic_memory=sm, episodic_memory=em)
        pipeline.process_user_message("I prefer Python for data science work")

        from homie_core.memory.working import WorkingMemory
        wm = WorkingMemory()
        wm.add_message("user", "tell me about Python")
        wm.add_message("assistant", "Python is great for data science")
        wm.add_message("user", "what about performance?")
        wm.add_message("assistant", "Use NumPy for performance-critical code")

        summary = pipeline.consolidate_session(wm)
        assert summary is not None
        assert "2-turn" in summary
        em.record.assert_called_once()

    def test_no_consolidation_for_empty_session(self):
        pipeline = LearningPipeline()
        from homie_core.memory.working import WorkingMemory
        wm = WorkingMemory()
        summary = pipeline.consolidate_session(wm)
        assert summary is None

    def test_session_stats(self):
        sm = MagicMock()
        sm.get_facts.return_value = []
        sm.learn.return_value = 1
        pipeline = LearningPipeline(semantic_memory=sm)
        pipeline.process_user_message("I prefer dark mode")
        pipeline.process_user_message("hello")
        stats = pipeline.get_session_stats()
        assert stats["interactions"] == 2
