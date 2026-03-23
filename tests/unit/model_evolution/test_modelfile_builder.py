import pytest
from homie_core.model_evolution.modelfile_builder import ModelfileBuilder


class TestModelfileBuilder:
    def test_builds_basic_modelfile(self):
        builder = ModelfileBuilder(base_model="lfm2")
        content = builder.build()
        assert "FROM lfm2" in content
        assert "SYSTEM" in content

    def test_includes_base_personality(self):
        builder = ModelfileBuilder(base_model="lfm2", user_name="Master")
        content = builder.build()
        assert "Homie" in content
        assert "Master" in content

    def test_includes_preferences_layer(self):
        builder = ModelfileBuilder(base_model="lfm2")
        builder.set_preferences(verbosity="concise", formality="casual", depth="expert", format_pref="bullets")
        content = builder.build()
        assert "concise" in content.lower() or "brief" in content.lower()
        assert "bullet" in content.lower()

    def test_includes_knowledge_layer(self):
        builder = ModelfileBuilder(base_model="lfm2")
        builder.set_knowledge(["Works on Homie AI project", "Uses Python and ChromaDB"])
        content = builder.build()
        assert "Homie AI" in content
        assert "ChromaDB" in content

    def test_includes_instructions_layer(self):
        builder = ModelfileBuilder(base_model="lfm2")
        builder.set_instructions(["Show code diffs first", "Morning greeting includes git summary"])
        content = builder.build()
        assert "diff" in content.lower()

    def test_includes_customizations_layer(self):
        builder = ModelfileBuilder(base_model="lfm2")
        builder.set_customizations(["/standup: show git + calendar", "Morning briefing with project status"])
        content = builder.build()
        assert "standup" in content.lower()

    def test_includes_parameters(self):
        builder = ModelfileBuilder(base_model="lfm2")
        builder.set_parameters(temperature=0.5, num_ctx=32768)
        content = builder.build()
        assert "PARAMETER temperature 0.5" in content
        assert "PARAMETER num_ctx 32768" in content

    def test_write_to_file(self, tmp_path):
        builder = ModelfileBuilder(base_model="lfm2")
        path = tmp_path / "Modelfile"
        builder.write(path)
        assert path.exists()
        assert "FROM lfm2" in path.read_text()

    def test_content_hash(self):
        builder = ModelfileBuilder(base_model="lfm2")
        h1 = builder.content_hash()
        builder.set_knowledge(["New fact"])
        h2 = builder.content_hash()
        assert h1 != h2  # hash changes with content

    def test_same_content_same_hash(self):
        b1 = ModelfileBuilder(base_model="lfm2", user_name="Test")
        b2 = ModelfileBuilder(base_model="lfm2", user_name="Test")
        assert b1.content_hash() == b2.content_hash()
