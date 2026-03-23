"""Tests for ModelTrainer — training lifecycle management."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from homie_core.ml.trainer import ModelTrainer
from homie_core.ml.classifier import TextClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEXTS = ["hello", "world", "good", "bad", "great", "terrible"]
LABELS = ["pos", "pos", "pos", "neg", "pos", "neg"]


class FakeStorage:
    """Minimal storage stub with get/put."""

    def __init__(self):
        self._data = {}

    def get(self, key):
        return self._data.get(key)

    def put(self, key, value):
        self._data[key] = value


@pytest.fixture
def storage():
    s = FakeStorage()
    s.put("intent_data", {"X": TEXTS, "y": LABELS})
    s.put("list_data", [
        {"text": "nice", "label": "pos"},
        {"text": "awful", "label": "neg"},
    ])
    return s


@pytest.fixture
def trainer(storage, tmp_path):
    return ModelTrainer(storage=storage, models_dir=tmp_path / "models")


@pytest.fixture
def clf():
    return TextClassifier("sentiment", classes=["pos", "neg"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCollectTrainingData:
    def test_collect_dict_format(self, trainer):
        n = trainer.collect_training_data("sentiment", "intent_data")
        assert n == len(TEXTS)

    def test_collect_list_format(self, trainer):
        n = trainer.collect_training_data("sentiment", "list_data")
        assert n == 2

    def test_collect_missing_source(self, trainer):
        n = trainer.collect_training_data("sentiment", "nonexistent")
        assert n == 0

    def test_collect_no_storage_raises(self, tmp_path):
        trainer = ModelTrainer(storage=None, models_dir=tmp_path / "m")
        with pytest.raises(RuntimeError, match="No storage"):
            trainer.collect_training_data("x", "y")


class TestAddTrainingData:
    def test_add_directly(self, trainer):
        n = trainer.add_training_data("model", ["a", "b"], ["x", "y"])
        assert n == 2

    def test_add_mismatched_raises(self, trainer):
        with pytest.raises(ValueError, match="same length"):
            trainer.add_training_data("model", ["a"], ["x", "y"])


class TestTrainModel:
    def test_train_returns_metrics(self, trainer, clf):
        trainer.register_model(clf)
        trainer.collect_training_data("sentiment", "intent_data")
        metrics = trainer.train_model("sentiment")
        assert "accuracy" in metrics
        assert "training_time_s" in metrics
        assert "artifact_path" in metrics

    def test_train_unregistered_raises(self, trainer):
        with pytest.raises(KeyError):
            trainer.train_model("ghost")

    def test_train_no_data_raises(self, trainer, clf):
        trainer.register_model(clf)
        with pytest.raises(ValueError, match="No training data"):
            trainer.train_model("sentiment")


class TestEvaluateModel:
    def test_evaluate(self, trainer, clf):
        trainer.register_model(clf)
        trainer.add_training_data("sentiment", TEXTS, LABELS)
        trainer.train_model("sentiment")
        test_data = [{"x": "hello", "y": "pos"}, {"x": "terrible", "y": "neg"}]
        result = trainer.evaluate_model("sentiment", test_data)
        assert "accuracy" in result
        assert result["n_test_samples"] == 2

    def test_evaluate_untrained_raises(self, trainer, clf):
        trainer.register_model(clf)
        with pytest.raises(RuntimeError, match="not been trained"):
            trainer.evaluate_model("sentiment", [{"x": "hi", "y": "pos"}])


class TestDeployModel:
    def test_deploy_trained_model(self, trainer, clf):
        trainer.register_model(clf)
        trainer.add_training_data("sentiment", TEXTS, LABELS)
        trainer.train_model("sentiment")
        assert trainer.deploy_model("sentiment") is True

    def test_deploy_untrained_raises(self, trainer, clf):
        trainer.register_model(clf)
        with pytest.raises(RuntimeError, match="untrained"):
            trainer.deploy_model("sentiment")


class TestListModels:
    def test_list_models(self, trainer, clf):
        trainer.register_model(clf)
        models = trainer.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "sentiment"
        assert models[0]["trained"] is False
