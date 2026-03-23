"""Homie ML — Local machine learning pipeline for classification, embeddings, and pattern recognition."""

from homie_core.ml.base import LocalModel
from homie_core.ml.classifier import TextClassifier
from homie_core.ml.embedder import LocalEmbedder
from homie_core.ml.pattern_detector import PatternDetector
from homie_core.ml.trainer import ModelTrainer
from homie_core.ml.registry import MLModelRegistry

__all__ = [
    "LocalModel",
    "TextClassifier",
    "LocalEmbedder",
    "PatternDetector",
    "ModelTrainer",
    "MLModelRegistry",
]
