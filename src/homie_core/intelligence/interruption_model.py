from __future__ import annotations

import math
from typing import Any


_CATEGORIES = ["health", "calendar", "suggestion", "task", "reminder", "other"]


def _category_features(category: str) -> list[float]:
    vec = [0.0] * len(_CATEGORIES)
    idx = _CATEGORIES.index(category) if category in _CATEGORIES else len(_CATEGORIES) - 1
    vec[idx] = 1.0
    return vec


def _build_features(minutes_in_task: float, switch_freq_10min: float,
                    minutes_since_interaction: float, category: str) -> list[float]:
    return [
        min(minutes_in_task / 120.0, 1.0),
        min(switch_freq_10min / 20.0, 1.0),
        min(minutes_since_interaction / 120.0, 1.0),
    ] + _category_features(category)


def _sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class InterruptionModel:
    """Logistic regression predicting user acceptance of interruptions.
    Trained online via SGD. Pure Python, no external deps."""

    def __init__(self, threshold: float = 0.7, learning_rate: float = 0.1):
        n_features = 3 + len(_CATEGORIES)
        self._weights = [0.0] * n_features
        self._bias = 0.0
        self._threshold = threshold
        self._lr = learning_rate
        self._n_samples = 0

    def predict(self, minutes_in_task: float, switch_freq_10min: float,
                minutes_since_interaction: float, category: str) -> float:
        features = _build_features(minutes_in_task, switch_freq_10min,
                                   minutes_since_interaction, category)
        z = self._bias + sum(w * x for w, x in zip(self._weights, features))
        return _sigmoid(z)

    def should_interrupt(self, minutes_in_task: float, switch_freq_10min: float,
                         minutes_since_interaction: float, category: str) -> bool:
        return self.predict(minutes_in_task, switch_freq_10min,
                           minutes_since_interaction, category) >= self._threshold

    def record_feedback(self, accepted: bool, minutes_in_task: float,
                        switch_freq_10min: float, minutes_since_interaction: float,
                        category: str) -> None:
        features = _build_features(minutes_in_task, switch_freq_10min,
                                   minutes_since_interaction, category)
        y = 1.0 if accepted else 0.0
        p = self.predict(minutes_in_task, switch_freq_10min,
                         minutes_since_interaction, category)
        error = y - p
        for i in range(len(self._weights)):
            self._weights[i] += self._lr * error * features[i]
        self._bias += self._lr * error
        self._n_samples += 1

    def serialize(self) -> dict:
        return {
            "weights": list(self._weights), "bias": self._bias,
            "threshold": self._threshold, "lr": self._lr,
            "n_samples": self._n_samples,
        }

    @classmethod
    def deserialize(cls, data: dict) -> InterruptionModel:
        model = cls(threshold=data["threshold"], learning_rate=data["lr"])
        model._weights = list(data["weights"])
        model._bias = data["bias"]
        model._n_samples = data["n_samples"]
        return model
