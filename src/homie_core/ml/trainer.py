"""Training pipeline — data collection, training, evaluation, and deployment of local ML models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from homie_core.ml.base import LocalModel


class ModelTrainer:
    """Manages the full training lifecycle for locally-trained models.

    Parameters
    ----------
    storage
        Any object that exposes a ``get(key)`` and ``put(key, value)`` method.
        Used for retrieving training data.  Can be ``None`` for manual usage.
    models_dir
        Directory where trained model artifacts are saved.
    """

    def __init__(self, storage: Any, models_dir: Path) -> None:
        self.storage = storage
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._models: dict[str, LocalModel] = {}
        self._training_data: dict[str, dict[str, list]] = {}  # model_name -> {X: [], y: []}
        self._history: list[dict[str, Any]] = []
        self._deployed: dict[str, str] = {}  # model_name -> version_tag

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_training_data(self, model_name: str, data_source: str) -> int:
        """Collect training data for *model_name* from *data_source*.

        *data_source* is a key that can be looked up in ``self.storage``.
        Returns the number of samples collected.
        """
        if self.storage is None:
            raise RuntimeError("No storage backend configured.")

        raw = self.storage.get(data_source)
        if raw is None:
            return 0

        if isinstance(raw, dict) and "X" in raw and "y" in raw:
            X = raw["X"]
            y = raw["y"]
        elif isinstance(raw, list):
            # Assume list of {"text": ..., "label": ...} dicts
            X = [item.get("text", item.get("x", "")) for item in raw]
            y = [item.get("label", item.get("y", "")) for item in raw]
        else:
            return 0

        entry = self._training_data.setdefault(model_name, {"X": [], "y": []})
        entry["X"].extend(X)
        entry["y"].extend(y)
        return len(X)

    def add_training_data(self, model_name: str, X: list, y: list) -> int:
        """Directly add training samples for *model_name*."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        entry = self._training_data.setdefault(model_name, {"X": [], "y": []})
        entry["X"].extend(X)
        entry["y"].extend(y)
        return len(X)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def register_model(self, model: LocalModel) -> None:
        """Register a model instance to be managed by this trainer."""
        self._models[model.name] = model

    def train_model(self, model_name: str) -> dict:
        """Train the registered model *model_name* on its collected data.

        Returns training metrics.
        """
        model = self._models.get(model_name)
        if model is None:
            raise KeyError(f"No model registered with name {model_name!r}.")
        data = self._training_data.get(model_name)
        if data is None or not data["X"]:
            raise ValueError(f"No training data collected for {model_name!r}.")

        start = time.time()
        metrics = model.train(data["X"], data["y"])
        elapsed = time.time() - start
        metrics["training_time_s"] = round(elapsed, 4)

        # Save to history
        record = {
            "model_name": model_name,
            "timestamp": time.time(),
            "metrics": metrics,
            "n_samples": len(data["X"]),
        }
        self._history.append(record)

        # Auto-save artifact
        artifact_path = self.models_dir / f"{model_name}.model"
        model.save(artifact_path)
        metrics["artifact_path"] = str(artifact_path)

        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_model(self, model_name: str, test_data: list) -> dict:
        """Evaluate *model_name* on *test_data*.

        *test_data* should be a list of ``{"x": ..., "y": ...}`` dicts.
        Returns evaluation metrics including ``accuracy``.
        """
        model = self._models.get(model_name)
        if model is None:
            raise KeyError(f"No model registered with name {model_name!r}.")
        if not model.is_trained:
            raise RuntimeError(f"Model {model_name!r} has not been trained yet.")

        X_test = [item.get("x", item.get("text", "")) for item in test_data]
        y_true = [item.get("y", item.get("label", "")) for item in test_data]

        y_pred = model.predict(X_test)
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        accuracy = correct / len(y_true) if y_true else 0.0

        return {
            "model_name": model_name,
            "accuracy": accuracy,
            "n_test_samples": len(test_data),
            "correct": correct,
        }

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------

    def deploy_model(self, model_name: str) -> bool:
        """Mark *model_name* as the active/deployed version.

        Returns ``True`` on success.
        """
        model = self._models.get(model_name)
        if model is None:
            raise KeyError(f"No model registered with name {model_name!r}.")
        if not model.is_trained:
            raise RuntimeError(f"Cannot deploy untrained model {model_name!r}.")

        version = f"v{len(self._history)}"
        self._deployed[model_name] = version

        # Copy artifact to a "deployed" directory
        deployed_dir = self.models_dir / "deployed"
        deployed_dir.mkdir(parents=True, exist_ok=True)
        src = self.models_dir / f"{model_name}.model"
        dst = deployed_dir / f"{model_name}.model"
        if src.exists():
            dst.write_bytes(src.read_bytes())

        return True

    # ------------------------------------------------------------------
    # Listing / introspection
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Return a summary of every registered model."""
        result = []
        for name, model in self._models.items():
            data = self._training_data.get(name, {"X": [], "y": []})
            result.append({
                "name": name,
                "type": model.model_type,
                "trained": model.is_trained,
                "deployed": name in self._deployed,
                "n_training_samples": len(data["X"]),
                "metrics": model.metrics,
            })
        return result

    def get_training_history(self) -> list[dict]:
        """Return the full training history."""
        return list(self._history)
