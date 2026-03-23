"""Data curator — collects and exports SFT/DPO training data."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DataCurator:
    """Curates training data from Homie's interactions."""

    def __init__(self, storage) -> None:
        self._storage = storage

    def collect_sft(
        self,
        system_prompt: str,
        user_message: str,
        response: str,
        quality_score: float,
    ) -> None:
        """Collect an SFT training example."""
        data = {
            "instruction": system_prompt,
            "input": user_message,
            "output": response,
        }
        self._storage.save_training_example(
            example_type="sft",
            data=json.dumps(data),
            quality_score=quality_score,
        )

    def collect_dpo(
        self,
        user_message: str,
        chosen: str,
        rejected: str,
    ) -> None:
        """Collect a DPO preference pair."""
        data = {
            "prompt": user_message,
            "chosen": chosen,
            "rejected": rejected,
        }
        self._storage.save_training_example(
            example_type="dpo",
            data=json.dumps(data),
            quality_score=0.0,
        )

    def export_sft(self, output_path: Path | str, min_quality: float = 0.5) -> int:
        """Export SFT examples as JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        examples = self._storage.get_training_examples(example_type="sft")
        count = 0
        with open(output_path, "w") as f:
            for ex in examples:
                if ex.get("quality_score", 0) >= min_quality:
                    f.write(ex["data"] + "\n")
                    count += 1
        logger.info("Exported %d SFT examples to %s", count, output_path)
        return count

    def export_dpo(self, output_path: Path | str) -> int:
        """Export DPO pairs as JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        examples = self._storage.get_training_examples(example_type="dpo")
        count = 0
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(ex["data"] + "\n")
                count += 1
        logger.info("Exported %d DPO pairs to %s", count, output_path)
        return count

    def get_stats(self) -> dict:
        """Get training data statistics."""
        return self._storage.count_training_examples()
