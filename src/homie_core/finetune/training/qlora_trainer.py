"""QLoRA trainer wrapping unsloth/peft + trl for SFT and DPO training."""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Callable, Optional

from homie_core.finetune.config import TrainingConfig

logger = logging.getLogger(__name__)


def _check_gpu_available() -> bool:
    """Return True when a CUDA GPU is reachable."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _detect_wsl2() -> bool:
    """Return True when running inside WSL2."""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


class QLoRATrainer:
    """Orchestrates QLoRA-based SFT and DPO training runs."""

    def __init__(
        self,
        base_model: str,
        config: TrainingConfig,
        output_dir: str | Path,
    ):
        self.base_model = base_model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def preflight_check(self) -> tuple[bool, str]:
        """Check GPU available, VRAM sufficient, deps installed."""
        if not _check_gpu_available():
            return False, "No GPU available or insufficient VRAM"
        # Check deps: try import unsloth, fallback peft
        try:
            import unsloth  # noqa: F401

            logger.info("Using unsloth backend")
        except ImportError:
            try:
                import peft  # noqa: F401

                logger.info("Falling back to peft backend")
            except ImportError:
                return False, "Neither unsloth nor peft is installed"
        return True, "Ready"

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def format_sft_example(self, example: dict) -> str:
        """Format a ChatML example to a training string."""
        system = example.get("system", "")
        user = example.get("user", "")
        assistant = example.get("assistant", "")
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}"

    def format_dpo_example(self, example: dict) -> dict:
        """Format a DPO example to trl DPOTrainer format."""
        prompt = (
            f"<|system|>\n{example.get('system', '')}\n"
            f"<|user|>\n{example.get('user', '')}"
        )
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    @staticmethod
    def convert_alpaca_to_chatml(example: dict) -> dict:
        """Convert Alpaca ``{instruction, input, output}`` to ChatML ``{system, user, assistant}``."""
        user_msg = example.get("instruction", "")
        inp = example.get("input", "")
        if inp:
            user_msg = f"{user_msg}\n{inp}"
        return {
            "system": "",
            "user": user_msg,
            "assistant": example.get("output", ""),
        }

    # ------------------------------------------------------------------
    # Training entry-points (require GPU)
    # ------------------------------------------------------------------

    def train_sft(
        self,
        dataset_path: Path,
        checkpoint_callback: Optional[Callable] = None,
    ) -> dict:
        """Run SFT training. Returns ``{"loss": float, "adapter_path": str}``."""
        raise NotImplementedError("Requires GPU -- use mock in tests")

    def train_dpo(
        self,
        dataset_path: Path,
        sft_adapter_path: Path,
        checkpoint_callback: Optional[Callable] = None,
    ) -> dict:
        """Run DPO training. Returns ``{"loss": float, "adapter_path": str}``."""
        raise NotImplementedError("Requires GPU -- use mock in tests")
