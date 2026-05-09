"""Homie Visual Understanding Plugin â€” Qwen3-VL-2B-Instruct

Provides local vision-language capabilities using the Qwen3-VL-2B-Instruct model.
Supports image captioning, visual question answering, and structured extraction
from images, all running entirely on-device with no network calls.

The plugin loads the model lazily on first use and exposes methods that Homie
can invoke to understand screenshots, photos, documents, or any visual input
a user provides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_DEVICE = "cpu"


@dataclass
class VisionResult:
    """Container for a single vision inference result."""

    query: str
    response: str
    image_path: str
    model: str = MODEL_ID
    error: Optional[str] = None


class Qwen3VLVisualUnderstandingPlugin:
    """Local vision-language plugin powered by Qwen3-VL-2B-Instruct.

    Provides on-device image understanding â€” captioning, visual QA, and
    structured extraction â€” without any network dependency after the
    initial model download.

    Usage::

        plugin = Qwen3VLVisualUnderstandingPlugin()
        plugin.activate()
        result = plugin.ask("screenshot.png", "What application is open?")
        print(result.response)
        plugin.deactivate()
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = DEFAULT_DEVICE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        torch_dtype: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype

        self._model: Any = None
        self._processor: Any = None
        self._active: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Load model and processor into memory.

        This is intentionally lazy â€” nothing is downloaded or loaded until
        the host application explicitly calls ``activate()``.
        """
        if self._active:
            logger.debug("Plugin already active, skipping reload.")
            return

        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Required packages missing. Install with: "
                "pip install transformers torch accelerate qwen-vl-utils"
            ) from exc

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        resolved_dtype = dtype_map.get(self.torch_dtype, "auto")

        logger.info("Loading %s on device=%s ...", self.model_id, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=resolved_dtype,
            device_map=self.device if self.device != "cpu" else None,
        )
        if self.device == "cpu":
            self._model = self._model.to("cpu")

        self._model.eval()
        self._active = True
        logger.info("Visual understanding plugin activated.")

    def deactivate(self) -> None:
        """Unload model and free memory."""
        self._model = None
        self._processor = None
        self._active = False
        logger.info("Visual understanding plugin deactivated.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        image_path: str | Path,
        question: str = "Describe this image in detail.",
    ) -> VisionResult:
        """Ask a question about a single image.

        Args:
            image_path: Path to a local image file (PNG, JPEG, BMP, etc.).
            question: Natural-language question about the image.

        Returns:
            A ``VisionResult`` with the model's response.
        """
        self._ensure_active()
        image_path = Path(image_path)
        if not image_path.exists():
            return VisionResult(
                query=question,
                response="",
                image_path=str(image_path),
                error=f"Image not found: {image_path}",
            )

        try:
            response_text = self._infer(image_path, question)
            return VisionResult(
                query=question,
                response=response_text,
                image_path=str(image_path),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Vision inference failed for %s", image_path)
            return VisionResult(
                query=question,
                response="",
                image_path=str(image_path),
                error=str(exc),
            )

    def caption(self, image_path: str | Path) -> VisionResult:
        """Generate a descriptive caption for an image."""
        return self.ask(image_path, "Describe this image in one concise paragraph.")

    def extract(self, image_path: str | Path, schema_hint: str = "") -> VisionResult:
        """Extract structured information from an image (e.g. a form, receipt).

        Args:
            image_path: Path to the image.
            schema_hint: Optional hint describing desired output structure,
                e.g. ``"JSON with fields: date, total, items"``.
        """
        prompt = "Extract all visible text and structured data from this image."
        if schema_hint:
            prompt += f" Output format: {schema_hint}"
        return self.ask(image_path, prompt)

    def compare(self, image_paths: Sequence[str | Path], question: str = "") -> VisionResult:
        """Compare multiple images by asking about them sequentially.

        Processes images one at a time (single-image model) and combines
        the descriptions, then uses the model to summarise differences.
        """
        self._ensure_active()
        descriptions: List[str] = []
        for idx, img in enumerate(image_paths):
            result = self.caption(img)
            if result.error:
                descriptions.append(f"Image {idx + 1}: [error: {result.error}]")
            else:
                descriptions.append(f"Image {idx + 1}: {result.response}")

        combined = "\n".join(descriptions)
        comparison_prompt = (
            f"Given these image descriptions:\n{combined}\n\n"
            f"{question or 'What are the key differences between these images?'}"
        )
        # Use the model as a text-only follow-up to compare.
        return self._text_only_infer(comparison_prompt, image_paths)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_active(self) -> None:
        if not self._active or self._model is None:
            raise RuntimeError(
                "Plugin is not active. Call activate() before inference."
            )

    def _infer(self, image_path: Path, question: str) -> str:
        """Run single-image VQA through the Qwen3-VL chat template."""
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path.resolve()}"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        # Trim input tokens from output
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def _text_only_infer(
        self, prompt: str, ref_paths: Sequence[str | Path]
    ) -> VisionResult:
        """Run a text-only generation for comparison summaries."""
        import torch

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_input], return_tensors="pt"
        ).to(self._model.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        response = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return VisionResult(
            query=prompt,
            response=response,
            image_path="; ".join(str(p) for p in ref_paths),
        )


# ------------------------------------------------------------------
# Module-level convenience for Homie plugin registration
# ------------------------------------------------------------------

_plugin_instance: Optional[Qwen3VLVisualUnderstandingPlugin] = None


def register(config: Optional[Dict[str, Any]] = None) -> Qwen3VLVisualUnderstandingPlugin:
    """Register and return the plugin singleton.

    Called by the Homie plugin loader. The plugin is *not* activated
    automatically â€” call ``activate()`` when the user opts in to
    vision features so we avoid loading ~2 GB into RAM eagerly.

    Args:
        config: Optional dict with keys ``device``, ``max_new_tokens``,
            ``torch_dtype`` to override defaults.
    """
    global _plugin_instance
    if _plugin_instance is not None:
        return _plugin_instance

    cfg = config or {}
    _plugin_instance = Qwen3VLVisualUnderstandingPlugin(
        device=cfg.get("device", DEFAULT_DEVICE),
        max_new_tokens=int(cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
        torch_dtype=cfg.get("torch_dtype", "auto"),
    )
    logger.info("Qwen3-VL visual understanding plugin registered.")
    return _plugin_instance


def activate() -> Qwen3VLVisualUnderstandingPlugin:
    """Convenience: register (if needed) then activate."""
    plugin = register()
    plugin.activate()
    return plugin


def deactivate() -> None:
    """Deactivate the plugin and release model memory."""
    global _plugin_instance
    if _plugin_instance is not None:
        _plugin_instance.deactivate()
        _plugin_instance = None
