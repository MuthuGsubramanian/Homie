"""Homie Vision Plugin — Qwen3-VL-2B-Instruct.

A local vision-language plugin that gives Homie the ability to *see*:
screenshots, photos, diagrams, scanned documents. It loads
``Qwen/Qwen3-VL-2B-Instruct`` via Hugging Face ``transformers`` and
runs inference fully on-device once the weights are cached locally.

Design notes
------------
* No network traffic at runtime. The first-time weight download is
  *opt-in* via ``plugins.qwen3_vl.allow_download = true`` in
  ``homie.config.yaml`` — otherwise the plugin refuses to fetch and
  raises :class:`VisionPluginError`. This matches Homie's local-first
  posture (no implicit egress).
* Heavy dependencies (``torch``, ``transformers``, ``PIL``) are imported
  lazily inside :meth:`Qwen3VLPlugin._load_runtime` so that simply
  importing this module is cheap and does not hard-require a GPU stack.
* Lifecycle follows the existing Homie convention: a ``register()``
  factory plus ``activate()`` / ``deactivate()`` pair, mirroring how
  :mod:`homie.controller.scheduler` and :mod:`homie.controller.automations`
  are wired up.
* Configuration is read through :func:`homie.config.cfg_get`, the same
  helper used by the orchestrator and LLM client.

Expected ``homie.config.yaml`` block::

    plugins:
      qwen3_vl:
        model_id: Qwen/Qwen3-VL-2B-Instruct
        device: auto          # auto | cpu | cuda | mps
        dtype: auto           # auto | float16 | bfloat16 | float32
        allow_download: false # flip to true to fetch weights once
        max_new_tokens: 256
        cache_dir: null       # optional HF cache override

Typical use from another Homie module::

    from homie.plugins.qwen3_vl_plugin import activate
    vision = activate(cfg)
    print(vision.describe_image("/tmp/screenshot.png"))
    print(vision.ask_about_image("/tmp/chart.png", "What is the y-axis unit?"))
    print(vision.ocr_image("/tmp/receipt.jpg"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from homie.config import HomieConfig, cfg_get

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 256

ImageInput = Union[str, Path, "object"]  # path or PIL.Image.Image


class VisionPluginError(Exception):
    """Raised when the vision plugin cannot fulfil a request."""


@dataclass
class VisionConfig:
    """Resolved configuration for the Qwen3-VL plugin."""

    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    dtype: str = "auto"
    allow_download: bool = False
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None

    @classmethod
    def from_homie(cls, cfg: HomieConfig) -> "VisionConfig":
        """Build a :class:`VisionConfig` from a loaded :class:`HomieConfig`."""
        return cls(
            model_id=cfg_get(cfg, "plugins", "qwen3_vl", "model_id", default=DEFAULT_MODEL_ID),
            device=cfg_get(cfg, "plugins", "qwen3_vl", "device", default="auto"),
            dtype=cfg_get(cfg, "plugins", "qwen3_vl", "dtype", default="auto"),
            allow_download=bool(
                cfg_get(cfg, "plugins", "qwen3_vl", "allow_download", default=False)
            ),
            max_new_tokens=int(
                cfg_get(
                    cfg,
                    "plugins",
                    "qwen3_vl",
                    "max_new_tokens",
                    default=DEFAULT_MAX_NEW_TOKENS,
                )
            ),
            trust_remote_code=bool(
                cfg_get(cfg, "plugins", "qwen3_vl", "trust_remote_code", default=True)
            ),
            cache_dir=cfg_get(cfg, "plugins", "qwen3_vl", "cache_dir", default=None),
        )


class Qwen3VLPlugin:
    """Local vision-language plugin powered by ``Qwen3-VL-2B-Instruct``.

    The plugin is intentionally stateless across calls: the model and
    processor are loaded once on :meth:`activate` and released on
    :meth:`deactivate`. Each ``describe_image`` / ``ask_about_image`` /
    ``ocr_image`` call constructs a fresh single-turn chat — there is no
    conversation memory kept inside the plugin, which keeps it safe to
    call concurrently from the orchestrator.
    """

    name: str = "qwen3_vl"
    version: str = "0.1.0"
    capabilities: Tuple[str, ...] = (
        "image-text-to-text",
        "ocr",
        "screenshot-understanding",
    )

    def __init__(self, vcfg: VisionConfig) -> None:
        self.vcfg = vcfg
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device: Optional[str] = None
        self._active: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self) -> None:
        """Load model + processor into memory. Idempotent."""
        if self._active:
            return
        self._load_runtime()
        self._active = True
        logger.info(
            "qwen3_vl plugin activated (model=%s, device=%s, dtype=%s)",
            self.vcfg.model_id,
            self._device,
            self.vcfg.dtype,
        )

    def deactivate(self) -> None:
        """Release model + processor references. Idempotent."""
        if not self._active:
            return
        self._model = None
        self._processor = None
        if self._torch is not None:
            try:
                if self._device == "cuda" and self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                logger.debug("qwen3_vl: cuda empty_cache failed", exc_info=True)
        self._active = False
        logger.info("qwen3_vl plugin deactivated")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def describe_image(
        self,
        image: ImageInput,
        prompt: str = "Describe the image in detail.",
    ) -> str:
        """Return a natural-language description of ``image``."""
        return self._chat(image, prompt)

    def ask_about_image(self, image: ImageInput, question: str) -> str:
        """Answer a free-form ``question`` grounded in ``image``."""
        if not question or not question.strip():
            raise VisionPluginError("question must be a non-empty string")
        return self._chat(image, question)

    def ocr_image(self, image: ImageInput) -> str:
        """Extract text content from ``image`` (best-effort OCR)."""
        return self._chat(
            image,
            "Extract all readable text from this image verbatim. "
            "Preserve line breaks and reading order. Output text only.",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_runtime(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoProcessor,
                AutoModelForImageTextToText,
            )
        except ImportError as exc:
            raise VisionPluginError(
                "Missing dependencies. Install with: "
                "pip install torch transformers pillow"
            ) from exc

        self._torch = torch
        self._device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch)

        common_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.vcfg.trust_remote_code,
        }
        if self.vcfg.cache_dir:
            common_kwargs["cache_dir"] = self.vcfg.cache_dir
        if not self.vcfg.allow_download:
            common_kwargs["local_files_only"] = True

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.vcfg.model_id, **common_kwargs
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.vcfg.model_id, torch_dtype=dtype, **common_kwargs
            )
        except OSError as exc:
            raise VisionPluginError(
                f"Model '{self.vcfg.model_id}' is not available locally. "
                "Set plugins.qwen3_vl.allow_download=true in homie.config.yaml "
                "to fetch the weights once, then flip it back to false."
            ) from exc

        self._model.to(self._device)
        self._model.eval()

    def _resolve_device(self, torch: Any) -> str:
        choice = (self.vcfg.device or "auto").lower()
        if choice != "auto":
            return choice
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, torch: Any) -> Any:
        choice = (self.vcfg.dtype or "auto").lower()
        if choice == "float16":
            return torch.float16
        if choice == "bfloat16":
            return torch.bfloat16
        if choice == "float32":
            return torch.float32
        # auto
        if self._device == "cuda":
            return torch.float16
        if self._device == "mps":
            return torch.float16
        return torch.float32

    def _load_image(self, image: ImageInput) -> Any:
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise VisionPluginError(
                "Pillow is required for image loading. pip install pillow"
            ) from exc

        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise VisionPluginError(f"Image not found: {path}")
            return Image.open(str(path)).convert("RGB")
        # Assume it is already a PIL.Image.Image
        return image

    def _build_messages(self, pil_image: Any, prompt: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _chat(self, image: ImageInput, prompt: str) -> str:
        if not self._active:
            self.activate()

        pil_image = self._load_image(image)
        messages = self._build_messages(pil_image, prompt)

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:  # noqa: BLE001
            # Older processor variants: fall back to the two-step path.
            text = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self._processor(
                text=[text], images=[pil_image], return_tensors="pt"
            )

        inputs = inputs.to(self._device)
        input_len = int(inputs["input_ids"].shape[1])

        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.vcfg.max_new_tokens,
                do_sample=False,
            )

        trimmed = generated[:, input_len:]
        decoded = self._processor.batch_decode(trimmed, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""


# ----------------------------------------------------------------------
# Module-level lifecycle helpers (Homie plugin entrypoints)
# ----------------------------------------------------------------------
def register(cfg: HomieConfig) -> Qwen3VLPlugin:
    """Construct the plugin without loading the model. Cheap; safe at import."""
    vcfg = VisionConfig.from_homie(cfg)
    return Qwen3VLPlugin(vcfg)


def activate(cfg: HomieConfig) -> Qwen3VLPlugin:
    """Construct and immediately activate the plugin."""
    plugin = register(cfg)
    plugin.activate()
    return plugin


def deactivate(plugin: Qwen3VLPlugin) -> None:
    """Release resources held by ``plugin``."""
    plugin.deactivate()


__all__ = [
    "Qwen3VLPlugin",
    "VisionConfig",
    "VisionPluginError",
    "register",
    "activate",
    "deactivate",
    "DEFAULT_MODEL_ID",
]
