"""Qwen3-VL-2B-Instruct vision-language plugin for Homie.

Provides on-device image and screenshot question answering using the
``Qwen/Qwen3-VL-2B-Instruct`` multimodal model from Hugging Face. The plugin
loads the model lazily into memory the first time it is invoked, so activation
is cheap; subsequent calls reuse the cached model and processor.

All inference runs locally via ``transformers``. The plugin makes no network
calls at runtime once the model weights are cached on disk. The initial
weight download from the Hugging Face Hub is opt-in: callers must pass
``allow_download=True`` (or set ``HOMIE_QWEN_VL_ALLOW_DOWNLOAD=1``) the first
time the plugin is used, mirroring Homie's local-first stance.

Typical usage from the Homie shell or another plugin:

    from homie.plugins.qwen3_vl_vision import register

    plugin = register()
    plugin.activate()
    answer = plugin.describe_image("/tmp/screenshot.png", "What is on screen?")
    plugin.deactivate()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from homie.config import HomieConfig, cfg_get

logger = logging.getLogger(__name__)

PLUGIN_NAME = "qwen3_vl_vision"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_DEVICE = "auto"

ImageInput = Union[str, Path, "PIL.Image.Image"]


@dataclass
class VisionPluginSettings:
    """Runtime settings resolved from Homie config + environment."""

    model_id: str = DEFAULT_MODEL_ID
    device: str = DEFAULT_DEVICE
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    allow_download: bool = False
    cache_dir: Optional[Path] = None
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Optional[HomieConfig]) -> "VisionPluginSettings":
        """Build settings from a Homie config, falling back to env + defaults."""
        if cfg is None:
            return cls(
                allow_download=os.getenv("HOMIE_QWEN_VL_ALLOW_DOWNLOAD", "0") == "1",
            )
        cache_dir = cfg_get(cfg, "plugins", PLUGIN_NAME, "cache_dir", default=None)
        return cls(
            model_id=cfg_get(cfg, "plugins", PLUGIN_NAME, "model_id", default=DEFAULT_MODEL_ID),
            device=cfg_get(cfg, "plugins", PLUGIN_NAME, "device", default=DEFAULT_DEVICE),
            max_new_tokens=int(
                cfg_get(cfg, "plugins", PLUGIN_NAME, "max_new_tokens", default=DEFAULT_MAX_NEW_TOKENS)
            ),
            allow_download=bool(
                cfg_get(cfg, "plugins", PLUGIN_NAME, "allow_download", default=False)
            )
            or os.getenv("HOMIE_QWEN_VL_ALLOW_DOWNLOAD", "0") == "1",
            cache_dir=Path(cache_dir).expanduser() if cache_dir else None,
            torch_dtype=cfg_get(cfg, "plugins", PLUGIN_NAME, "torch_dtype", default="auto"),
            trust_remote_code=bool(
                cfg_get(cfg, "plugins", PLUGIN_NAME, "trust_remote_code", default=True)
            ),
        )


class Qwen3VLVisionPlugin:
    """Local vision-language plugin backed by Qwen3-VL-2B-Instruct.

    The plugin follows Homie's activate/deactivate lifecycle. Heavy imports
    (``torch``, ``transformers``, ``PIL``) are deferred until ``activate`` so
    that simply importing this module is cheap and doesn't drag the GPU/CPU
    inference stack into every Homie session.
    """

    name = PLUGIN_NAME
    description = "On-device image and screenshot Q&A using Qwen3-VL-2B-Instruct."

    def __init__(self, cfg: Optional[HomieConfig] = None, settings: Optional[VisionPluginSettings] = None) -> None:
        self.cfg = cfg
        self.settings = settings or VisionPluginSettings.from_config(cfg)
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._active: bool = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def activate(self) -> None:
        """Mark the plugin as ready. Model loading is deferred to first use."""
        if self._active:
            return
        logger.info("Activating %s (model=%s, device=%s)", PLUGIN_NAME, self.settings.model_id, self.settings.device)
        self._active = True

    def deactivate(self) -> None:
        """Release model + processor references and free GPU memory if possible."""
        if not self._active:
            return
        logger.info("Deactivating %s", PLUGIN_NAME)
        self._model = None
        self._processor = None
        try:
            if self._torch is not None and hasattr(self._torch, "cuda") and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            logger.debug("cuda empty_cache failed", exc_info=True)
        self._active = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def describe_image(
        self,
        image: ImageInput,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Answer a natural-language ``prompt`` about a single ``image``.

        Args:
            image: Path to an image file or a preloaded ``PIL.Image``.
            prompt: Question or instruction for the model.
            max_new_tokens: Optional cap on response tokens.

        Returns:
            The model's textual response with any chat-template wrapping stripped.
        """
        return self.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_new_tokens=max_new_tokens,
        )

    def answer_screenshot(
        self,
        prompt: str = "What is currently on the screen?",
        screenshot_path: Optional[Union[str, Path]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Capture (or read) a screenshot and ask the model a question about it.

        Screenshot capture is opt-in: if ``screenshot_path`` is not provided the
        plugin will attempt to grab the current screen via ``mss`` if installed.
        If ``mss`` is not available an informative error is raised so the user
        can install it explicitly rather than pulling it in silently.
        """
        path = Path(screenshot_path) if screenshot_path else self._capture_screenshot()
        if not path.exists():
            raise FileNotFoundError(f"Screenshot not found: {path}")
        return self.describe_image(path, prompt=prompt, max_new_tokens=max_new_tokens)

    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Run a multimodal chat exchange with Qwen3-VL.

        ``messages`` follows the OpenAI-style multimodal schema used by
        ``transformers`` chat templates: each message is
        ``{"role": ..., "content": [{"type": "image"|"text", ...}]}``.
        """
        if not self._active:
            raise RuntimeError("Plugin not active. Call activate() first.")
        self._ensure_model_loaded()

        torch = self._torch
        processor = self._processor
        model = self._model

        images = self._collect_images(messages)
        text = processor.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=images or None,
            return_tensors="pt",
            padding=True,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.settings.max_new_tokens,
                do_sample=False,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch  # type: ignore
            from transformers import AutoProcessor  # type: ignore
            try:
                from transformers import Qwen3VLForConditionalGeneration as _ModelCls  # type: ignore
            except ImportError:
                from transformers import AutoModelForVision2Seq as _ModelCls  # type: ignore
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "qwen3_vl_vision requires 'torch', 'transformers', and 'Pillow'. "
                "Install with: pip install torch transformers pillow"
            ) from exc

        self._torch = torch

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.settings.trust_remote_code,
        }
        if self.settings.cache_dir is not None:
            load_kwargs["cache_dir"] = str(self.settings.cache_dir)
        if not self.settings.allow_download:
            # Force offline mode unless the user has explicitly opted in.
            load_kwargs["local_files_only"] = True

        dtype = self._resolve_dtype(torch, self.settings.torch_dtype)
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if self.settings.device and self.settings.device != "cpu":
            load_kwargs["device_map"] = self.settings.device

        logger.info("Loading Qwen3-VL model '%s'", self.settings.model_id)
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.settings.model_id,
                trust_remote_code=self.settings.trust_remote_code,
                local_files_only=load_kwargs.get("local_files_only", False),
                cache_dir=load_kwargs.get("cache_dir"),
            )
            self._model = _ModelCls.from_pretrained(self.settings.model_id, **load_kwargs)
            self._model.eval()
        except OSError as exc:
            if not self.settings.allow_download:
                raise RuntimeError(
                    f"Model weights for '{self.settings.model_id}' not found locally and "
                    "network downloads are disabled. Set allow_download=True or "
                    "HOMIE_QWEN_VL_ALLOW_DOWNLOAD=1 to fetch them."
                ) from exc
            raise

    @staticmethod
    def _resolve_dtype(torch: Any, dtype_name: str) -> Any:
        mapping = {
            "auto": None,
            "float16": getattr(torch, "float16", None),
            "fp16": getattr(torch, "float16", None),
            "bfloat16": getattr(torch, "bfloat16", None),
            "bf16": getattr(torch, "bfloat16", None),
            "float32": getattr(torch, "float32", None),
            "fp32": getattr(torch, "float32", None),
        }
        return mapping.get(dtype_name.lower(), None)

    @staticmethod
    def _collect_images(messages: Sequence[Dict[str, Any]]) -> List[Any]:
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError("Pillow is required for image inputs.") from exc

        images: List[Any] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image":
                    continue
                src = part.get("image")
                if isinstance(src, (str, Path)):
                    images.append(Image.open(src).convert("RGB"))
                else:
                    images.append(src)
        return images

    def _capture_screenshot(self) -> Path:
        try:
            import mss  # type: ignore
            import mss.tools  # type: ignore
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "Screenshot capture requires the 'mss' package. "
                "Install with: pip install mss"
            ) from exc

        out_dir = Path.home() / ".homie" / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"screen_{os.getpid()}.png"
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            shot = sct.grab(monitor)
            mss.tools.to_png(shot.rgb, shot.size, output=str(out_path))
        return out_path


# ---------------------------------------------------------------------- #
# Plugin entrypoints
# ---------------------------------------------------------------------- #
def register(cfg: Optional[HomieConfig] = None) -> Qwen3VLVisionPlugin:
    """Construct and return the plugin instance.

    Homie's plugin loader calls ``register`` once per session. The returned
    plugin is not activated yet; the caller should invoke ``activate()`` when
    ready, and ``deactivate()`` on shutdown.
    """
    return Qwen3VLVisionPlugin(cfg=cfg)


def activate(cfg: Optional[HomieConfig] = None) -> Qwen3VLVisionPlugin:
    """Convenience wrapper: register + activate in one call."""
    plugin = register(cfg=cfg)
    plugin.activate()
    return plugin


def deactivate(plugin: Qwen3VLVisionPlugin) -> None:
    """Deactivate a plugin instance returned by ``activate``/``register``."""
    plugin.deactivate()


__all__ = [
    "PLUGIN_NAME",
    "Qwen3VLVisionPlugin",
    "VisionPluginSettings",
    "register",
    "activate",
    "deactivate",
]
