"""Qwen3-VL local image Q&A plugin for Homie.

Wraps the Qwen/Qwen3-VL-2B-Instruct vision-language model so Homie can answer
questions about screenshots and local images entirely on-device. Inference runs
through Hugging Face ``transformers`` with safetensors weights; the first call
downloads the model into the local HF cache (this is the only network access
and only when ``allow_download`` is enabled in config). Subsequent calls are
fully offline.

Integration:
    The plugin registers a ``vision.image_qa`` capability with the host. Other
    Homie components (e.g. the AI planner, screenshot watcher, voice loop) can
    dispatch ``{"action": "image_qa", "image": <path>, "prompt": <text>}`` and
    receive a textual answer suitable for TTS or chat display.

Config keys (homie.config.yaml)::

    plugins:
      qwen3_vl_image_qa:
        enabled: true
        model_id: "Qwen/Qwen3-VL-2B-Instruct"
        device: "auto"            # auto | cpu | cuda | mps
        dtype: "auto"             # auto | float16 | bfloat16 | float32
        max_new_tokens: 256
        allow_download: false      # must be opt-in; offline by default
        cache_dir: null            # optional HF cache override
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from homie.config import HomieConfig, cfg_get

logger = logging.getLogger(__name__)

PLUGIN_NAME = "qwen3_vl_image_qa"
PLUGIN_CAPABILITY = "vision.image_qa"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 256


@dataclass
class PluginSettings:
    """Resolved plugin settings pulled from ``HomieConfig``."""

    enabled: bool = False
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    allow_download: bool = False
    cache_dir: Optional[str] = None
    system_prompt: str = (
        "You are Homie's local vision assistant. Describe images and answer "
        "questions about them concisely and factually. If you are unsure, say so."
    )
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: HomieConfig) -> "PluginSettings":
        raw = cfg_get(cfg, "plugins", PLUGIN_NAME, default={}) or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            model_id=str(raw.get("model_id") or DEFAULT_MODEL_ID),
            device=str(raw.get("device") or "auto"),
            dtype=str(raw.get("dtype") or "auto"),
            max_new_tokens=int(raw.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS),
            allow_download=bool(raw.get("allow_download", False)),
            cache_dir=raw.get("cache_dir"),
            system_prompt=str(raw.get("system_prompt") or cls.system_prompt),
            extra={k: v for k, v in raw.items() if k not in {
                "enabled", "model_id", "device", "dtype", "max_new_tokens",
                "allow_download", "cache_dir", "system_prompt",
            }},
        )


class Qwen3VLImageQAPlugin:
    """Local image Q&A plugin backed by Qwen3-VL-2B-Instruct.

    The model and processor are loaded lazily on the first request so that
    plugin activation stays cheap and Homie can boot quickly. Loading happens
    under a lock so concurrent callers do not race during the first inference.
    """

    def __init__(self, settings: PluginSettings) -> None:
        self.settings = settings
        self._model = None
        self._processor = None
        self._torch = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self, host: Optional[Any] = None) -> None:
        """Register the plugin with the Homie host (lazy model load).

        :param host: Optional Homie host/registry object. If it exposes a
            ``register_capability`` method, the plugin's ``image_qa`` entry
            point is registered under :data:`PLUGIN_CAPABILITY`.
        """
        if not self.settings.enabled:
            logger.info("%s disabled via config; skipping activation", PLUGIN_NAME)
            return

        if host is not None and hasattr(host, "register_capability"):
            host.register_capability(PLUGIN_CAPABILITY, self.image_qa)
            logger.info("Registered capability %s with host", PLUGIN_CAPABILITY)
        else:
            logger.debug("No host registry provided; plugin usable directly")

    def deactivate(self, host: Optional[Any] = None) -> None:
        """Unregister and release the model from memory."""
        if host is not None and hasattr(host, "unregister_capability"):
            try:
                host.unregister_capability(PLUGIN_CAPABILITY)
            except Exception:  # noqa: BLE001
                logger.debug("Host did not support unregister_capability cleanly")

        with self._lock:
            self._model = None
            self._processor = None
            self._loaded = False
            if self._torch is not None:
                try:
                    if self._device == "cuda" and self._torch.cuda.is_available():
                        self._torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
        logger.info("%s deactivated", PLUGIN_NAME)

    # ------------------------------------------------------------------
    # Capability entry point
    # ------------------------------------------------------------------
    def image_qa(
        self,
        image: Union[str, Path, List[Union[str, Path]]],
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Answer ``prompt`` about ``image`` using the local Qwen3-VL model.

        :param image: One or more local paths to image files (PNG, JPG, etc.).
            Remote URLs are rejected to keep the plugin offline-safe.
        :param prompt: Natural-language question or instruction about the image.
        :param max_new_tokens: Optional override for generation length.
        :returns: ``{"answer": str, "model": str, "image_count": int}``
        :raises RuntimeError: If the model fails to load (e.g. offline and
            weights not cached).
        :raises ValueError: If inputs are invalid or paths are not local files.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        image_paths = self._validate_images(image)
        self._ensure_loaded()

        messages = [
            {"role": "system", "content": self.settings.system_prompt},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": str(p)} for p in image_paths],
                    {"type": "text", "text": prompt.strip()},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pil_images = [self._open_image(p) for p in image_paths]
        inputs = self._processor(
            text=[text], images=pil_images, return_tensors="pt", padding=True
        ).to(self._device)

        generate_kwargs = {
            "max_new_tokens": int(max_new_tokens or self.settings.max_new_tokens),
            "do_sample": False,
        }

        with self._torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generate_kwargs)

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        answer = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return {
            "answer": answer,
            "model": self.settings.model_id,
            "image_count": len(image_paths),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _validate_images(
        self, image: Union[str, Path, List[Union[str, Path]]]
    ) -> List[Path]:
        if isinstance(image, (str, Path)):
            candidates = [image]
        elif isinstance(image, list):
            candidates = image
        else:
            raise ValueError("image must be a path or list of paths")

        resolved: List[Path] = []
        for item in candidates:
            s = str(item)
            if "://" in s and not s.startswith("file://"):
                raise ValueError(
                    f"Remote URLs are not allowed (offline-safe plugin): {s}"
                )
            path = Path(s[7:] if s.startswith("file://") else s).expanduser().resolve()
            if not path.exists() or not path.is_file():
                raise ValueError(f"Image path does not exist: {path}")
            resolved.append(path)
        if not resolved:
            raise ValueError("At least one image path is required")
        return resolved

    def _open_image(self, path: Path):
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "Pillow is required for image loading. Install with `pip install pillow`."
            ) from exc
        img = Image.open(path)
        return img.convert("RGB")

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load_model()
            self._loaded = True

    def _load_model(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoProcessor,
                Qwen3VLForConditionalGeneration,
            )
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "qwen3_vl_image_qa requires `transformers>=4.45` and `torch`. "
                "Install with `pip install transformers torch pillow safetensors`."
            ) from exc

        self._torch = torch
        self._device = self._resolve_device(torch)
        torch_dtype = self._resolve_dtype(torch)

        env_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        if not self.settings.allow_download:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            load_kwargs: Dict[str, Any] = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
            }
            if self.settings.cache_dir:
                load_kwargs["cache_dir"] = self.settings.cache_dir

            logger.info(
                "Loading %s (device=%s, dtype=%s, allow_download=%s)",
                self.settings.model_id,
                self._device,
                torch_dtype,
                self.settings.allow_download,
            )
            self._processor = AutoProcessor.from_pretrained(
                self.settings.model_id,
                cache_dir=self.settings.cache_dir,
            )
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.settings.model_id, **load_kwargs
            ).to(self._device)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load {self.settings.model_id}: {exc}. "
                "If the weights are not cached, set `allow_download: true` once "
                "to fetch them, then you can run fully offline."
            ) from exc
        finally:
            if env_offline is None and not self.settings.allow_download:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            elif env_offline is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = env_offline

    def _resolve_device(self, torch) -> str:
        requested = (self.settings.device or "auto").lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return requested

    def _resolve_dtype(self, torch):
        requested = (self.settings.dtype or "auto").lower()
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if requested in mapping:
            return mapping[requested]
        if self._device == "cuda":
            return torch.float16
        if self._device == "mps":
            return torch.float16
        return torch.float32


_INSTANCE: Optional[Qwen3VLImageQAPlugin] = None


def register(cfg: HomieConfig, host: Optional[Any] = None) -> Qwen3VLImageQAPlugin:
    """Entry point used by the Homie plugin loader.

    Returns the activated plugin instance so the host can keep a reference
    and call ``deactivate`` during shutdown.
    """
    global _INSTANCE
    settings = PluginSettings.from_config(cfg)
    _INSTANCE = Qwen3VLImageQAPlugin(settings)
    _INSTANCE.activate(host)
    return _INSTANCE


def activate(cfg: HomieConfig, host: Optional[Any] = None) -> Qwen3VLImageQAPlugin:
    """Alias for :func:`register` to match the activate/deactivate convention."""
    return register(cfg, host)


def deactivate(host: Optional[Any] = None) -> None:
    """Tear down the singleton plugin instance, if one is active."""
    global _INSTANCE
    if _INSTANCE is not None:
        _INSTANCE.deactivate(host)
        _INSTANCE = None


__all__ = [
    "PLUGIN_NAME",
    "PLUGIN_CAPABILITY",
    "PluginSettings",
    "Qwen3VLImageQAPlugin",
    "register",
    "activate",
    "deactivate",
]
