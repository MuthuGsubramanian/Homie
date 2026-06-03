"""HOMIE Flux Image Generator Plugin.

Local text-to-image generation using ``black-forest-labs/FLUX.1-schnell``,
a distilled diffusion model that produces high-quality 1024x1024 images in
1-4 inference steps. Designed for Homie users with modest GPUs (8-16GB VRAM
via CPU offload) or Apple Silicon (MPS).

Integration notes:
  * Pure local inference via ``diffusers``; no network calls at generation time.
  * Model weights (~24GB) are NOT downloaded automatically. The user must
    opt-in by setting ``plugins.flux_image.allow_download: true`` in
    ``homie.config.yaml``, or by pre-caching the model with the Hugging Face CLI.
  * Honors the Homie convention of ``activate()`` / ``deactivate()`` lifecycle
    hooks plus a ``register()`` entrypoint for the plugin loader.
  * Generated images are written to ``~/.homie/generated_images`` by default.

Example (Python):
    >>> from homie.config import load_config
    >>> from homie.plugins.flux_image_generator import activate
    >>> gen = activate(load_config())
    >>> path = gen.generate("a watercolor fox reading a book", seed=42)
    >>> print(path)
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from homie.config import HomieConfig, cfg_get


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEFAULT_STEPS = 4  # schnell is distilled for 1-4 step inference
DEFAULT_GUIDANCE = 0.0  # schnell ignores classifier-free guidance
DEFAULT_OUTPUT_DIR = Path.home() / ".homie" / "generated_images"
DEFAULT_RESOLUTION = 1024


class FluxImageGenerator:
    """Lazy-loaded FLUX.1-schnell text-to-image pipeline.

    The underlying ``diffusers`` pipeline is only constructed on first call to
    :meth:`generate`, so plugin activation is cheap and safe at startup even on
    machines without a GPU or the optional dependencies installed.
    """

    def __init__(self, cfg: HomieConfig) -> None:
        self.cfg = cfg
        self.model_id: str = cfg_get(
            cfg, "plugins", "flux_image", "model_id", default=DEFAULT_MODEL_ID
        )
        self.steps: int = int(
            cfg_get(cfg, "plugins", "flux_image", "steps", default=DEFAULT_STEPS)
        )
        self.guidance: float = float(
            cfg_get(cfg, "plugins", "flux_image", "guidance", default=DEFAULT_GUIDANCE)
        )
        self.device: str = cfg_get(
            cfg, "plugins", "flux_image", "device", default="auto"
        )
        self.allow_download: bool = bool(
            cfg_get(cfg, "plugins", "flux_image", "allow_download", default=False)
        )
        self.enable_cpu_offload: bool = bool(
            cfg_get(cfg, "plugins", "flux_image", "cpu_offload", default=True)
        )
        out = cfg_get(
            cfg,
            "plugins",
            "flux_image",
            "output_dir",
            default=str(DEFAULT_OUTPUT_DIR),
        )
        self.output_dir: Path = Path(out).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline: Optional[Any] = None

    def _resolve_device(self) -> str:
        """Pick the best available device unless the user pinned one in config."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _is_model_cached(self) -> bool:
        """Return True if the model is already in the local HF cache."""
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return False
        try:
            cached = try_to_load_from_cache(self.model_id, "model_index.json")
            return bool(cached)
        except Exception:  # noqa: BLE001
            return False

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from diffusers import FluxPipeline
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "FLUX image generation requires optional deps. Install with:\n"
                "  pip install diffusers transformers accelerate torch sentencepiece"
            ) from exc

        if not self.allow_download and not self._is_model_cached():
            raise RuntimeError(
                f"Model '{self.model_id}' is not in the local Hugging Face cache "
                "and 'plugins.flux_image.allow_download' is False. Set it to true "
                "in homie.config.yaml to opt into the ~24GB download, or pre-cache "
                "the model with `huggingface-cli download`."
            )

        device = self._resolve_device()
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        LOGGER.info("Loading FLUX pipeline on %s (dtype=%s)", device, dtype)
        pipe = FluxPipeline.from_pretrained(self.model_id, torch_dtype=dtype)

        if device == "cuda" and self.enable_cpu_offload:
            # Keeps VRAM usage workable on 8-16GB GPUs.
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self._pipeline = pipe
        return pipe

    def generate(
        self,
        prompt: str,
        *,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        height: int = DEFAULT_RESOLUTION,
        width: int = DEFAULT_RESOLUTION,
        seed: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Generate a single image and return the saved file path.

        Args:
            prompt: Natural-language description of the image to create.
            steps: Override the configured number of inference steps (1-4 recommended).
            guidance: Override the configured guidance scale (schnell uses 0.0).
            height: Image height in pixels (must be a multiple of 16).
            width: Image width in pixels (must be a multiple of 16).
            seed: Optional RNG seed for reproducible output.
            filename: Optional output filename; default is a UTC timestamp.

        Returns:
            Absolute path to the saved PNG file.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if height % 16 or width % 16:
            raise ValueError("height and width must be multiples of 16")

        pipe = self._load_pipeline()
        import torch  # safe: _load_pipeline already validated availability

        generator = None
        if seed is not None:
            gen_device = "cpu" if self._resolve_device() == "mps" else self._resolve_device()
            generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        result = pipe(
            prompt=prompt,
            num_inference_steps=steps or self.steps,
            guidance_scale=guidance if guidance is not None else self.guidance,
            height=height,
            width=width,
            generator=generator,
        )
        image = result.images[0]

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        name = filename or f"flux_{ts}.png"
        out_path = self.output_dir / name
        image.save(out_path)
        LOGGER.info("Saved image %s", out_path)
        return out_path

    def unload(self) -> None:
        """Release the pipeline and free GPU memory if possible."""
        if self._pipeline is None:
            return
        self._pipeline = None
        try:
            import torch
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


_INSTANCE: Optional[FluxImageGenerator] = None


def activate(cfg: HomieConfig) -> FluxImageGenerator:
    """Instantiate the plugin singleton (no model load yet)."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = FluxImageGenerator(cfg)
        LOGGER.info(
            "Flux image generator activated (model=%s, output_dir=%s)",
            _INSTANCE.model_id,
            _INSTANCE.output_dir,
        )
    return _INSTANCE


def deactivate() -> None:
    """Tear down the plugin singleton and free resources."""
    global _INSTANCE
    if _INSTANCE is not None:
        _INSTANCE.unload()
        _INSTANCE = None
        LOGGER.info("Flux image generator deactivated")


def register(cfg: HomieConfig) -> Dict[str, Any]:
    """Standard Homie plugin registration hook.

    Returns a descriptor the plugin loader can use to expose the
    ``generate_image`` command throughout Homie (CLI, dashboard, orchestrator).
    """
    instance = activate(cfg)
    return {
        "name": "flux_image_generator",
        "version": "0.1.0",
        "description": "Local text-to-image generation via FLUX.1-schnell",
        "commands": {
            "generate_image": instance.generate,
        },
        "instance": instance,
        "teardown": deactivate,
    }


__all__ = [
    "FluxImageGenerator",
    "activate",
    "deactivate",
    "register",
    "DEFAULT_MODEL_ID",
]
