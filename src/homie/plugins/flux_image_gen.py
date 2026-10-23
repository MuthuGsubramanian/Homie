"""FLUX.1-schnell local image generation plugin for HOMIE.

Generates images entirely on-device using the Apache-2.0 licensed
FLUX.1-schnell model (black-forest-labs/FLUX.1-schnell). schnell is a
timestep-distilled model that produces strong results in only 4 inference
steps with guidance disabled, which makes it the most practical FLUX
variant for local, offline image generation.

Design notes (matches HOMIE conventions):
- Offline-first: the model is loaded with ``local_files_only=True`` unless
  the user explicitly opts in to downloads via config
  (``plugins.flux_image.allow_download: true``). No network calls happen
  otherwise.
- Lazy loading: importing this module is cheap. ``torch`` and ``diffusers``
  are only imported, and the ~24 GB of weights only touched, on the first
  ``generate()`` call.
- Lifecycle: ``activate()`` validates dependencies and config,
  ``deactivate()`` releases the pipeline and frees VRAM.

Config (homie.config.yaml)::

    plugins:
      flux_image:
        enabled: true
        model_id: black-forest-labs/FLUX.1-schnell
        allow_download: false      # opt-in network access
        steps: 4                   # schnell is distilled for 4 steps
        width: 1024
        height: 1024
        device: auto               # auto|cuda|mps|cpu
        cpu_offload: true          # lets 12-16GB VRAM cards run FLUX
        output_dir: ~/.homie/images
        open_after_generate: false

Dependencies (installed explicitly by the user; not pulled in by HOMIE core)::

    pip install torch diffusers transformers accelerate sentencepiece protobuf
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from homie.config import HomieConfig, cfg_get


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEFAULT_STEPS = 4
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
# schnell is guidance-distilled: CFG must stay at 0.0 or quality degrades.
SCHNELL_GUIDANCE_SCALE = 0.0
DEFAULT_OUTPUT_DIR = Path.home() / ".homie" / "images"


class FluxPluginError(Exception):
    """Raised when the FLUX plugin cannot load dependencies or generate."""


@dataclass
class FluxSettings:
    """Resolved plugin settings from homie.config.yaml."""

    model_id: str = DEFAULT_MODEL_ID
    allow_download: bool = False
    steps: int = DEFAULT_STEPS
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    device: str = "auto"
    cpu_offload: bool = True
    output_dir: Path = DEFAULT_OUTPUT_DIR
    open_after_generate: bool = False

    @classmethod
    def from_config(cls, cfg: Optional[HomieConfig]) -> "FluxSettings":
        if cfg is None:
            return cls()

        def get(key: str, default: Any) -> Any:
            return cfg_get(cfg, "plugins", "flux_image", key, default=default)

        return cls(
            model_id=str(get("model_id", DEFAULT_MODEL_ID)),
            allow_download=bool(get("allow_download", False)),
            steps=int(get("steps", DEFAULT_STEPS)),
            width=int(get("width", DEFAULT_WIDTH)),
            height=int(get("height", DEFAULT_HEIGHT)),
            device=str(get("device", "auto")),
            cpu_offload=bool(get("cpu_offload", True)),
            output_dir=Path(str(get("output_dir", str(DEFAULT_OUTPUT_DIR)))).expanduser(),
            open_after_generate=bool(get("open_after_generate", False)),
        )


def _slugify(text: str, max_len: int = 48) -> str:
    """Turn a prompt into a filesystem-safe filename fragment."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "image"


def _open_file(path: Path) -> None:
    """Best-effort open in the OS default image viewer (opt-in via config)."""
    system = platform.system().lower()
    try:
        if system == "windows":
            import os

            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:  # noqa: BLE001
        logging.debug("Could not open %s in a viewer", path)


class FluxImagePlugin:
    """Local text-to-image generation via FLUX.1-schnell.

    Heavy dependencies (torch, diffusers) and the model weights load lazily
    on first use so HOMIE startup stays fast. All generation happens on the
    local machine; the only possible network access is the initial model
    download, and that is disabled unless explicitly enabled in config.
    """

    name = "flux_image"

    def __init__(self, cfg: Optional[HomieConfig] = None) -> None:
        self.cfg = cfg
        self.settings = FluxSettings.from_config(cfg)
        self._pipeline: Any = None
        self._active = False

    # -- lifecycle ----------------------------------------------------

    def activate(self) -> None:
        """Validate dependencies and prepare the output directory.

        Does NOT load model weights -- that happens lazily on the first
        ``generate()`` call so activation stays instant.
        """
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise FluxPluginError(
                "FLUX plugin requires extra packages. Install with:\n"
                "  pip install torch diffusers transformers accelerate "
                "sentencepiece protobuf"
            ) from exc

        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self._active = True
        logging.info(
            "flux_image activated | model=%s mode=%s steps=%d size=%dx%d",
            self.settings.model_id,
            "download-opt-in" if self.settings.allow_download else "offline-only",
            self.settings.steps,
            self.settings.width,
            self.settings.height,
        )

    def deactivate(self) -> None:
        """Release the pipeline and free GPU/CPU memory."""
        if self._pipeline is not None:
            self._pipeline = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        self._active = False
        logging.info("flux_image deactivated")

    # -- internals ----------------------------------------------------

    def _pick_device(self) -> str:
        import torch

        if self.settings.device != "auto":
            return self.settings.device
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from diffusers import FluxPipeline

        device = self._pick_device()
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        logging.info(
            "Loading %s (device=%s, dtype=%s). First load can take minutes.",
            self.settings.model_id,
            device,
            dtype,
        )
        try:
            pipe = FluxPipeline.from_pretrained(
                self.settings.model_id,
                torch_dtype=dtype,
                local_files_only=not self.settings.allow_download,
            )
        except Exception as exc:  # noqa: BLE001
            if not self.settings.allow_download:
                raise FluxPluginError(
                    f"Model '{self.settings.model_id}' is not in the local "
                    "HuggingFace cache and downloads are disabled. Either "
                    "pre-download it on a machine you trust, or set "
                    "plugins.flux_image.allow_download: true in "
                    "homie.config.yaml to opt in to a one-time ~24GB download."
                ) from exc
            raise FluxPluginError(f"Failed to load FLUX pipeline: {exc}") from exc

        if device == "cuda" and self.settings.cpu_offload:
            # Streams weights through VRAM; lets 12-16GB cards run FLUX.
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        self._pipeline = pipe
        return pipe

    # -- public API ---------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        steps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Generate an image from ``prompt`` and return the saved PNG path.

        Args:
            prompt: Text description of the desired image.
            steps: Inference steps; defaults to config (4 for schnell).
            width: Image width in pixels (multiple of 16 recommended).
            height: Image height in pixels (multiple of 16 recommended).
            seed: Optional seed for reproducible output.
            output_path: Explicit save path; auto-named from the prompt
                and a timestamp when omitted.

        Raises:
            FluxPluginError: if the plugin is inactive, the prompt is
                empty, or the model cannot be loaded.
        """
        if not self._active:
            raise FluxPluginError("Plugin not active. Call activate() first.")
        if not prompt or not prompt.strip():
            raise FluxPluginError("Prompt must be a non-empty string.")

        import torch

        pipe = self._load_pipeline()
        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(int(seed))

        started = time.time()
        result = pipe(
            prompt=prompt,
            num_inference_steps=int(steps or self.settings.steps),
            guidance_scale=SCHNELL_GUIDANCE_SCALE,
            width=int(width or self.settings.width),
            height=int(height or self.settings.height),
            max_sequence_length=256,
            generator=generator,
        )
        image = result.images[0]

        if output_path is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = self.settings.output_dir / f"{stamp}-{_slugify(prompt)}.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        logging.info("Generated %s in %.1fs", output_path, time.time() - started)

        if self.settings.open_after_generate:
            _open_file(output_path)
        return output_path


def register(cfg: Optional[HomieConfig] = None) -> FluxImagePlugin:
    """Entry point used by the HOMIE plugin loader."""
    return FluxImagePlugin(cfg)


__all__ = ["FluxImagePlugin", "FluxSettings", "FluxPluginError", "register"]


if __name__ == "__main__":
    # Manual smoke test: python flux_image_gen.py "a red fox in snow"
    import sys

    logging.basicConfig(level=logging.INFO)
    plugin = register()
    plugin.activate()
    out = plugin.generate(
        sys.argv[1] if len(sys.argv) > 1 else "a cozy cabin at dusk, watercolor"
    )
    print(f"Saved: {out}")
    plugin.deactivate()
