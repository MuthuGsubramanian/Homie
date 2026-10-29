"""
Local on-device image-editing capability probe for HOMIE.

Inspired by MoI-Bench (arXiv:2610.37599), a benchmark that evaluates
on-device image generation/editing models against a taxonomy of
operations (e.g. removal, replacement, resize, style/color adjustment)
and visual target categories (portraits, objects, text, scenery, etc.).

HOMIE does not bundle a diffusion/image-editing model today. This plugin
does NOT download or run one either -- it runs a small, dependency-light
set of classic Pillow-based approximations for a subset of the MoI-Bench
operation taxonomy (crop/removal, box-fill/replacement, resize, rotate,
flip, brightness/contrast/color adjust, blur, sharpen) against a user
supplied image, and reports timing + success per operation.

The resulting report answers a narrow, practical question: does this
machine have enough local headroom (CPU time, memory) to make bundling a
real on-device image-editing model worthwhile, and which operation
categories would benefit most? It is a decision-support probe, not an
image model. All work happens on local files; there are no network calls
anywhere in this module.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
except ImportError as exc:  # pragma: no cover - surfaced at activate() time
    Image = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None


DEFAULT_WORKSPACE = Path.home() / ".homie" / "image_bench"

# A deliberately small, local-only subset of MoI-Bench's 16 operations.
# Each maps to a classic (non-model) Pillow approximation so the plugin
# stays safe and dependency-light while still exercising comparable
# image I/O and compute cost categories.
SUPPORTED_OPERATIONS = (
    "resize",
    "crop_removal",
    "box_fill_replacement",
    "rotate",
    "flip_horizontal",
    "brightness_adjust",
    "contrast_adjust",
    "color_adjust",
    "blur_smooth",
    "sharpen",
)


@dataclass
class OperationResult:
    """Outcome of a single benchmark operation."""

    operation: str
    ok: bool
    elapsed_ms: float
    output_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Aggregate result of running a batch of operations on one image."""

    source_image: str
    target_category: str
    results: List[OperationResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.ok) / len(self.results)

    @property
    def total_elapsed_ms(self) -> float:
        return sum(r.elapsed_ms for r in self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_image": self.source_image,
            "target_category": self.target_category,
            "success_rate": self.success_rate,
            "total_elapsed_ms": self.total_elapsed_ms,
            "results": [asdict(r) for r in self.results],
        }


class ImageEditBenchPlugin:
    """Runs local, classic-CV approximations of MoI-Bench operations.

    This plugin exists to generate a local capability signal (success
    rate + latency per operation) that HOMIE's maintainers can use to
    decide which real on-device image-editing model, if any, would be
    worth bundling next. It never reaches the network and never loads a
    deep-learning model; everything here is Pillow.
    """

    name = "image_edit_bench"

    def __init__(self, workspace: Optional[Path] = None) -> None:
        self.workspace = Path(workspace) if workspace else DEFAULT_WORKSPACE
        self._active = False
        self._operations: Dict[str, Callable[[Any], Any]] = {
            "resize": self._op_resize,
            "crop_removal": self._op_crop_removal,
            "box_fill_replacement": self._op_box_fill_replacement,
            "rotate": self._op_rotate,
            "flip_horizontal": self._op_flip_horizontal,
            "brightness_adjust": self._op_brightness_adjust,
            "contrast_adjust": self._op_contrast_adjust,
            "color_adjust": self._op_color_adjust,
            "blur_smooth": self._op_blur_smooth,
            "sharpen": self._op_sharpen,
        }

    # -- lifecycle -----------------------------------------------------

    def activate(self) -> None:
        """Prepare the local workspace. Raises if Pillow is unavailable."""
        if Image is None:
            raise RuntimeError(
                "ImageEditBenchPlugin requires Pillow ('pip install Pillow'): "
                f"{_PIL_IMPORT_ERROR}"
            )
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._active = True
        logging.info("ImageEditBenchPlugin activated, workspace=%s", self.workspace)

    def deactivate(self) -> None:
        self._active = False
        logging.info("ImageEditBenchPlugin deactivated")

    # -- public API ------------------------------------------------------

    def list_supported_operations(self) -> List[str]:
        return list(SUPPORTED_OPERATIONS)

    def run_benchmark(
        self,
        image_path: str | Path,
        operations: Optional[List[str]] = None,
        target_category: str = "generic",
    ) -> BenchmarkReport:
        """Run the requested (or all supported) operations on one image.

        Args:
            image_path: Local path to a source image.
            operations: Subset of SUPPORTED_OPERATIONS to run; defaults to all.
            target_category: Free-text label for the MoI-Bench-style visual
                target category this image represents (e.g. "portrait",
                "text", "scenery") -- used only for reporting, never inferred
                automatically.

        Returns:
            A BenchmarkReport with per-operation timing and success state.
        """
        if not self._active:
            raise RuntimeError("Plugin not activated. Call activate() first.")

        src = Path(image_path)
        if not src.is_file():
            raise FileNotFoundError(f"No such image: {src}")

        chosen = operations or list(SUPPORTED_OPERATIONS)
        unknown = [op for op in chosen if op not in self._operations]
        if unknown:
            raise ValueError(f"Unsupported operation(s): {unknown}")

        report = BenchmarkReport(source_image=str(src), target_category=target_category)

        for op_name in chosen:
            start = time.perf_counter()
            try:
                with Image.open(src) as img:
                    img = img.convert("RGB")
                    edited = self._operations[op_name](img)
                    out_path = self.workspace / f"{src.stem}_{op_name}.png"
                    edited.save(out_path, format="PNG")
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                report.results.append(
                    OperationResult(
                        operation=op_name,
                        ok=True,
                        elapsed_ms=round(elapsed_ms, 2),
                        output_path=str(out_path),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - we want per-op isolation
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                report.results.append(
                    OperationResult(
                        operation=op_name,
                        ok=False,
                        elapsed_ms=round(elapsed_ms, 2),
                        error=str(exc),
                    )
                )
                logging.warning("Operation %s failed: %s", op_name, exc)

        self._write_report(report)
        return report

    # -- reporting -------------------------------------------------------

    def _write_report(self, report: BenchmarkReport) -> Path:
        report_path = self.workspace / f"{Path(report.source_image).stem}_report.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return report_path

    # -- operation approximations (classic CV, no models, no network) ---

    @staticmethod
    def _op_resize(img: "Image.Image") -> "Image.Image":
        w, h = img.size
        return img.resize((max(1, w // 2), max(1, h // 2)))

    @staticmethod
    def _op_crop_removal(img: "Image.Image") -> "Image.Image":
        w, h = img.size
        # Approximates "removal" by cropping out a central region.
        box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
        return img.crop(box)

    @staticmethod
    def _op_box_fill_replacement(img: "Image.Image") -> "Image.Image":
        # Approximates "replacement" by flat-filling a region, the
        # classic placeholder for a generative inpaint/replace step.
        out = img.copy()
        w, h = out.size
        draw = ImageDraw.Draw(out)
        box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
        draw.rectangle(box, fill=(128, 128, 128))
        return out

    @staticmethod
    def _op_rotate(img: "Image.Image") -> "Image.Image":
        return img.rotate(15, expand=True)

    @staticmethod
    def _op_flip_horizontal(img: "Image.Image") -> "Image.Image":
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    @staticmethod
    def _op_brightness_adjust(img: "Image.Image") -> "Image.Image":
        return ImageEnhance.Brightness(img).enhance(1.3)

    @staticmethod
    def _op_contrast_adjust(img: "Image.Image") -> "Image.Image":
        return ImageEnhance.Contrast(img).enhance(1.3)

    @staticmethod
    def _op_color_adjust(img: "Image.Image") -> "Image.Image":
        return ImageEnhance.Color(img).enhance(1.5)

    @staticmethod
    def _op_blur_smooth(img: "Image.Image") -> "Image.Image":
        return img.filter(ImageFilter.GaussianBlur(radius=3))

    @staticmethod
    def _op_sharpen(img: "Image.Image") -> "Image.Image":
        return img.filter(ImageFilter.SHARPEN)


def register(workspace: Optional[Path] = None) -> ImageEditBenchPlugin:
    """Create, activate, and return an ImageEditBenchPlugin instance."""
    plugin = ImageEditBenchPlugin(workspace=workspace)
    plugin.activate()
    return plugin


__all__ = [
    "ImageEditBenchPlugin",
    "BenchmarkReport",
    "OperationResult",
    "SUPPORTED_OPERATIONS",
    "register",
]
