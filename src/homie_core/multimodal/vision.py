from __future__ import annotations

import base64
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import io as _io

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import easyocr as _easyocr

    _HAS_EASYOCR = True
except ImportError:
    _HAS_EASYOCR = False

try:
    import pytesseract as _pytesseract

    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False


class VisionEngine:
    """Understands images and screen content."""

    def __init__(self, inference_fn: Optional[Callable[..., str]] = None):
        self._infer = inference_fn
        self._ocr_reader: object | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe_image(self, image_path: str) -> str:
        """Generate a natural-language description of an image.

        Uses the LLM inference function when available; falls back to OCR text
        extraction so the caller always gets *something* useful.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self._infer is not None:
            try:
                b64 = self._encode_image_b64(image_path)
                prompt = (
                    "Describe this image in detail. Include the main subject, "
                    "colours, layout, and any text visible."
                )
                return self._infer(prompt, image_b64=b64)
            except Exception:
                logger.debug("LLM image description failed, falling back to OCR", exc_info=True)

        # Fallback — extract whatever text is in the image
        text = self.extract_text_from_image(image_path)
        if text and text.strip():
            return f"[OCR fallback] Text found in image: {text.strip()}"
        return "[No description available — vision model and OCR unavailable]"

    def extract_text_from_image(self, image_path: str) -> str:
        """OCR text extraction from an image file.

        Tries easyocr first, then pytesseract.  Returns empty string when
        neither library is installed.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        text = self._ocr_easyocr(image_path)
        if text is not None:
            return text

        text = self._ocr_tesseract(image_path)
        if text is not None:
            return text

        logger.debug("No OCR backend available")
        return ""

    def analyze_screenshot(self, screenshot_path: str) -> dict:
        """Analyze a screenshot: identify UI elements, text, and layout.

        Returns a dict with keys ``description``, ``text``, ``ui_elements``.
        """
        if not os.path.isfile(screenshot_path):
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        result: dict = {
            "description": "",
            "text": "",
            "ui_elements": [],
        }

        # Text extraction
        result["text"] = self.extract_text_from_image(screenshot_path)

        # LLM-based analysis
        if self._infer is not None:
            try:
                b64 = self._encode_image_b64(screenshot_path)
                prompt = (
                    "Analyze this screenshot. List the UI elements you see "
                    "(buttons, text fields, menus, etc.), describe the layout, "
                    "and summarize what application or webpage is shown."
                )
                analysis = self._infer(prompt, image_b64=b64)
                result["description"] = analysis
            except Exception:
                logger.debug("LLM screenshot analysis failed", exc_info=True)

        # Basic image metadata via PIL
        if _HAS_PIL:
            try:
                img = Image.open(screenshot_path)
                result["resolution"] = f"{img.width}x{img.height}"
                result["mode"] = img.mode
            except Exception:
                logger.debug("PIL metadata extraction failed", exc_info=True)

        return result

    def compare_images(self, path_a: str, path_b: str) -> dict:
        """Compare two images for differences.

        Returns a dict with ``same_size``, ``similarity`` (0-1 float when PIL
        is available), and ``description`` (when an LLM is available).
        """
        for p in (path_a, path_b):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Image not found: {p}")

        result: dict = {"same_size": False, "similarity": None, "description": ""}

        if _HAS_PIL:
            try:
                img_a = Image.open(path_a)
                img_b = Image.open(path_b)
                result["same_size"] = img_a.size == img_b.size
                result["size_a"] = f"{img_a.width}x{img_a.height}"
                result["size_b"] = f"{img_b.width}x{img_b.height}"

                # Simple pixel-level similarity when same size
                if result["same_size"]:
                    result["similarity"] = self._pixel_similarity(img_a, img_b)
            except Exception:
                logger.debug("PIL image comparison failed", exc_info=True)

        if self._infer is not None:
            try:
                b64_a = self._encode_image_b64(path_a)
                b64_b = self._encode_image_b64(path_b)
                prompt = "Compare these two images. Describe the differences."
                result["description"] = self._infer(prompt, image_b64=b64_a, image_b64_2=b64_b)
            except Exception:
                logger.debug("LLM image comparison failed", exc_info=True)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image_b64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _ocr_easyocr(image_path: str) -> str | None:
        if not _HAS_EASYOCR:
            return None
        try:
            reader = _easyocr.Reader(["en"], gpu=False, verbose=False)
            results = reader.readtext(image_path, detail=0)
            return "\n".join(results)
        except Exception:
            logger.debug("easyocr failed", exc_info=True)
            return None

    @staticmethod
    def _ocr_tesseract(image_path: str) -> str | None:
        if not (_HAS_TESSERACT and _HAS_PIL):
            return None
        try:
            img = Image.open(image_path)
            return _pytesseract.image_to_string(img)
        except Exception:
            logger.debug("pytesseract failed", exc_info=True)
            return None

    @staticmethod
    def _pixel_similarity(img_a: "Image.Image", img_b: "Image.Image") -> float:
        """Return 0.0–1.0 similarity based on pixel-level comparison."""
        a_bytes = img_a.convert("RGB").tobytes()
        b_bytes = img_b.convert("RGB").tobytes()
        if len(a_bytes) != len(b_bytes):
            return 0.0
        matching = sum(1 for x, y in zip(a_bytes, b_bytes) if x == y)
        return matching / len(a_bytes)
