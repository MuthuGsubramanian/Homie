from __future__ import annotations

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from homie_core.multimodal.vision import VisionEngine


@pytest.fixture
def dummy_image(tmp_path):
    """Create a tiny valid file to act as an image path."""
    p = tmp_path / "test.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return str(p)


@pytest.fixture
def two_images(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"\x89PNG" + b"\x00" * 32)
    b.write_bytes(b"\x89PNG" + b"\x01" * 32)
    return str(a), str(b)


class TestVisionEngine:
    def test_init_no_inference(self):
        engine = VisionEngine()
        assert engine._infer is None

    def test_init_with_inference(self):
        fn = lambda prompt, **kw: "described"
        engine = VisionEngine(inference_fn=fn)
        assert engine._infer is fn

    def test_describe_image_file_not_found(self):
        engine = VisionEngine()
        with pytest.raises(FileNotFoundError):
            engine.describe_image("/nonexistent/image.png")

    def test_describe_image_with_inference_fn(self, dummy_image):
        fn = MagicMock(return_value="A photo of a cat.")
        engine = VisionEngine(inference_fn=fn)
        result = engine.describe_image(dummy_image)
        assert result == "A photo of a cat."
        fn.assert_called_once()

    def test_describe_image_inference_failure_falls_back_to_ocr(self, dummy_image):
        """When inference raises, engine falls back to OCR (which also fails here)."""
        fn = MagicMock(side_effect=RuntimeError("model down"))
        engine = VisionEngine(inference_fn=fn)
        # Both OCR backends are unavailable in test env so we get the fallback msg
        result = engine.describe_image(dummy_image)
        assert "OCR fallback" in result or "No description available" in result

    def test_describe_image_no_inference_no_ocr(self, dummy_image):
        """Without inference or OCR, a fallback message is returned."""
        engine = VisionEngine()
        with patch("homie_core.multimodal.vision._HAS_EASYOCR", False), \
             patch("homie_core.multimodal.vision._HAS_TESSERACT", False):
            result = engine.describe_image(dummy_image)
            assert "No description available" in result

    def test_extract_text_no_ocr_backends(self, dummy_image):
        engine = VisionEngine()
        with patch("homie_core.multimodal.vision._HAS_EASYOCR", False), \
             patch("homie_core.multimodal.vision._HAS_TESSERACT", False):
            assert engine.extract_text_from_image(dummy_image) == ""

    def test_extract_text_file_not_found(self):
        engine = VisionEngine()
        with pytest.raises(FileNotFoundError):
            engine.extract_text_from_image("/no/such/file.png")

    def test_analyze_screenshot_returns_dict(self, dummy_image):
        engine = VisionEngine()
        result = engine.analyze_screenshot(dummy_image)
        assert isinstance(result, dict)
        assert "text" in result
        assert "description" in result
        assert "ui_elements" in result

    def test_analyze_screenshot_with_inference(self, dummy_image):
        fn = MagicMock(return_value="Browser with Google open")
        engine = VisionEngine(inference_fn=fn)
        result = engine.analyze_screenshot(dummy_image)
        assert result["description"] == "Browser with Google open"

    def test_compare_images_missing_file(self, dummy_image):
        engine = VisionEngine()
        with pytest.raises(FileNotFoundError):
            engine.compare_images(dummy_image, "/does/not/exist.png")

    def test_compare_images_returns_dict(self, two_images):
        engine = VisionEngine()
        result = engine.compare_images(*two_images)
        assert isinstance(result, dict)
        assert "same_size" in result

    def test_encode_image_b64(self, dummy_image):
        b64 = VisionEngine._encode_image_b64(dummy_image)
        assert isinstance(b64, str)
        assert len(b64) > 0
