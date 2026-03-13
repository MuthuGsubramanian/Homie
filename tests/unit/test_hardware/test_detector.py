from homie_core.hardware.detector import HardwareInfo, detect_hardware
from homie_core.hardware.profiles import recommend_model


def test_detect_hardware_returns_info():
    info = detect_hardware()
    assert isinstance(info, HardwareInfo)
    assert info.ram_gb > 0
    assert info.os_name in ("Windows", "Linux", "Darwin")


def test_recommend_model_16gb():
    rec = recommend_model(gpu_vram_gb=16.0)
    assert rec["quant"] == "Q4_K_M"
    assert rec["model"] == "Qwen3.5-35B-A3B"
    assert rec["context_length"] == 65536
    assert rec["repo_id"] != ""


def test_recommend_model_15gb():
    """RTX 4080 reports ~15.9GB, should still get the top-tier model."""
    rec = recommend_model(gpu_vram_gb=15.9)
    assert rec["model"] == "Qwen3.5-35B-A3B"


def test_recommend_model_8gb():
    rec = recommend_model(gpu_vram_gb=8.0)
    assert "8B" in rec["model"]


def test_recommend_model_no_gpu():
    rec = recommend_model(gpu_vram_gb=0)
    assert rec["format"] == "gguf"
    assert rec["backend"] == "cpu"
