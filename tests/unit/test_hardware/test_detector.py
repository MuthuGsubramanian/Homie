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
    assert "72B" in rec["model"] or "70B" in rec["model"]


def test_recommend_model_8gb():
    rec = recommend_model(gpu_vram_gb=8.0)
    assert "8B" in rec["model"]


def test_recommend_model_no_gpu():
    rec = recommend_model(gpu_vram_gb=0)
    assert rec["format"] == "gguf"
    assert rec["backend"] == "cpu"
