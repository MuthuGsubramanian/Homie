from __future__ import annotations

from pathlib import Path

from homie_core.config import HomieConfig, load_config
from homie_core.hardware.detector import detect_hardware
from homie_core.hardware.profiles import recommend_model
from homie_core.model.registry import ModelRegistry

import yaml


def run_init(auto: bool = False, config_path: str | None = None) -> HomieConfig:
    print("=" * 50)
    print("  Welcome to Homie AI Setup")
    print("=" * 50)

    # Step 1: Detect hardware
    print("\n[1/5] Detecting hardware...")
    hw = detect_hardware()
    print(f"  OS: {hw.os_name} {hw.os_version}")
    print(f"  CPU: {hw.cpu_cores} cores")
    print(f"  RAM: {hw.ram_gb} GB")
    if hw.gpus:
        for gpu in hw.gpus:
            print(f"  GPU: {gpu.name} ({gpu.vram_mb} MB VRAM)")
    else:
        print("  GPU: None detected")
    print(f"  Microphone: {'Yes' if hw.has_microphone else 'No'}")

    # Step 2: Recommend model
    print("\n[2/5] Selecting optimal model...")
    rec = recommend_model(hw.best_gpu_vram_gb)
    print(f"  Recommended: {rec['model']} ({rec['quant']}, {rec['format']})")
    print(f"  Backend: {rec['backend']}")

    # Step 3: Create config
    print("\n[3/5] Creating configuration...")
    cfg = HomieConfig()
    cfg.llm.backend = rec["format"]
    cfg.voice.enabled = hw.has_microphone
    if hw.has_microphone:
        cfg.voice.mode = "push_to_talk"

    # Step 4: Get user name
    if not auto:
        try:
            name = input("\n[4/5] What should I call you? ").strip()
            if name:
                cfg.user_name = name
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        print("\n[4/5] Skipping (auto mode)")

    # Step 5: Save config
    print("\n[5/5] Saving configuration...")
    storage_path = Path(cfg.storage.path)
    storage_path.mkdir(parents=True, exist_ok=True)

    config_file = Path("homie.config.yaml")
    config_data = cfg.model_dump()
    config_file.write_text(yaml.dump(config_data, default_flow_style=False))
    print(f"  Config saved to {config_file}")
    print(f"  Data directory: {storage_path}")

    # Initialize model registry
    registry = ModelRegistry(storage_path / cfg.storage.models_dir)
    registry.initialize()

    print("\n" + "=" * 50)
    print("  Setup complete! Run 'homie start' to begin.")
    print("=" * 50)

    return cfg
