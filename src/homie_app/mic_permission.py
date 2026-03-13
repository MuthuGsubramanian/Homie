from __future__ import annotations
import logging
import os
import sys

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
except ImportError:
    sd = None


def _has_microphone() -> bool:
    if sd is None:
        return False
    try:
        devices = sd.query_devices()
        return any(d.get("max_input_channels", 0) > 0 for d in devices)
    except Exception:
        return False


def _trigger_os_prompt() -> bool:
    """Briefly open a mic stream to trigger Windows permission dialog."""
    if sd is None:
        return False
    try:
        stream = sd.InputStream(channels=1, samplerate=16000, blocksize=512)
        stream.start()
        stream.stop()
        stream.close()
        return True
    except Exception:
        return False


def _open_settings() -> None:
    """Open Windows Privacy > Microphone settings."""
    if sys.platform == "win32":
        os.startfile("ms-settings:privacy-microphone")


def request_microphone_access(interactive: bool = True, max_retries: int = 3) -> bool:
    """Attempt to get microphone access through progressive steps.

    1. Check if mic is already available
    2. Try triggering OS permission prompt
    3. Open Windows Settings and wait for user

    Returns True if mic access was obtained.
    """
    # Step 1: Check directly
    if _has_microphone():
        return True

    # Step 2: Trigger OS prompt
    _trigger_os_prompt()
    if _has_microphone():
        return True

    if not interactive:
        return False

    # Step 3: Open Settings with guidance
    for attempt in range(max_retries):
        print("\n  Microphone access is needed for voice features.")
        print("  Opening Windows Settings > Privacy > Microphone...")
        _open_settings()
        input(f"  Enable microphone access, then press Enter to retry ({attempt + 1}/{max_retries}): ")
        if _has_microphone():
            return True

    print("  No microphone detected. Voice will be disabled.")
    print("  You can enable it later in Homie settings.")
    return False
