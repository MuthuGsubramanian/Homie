#!/usr/bin/env python3
"""Build Homie AI .dmg for macOS."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INSTALLER = ROOT / "installer"
MACOS = INSTALLER / "macos"
DIST = ROOT / "dist"


def get_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


def main() -> None:
    version = get_version()
    print(f"Building Homie AI .dmg v{version}")

    skip_freeze = "--skip-freeze" in sys.argv
    if skip_freeze:
        app_path = DIST / "Homie AI.app"
    else:
        print("\n=== Stage 1: PyInstaller freeze ===")
        if not shutil.which("pyinstaller"):
            print("ERROR: pyinstaller not found")
            sys.exit(1)
        spec = MACOS / "homie-macos.spec"
        run(["pyinstaller", str(spec), "--noconfirm", "--distpath", str(DIST)])
        app_path = DIST / "Homie AI.app"

    if not app_path.exists():
        print(f"ERROR: {app_path} not found")
        sys.exit(1)

    resources = app_path / "Contents" / "Resources"
    resources.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MACOS / "uninstall.sh", resources / "uninstall.sh")
    shutil.copy2(MACOS / "com.heyhomie.daemon.plist", resources / "com.heyhomie.daemon.plist")

    print("\n=== Stage 2: Create .dmg ===")
    dmg_name = f"HomieAI-{version}.dmg"
    dmg_path = DIST / dmg_name
    if dmg_path.exists():
        dmg_path.unlink()

    if shutil.which("create-dmg"):
        run([
            "create-dmg",
            "--volname", "Homie AI",
            "--window-size", "600", "400",
            "--app-drop-link", "400", "200",
            "--icon", "Homie AI.app", "200", "200",
            str(dmg_path), str(app_path),
        ])
    else:
        run([
            "hdiutil", "create",
            "-volname", "Homie AI",
            "-srcfolder", str(app_path),
            "-ov", "-format", "UDZO",
            str(dmg_path),
        ])

    if not dmg_path.exists():
        print(f"ERROR: .dmg not created at {dmg_path}")
        sys.exit(1)

    size_mb = dmg_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 50}")
    print(f"  SUCCESS: {dmg_name} ({size_mb:.1f} MB)")
    print(f"  NOTE: This .dmg is unsigned. Users must right-click > Open")
    print(f"        to bypass Gatekeeper on first launch.")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
