# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Homie AI on macOS — .app bundle."""
from pathlib import Path
import os
import tomllib

block_cipher = None
ROOT = Path(SPECPATH).parent.parent
if not (ROOT / "src" / "homie_app" / "cli.py").exists():
    ROOT = Path(os.getcwd())

# Read version dynamically
with open(ROOT / "pyproject.toml", "rb") as f:
    _VERSION = tomllib.load(f)["project"]["version"]

a_cli = Analysis(
    [str(ROOT / "src" / "homie_app" / "cli.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[(str(ROOT / "homie.config.yaml"), ".")],
    hiddenimports=[
        "pydantic", "pydantic_core", "pydantic.deprecated",
        "pydantic.deprecated.decorator",
        "cryptography", "cryptography.hazmat", "cryptography.hazmat.primitives",
        "cffi", "_cffi_backend",
        "keyring", "keyring.backends",
        "feedparser",
        "prompt_toolkit", "prompt_toolkit.shortcuts", "prompt_toolkit.history",
        "psutil", "requests", "yaml",
        "homie_app", "homie_app.cli", "homie_app.daemon",
        "homie_app.console", "homie_app.console.console",
        "homie_app.console.router", "homie_app.console.commands",
        "homie_app.tray", "homie_app.tray.app",
        "homie_app.wizard",
        "homie_app.service",
        "homie_core", "homie_core.brain",
        "homie_core.brain.engine", "homie_core.brain.tool_registry",
        "homie_core.config",
        "homie_core.memory", "homie_core.memory.working",
        "homie_core.memory.semantic", "homie_core.memory.episodic",
        "homie_core.inference", "homie_core.inference.router",
        "homie_core.inference.qubrid",
        "homie_core.network", "homie_core.network.protocol",
        "homie_core.network.discovery", "homie_core.network.server",
        "homie_core.email", "homie_core.email.gmail_provider",
        "homie_core.context",
        "homie_core.hardware", "homie_core.hardware.detector",
        "homie_core.model", "homie_core.model.downloader",
        "homie_core.neural", "homie_core.neural.model_manager",
        "homie_core.notifications",
        "homie_core.voice",
        "homie_core.storage", "homie_core.storage.database",
        "homie_core.storage.vectors",
        "homie_core.rag",
        "homie_core.plugins",
        "zeroconf", "websockets",
        "chromadb", "chromadb.config",
        "uvicorn", "fastapi",
        "apscheduler", "apscheduler.schedulers.background",
        "numpy",
        "google.auth", "google.oauth2.credentials",
        "googleapiclient",
    ],
    excludes=[
        "matplotlib", "scipy", "notebook", "jupyterlab",
        "pytest", "sphinx", "setuptools", "pip", "wheel",
        "tkinter", "_tkinter", "IPython", "ipykernel",
    ],
    noarchive=False,
)

a_daemon = Analysis(
    [str(ROOT / "src" / "homie_app" / "daemon.py")],
    pathex=[str(ROOT / "src")],
    hiddenimports=a_cli.hiddenimports,
    excludes=a_cli.excludes,
    noarchive=False,
)

pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)
pyz_daemon = PYZ(a_daemon.pure, a_daemon.zipped_data, cipher=block_cipher)

exe_cli = EXE(
    pyz_cli, a_cli.scripts, [],
    exclude_binaries=True, name="homie",
    debug=False, strip=False, upx=True, console=True,
)
exe_daemon = EXE(
    pyz_daemon, a_daemon.scripts, [],
    exclude_binaries=True, name="homie-daemon",
    debug=False, strip=False, upx=True, console=False,
)

coll = COLLECT(
    exe_cli, a_cli.binaries, a_cli.zipfiles, a_cli.datas,
    exe_daemon, a_daemon.binaries, a_daemon.zipfiles, a_daemon.datas,
    strip=False, upx=True, name="homie",
)

app = BUNDLE(
    coll,
    name="Homie AI.app",
    bundle_identifier="com.heyhomie.app",
    info_plist={
        "CFBundleName": "Homie AI",
        "CFBundleDisplayName": "Homie AI",
        "CFBundleVersion": _VERSION,
        "CFBundleShortVersionString": _VERSION,
        "LSMinimumSystemVersion": "10.15",
        "NSHighResolutionCapable": True,
    },
)
