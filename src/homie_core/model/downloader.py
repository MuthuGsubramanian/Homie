from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import hf_hub_download, snapshot_download


class ModelDownloader:
    def __init__(self, models_dir: Path | str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def download_gguf(self, repo_id: str, filename: str, progress_callback: Optional[Callable[[float], None]] = None) -> Path:
        local_path = hf_hub_download(
            repo_id=repo_id, filename=filename,
            local_dir=str(self.models_dir / repo_id.replace("/", "_")),
            resume_download=True,
        )
        return Path(local_path)

    def download_safetensors(self, repo_id: str, progress_callback: Optional[Callable[[float], None]] = None) -> Path:
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(self.models_dir / repo_id.replace("/", "_")),
            ignore_patterns=["*.bin", "*.pt", "*.ot"],
            resume_download=True,
        )
        return Path(local_path)
