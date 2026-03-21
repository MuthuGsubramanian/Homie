"""Recovery playbook — seed rules plus learned strategies."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .engine import RecoveryTier


@dataclass
class PlaybookEntry:
    """A single recovery strategy entry."""

    module: str
    failure_type: str
    tier: RecoveryTier
    action: str
    success_count: int = 0
    fail_count: int = 0
    learned: bool = False
    deprecated: bool = False

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0


def _seed_entries() -> list[PlaybookEntry]:
    """Static seed playbook — the starting point that Homie evolves from."""
    return [
        # Inference
        PlaybookEntry("inference", "timeout", RecoveryTier.RETRY, "retry with shorter max_tokens"),
        PlaybookEntry("inference", "timeout", RecoveryTier.FALLBACK, "reduce context_length and retry"),
        PlaybookEntry("inference", "timeout", RecoveryTier.REBUILD, "switch to smaller model"),
        PlaybookEntry("inference", "timeout", RecoveryTier.DEGRADE, "serve cached responses"),
        PlaybookEntry("inference", "oom", RecoveryTier.RETRY, "retry with fewer gpu_layers"),
        PlaybookEntry("inference", "oom", RecoveryTier.FALLBACK, "reduce batch size, offload to CPU"),
        PlaybookEntry("inference", "oom", RecoveryTier.REBUILD, "switch to full CPU inference"),
        PlaybookEntry("inference", "oom", RecoveryTier.DEGRADE, "queue requests, serve cache"),
        PlaybookEntry("inference", "model_corrupt", RecoveryTier.RETRY, "re-verify GGUF hash"),
        PlaybookEntry("inference", "model_corrupt", RecoveryTier.FALLBACK, "re-download model"),
        PlaybookEntry("inference", "model_corrupt", RecoveryTier.REBUILD, "fall back to alternative model"),
        PlaybookEntry("inference", "gpu_crash", RecoveryTier.RETRY, "retry after cooldown"),
        PlaybookEntry("inference", "gpu_crash", RecoveryTier.FALLBACK, "restart with CPU fallback"),
        PlaybookEntry("inference", "gpu_crash", RecoveryTier.DEGRADE, "CPU-only mode"),
        # Storage
        PlaybookEntry("storage", "sqlite_locked", RecoveryTier.RETRY, "retry with backoff"),
        PlaybookEntry("storage", "sqlite_locked", RecoveryTier.FALLBACK, "force close stale connections"),
        PlaybookEntry("storage", "sqlite_locked", RecoveryTier.REBUILD, "copy DB, repair, swap"),
        PlaybookEntry("storage", "sqlite_corrupt", RecoveryTier.RETRY, "run integrity_check and auto-fix"),
        PlaybookEntry("storage", "sqlite_corrupt", RecoveryTier.FALLBACK, "restore from backup"),
        PlaybookEntry("storage", "sqlite_corrupt", RecoveryTier.REBUILD, "rebuild from available data"),
        PlaybookEntry("storage", "sqlite_corrupt", RecoveryTier.DEGRADE, "start fresh DB"),
        PlaybookEntry("storage", "chromadb_down", RecoveryTier.RETRY, "restart ChromaDB process"),
        PlaybookEntry("storage", "chromadb_down", RecoveryTier.FALLBACK, "rebuild collection"),
        PlaybookEntry("storage", "chromadb_down", RecoveryTier.DEGRADE, "fall back to keyword search"),
        PlaybookEntry("storage", "disk_full", RecoveryTier.RETRY, "emergency cleanup"),
        PlaybookEntry("storage", "disk_full", RecoveryTier.FALLBACK, "compress data"),
        PlaybookEntry("storage", "disk_full", RecoveryTier.REBUILD, "aggressive retention 7 days"),
        PlaybookEntry("storage", "disk_full", RecoveryTier.DEGRADE, "read-only mode"),
        # Voice
        PlaybookEntry("voice", "stt_crash", RecoveryTier.RETRY, "reload Whisper model"),
        PlaybookEntry("voice", "stt_crash", RecoveryTier.FALLBACK, "switch quality medium to small"),
        PlaybookEntry("voice", "stt_crash", RecoveryTier.DEGRADE, "text-only input"),
        PlaybookEntry("voice", "tts_failure", RecoveryTier.RETRY, "retry synthesis"),
        PlaybookEntry("voice", "tts_failure", RecoveryTier.FALLBACK, "switch TTS engine"),
        PlaybookEntry("voice", "tts_failure", RecoveryTier.DEGRADE, "text-only output"),
        PlaybookEntry("voice", "audio_device_lost", RecoveryTier.RETRY, "re-enumerate devices"),
        PlaybookEntry("voice", "audio_device_lost", RecoveryTier.DEGRADE, "text-only mode"),
        # Config
        PlaybookEntry("config", "parse_error", RecoveryTier.RETRY, "re-read and validate YAML"),
        PlaybookEntry("config", "parse_error", RecoveryTier.FALLBACK, "use last known good config"),
        PlaybookEntry("config", "parse_error", RecoveryTier.REBUILD, "merge defaults with parseable fields"),
        PlaybookEntry("config", "parse_error", RecoveryTier.DEGRADE, "boot with full defaults"),
        PlaybookEntry("config", "invalid_values", RecoveryTier.RETRY, "clamp to valid range"),
        PlaybookEntry("config", "invalid_values", RecoveryTier.FALLBACK, "reset section to defaults"),
    ]


class RecoveryPlaybook:
    """Manages recovery strategies — seed rules plus learned mutations."""

    def __init__(self, seed: bool = True) -> None:
        self._entries: list[PlaybookEntry] = _seed_entries() if seed else []

    def get_entries(self, module: str) -> list[PlaybookEntry]:
        """Get all entries for a module."""
        return [e for e in self._entries if e.module == module and not e.deprecated]

    def get_best_entry(
        self, module: str, failure_type: str, tier: RecoveryTier
    ) -> Optional[PlaybookEntry]:
        """Get the best strategy for a specific failure at a specific tier."""
        candidates = [
            e for e in self._entries
            if e.module == module and e.failure_type == failure_type
            and e.tier == tier and not e.deprecated
        ]
        if not candidates:
            return None
        # Prefer highest success rate, then seed over learned
        return max(candidates, key=lambda e: (e.success_rate, not e.learned))

    def record_outcome(self, entry: PlaybookEntry, success: bool) -> None:
        """Record whether a strategy succeeded or failed."""
        if success:
            entry.success_count += 1
        else:
            entry.fail_count += 1

    def add_entry(self, entry: PlaybookEntry) -> None:
        """Add a learned strategy to the playbook."""
        self._entries.append(entry)

    def deprecate_entry(self, entry: PlaybookEntry) -> None:
        """Mark a strategy as deprecated (never succeeded)."""
        entry.deprecated = True

    def export_to_file(self, path: Path | str) -> None:
        """Export playbook to JSON for persistence."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for e in self._entries:
            d = {
                "module": e.module,
                "failure_type": e.failure_type,
                "tier": e.tier.value,
                "action": e.action,
                "success_count": e.success_count,
                "fail_count": e.fail_count,
                "learned": e.learned,
                "deprecated": e.deprecated,
            }
            data.append(d)
        path.write_text(json.dumps(data, indent=2))

    def import_from_file(self, path: Path | str) -> None:
        """Import playbook from JSON."""
        path = Path(path)
        data = json.loads(path.read_text())
        for d in data:
            self._entries.append(PlaybookEntry(
                module=d["module"],
                failure_type=d["failure_type"],
                tier=RecoveryTier(d["tier"]),
                action=d["action"],
                success_count=d.get("success_count", 0),
                fail_count=d.get("fail_count", 0),
                learned=d.get("learned", False),
                deprecated=d.get("deprecated", False),
            ))
