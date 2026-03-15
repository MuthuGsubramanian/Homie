"""Tests for ModelRegistry."""
from __future__ import annotations

import pytest
from pathlib import Path

from homie_core.config import HomieConfig, ModelProfile, ModelTier
from homie_core.inference.model_registry import ModelRegistry, _SMALL_THRESHOLD, _MEDIUM_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> HomieConfig:
    cfg = HomieConfig()
    cfg.storage.path = str(tmp_path / ".homie")
    Path(cfg.storage.path).mkdir(parents=True, exist_ok=True)
    return cfg


def _make_registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(_make_config(tmp_path))


# ---------------------------------------------------------------------------
# Basic empty-registry behaviour
# ---------------------------------------------------------------------------

def test_empty_registry_available_returns_empty(tmp_path):
    reg = _make_registry(tmp_path)
    assert reg.available() == []


def test_empty_registry_best_for_returns_none(tmp_path):
    reg = _make_registry(tmp_path)
    assert reg.best_for(ModelTier.SMALL) is None
    assert reg.best_for(ModelTier.MEDIUM) is None
    assert reg.best_for(ModelTier.LARGE) is None


# ---------------------------------------------------------------------------
# Manual add + lookup
# ---------------------------------------------------------------------------

def test_manual_add_and_lookup_by_tier(tmp_path):
    reg = _make_registry(tmp_path)
    profile = ModelProfile(name="tiny-llm", tier=ModelTier.SMALL, priority=10)
    reg._add(profile)

    result = reg.best_for(ModelTier.SMALL)
    assert result is not None
    assert result.name == "tiny-llm"


def test_available_filters_by_tier(tmp_path):
    reg = _make_registry(tmp_path)
    reg._add(ModelProfile(name="small-model", tier=ModelTier.SMALL, priority=10))
    reg._add(ModelProfile(name="large-model", tier=ModelTier.LARGE, priority=5))

    small_list = reg.available(ModelTier.SMALL)
    assert len(small_list) == 1
    assert small_list[0].name == "small-model"

    large_list = reg.available(ModelTier.LARGE)
    assert len(large_list) == 1
    assert large_list[0].name == "large-model"


def test_available_no_filter_returns_all(tmp_path):
    reg = _make_registry(tmp_path)
    reg._add(ModelProfile(name="m1", tier=ModelTier.SMALL, priority=1))
    reg._add(ModelProfile(name="m2", tier=ModelTier.MEDIUM, priority=2))
    reg._add(ModelProfile(name="m3", tier=ModelTier.LARGE, priority=3))

    all_profiles = reg.available()
    assert len(all_profiles) == 3


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

def test_best_for_returns_highest_priority(tmp_path):
    reg = _make_registry(tmp_path)
    reg._add(ModelProfile(name="low-prio", tier=ModelTier.MEDIUM, priority=1))
    reg._add(ModelProfile(name="high-prio", tier=ModelTier.MEDIUM, priority=99))
    reg._add(ModelProfile(name="mid-prio", tier=ModelTier.MEDIUM, priority=50))

    best = reg.best_for(ModelTier.MEDIUM)
    assert best is not None
    assert best.name == "high-prio"


def test_available_sorted_highest_priority_first(tmp_path):
    reg = _make_registry(tmp_path)
    reg._add(ModelProfile(name="b", tier=ModelTier.SMALL, priority=5))
    reg._add(ModelProfile(name="a", tier=ModelTier.SMALL, priority=20))
    reg._add(ModelProfile(name="c", tier=ModelTier.SMALL, priority=1))

    results = reg.available(ModelTier.SMALL)
    names = [p.name for p in results]
    assert names == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# scan_local
# ---------------------------------------------------------------------------

def _models_dir(config: HomieConfig) -> Path:
    return Path(config.storage.path) / "models"


def test_scan_local_finds_gguf_files(tmp_path):
    cfg = _make_config(tmp_path)
    mdir = _models_dir(cfg)
    mdir.mkdir(parents=True, exist_ok=True)

    # Write a small test file (< 4 GB)
    (mdir / "tiny-model.gguf").write_bytes(b"\x00" * 1024)

    reg = ModelRegistry(cfg)
    reg._scan_local()

    assert len(reg.available()) == 1
    profile = reg.available()[0]
    assert profile.name == "tiny-model"
    assert profile.tier == ModelTier.SMALL
    assert profile.location == "local"


def test_scan_local_tier_by_size(tmp_path):
    cfg = _make_config(tmp_path)
    mdir = _models_dir(cfg)
    mdir.mkdir(parents=True, exist_ok=True)

    # Create files of different sizes by writing sparse files or small placeholders
    # We test tier assignment using the thresholds directly by writing the expected size
    # via truncate (fast, no large disk allocation needed).
    small_file = mdir / "small.gguf"
    medium_file = mdir / "medium.gguf"
    large_file = mdir / "large.gguf"

    # Use truncate to create files of the desired logical sizes without writing actual data
    with open(small_file, "wb") as f:
        f.truncate(_SMALL_THRESHOLD - 1)          # just under 4 GB
    with open(medium_file, "wb") as f:
        f.truncate(_SMALL_THRESHOLD + 1)          # just over 4 GB -> MEDIUM
    with open(large_file, "wb") as f:
        f.truncate(_MEDIUM_THRESHOLD + 1)         # just over 16 GB -> LARGE

    reg = ModelRegistry(cfg)
    reg._scan_local()

    tiers = {p.name: p.tier for p in reg.available()}
    assert tiers["small"] == ModelTier.SMALL
    assert tiers["medium"] == ModelTier.MEDIUM
    assert tiers["large"] == ModelTier.LARGE


def test_scan_local_ignores_non_gguf(tmp_path):
    cfg = _make_config(tmp_path)
    mdir = _models_dir(cfg)
    mdir.mkdir(parents=True, exist_ok=True)

    (mdir / "not-a-model.bin").write_bytes(b"\x00" * 100)
    (mdir / "readme.txt").write_text("hello")

    reg = ModelRegistry(cfg)
    reg._scan_local()
    assert reg.available() == []


def test_scan_local_no_models_dir(tmp_path):
    """If the models directory doesn't exist, scan_local should return silently."""
    cfg = _make_config(tmp_path)
    # Do NOT create the models dir
    reg = ModelRegistry(cfg)
    reg._scan_local()
    assert reg.available() == []


# ---------------------------------------------------------------------------
# refresh clears and re-scans
# ---------------------------------------------------------------------------

def test_refresh_clears_previous_profiles(tmp_path):
    cfg = _make_config(tmp_path)
    reg = ModelRegistry(cfg)
    reg._add(ModelProfile(name="stale-model", tier=ModelTier.SMALL, priority=1))
    assert len(reg.available()) == 1

    reg.refresh()  # no models dir -> empty after refresh
    assert reg.available() == []


# ---------------------------------------------------------------------------
# scan_qubrid — skipped without api_key
# ---------------------------------------------------------------------------

def test_scan_qubrid_skipped_without_api_key(tmp_path):
    cfg = _make_config(tmp_path)
    cfg.llm.api_key = ""  # ensure no key

    reg = ModelRegistry(cfg)
    reg._scan_qubrid()

    assert reg.available() == []


def test_scan_qubrid_registers_model_with_api_key(tmp_path):
    cfg = _make_config(tmp_path)
    cfg.llm.api_key = "test-key-123"
    cfg.inference.qubrid.model = "Qwen/Qwen3.5-Flash"

    reg = ModelRegistry(cfg)
    reg._scan_qubrid()

    profiles = reg.available()
    assert len(profiles) == 1
    assert profiles[0].location == "qubrid"
    assert profiles[0].tier == ModelTier.LARGE
