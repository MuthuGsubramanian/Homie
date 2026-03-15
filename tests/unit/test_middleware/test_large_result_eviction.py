from __future__ import annotations

import pytest

from homie_core.backend.state import StateBackend
from homie_core.config import HomieConfig, ContextConfig
from homie_core.middleware.large_result_eviction import LargeResultEvictionMiddleware


def make_mw(threshold: int = 100) -> tuple[LargeResultEvictionMiddleware, StateBackend]:
    cfg = HomieConfig(context=ContextConfig(large_result_threshold=threshold))
    backend = StateBackend()
    mw = LargeResultEvictionMiddleware(cfg, backend)
    return mw, backend


def make_large_result(num_lines: int, line_len: int = 20) -> str:
    return "\n".join(f"line{i:04d} " + "x" * (line_len - 8) for i in range(num_lines))


# ---------------------------------------------------------------------------
# Passthrough cases
# ---------------------------------------------------------------------------

def test_result_below_threshold_passthrough():
    mw, _ = make_mw(threshold=200)
    result = "small result"
    assert mw.wrap_tool_result("some_tool", result) == result


def test_result_exactly_at_threshold_passthrough():
    mw, _ = make_mw(threshold=10)
    result = "a" * 10
    assert mw.wrap_tool_result("some_tool", result) == result


def test_excluded_tool_passthrough_even_if_large():
    mw, backend = make_mw(threshold=5)
    large = "x" * 1000
    for tool in ["ls", "glob", "grep", "read_file", "edit_file", "write_file", "search_files"]:
        result = mw.wrap_tool_result(tool, large)
        assert result == large, f"Expected passthrough for excluded tool {tool!r}"
    # Nothing should be written to backend for excluded tools
    assert backend.glob("/**") == []


# ---------------------------------------------------------------------------
# Eviction cases
# ---------------------------------------------------------------------------

def test_large_result_is_evicted():
    mw, backend = make_mw(threshold=50)
    large = make_large_result(20, line_len=10)  # well above 50 chars
    output = mw.wrap_tool_result("my_tool", large)
    # Output should not equal the original
    assert output != large
    assert "Full output saved to:" in output


def test_full_content_stored_in_backend():
    mw, backend = make_mw(threshold=50)
    large = make_large_result(20, line_len=10)
    mw.wrap_tool_result("my_tool", large)
    # Something was written
    paths = backend.glob("/large_tool_results/**")
    # glob may not match deeply; check the directory listing
    entries = backend.ls("/large_tool_results")
    assert len(entries) == 1
    stored = backend.read(entries[0].path, limit=1000).content
    # The stored content is the full original (possibly truncated by read's default limit)
    assert stored.startswith("line0000")


def test_evicted_preview_contains_first_five_lines():
    mw, _ = make_mw(threshold=5)
    lines = [f"line{i}" for i in range(20)]
    result_text = "\n".join(lines)
    output = mw.wrap_tool_result("my_tool", result_text)
    for i in range(5):
        assert lines[i] in output


def test_evicted_preview_contains_last_five_lines():
    mw, _ = make_mw(threshold=5)
    lines = [f"line{i:02d}" for i in range(20)]
    result_text = "\n".join(lines)
    output = mw.wrap_tool_result("my_tool", result_text)
    for i in range(15, 20):
        assert lines[i] in output


def test_evicted_preview_contains_line_count_and_char_count():
    mw, _ = make_mw(threshold=5)
    lines = [f"line{i}" for i in range(10)]
    result_text = "\n".join(lines)
    output = mw.wrap_tool_result("my_tool", result_text)
    assert "10 total lines" in output
    assert f"{len(result_text)} chars" in output


def test_evicted_path_uses_safe_name():
    mw, backend = make_mw(threshold=5)
    large = "x" * 200
    output = mw.wrap_tool_result("tool with spaces!", large)
    assert "Full output saved to: /large_tool_results/" in output
    # The path in the output should not contain spaces or exclamation marks
    path_line = [l for l in output.split("\n") if "Full output saved to:" in l][0]
    path = path_line.split("Full output saved to: ")[1].strip()
    assert " " not in path
    assert "!" not in path


def test_name_and_order():
    mw, _ = make_mw()
    assert mw.name == "large_result_eviction"
    assert mw.order == 90


def test_excluded_tools_set():
    assert "ls" in LargeResultEvictionMiddleware.EXCLUDED_TOOLS
    assert "glob" in LargeResultEvictionMiddleware.EXCLUDED_TOOLS
    assert "grep" in LargeResultEvictionMiddleware.EXCLUDED_TOOLS
    assert "read_file" in LargeResultEvictionMiddleware.EXCLUDED_TOOLS
    assert "edit_file" in LargeResultEvictionMiddleware.EXCLUDED_TOOLS
    assert "write_file" in LargeResultEvictionMiddleware.EXCLUDED_TOOLS
    assert "search_files" in LargeResultEvictionMiddleware.EXCLUDED_TOOLS
