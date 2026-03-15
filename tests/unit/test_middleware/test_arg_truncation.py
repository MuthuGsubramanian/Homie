from __future__ import annotations

import pytest

from homie_core.config import HomieConfig, ContextConfig
from homie_core.memory.working import WorkingMemory
from homie_core.middleware.arg_truncation import ArgTruncationMiddleware


def make_mw(threshold: int = 50) -> tuple[ArgTruncationMiddleware, WorkingMemory]:
    cfg = HomieConfig(context=ContextConfig(arg_truncation_threshold=threshold))
    wm = WorkingMemory()
    mw = ArgTruncationMiddleware(cfg, wm)
    return mw, wm


def fill_conversation(wm: WorkingMemory, messages: list[dict]) -> None:
    """Populate WorkingMemory conversation with given messages."""
    for msg in messages:
        wm.add_message(msg["role"], msg["content"])


# ---------------------------------------------------------------------------
# Basic no-op cases
# ---------------------------------------------------------------------------

def test_conversation_less_than_6_messages_no_truncation():
    mw, wm = make_mw(threshold=10)
    # 5 messages — below the guard
    fill_conversation(wm, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "write_file " + "x" * 200},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "write_file " + "x" * 200},
        {"role": "user", "content": "done"},
    ])
    result = mw.before_turn("new message", {})
    assert result == "new message"
    conv = wm.get_conversation()
    # Nothing should have been truncated because len(conv) == 5 < 6
    assert "write_file " + "x" * 200 in [m["content"] for m in conv]


def test_message_returns_original_message_string():
    mw, wm = make_mw()
    fill_conversation(wm, [{"role": "user", "content": "x"} for _ in range(6)])
    result = mw.before_turn("the user message", {})
    assert result == "the user message"


def test_short_assistant_message_not_truncated():
    mw, wm = make_mw(threshold=500)
    msgs = []
    for i in range(6):
        msgs.append({"role": "user", "content": "question"})
        msgs.append({"role": "assistant", "content": "write_file short"})
    fill_conversation(wm, msgs)
    mw.before_turn("new", {})
    conv = wm.get_conversation()
    # No assistant message should be truncated — all are <= threshold
    for m in conv:
        if m["role"] == "assistant":
            assert "truncated" not in m["content"]


def test_message_without_tool_name_not_truncated():
    mw, wm = make_mw(threshold=10)
    # Large assistant messages but no truncatable tool name
    fill_conversation(wm, [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a" * 500},  # large but no tool
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "b" * 500},  # large but no tool
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "c" * 500},  # large but no tool
        {"role": "user", "content": "q"},
    ])
    mw.before_turn("new", {})
    conv = wm.get_conversation()
    for m in conv:
        if m["role"] == "assistant":
            assert "truncated" not in m["content"]


# ---------------------------------------------------------------------------
# Truncation cases
# ---------------------------------------------------------------------------

def test_old_write_file_message_with_large_content_is_truncated():
    mw, wm = make_mw(threshold=50)
    large_content = "write_file " + "x" * 500
    fill_conversation(wm, [
        {"role": "user", "content": "please write"},
        {"role": "assistant", "content": large_content},  # old, will be truncated
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "anything else?"},
        {"role": "assistant", "content": "no"},
        {"role": "user", "content": "new question"},
    ])
    mw.before_turn("new question", {})
    conv = wm.get_conversation()
    assistant_msgs = [m for m in conv if m["role"] == "assistant"]
    # The very first assistant message should be truncated
    assert "...(argument truncated)" in assistant_msgs[0]["content"]
    assert len(assistant_msgs[0]["content"]) < len(large_content)


def test_truncated_content_preserves_first_20_chars():
    mw, wm = make_mw(threshold=50)
    large_content = "write_file ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "x" * 500
    fill_conversation(wm, [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": large_content},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "c"},
        {"role": "user", "content": "d"},
        {"role": "assistant", "content": "e"},
        {"role": "user", "content": "f"},
    ])
    mw.before_turn("new", {})
    conv = wm.get_conversation()
    old_msg = [m for m in conv if m["role"] == "assistant"][0]
    assert old_msg["content"].startswith(large_content[:20])
    assert old_msg["content"].endswith("...(argument truncated)")


def test_edit_file_message_is_truncated():
    mw, wm = make_mw(threshold=50)
    large_content = "edit_file " + "z" * 500
    fill_conversation(wm, [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": large_content},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "new"},
    ])
    mw.before_turn("new", {})
    conv = wm.get_conversation()
    first_assistant = [m for m in conv if m["role"] == "assistant"][0]
    assert "...(argument truncated)" in first_assistant["content"]


def test_recent_messages_last_3_are_protected():
    mw, wm = make_mw(threshold=50)
    # Build a conversation where the last 3 assistant messages are large+tool
    fill_conversation(wm, [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "old " + "x" * 500},          # old (idx 1 of 10)
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "write_file " + "x" * 500},   # old, should truncate
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "write_file " + "y" * 500},   # old, should truncate
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "write_file " + "z" * 500},   # last 3 → protected
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "write_file " + "w" * 500},   # last 3 → protected
    ])
    mw.before_turn("new msg", {})
    conv = wm.get_conversation()
    assistant_msgs = [m for m in conv if m["role"] == "assistant"]
    # Index 1 (write_file + x*500) should be truncated
    assert "...(argument truncated)" in assistant_msgs[1]["content"]
    # Index 2 (write_file + y*500) should be truncated
    assert "...(argument truncated)" in assistant_msgs[2]["content"]
    # Index 3 and 4 are in last 3 → protected
    assert "z" * 10 in assistant_msgs[3]["content"]  # still has original content
    assert "w" * 10 in assistant_msgs[4]["content"]  # still has original content


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_name_and_order():
    mw, _ = make_mw()
    assert mw.name == "arg_truncation"
    assert mw.order == 8


def test_truncatable_tools_set():
    assert "write_file" in ArgTruncationMiddleware.TRUNCATABLE_TOOLS
    assert "edit_file" in ArgTruncationMiddleware.TRUNCATABLE_TOOLS
