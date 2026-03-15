from __future__ import annotations

import pytest

from homie_core.memory.working import WorkingMemory
from homie_core.middleware.dangling_tool_repair import DanglingToolCallMiddleware


def make_mw() -> tuple[DanglingToolCallMiddleware, WorkingMemory]:
    wm = WorkingMemory()
    mw = DanglingToolCallMiddleware(wm)
    return mw, wm


def fill_conversation(wm: WorkingMemory, messages: list[dict]) -> None:
    for msg in messages:
        wm.add_message(msg["role"], msg["content"])


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------

def test_no_tool_calls_no_change():
    mw, wm = make_mw()
    fill_conversation(wm, [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "how are you?"},
    ])
    original = wm.get_conversation()
    result = mw.before_turn("next msg", {})
    assert result == "next msg"
    assert wm.get_conversation() == original


def test_tool_call_with_matching_result_no_change():
    mw, wm = make_mw()
    fill_conversation(wm, [
        {"role": "user", "content": "search for something"},
        {"role": "assistant", "content": "<tool>search(query='foo')</tool>"},
        {"role": "system", "content": "[Tool: search] Result: found 5 items"},
        {"role": "user", "content": "thanks"},
    ])
    original_len = len(wm.get_conversation())
    mw.before_turn("next", {})
    assert len(wm.get_conversation()) == original_len


def test_returns_original_message():
    mw, wm = make_mw()
    fill_conversation(wm, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    result = mw.before_turn("my message", {})
    assert result == "my message"


def test_single_message_no_change():
    mw, wm = make_mw()
    fill_conversation(wm, [
        {"role": "user", "content": "hello"},
    ])
    original = wm.get_conversation()
    mw.before_turn("next", {})
    assert wm.get_conversation() == original


def test_empty_conversation_no_change():
    mw, wm = make_mw()
    mw.before_turn("msg", {})
    assert wm.get_conversation() == []


# ---------------------------------------------------------------------------
# Dangling tool call cases
# ---------------------------------------------------------------------------

def test_dangling_tool_call_injects_synthetic_result():
    mw, wm = make_mw()
    fill_conversation(wm, [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "<tool>search(query='test')</tool>"},
        {"role": "user", "content": "actually nevermind"},
    ])
    mw.before_turn("new msg", {})
    conv = wm.get_conversation()
    # A synthetic result should be injected after the assistant message (index 1)
    # so conv[2] should be the synthetic result
    synthetic = conv[2]
    assert synthetic["role"] == "system"
    assert "[Tool: search]" in synthetic["content"]
    assert "cancelled" in synthetic["content"].lower() or "Result" in synthetic["content"]


def test_dangling_tool_call_conversation_length_increases():
    mw, wm = make_mw()
    fill_conversation(wm, [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "<tool>read_file(path='/tmp/x')</tool>"},
        {"role": "user", "content": "cancel that"},
    ])
    original_len = len(wm.get_conversation())
    mw.before_turn("new msg", {})
    assert len(wm.get_conversation()) == original_len + 1


def test_multiple_dangling_calls_all_get_synthetic_results():
    mw, wm = make_mw()
    fill_conversation(wm, [
        {"role": "user", "content": "do stuff"},
        {"role": "assistant", "content": "<tool>tool_a(x=1)</tool>"},
        {"role": "user", "content": "also do this"},
        {"role": "assistant", "content": "<tool>tool_b(y=2)</tool>"},
        {"role": "user", "content": "forget it"},
    ])
    original_len = len(wm.get_conversation())
    mw.before_turn("new", {})
    conv = wm.get_conversation()
    # Two synthetic results should have been inserted
    assert len(conv) == original_len + 2
    system_msgs = [m for m in conv if m["role"] == "system"]
    assert len(system_msgs) == 2
    tool_names = {m["content"].split("[Tool: ")[1].split("]")[0] for m in system_msgs}
    assert "tool_a" in tool_names
    assert "tool_b" in tool_names


def test_last_message_is_dangling_tool_call_synthetic_added():
    mw, wm = make_mw()
    fill_conversation(wm, [
        {"role": "user", "content": "run this"},
        {"role": "assistant", "content": "<tool>execute(cmd='ls')</tool>"},
    ])
    original_len = len(wm.get_conversation())
    mw.before_turn("interrupt", {})
    conv = wm.get_conversation()
    assert len(conv) == original_len + 1
    assert conv[-1]["role"] == "system"
    assert "[Tool: execute]" in conv[-1]["content"]


def test_tool_call_with_error_result_not_duplicated():
    mw, wm = make_mw()
    fill_conversation(wm, [
        {"role": "user", "content": "try something"},
        {"role": "assistant", "content": "<tool>search(q='x')</tool>"},
        {"role": "system", "content": "[Tool: search] Error: timeout"},
        {"role": "user", "content": "ok"},
    ])
    original_len = len(wm.get_conversation())
    mw.before_turn("next", {})
    # Error result should count as a valid result — no injection
    assert len(wm.get_conversation()) == original_len


def test_non_assistant_tool_call_like_text_ignored():
    mw, wm = make_mw()
    fill_conversation(wm, [
        {"role": "user", "content": "<tool>search(q='x')</tool>"},
        {"role": "assistant", "content": "I see you mentioned a tool"},
    ])
    original_len = len(wm.get_conversation())
    mw.before_turn("hi", {})
    # Tool-call-like text in user messages should not trigger injection
    assert len(wm.get_conversation()) == original_len


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_name_and_order():
    mw, _ = make_mw()
    assert mw.name == "dangling_tool_repair"
    assert mw.order == 2
