from __future__ import annotations

import pytest
from homie_core.middleware.base import HomieMiddleware


def test_default_name():
    mw = HomieMiddleware()
    assert mw.name == "unnamed"


def test_default_order():
    mw = HomieMiddleware()
    assert mw.order == 100


def test_modify_tools_passthrough():
    mw = HomieMiddleware()
    tools = [{"name": "tool1"}, {"name": "tool2"}]
    result = mw.modify_tools(tools)
    assert result == tools


def test_modify_prompt_passthrough():
    mw = HomieMiddleware()
    result = mw.modify_prompt("hello world")
    assert result == "hello world"


def test_before_turn_returns_message():
    mw = HomieMiddleware()
    result = mw.before_turn("user input", {})
    assert result == "user input"


def test_after_turn_passthrough():
    mw = HomieMiddleware()
    result = mw.after_turn("response text", {})
    assert result == "response text"


def test_wrap_tool_call_passthrough():
    mw = HomieMiddleware()
    args = {"param": "value"}
    result = mw.wrap_tool_call("my_tool", args)
    assert result == args


def test_wrap_tool_result_passthrough():
    mw = HomieMiddleware()
    result = mw.wrap_tool_result("my_tool", "result_data")
    assert result == "result_data"


def test_subclass_can_override_name_and_order():
    class MyMW(HomieMiddleware):
        name = "my_middleware"
        order = 10

    mw = MyMW()
    assert mw.name == "my_middleware"
    assert mw.order == 10


def test_subclass_can_override_before_turn_to_block():
    class BlockingMW(HomieMiddleware):
        def before_turn(self, message: str, state: dict) -> str | None:
            return None  # blocks

    mw = BlockingMW()
    result = mw.before_turn("msg", {})
    assert result is None


def test_subclass_can_override_wrap_tool_call_to_block():
    class BlockingMW(HomieMiddleware):
        def wrap_tool_call(self, name: str, args: dict) -> dict | None:
            return None  # blocks

    mw = BlockingMW()
    result = mw.wrap_tool_call("tool", {"a": 1})
    assert result is None


def test_subclass_can_modify_prompt():
    class PrefixMW(HomieMiddleware):
        def modify_prompt(self, prompt: str) -> str:
            return "[PREFIX] " + prompt

    mw = PrefixMW()
    assert mw.modify_prompt("hello") == "[PREFIX] hello"
