# tests/unit/neural/test_agent_bus.py
"""Tests for AgentBus and AgentMessage."""

import time

import pytest

from homie_core.neural.communication.agent_bus import AgentBus, AgentMessage


class TestAgentMessage:
    def test_creation_defaults(self):
        msg = AgentMessage(
            from_agent="reasoning",
            to_agent="action",
            message_type="goal",
            content={"task": "analyze"},
        )
        assert msg.from_agent == "reasoning"
        assert msg.to_agent == "action"
        assert msg.priority == 0
        assert msg.parent_goal_id == ""
        assert msg.timestamp > 0

    def test_to_dict(self):
        msg = AgentMessage(
            from_agent="a",
            to_agent="b",
            message_type="result",
            content={"ok": True},
            priority=5,
            parent_goal_id="g1",
        )
        d = msg.to_dict()
        assert d["from_agent"] == "a"
        assert d["priority"] == 5
        assert d["parent_goal_id"] == "g1"
        assert "timestamp" in d


class TestAgentBus:
    def test_subscribe_agent_and_publish(self):
        bus = AgentBus()
        received = []
        bus.subscribe_agent("action", lambda m: received.append(m))
        msg = AgentMessage(
            from_agent="meta", to_agent="action", message_type="goal", content={}
        )
        bus.publish(msg)
        time.sleep(0.1)
        assert len(received) == 1
        assert received[0].from_agent == "meta"
        bus.shutdown()

    def test_subscribe_type_and_publish(self):
        bus = AgentBus()
        received = []
        bus.subscribe_type("error", lambda m: received.append(m))
        bus.publish(
            AgentMessage(
                from_agent="a", to_agent="b", message_type="error", content={"msg": "fail"}
            )
        )
        time.sleep(0.1)
        assert len(received) == 1
        bus.shutdown()

    def test_wildcard_receives_all(self):
        bus = AgentBus()
        received = []
        bus.subscribe_type("*", lambda m: received.append(m))
        bus.publish(AgentMessage(from_agent="a", to_agent="b", message_type="goal", content={}))
        bus.publish(AgentMessage(from_agent="c", to_agent="d", message_type="result", content={}))
        time.sleep(0.1)
        assert len(received) == 2
        bus.shutdown()

    def test_unsubscribe_agent(self):
        bus = AgentBus()
        received = []
        cb = lambda m: received.append(m)
        bus.subscribe_agent("x", cb)
        bus.unsubscribe_agent("x", cb)
        bus.publish(AgentMessage(from_agent="a", to_agent="x", message_type="goal", content={}))
        time.sleep(0.1)
        assert len(received) == 0
        bus.shutdown()

    def test_priority_ordering(self):
        """Higher-priority messages should be dispatched before lower ones."""
        bus = AgentBus()
        received = []
        bus.subscribe_agent("target", lambda m: received.append(m.priority))

        # Pause processing by shutting down worker, enqueue manually, restart
        bus._running = False
        bus._queue.put((0, 0, None))  # stop current worker
        bus._worker.join(timeout=1)

        # Enqueue directly — low priority first, high second
        low = AgentMessage(from_agent="a", to_agent="target", message_type="goal", content={}, priority=1)
        high = AgentMessage(from_agent="a", to_agent="target", message_type="goal", content={}, priority=10)
        bus._queue.put((-low.priority, 1, low))
        bus._queue.put((-high.priority, 2, high))

        # Restart worker
        bus._running = True
        import threading
        bus._worker = threading.Thread(target=bus._process_loop, daemon=True)
        bus._worker.start()
        time.sleep(0.15)

        assert len(received) == 2
        assert received[0] == 10  # high priority first
        assert received[1] == 1
        bus.shutdown()

    def test_handler_exception_does_not_crash_bus(self):
        bus = AgentBus()
        good = []

        def bad(m):
            raise RuntimeError("boom")

        bus.subscribe_agent("t", bad)
        bus.subscribe_agent("t", lambda m: good.append(m))
        bus.publish(AgentMessage(from_agent="a", to_agent="t", message_type="goal", content={}))
        time.sleep(0.1)
        assert len(good) == 1
        bus.shutdown()

    def test_shutdown_stops_processing(self):
        bus = AgentBus()
        bus.shutdown()
        # Publish after shutdown — should not raise
        bus.publish(AgentMessage(from_agent="a", to_agent="b", message_type="goal", content={}))

    def test_multiple_agent_subscribers(self):
        bus = AgentBus()
        r1, r2 = [], []
        bus.subscribe_agent("x", lambda m: r1.append(m))
        bus.subscribe_agent("x", lambda m: r2.append(m))
        bus.publish(AgentMessage(from_agent="a", to_agent="x", message_type="goal", content={}))
        time.sleep(0.1)
        assert len(r1) == 1
        assert len(r2) == 1
        bus.shutdown()
