# tests/unit/neural/test_base_agent.py
"""Tests for the BaseAgent abstract interface."""

import time
from unittest.mock import MagicMock

import pytest

from homie_core.neural.agents.base_agent import BaseAgent
from homie_core.neural.communication.agent_bus import AgentBus, AgentMessage


class ConcreteAgent(BaseAgent):
    """Minimal concrete subclass for testing."""

    async def process(self, message: AgentMessage) -> AgentMessage:
        return AgentMessage(
            from_agent=self.name,
            to_agent=message.from_agent,
            message_type="result",
            content={"echo": message.content},
            parent_goal_id=message.parent_goal_id,
        )


class TestBaseAgent:
    def _make_agent(self):
        bus = AgentBus()
        inference = MagicMock(return_value="mocked")
        agent = ConcreteAgent(name="test_agent", agent_bus=bus, inference_fn=inference)
        return agent, bus

    def test_cannot_instantiate_abstract(self):
        bus = AgentBus()
        with pytest.raises(TypeError):
            BaseAgent(name="x", agent_bus=bus, inference_fn=lambda x: x)
        bus.shutdown()

    def test_name_and_inference_fn(self):
        agent, bus = self._make_agent()
        assert agent.name == "test_agent"
        assert agent.inference_fn("hi") == "mocked"
        bus.shutdown()

    def test_send_publishes_to_bus(self):
        agent, bus = self._make_agent()
        received = []
        bus.subscribe_agent("target", lambda m: received.append(m))
        agent.send(to="target", message_type="query", content={"q": "hello"})
        time.sleep(0.1)
        assert len(received) == 1
        assert received[0].from_agent == "test_agent"
        assert received[0].content == {"q": "hello"}
        bus.shutdown()

    def test_subscribe_registers_on_bus(self):
        agent, bus = self._make_agent()
        agent.subscribe()
        # Verify agent receives messages addressed to it
        received = []
        original = agent._on_message
        agent._on_message = lambda m: received.append(m)
        # Re-subscribe with patched handler
        bus.subscribe_agent(agent.name, agent._on_message)
        bus.publish(AgentMessage(
            from_agent="other", to_agent="test_agent", message_type="goal", content={}
        ))
        time.sleep(0.1)
        assert len(received) == 1
        bus.shutdown()

    def test_process_returns_response(self):
        import asyncio
        agent, bus = self._make_agent()
        msg = AgentMessage(
            from_agent="meta", to_agent="test_agent", message_type="goal",
            content={"data": 42}, parent_goal_id="g1",
        )
        response = asyncio.run(agent.process(msg))
        assert response.from_agent == "test_agent"
        assert response.to_agent == "meta"
        assert response.content["echo"] == {"data": 42}
        bus.shutdown()
