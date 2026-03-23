"""Base class for all Neural Reasoning Engine agents."""

from abc import ABC, abstractmethod
from typing import Callable

from ..communication.agent_bus import AgentBus, AgentMessage


class BaseAgent(ABC):
    """Abstract base agent — every specialized agent inherits this.

    Parameters
    ----------
    name:
        Unique agent identifier (e.g. "reasoning", "research").
    agent_bus:
        The shared inter-agent communication bus.
    inference_fn:
        Callable that performs LLM inference.  Signature:
        ``inference_fn(prompt: str, **kwargs) -> str``.
    """

    name: str

    def __init__(self, name: str, agent_bus: AgentBus, inference_fn: Callable) -> None:
        self.name = name
        self.bus = agent_bus
        self.inference_fn = inference_fn

    # ── abstract ────────────────────────────────────────────────────

    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentMessage:
        """Handle an incoming message and return a response message."""

    # ── helpers ─────────────────────────────────────────────────────

    def send(self, to: str, message_type: str, content: dict) -> None:
        """Convenience: publish a message on the bus from this agent."""
        msg = AgentMessage(
            from_agent=self.name,
            to_agent=to,
            message_type=message_type,
            content=content,
        )
        self.bus.publish(msg)

    def subscribe(self) -> None:
        """Register this agent on the bus to receive messages addressed to it."""
        self.bus.subscribe_agent(self.name, self._on_message)

    def _on_message(self, message: AgentMessage) -> None:
        """Default sync callback — subclasses can override for custom routing."""
        # By default, store for later async processing.
        pass
