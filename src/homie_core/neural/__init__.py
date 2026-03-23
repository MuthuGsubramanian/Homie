"""Neural Reasoning Engine — autonomous goal planning and execution.

Public API
----------
MetaAgent   : Top-level orchestrator
NeuralConfig: Engine configuration
Goal        : Goal data model
ThoughtChain: Chain-of-thought plan
Planner     : Goal decomposition
Replanner   : Failure recovery planning
GoalMemory  : Goal persistence
AgentBus    : Inter-agent communication
"""

from .config import NeuralConfig
from .meta_agent import MetaAgent
from .planning.goal import Goal, ThoughtChain, ThoughtStep
from .planning.planner import Planner
from .planning.replanner import Replanner
from .planning.goal_memory import GoalMemory
from .communication.agent_bus import AgentBus, AgentMessage

__all__ = [
    "MetaAgent",
    "NeuralConfig",
    "Goal",
    "ThoughtChain",
    "ThoughtStep",
    "Planner",
    "Replanner",
    "GoalMemory",
    "AgentBus",
    "AgentMessage",
]
