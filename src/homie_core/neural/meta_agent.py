"""MetaAgent — top-level orchestrator for the Neural Reasoning Engine.

Receives goals, plans them via the Planner, delegates steps to specialised
agents, monitors execution, invokes the Replanner on failure, and validates
final results through the ValidationAgent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

from .communication.agent_bus import AgentBus, AgentMessage
from .config import NeuralConfig
from .planning.goal import Goal, ThoughtChain, ThoughtStep
from .planning.goal_memory import GoalMemory
from .planning.planner import Planner
from .planning.replanner import Replanner

logger = logging.getLogger(__name__)


class MetaAgent:
    """Top-level orchestrator — receives goals, plans, delegates, monitors.

    Parameters
    ----------
    inference_fn:
        LLM inference callable ``(prompt: str, **kw) -> str``.
    goal_memory:
        Persistence layer for goals.
    planner:
        Decomposes goals into ThoughtChains.
    replanner:
        Generates alternative plans when a step fails.
    agents:
        Mapping of agent name to agent instance (must implement ``process``).
    agent_bus:
        Shared inter-agent communication bus.
    config:
        Engine-wide configuration.
    """

    def __init__(
        self,
        inference_fn: Callable,
        goal_memory: GoalMemory,
        planner: Planner,
        replanner: Replanner,
        agents: dict,
        agent_bus: AgentBus,
        config: Optional[NeuralConfig] = None,
    ) -> None:
        self.inference_fn = inference_fn
        self.goal_memory = goal_memory
        self.planner = planner
        self.replanner = replanner
        self.agents = agents
        self.bus = agent_bus
        self.config = config or NeuralConfig()

        # In-memory tracking of active goals (superset of goal_memory for speed)
        self._active_goals: dict[str, Goal] = {}

    # ── public API ──────────────────────────────────────────────────

    def submit_goal(self, description: str, priority: int = 5) -> Goal:
        """Submit a new goal for autonomous execution.

        1. Create a Goal object.
        2. Plan using the Planner.
        3. Persist to GoalMemory.
        4. Register as active.

        Returns the fully planned Goal (not yet executed).
        """
        if len(self._active_goals) >= self.config.max_concurrent_goals:
            raise RuntimeError(
                f"Max concurrent goals ({self.config.max_concurrent_goals}) reached"
            )

        goal = Goal(
            id=Goal.new_id(),
            description=description,
            priority=priority,
        )

        # Plan
        thought_chain = self.planner.plan(description)
        goal.thought_chain = thought_chain

        # Persist and track
        self.goal_memory.save_goal(goal)
        self._active_goals[goal.id] = goal

        logger.info(
            "Goal submitted: id=%s  desc=%r  steps=%d",
            goal.id,
            description[:80],
            len(thought_chain.steps),
        )
        return goal

    def execute_goal(self, goal: Goal) -> dict:
        """Execute a planned goal step by step (synchronous).

        For each ready step in the thought chain:
        1. Check autonomy level — pause if supervised and agent is 'action'.
        2. Delegate to the appropriate agent.
        3. Collect result and mark step complete/failed.
        4. If a step fails, attempt re-planning.
        5. Once all steps complete, run validation.

        Returns a summary dict with keys: ``goal_id``, ``status``, ``result``,
        ``validation``.
        """
        if goal.thought_chain is None:
            return {
                "goal_id": goal.id,
                "status": "failed",
                "result": None,
                "validation": None,
                "error": "Goal has no thought chain (not planned).",
            }

        chain = goal.thought_chain
        chain.status = "executing"
        last_result: Optional[dict] = None

        while not chain.is_complete:
            step = chain.advance()
            if step is None:
                # No ready steps and not complete → stuck or failed
                if chain.has_failed:
                    break
                # Possibly all remaining steps are blocked
                break

            # Autonomy gate: supervised mode pauses before action steps
            if (
                self.config.autonomy_level == "supervised"
                and step.agent == "action"
            ):
                step.status = "pending"
                logger.info(
                    "Supervised mode: pausing before action step %s", step.id
                )
                return {
                    "goal_id": goal.id,
                    "status": "paused",
                    "paused_step": step.id,
                    "result": last_result,
                    "validation": None,
                }

            # Delegate to agent
            result = self._delegate_step(step, goal)

            if step.status == "failed":
                # Attempt re-plan
                replan_result = self._try_replan(goal, step, result)
                if replan_result is not None:
                    # Re-planning succeeded — update chain and continue
                    chain = replan_result
                    goal.thought_chain = chain
                    continue
                else:
                    # Replan exhausted
                    break
            else:
                last_result = result

        # Determine final status
        if chain.is_complete:
            chain.status = "complete"
            # Validate with the validation agent
            validation = self._validate_goal(goal, last_result)
            goal.completed_at = time.time()
            goal.outcome = "completed"
            self.goal_memory.save_goal(goal)
            self._active_goals.pop(goal.id, None)
            self.replanner.reset(chain.goal)
            return {
                "goal_id": goal.id,
                "status": "completed",
                "result": last_result,
                "validation": validation,
            }
        else:
            chain.status = "failed"
            goal.outcome = "failed"
            self.goal_memory.save_goal(goal)
            self._active_goals.pop(goal.id, None)
            return {
                "goal_id": goal.id,
                "status": "failed",
                "result": last_result,
                "validation": None,
            }

    def get_active_goals(self) -> list[Goal]:
        """List currently active goals, ordered by priority."""
        goals = list(self._active_goals.values())
        goals.sort(key=lambda g: g.priority)
        return goals

    def get_goal_status(self, goal_id: str) -> dict:
        """Get status of a specific goal."""
        goal = self._active_goals.get(goal_id)
        if goal is None:
            goal = self.goal_memory.get_goal(goal_id)
        if goal is None:
            return {"goal_id": goal_id, "status": "not_found"}

        chain = goal.thought_chain
        if chain is None:
            return {"goal_id": goal_id, "status": "unplanned"}

        completed = sum(1 for s in chain.steps if s.status == "complete")
        total = len(chain.steps)
        return {
            "goal_id": goal_id,
            "status": chain.status,
            "progress": f"{completed}/{total}",
            "description": goal.description,
            "priority": goal.priority,
        }

    def cancel_goal(self, goal_id: str) -> bool:
        """Cancel an active goal. Returns True if successfully cancelled."""
        goal = self._active_goals.pop(goal_id, None)
        if goal is None:
            return False

        if goal.thought_chain:
            goal.thought_chain.status = "failed"
        goal.outcome = "cancelled"
        goal.completed_at = time.time()
        self.goal_memory.save_goal(goal)
        logger.info("Goal cancelled: %s", goal_id)
        return True

    # ── internal helpers ────────────────────────────────────────────

    def _delegate_step(self, step: ThoughtStep, goal: Goal) -> dict:
        """Send a step to the appropriate agent and collect the result."""
        agent = self.agents.get(step.agent)
        if agent is None:
            logger.error("No agent registered for %r", step.agent)
            step.status = "failed"
            step.result = {"error": f"No agent for {step.agent!r}"}
            return step.result

        message = AgentMessage(
            from_agent="meta",
            to_agent=step.agent,
            message_type="goal",
            content={
                "action": step.action,
                "action_spec": {
                    "type": step.agent,
                    "description": step.action,
                },
                "goal": goal.description,
                "query": step.action,
                "context": {
                    "reasoning": step.reasoning,
                    "expected_outcome": step.expected_outcome,
                },
                "result": {},
            },
            parent_goal_id=goal.id,
        )

        try:
            # Agents expose an async process() method — run it synchronously
            loop = asyncio.new_event_loop()
            try:
                response = loop.run_until_complete(agent.process(message))
            finally:
                loop.close()
            step.result = response.content
            step.status = "complete"
            return response.content
        except Exception as exc:
            logger.exception("Agent %s failed on step %s", step.agent, step.id)
            step.status = "failed"
            step.result = {"error": str(exc)}
            return step.result

    def _try_replan(
        self, goal: Goal, failed_step: ThoughtStep, error_result: dict
    ) -> Optional[ThoughtChain]:
        """Attempt to re-plan around a failed step. Returns new chain or None."""
        chain = goal.thought_chain
        if chain is None:
            return None

        error_str = str(error_result.get("error", "unknown error"))

        if not self.replanner.can_replan(chain):
            logger.warning(
                "Replan budget exhausted for goal %s", goal.id
            )
            return None

        try:
            new_chain = self.replanner.replan(chain, failed_step, error_str)
            new_chain.status = "executing"
            logger.info(
                "Re-planned goal %s: %d new steps",
                goal.id,
                len(new_chain.steps),
            )
            return new_chain
        except Exception:
            logger.exception("Re-planning failed for goal %s", goal.id)
            return None

    def _validate_goal(self, goal: Goal, result: Optional[dict]) -> Optional[dict]:
        """Run the validation agent against the final result."""
        validator = self.agents.get("validation")
        if validator is None:
            logger.warning("No validation agent registered — skipping validation")
            return None

        try:
            validation = validator.validate(
                goal=goal.description,
                result=result or {},
            )
            return validation
        except Exception:
            logger.exception("Validation failed for goal %s", goal.id)
            return None
