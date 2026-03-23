"""Neural Reasoning Engine configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NeuralConfig:
    """Configuration for the Neural Reasoning Engine.

    Attributes
    ----------
    enabled:
        Master switch for the neural reasoning engine.
    autonomy_level:
        Controls how autonomous the engine behaves:
        - ``"full"``: goals execute end-to-end without human approval.
        - ``"supervised"``: pauses before action-agent steps for confirmation.
        - ``"assisted"``: plans are generated but never auto-executed.
    max_concurrent_goals:
        Upper limit on goals executing in parallel.
    max_recursion_depth:
        Maximum depth for hierarchical sub-goal decomposition.
    max_replan_attempts:
        How many times the replanner can retry a failed goal before giving up.
    validation_threshold:
        Minimum validation score (0-1) for a goal result to be accepted.
    proactive_enabled:
        Whether the proactive trigger engine is active.
    proactive_check_interval:
        Seconds between proactive trigger checks.
    default_planning_strategy:
        Fallback planning strategy when complexity classification fails.
    """

    enabled: bool = True
    autonomy_level: str = "full"
    max_concurrent_goals: int = 5
    max_recursion_depth: int = 10
    max_replan_attempts: int = 3
    validation_threshold: float = 0.7
    proactive_enabled: bool = True
    proactive_check_interval: float = 60.0
    default_planning_strategy: str = "hierarchical"

    def __post_init__(self) -> None:
        valid_levels = {"full", "supervised", "assisted"}
        if self.autonomy_level not in valid_levels:
            raise ValueError(
                f"autonomy_level must be one of {valid_levels}, "
                f"got {self.autonomy_level!r}"
            )
