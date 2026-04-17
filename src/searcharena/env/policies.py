"""Environment policies for budgets and termination."""

from __future__ import annotations
from dataclasses import dataclass
from ..models import ActionType

@dataclass
class TerminationInfo:
    """Normalized termination metadata for one step."""

    reason: str | None = None
    truncated: bool = False
    final_reward_applied: bool = False


class BudgetPolicy:
    """Applies budget thresholds to environment actions."""

    def __init__(
        self,
        *,
        max_context_tokens: int,
        hard_budget_threshold: float,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.hard_budget_threshold = hard_budget_threshold

    def usage(self, context_token_count: int) -> float:
        """Return budget usage in [0, 1]."""
        if self.max_context_tokens <= 0:
            return 0.0
        return context_token_count / self.max_context_tokens

    def blocks(self, action_type: ActionType, context_token_count: int) -> bool:
        """Return whether a non-terminal action is blocked by budget."""
        if action_type in {ActionType.PRUNE, ActionType.ANSWER}:
            return False
        return self.usage(context_token_count) >= self.hard_budget_threshold

    def blocked_result(self) -> dict[str, str]:
        """Standard payload for budget-blocked actions."""
        return {"error": "Token budget exceeded. Only prune or answer allowed."}


class TerminationPolicy:
    """Applies step-based termination rules."""

    def __init__(self, *, max_steps: int) -> None:
        self.max_steps = max_steps

    def after_step(self, step_count: int) -> TerminationInfo | None:
        """Return terminal metadata after a step if the limit is hit."""
        if step_count >= self.max_steps:
            return TerminationInfo(
                reason="max_steps",
                truncated=True,
                final_reward_applied=False,
            )
        return None

