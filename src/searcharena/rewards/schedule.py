"""Reward parameter schedules."""

from __future__ import annotations


class BetaScheduler:
    """Schedule beta for curriculum learning."""

    def __init__(
        self,
        start_beta: float = 4.0,
        end_beta: float = 2.0,
        warmup_steps: int = 1000,
        decay_steps: int = 10000,
    ) -> None:
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def get_beta(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.start_beta

        decay_progress = min(1.0, (step - self.warmup_steps) / self.decay_steps)
        return self.start_beta + (self.end_beta - self.start_beta) * decay_progress
