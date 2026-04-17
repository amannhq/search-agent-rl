"""Logging utilities for submission format."""

from __future__ import annotations
from .action import clean


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start (mandatory submission format)."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    """Log step result (mandatory submission format)."""
    print(
        f"[STEP] step={step} action={clean(action)[:120]} "
        f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """Log episode end (mandatory submission format)."""
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )
