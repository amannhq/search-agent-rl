"""Metrics collection and logging."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    task_id: str
    level: int = 0
    domain: str = "general"

    # Core metrics
    total_reward: float = 0.0
    success: bool = False
    num_steps: int = 0

    # Search metrics
    num_searches: int = 0
    num_reads: int = 0
    num_prunes: int = 0
    unique_chunks_seen: int = 0

    # Reward components
    f_beta: float = 0.0
    trajectory_recall: float = 0.0
    output_recall: float = 0.0
    output_precision: float = 0.0
    answer_correct: bool = False

    # Efficiency
    tokens_used: int = 0
    budget_utilization: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "level": self.level,
            "domain": self.domain,
            "total_reward": self.total_reward,
            "success": self.success,
            "num_steps": self.num_steps,
            "num_searches": self.num_searches,
            "num_reads": self.num_reads,
            "num_prunes": self.num_prunes,
            "unique_chunks_seen": self.unique_chunks_seen,
            "f_beta": self.f_beta,
            "trajectory_recall": self.trajectory_recall,
            "output_recall": self.output_recall,
            "output_precision": self.output_precision,
            "answer_correct": self.answer_correct,
            "tokens_used": self.tokens_used,
            "budget_utilization": self.budget_utilization,
        }


@dataclass
class TrainingMetrics:
    """Aggregated training metrics over multiple episodes."""

    step: int = 0
    timestamp: float = field(default_factory=time.time)

    # Episode counts
    episodes_total: int = 0
    episodes_successful: int = 0

    # Reward statistics
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0

    # Success rates by level
    success_rate_by_level: dict[int, float] = field(default_factory=dict)

    # Efficiency metrics
    avg_steps: float = 0.0
    avg_searches: float = 0.0
    avg_budget_utilization: float = 0.0

    # Learning metrics
    loss: float | None = None
    policy_loss: float | None = None
    value_loss: float | None = None
    entropy: float | None = None
    kl_divergence: float | None = None
    learning_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        data = {
            "step": self.step,
            "timestamp": self.timestamp,
            "episodes_total": self.episodes_total,
            "episodes_successful": self.episodes_successful,
            "success_rate": self.episodes_successful / max(1, self.episodes_total),
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "success_rate_by_level": self.success_rate_by_level,
            "avg_steps": self.avg_steps,
            "avg_searches": self.avg_searches,
            "avg_budget_utilization": self.avg_budget_utilization,
        }

        # Add optional learning metrics
        if self.loss is not None:
            data["loss"] = self.loss
        if self.policy_loss is not None:
            data["policy_loss"] = self.policy_loss
        if self.value_loss is not None:
            data["value_loss"] = self.value_loss
        if self.entropy is not None:
            data["entropy"] = self.entropy
        if self.kl_divergence is not None:
            data["kl_divergence"] = self.kl_divergence
        if self.learning_rate is not None:
            data["learning_rate"] = self.learning_rate

        return data

    @classmethod
    def from_episodes(
        cls,
        episodes: list[EpisodeMetrics],
        step: int = 0,
    ) -> TrainingMetrics:
        """Compute aggregated metrics from episode metrics."""
        if not episodes:
            return cls(step=step)

        rewards = [e.total_reward for e in episodes]
        n = len(episodes)

        # Basic stats
        reward_mean = sum(rewards) / n
        reward_std = (sum((r - reward_mean) ** 2 for r in rewards) / n) ** 0.5

        # Success rates by level
        by_level: dict[int, list[EpisodeMetrics]] = {}
        for e in episodes:
            if e.level not in by_level:
                by_level[e.level] = []
            by_level[e.level].append(e)

        success_rate_by_level = {
            level: sum(1 for e in eps if e.success) / max(1, len(eps))
            for level, eps in by_level.items()
        }

        return cls(
            step=step,
            episodes_total=n,
            episodes_successful=sum(1 for e in episodes if e.success),
            reward_mean=reward_mean,
            reward_std=reward_std,
            reward_min=min(rewards),
            reward_max=max(rewards),
            success_rate_by_level=success_rate_by_level,
            avg_steps=sum(e.num_steps for e in episodes) / n,
            avg_searches=sum(e.num_searches for e in episodes) / n,
            avg_budget_utilization=sum(e.budget_utilization for e in episodes) / n,
        )


class MetricsLogger:
    """Logs metrics to file, console, and optionally W&B."""

    def __init__(
        self,
        output_dir: str | Path = "./logs",
        run_name: str = "training",
        use_wandb: bool = False,
        wandb_project: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.use_wandb = use_wandb

        # Setup file logging
        self.metrics_file = self.output_dir / f"{run_name}_metrics.jsonl"
        self.episodes_file = self.output_dir / f"{run_name}_episodes.jsonl"

        # Setup wandb if requested
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                if wandb_project:
                    wandb.init(project=wandb_project, name=run_name)
            except ImportError:
                print("Warning: wandb not installed, disabling W&B logging")
                self.use_wandb = False

        self._step = 0
        self._start_time = time.time()

    def log_episode(self, metrics: EpisodeMetrics) -> None:
        """Log metrics for a single episode."""
        data = metrics.to_dict()
        data["logged_at"] = time.time()

        with open(self.episodes_file, "a") as f:
            f.write(json.dumps(data) + "\n")

    def log_step(self, metrics: TrainingMetrics) -> None:
        """Log aggregated metrics for a training step."""
        self._step = metrics.step
        data = metrics.to_dict()
        data["elapsed_time"] = time.time() - self._start_time

        # File logging
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(data) + "\n")

        # Wandb logging
        if self._wandb and self.use_wandb:
            self._wandb.log(data, step=metrics.step)

    def log_scalar(self, name: str, value: float, step: int | None = None) -> None:
        """Log a single scalar value."""
        step = step or self._step
        data = {"step": step, "name": name, "value": value, "timestamp": time.time()}

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(data) + "\n")

        if self._wandb and self.use_wandb:
            self._wandb.log({name: value}, step=step)

    def log_histogram(self, name: str, values: list[float], step: int | None = None) -> None:
        """Log a histogram of values."""
        step = step or self._step

        if self._wandb and self.use_wandb:
            import wandb
            self._wandb.log({name: wandb.Histogram(values)}, step=step)

    def print_summary(self, metrics: TrainingMetrics) -> None:
        """Print a formatted summary to console."""
        elapsed = time.time() - self._start_time
        eps = metrics.episodes_total / max(1, elapsed)

        print(f"\n{'='*60}")
        print(f"Step {metrics.step:,} | Time: {elapsed:.1f}s | {eps:.2f} ep/s")
        print(f"{'='*60}")
        print(f"Reward: {metrics.reward_mean:.3f} +/- {metrics.reward_std:.3f}")
        print(f"Success Rate: {metrics.episodes_successful}/{metrics.episodes_total} "
              f"({100*metrics.episodes_successful/max(1,metrics.episodes_total):.1f}%)")
        if metrics.success_rate_by_level:
            level_strs = [f"L{level}: {100*rate:.1f}%" for level, rate in sorted(metrics.success_rate_by_level.items())]
            print(f"  {' | '.join(level_strs)}")
        print(f"Avg Steps: {metrics.avg_steps:.1f} | "
              f"Avg Searches: {metrics.avg_searches:.1f}")

        if metrics.loss is not None:
            print(f"Loss: {metrics.loss:.4f}")
        print(f"{'='*60}\n")

    def close(self) -> None:
        """Finalize logging."""
        if self._wandb and self.use_wandb:
            self._wandb.finish()
