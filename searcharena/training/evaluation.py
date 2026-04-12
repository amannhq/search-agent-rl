"""Evaluation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import SearchTask
    from ..engine import SearchEnvironment
    from .metrics import EpisodeMetrics


@dataclass
class EvaluationResult:
    """Complete evaluation results."""

    # Summary metrics
    total_episodes: int = 0
    successful_episodes: int = 0
    mean_reward: float = 0.0
    std_reward: float = 0.0

    # Per-level breakdown
    metrics_by_level: dict[int, dict[str, float]] = field(default_factory=dict)

    # Per-domain breakdown
    metrics_by_domain: dict[str, dict[str, float]] = field(default_factory=dict)

    # Detailed episode results
    episode_results: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    model_name: str = ""
    eval_timestamp: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        if self.total_episodes == 0:
            return 0.0
        return self.successful_episodes / self.total_episodes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary": {
                "total_episodes": self.total_episodes,
                "successful_episodes": self.successful_episodes,
                "success_rate": self.success_rate,
                "mean_reward": self.mean_reward,
                "std_reward": self.std_reward,
            },
            "by_level": self.metrics_by_level,
            "by_domain": self.metrics_by_domain,
            "metadata": {
                "model_name": self.model_name,
                "timestamp": self.eval_timestamp,
                "config": self.config,
            },
            "episodes": self.episode_results if self.episode_results else None,
        }

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> EvaluationResult:
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            total_episodes=data["summary"]["total_episodes"],
            successful_episodes=data["summary"]["successful_episodes"],
            mean_reward=data["summary"]["mean_reward"],
            std_reward=data["summary"]["std_reward"],
            metrics_by_level=data.get("by_level", {}),
            metrics_by_domain=data.get("by_domain", {}),
            episode_results=data.get("episodes", []),
            model_name=data["metadata"].get("model_name", ""),
            eval_timestamp=data["metadata"].get("timestamp", ""),
            config=data["metadata"].get("config", {}),
        )

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nOverall: {self.successful_episodes}/{self.total_episodes} "
              f"({100*self.success_rate:.1f}% success)")
        print(f"Mean Reward: {self.mean_reward:.3f} +/- {self.std_reward:.3f}")

        if self.metrics_by_level:
            print("\nBy Level:")
            for level, metrics in sorted(self.metrics_by_level.items()):
                sr = metrics.get("success_rate", 0) * 100
                mr = metrics.get("mean_reward", 0)
                n = metrics.get("count", 0)
                print(f"  {level:3d}: {sr:5.1f}% success, {mr:.3f} reward ({n} episodes)")

        if self.metrics_by_domain:
            print("\nBy Domain:")
            for domain, metrics in sorted(self.metrics_by_domain.items()):
                sr = metrics.get("success_rate", 0) * 100
                mr = metrics.get("mean_reward", 0)
                n = metrics.get("count", 0)
                print(f"  {domain:12s}: {sr:5.1f}% success, {mr:.3f} reward ({n} episodes)")

        print("=" * 60 + "\n")


class Evaluator:
    """Runs evaluation with per-level/domain metrics."""

    def __init__(
        self,
        tasks: list["SearchTask"],
        env: "SearchEnvironment | None" = None,
        save_trajectories: bool = False,
    ):
        self.tasks = tasks
        self.env = env
        self.save_trajectories = save_trajectories

        # Index tasks
        self._by_level: dict[int, list["SearchTask"]] = {}
        self._by_domain: dict[str, list["SearchTask"]] = {}

        for task in tasks:
            if task.level not in self._by_level:
                self._by_level[task.level] = []
            self._by_level[task.level].append(task)

            if task.domain not in self._by_domain:
                self._by_domain[task.domain] = []
            self._by_domain[task.domain].append(task)

    def evaluate_episode(
        self,
        task: "SearchTask",
        trajectory: list[tuple[Any, Any, float]],
        final_metrics: "EpisodeMetrics",
    ) -> dict[str, Any]:
        """
        Evaluate a single episode and return metrics.

        Args:
            task: The task that was evaluated
            trajectory: List of (observation, action, reward) tuples
            final_metrics: Metrics from the episode

        Returns:
            Dictionary of evaluation metrics
        """
        result = {
            "task_id": task.task_id,
            "level": task.level,
            "domain": task.domain,
            "success": final_metrics.success,
            "total_reward": final_metrics.total_reward,
            "num_steps": final_metrics.num_steps,
            "f_beta": final_metrics.f_beta,
            "trajectory_recall": final_metrics.trajectory_recall,
            "output_recall": final_metrics.output_recall,
            "output_precision": final_metrics.output_precision,
            "answer_correct": final_metrics.answer_correct,
        }

        if self.save_trajectories:
            result["trajectory"] = [
                {
                    "step": i,
                    "action": str(action),
                    "reward": reward,
                }
                for i, (_, action, reward) in enumerate(trajectory)
            ]

        return result

    def aggregate_results(
        self,
        episode_results: list[dict[str, Any]],
        model_name: str = "",
    ) -> EvaluationResult:
        """
        Aggregate episode results into final evaluation.

        Args:
            episode_results: List of per-episode result dicts
            model_name: Name of the evaluated model

        Returns:
            Complete EvaluationResult
        """
        import datetime

        if not episode_results:
            return EvaluationResult()

        # Basic aggregation
        rewards = [e["total_reward"] for e in episode_results]
        mean_reward = sum(rewards) / len(rewards)
        std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5

        # By level
        by_level: dict[int, dict[str, float]] = {}
        for level in self._by_level.keys():
            level_episodes = [e for e in episode_results if e["level"] == level]
            if level_episodes:
                level_rewards = [e["total_reward"] for e in level_episodes]
                by_level[level] = {
                    "count": len(level_episodes),
                    "success_rate": sum(1 for e in level_episodes if e["success"]) / len(level_episodes),
                    "mean_reward": sum(level_rewards) / len(level_rewards),
                    "mean_f_beta": sum(e["f_beta"] for e in level_episodes) / len(level_episodes),
                }

        # By domain
        by_domain: dict[str, dict[str, float]] = {}
        for domain in self._by_domain.keys():
            domain_episodes = [e for e in episode_results if e["domain"] == domain]
            if domain_episodes:
                domain_rewards = [e["total_reward"] for e in domain_episodes]
                by_domain[domain] = {
                    "count": len(domain_episodes),
                    "success_rate": sum(1 for e in domain_episodes if e["success"]) / len(domain_episodes),
                    "mean_reward": sum(domain_rewards) / len(domain_rewards),
                }

        return EvaluationResult(
            total_episodes=len(episode_results),
            successful_episodes=sum(1 for e in episode_results if e["success"]),
            mean_reward=mean_reward,
            std_reward=std_reward,
            metrics_by_level=by_level,
            metrics_by_domain=by_domain,
            episode_results=episode_results if self.save_trajectories else [],
            model_name=model_name,
            eval_timestamp=datetime.datetime.now().isoformat(),
        )

    def get_tasks_by_level(
        self,
        level: int,
        n: int | None = None,
    ) -> list["SearchTask"]:
        """Get tasks of a specific level."""
        tasks = self._by_level.get(level, [])
        if n is not None:
            return tasks[:n]
        return tasks

    def get_tasks_by_domain(
        self,
        domain: str,
        n: int | None = None,
    ) -> list["SearchTask"]:
        """Get tasks of a specific domain."""
        tasks = self._by_domain.get(domain, [])
        if n is not None:
            return tasks[:n]
        return tasks
