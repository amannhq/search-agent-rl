"""Task sampling, episode buffering, dataset construction."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Any

if TYPE_CHECKING:
    from ..models import SearchTask, SearchObservation, SearchAction


@dataclass
class Episode:
    """A complete training episode."""

    task_id: str
    observations: list["SearchObservation"] = field(default_factory=list)
    actions: list["SearchAction"] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False

    # Metadata
    difficulty: str = "medium"
    domain: str = "general"
    num_steps: int = 0

    def add_step(
        self,
        observation: "SearchObservation",
        action: "SearchAction",
        reward: float,
    ) -> None:
        """Add a step to the episode."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.num_steps += 1

    def finalize(self, final_reward: float, success: bool) -> None:
        """Mark episode as complete."""
        self.total_reward = final_reward
        self.success = success

    def get_trajectory(self) -> list[tuple["SearchObservation", "SearchAction", float]]:
        """Get the full trajectory as (obs, action, reward) tuples."""
        return list(zip(self.observations, self.actions, self.rewards))


class TaskSampler:
    """Samples tasks with stratification by difficulty/domain."""

    def __init__(
        self,
        tasks: list["SearchTask"],
        seed: int | None = None,
    ):
        self.tasks = tasks
        self.rng = random.Random(seed)

        # Index by difficulty and domain
        self._by_difficulty: dict[str, list["SearchTask"]] = {}
        self._by_domain: dict[str, list["SearchTask"]] = {}

        for task in tasks:
            diff = task.difficulty
            if diff not in self._by_difficulty:
                self._by_difficulty[diff] = []
            self._by_difficulty[diff].append(task)

            domain = task.domain
            if domain not in self._by_domain:
                self._by_domain[domain] = []
            self._by_domain[domain].append(task)

    def sample(self, n: int = 1) -> list["SearchTask"]:
        """Sample n tasks uniformly."""
        return self.rng.choices(self.tasks, k=n)

    def sample_by_difficulty(
        self,
        n: int = 1,
        weights: dict[str, float] | None = None,
    ) -> list["SearchTask"]:
        """
        Sample tasks with difficulty-based weights.

        Args:
            n: Number of tasks to sample
            weights: Difficulty -> weight mapping (e.g., {"easy": 0.5, "medium": 0.3, "hard": 0.2})
        """
        if weights is None:
            weights = {"easy": 1.0, "medium": 1.0, "hard": 1.0}

        # Build weighted task list
        weighted_tasks: list[tuple["SearchTask", float]] = []
        for task in self.tasks:
            w = weights.get(task.difficulty, 1.0)
            weighted_tasks.append((task, w))

        tasks, task_weights = zip(*weighted_tasks)
        return self.rng.choices(list(tasks), weights=list(task_weights), k=n)

    def sample_stratified(
        self,
        n_per_difficulty: int = 1,
        difficulties: list[str] | None = None,
    ) -> list["SearchTask"]:
        """Sample equal numbers from each difficulty."""
        difficulties = difficulties or list(self._by_difficulty.keys())

        sampled = []
        for diff in difficulties:
            if diff in self._by_difficulty:
                pool = self._by_difficulty[diff]
                sampled.extend(self.rng.choices(pool, k=min(n_per_difficulty, len(pool))))

        return sampled

    def sample_from_difficulty(
        self,
        difficulty: str,
        n: int = 1,
    ) -> list["SearchTask"]:
        """Sample tasks from a specific difficulty."""
        pool = self._by_difficulty.get(difficulty, [])
        if not pool:
            return []
        return self.rng.choices(pool, k=min(n, len(pool)))

    def sample_from_domain(
        self,
        domain: str,
        n: int = 1,
    ) -> list["SearchTask"]:
        """Sample tasks from a specific domain."""
        pool = self._by_domain.get(domain, [])
        if not pool:
            return []
        return self.rng.choices(pool, k=min(n, len(pool)))

    @property
    def difficulties(self) -> list[str]:
        """Get available difficulties."""
        return list(self._by_difficulty.keys())

    @property
    def domains(self) -> list[str]:
        """Get available domains."""
        return list(self._by_domain.keys())


class EpisodeBuffer:
    """Circular buffer for training episodes."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: deque[Episode] = deque(maxlen=max_size)
        self._success_count = 0
        self._total_reward = 0.0

    def add(self, episode: Episode) -> None:
        """Add an episode to the buffer."""
        if len(self._buffer) == self.max_size:
            # Track stats of removed episode
            old = self._buffer[0]
            if old.success:
                self._success_count -= 1
            self._total_reward -= old.total_reward

        self._buffer.append(episode)
        if episode.success:
            self._success_count += 1
        self._total_reward += episode.total_reward

    def sample(self, n: int = 1) -> list[Episode]:
        """Sample n episodes uniformly."""
        if len(self._buffer) < n:
            return list(self._buffer)
        return random.sample(list(self._buffer), n)

    def sample_successful(self, n: int = 1) -> list[Episode]:
        """Sample only successful episodes."""
        successful = [e for e in self._buffer if e.success]
        if len(successful) < n:
            return successful
        return random.sample(successful, n)

    def get_recent(self, n: int = 10) -> list[Episode]:
        """Get n most recent episodes."""
        return list(self._buffer)[-n:]

    @property
    def size(self) -> int:
        """Current buffer size."""
        return len(self._buffer)

    @property
    def success_rate(self) -> float:
        """Success rate of buffered episodes."""
        if not self._buffer:
            return 0.0
        return self._success_count / len(self._buffer)

    @property
    def mean_reward(self) -> float:
        """Mean reward of buffered episodes."""
        if not self._buffer:
            return 0.0
        return self._total_reward / len(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._success_count = 0
        self._total_reward = 0.0


class DatasetBuilder:
    """Converts episodes to training examples."""

    def __init__(
        self,
        include_failed: bool = True,
        normalize_rewards: bool = True,
    ):
        self.include_failed = include_failed
        self.normalize_rewards = normalize_rewards

    def build_from_episodes(
        self,
        episodes: list[Episode],
    ) -> list[dict[str, Any]]:
        """
        Convert episodes to training examples.

        Each example contains:
        - observation: The state at each step
        - action: The action taken
        - reward: The reward received
        - return: Discounted return from this step
        """
        examples = []

        for episode in episodes:
            if not self.include_failed and not episode.success:
                continue

            # Compute returns (simple sum for now, can add discounting)
            returns = self._compute_returns(episode.rewards)

            for i, (obs, action, reward) in enumerate(episode.get_trajectory()):
                examples.append({
                    "task_id": episode.task_id,
                    "step": i,
                    "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
                    "action": action.model_dump() if hasattr(action, "model_dump") else action,
                    "reward": reward,
                    "return": returns[i],
                    "episode_success": episode.success,
                    "episode_reward": episode.total_reward,
                })

        if self.normalize_rewards and examples:
            self._normalize(examples)

        return examples

    def _compute_returns(
        self,
        rewards: list[float],
        gamma: float = 0.99,
    ) -> list[float]:
        """Compute discounted returns from rewards."""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns

    def _normalize(self, examples: list[dict[str, Any]]) -> None:
        """Normalize returns in-place."""
        returns = [e["return"] for e in examples]
        mean = sum(returns) / len(returns)
        std = (sum((r - mean) ** 2 for r in returns) / len(returns)) ** 0.5

        if std > 0:
            for e in examples:
                e["return_normalized"] = (e["return"] - mean) / std
        else:
            for e in examples:
                e["return_normalized"] = 0.0

    def iterate_batches(
        self,
        examples: list[dict[str, Any]],
        batch_size: int = 8,
        shuffle: bool = True,
    ) -> Iterator[list[dict[str, Any]]]:
        """Iterate over batches of examples."""
        if shuffle:
            examples = examples.copy()
            random.shuffle(examples)

        for i in range(0, len(examples), batch_size):
            yield examples[i:i + batch_size]
