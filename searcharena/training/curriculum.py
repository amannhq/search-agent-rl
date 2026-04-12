"""Curriculum learning with adaptive difficulty."""

from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import EpisodeMetrics


class DifficultyLevel(IntEnum):
    """Difficulty levels for curriculum learning."""

    EASY = 0
    MEDIUM = 1
    HARD = 2

    @classmethod
    def from_string(cls, s: str) -> DifficultyLevel:
        """Convert string to DifficultyLevel."""
        mapping = {"easy": cls.EASY, "medium": cls.MEDIUM, "hard": cls.HARD}
        return mapping.get(s.lower(), cls.MEDIUM)

    def to_string(self) -> str:
        """Convert to string."""
        return ["easy", "medium", "hard"][self.value]


@dataclass
class CurriculumState:
    """Tracks curriculum learning state."""

    current_level: DifficultyLevel = DifficultyLevel.EASY
    steps_at_level: int = 0
    successes_at_level: int = 0
    attempts_at_level: int = 0
    level_history: list[tuple[int, DifficultyLevel]] = field(default_factory=list)

    @property
    def success_rate_at_level(self) -> float:
        """Current success rate at this difficulty."""
        if self.attempts_at_level == 0:
            return 0.0
        return self.successes_at_level / self.attempts_at_level


class CurriculumScheduler:
    """Advances difficulty based on success rate."""

    def __init__(
        self,
        warmup_steps: int = 1000,
        advance_threshold: float = 0.7,
        regress_threshold: float = 0.3,
        min_attempts_before_advance: int = 20,
        min_attempts_before_regress: int = 50,
        allow_regression: bool = True,
    ):
        """
        Initialize curriculum scheduler.

        Args:
            warmup_steps: Steps to stay at EASY before considering advancement
            advance_threshold: Success rate required to advance difficulty
            regress_threshold: Success rate below which to regress difficulty
            min_attempts_before_advance: Minimum attempts at level before advancing
            min_attempts_before_regress: Minimum attempts at level before regressing
            allow_regression: Whether to allow going back to easier levels
        """
        self.warmup_steps = warmup_steps
        self.advance_threshold = advance_threshold
        self.regress_threshold = regress_threshold
        self.min_attempts_before_advance = min_attempts_before_advance
        self.min_attempts_before_regress = min_attempts_before_regress
        self.allow_regression = allow_regression

        self._state = CurriculumState()
        self._global_step = 0

    @property
    def current_difficulty(self) -> str:
        """Get current difficulty as string."""
        return self._state.current_level.to_string()

    @property
    def current_level(self) -> DifficultyLevel:
        """Get current difficulty level."""
        return self._state.current_level

    def get_difficulty_weights(self) -> dict[str, float]:
        """
        Get sampling weights for each difficulty.

        Returns weights that favor the current curriculum level
        while still sampling some tasks from other levels.
        """
        level = self._state.current_level

        if level == DifficultyLevel.EASY:
            return {"easy": 0.8, "medium": 0.2, "hard": 0.0}
        elif level == DifficultyLevel.MEDIUM:
            return {"easy": 0.2, "medium": 0.6, "hard": 0.2}
        else:  # HARD
            return {"easy": 0.1, "medium": 0.3, "hard": 0.6}

    def record_episode(
        self,
        metrics: "EpisodeMetrics",
        step: int | None = None,
    ) -> None:
        """
        Record episode result and update curriculum state.

        Args:
            metrics: Metrics from the completed episode
            step: Current training step (optional)
        """
        if step is not None:
            self._global_step = step

        # Only track if episode matches current curriculum level
        episode_diff = DifficultyLevel.from_string(metrics.difficulty)
        if episode_diff != self._state.current_level:
            return

        self._state.steps_at_level += 1
        self._state.attempts_at_level += 1
        if metrics.success:
            self._state.successes_at_level += 1

    def maybe_advance(self) -> bool:
        """
        Check if we should advance difficulty level.

        Returns:
            True if difficulty was advanced
        """
        # Still in warmup at EASY
        if (
            self._state.current_level == DifficultyLevel.EASY
            and self._global_step < self.warmup_steps
        ):
            return False

        # Already at max difficulty
        if self._state.current_level == DifficultyLevel.HARD:
            return False

        # Not enough attempts
        if self._state.attempts_at_level < self.min_attempts_before_advance:
            return False

        # Check success rate
        if self._state.success_rate_at_level >= self.advance_threshold:
            self._advance_level()
            return True

        return False

    def maybe_regress(self) -> bool:
        """
        Check if we should regress difficulty level.

        Returns:
            True if difficulty was regressed
        """
        if not self.allow_regression:
            return False

        # Already at min difficulty
        if self._state.current_level == DifficultyLevel.EASY:
            return False

        # Not enough attempts
        if self._state.attempts_at_level < self.min_attempts_before_regress:
            return False

        # Check success rate
        if self._state.success_rate_at_level < self.regress_threshold:
            self._regress_level()
            return True

        return False

    def step(self, metrics: "EpisodeMetrics", step: int | None = None) -> str | None:
        """
        Full curriculum step: record, check advance/regress.

        Args:
            metrics: Episode metrics
            step: Current training step

        Returns:
            "advanced", "regressed", or None if no change
        """
        self.record_episode(metrics, step)

        if self.maybe_advance():
            return "advanced"
        if self.maybe_regress():
            return "regressed"
        return None

    def _advance_level(self) -> None:
        """Advance to next difficulty level."""
        old_level = self._state.current_level
        new_level = DifficultyLevel(min(old_level.value + 1, DifficultyLevel.HARD.value))

        self._state.level_history.append((self._global_step, old_level))
        self._state.current_level = new_level
        self._state.steps_at_level = 0
        self._state.successes_at_level = 0
        self._state.attempts_at_level = 0

    def _regress_level(self) -> None:
        """Regress to previous difficulty level."""
        old_level = self._state.current_level
        new_level = DifficultyLevel(max(old_level.value - 1, DifficultyLevel.EASY.value))

        self._state.level_history.append((self._global_step, old_level))
        self._state.current_level = new_level
        self._state.steps_at_level = 0
        self._state.successes_at_level = 0
        self._state.attempts_at_level = 0

    def reset(self) -> None:
        """Reset curriculum to initial state."""
        self._state = CurriculumState()
        self._global_step = 0

    def get_state(self) -> dict:
        """Get serializable state for checkpointing."""
        return {
            "current_level": self._state.current_level.value,
            "steps_at_level": self._state.steps_at_level,
            "successes_at_level": self._state.successes_at_level,
            "attempts_at_level": self._state.attempts_at_level,
            "level_history": self._state.level_history,
            "global_step": self._global_step,
        }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint."""
        self._state.current_level = DifficultyLevel(state["current_level"])
        self._state.steps_at_level = state["steps_at_level"]
        self._state.successes_at_level = state["successes_at_level"]
        self._state.attempts_at_level = state["attempts_at_level"]
        self._state.level_history = state["level_history"]
        self._global_step = state["global_step"]
