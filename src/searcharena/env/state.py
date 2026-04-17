"""Episode state primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from uuid import uuid4
from openenv.core.env_server.types import State
from ..models import Chunk, SearchTask
from ..rewards import RewardMetrics, TrajectoryTracker


@dataclass
class EpisodeState:
    """All mutable state for one environment episode."""

    state: State = field(
        default_factory=lambda: State(episode_id=str(uuid4()), step_count=0)
    )
    current_task: SearchTask | None = None
    tracker: TrajectoryTracker = field(default_factory=TrajectoryTracker)
    context_chunks: dict[str, Chunk] = field(default_factory=dict)
    context_token_count: int = 0
    chunks_seen: set[str] = field(default_factory=set)
    seen_texts: list[str] = field(default_factory=list)
    done: bool = False
    last_metrics: RewardMetrics | None = None
    rng_seed: int | None = None
    rng: Random = field(default_factory=Random)
    terminated: bool = False
    truncated: bool = False
    termination_reason: str | None = None

    @classmethod
    def create(
        cls,
        *,
        episode_id: str | None = None,
        seed: int | None = None,
    ) -> EpisodeState:
        """Create a fresh episode state."""
        runtime_state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        rng = Random(seed)
        return cls(
            state=runtime_state,
            rng_seed=seed,
            rng=rng,
        )

