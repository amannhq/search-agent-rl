"""Episode trajectory tracking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryTracker:
    """Tracks chunks and queries encountered during an episode."""

    chunks_seen: set[str] = field(default_factory=set)
    chunks_in_context: set[str] = field(default_factory=set)
    queries: list[str] = field(default_factory=list)
    consecutive_prunes: int = 0
    total_prunes: int = 0

    def record_search(self, query: str, chunk_ids: list[str]) -> None:
        self.queries.append(query)
        self.chunks_seen.update(chunk_ids)
        self.consecutive_prunes = 0

    def record_read(self, chunk_ids: list[str]) -> None:
        self.chunks_in_context.update(chunk_ids)
        self.chunks_seen.update(chunk_ids)
        self.consecutive_prunes = 0

    def record_prune(self, chunk_ids: list[str]) -> None:
        self.chunks_in_context -= set(chunk_ids)
        self.consecutive_prunes += 1
        self.total_prunes += 1

    def reset(self) -> None:
        self.chunks_seen.clear()
        self.chunks_in_context.clear()
        self.queries.clear()
        self.consecutive_prunes = 0
        self.total_prunes = 0
