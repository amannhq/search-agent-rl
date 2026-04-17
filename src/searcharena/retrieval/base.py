"""Stable retrieval interfaces."""

from __future__ import annotations

from typing import Protocol

from ..models import Chunk, ChunkSummary


class ChunkRetriever(Protocol):
    """Stable retrieval surface used by the environment."""

    def search(
        self,
        query: str,
        top_k: int = 10,
        exclude_ids: set[str] | None = None,
        snippet_length: int = 200,
    ) -> list[ChunkSummary]:
        """Search for chunk summaries."""
        ...

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Fetch one chunk."""
        ...

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch many chunks."""
        ...


class TokenCounter(Protocol):
    """Token estimation abstraction."""

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens for a string."""
        ...
