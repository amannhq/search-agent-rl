"""Reranker hooks for retrieval pipelines."""

from __future__ import annotations

from typing import Protocol

from ..models import ChunkSummary


class ChunkReranker(Protocol):
    """Optional reranker for retrieved chunk summaries."""

    def rerank(self, query: str, results: list[ChunkSummary]) -> list[ChunkSummary]:
        """Return reranked results."""


class NoOpReranker:
    """Default reranker that preserves input order."""

    def rerank(self, query: str, results: list[ChunkSummary]) -> list[ChunkSummary]:
        _ = query
        return results
