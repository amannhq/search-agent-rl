"""Action dispatching for SearchArena."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from ..models import ActionType, SearchAction, SearchEnvConfig
from ..retrieval import DocumentCorpus
from .state import EpisodeState


@dataclass
class DispatchResult:
    """Normalized result from a dispatched action."""

    action_result: dict[str, Any] = field(default_factory=dict)
    reward_delta: float = 0.0
    answer: str | None = None
    supporting_chunk_ids: list[str] = field(default_factory=list)


class ActionDispatcher:
    """Handles action branching and state mutation."""

    def __init__(
        self,
        *,
        corpus: DocumentCorpus,
        config: SearchEnvConfig,
    ) -> None:
        self.corpus = corpus
        self.config = config

    def dispatch(
        self,
        action: SearchAction,
        episode: EpisodeState,
    ) -> DispatchResult:
        """Apply one agent action to the episode."""
        if action.action_type == ActionType.SEARCH:
            return self._handle_search(action, episode)
        if action.action_type == ActionType.READ:
            return self._handle_read(action, episode)
        if action.action_type == ActionType.PRUNE:
            return self._handle_prune(action, episode)
        if action.action_type == ActionType.ANSWER:
            return self._handle_answer(action)

        return DispatchResult(action_result={"error": "Unsupported action type"})

    def _handle_search(
        self,
        action: SearchAction,
        episode: EpisodeState,
    ) -> DispatchResult:
        if action.search is None:
            return DispatchResult(action_result={"error": "Missing search payload"})

        exclude_ids = episode.chunks_seen if self.config.deduplicate_searches else None
        try:
            results = self.corpus.search(
                query=action.search.query,
                top_k=action.search.top_k,
                exclude_ids=exclude_ids,
                snippet_length=self.config.snippet_length,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return DispatchResult(action_result={"error": str(exc)})

        chunk_ids = [result.chunk_id for result in results]
        episode.tracker.record_search(action.search.query, chunk_ids)
        episode.chunks_seen.update(chunk_ids)
        episode.seen_texts.extend(
            result.snippet for result in results if result.snippet
        )

        return DispatchResult(
            action_result={
                "query": action.search.query,
                "results": [result.model_dump() for result in results],
                "total_found": len(results),
            }
        )

    def _handle_read(
        self,
        action: SearchAction,
        episode: EpisodeState,
    ) -> DispatchResult:
        if action.read is None:
            return DispatchResult(action_result={"error": "Missing read payload"})

        chunks_added = []
        tokens_added = 0
        budget_exceeded = False
        chunks_truncated = 0
        remaining_budget = self.config.max_context_tokens - episode.context_token_count

        for chunk_id in action.read.chunk_ids:
            if chunk_id in episode.context_chunks:
                continue

            try:
                chunk = self.corpus.get_chunk(chunk_id)
            except Exception as exc:  # pragma: no cover - defensive
                return DispatchResult(action_result={"error": str(exc)})

            if chunk is None:
                continue

            if tokens_added + chunk.token_count > remaining_budget:
                budget_exceeded = True
                chunks_truncated += 1
                continue

            episode.context_chunks[chunk_id] = chunk
            episode.context_token_count += chunk.token_count
            tokens_added += chunk.token_count
            chunks_added.append(chunk)

        episode.tracker.record_read([chunk.chunk_id for chunk in chunks_added])
        episode.chunks_seen.update(action.read.chunk_ids)

        return DispatchResult(
            action_result={
                "chunks": [chunk.model_dump() for chunk in chunks_added],
                "tokens_added": tokens_added,
                "budget_exceeded": budget_exceeded,
                "chunks_truncated": chunks_truncated,
            }
        )

    def _handle_prune(
        self,
        action: SearchAction,
        episode: EpisodeState,
    ) -> DispatchResult:
        if action.prune is None:
            return DispatchResult(action_result={"error": "Missing prune payload"})

        chunks_removed = 0
        tokens_freed = 0
        invalid_ids: list[str] = []

        for chunk_id in action.prune.chunk_ids:
            if chunk_id in episode.context_chunks:
                chunk = episode.context_chunks.pop(chunk_id)
                episode.context_token_count -= chunk.token_count
                tokens_freed += chunk.token_count
                chunks_removed += 1
            else:
                invalid_ids.append(chunk_id)

        episode.tracker.record_prune(action.prune.chunk_ids)

        return DispatchResult(
            action_result={
                "chunks_removed": chunks_removed,
                "tokens_freed": tokens_freed,
                "invalid_ids": invalid_ids,
            }
        )

    def _handle_answer(self, action: SearchAction) -> DispatchResult:
        if action.answer is None:
            return DispatchResult(action_result={"error": "Missing answer payload"})

        return DispatchResult(
            answer=action.answer.answer,
            supporting_chunk_ids=list(action.answer.supporting_chunk_ids),
        )

