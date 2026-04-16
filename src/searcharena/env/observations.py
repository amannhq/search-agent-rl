"""Observation rendering for SearchArena."""

from __future__ import annotations

from ..models import ChunkSummary, SearchEnvConfig, SearchObservation
from .state import EpisodeState


def get_budget_warning(
    usage: float,
    soft_threshold: float,
    hard_threshold: float,
) -> str | None:
    """Return a user-facing budget warning."""
    if usage >= hard_threshold:
        return (
            f"HARD LIMIT: Context at {usage:.0%} capacity. "
            "Only prune or answer actions allowed."
        )
    if usage >= soft_threshold:
        return (
            f"WARNING: Context at {usage:.0%} capacity. "
            "Consider pruning irrelevant chunks or submitting answer."
        )
    return None


def render_observation(
    *,
    episode: EpisodeState,
    config: SearchEnvConfig,
    action_result: dict | None = None,
    action_type: str | None = None,
    reward: float = 0.0,
) -> SearchObservation:
    """Render an agent-visible observation from internal episode state."""
    context_summaries = [
        ChunkSummary(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.metadata.get("title", chunk.document_id),
            snippet=(
                chunk.content[: config.snippet_length] + "..."
                if len(chunk.content) > config.snippet_length
                else chunk.content
            ),
            score=chunk.retrieval_score or 0.0,
            token_count=chunk.token_count,
        )
        for chunk in episode.context_chunks.values()
    ]

    if config.max_context_tokens <= 0:
        budget_usage = 0.0
    else:
        budget_usage = episode.context_token_count / config.max_context_tokens

    return SearchObservation(
        question=episode.current_task.question if episode.current_task else "",
        context_chunks=context_summaries,
        context_token_count=episode.context_token_count,
        context_token_budget=config.max_context_tokens,
        budget_usage_percent=budget_usage * 100,
        budget_warning=get_budget_warning(
            budget_usage,
            config.soft_budget_threshold,
            config.hard_budget_threshold,
        ),
        action_result=action_result,
        action_type=action_type,
        step_count=episode.state.step_count,
        max_steps=config.max_steps,
        terminated=episode.terminated,
        truncated=episode.truncated,
        termination_reason=episode.termination_reason,
        queries_issued=list(episode.tracker.queries),
        chunks_seen_count=len(episode.chunks_seen),
        done=episode.done,
        reward=reward,
    )

