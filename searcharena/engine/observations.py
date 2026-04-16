"""Observation rendering."""

from __future__ import annotations
from ..models import Chunk, ChunkSummary, SearchObservation, SearchTask
from .rewards import TrajectoryTracker


def get_budget_warning(
    usage: float,
    soft_threshold: float,
    hard_threshold: float,
) -> str | None:
    """Get budget warning message based on usage."""
    if usage >= hard_threshold:
        return (
            f"HARD LIMIT: Context at {usage:.0%} capacity. "
            "Only prune or answer actions allowed."
        )
    elif usage >= soft_threshold:
        return (
            f"WARNING: Context at {usage:.0%} capacity. "
            "Consider pruning irrelevant chunks or submitting answer."
        )
    return None


def create_observation(
    current_task: "SearchTask | None",
    context_chunks: dict[str, "Chunk"],
    context_token_count: int,
    context_token_budget: int,
    snippet_length: int,
    soft_budget_threshold: float,
    hard_budget_threshold: float,
    step_count: int,
    max_steps: int,
    tracker: "TrajectoryTracker",
    chunks_seen: set[str],
    done: bool,
    action_result: dict | None = None,
    action_type: str | None = None,
    reward: float = 0.0,
) -> "SearchObservation":
    """Create observation from current environment state."""
    from ..models import ChunkSummary, SearchObservation

    # Create chunk summaries for context
    context_summaries = []
    for chunk in context_chunks.values():
        summary = ChunkSummary(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.metadata.get("title", chunk.document_id),
            snippet=chunk.content[:snippet_length] + "..."
            if len(chunk.content) > snippet_length
            else chunk.content,
            score=chunk.retrieval_score or 0.0,
            token_count=chunk.token_count,
        )
        context_summaries.append(summary)

    # Calculate budget usage
    if context_token_budget <= 0:
        budget_usage = 0.0
    else:
        budget_usage = context_token_count / context_token_budget

    return SearchObservation(
        question=current_task.question if current_task else "",
        context_chunks=context_summaries,
        context_token_count=context_token_count,
        context_token_budget=context_token_budget,
        budget_usage_percent=budget_usage * 100,
        budget_warning=get_budget_warning(budget_usage, soft_budget_threshold, hard_budget_threshold),
        action_result=action_result,
        action_type=action_type,
        step_count=step_count,
        max_steps=max_steps,
        queries_issued=list(tracker.queries),
        chunks_seen_count=len(chunks_seen),
        done=done,
        reward=reward,
    )
