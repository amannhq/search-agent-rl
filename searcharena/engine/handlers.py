"""Action handlers for search, read, prune, answer."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Chunk
    from .rewards import RewardCalculator, RewardMetrics, TrajectoryTracker
    from .retrieval import DocumentCorpus


def handle_search(
    query: str,
    top_k: int,
    corpus: "DocumentCorpus",
    tracker: "TrajectoryTracker",
    chunks_seen: set[str],
    seen_texts: list[str],
    deduplicate: bool,
    snippet_length: int,
) -> dict[str, Any]:
    """Handle search action."""
    exclude_ids = chunks_seen if deduplicate else None

    try:
        results = corpus.search(
            query=query,
            top_k=top_k,
            exclude_ids=exclude_ids,
            snippet_length=snippet_length,
        )
    except Exception as exc:
        return {"error": str(exc)}

    chunk_ids = [r.chunk_id for r in results]
    tracker.record_search(query, chunk_ids)
    chunks_seen.update(chunk_ids)

    for r in results:
        if r.snippet:
            seen_texts.append(r.snippet)

    return {
        "query": query,
        "results": [r.model_dump() for r in results],
        "total_found": len(results),
    }


def handle_read(
    chunk_ids: list[str],
    corpus: "DocumentCorpus",
    tracker: "TrajectoryTracker",
    context_chunks: dict[str, "Chunk"],
    context_token_count: int,
    chunks_seen: set[str],
    max_context_tokens: int,
) -> tuple[dict[str, Any], int]:
    """
    Handle read action.

    Returns:
        Tuple of (action_result dict, new context_token_count)
    """
    from ..models import Chunk

    chunks_added: list[Chunk] = []
    tokens_added = 0
    budget_exceeded = False
    chunks_truncated = 0

    remaining_budget = max_context_tokens - context_token_count

    for chunk_id in chunk_ids:
        if chunk_id in context_chunks:
            continue

        try:
            chunk = corpus.get_chunk(chunk_id)
        except Exception as exc:
            return {"error": str(exc)}, context_token_count
        if chunk is None:
            continue

        if tokens_added + chunk.token_count > remaining_budget:
            budget_exceeded = True
            chunks_truncated += 1
            continue

        context_chunks[chunk_id] = chunk
        context_token_count += chunk.token_count
        tokens_added += chunk.token_count
        chunks_added.append(chunk)

    tracker.record_read([c.chunk_id for c in chunks_added])
    chunks_seen.update(chunk_ids)

    return {
        "chunks": [c.model_dump() for c in chunks_added],
        "tokens_added": tokens_added,
        "budget_exceeded": budget_exceeded,
        "chunks_truncated": chunks_truncated,
    }, context_token_count


def handle_prune(
    chunk_ids: list[str],
    tracker: "TrajectoryTracker",
    context_chunks: dict[str, "Chunk"],
    context_token_count: int,
) -> tuple[dict[str, Any], int]:
    """
    Handle prune action.

    Returns:
        Tuple of (action_result dict, new context_token_count)
    """
    chunks_removed = 0
    tokens_freed = 0
    invalid_ids: list[str] = []

    for chunk_id in chunk_ids:
        if chunk_id in context_chunks:
            chunk = context_chunks.pop(chunk_id)
            context_token_count -= chunk.token_count
            tokens_freed += chunk.token_count
            chunks_removed += 1
        else:
            invalid_ids.append(chunk_id)

    tracker.record_prune(chunk_ids)

    return {
        "chunks_removed": chunks_removed,
        "tokens_freed": tokens_freed,
        "invalid_ids": invalid_ids,
    }, context_token_count


def handle_answer(
    answer: str,
    supporting_chunk_ids: list[str],
    reward_calculator: "RewardCalculator",
    tracker: "TrajectoryTracker",
    context_chunks: dict[str, "Chunk"],
    context_token_count: int,
    seen_texts: list[str],
    gold_chunks: set[str],
    gold_answer: str,
    steps_used: int,
    max_steps: int,
    max_tokens: int,
) -> tuple[dict[str, Any], "RewardMetrics"]:
    """
    Handle answer action - computes final reward.

    Returns:
        Tuple of (action_result dict, RewardMetrics)
    """
    metrics = reward_calculator.calculate_reward(
        tracker=tracker,
        gold_chunks=gold_chunks,
        gold_answer=gold_answer,
        predicted_answer=answer,
        context_texts=[chunk.content for chunk in context_chunks.values()],
        steps_used=steps_used,
        max_steps=max_steps,
        tokens_used=context_token_count,
        max_tokens=max_tokens,
        all_seen_texts=seen_texts if seen_texts else None,
    )

    return {
        "answer_submitted": answer,
        "final_reward": metrics.total_reward,
        "trajectory_recall": metrics.trajectory_recall,
        "output_recall": metrics.output_recall,
        "output_precision": metrics.output_precision,
        "f_beta": metrics.f_beta,
        "beta_used": metrics.beta,
        "answer_correct": metrics.answer_correct,
        "answer_found_in_context": metrics.answer_found_in_context,
        "answer_similarity": metrics.answer_similarity,
        "f_beta_reward": metrics.f_beta_reward,
        "trajectory_reward": metrics.trajectory_reward,
        "answer_reward": metrics.answer_reward,
        "turn_penalty": metrics.turn_penalty,
        "prune_penalty": metrics.prune_penalty,
        "pre_penalty_reward": metrics.pre_penalty_reward,
        "reward_floor": metrics.reward_floor,
    }, metrics
