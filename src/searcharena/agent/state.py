"""State serialization and message building for LLM."""

from __future__ import annotations

import json
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from .action import truncate


def build_state(
    obs: Any,
    step: int,
    max_steps: int,
    full_context: dict[str, str],
    char_limit: int = 800,
    soft_budget_threshold: float = 0.75,
    hard_budget_threshold: float = 0.95,
) -> str:
    """Build state description for LLM prompt."""
    context_items = []
    for chunk in (obs.context_chunks or [])[:3]:
        text = full_context.get(chunk.chunk_id) or getattr(chunk, "snippet", "")
        context_items.append({
            "id": chunk.chunk_id,
            "title": truncate(chunk.title, 40),
            "tokens": chunk.token_count,
            "text": truncate(text, char_limit),
        })

    pct = obs.budget_usage_percent
    if pct >= hard_budget_threshold * 100:
        status = "CRITICAL - must prune or answer"
    elif pct >= soft_budget_threshold * 100:
        status = "high - consider pruning"
    else:
        status = "ok"

    state = {
        "step": step,
        "remaining": max_steps - step,
        "budget": {
            "used": obs.context_token_count,
            "limit": obs.context_token_budget,
            "percent": round(pct, 1),
            "status": status,
        },
        "context": context_items,
        "last_action": summarize_action(obs),
    }
    return f"Question: {obs.question}\n\nState:\n{json.dumps(state, indent=2)}"


def summarize_action(obs: Any) -> dict[str, Any]:
    """Summarize the last action result."""
    at = obs.action_type or "none"
    r = obs.action_result or {}

    if at == "search":
        return {
            "type": "search",
            "query": r.get("query"),
            "results": [
                {
                    "id": x.get("chunk_id"),
                    "title": truncate(x.get("title", ""), 40),
                    "snippet": truncate(x.get("snippet", ""), 100),
                }
                for x in r.get("results", [])[:4]
            ],
        }

    if at == "read":
        return {
            "type": "read",
            "tokens_added": r.get("tokens_added"),
            "chunks": [
                {"id": c.get("chunk_id"), "text": truncate(c.get("content", ""), 150)}
                for c in r.get("chunks", [])[:2]
            ],
        }

    if at == "prune":
        return {
            "type": "prune",
            "removed": r.get("chunks_removed"),
            "freed": r.get("tokens_freed"),
        }

    if at == "answer":
        return {
            "type": "answer",
            "reward": r.get("final_reward"),
            "correct": r.get("answer_correct"),
        }

    return {"type": at}


def build_tool_result(
    tool_id: str,
    obs: Any,
    step: int,
    max_steps: int,
) -> ChatCompletionMessageParam:
    """Build tool result message for conversation."""
    return {
        "role": "tool",
        "tool_call_id": tool_id,
        "content": json.dumps({
            "step": step + 1,
            "remaining": max_steps - step - 1,
            "done": obs.done,
            "reward": obs.reward or 0.0,
            "budget": {
                "used": obs.context_token_count,
                "percent": round(obs.budget_usage_percent, 1),
            },
            "last_action": summarize_action(obs),
        }),
    }


def update_context_cache(obs: Any, cache: dict[str, str]) -> dict[str, str]:
    """Update context cache with newly read chunks."""
    if obs.action_type == "read":
        for chunk in (obs.action_result or {}).get("chunks", []):
            if chunk.get("chunk_id") and chunk.get("content"):
                cache[chunk["chunk_id"]] = chunk["content"]
    current_ids = {c.chunk_id for c in (obs.context_chunks or [])}
    return {k: v for k, v in cache.items() if k in current_ids}
