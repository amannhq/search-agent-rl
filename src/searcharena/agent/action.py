"""Action building and conversion from LLM tool calls."""

from __future__ import annotations

import json
from typing import Any

from ..models import SearchAction
from .config import RE_WHITESPACE


def clean(text: str) -> str:
    """Normalize whitespace in text."""
    return RE_WHITESPACE.sub(" ", text).strip()


def truncate(text: str, limit: int = 200) -> str:
    """Truncate text to limit with ellipsis."""
    text = clean(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class ActionBuilder:
    """Converts LLM tool calls to SearchAction objects."""

    def __init__(
        self,
        observation: Any,
        search_top_k: int = 5,
    ) -> None:
        self.obs = observation
        self.search_top_k = search_top_k
        self.result: dict[str, Any] = observation.action_result or {}
        self.context = observation.context_chunks or []
        self.context_ids: set[str] = {c.chunk_id for c in self.context}
        self.budget_pct: float = observation.budget_usage_percent / 100.0

    def search(self, query: str, top_k: int | None = None) -> tuple[SearchAction, str]:
        """Build a search action."""
        k = top_k if top_k is not None else self.search_top_k
        return SearchAction.make_search(query, k), f"search('{truncate(query, 40)}', k={k})"

    def read(self, chunk_ids: list[str]) -> tuple[SearchAction, str]:
        """Build a read action."""
        return SearchAction.make_read(chunk_ids), f"read({len(chunk_ids)} chunks)"

    def prune(self, chunk_ids: list[str]) -> tuple[SearchAction, str]:
        """Build a prune action."""
        return SearchAction.make_prune(chunk_ids), f"prune({len(chunk_ids)} chunks)"

    def answer(self, text: str, support_ids: list[str] | None = None) -> tuple[SearchAction, str]:
        """Build an answer action."""
        ids = support_ids if support_ids else list(self.context_ids)
        return SearchAction.make_answer(text, ids), f"answer('{truncate(text, 30)}')"

    def from_tool_call(self, tool_call: Any) -> tuple[SearchAction, str, str]:
        """
        Convert an LLM tool call to a SearchAction.

        Returns:
            Tuple of (action, description, tool_call_id)

        Raises:
            ValueError: If tool call is invalid or unknown
        """
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")

        if name == "search":
            query = clean(args.get("query", "")) or self.obs.question
            action, desc = self.search(query, args.get("top_k"))
            return action, desc, tool_call.id

        if name == "read":
            ids = [str(c) for c in args.get("chunk_ids", []) if c]
            if not ids:
                raise ValueError("read requires chunk_ids")
            return *self.read(ids), tool_call.id

        if name == "prune":
            ids = [str(c) for c in args.get("chunk_ids", []) if c]
            if not ids:
                raise ValueError("prune requires chunk_ids")
            return *self.prune(ids), tool_call.id

        if name == "answer":
            text = clean(args.get("answer", ""))
            if not text:
                raise ValueError("answer requires text")
            support = [s for s in args.get("supporting_chunk_ids", []) if s in self.context_ids]
            return *self.answer(text, support or None), tool_call.id

        raise ValueError(f"Unknown tool: {name}")
