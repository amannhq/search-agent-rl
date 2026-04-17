"""Heuristic fallback policy when LLM is unavailable."""

from __future__ import annotations

from typing import Any

from ..models import SearchAction
from .action import ActionBuilder, clean
from .config import (
    STOPWORDS,
    RE_WORD,
    RE_SENTENCE_SPLIT,
    RE_DIGIT,
)


class HeuristicPolicy:
    """Fallback policy using heuristics when LLM fails."""

    def __init__(
        self,
        builder: ActionBuilder,
        soft_budget_threshold: float = 0.75,
        hard_budget_threshold: float = 0.95,
        prune_target_threshold: float = 0.60,
        read_top_k: int = 2,
    ) -> None:
        self.builder = builder
        self.soft_budget_threshold = soft_budget_threshold
        self.hard_budget_threshold = hard_budget_threshold
        self.prune_target_threshold = prune_target_threshold
        self.read_top_k = read_top_k

    def get_action(self) -> tuple[SearchAction, str]:
        """Select action using heuristics."""
        if self._should_prune():
            return self._prune_lowest()
        if self._has_unread_results():
            return self._read_top()
        if self.builder.context:
            return self.builder.answer(self._extract_answer())
        return self.builder.search(self.builder.obs.question)

    def _should_prune(self) -> bool:
        """Check if pruning is needed based on budget."""
        if len(self.builder.context) < 2:
            return False
        return (
            self.builder.budget_pct >= self.hard_budget_threshold
            or (
                self.builder.budget_pct >= self.soft_budget_threshold
                and len(self.builder.context) > 3
            )
        )

    def _prune_lowest(self) -> tuple[SearchAction, str]:
        """Prune lowest-scored chunks to free budget."""
        sorted_chunks = sorted(
            self.builder.context,
            key=lambda c: getattr(c, "score", 0),
        )
        target = int(
            self.builder.obs.context_token_count
            * (1 - self.prune_target_threshold / self.builder.budget_pct)
        )
        to_prune: list[str] = []
        freed = 0
        for chunk in sorted_chunks:
            if freed >= target:
                break
            to_prune.append(chunk.chunk_id)
            freed += chunk.token_count
        if not to_prune:
            to_prune = [sorted_chunks[0].chunk_id]
        return self.builder.prune(to_prune)

    def _has_unread_results(self) -> bool:
        """Check if there are unread search results."""
        results = self.builder.result.get("results", [])
        return bool(results) and any(
            r.get("chunk_id") not in self.builder.context_ids for r in results
        )

    def _read_top(self) -> tuple[SearchAction, str]:
        """Read top unread results."""
        results = self.builder.result.get("results", [])
        ids = [
            r["chunk_id"]
            for r in results[: self.read_top_k]
            if r.get("chunk_id") and r["chunk_id"] not in self.builder.context_ids
        ]
        if ids:
            return self.builder.read(ids)
        return self.builder.answer(self._extract_answer())

    def _extract_answer(self) -> str:
        """Extract answer from available context."""
        q_words = {
            w
            for w in RE_WORD.findall(self.builder.obs.question.lower())
            if w not in STOPWORDS
        }

        texts: list[str] = []
        for chunk in self.builder.result.get("chunks", []):
            texts.append(clean(chunk.get("content", "")))
        for chunk in self.builder.context:
            texts.append(clean(getattr(chunk, "snippet", "")))

        scored: list[tuple[int, str]] = []
        for text in texts:
            for sent in RE_SENTENCE_SPLIT.split(text):
                sent = clean(sent)
                if not sent:
                    continue
                words = set(RE_WORD.findall(sent.lower()))
                sc = len(words & q_words) + (1 if RE_DIGIT.search(sent) else 0)
                if sc > 0:
                    scored.append((sc, sent))

        if scored:
            scored.sort(key=lambda x: (-x[0], len(x[1])))
            return " ".join(s for _, s in scored[:3])

        if texts:
            return texts[0]
        return f"Based on retrieved evidence: {self.builder.obs.question}"
