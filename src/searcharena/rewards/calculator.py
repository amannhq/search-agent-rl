"""Reward calculation logic."""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher

from .metrics import RewardMetrics
from .tracker import TrajectoryTracker


class RewardCalculator:
    """Calculate rewards for search episodes."""

    def __init__(
        self,
        beta: float = 4.0,
        f_beta_weight: float = 1.0,
        answer_reward_weight: float = 1.0,
        trajectory_reward_weight: float = 0.0,
        efficiency_reward_weight: float = 0.0,
        successful_trajectory_floor: float = 0.0,
        use_trajectory_reward: bool = False,
        prune_penalty_threshold: int = 3,
        prune_penalty_per_excess: float = 0.1,
        prune_penalty_cap: float = 0.5,
        turn_penalty_start: int = 15,
        turn_penalty_end: int = 20,
        turn_penalty_max: float = 0.5,
    ) -> None:
        self.beta = beta
        self.f_beta_weight = f_beta_weight
        self.answer_reward_weight = answer_reward_weight
        self.trajectory_reward_weight = trajectory_reward_weight
        self.efficiency_reward_weight = efficiency_reward_weight
        self.successful_trajectory_floor = successful_trajectory_floor
        self.use_trajectory_reward = use_trajectory_reward
        self.prune_penalty_threshold = prune_penalty_threshold
        self.prune_penalty_per_excess = prune_penalty_per_excess
        self.prune_penalty_cap = prune_penalty_cap
        self.turn_penalty_start = turn_penalty_start
        self.turn_penalty_end = turn_penalty_end
        self.turn_penalty_max = turn_penalty_max

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def compute_f_beta(self, precision: float, recall: float, beta: float | None = None) -> float:
        beta = beta if beta is not None else self.beta
        if precision + recall == 0:
            return 0.0

        beta_sq = beta**2
        return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

    def compute_precision_recall(
        self,
        retrieved_chunks: set[str],
        gold_chunks: set[str],
    ) -> tuple[float, float]:
        if not retrieved_chunks:
            return 0.0, 0.0
        if not gold_chunks:
            return 0.0, 1.0

        true_positives = len(retrieved_chunks & gold_chunks)
        precision = true_positives / len(retrieved_chunks)
        recall = true_positives / len(gold_chunks)
        return precision, recall

    def compute_answer_similarity(
        self,
        predicted: str,
        gold: str,
        method: str = "fuzzy",
    ) -> tuple[bool, float]:
        pred_lower = self._normalize_text(predicted)
        gold_lower = self._normalize_text(gold)

        if method == "exact":
            is_correct = pred_lower == gold_lower
            similarity = 1.0 if is_correct else 0.0
        elif method == "fuzzy":
            similarity = SequenceMatcher(None, pred_lower, gold_lower).ratio()
            is_correct = similarity >= 0.8
        elif method == "contains":
            is_correct = gold_lower in pred_lower or pred_lower in gold_lower
            similarity = 1.0 if is_correct else SequenceMatcher(None, pred_lower, gold_lower).ratio()
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown method: {method}")

        return is_correct, similarity

    def compute_answer_found_in_context(
        self,
        context_texts: Iterable[str] | None,
        gold_answer: str,
    ) -> bool:
        if not context_texts:
            return False

        gold_normalized = self._normalize_text(gold_answer)
        if not gold_normalized:
            return False

        key_phrases: list[str] = []
        for sub in re.split(r"[,;:\-\(\)]", gold_normalized):
            sub = sub.strip()
            if len(sub) > 3:
                key_phrases.append(sub)
        key_phrases.append(gold_normalized)

        context_combined = " ".join(self._normalize_text(text) for text in context_texts)
        if gold_normalized in context_combined:
            return True

        found_count = sum(1 for phrase in key_phrases if phrase in context_combined)
        required = min(2, max(1, len(key_phrases) // 2))
        return found_count >= required

    def compute_efficiency_reward(
        self,
        steps_used: int,
        max_steps: int,
        tokens_used: int,
        max_tokens: int,
    ) -> float:
        step_efficiency = 1 - (steps_used / max_steps) if max_steps > 0 else 0
        token_efficiency = 1 - (tokens_used / max_tokens) if max_tokens > 0 else 0
        return (step_efficiency + token_efficiency) / 2

    def compute_turn_penalty(self, steps: int) -> float:
        if steps <= self.turn_penalty_start:
            return 0.0
        if self.turn_penalty_end == self.turn_penalty_start:
            return self.turn_penalty_max

        progress = (steps - self.turn_penalty_start) / (self.turn_penalty_end - self.turn_penalty_start)
        return min(self.turn_penalty_max, self.turn_penalty_max * progress)

    def compute_prune_penalty(self, consecutive_prunes: int) -> float:
        if consecutive_prunes <= self.prune_penalty_threshold:
            return 0.0

        excess = consecutive_prunes - self.prune_penalty_threshold
        return min(self.prune_penalty_cap, excess * self.prune_penalty_per_excess)

    def compute_evidence_metrics(
        self,
        supporting_chunk_ids: list[str],
        gold_chunks: set[str],
    ) -> tuple[float, float, int]:
        predicted = set(supporting_chunk_ids)
        if not predicted:
            return 0.0, 0.0, 0
        if not gold_chunks:
            return 0.0, 1.0, len(predicted)

        true_positives = len(predicted & gold_chunks)
        precision = true_positives / len(predicted)
        recall = true_positives / len(gold_chunks)
        unsupported = len(predicted - gold_chunks)
        return precision, recall, unsupported

    def calculate_reward(
        self,
        tracker: TrajectoryTracker,
        gold_chunks: set[str],
        gold_answer: str,
        predicted_answer: str,
        context_texts: list[str] | None,
        steps_used: int,
        max_steps: int,
        tokens_used: int,
        max_tokens: int,
        supporting_chunk_ids: list[str] | None = None,
        answer_method: str = "fuzzy",
        all_seen_texts: list[str] | None = None,
    ) -> RewardMetrics:
        metrics = RewardMetrics(
            beta=self.beta,
            steps_used=steps_used,
            max_steps=max_steps,
            tokens_used=tokens_used,
            max_tokens=max_tokens,
        )

        output_precision, output_recall = self.compute_precision_recall(
            tracker.chunks_in_context,
            gold_chunks,
        )
        _, trajectory_recall = self.compute_precision_recall(tracker.chunks_seen, gold_chunks)

        if trajectory_recall == 0 and gold_answer:
            gold_norm = self._normalize_text(gold_answer)
            if all_seen_texts:
                seen_relevant = sum(1 for text in all_seen_texts if gold_norm in self._normalize_text(text))
                if seen_relevant > 0:
                    trajectory_recall = 1.0
            if context_texts:
                context_relevant = sum(1 for text in context_texts if gold_norm in self._normalize_text(text))
                if context_relevant > 0 and len(context_texts) > 0:
                    output_recall = 1.0
                    output_precision = context_relevant / len(context_texts)

        metrics.output_precision = output_precision
        metrics.output_recall = output_recall
        metrics.trajectory_recall = trajectory_recall
        metrics.f_beta = self.compute_f_beta(output_precision, output_recall)

        answer_correct, answer_similarity = self.compute_answer_similarity(
            predicted_answer,
            gold_answer,
            answer_method,
        )
        metrics.answer_correct = answer_correct
        metrics.answer_similarity = answer_similarity
        metrics.answer_found_in_context = self.compute_answer_found_in_context(context_texts, gold_answer)

        evidence_precision, evidence_recall, unsupported = self.compute_evidence_metrics(
            supporting_chunk_ids or [],
            gold_chunks,
        )
        metrics.evidence_precision = evidence_precision
        metrics.evidence_recall = evidence_recall
        metrics.unsupported_support_count = unsupported
        metrics.evidence_reward = max(0.0, evidence_precision * evidence_recall - (unsupported * 0.05))

        efficiency = self.compute_efficiency_reward(steps_used, max_steps, tokens_used, max_tokens)
        metrics.efficiency_reward = efficiency * self.efficiency_reward_weight
        metrics.turn_penalty = self.compute_turn_penalty(steps_used)
        metrics.prune_penalty = self.compute_prune_penalty(tracker.consecutive_prunes)
        metrics.f_beta_reward = metrics.f_beta * self.f_beta_weight
        metrics.trajectory_reward = trajectory_recall * self.trajectory_reward_weight if self.use_trajectory_reward else 0.0
        metrics.answer_reward = self.answer_reward_weight if metrics.answer_found_in_context else 0.0
        metrics.pre_penalty_reward = (
            metrics.f_beta_reward
            + metrics.trajectory_reward
            + metrics.answer_reward
            + metrics.evidence_reward
        )
        metrics.reward_floor = self.successful_trajectory_floor

        total = metrics.pre_penalty_reward - metrics.turn_penalty - metrics.prune_penalty
        metrics.total_reward = max(0.001, min(0.999, max(metrics.reward_floor, total)))
        return metrics
