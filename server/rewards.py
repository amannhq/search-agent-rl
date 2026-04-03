"""
Reward functions for the Search RL Environment.
Implements F-beta reward with trajectory tracking.
"""

from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set


@dataclass
class RewardMetrics:
    """Metrics computed for reward calculation."""

    # Precision and recall
    output_precision: float = 0.0
    output_recall: float = 0.0
    trajectory_recall: float = 0.0

    # F-beta score
    f_beta: float = 0.0
    beta: float = 4.0

    # Answer evaluation
    answer_correct: bool = False
    answer_similarity: float = 0.0
    answer_found_in_context: bool = False

    # Efficiency
    steps_used: int = 0
    max_steps: int = 20
    tokens_used: int = 0
    max_tokens: int = 32768

    # Component rewards
    f_beta_reward: float = 0.0
    trajectory_reward: float = 0.0
    answer_reward: float = 0.0
    efficiency_reward: float = 0.0

    # Penalties
    turn_penalty: float = 0.0
    prune_penalty: float = 0.0

    # Pre/post clamp bookkeeping
    pre_penalty_reward: float = 0.0
    reward_floor: float = 0.0

    # Final reward
    total_reward: float = 0.0


@dataclass
class TrajectoryTracker:
    """
    Tracks chunks encountered during an episode.

    Used to compute trajectory recall - rewarding exploration even if
    relevant chunks are later pruned.
    """

    # All chunks ever seen during search
    chunks_seen: Set[str] = field(default_factory=set)

    # Chunks currently in context
    chunks_in_context: Set[str] = field(default_factory=set)

    # Search queries issued
    queries: List[str] = field(default_factory=list)

    # Prune history (for penalty calculation)
    consecutive_prunes: int = 0
    total_prunes: int = 0

    def record_search(self, query: str, chunk_ids: List[str]) -> None:
        """Record a search action."""
        self.queries.append(query)
        self.chunks_seen.update(chunk_ids)
        self.consecutive_prunes = 0  # Reset prune streak

    def record_read(self, chunk_ids: List[str]) -> None:
        """Record chunks added to context."""
        self.chunks_in_context.update(chunk_ids)
        self.chunks_seen.update(chunk_ids)
        self.consecutive_prunes = 0

    def record_prune(self, chunk_ids: List[str]) -> None:
        """Record chunks removed from context."""
        self.chunks_in_context -= set(chunk_ids)
        self.consecutive_prunes += 1
        self.total_prunes += 1

    def reset(self) -> None:
        """Reset tracker for new episode."""
        self.chunks_seen.clear()
        self.chunks_in_context.clear()
        self.queries.clear()
        self.consecutive_prunes = 0
        self.total_prunes = 0


class RewardCalculator:
    """
    Calculate rewards for the Search RL Environment.

    - Weighted F-beta outcome score
    - Answer bonus for finding the answer
    - Trajectory recall process reward
    - Penalties for degenerate behavior
    """

    def __init__(
        self,
        beta: float = 4.0,
        f_beta_weight: float = 0.7,
        answer_reward_weight: float = 1.0,
        trajectory_reward_weight: float = 0.3,
        efficiency_reward_weight: float = 0.0,
        successful_trajectory_floor: float = 0.01,
        use_trajectory_reward: bool = True,
        prune_penalty_threshold: int = 3,
        prune_penalty_per_excess: float = 0.1,
        prune_penalty_cap: float = 0.5,
        turn_penalty_start: int = 64,
        turn_penalty_end: int = 128,
        turn_penalty_max: float = 0.5,
    ):
        """
        Initialize reward calculator.

        Args:
            beta: F-beta parameter. >1 favors recall, <1 favors precision
            f_beta_weight: Weight for the F-beta outcome component
            answer_reward_weight: Weight for answer bonus
            trajectory_reward_weight: Weight for trajectory recall component
            efficiency_reward_weight: Weight for efficiency diagnostics
            successful_trajectory_floor: Minimum reward for successful completions
            use_trajectory_reward: Whether to include trajectory recall
            prune_penalty_threshold: Consecutive prunes before penalty
            prune_penalty_per_excess: Penalty per excess prune
            prune_penalty_cap: Maximum prune penalty
            turn_penalty_start: Turn at which penalty begins
            turn_penalty_end: Turn at which penalty is maximum
            turn_penalty_max: Maximum turn penalty
        """
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
        """Normalize text for lightweight string matching."""
        return " ".join(text.strip().lower().split())

    def compute_f_beta(
        self, precision: float, recall: float, beta: Optional[float] = None
    ) -> float:
        """
        Compute F-beta score.

        Args:
            precision: Precision value [0, 1]
            recall: Recall value [0, 1]
            beta: Beta parameter (uses self.beta if None)

        Returns:
            F-beta score [0, 1]
        """
        beta = beta if beta is not None else self.beta

        if precision + recall == 0:
            return 0.0

        beta_sq = beta**2
        f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
        return f_beta

    def compute_precision_recall(
        self,
        retrieved_chunks: Set[str],
        gold_chunks: Set[str],
    ) -> tuple[float, float]:
        """
        Compute precision and recall.

        Args:
            retrieved_chunks: Set of chunk IDs in output
            gold_chunks: Set of gold chunk IDs

        Returns:
            Tuple of (precision, recall)
        """
        if not retrieved_chunks:
            return 0.0, 0.0

        if not gold_chunks:
            return 0.0, 1.0  # No gold chunks means recall is trivially 1

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
        """
        Evaluate answer quality.

        Args:
            predicted: Predicted answer
            gold: Gold answer
            method: Evaluation method ("exact", "fuzzy", "contains")

        Returns:
            Tuple of (is_correct, similarity_score)
        """
        pred_lower = self._normalize_text(predicted)
        gold_lower = self._normalize_text(gold)

        if method == "exact":
            is_correct = pred_lower == gold_lower
            similarity = 1.0 if is_correct else 0.0

        elif method == "fuzzy":
            similarity = SequenceMatcher(None, pred_lower, gold_lower).ratio()
            is_correct = similarity >= 0.8  # 80% threshold

        elif method == "contains":
            is_correct = gold_lower in pred_lower or pred_lower in gold_lower
            similarity = (
                1.0
                if is_correct
                else SequenceMatcher(None, pred_lower, gold_lower).ratio()
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        return is_correct, similarity

    def compute_answer_found_in_context(
        self,
        context_texts: Optional[Iterable[str]],
        gold_answer: str,
    ) -> bool:
        """
        Check whether the final context includes key parts of the gold answer.

        Uses a lenient matching approach: extracts key entities/facts from
        the gold answer and checks if they appear in the context.
        """
        if not context_texts:
            return False

        gold_normalized = self._normalize_text(gold_answer)
        if not gold_normalized:
            return False

        # Extract key phrases from gold answer (split on common separators)
        # This handles cases like "Mark Zuckerberg was born in White Plains, New York"
        # where we want to find both "mark zuckerberg" and "white plains"
        import re

        key_phrases = []

        # Extract proper nouns and key facts (words that aren't stopwords)
        stopwords = {
            "a",
            "an",
            "the",
            "was",
            "were",
            "is",
            "are",
            "in",
            "on",
            "at",
            "for",
            "to",
            "of",
            "and",
            "or",
            "which",
            "that",
            "who",
            "whom",
            "born",
            "paid",
            "acquired",
            "went",
            "public",
            "founded",
            "compared",
        }

        # Find capitalized phrases or numbers with context
        words = gold_normalized.split()
        current_phrase = []
        for word in words:
            # Keep numbers, dollar amounts, years, or multi-word names
            if (
                word.replace("$", "").replace(",", "").replace(".", "").isdigit()
                or word not in stopwords
                or len(word) > 3
            ):
                current_phrase.append(word)
            else:
                if current_phrase:
                    phrase = " ".join(current_phrase)
                    if len(phrase) > 3:
                        key_phrases.append(phrase)
                    current_phrase = []
        if current_phrase:
            phrase = " ".join(current_phrase)
            if len(phrase) > 3:
                key_phrases.append(phrase)

        # Also add the full normalized gold answer
        key_phrases.append(gold_normalized)

        # Check if context contains either the full answer or key phrases
        context_combined = " ".join(self._normalize_text(t) for t in context_texts)

        # Full answer match
        if gold_normalized in context_combined:
            return True

        # Key phrase matching: require majority of key phrases found
        if key_phrases:
            found_count = sum(1 for phrase in key_phrases if phrase in context_combined)
            # At least half of key phrases (or 2, whichever is smaller) must be found
            required = min(2, max(1, len(key_phrases) // 2))
            if found_count >= required:
                return True

        return False

    def compute_efficiency_reward(
        self,
        steps_used: int,
        max_steps: int,
        tokens_used: int,
        max_tokens: int,
    ) -> float:
        """
        Compute efficiency reward.

        Args:
            steps_used: Number of steps taken
            max_steps: Maximum allowed steps
            tokens_used: Tokens used in context
            max_tokens: Maximum token budget

        Returns:
            Efficiency reward [0, 1]
        """
        step_efficiency = 1 - (steps_used / max_steps) if max_steps > 0 else 0
        token_efficiency = 1 - (tokens_used / max_tokens) if max_tokens > 0 else 0

        return (step_efficiency + token_efficiency) / 2

    def compute_turn_penalty(self, steps: int) -> float:
        """
        Compute turn count penalty.

        Linear penalty from turn_penalty_start to turn_penalty_end.
        """
        if steps <= self.turn_penalty_start:
            return 0.0

        progress = (steps - self.turn_penalty_start) / (
            self.turn_penalty_end - self.turn_penalty_start
        )
        return min(self.turn_penalty_max, self.turn_penalty_max * progress)

    def compute_prune_penalty(self, consecutive_prunes: int) -> float:
        """
        Compute penalty for excessive consecutive pruning.

        Discourages pruning one chunk at a time.
        """
        if consecutive_prunes <= self.prune_penalty_threshold:
            return 0.0

        excess = consecutive_prunes - self.prune_penalty_threshold
        return min(self.prune_penalty_cap, excess * self.prune_penalty_per_excess)

    def calculate_reward(
        self,
        tracker: TrajectoryTracker,
        gold_chunks: Set[str],
        gold_answer: str,
        predicted_answer: str,
        context_texts: Optional[List[str]],
        steps_used: int,
        max_steps: int,
        tokens_used: int,
        max_tokens: int,
        answer_method: str = "fuzzy",
        all_seen_texts: Optional[List[str]] = None,
    ) -> RewardMetrics:
        """
        Calculate full reward for an episode.

        Args:
            tracker: Trajectory tracker with episode history
            gold_chunks: Set of gold chunk IDs
            gold_answer: Gold answer string
            predicted_answer: Agent's predicted answer
            context_texts: Full text of chunks currently kept in context
            steps_used: Number of steps taken
            max_steps: Maximum allowed steps
            tokens_used: Tokens used in context
            max_tokens: Maximum token budget
            answer_method: Method for answer evaluation
            all_seen_texts: Optional list of all chunk texts seen (for content-based matching)

        Returns:
            RewardMetrics with all components
        """
        metrics = RewardMetrics(
            beta=self.beta,
            steps_used=steps_used,
            max_steps=max_steps,
            tokens_used=tokens_used,
            max_tokens=max_tokens,
        )

        # Try ID-based matching first
        output_precision, output_recall = self.compute_precision_recall(
            tracker.chunks_in_context, gold_chunks
        )
        _, trajectory_recall = self.compute_precision_recall(
            tracker.chunks_seen, gold_chunks
        )

        # If ID matching fails (web search mode), use content-based matching
        if trajectory_recall == 0 and gold_answer:
            gold_norm = self._normalize_text(gold_answer)

            # Check if any seen content contains the gold answer
            if all_seen_texts:
                seen_relevant = sum(
                    1
                    for text in all_seen_texts
                    if gold_norm in self._normalize_text(text)
                )
                if seen_relevant > 0:
                    trajectory_recall = 1.0  # Found relevant content

            # Check if context contains gold answer
            if context_texts:
                context_relevant = sum(
                    1
                    for text in context_texts
                    if gold_norm in self._normalize_text(text)
                )
                if context_relevant > 0 and len(context_texts) > 0:
                    output_recall = 1.0
                    output_precision = context_relevant / len(context_texts)

        metrics.output_precision = output_precision
        metrics.output_recall = output_recall
        metrics.trajectory_recall = trajectory_recall

        # F-beta score on output
        metrics.f_beta = self.compute_f_beta(output_precision, output_recall)

        # Answer evaluation
        answer_correct, answer_similarity = self.compute_answer_similarity(
            predicted_answer, gold_answer, answer_method
        )
        metrics.answer_correct = answer_correct
        metrics.answer_similarity = answer_similarity
        metrics.answer_found_in_context = self.compute_answer_found_in_context(
            context_texts, gold_answer
        )

        # Efficiency is still tracked for diagnostics, but not used in the
        # final reward formula.
        efficiency = self.compute_efficiency_reward(
            steps_used, max_steps, tokens_used, max_tokens
        )
        metrics.efficiency_reward = efficiency * self.efficiency_reward_weight

        # Penalties
        metrics.turn_penalty = self.compute_turn_penalty(steps_used)
        metrics.prune_penalty = self.compute_prune_penalty(tracker.consecutive_prunes)

        # Component rewards
        metrics.f_beta_reward = metrics.f_beta * self.f_beta_weight
        metrics.trajectory_reward = (
            trajectory_recall * self.trajectory_reward_weight
            if self.use_trajectory_reward
            else 0.0
        )
        metrics.answer_reward = (
            self.answer_reward_weight if metrics.answer_found_in_context else 0.0
        )

        # Total reward: clamp(0.7 * F_beta + 0.3 * r_traj + r_fa - penalties, floor, pre_penalty_reward)
        metrics.pre_penalty_reward = (
            metrics.f_beta_reward + metrics.trajectory_reward + metrics.answer_reward
        )
        metrics.reward_floor = self.successful_trajectory_floor

        total = metrics.pre_penalty_reward
        total -= metrics.turn_penalty + metrics.prune_penalty
        metrics.total_reward = min(
            metrics.pre_penalty_reward,
            max(metrics.reward_floor, total),
        )

        return metrics


class BetaScheduler:
    """
    Schedule beta parameter for curriculum learning.

    High beta (e.g., 4.0) → Emphasizes recall → Agent learns to find relevant docs
    Lower beta (e.g., 2.0) → Shifts toward precision as pruning improves
    """

    def __init__(
        self,
        start_beta: float = 4.0,
        end_beta: float = 2.0,
        warmup_steps: int = 1000,
        decay_steps: int = 10000,
    ):
        """
        Initialize beta scheduler.

        Args:
            start_beta: Initial beta value (recall-heavy)
            end_beta: Final beta value (balanced)
            warmup_steps: Steps to hold at start_beta
            decay_steps: Steps over which to decay to end_beta
        """
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def get_beta(self, step: int) -> float:
        """
        Get beta value for a given training step.

        Args:
            step: Current training step

        Returns:
            Beta value for this step
        """
        if step < self.warmup_steps:
            return self.start_beta

        decay_progress = min(1.0, (step - self.warmup_steps) / self.decay_steps)

        return self.start_beta + (self.end_beta - self.start_beta) * decay_progress
