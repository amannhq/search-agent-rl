"""Reward-related metrics data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardMetrics:
    """Metrics computed for reward calculation."""

    output_precision: float = 0.0
    output_recall: float = 0.0
    trajectory_recall: float = 0.0
    f_beta: float = 0.0
    beta: float = 4.0
    answer_correct: bool = False
    answer_similarity: float = 0.0
    answer_found_in_context: bool = False
    steps_used: int = 0
    max_steps: int = 20
    tokens_used: int = 0
    max_tokens: int = 32768
    f_beta_reward: float = 0.0
    trajectory_reward: float = 0.0
    answer_reward: float = 0.0
    efficiency_reward: float = 0.0
    turn_penalty: float = 0.0
    prune_penalty: float = 0.0
    pre_penalty_reward: float = 0.0
    reward_floor: float = 0.0
    evidence_precision: float = 0.0
    evidence_recall: float = 0.0
    evidence_reward: float = 0.0
    unsupported_support_count: int = 0
    total_reward: float = 0.0
