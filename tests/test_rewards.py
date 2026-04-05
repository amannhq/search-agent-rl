"""Tests for reward calculation and beta scheduling."""

from typing import List

from models import SearchAction
from server.environment import SearchEnvironment
from server.retrieval import DocumentCorpus
from server.rewards import BetaScheduler, RewardCalculator, TrajectoryTracker


class TestRewardCalculation:
    """Tests for reward calculation in environment."""

    def test_reward_in_valid_range(self, env: SearchEnvironment) -> None:
        """Rewards should be in a reasonable range."""
        env.reset()

        obs = env.step(SearchAction.make_answer("test"))

        # Reward should be bounded (per Context-1 paper: -0.5 to 2.0)
        assert obs.reward is not None
        assert -1.0 <= float(obs.reward) <= 3.0

    def test_correct_answer_higher_reward(
        self, corpus: DocumentCorpus, tasks: List
    ) -> None:
        """Correct answer should yield higher reward than wrong answer."""
        env = SearchEnvironment(corpus=corpus, tasks=tasks)

        # Task 1: answer is "2010"
        env.reset()

        # Search and read relevant chunk
        env.step(SearchAction.make_search("Instagram launched"))
        search_obs = env.step(SearchAction.make_search("photo sharing app 2010"))
        assert search_obs.action_result is not None
        if search_obs.action_result.get("results"):
            chunk_id = search_obs.action_result["results"][0]["chunk_id"]
            env.step(SearchAction.make_read([chunk_id]))

        correct_obs = env.step(SearchAction.make_answer("2010"))
        assert correct_obs.reward is not None
        correct_reward = float(correct_obs.reward)

        # Reset and give wrong answer
        env.reset()
        wrong_obs = env.step(SearchAction.make_answer("1999"))
        assert wrong_obs.reward is not None
        wrong_reward = float(wrong_obs.reward)

        assert correct_reward >= wrong_reward

    def test_reward_deterministic(self, corpus: DocumentCorpus, tasks: List) -> None:
        """Same actions should produce same reward."""
        rewards: List[float] = []
        for _ in range(3):
            env = SearchEnvironment(corpus=corpus, tasks=tasks)
            env.reset()
            env.step(SearchAction.make_search("Facebook"))
            obs = env.step(SearchAction.make_answer("test"))
            assert obs.reward is not None
            rewards.append(float(obs.reward))

        assert rewards[0] == rewards[1] == rewards[2]


class TestRewardCalculator:
    """Tests for the RewardCalculator class."""

    def test_f_beta_calculation(self) -> None:
        """F-beta score should be calculated correctly."""
        calc = RewardCalculator(beta=4.0)
        tracker = TrajectoryTracker()
        tracker.chunks_in_context = {"gold_1", "gold_2", "noise"}
        tracker.chunks_seen = {"gold_1", "gold_2", "noise"}

        metrics = calc.calculate_reward(
            tracker=tracker,
            gold_chunks={"gold_1", "gold_2"},
            gold_answer="answer",
            predicted_answer="answer",
            context_texts=["text"],
            steps_used=5,
            max_steps=20,
            tokens_used=1000,
            max_tokens=32768,
        )

        # output_recall = 2/2 = 1.0 (both gold chunks in context)
        # output_precision = 2/3 ≈ 0.67 (2 gold out of 3 total)
        assert metrics.output_recall == 1.0
        assert 0.6 < metrics.output_precision < 0.7

    def test_trajectory_recall(self) -> None:
        """Trajectory recall should credit exploration."""
        calc = RewardCalculator(beta=4.0, use_trajectory_reward=True)
        tracker = TrajectoryTracker()
        tracker.chunks_seen = {"gold_1", "gold_2", "noise"}  # Saw gold chunks
        tracker.chunks_in_context = {"noise"}  # But pruned them

        metrics = calc.calculate_reward(
            tracker=tracker,
            gold_chunks={"gold_1", "gold_2"},
            gold_answer="answer",
            predicted_answer="wrong",
            context_texts=["text"],
            steps_used=5,
            max_steps=20,
            tokens_used=1000,
            max_tokens=32768,
        )

        # trajectory_recall = 2/2 = 1.0 (found both gold, even if pruned)
        assert metrics.trajectory_recall == 1.0
        # output_recall = 0/2 = 0.0 (no gold in final context)
        assert metrics.output_recall == 0.0

    def test_turn_penalty_same_start_end(self) -> None:
        """Turn penalty should handle start == end without division by zero."""
        calc = RewardCalculator(
            turn_penalty_start=10,
            turn_penalty_end=10,  # Same as start
            turn_penalty_max=0.5,
        )

        # Should not crash
        penalty = calc.compute_turn_penalty(15)
        assert penalty == 0.5  # Should be max penalty

    def test_empty_gold_chunks(self) -> None:
        """Reward with empty gold chunks should handle gracefully."""
        calc = RewardCalculator()
        tracker = TrajectoryTracker()

        metrics = calc.calculate_reward(
            tracker=tracker,
            gold_chunks=set(),  # Empty
            gold_answer="test",
            predicted_answer="test",
            context_texts=[],
            steps_used=0,
            max_steps=20,
            tokens_used=0,
            max_tokens=32768,
        )

        # Should produce valid metrics
        assert metrics is not None
        assert isinstance(metrics.total_reward, float)


class TestBetaScheduler:
    """Tests for BetaScheduler curriculum learning."""

    def test_warmup_phase(self) -> None:
        """Beta should stay at start value during warmup."""
        scheduler = BetaScheduler(
            start_beta=4.0,
            end_beta=1.0,
            warmup_steps=100,
            decay_steps=1000,
        )

        # During warmup, beta should be at start value
        assert scheduler.get_beta(0) == 4.0
        assert scheduler.get_beta(50) == 4.0
        assert scheduler.get_beta(99) == 4.0

    def test_decay_phase(self) -> None:
        """Beta should decay after warmup."""
        scheduler = BetaScheduler(
            start_beta=4.0,
            end_beta=1.0,
            warmup_steps=100,
            decay_steps=1000,
        )

        # After warmup, beta should start decaying
        beta_at_warmup = scheduler.get_beta(100)
        beta_mid_decay = scheduler.get_beta(500)
        beta_end_decay = scheduler.get_beta(1100)

        assert beta_at_warmup <= 4.0
        assert beta_mid_decay < beta_at_warmup
        assert beta_end_decay == 1.0  # At end, should be at end_beta

    def test_beta_never_below_end(self) -> None:
        """Beta should never go below end_beta."""
        scheduler = BetaScheduler(
            start_beta=4.0,
            end_beta=1.0,
            warmup_steps=10,
            decay_steps=100,
        )

        # Way past decay period
        beta = scheduler.get_beta(10000)
        assert beta >= 1.0
