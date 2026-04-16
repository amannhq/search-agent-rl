"""Tests for the searcharena.training module."""

import pytest
from searcharena.training import (
    CurriculumScheduler,
    DifficultyLevel,
    EpisodeBuffer,
    EpisodeMetrics,
    MetricsLogger,
    PromptBuilder,
    TaskSampler,
    TrainingConfig,
    TrainingMetrics,
)
from searcharena.training.datasets import Episode


class TestCurriculumScheduler:
    """Tests for curriculum learning scheduler."""

    def test_initial_level_is_zero(self):
        """Scheduler starts at level 0."""
        scheduler = CurriculumScheduler()
        assert scheduler.current_level_int == 0

    def test_advance_after_success_threshold(self):
        """Scheduler advances after reaching success threshold."""
        scheduler = CurriculumScheduler(
            warmup_steps=0,
            advance_threshold=0.5,
            min_attempts_before_advance=5,
        )

        # Record successful episodes
        for _ in range(5):
            metrics = EpisodeMetrics(task_id="test", level=0, success=True)
            scheduler.record_episode(metrics, step=100)

        advanced = scheduler.maybe_advance()
        assert advanced
        assert scheduler.current_level_int == 1

    def test_no_advance_below_threshold(self):
        """Scheduler does not advance below success threshold."""
        scheduler = CurriculumScheduler(
            warmup_steps=0,
            advance_threshold=0.8,
            min_attempts_before_advance=5,
        )

        # Record mixed results (60% success)
        for i in range(5):
            metrics = EpisodeMetrics(
                task_id=f"test_{i}",
                level=0,
                success=i < 3,
            )
            scheduler.record_episode(metrics, step=100)

        advanced = scheduler.maybe_advance()
        assert not advanced
        assert scheduler.current_level_int == 0

    def test_get_level_weights(self):
        """Level weights sum to 1.0."""
        scheduler = CurriculumScheduler()
        weights = scheduler.get_level_weights()
        assert abs(sum(weights.values()) - 1.0) < 0.001


class TestEpisodeBuffer:
    """Tests for episode buffer."""

    def test_add_and_sample(self):
        """Can add and sample episodes."""
        buffer = EpisodeBuffer(max_size=100)

        for i in range(10):
            episode = Episode(
                task_id=f"task_{i}",
                total_reward=float(i),
                success=i > 5,
            )
            buffer.add(episode)

        assert buffer.size == 10
        sampled = buffer.sample(3)
        assert len(sampled) == 3

    def test_success_rate_tracking(self):
        """Buffer tracks success rate correctly."""
        buffer = EpisodeBuffer()

        # Add 4 successful, 6 failed
        for i in range(10):
            episode = Episode(task_id=f"task_{i}", success=i < 4)
            buffer.add(episode)

        assert abs(buffer.success_rate - 0.4) < 0.001

    def test_max_size_eviction(self):
        """Buffer evicts old episodes at max size."""
        buffer = EpisodeBuffer(max_size=5)

        for i in range(10):
            episode = Episode(task_id=f"task_{i}")
            buffer.add(episode)

        assert buffer.size == 5


class TestTrainingMetrics:
    """Tests for training metrics."""

    def test_from_episodes(self):
        """Can aggregate metrics from episodes."""
        episodes = [
            EpisodeMetrics(task_id="1", total_reward=0.5, success=True, level=0),
            EpisodeMetrics(task_id="2", total_reward=0.3, success=False, level=1),
            EpisodeMetrics(task_id="3", total_reward=0.8, success=True, level=0),
        ]

        metrics = TrainingMetrics.from_episodes(episodes, step=100)

        assert metrics.episodes_total == 3
        assert metrics.episodes_successful == 2
        assert abs(metrics.reward_mean - 0.533) < 0.01
        assert metrics.success_rate_by_level[0] == 1.0
        assert metrics.success_rate_by_level[1] == 0.0


class TestPromptBuilder:
    """Tests for prompt builder."""

    def test_system_prompts_exist(self):
        """All expected system prompts exist."""
        from searcharena.training.prompts import SYSTEM_PROMPTS

        expected = ["default", "exploration", "exploitation", "curriculum_easy", "curriculum_hard"]
        for key in expected:
            assert key in SYSTEM_PROMPTS

    def test_build_system_prompt(self):
        """Can build system prompts."""
        builder = PromptBuilder()
        prompt = builder.build_system_prompt("default")
        assert "search" in prompt.lower()
        assert len(prompt) > 100


class TestTaskSampler:
    """Tests for task sampler."""

    def test_sample_by_level(self):
        """Can sample tasks by level weights."""
        from searcharena.models import SearchTask, SupportingItem

        tasks = [
            SearchTask(task_id="1", question="Q1", truth="A1", supporting_items=[SupportingItem(id="c1", reasoning="r")], level=0),
            SearchTask(task_id="2", question="Q2", truth="A2", supporting_items=[SupportingItem(id="c2", reasoning="r")], level=1),
            SearchTask(task_id="3", question="Q3", truth="A3", supporting_items=[SupportingItem(id="c3", reasoning="r")], level=2),
        ]

        sampler = TaskSampler(tasks, seed=42)

        # Sample with 100% level 0 weight
        samples = sampler.sample_by_level(n=10, weights={0: 1.0, 1: 0.0, 2: 0.0})
        assert all(t.level == 0 for t in samples)

    def test_stratified_sampling(self):
        """Stratified sampling gets tasks from each level."""
        from searcharena.models import SearchTask, SupportingItem

        tasks = [
            SearchTask(task_id="1", question="Q1", truth="A1", supporting_items=[SupportingItem(id="c1", reasoning="r")], level=0),
            SearchTask(task_id="2", question="Q2", truth="A2", supporting_items=[SupportingItem(id="c2", reasoning="r")], level=1),
            SearchTask(task_id="3", question="Q3", truth="A3", supporting_items=[SupportingItem(id="c3", reasoning="r")], level=2),
        ]

        sampler = TaskSampler(tasks, seed=42)
        samples = sampler.sample_stratified(n_per_level=1)

        levels = {t.level for t in samples}
        assert levels == {0, 1, 2}


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = TrainingConfig()
        assert config.total_steps > 0
        assert config.learning_rate > 0
        assert config.batch_size > 0

    def test_config_serialization(self):
        """Config can be serialized and deserialized."""
        config = TrainingConfig(run_name="test-run", total_steps=500)
        data = config.to_dict()
        restored = TrainingConfig.from_dict(data)

        assert restored.run_name == "test-run"
        assert restored.total_steps == 500
