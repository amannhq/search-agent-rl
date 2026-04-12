"""Tests for task loading and levels."""

from typing import List

import pytest

from searcharena import SearchTask
from sample import (
    get_sample_tasks,
    get_sample_tasks_by_level,
    get_sample_tasks_by_domain,
    get_sample_statistics,
)


class TestTaskLoading:
    """Tests for loading tasks from per-seed JSON files."""

    def test_get_sample_tasks_returns_list(self) -> None:
        """get_sample_tasks should return a list of SearchTask objects."""
        tasks = get_sample_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) > 0

    def test_get_task_by_id_found(self) -> None:
        """Should be able to find a task by ID."""
        tasks = get_sample_tasks()
        task_id = tasks[0].task_id

        found = [t for t in tasks if t.task_id == task_id]

        assert len(found) == 1
        assert found[0].task_id == task_id

    def test_get_tasks_by_level(self) -> None:
        """get_sample_tasks_by_level should filter correctly."""
        level_0 = get_sample_tasks_by_level(0)
        level_1 = get_sample_tasks_by_level(1)
        level_2 = get_sample_tasks_by_level(2)

        assert all(t.level == 0 for t in level_0)
        assert all(t.level == 1 for t in level_1)
        assert all(t.level == 2 for t in level_2)

    def test_get_tasks_by_domain(self) -> None:
        """get_sample_tasks_by_domain should filter correctly."""
        tech_tasks = get_sample_tasks_by_domain("tech")

        assert len(tech_tasks) > 0
        assert all(t.domain == "tech" for t in tech_tasks)

    def test_get_sample_statistics(self) -> None:
        """get_sample_statistics should return valid stats."""
        stats = get_sample_statistics()

        assert "total_tasks" in stats
        assert "by_level" in stats
        assert "domains" in stats
        assert "files" in stats
        assert stats["total_tasks"] > 0


class TestTaskLevels:
    """Tests for task levels."""

    def test_level_0_tasks_exist(self, tasks: List[SearchTask]) -> None:
        """Should have level 0 tasks."""
        level_0_tasks = [t for t in tasks if t.level == 0]
        assert len(level_0_tasks) >= 1

    def test_level_1_tasks_exist(self, tasks: List[SearchTask]) -> None:
        """Should have level 1 tasks."""
        level_1_tasks = [t for t in tasks if t.level == 1]
        assert len(level_1_tasks) >= 1

    def test_level_2_tasks_exist(self, tasks: List[SearchTask]) -> None:
        """Should have level 2 tasks."""
        level_2_tasks = [t for t in tasks if t.level == 2]
        assert len(level_2_tasks) >= 1

    def test_tasks_have_required_fields(self, tasks: List[SearchTask]) -> None:
        """All tasks should have required fields."""
        for task in tasks:
            assert task.task_id
            assert task.question
            assert task.truth
            assert len(task.supporting_items) > 0

    def test_higher_level_tasks_require_multiple_chunks(self, tasks: List[SearchTask]) -> None:
        """Higher level tasks should typically require multiple supporting items."""
        level_2_tasks = [t for t in tasks if t.level == 2]
        multi_chunk_tasks = [t for t in level_2_tasks if len(t.supporting_items) > 1]
        # At least some level 2 tasks should need multiple chunks
        assert len(multi_chunk_tasks) >= 1

    def test_task_domains_are_diverse(self, tasks: List[SearchTask]) -> None:
        """Tasks should cover multiple domains."""
        domains = set(t.domain for t in tasks)
        assert len(domains) >= 3  # At least 3 different domains


class TestTaskQuality:
    """Tests for task quality and consistency."""

    def test_questions_are_non_empty(self, tasks: List[SearchTask]) -> None:
        """All task questions should be non-empty strings."""
        for task in tasks:
            assert isinstance(task.question, str)
            assert len(task.question.strip()) > 10  # Meaningful length

    def test_truth_answers_are_non_empty(self, tasks: List[SearchTask]) -> None:
        """All truth answers should be non-empty."""
        for task in tasks:
            assert isinstance(task.truth, str)
            assert len(task.truth.strip()) > 0

    def test_task_ids_are_unique(self, tasks: List[SearchTask]) -> None:
        """All task IDs should be unique."""
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_supporting_item_ids_format(self, tasks: List[SearchTask]) -> None:
        """Supporting item IDs should follow expected format."""
        for task in tasks:
            for item in task.supporting_items:
                # Should contain 'chunk' reference
                assert "chunk" in item.id.lower() or "_" in item.id
