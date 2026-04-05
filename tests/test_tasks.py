"""Tests for task loading and difficulty levels."""

from typing import List

import pytest

from models import SearchTask
from server.tasks import (
    get_all_tasks,
    get_documents,
    get_task_by_id,
    get_task_statistics,
    get_tasks_by_difficulty,
    get_tasks_by_domain,
)


class TestTaskLoading:
    """Tests for loading tasks from JSON files."""

    def test_get_all_tasks_returns_list(self) -> None:
        """get_all_tasks should return a list of SearchTask objects."""
        tasks = get_all_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) > 0

    def test_get_documents_returns_list(self) -> None:
        """get_documents should return a list of document dicts."""
        docs = get_documents()
        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_get_task_by_id_found(self) -> None:
        """get_task_by_id should return task when ID exists."""
        tasks = get_all_tasks()
        task_id = tasks[0].task_id

        found = get_task_by_id(task_id)

        assert found is not None
        assert found.task_id == task_id

    def test_get_task_by_id_not_found(self) -> None:
        """get_task_by_id should return None for non-existent ID."""
        found = get_task_by_id("nonexistent_task_id_12345")
        assert found is None

    def test_get_tasks_by_difficulty(self) -> None:
        """get_tasks_by_difficulty should filter correctly."""
        easy = get_tasks_by_difficulty("easy")
        medium = get_tasks_by_difficulty("medium")
        hard = get_tasks_by_difficulty("hard")

        assert all(t.difficulty == "easy" for t in easy)
        assert all(t.difficulty == "medium" for t in medium)
        assert all(t.difficulty == "hard" for t in hard)

    def test_get_tasks_by_difficulty_invalid(self) -> None:
        """get_tasks_by_difficulty should raise for invalid difficulty."""
        with pytest.raises(ValueError):
            get_tasks_by_difficulty("invalid")

    def test_get_tasks_by_domain(self) -> None:
        """get_tasks_by_domain should filter correctly."""
        tech_tasks = get_tasks_by_domain("tech")

        assert len(tech_tasks) > 0
        assert all(t.domain == "tech" for t in tech_tasks)

    def test_get_task_statistics(self) -> None:
        """get_task_statistics should return valid stats."""
        stats = get_task_statistics()

        assert "total_tasks" in stats
        assert "by_difficulty" in stats
        assert "domains" in stats
        assert "num_documents" in stats
        assert stats["total_tasks"] > 0
        assert stats["num_documents"] > 0


class TestTaskDifficulty:
    """Tests for task difficulty levels."""

    def test_easy_tasks_exist(self, tasks: List[SearchTask]) -> None:
        """Should have easy difficulty tasks."""
        easy_tasks = [t for t in tasks if t.difficulty == "easy"]
        assert len(easy_tasks) >= 1

    def test_medium_tasks_exist(self, tasks: List[SearchTask]) -> None:
        """Should have medium difficulty tasks."""
        medium_tasks = [t for t in tasks if t.difficulty == "medium"]
        assert len(medium_tasks) >= 1

    def test_hard_tasks_exist(self, tasks: List[SearchTask]) -> None:
        """Should have hard difficulty tasks."""
        hard_tasks = [t for t in tasks if t.difficulty == "hard"]
        assert len(hard_tasks) >= 1

    def test_tasks_have_required_fields(self, tasks: List[SearchTask]) -> None:
        """All tasks should have required fields."""
        for task in tasks:
            assert task.task_id
            assert task.question
            assert task.gold_answer
            assert len(task.gold_chunk_ids) > 0

    def test_hard_tasks_require_multiple_chunks(self, tasks: List[SearchTask]) -> None:
        """Hard tasks should typically require multiple gold chunks."""
        hard_tasks = [t for t in tasks if t.difficulty == "hard"]
        multi_chunk_hard = [t for t in hard_tasks if len(t.gold_chunk_ids) > 1]
        # At least some hard tasks should need multiple chunks
        assert len(multi_chunk_hard) >= 1

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

    def test_gold_answers_are_non_empty(self, tasks: List[SearchTask]) -> None:
        """All gold answers should be non-empty."""
        for task in tasks:
            assert isinstance(task.gold_answer, str)
            assert len(task.gold_answer.strip()) > 0

    def test_task_ids_are_unique(self, tasks: List[SearchTask]) -> None:
        """All task IDs should be unique."""
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_gold_chunk_ids_format(self, tasks: List[SearchTask]) -> None:
        """Gold chunk IDs should follow expected format."""
        for task in tasks:
            for chunk_id in task.gold_chunk_ids:
                # Should contain 'chunk' and a document reference
                assert "chunk" in chunk_id
                assert "doc_" in chunk_id
