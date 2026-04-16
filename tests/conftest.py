"""Shared pytest fixtures for the Search RL Environment tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path before project imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from collections.abc import Callable

import pytest

from searcharena import (
    DocumentCorpus,
    SearchEnvConfig,
    SearchEnvironment,
    SearchTask,
    create_sample_corpus,
    create_sample_tasks,
)


@pytest.fixture(scope="session")
def corpus() -> DocumentCorpus:
    """Create a sample document corpus (shared across all tests)."""
    return create_sample_corpus()


@pytest.fixture(scope="session")
def tasks() -> list[SearchTask]:
    """Create sample tasks (shared across all tests)."""
    return create_sample_tasks()


@pytest.fixture
def env(corpus: DocumentCorpus, tasks: list[SearchTask]) -> SearchEnvironment:
    """Create a configured environment with corpus and tasks."""
    return SearchEnvironment(corpus=corpus, tasks=tasks)


@pytest.fixture
def env_with_config(
    corpus: DocumentCorpus, tasks: list[SearchTask]
) -> Callable[[SearchEnvConfig], SearchEnvironment]:
    """Factory fixture to create environment with custom config."""

    def _create(config: SearchEnvConfig) -> SearchEnvironment:
        return SearchEnvironment(config=config, corpus=corpus, tasks=tasks)

    return _create


@pytest.fixture
def empty_corpus() -> DocumentCorpus:
    """Create an empty document corpus."""
    return DocumentCorpus()


@pytest.fixture
def custom_task() -> SearchTask:
    """Create a custom task for testing."""
    from searcharena.models import SupportingItem

    return SearchTask(
        task_id="custom_task",
        question="What is the meaning of life?",
        truth="42",
        supporting_items=[
            SupportingItem(id="doc_test_chunk_0", reasoning="Contains the answer")
        ],
        items_and_contents={"doc_test_chunk_0": "The answer to life is 42."},
        level=0,
    )
