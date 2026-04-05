"""Shared pytest fixtures for the Search RL Environment tests."""

import sys
from pathlib import Path
from typing import Callable, List

import pytest

from models import SearchEnvConfig, SearchTask
from server.environment import (
    SearchEnvironment,
    create_sample_corpus,
    create_sample_tasks,
)
from server.retrieval import DocumentCorpus

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def corpus() -> DocumentCorpus:
    """Create a sample document corpus."""
    return create_sample_corpus()


@pytest.fixture
def tasks() -> List[SearchTask]:
    """Create sample tasks."""
    return create_sample_tasks()


@pytest.fixture
def env(corpus: DocumentCorpus, tasks: List[SearchTask]) -> SearchEnvironment:
    """Create a configured environment with corpus and tasks."""
    return SearchEnvironment(corpus=corpus, tasks=tasks)


@pytest.fixture
def env_with_config(
    corpus: DocumentCorpus, tasks: List[SearchTask]
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
    return SearchTask(
        task_id="custom_task",
        question="What is the meaning of life?",
        gold_answer="42",
        gold_chunk_ids=["doc_test_chunk_0"],
    )
