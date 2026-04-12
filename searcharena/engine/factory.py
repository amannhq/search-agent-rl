"""Factory functions for corpus and tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import SearchEnvConfig, SearchTask
    from .retrieval import DocumentCorpus


def create_sample_corpus(
    config: "SearchEnvConfig | None" = None,
) -> "DocumentCorpus":
    """
    Create a sample corpus for testing.

    Documents are loaded from the data module for better organization.
    """
    from .retrieval import DocumentCorpus

    # Data loading
    try:
        from data import get_documents
    except ImportError:
        from ...data import get_documents

    config_dict = config.model_dump() if config is not None else None
    corpus = DocumentCorpus(config=config_dict)

    documents = get_documents()

    for doc in documents:
        corpus.add_document(
            doc_id=doc["doc_id"],
            content=doc["content"],
            metadata=doc["metadata"],
            chunk_size=500,
            chunk_overlap=50,
        )

    return corpus


def create_sample_tasks() -> list["SearchTask"]:
    """
    Create sample tasks for testing.

    Tasks are loaded from the data module which organizes them by difficulty.

    Tasks follow the Context-1 paper style:
    - Obfuscated clues (don't mention entities directly)
    - Short, verifiable answers (exist verbatim in documents)
    - Multi-constraint questions requiring decomposition

    Includes easy, medium, and hard difficulties across multiple domains.
    """
    try:
        from data import get_all_tasks
    except ImportError:
        from ...data import get_all_tasks

    return get_all_tasks()
