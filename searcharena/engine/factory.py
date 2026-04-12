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

    Documents are built from sample task data (items_and_contents, distractors_and_contents).
    This ensures the corpus matches exactly what tasks expect.
    """
    from .retrieval import Chunk, DocumentCorpus

    try:
        from sample import get_sample_tasks
    except ImportError:
        from ...sample import get_sample_tasks

    config_dict = config.model_dump() if config is not None else None
    corpus = DocumentCorpus(config=config_dict)

    # Collect all unique chunks from tasks
    seen_chunks: set[str] = set()
    tasks = get_sample_tasks()

    for task in tasks:
        # Add supporting item contents
        for chunk_id, content in task.items_and_contents.items():
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                # Create chunk directly with the exact ID expected by tasks
                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=chunk_id.rsplit("_", 1)[0],  # Extract doc_id from chunk_id
                    content=content,
                    metadata={"source": "sample", "domain": task.domain},
                )
                corpus.add_chunk(chunk)

        # Add distractor contents
        if task.distractors_and_contents:
            for chunk_id, content in task.distractors_and_contents.items():
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        document_id=chunk_id.rsplit("_", 1)[0],
                        content=content,
                        metadata={"source": "sample", "domain": task.domain, "distractor": True},
                    )
                    corpus.add_chunk(chunk)

    return corpus


def create_sample_tasks() -> list["SearchTask"]:
    """
    Create sample tasks for testing.

    Tasks are loaded from the sample module which provides mock data
    matching the production generator format.

    Tasks follow the Context-1 paper style:
    - Obfuscated clues (don't mention entities directly)
    - Short, verifiable answers (exist verbatim in documents)
    - Multi-constraint questions requiring decomposition

    Includes levels 0, 1, 2 across multiple domains.
    """
    try:
        from sample import get_sample_tasks
    except ImportError:
        from ...sample import get_sample_tasks

    return get_sample_tasks()
