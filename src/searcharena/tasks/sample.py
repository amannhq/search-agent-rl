"""Bundled sample tasks and corpus helpers."""

from __future__ import annotations

import json
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Any

from ..models import Chunk, SearchEnvConfig, SearchTask
from ..retrieval import DocumentCorpus

_SAMPLE_PACKAGE = "searcharena.tasks.resources.sample"
_cached_tasks: list[SearchTask] | None = None


def _iter_sample_files() -> list[Traversable]:
    sample_root = files(_SAMPLE_PACKAGE)
    return sorted(
        (entry for entry in sample_root.iterdir() if entry.name.endswith(".json")),
        key=lambda entry: entry.name,
    )


def _load_json(file_path: Traversable) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_tasks_from_files() -> list[SearchTask]:
    all_tasks: list[SearchTask] = []
    for task_file in _iter_sample_files():
        try:
            data = _load_json(task_file)
        except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - defensive
            print(f"Warning: Skipping {task_file}: {exc}")
            continue

        for task_data in data.get("tasks", []):
            try:
                all_tasks.append(SearchTask(**task_data))
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Warning: Skipping invalid task in {task_file}: {exc}")
    return all_tasks


def get_sample_tasks() -> list[SearchTask]:
    """Return bundled sample tasks."""
    global _cached_tasks
    if _cached_tasks is None:
        _cached_tasks = _load_tasks_from_files()
    return list(_cached_tasks)


def get_sample_tasks_by_level(level: int) -> list[SearchTask]:
    """Return bundled tasks for one level."""
    return [task for task in get_sample_tasks() if task.level == level]


def get_sample_tasks_by_domain(domain: str) -> list[SearchTask]:
    """Return bundled tasks for one domain."""
    return [task for task in get_sample_tasks() if task.domain == domain]


def get_sample_statistics() -> dict[str, Any]:
    """Return summary stats for bundled tasks."""
    all_tasks = get_sample_tasks()
    by_level: dict[int, int] = {}
    for task in all_tasks:
        by_level[task.level] = by_level.get(task.level, 0) + 1

    return {
        "total_tasks": len(all_tasks),
        "by_level": by_level,
        "domains": sorted({task.domain for task in all_tasks}),
        "files": [path.name for path in _iter_sample_files()],
    }


def create_sample_tasks() -> list[SearchTask]:
    """Compatibility helper for creating bundled tasks."""
    return get_sample_tasks()


def create_sample_corpus(
    config: SearchEnvConfig | None = None,
) -> DocumentCorpus:
    """Create a corpus that exactly mirrors the bundled sample tasks."""
    config_dict = config.model_dump() if config is not None else None
    corpus = DocumentCorpus(config=config_dict)

    seen_chunks: set[str] = set()
    for task in get_sample_tasks():
        for chunk_id, content in task.items_and_contents.items():
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            corpus.add_chunk(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=chunk_id.rsplit("_", 1)[0],
                    content=content,
                    metadata={"source": "sample", "domain": task.domain},
                )
            )

        for chunk_id, content in task.distractors_and_contents.items():
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            corpus.add_chunk(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=chunk_id.rsplit("_", 1)[0],
                    content=content,
                    metadata={
                        "source": "sample",
                        "domain": task.domain,
                        "distractor": True,
                    },
                )
            )

    return corpus
