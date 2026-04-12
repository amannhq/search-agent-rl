"""Task and corpus data for the Search RL Environment.

Static data:
    - get_documents(): Load corpus documents
    - get_all_tasks(): Load all tasks
    - get_tasks_by_difficulty(): Filter by difficulty

Data generation (requires datagen extra):
    python -m data.generator.domains.web --seeds seeds.txt --output ./output
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from searcharena.models import SearchTask
except ImportError:
    from models import SearchTask


_DATA_DIR = Path(__file__).parent
_CORPUS_FILE = _DATA_DIR / "corpus.json"
_TASKS_FILE = _DATA_DIR / "tasks.json"

_cached_documents: list[dict[str, Any]] | None = None
_cached_tasks: list[SearchTask] | None = None


def _load_json(file_path: Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(file_path: Path, data: dict[str, Any]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _invalidate_cache() -> None:
    global _cached_documents, _cached_tasks
    _cached_documents = None
    _cached_tasks = None


def get_documents() -> list[dict[str, Any]]:
    """Load documents from corpus.json (cached after first call)."""
    global _cached_documents
    if _cached_documents is not None:
        return list(_cached_documents)
    result: list[dict[str, Any]] = _load_json(_CORPUS_FILE).get("documents", [])
    _cached_documents = result
    return list(result)


def get_all_tasks() -> list[SearchTask]:
    """Load all tasks from tasks.json (cached after first call)."""
    global _cached_tasks
    if _cached_tasks is not None:
        return list(_cached_tasks)
    result = [SearchTask(**t) for t in _load_json(_TASKS_FILE).get("tasks", [])]
    _cached_tasks = result
    return list(result)


def get_tasks_by_difficulty(difficulty: str) -> list[SearchTask]:
    """Get tasks filtered by difficulty (easy, medium, hard)."""
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return [t for t in get_all_tasks() if t.difficulty == difficulty]


def get_tasks_by_domain(domain: str) -> list[SearchTask]:
    """Get tasks filtered by domain."""
    return [t for t in get_all_tasks() if t.domain == domain]


def get_task_by_id(task_id: str) -> SearchTask | None:
    """Get a specific task by ID."""
    for task in get_all_tasks():
        if task.task_id == task_id:
            return task
    return None


def get_task_statistics() -> dict[str, Any]:
    """Get statistics about tasks and corpus."""
    all_tasks = get_all_tasks()
    documents = get_documents()

    difficulties = {"easy": 0, "medium": 0, "hard": 0}
    for task in all_tasks:
        if task.difficulty in difficulties:
            difficulties[task.difficulty] += 1

    return {
        "total_tasks": len(all_tasks),
        "by_difficulty": difficulties,
        "domains": list(set(t.domain for t in all_tasks)),
        "num_documents": len(documents),
    }


def add_task(task: SearchTask) -> None:
    """Add a new task to tasks.json."""
    data = _load_json(_TASKS_FILE)
    data["tasks"].append(task.model_dump())
    _save_json(_TASKS_FILE, data)
    _invalidate_cache()


def add_document(doc: dict[str, Any]) -> None:
    """Add a new document to corpus.json."""
    data = _load_json(_CORPUS_FILE)
    data["documents"].append(doc)
    _save_json(_CORPUS_FILE, data)
    _invalidate_cache()


def remove_task(task_id: str) -> bool:
    """Remove a task by ID. Returns True if removed."""
    data = _load_json(_TASKS_FILE)
    original = data["tasks"]
    data["tasks"] = [t for t in original if t.get("task_id") != task_id]
    if len(data["tasks"]) < len(original):
        _save_json(_TASKS_FILE, data)
        _invalidate_cache()
        return True
    return False


def remove_document(doc_id: str) -> bool:
    """Remove a document by ID. Returns True if removed."""
    data = _load_json(_CORPUS_FILE)
    original = data["documents"]
    data["documents"] = [d for d in original if d.get("doc_id") != doc_id]
    if len(data["documents"]) < len(original):
        _save_json(_CORPUS_FILE, data)
        _invalidate_cache()
        return True
    return False
