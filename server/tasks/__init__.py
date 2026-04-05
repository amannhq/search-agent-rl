"""
Task and corpus loader for the Search RL Environment.

Loads tasks and documents from JSON files for easy management and updates.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from models import SearchTask


# Path to data files
_TASKS_DIR = Path(__file__).parent
_CORPUS_FILE = _TASKS_DIR / "corpus.json"
_TASKS_FILE = _TASKS_DIR / "tasks.json"


def _load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(file_path: Path, data: Dict[str, Any]) -> None:
    """Save data to JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_documents() -> List[Dict[str, Any]]:
    """Load documents from corpus.json."""
    data = _load_json(_CORPUS_FILE)
    return data.get("documents", [])


def get_all_tasks() -> List[SearchTask]:
    """Load all tasks from tasks.json."""
    data = _load_json(_TASKS_FILE)
    return [SearchTask(**t) for t in data.get("tasks", [])]


def get_tasks_by_difficulty(difficulty: str) -> List[SearchTask]:
    """Get tasks filtered by difficulty (easy, medium, hard)."""
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return [t for t in get_all_tasks() if t.difficulty == difficulty]


def get_tasks_by_domain(domain: str) -> List[SearchTask]:
    """Get tasks filtered by domain."""
    return [t for t in get_all_tasks() if t.domain == domain]


def get_task_by_id(task_id: str) -> Optional[SearchTask]:
    """Get a specific task by ID."""
    for task in get_all_tasks():
        if task.task_id == task_id:
            return task
    return None


def get_task_statistics() -> Dict[str, Any]:
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


def add_document(doc: Dict[str, Any]) -> None:
    """Add a new document to corpus.json."""
    data = _load_json(_CORPUS_FILE)
    data["documents"].append(doc)
    _save_json(_CORPUS_FILE, data)


def remove_task(task_id: str) -> bool:
    """Remove a task by ID. Returns True if removed."""
    data = _load_json(_TASKS_FILE)
    original_count = len(data["tasks"])
    data["tasks"] = [t for t in data["tasks"] if t.get("task_id") != task_id]
    if len(data["tasks"]) < original_count:
        _save_json(_TASKS_FILE, data)
        return True
    return False


def remove_document(doc_id: str) -> bool:
    """Remove a document by ID. Returns True if removed."""
    data = _load_json(_CORPUS_FILE)
    original_count = len(data["documents"])
    data["documents"] = [d for d in data["documents"] if d.get("doc_id") != doc_id]
    if len(data["documents"]) < original_count:
        _save_json(_CORPUS_FILE, data)
        return True
    return False
