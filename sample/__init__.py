"""Sample/mock data for testing the Search RL Environment.

This module provides sample data that matches the exact format produced by
the data generators in data/generator/. Use this for testing and development.

Production file structure (one JSON per seed):
    sample/
    ├── instagram.json        # Tech: Instagram task (level 0)
    ├── whatsapp.json         # Tech: WhatsApp task (level 1)
    ├── facebook_acquisitions.json  # Tech: Multi-source task (level 2)
    ├── curie.json            # Science: Marie Curie task (level 0)
    └── berlin_wall.json      # History: Berlin Wall task (level 0)

Each file contains:
    {
        "seed": "topic_name",
        "domain": "tech|science|history",
        "tasks": [
            {"level": 0, "truth": "...", "supporting_items": [...], ...},
            {"level": 1, ...},  // extension tasks (if any)
        ]
    }

Usage:
    from sample import get_sample_tasks, get_sample_tasks_by_level

    tasks = get_sample_tasks()
    level_0_tasks = get_sample_tasks_by_level(0)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from searcharena.models import SearchTask
except ImportError:
    from models import SearchTask


_SAMPLE_DIR = Path(__file__).parent

_cached_tasks: list[SearchTask] | None = None


def _load_json(file_path: Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_tasks_from_files() -> list[SearchTask]:
    """Load all tasks from per-seed JSON files in sample/."""
    all_tasks: list[SearchTask] = []

    # Get all JSON files except __pycache__ etc
    task_files = list(_SAMPLE_DIR.glob("*.json"))

    for task_file in task_files:
        try:
            data = _load_json(task_file)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Skipping {task_file}: {e}")
            continue

        tasks_data = data.get("tasks", [])
        for task_data in tasks_data:
            try:
                task = SearchTask(**task_data)
                all_tasks.append(task)
            except Exception as e:
                print(f"Warning: Skipping invalid task in {task_file}: {e}")
                continue

    return all_tasks


def get_sample_tasks() -> list[SearchTask]:
    """Load all sample tasks from per-seed JSON files (cached after first call).

    Tasks match the format produced by the data generators:
    - level: int (0, 1, 2, ...)
    - truth: str (the ground truth answer)
    - supporting_items: list of SupportingItem
    - items_and_contents: dict mapping chunk IDs to content
    - valid_distractors: list of DistractorItem
    - distractors_and_contents: dict mapping distractor IDs to content
    - clues: str
    - truth_type: str
    - passed_verification: bool
    """
    global _cached_tasks
    if _cached_tasks is not None:
        return list(_cached_tasks)
    result = _load_tasks_from_files()
    _cached_tasks = result
    return list(result)


def get_sample_tasks_by_level(level: int) -> list[SearchTask]:
    """Get sample tasks filtered by level."""
    return [t for t in get_sample_tasks() if t.level == level]


def get_sample_tasks_by_domain(domain: str) -> list[SearchTask]:
    """Get sample tasks filtered by domain."""
    return [t for t in get_sample_tasks() if t.domain == domain]


def get_sample_statistics() -> dict[str, Any]:
    """Get statistics about sample tasks."""
    all_tasks = get_sample_tasks()

    by_level: dict[int, int] = {}
    for task in all_tasks:
        by_level[task.level] = by_level.get(task.level, 0) + 1

    return {
        "total_tasks": len(all_tasks),
        "by_level": by_level,
        "domains": list(set(t.domain for t in all_tasks)),
        "files": [f.name for f in _SAMPLE_DIR.glob("*.json")],
    }
