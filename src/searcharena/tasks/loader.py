"""Load generated task datasets from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import SearchTask


def _load_json(file_path: Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tasks_from_directory(
    directory: str | Path,
    verified_only: bool = True,
    min_level: int | None = None,
    max_level: int | None = None,
) -> list[SearchTask]:
    """Load tasks from a generator output directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_tasks: list[SearchTask] = []
    task_files = [path for path in directory.glob("*.json") if not path.name.startswith("index_")]

    for task_file in task_files:
        try:
            data = _load_json(task_file)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: Skipping {task_file}: {exc}")
            continue

        for task_data in data.get("tasks", []):
            if verified_only and not task_data.get("passed_verification", False):
                continue

            level = task_data.get("level", 0)
            if min_level is not None and level < min_level:
                continue
            if max_level is not None and level > max_level:
                continue

            if "task_id" not in task_data:
                task_data["task_id"] = f"{task_file.stem}_level_{level}"

            task_data.setdefault("domain", _infer_domain(task_file, data))

            try:
                all_tasks.append(SearchTask(**task_data))
            except Exception as exc:  # pragma: no cover - defensive data loading
                print(f"Warning: Skipping invalid task in {task_file}: {exc}")

    return all_tasks


def load_verified_tasks(directory: str | Path) -> list[SearchTask]:
    """Load only verified tasks."""
    return load_tasks_from_directory(directory, verified_only=True)


def load_tasks_by_level(
    directory: str | Path,
    level: int,
    verified_only: bool = True,
) -> list[SearchTask]:
    """Load tasks at a specific level."""
    return load_tasks_from_directory(
        directory,
        verified_only=verified_only,
        min_level=level,
        max_level=level,
    )


def get_directory_statistics(directory: str | Path) -> dict[str, Any]:
    """Summarize tasks in a generated-data directory."""
    directory = Path(directory)
    if not directory.exists():
        return {"error": f"Directory not found: {directory}"}

    stats: dict[str, Any] = {
        "total_files": 0,
        "total_tasks": 0,
        "verified_tasks": 0,
        "unverified_tasks": 0,
        "by_level": {},
        "by_domain": {},
        "files_with_errors": [],
    }

    task_files = [path for path in directory.glob("*.json") if not path.name.startswith("index_")]
    stats["total_files"] = len(task_files)

    for task_file in task_files:
        try:
            data = _load_json(task_file)
        except Exception as exc:  # pragma: no cover - defensive data loading
            stats["files_with_errors"].append({"file": str(task_file), "error": str(exc)})
            continue

        for task_data in data.get("tasks", []):
            stats["total_tasks"] += 1
            if task_data.get("passed_verification", False):
                stats["verified_tasks"] += 1
            else:
                stats["unverified_tasks"] += 1

            level = task_data.get("level", 0)
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

            domain = task_data.get("domain", _infer_domain(task_file, data))
            stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1

    return stats


def _infer_domain(task_file: Path, data: dict[str, Any]) -> str:
    """Infer task domain from file contents or naming patterns."""
    if "domain" in data:
        return data["domain"]
    if "ticker" in data:
        return "sec"

    for task in data.get("tasks", []):
        for item in task.get("supporting_items", []):
            item_id = item.get("id", "")
            if item_id.startswith(("http://", "https://")):
                return "web"
            if item_id.startswith("thread_") or ("_" in item_id and item_id.split("_")[0].isdigit()):
                return "epstein"

    name = task_file.stem
    if name.isupper() and len(name) <= 5:
        return "sec"
    if name.isdigit():
        return "epstein"
    return "web"
