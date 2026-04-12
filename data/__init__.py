"""Task data loading for the Search RL Environment.

Production data (from generators):
    - load_tasks_from_directory(path): Load per-seed task files from output/
    - load_verified_tasks(path): Load only tasks with passed_verification=True
    - load_tasks_by_level(path, level): Load tasks of a specific level

Data generation (requires datagen extra):
    python -m data.generator.domains.web --seeds seeds.txt --output ./output
    python -m data.generator.domains.sec --tickers tickers.txt --output ./output

Production file structure:
    output/
    ├── {seed}.json          # Web: one file per seed (e.g., machine_learning.json)
    ├── {TICKER}.json        # SEC: one file per ticker (e.g., AAPL.json)
    └── {number}.json        # Epstein: numbered files (0.json, 1.json, ...)

Each file contains:
    {
        "seed": "topic_name",
        "domain": "web|sec|epstein",
        "tasks": [
            {"level": 0, "truth": "...", "supporting_items": [...], ...},
            {"level": 1, ...},  // extension tasks
        ]
    }

For sample/mock data during development, use the sample module:
    from sample import get_sample_tasks, get_sample_tasks_by_level
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from searcharena.models import SearchTask
except ImportError:
    from models import SearchTask


def _load_json(file_path: Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tasks_from_directory(
    directory: str | Path,
    verified_only: bool = True,
    min_level: int | None = None,
    max_level: int | None = None,
) -> list[SearchTask]:
    """
    Load tasks from a directory of per-seed JSON files (production format).

    This is the format produced by the data generators:
    - Web domain: {seed}.json (e.g., machine_learning.json)
    - SEC domain: {TICKER}.json (e.g., AAPL.json)
    - Epstein domain: {number}.json (e.g., 0.json, 1.json)

    Each file contains a "tasks" array with level 0, 1, 2, etc.

    Args:
        directory: Path to the output directory containing task JSON files
        verified_only: Only include tasks with passed_verification=True
        min_level: Minimum task level to include (None = no minimum)
        max_level: Maximum task level to include (None = no maximum)

    Returns:
        List of SearchTask objects from all files in the directory
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_tasks: list[SearchTask] = []
    task_files = list(directory.glob("*.json"))

    # Exclude index output files
    task_files = [f for f in task_files if not f.name.startswith("index_")]

    for task_file in task_files:
        try:
            data = _load_json(task_file)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Skipping {task_file}: {e}")
            continue

        tasks_data = data.get("tasks", [])
        if not tasks_data:
            continue

        for task_data in tasks_data:
            # Filter by verification status
            if verified_only and not task_data.get("passed_verification", False):
                continue

            # Filter by level
            level = task_data.get("level", 0)
            if min_level is not None and level < min_level:
                continue
            if max_level is not None and level > max_level:
                continue

            # Generate task_id if not present
            if "task_id" not in task_data:
                task_data["task_id"] = f"{task_file.stem}_level_{level}"

            # Add source file info
            task_data.setdefault("domain", _infer_domain(task_file, data))

            try:
                task = SearchTask(**task_data)
                all_tasks.append(task)
            except Exception as e:
                print(f"Warning: Skipping invalid task in {task_file}: {e}")
                continue

    return all_tasks


def load_verified_tasks(directory: str | Path) -> list[SearchTask]:
    """
    Load only verified tasks from a production output directory.

    Shorthand for load_tasks_from_directory(directory, verified_only=True).
    """
    return load_tasks_from_directory(directory, verified_only=True)


def load_tasks_by_level(
    directory: str | Path,
    level: int,
    verified_only: bool = True,
) -> list[SearchTask]:
    """Load tasks of a specific level from a production output directory."""
    return load_tasks_from_directory(
        directory,
        verified_only=verified_only,
        min_level=level,
        max_level=level,
    )


def get_directory_statistics(directory: str | Path) -> dict[str, Any]:
    """
    Get statistics about tasks in a production output directory.

    Returns:
        Dictionary with counts by level, domain, verification status, etc.
    """
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

    task_files = [f for f in directory.glob("*.json") if not f.name.startswith("index_")]
    stats["total_files"] = len(task_files)

    for task_file in task_files:
        try:
            data = _load_json(task_file)
        except Exception as e:
            stats["files_with_errors"].append({"file": str(task_file), "error": str(e)})
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
    """Infer the domain from file content or naming patterns."""
    # Check if domain is explicitly set
    if "domain" in data:
        return data["domain"]

    # SEC domain has ticker field
    if "ticker" in data:
        return "sec"

    # Check for URL-based IDs (web domain)
    for task in data.get("tasks", []):
        for item in task.get("supporting_items", []):
            item_id = item.get("id", "")
            if item_id.startswith("http://") or item_id.startswith("https://"):
                return "web"
            if item_id.startswith("thread_") or "_" in item_id and item_id.split("_")[0].isdigit():
                return "epstein"

    # Fallback based on filename pattern
    name = task_file.stem
    if name.isupper() and len(name) <= 5:  # Looks like a ticker
        return "sec"
    if name.isdigit():  # Numbered file
        return "epstein"

    return "web"  # Default to web domain
