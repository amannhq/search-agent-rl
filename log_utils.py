"""Shared JSON log writers for inference and retrieval code."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def append_log_record(log_file: str, record: Dict[str, Any]) -> None:
    """Append one record to a JSON or JSONL log file."""
    if not log_file:
        return

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                existing = handle.read().strip()
                if not existing:
                    payload: List[Dict[str, Any]] = []
                else:
                    try:
                        loaded = json.loads(existing)
                        payload = loaded if isinstance(loaded, list) else [loaded]
                    except json.JSONDecodeError:
                        payload = [
                            json.loads(line)
                            for line in existing.splitlines()
                            if line.strip()
                        ]
                payload.append(record)
                handle.seek(0)
                handle.truncate()
                json.dump(payload, handle, ensure_ascii=True, indent=2)
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


async def append_log_record_async(log_file: str, record: Dict[str, Any]) -> None:
    """Write one record without blocking the event loop."""
    await asyncio.to_thread(append_log_record, log_file, record)
