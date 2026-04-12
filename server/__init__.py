"""
Server package - FastAPI wrapper for SearchArena.

This package contains only the server/API layer.
All core logic lives in the searcharena package.
"""

from .app import app, create_environment
from .environment import SearchEnvironment, create_sample_corpus, create_sample_tasks

__all__ = [
    "app",
    "create_environment",
    "SearchEnvironment",
    "create_sample_corpus",
    "create_sample_tasks",
]
