"""Contracts for package layout and entrypoints."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path


def test_searcharena_is_the_canonical_package() -> None:
    """The runtime package should import from src/searcharena."""
    package = import_module("searcharena")
    assert hasattr(package, "SearchEnvironment")
    assert hasattr(package, "create_sample_tasks")


def test_pyproject_entrypoints_target_searcharena() -> None:
    """Console scripts should resolve into the searcharena package."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'server = "searcharena.server.app:main"' in text
    assert 'inference = "searcharena.cli.inference:main"' in text
