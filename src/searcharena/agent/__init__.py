"""CLI inference module for SearchArena."""


def cli() -> None:
    """Run the SearchArena inference CLI lazily."""
    from .inference import cli as _cli

    _cli()


def main() -> None:
    """Run the SearchArena inference entrypoint lazily."""
    from .inference import main as _main

    _main()


__all__ = ["main", "cli"]
