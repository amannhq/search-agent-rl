"""Server entrypoint for the search environment."""

import argparse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SearchAction, SearchEnvConfig, SearchObservation
    from .environment import (
        SearchEnvironment,
        create_sample_corpus,
        create_sample_tasks,
    )
except ImportError:
    from models import SearchAction, SearchEnvConfig, SearchObservation
    from server.environment import (
        SearchEnvironment,
        create_sample_corpus,
        create_sample_tasks,
    )


def create_environment() -> SearchEnvironment:
    """Create one environment instance."""
    config = SearchEnvConfig()
    return SearchEnvironment(
        config=config,
        corpus=create_sample_corpus(config),
        tasks=create_sample_tasks(),
    )


app = create_app(
    create_environment,
    SearchAction,
    SearchObservation,
    env_name="search_env",
    max_concurrent_envs=4,
)


def main() -> None:
    """Run the server directly."""
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
