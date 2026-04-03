"""Server entrypoint for the search environment."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

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


load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

CONFIG_CASTERS = {
    "search_backend": str,
    "serper_api_key": str,
    "max_steps": int,
    "max_context_tokens": int,
    "search_top_k": int,
}
CONFIG_ENV_NAMES = {
    "search_backend": "SEARCH_BACKEND",
    "serper_api_key": "SERPER_API_KEY",
    "max_steps": "MAX_STEPS",
    "max_context_tokens": "MAX_CONTEXT_TOKENS",
    "search_top_k": "SEARCH_TOP_K",
}


def _build_config_from_env() -> SearchEnvConfig:
    """Build the environment config from a small set of supported env vars."""
    overrides = {}
    for field_name, env_name in CONFIG_ENV_NAMES.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        value = raw.strip()
        if not value:
            continue
        overrides[field_name] = CONFIG_CASTERS[field_name](value)
    return SearchEnvConfig().model_copy(update=overrides) if overrides else SearchEnvConfig()


def create_environment() -> SearchEnvironment:
    """Create one environment instance."""
    config = _build_config_from_env()
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
