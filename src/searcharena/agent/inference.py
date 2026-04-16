"""
Inference script for the Search RL Environment.

Submission system injects: API_BASE_URL, API_KEY, MODEL_NAME
Environment connection: LOCAL_IMAGE_NAME or ENV_BASE_URL
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from openai import AsyncOpenAI

from ..client import SearchEnv
from ..models import SearchAction
from .config import InferenceConfig
from .logging import log_start, log_end
from .runner import run_episode


class LocalEnvWrapper:
    """Run SearchEnvironment in-process without HTTP server."""

    def __init__(self) -> None:
        from searcharena import (
            SearchEnvironment,
            SearchEnvConfig,
            create_sample_corpus,
            create_sample_tasks,
        )
        config = SearchEnvConfig()
        self._env = SearchEnvironment(
            config=config,
            corpus=create_sample_corpus(config),
            tasks=create_sample_tasks(),
        )

    async def reset(self, task_id: str | None = None, **kwargs: Any) -> Any:
        from openenv.core.client_types import StepResult
        task = None
        if task_id:
            for t in self._env.tasks:
                if t.task_id == task_id:
                    task = t
                    break
        obs = self._env.reset(task=task, **kwargs)
        return StepResult(observation=obs, reward=0.0, done=False)

    async def step(self, action: SearchAction) -> Any:
        from openenv.core.client_types import StepResult
        obs = self._env.step(action)
        return StepResult(
            observation=obs,
            reward=obs.reward if obs.reward is not None else 0.0,
            done=obs.done,
        )

    async def close(self) -> None:
        self._env.close()


async def create_env(config: InferenceConfig) -> Any:
    """Create environment based on configuration."""
    if config.local_image_name:
        env_keys = ["MAX_STEPS", "MAX_CONTEXT_TOKENS", "SEARCH_TOP_K"]
        env_vars = {k: v for k in env_keys if (v := os.getenv(k))}
        return await SearchEnv.from_docker_image(
            config.local_image_name, env_vars=env_vars
        )

    if config.env_base_url:
        env = SearchEnv(base_url=config.env_base_url)
        await env.connect()
        return env

    return LocalEnvWrapper()


async def run_main() -> None:
    """Main entry point for inference."""
    config = InferenceConfig()
    config.validate()

    print(
        f"Config: base_url={config.api_base_url} model={config.model_name} "
        f"api_key={'set' if config.api_key else 'MISSING'}",
        file=sys.stderr,
        flush=True,
    )
    print(f"Tasks: {config.task_ids}", file=sys.stderr, flush=True)

    env = None
    scores: dict[str, float] = {}

    try:
        client = AsyncOpenAI(
            base_url=config.api_base_url,
            api_key=config.api_key,
        )
        env = await create_env(config)

        for task_id in config.task_ids:
            try:
                success, steps, score, rewards = await run_episode(
                    env, client, config, task_id=task_id
                )
                scores[task_id] = score
            except Exception as e:
                print(f"Task {task_id} failed: {e}", file=sys.stderr, flush=True)
                log_end(success=False, steps=0, score=0.001, rewards=[])
                scores[task_id] = 0.001

        print("\n--- SUMMARY ---", flush=True)
        for tid, sc in scores.items():
            print(f"  {tid}: {sc:.3f}", flush=True)
        print(f"  Average: {sum(scores.values()) / len(scores):.3f}", flush=True)

    except Exception as e:
        print(f"Fatal error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        log_start(task="error", env=config.benchmark, model=config.model_name)
        log_end(success=False, steps=0, score=0.001, rewards=[])
        raise

    finally:
        if env:
            try:
                await env.close()
            except Exception:
                pass


def cli() -> None:
    """CLI entry point."""
    asyncio.run(run_main())


def main() -> None:
    """Console-script entrypoint."""
    cli()


if __name__ == "__main__":
    cli()
