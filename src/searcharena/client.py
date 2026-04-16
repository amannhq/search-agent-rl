"""Search RL Environment Client."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    ChunkSummary,
    SearchAction,
    SearchObservation,
)


class SearchEnv(EnvClient[SearchAction, SearchObservation, State]):
    """
    Client for the Search RL Environment.

    This client provides convenient methods for all action types:
    - search(): Issue a search query
    - read(): Read full content of chunks
    - prune(): Remove chunks from context
    - answer(): Submit final answer

    Example:
        >>> with SearchEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(f"Question: {result.observation.question}")
        ...
        ...     # Search for relevant documents
        ...     result = env.search("Facebook Instagram acquisition")
        ...     for chunk in result.observation.action_result["results"][:3]:
        ...         print(f"Found: {chunk['snippet'][:50]}...")
        ...
        ...     # Read the top result
        ...     chunk_id = result.observation.action_result["results"][0]["chunk_id"]
        ...     result = env.read([chunk_id])
        ...
        ...     # Submit answer
        ...     result = env.answer("Mark Zuckerberg", [chunk_id])
        ...     print(f"Reward: {result.reward}")

    Example with Docker:
        >>> client = SearchEnv.from_docker_image("search-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.search("test query")
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SearchAction) -> dict[str, Any]:
        """Convert SearchAction to JSON payload for step message."""
        return action.model_dump(mode="json", exclude_none=True, exclude={"metadata"})

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[SearchObservation]:
        """Parse server response into StepResult[SearchObservation]."""
        obs_data = payload.get("observation", {})

        context_chunks = [
            ChunkSummary.model_validate(c)
            for c in obs_data.get("context_chunks", [])
        ]

        observation = SearchObservation(
            question=obs_data.get("question", ""),
            context_chunks=context_chunks,
            context_token_count=obs_data.get("context_token_count", 0),
            context_token_budget=obs_data.get("context_token_budget", 32768),
            budget_usage_percent=obs_data.get("budget_usage_percent", 0.0),
            budget_warning=obs_data.get("budget_warning"),
            action_result=obs_data.get("action_result"),
            action_type=obs_data.get("action_type"),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 20),
            terminated=obs_data.get("terminated", False),
            truncated=obs_data.get("truncated", False),
            termination_reason=obs_data.get("termination_reason"),
            queries_issued=obs_data.get("queries_issued", []),
            chunks_seen_count=obs_data.get("chunks_seen_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    async def search(
        self, query: str, top_k: int = 10
    ) -> StepResult[SearchObservation]:
        """Issue a search query."""
        return await self.step(SearchAction.make_search(query, top_k))

    async def read(self, chunk_ids: list[str]) -> StepResult[SearchObservation]:
        """Read full content of chunks into context."""
        return await self.step(SearchAction.make_read(chunk_ids))

    async def prune(self, chunk_ids: list[str]) -> StepResult[SearchObservation]:
        """Remove chunks from context to free token budget."""
        return await self.step(SearchAction.make_prune(chunk_ids))

    async def answer(
        self,
        answer: str,
        supporting_chunk_ids: list[str] | None = None,
    ) -> StepResult[SearchObservation]:
        """Submit final answer and end episode."""
        return await self.step(SearchAction.make_answer(answer, supporting_chunk_ids))
