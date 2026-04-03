"""Search RL Environment Client."""

from typing import Any, Dict, List, Optional
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

    def _step_payload(self, action: SearchAction) -> Dict[str, Any]:
        """
        Convert SearchAction to JSON payload for step message.

        Args:
            action: SearchAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict[str, Any] = {"action_type": action.action_type.value}

        if action.search is not None:
            payload["search"] = {
                "query": action.search.query,
                "top_k": action.search.top_k,
            }
        if action.read is not None:
            payload["read"] = {"chunk_ids": action.read.chunk_ids}
        if action.prune is not None:
            payload["prune"] = {"chunk_ids": action.prune.chunk_ids}
        if action.answer is not None:
            payload["answer"] = {
                "answer": action.answer.answer,
                "supporting_chunk_ids": action.answer.supporting_chunk_ids,
            }

        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SearchObservation]:
        """
        Parse server response into StepResult[SearchObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SearchObservation
        """
        obs_data = payload.get("observation", {})

        # Parse context chunks
        context_chunks = []
        for chunk_data in obs_data.get("context_chunks", []):
            context_chunks.append(
                ChunkSummary(
                    chunk_id=chunk_data.get("chunk_id", ""),
                    document_id=chunk_data.get("document_id", ""),
                    title=chunk_data.get("title", ""),
                    snippet=chunk_data.get("snippet", ""),
                    score=chunk_data.get("score", 0.0),
                    token_count=chunk_data.get("token_count", 0),
                )
            )

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
            queries_issued=obs_data.get("queries_issued", []),
            chunks_seen_count=obs_data.get("chunks_seen_count", 0),
            search_backend=obs_data.get("search_backend", "sample"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    async def search(
        self, query: str, top_k: int = 10
    ) -> StepResult[SearchObservation]:
        """
        Issue a search query.

        Args:
            query: Natural language search query
            top_k: Number of results to retrieve

        Returns:
            StepResult with search results in observation.action_result
        """
        action = SearchAction.make_search(query, top_k)
        return await self.step(action)

    async def read(self, chunk_ids: List[str]) -> StepResult[SearchObservation]:
        """
        Read full content of chunks into context.

        Args:
            chunk_ids: List of chunk IDs to read

        Returns:
            StepResult with read result in observation.action_result
        """
        action = SearchAction.make_read(chunk_ids)
        return await self.step(action)

    async def prune(self, chunk_ids: List[str]) -> StepResult[SearchObservation]:
        """
        Remove chunks from context to free token budget.

        Args:
            chunk_ids: List of chunk IDs to remove

        Returns:
            StepResult with prune result in observation.action_result
        """
        action = SearchAction.make_prune(chunk_ids)
        return await self.step(action)

    async def answer(
        self,
        answer: str,
        supporting_chunk_ids: Optional[List[str]] = None,
    ) -> StepResult[SearchObservation]:
        """
        Submit final answer and end episode.

        Args:
            answer: The agent's final answer
            supporting_chunk_ids: Chunk IDs that support the answer

        Returns:
            StepResult with final reward and evaluation metrics
        """
        action = SearchAction.make_answer(answer, supporting_chunk_ids)
        return await self.step(action)
