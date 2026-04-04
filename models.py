"""
Data models for the Search RL Environment.

This environment trains agents to perform multi-hop document retrieval tasks.
Uses an F-beta-based reward curriculum for iterative retrieval.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A chunk of text from a document."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Text content of the chunk")
    token_count: int = Field(default=0, description="Approximate token count")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # Search scores (populated when retrieved)
    retrieval_score: Optional[float] = Field(
        default=None, description="Score from retrieval"
    )
    rerank_score: Optional[float] = Field(
        default=None, description="Score from reranker"
    )


class ChunkSummary(BaseModel):
    """Summary of a chunk (returned from search, not full content)."""

    chunk_id: str = Field(..., description="Unique identifier")
    document_id: str = Field(..., description="Parent document ID")
    title: str = Field(default="", description="Document title if available")
    snippet: str = Field(..., description="Short preview of content")
    score: float = Field(default=0.0, description="Retrieval score")
    token_count: int = Field(default=0, description="Token count if read")


class ActionType(str, Enum):
    """Types of actions the agent can take."""

    SEARCH = "search"
    READ = "read"
    PRUNE = "prune"
    ANSWER = "answer"


class SearchActionPayload(BaseModel):
    """Payload for search action."""

    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=10, description="Number of results to retrieve")


class ReadActionPayload(BaseModel):
    """Payload for read action."""

    chunk_ids: List[str] = Field(..., description="Chunk IDs to read in full")


class PruneActionPayload(BaseModel):
    """Payload for prune action."""

    chunk_ids: List[str] = Field(..., description="Chunk IDs to remove from context")


class AnswerActionPayload(BaseModel):
    """Payload for answer action."""

    answer: str = Field(..., description="The agent's final answer")
    supporting_chunk_ids: List[str] = Field(
        default_factory=list, description="Chunks supporting the answer"
    )


class SearchAction(Action):
    """
    Action for the Search RL Environment.

    The agent can:
    - search: Issue a search query to find relevant documents
    - read: Read full content of specific chunks
    - prune: Remove chunks from context to free token budget
    - answer: Submit final answer and end episode
    """

    action_type: ActionType = Field(..., description="Type of action to take")

    # Payloads for each action type (only one should be set)
    search: Optional[SearchActionPayload] = Field(
        default=None, description="Search parameters"
    )
    read: Optional[ReadActionPayload] = Field(
        default=None, description="Read parameters"
    )
    prune: Optional[PruneActionPayload] = Field(
        default=None, description="Prune parameters"
    )
    answer: Optional[AnswerActionPayload] = Field(
        default=None, description="Answer parameters"
    )

    @classmethod
    def make_search(cls, query: str, top_k: int = 10) -> "SearchAction":
        """Create a search action."""
        return cls(
            action_type=ActionType.SEARCH,
            search=SearchActionPayload(query=query, top_k=top_k),
        )

    @classmethod
    def make_read(cls, chunk_ids: List[str]) -> "SearchAction":
        """Create a read action."""
        return cls(
            action_type=ActionType.READ,
            read=ReadActionPayload(chunk_ids=chunk_ids),
        )

    @classmethod
    def make_prune(cls, chunk_ids: List[str]) -> "SearchAction":
        """Create a prune action."""
        return cls(
            action_type=ActionType.PRUNE,
            prune=PruneActionPayload(chunk_ids=chunk_ids),
        )

    @classmethod
    def make_answer(
        cls, answer: str, supporting_chunk_ids: Optional[List[str]] = None
    ) -> "SearchAction":
        """Create an answer action."""
        return cls(
            action_type=ActionType.ANSWER,
            answer=AnswerActionPayload(
                answer=answer, supporting_chunk_ids=supporting_chunk_ids or []
            ),
        )


class SearchResult(BaseModel):
    """Result of a search action."""

    query: str = Field(..., description="The query that was issued")
    results: List[ChunkSummary] = Field(
        default_factory=list, description="Search results"
    )
    total_found: int = Field(default=0, description="Total matching documents")


class ReadResult(BaseModel):
    """Result of a read action."""

    chunks: List[Chunk] = Field(default_factory=list, description="Full chunk content")
    tokens_added: int = Field(default=0, description="Tokens added to context")
    budget_exceeded: bool = Field(
        default=False, description="True if couldn't fit all chunks"
    )
    chunks_truncated: int = Field(
        default=0, description="Number of chunks that couldn't fit"
    )


class PruneResult(BaseModel):
    """Result of a prune action."""

    chunks_removed: int = Field(default=0, description="Number of chunks removed")
    tokens_freed: int = Field(default=0, description="Tokens freed from context")
    invalid_ids: List[str] = Field(
        default_factory=list, description="IDs that weren't in context"
    )


class AnswerResult(BaseModel):
    """Result of an answer action (episode end)."""

    answer_submitted: str = Field(..., description="The submitted answer")
    final_reward: float = Field(default=0.0, description="Final episode reward")

    # Evaluation metrics
    trajectory_recall: float = Field(
        default=0.0, description="% of gold chunks ever retrieved"
    )
    output_recall: float = Field(
        default=0.0, description="% of gold chunks in final context"
    )
    output_precision: float = Field(
        default=0.0, description="% of context that is gold"
    )
    f_beta: float = Field(default=0.0, description="F-beta score")
    beta_used: float = Field(
        default=4.0, description="Beta value used when computing F-beta"
    )
    answer_correct: bool = Field(
        default=False, description="Whether answer matches gold"
    )
    answer_found_in_context: bool = Field(
        default=False,
        description="Whether a kept chunk directly contains the gold answer",
    )
    answer_similarity: float = Field(
        default=0.0, description="Similarity between answer and gold answer"
    )
    f_beta_reward: float = Field(
        default=0.0, description="Weighted F-beta contribution to total reward"
    )
    trajectory_reward: float = Field(
        default=0.0, description="Weighted trajectory contribution to total reward"
    )
    answer_reward: float = Field(
        default=0.0, description="Evidence-backed answer bonus contribution"
    )
    turn_penalty: float = Field(
        default=0.0, description="Turn-count penalty applied to the episode"
    )
    prune_penalty: float = Field(
        default=0.0, description="Excessive-pruning penalty applied to the episode"
    )
    pre_penalty_reward: float = Field(
        default=0.0, description="Reward before penalties are subtracted"
    )
    reward_floor: float = Field(
        default=0.0, description="Lower clamp floor for completed trajectories"
    )


ActionResult = Union[SearchResult, ReadResult, PruneResult, AnswerResult]


class SearchObservation(Observation):
    """
    Observation from the Search RL Environment.

    Contains the current state visible to the agent after each action.
    """

    # The question to answer (always visible)
    question: str = Field(default="", description="The question to answer")

    # Current context state
    context_chunks: List[ChunkSummary] = Field(
        default_factory=list, description="Summaries of chunks in context"
    )
    context_token_count: int = Field(default=0, description="Current tokens in context")
    context_token_budget: int = Field(default=32768, description="Maximum token budget")

    # Budget status (for agent awareness)
    budget_usage_percent: float = Field(
        default=0.0, description="Percentage of budget used"
    )
    budget_warning: Optional[str] = Field(
        default=None, description="Warning if near limit"
    )

    # Last action result
    action_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Result of last action"
    )
    action_type: Optional[str] = Field(default=None, description="Type of last action")

    # Episode progress
    step_count: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=20, description="Maximum steps allowed")

    # Search history (for deduplication awareness)
    queries_issued: List[str] = Field(
        default_factory=list, description="All queries made"
    )
    chunks_seen_count: int = Field(
        default=0, description="Total unique chunks encountered"
    )


class SearchTask(BaseModel):
    """
    A single search task/episode.

    Defines the question, gold answer, and supporting evidence.
    """

    task_id: str = Field(..., description="Unique task identifier")
    question: str = Field(..., description="The multi-hop question to answer")
    gold_answer: str = Field(..., description="Ground truth answer")
    gold_chunk_ids: List[str] = Field(..., description="Chunk IDs containing evidence")

    # Task metadata
    difficulty: str = Field(default="medium", description="easy, medium, hard")
    num_hops: int = Field(default=1, description="Number of reasoning hops required")
    domain: str = Field(default="general", description="Task domain")

    # Optional
    clues: List[str] = Field(default_factory=list, description="Intermediate clues")
    distractor_chunk_ids: List[str] = Field(
        default_factory=list, description="Hard negative chunk IDs"
    )


class SearchEnvConfig(BaseModel):
    """Configuration for the Search RL Environment."""

    # Budget settings
    max_steps: int = Field(default=20, description="Maximum actions per episode")
    max_context_tokens: int = Field(default=32768, description="Context window budget")
    soft_budget_threshold: float = Field(
        default=0.75, description="Warn at this % usage"
    )
    hard_budget_threshold: float = Field(
        default=0.95, description="Block searches at this %"
    )

    # Reward settings
    beta: float = Field(
        default=4.0,
        description="F-beta parameter (paper-aligned default emphasizes recall)",
    )
    f_beta_weight: float = Field(
        default=0.7, description="Weight for the F-beta outcome term"
    )
    answer_reward_weight: float = Field(
        default=1.0, description="Weight for answer bonus"
    )
    use_trajectory_reward: bool = Field(
        default=True, description="Include trajectory recall"
    )
    trajectory_reward_weight: float = Field(
        default=0.3, description="Weight for the trajectory recall term"
    )
    successful_trajectory_floor: float = Field(
        default=0.01,
        description="Small positive floor for completed trajectories after penalties",
    )
    use_beta_schedule: bool = Field(
        default=False,
        description="Use BetaScheduler when reset() receives training_step",
    )
    beta_schedule_start: float = Field(
        default=4.0, description="Initial beta for curriculum training"
    )
    beta_schedule_end: float = Field(
        default=2.0, description="Final beta for curriculum training"
    )
    beta_schedule_warmup_steps: int = Field(
        default=1000, description="Warmup steps before beta decay begins"
    )
    beta_schedule_decay_steps: int = Field(
        default=10000, description="Steps over which beta decays to the final value"
    )

    # Retrieval settings
    search_top_k: int = Field(default=10, description="Default results per search")
    tokens_per_search: int = Field(
        default=4096, description="Max tokens per search result"
    )
    snippet_length: int = Field(
        default=200, description="Characters in search snippets"
    )

    # Deduplication
    deduplicate_searches: bool = Field(default=True, description="Exclude seen chunks")
