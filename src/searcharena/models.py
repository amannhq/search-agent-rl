"""Pydantic models for actions, observations, tasks, and config."""

from enum import Enum
from typing import Any, Dict, List, Optional

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
    terminated: bool = Field(default=False, description="Episode reached a terminal outcome")
    truncated: bool = Field(default=False, description="Episode stopped due to a non-terminal limit")
    termination_reason: Optional[str] = Field(
        default=None,
        description="Why the episode ended, if known",
    )

    # Search history (for deduplication awareness)
    queries_issued: List[str] = Field(
        default_factory=list, description="All queries made"
    )
    chunks_seen_count: int = Field(
        default=0, description="Total unique chunks encountered"
    )


class SupportingItem(BaseModel):
    """A supporting item from data generation."""

    id: str = Field(..., description="Item/chunk identifier")
    reasoning: str = Field(default="", description="Why this item supports the answer")

    # Quote fields added by verification
    clue_quotes: List[str] = Field(default_factory=list)
    item_quotes: List[str] = Field(default_factory=list)
    truth_quotes: List[str] = Field(default_factory=list)
    contains_truth: bool = Field(default=False)
    not_relevant: bool = Field(default=False)


class DistractorItem(BaseModel):
    """A distractor item that doesn't contain the truth."""

    id: str = Field(..., description="Item/chunk identifier")
    reasoning: str = Field(default="", description="Why this is a good distractor")
    contains_truth: bool = Field(default=False)


class SearchTask(BaseModel):
    """
    A single search task/episode.

    Uses the same field names as data generator output for direct compatibility.
    """

    # Core fields (matching generator output)
    question: str = Field(..., description="The multi-hop question to answer")
    truth: str = Field(..., description="Ground truth answer")
    clues: str = Field(default="", description="Clues text that hints at the answer")
    supporting_items: List[SupportingItem] = Field(
        default_factory=list, description="Items that support the answer"
    )
    items_and_contents: Dict[str, str] = Field(
        default_factory=dict, description="Map of item IDs to their content"
    )

    # Distractor fields
    valid_distractors: List[DistractorItem] = Field(
        default_factory=list, description="Distractor items that don't contain truth"
    )
    distractors_and_contents: Dict[str, str] = Field(
        default_factory=dict, description="Map of distractor IDs to content"
    )

    # Task metadata
    task_id: str = Field(default="", description="Unique task identifier")
    level: int = Field(default=0, description="Task level (0=base, 1+=extended)")
    domain: str = Field(default="general", description="Task domain")
    truth_type: str = Field(default="", description="Type of truth")

    # Verification fields
    passed_verification: bool = Field(default=True, description="Whether task passed verification")


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
    snippet_length: int = Field(
        default=200, description="Characters in search snippets"
    )

    # Deduplication
    deduplicate_searches: bool = Field(default=True, description="Exclude seen chunks")

    # Retrieval backend configuration
    token_estimation_chars_per_token: int = Field(
        default=4,
        description="Fallback token estimation ratio for heuristic token counting",
    )
    rerank_top_k: int = Field(
        default=0,
        description="Optional reranker depth; 0 disables reranking",
    )
