"""System prompts for training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import SearchObservation, SearchTask


# Default system prompts for different modes
SYSTEM_PROMPTS = {
    "default": """You are a search agent that retrieves information to answer questions.

You have access to these actions:
- search(query, top_k): Search the corpus for relevant documents
- read(chunk_ids): Read full content of specific chunks
- prune(chunk_ids): Remove chunks from your context to free space
- answer(answer, supporting_chunk_ids): Submit your final answer

Strategy:
1. Analyze the question to identify key concepts
2. Search for relevant documents using targeted queries
3. Read promising chunks to gather evidence
4. Prune irrelevant chunks to manage your context budget
5. Submit your answer with supporting evidence

Be efficient with your context budget. Focus on finding the most relevant information.""",

    "exploration": """You are a search agent learning to retrieve information efficiently.

Available actions:
- search(query, top_k): Search for documents
- read(chunk_ids): Read chunk contents
- prune(chunk_ids): Remove chunks from context
- answer(answer, supporting_chunk_ids): Submit answer

Explore different search strategies. Try:
- Broad queries to discover relevant topics
- Specific queries to find precise information
- Iterative refinement based on what you find

Learn from each search result to improve your next query.""",

    "exploitation": """You are an expert search agent optimizing for retrieval efficiency.

Available actions:
- search(query, top_k): Search for documents
- read(chunk_ids): Read chunk contents
- prune(chunk_ids): Remove chunks from context
- answer(answer, supporting_chunk_ids): Submit answer

Optimize your search strategy:
- Use precise, targeted queries
- Minimize unnecessary reads
- Aggressively prune irrelevant content
- Submit confident answers quickly""",

    "curriculum_easy": """You are learning to search for information.

Actions available:
- search(query): Find documents matching your query
- read(chunk_ids): Read the full content of chunks
- answer(answer): Submit your answer

For this task:
- Start with a simple search for the main topic
- Read the most relevant results
- Answer based on what you found""",

    "curriculum_hard": """You are an advanced search agent handling complex multi-hop questions.

Actions:
- search(query, top_k): Search corpus
- read(chunk_ids): Read chunks
- prune(chunk_ids): Remove chunks
- answer(answer, supporting_chunk_ids): Submit answer

This question requires multiple reasoning steps:
1. Decompose the question into sub-questions
2. Search for each piece of evidence separately
3. Combine information across documents
4. Verify your answer is supported by evidence""",
}


@dataclass
class PromptBuilder:
    """Builds prompts for training and inference."""

    system_prompt: str = field(default_factory=lambda: SYSTEM_PROMPTS["default"])
    include_budget_info: bool = True
    include_history: bool = True
    max_history_items: int = 5

    def build_system_prompt(
        self,
        mode: str = "default",
        custom_additions: str | None = None,
    ) -> str:
        """Build the system prompt for a given mode."""
        base = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["default"])

        if custom_additions:
            base = f"{base}\n\n{custom_additions}"

        return base

    def build_observation_prompt(
        self,
        observation: "SearchObservation",
        task: "SearchTask | None" = None,
    ) -> str:
        """Build user prompt from observation state."""
        _ = task
        parts = []

        # Question
        parts.append(f"Question: {observation.question}")
        parts.append("")

        # Budget info
        if self.include_budget_info:
            parts.append(f"Context Budget: {observation.budget_usage_percent:.1f}% used "
                        f"({observation.context_token_count}/{observation.context_token_budget} tokens)")
            if observation.budget_warning:
                parts.append(f"Warning: {observation.budget_warning}")
            parts.append("")

        # Step info
        parts.append(f"Step: {observation.step_count}/{observation.max_steps}")
        parts.append("")

        # Current context
        if observation.context_chunks:
            parts.append(f"Current Context ({len(observation.context_chunks)} chunks):")
            for chunk in observation.context_chunks:
                parts.append(f"  - [{chunk.chunk_id}] {chunk.title}: {chunk.snippet[:100]}...")
            parts.append("")

        # Last action result
        if observation.action_result:
            parts.append("Last Action Result:")
            if observation.action_type == "search":
                results = observation.action_result.get("results", [])
                parts.append(f"  Found {len(results)} results for: {observation.action_result.get('query', '')}")
                for r in results[:3]:
                    parts.append(f"    - [{r.get('chunk_id', '')}] score={r.get('score', 0):.3f}")
            elif observation.action_type == "read":
                parts.append(f"  Added {observation.action_result.get('tokens_added', 0)} tokens")
            elif observation.action_type == "prune":
                parts.append(f"  Freed {observation.action_result.get('tokens_freed', 0)} tokens")
            parts.append("")

        # Search history
        if self.include_history and observation.queries_issued:
            recent = observation.queries_issued[-self.max_history_items:]
            parts.append(f"Recent Queries: {recent}")
            parts.append("")

        return "\n".join(parts)

    def build_full_prompt(
        self,
        observation: "SearchObservation",
        mode: str = "default",
        task: "SearchTask | None" = None,
    ) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt)."""
        system = self.build_system_prompt(mode)
        user = self.build_observation_prompt(observation, task)
        return system, user
