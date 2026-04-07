"""
Search RL Environment Implementation.

A reinforcement learning environment for training agents to perform
multi-hop document retrieval tasks with explicit context management.
"""

from typing import Any, Dict, List, Optional, Set
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from .retrieval import DocumentCorpus
from .rewards import BetaScheduler, RewardCalculator, RewardMetrics, TrajectoryTracker

try:
    from .tasks import get_all_tasks, get_documents, get_task_statistics
except ImportError:
    from server.tasks import get_all_tasks, get_documents, get_task_statistics

try:
    from ..models import (
        ActionType,
        AnswerResult,
        Chunk,
        ChunkSummary,
        PruneResult,
        ReadResult,
        SearchAction,
        SearchEnvConfig,
        SearchObservation,
        SearchResult,
        SearchTask,
    )
except ImportError:
    from models import (
        ActionType,
        AnswerResult,
        Chunk,
        ChunkSummary,
        PruneResult,
        ReadResult,
        SearchAction,
        SearchEnvConfig,
        SearchObservation,
        SearchResult,
        SearchTask,
    )


class SearchEnvironment(Environment):
    """
    Search RL Environment for training agentic search models.

    The agent must:
    1. Issue search queries to find relevant documents
    2. Read documents to add them to context
    3. Prune irrelevant documents to manage token budget
    4. Submit a final answer based on retrieved evidence

    Rewards are based on F-beta score, trajectory recall, answer retrieval,
    and efficiency/degeneracy penalties.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        config: Optional[SearchEnvConfig] = None,
        corpus: Optional[DocumentCorpus] = None,
        tasks: Optional[List[SearchTask]] = None,
    ):
        """
        Initialize the Search RL Environment.

        Args:
            config: Environment configuration
            corpus: Pre-loaded document corpus (or will create empty one)
            tasks: List of tasks to sample from
        """
        super().__init__()
        self.config = config or SearchEnvConfig()
        self.corpus = corpus or DocumentCorpus(config=self.config.model_dump())
        self.tasks = tasks or []
        self._task_index = 0

        # Reward calculator
        self.reward_calculator = RewardCalculator(
            beta=self.config.beta,
            f_beta_weight=self.config.f_beta_weight,
            answer_reward_weight=self.config.answer_reward_weight,
            trajectory_reward_weight=self.config.trajectory_reward_weight,
            successful_trajectory_floor=self.config.successful_trajectory_floor,
            use_trajectory_reward=self.config.use_trajectory_reward,
        )
        self.beta_scheduler = (
            BetaScheduler(
                start_beta=self.config.beta_schedule_start,
                end_beta=self.config.beta_schedule_end,
                warmup_steps=self.config.beta_schedule_warmup_steps,
                decay_steps=self.config.beta_schedule_decay_steps,
            )
            if self.config.use_beta_schedule
            else None
        )

        # Episode state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: Optional[SearchTask] = None
        self._tracker = TrajectoryTracker()
        self._context_chunks: Dict[str, Chunk] = {}  # chunk_id -> Chunk
        self._context_token_count: int = 0
        self._chunks_seen: Set[str] = set()  # For deduplication
        self._seen_texts: List[
            str
        ] = []  # Track snippet/content for content-based matching
        self._done: bool = False
        self._last_metrics: Optional[RewardMetrics] = None

    def set_corpus(self, corpus: DocumentCorpus) -> None:
        """Set the document corpus."""
        self.corpus = corpus

    def set_tasks(self, tasks: List[SearchTask]) -> None:
        """Set the task list."""
        self.tasks = tasks
        self._task_index = 0

    def add_task(self, task: SearchTask) -> None:
        """Add a task to the task list."""
        self.tasks.append(task)

    def _get_next_task(self) -> Optional[SearchTask]:
        """Get the next task in sequence (cycles through tasks)."""
        if not self.tasks:
            return None
        task = self.tasks[self._task_index % len(self.tasks)]
        self._task_index += 1
        return task

    def _get_budget_warning(self) -> Optional[str]:
        if self.config.max_context_tokens <= 0:
            return None
        usage = self._context_token_count / self.config.max_context_tokens

        if usage >= self.config.hard_budget_threshold:
            return (
                f"HARD LIMIT: Context at {usage:.0%} capacity. "
                "Only prune or answer actions allowed."
            )
        elif usage >= self.config.soft_budget_threshold:
            return (
                f"WARNING: Context at {usage:.0%} capacity. "
                "Consider pruning irrelevant chunks or submitting answer."
            )
        return None

    def _create_observation(
        self,
        action_result: Optional[Dict[str, Any]] = None,
        action_type: Optional[str] = None,
        reward: float = 0.0,
    ) -> SearchObservation:
        """Create observation from current state."""
        # Create chunk summaries for context
        context_summaries = []
        for chunk in self._context_chunks.values():
            summary = ChunkSummary(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.metadata.get("title", chunk.document_id),
                snippet=chunk.content[: self.config.snippet_length] + "..."
                if len(chunk.content) > self.config.snippet_length
                else chunk.content,
                score=chunk.retrieval_score or 0.0,
                token_count=chunk.token_count,
            )
            context_summaries.append(summary)

        budget_usage = (
            self._context_token_count / self.config.max_context_tokens
            if self.config.max_context_tokens > 0
            else 0.0
        )

        return SearchObservation(
            question=self._current_task.question if self._current_task else "",
            context_chunks=context_summaries,
            context_token_count=self._context_token_count,
            context_token_budget=self.config.max_context_tokens,
            budget_usage_percent=budget_usage * 100,
            budget_warning=self._get_budget_warning(),
            action_result=action_result,
            action_type=action_type,
            step_count=self._state.step_count,
            max_steps=self.config.max_steps,
            queries_issued=list(self._tracker.queries),
            chunks_seen_count=len(self._chunks_seen),
            done=self._done,
            reward=reward,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[SearchTask] = None,
        **kwargs: Any,
    ) -> SearchObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            episode_id: Optional episode identifier
            task: Specific task to use (if None, samples from task list)

        Returns:
            Initial observation with question and empty context
        """
        # Reset state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._tracker.reset()
        self._context_chunks.clear()
        self._context_token_count = 0
        self._chunks_seen.clear()
        self._seen_texts.clear()
        self._done = False
        self._last_metrics = None
        self._configure_reward_beta(**kwargs)

        # Get task
        if task is not None:
            self._current_task = task
        else:
            self._current_task = self._get_next_task()

        if self._current_task is None:
            # No tasks available - create a dummy observation
            return SearchObservation(
                question="No tasks available. Please add tasks to the environment.",
                done=True,
                reward=0.0,
            )

        return self._create_observation()

    def step(
        self,
        action: SearchAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SearchObservation:
        """
        Execute an action in the environment.

        Args:
            action: SearchAction with action_type and payload

        Returns:
            SearchObservation with action result and updated state
        """
        if self._done:
            return self._create_observation(
                action_result={"error": "Episode already finished"},
                action_type=action.action_type.value,
                reward=0.0,
            )

        self._state.step_count += 1
        reward = 0.0
        action_result: Dict[str, Any] = {}

        # Check step limit
        if self._state.step_count >= self.config.max_steps:
            self._done = True
            # Force answer with empty response
            action_result = self._handle_answer("", [])
            reward = self._last_metrics.total_reward if self._last_metrics else 0.0
            return self._create_observation(
                action_result=action_result,
                action_type="timeout",
                reward=reward,
            )

        # Check budget for non-prune/answer actions
        if self.config.max_context_tokens > 0:
            budget_usage = self._context_token_count / self.config.max_context_tokens
        else:
            budget_usage = 0.0
        if (
            budget_usage >= self.config.hard_budget_threshold
            and action.action_type not in [ActionType.PRUNE, ActionType.ANSWER]
        ):
            return self._create_observation(
                action_result={
                    "error": "Token budget exceeded. Only prune or answer allowed."
                },
                action_type=action.action_type.value,
                reward=-0.1,  # Small penalty for invalid action
            )

        # Dispatch to action handler
        if action.action_type == ActionType.SEARCH:
            if action.search is None:
                action_result = {"error": "Missing search payload"}
            else:
                action_result = self._handle_search(
                    action.search.query, action.search.top_k
                )

        elif action.action_type == ActionType.READ:
            if action.read is None:
                action_result = {"error": "Missing read payload"}
            else:
                action_result = self._handle_read(action.read.chunk_ids)

        elif action.action_type == ActionType.PRUNE:
            if action.prune is None:
                action_result = {"error": "Missing prune payload"}
            else:
                action_result = self._handle_prune(action.prune.chunk_ids)

        elif action.action_type == ActionType.ANSWER:
            if action.answer is None:
                action_result = {"error": "Missing answer payload"}
            else:
                action_result = self._handle_answer(
                    action.answer.answer, action.answer.supporting_chunk_ids
                )
                reward = self._last_metrics.total_reward if self._last_metrics else 0.0

        return self._create_observation(
            action_result=action_result,
            action_type=action.action_type.value,
            reward=reward,
        )

    def _handle_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """Handle search action."""
        # Determine chunks to exclude (for deduplication)
        exclude_ids = self._chunks_seen if self.config.deduplicate_searches else None

        try:
            results = self.corpus.search(
                query=query,
                top_k=top_k,
                exclude_ids=exclude_ids,
                snippet_length=self.config.snippet_length,
            )
        except Exception as exc:
            return {"error": str(exc)}

        # Track results
        chunk_ids = [r.chunk_id for r in results]
        self._tracker.record_search(query, chunk_ids)
        self._chunks_seen.update(chunk_ids)

        # Track snippets for content-based matching fallback in reward calculation
        for r in results:
            if r.snippet:
                self._seen_texts.append(r.snippet)

        # Create result
        search_result = SearchResult(
            query=query,
            results=results,
            total_found=len(results),
        )

        return search_result.model_dump()

    def _handle_read(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """Handle read action."""
        chunks_added: List[Chunk] = []
        tokens_added = 0
        budget_exceeded = False
        chunks_truncated = 0

        remaining_budget = self.config.max_context_tokens - self._context_token_count

        for chunk_id in chunk_ids:
            # Skip if already in context
            if chunk_id in self._context_chunks:
                continue

            try:
                chunk = self.corpus.get_chunk(chunk_id)
            except Exception as exc:
                return {"error": str(exc)}
            if chunk is None:
                continue

            # Check if chunk fits in budget
            if tokens_added + chunk.token_count > remaining_budget:
                budget_exceeded = True
                chunks_truncated += 1
                continue

            # Add to context
            self._context_chunks[chunk_id] = chunk
            self._context_token_count += chunk.token_count
            tokens_added += chunk.token_count
            chunks_added.append(chunk)

        # Track
        self._tracker.record_read([c.chunk_id for c in chunks_added])
        self._chunks_seen.update(chunk_ids)

        read_result = ReadResult(
            chunks=chunks_added,
            tokens_added=tokens_added,
            budget_exceeded=budget_exceeded,
            chunks_truncated=chunks_truncated,
        )

        return read_result.model_dump()

    def _handle_prune(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """Handle prune action."""
        chunks_removed = 0
        tokens_freed = 0
        invalid_ids: List[str] = []

        for chunk_id in chunk_ids:
            if chunk_id in self._context_chunks:
                chunk = self._context_chunks.pop(chunk_id)
                self._context_token_count -= chunk.token_count
                tokens_freed += chunk.token_count
                chunks_removed += 1
            else:
                invalid_ids.append(chunk_id)

        # Track
        self._tracker.record_prune(chunk_ids)

        prune_result = PruneResult(
            chunks_removed=chunks_removed,
            tokens_freed=tokens_freed,
            invalid_ids=invalid_ids,
        )

        return prune_result.model_dump()

    def _handle_answer(
        self, answer: str, supporting_chunk_ids: List[str]
    ) -> Dict[str, Any]:
        """Handle answer action - ends the episode."""
        self._done = True

        if self._current_task is None:
            answer_result = AnswerResult(
                answer_submitted=answer,
                final_reward=0.0,
            )
            return answer_result.model_dump()

        # Calculate reward
        gold_chunks = set(self._current_task.gold_chunk_ids)

        metrics = self.reward_calculator.calculate_reward(
            tracker=self._tracker,
            gold_chunks=gold_chunks,
            gold_answer=self._current_task.gold_answer,
            predicted_answer=answer,
            context_texts=[chunk.content for chunk in self._context_chunks.values()],
            steps_used=self._state.step_count,
            max_steps=self.config.max_steps,
            tokens_used=self._context_token_count,
            max_tokens=self.config.max_context_tokens,
            all_seen_texts=self._seen_texts if self._seen_texts else None,
        )

        self._last_metrics = metrics

        answer_result = AnswerResult(
            answer_submitted=answer,
            final_reward=metrics.total_reward,
            trajectory_recall=metrics.trajectory_recall,
            output_recall=metrics.output_recall,
            output_precision=metrics.output_precision,
            f_beta=metrics.f_beta,
            beta_used=metrics.beta,
            answer_correct=metrics.answer_correct,
            answer_found_in_context=metrics.answer_found_in_context,
            answer_similarity=metrics.answer_similarity,
            f_beta_reward=metrics.f_beta_reward,
            trajectory_reward=metrics.trajectory_reward,
            answer_reward=metrics.answer_reward,
            turn_penalty=metrics.turn_penalty,
            prune_penalty=metrics.prune_penalty,
            pre_penalty_reward=metrics.pre_penalty_reward,
            reward_floor=metrics.reward_floor,
        )

        return answer_result.model_dump()

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def get_metrics(self) -> Optional[RewardMetrics]:
        """Get the last computed reward metrics."""
        return self._last_metrics

    def _configure_reward_beta(self, **kwargs: Any) -> None:
        """Set the reward beta for the next episode."""
        reward_beta = kwargs.get("reward_beta")
        training_step = kwargs.get("training_step")

        if reward_beta is not None:
            self.reward_calculator.beta = float(reward_beta)
            return

        if self.beta_scheduler is not None and training_step is not None:
            self.reward_calculator.beta = self.beta_scheduler.get_beta(
                int(training_step)
            )
            return

        self.reward_calculator.beta = self.config.beta


def create_sample_corpus(
    config: Optional[SearchEnvConfig] = None,
) -> DocumentCorpus:
    """
    Create a sample corpus for testing.

    Documents are loaded from the tasks module for better organization.
    """
    config_dict = config.model_dump() if config is not None else None
    corpus = DocumentCorpus(config=config_dict)

    # Load documents from tasks module
    documents = get_documents()

    for doc in documents:
        corpus.add_document(
            doc_id=doc["doc_id"],
            content=doc["content"],
            metadata=doc["metadata"],
            chunk_size=500,
            chunk_overlap=50,
        )

    return corpus


def create_sample_tasks() -> List[SearchTask]:
    """
    Create sample tasks for testing.

    Tasks are loaded from the tasks module which organizes them by difficulty.

    Tasks follow the Context-1 paper style:
    - Obfuscated clues (don't mention entities directly)
    - Short, verifiable answers (exist verbatim in documents)
    - Multi-constraint questions requiring decomposition

    Includes easy, medium, and hard difficulties across multiple domains.
    """
    return get_all_tasks()


if __name__ == "__main__":
    # Test the environment
    print("Creating sample corpus and tasks...")
    corpus = create_sample_corpus()
    tasks = create_sample_tasks()

    # Print task statistics
    stats = get_task_statistics()
    print(f"Corpus: {stats['num_documents']} documents, {corpus.num_chunks} chunks")
    print(f"Tasks: {stats['total_tasks']} total")
    print(f"  - Easy: {stats['by_difficulty']['easy']}")
    print(f"  - Medium: {stats['by_difficulty']['medium']}")
    print(f"  - Hard: {stats['by_difficulty']['hard']}")
    print(f"  - Domains: {stats['domains']}")

    # Create environment
    env = SearchEnvironment(corpus=corpus, tasks=tasks)

    # Helper to safely get from action_result
    def get_result(obs: SearchObservation, key: str, default: Any = None) -> Any:
        if obs.action_result is None:
            return default
        return obs.action_result.get(key, default)

    # Run a test episode
    print("\n--- Test Episode ---")
    obs = env.reset()
    print(f"Question: {obs.question}")
    print(f"Budget: {obs.context_token_count}/{obs.context_token_budget}")

    # Search
    action = SearchAction.make_search("Facebook Instagram acquisition")
    obs = env.step(action)
    results = get_result(obs, "results", [])
    print(f"\nSearch results: {len(results)} chunks")
    for r in results[:3]:
        print(f"  - {r['chunk_id']}: {r['snippet'][:50]}...")

    # Read first result
    if results:
        chunk_id = results[0]["chunk_id"]
        action = SearchAction.make_read([chunk_id])
        obs = env.step(action)
        print(f"\nRead: added {get_result(obs, 'tokens_added', 0)} tokens")
        print(f"Budget: {obs.context_token_count}/{obs.context_token_budget}")

    # Search for founder
    action = SearchAction.make_search("Mark Zuckerberg born birthplace")
    obs = env.step(action)
    results = get_result(obs, "results", [])
    print(f"\nSearch results: {len(results)} chunks")

    # Read
    if results:
        chunk_id = results[0]["chunk_id"]
        action = SearchAction.make_read([chunk_id])
        obs = env.step(action)
        print(f"Read: added {get_result(obs, 'tokens_added', 0)} tokens")

    # Answer
    action = SearchAction.make_answer(
        "Mark Zuckerberg was born in White Plains, New York",
        supporting_chunk_ids=list(env._context_chunks.keys()),
    )
    obs = env.step(action)
    print("\n--- Episode Complete ---")
    print(f"Answer correct: {get_result(obs, 'answer_correct')}")
    print(f"F-beta: {get_result(obs, 'f_beta', 0):.3f}")
    print(f"Trajectory recall: {get_result(obs, 'trajectory_recall', 0):.3f}")
    print(f"Output recall: {get_result(obs, 'output_recall', 0):.3f}")
    print(f"Final reward: {get_result(obs, 'final_reward', 0):.3f}")

    # Verify reward is in valid range: [0.0, 1.0] (clamped per OpenEnv spec)
    reward = get_result(obs, "final_reward", 0)
    assert 0.0 <= reward <= 1.0, f"Reward {reward} out of bounds [0.0, 1.0]!"
    print(f"\n✓ Reward {reward:.3f} is within valid range [0.0, 1.0]")
