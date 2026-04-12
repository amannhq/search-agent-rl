"""Core RL environment class."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .retrieval import DocumentCorpus
from .rewards import BetaScheduler, RewardCalculator, RewardMetrics, TrajectoryTracker
from .observations import create_observation
from .handlers import handle_search, handle_read, handle_prune, handle_answer
from ..models import (
    ActionType,
    Chunk,
    SearchAction,
    SearchEnvConfig,
    SearchObservation,
    SearchTask,
)


class SearchEnvironment(Environment):
    """
    Search RL environment. Agent searches, reads, prunes, and answers.
    Rewards based on F-beta score and trajectory recall.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        config: SearchEnvConfig | None = None,
        corpus: DocumentCorpus | None = None,
        tasks: list[SearchTask] | None = None,
    ):
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
        self._current_task: SearchTask | None = None
        self._tracker = TrajectoryTracker()
        self._context_chunks: dict[str, Chunk] = {}
        self._context_token_count: int = 0
        self._chunks_seen: set[str] = set()
        self._seen_texts: list[str] = []
        self._done: bool = False
        self._last_metrics: RewardMetrics | None = None

    def set_corpus(self, corpus: DocumentCorpus) -> None:
        """Set the document corpus."""
        self.corpus = corpus

    def set_tasks(self, tasks: list[SearchTask]) -> None:
        """Set the task list."""
        self.tasks = tasks
        self._task_index = 0

    def add_task(self, task: SearchTask) -> None:
        """Add a task to the task list."""
        self.tasks.append(task)

    def _get_next_task(self) -> SearchTask | None:
        """Get the next task in sequence (cycles through tasks)."""
        if not self.tasks:
            return None
        task = self.tasks[self._task_index % len(self.tasks)]
        self._task_index += 1
        return task

    @property
    def _budget_usage(self) -> float:
        if self.config.max_context_tokens <= 0:
            return 0.0
        return self._context_token_count / self.config.max_context_tokens

    def _create_observation(
        self,
        action_result: dict[str, Any] | None = None,
        action_type: str | None = None,
        reward: float = 0.0,
    ) -> SearchObservation:
        """Create observation from current state."""
        return create_observation(
            current_task=self._current_task,
            context_chunks=self._context_chunks,
            context_token_count=self._context_token_count,
            context_token_budget=self.config.max_context_tokens,
            snippet_length=self.config.snippet_length,
            soft_budget_threshold=self.config.soft_budget_threshold,
            hard_budget_threshold=self.config.hard_budget_threshold,
            step_count=self._state.step_count,
            max_steps=self.config.max_steps,
            tracker=self._tracker,
            chunks_seen=self._chunks_seen,
            done=self._done,
            action_result=action_result,
            action_type=action_type,
            reward=reward,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task: SearchTask | None = None,
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
        _ = seed

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
            return SearchObservation(
                question="No tasks available. Please add tasks to the environment.",
                done=True,
                reward=0.0,
            )

        return self._create_observation()

    def step(
        self,
        action: SearchAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SearchObservation:
        """
        Execute an action in the environment.

        Args:
            action: SearchAction with action_type and payload

        Returns:
            SearchObservation with action result and updated state
        """
        _ = timeout_s, kwargs

        if self._done:
            return self._create_observation(
                action_result={"error": "Episode already finished"},
                action_type=action.action_type.value,
                reward=0.0,
            )

        self._state.step_count += 1
        reward = 0.0
        action_result: dict[str, Any] = {}

        # Check step limit
        if self._state.step_count >= self.config.max_steps:
            self._done = True
            action_result, self._last_metrics = handle_answer(
                answer="",
                supporting_chunk_ids=[],
                reward_calculator=self.reward_calculator,
                tracker=self._tracker,
                context_chunks=self._context_chunks,
                context_token_count=self._context_token_count,
                seen_texts=self._seen_texts,
                gold_chunks=set(item.id for item in self._current_task.supporting_items) if self._current_task else set(),
                gold_answer=self._current_task.truth if self._current_task else "",
                steps_used=self._state.step_count,
                max_steps=self.config.max_steps,
                max_tokens=self.config.max_context_tokens,
            )
            reward = self._last_metrics.total_reward if self._last_metrics else 0.0
            return self._create_observation(
                action_result=action_result,
                action_type="timeout",
                reward=reward,
            )

        # Check budget constraint
        if (
            self._budget_usage >= self.config.hard_budget_threshold
            and action.action_type not in [ActionType.PRUNE, ActionType.ANSWER]
        ):
            return self._create_observation(
                action_result={
                    "error": "Token budget exceeded. Only prune or answer allowed."
                },
                action_type=action.action_type.value,
                reward=-0.1,
            )

        # Dispatch action
        if action.action_type == ActionType.SEARCH:
            if action.search is None:
                action_result = {"error": "Missing search payload"}
            else:
                action_result = handle_search(
                    query=action.search.query,
                    top_k=action.search.top_k,
                    corpus=self.corpus,
                    tracker=self._tracker,
                    chunks_seen=self._chunks_seen,
                    seen_texts=self._seen_texts,
                    deduplicate=self.config.deduplicate_searches,
                    snippet_length=self.config.snippet_length,
                )

        elif action.action_type == ActionType.READ:
            if action.read is None:
                action_result = {"error": "Missing read payload"}
            else:
                action_result, self._context_token_count = handle_read(
                    chunk_ids=action.read.chunk_ids,
                    corpus=self.corpus,
                    tracker=self._tracker,
                    context_chunks=self._context_chunks,
                    context_token_count=self._context_token_count,
                    chunks_seen=self._chunks_seen,
                    max_context_tokens=self.config.max_context_tokens,
                )

        elif action.action_type == ActionType.PRUNE:
            if action.prune is None:
                action_result = {"error": "Missing prune payload"}
            else:
                action_result, self._context_token_count = handle_prune(
                    chunk_ids=action.prune.chunk_ids,
                    tracker=self._tracker,
                    context_chunks=self._context_chunks,
                    context_token_count=self._context_token_count,
                )

        elif action.action_type == ActionType.ANSWER:
            if action.answer is None:
                action_result = {"error": "Missing answer payload"}
            else:
                self._done = True
                if self._current_task is None:
                    action_result = {"answer_submitted": action.answer.answer, "final_reward": 0.0}
                else:
                    action_result, self._last_metrics = handle_answer(
                        answer=action.answer.answer,
                        supporting_chunk_ids=action.answer.supporting_chunk_ids,
                        reward_calculator=self.reward_calculator,
                        tracker=self._tracker,
                        context_chunks=self._context_chunks,
                        context_token_count=self._context_token_count,
                        seen_texts=self._seen_texts,
                        gold_chunks=set(item.id for item in self._current_task.supporting_items),
                        gold_answer=self._current_task.truth,
                        steps_used=self._state.step_count,
                        max_steps=self.config.max_steps,
                        max_tokens=self.config.max_context_tokens,
                    )
                    reward = self._last_metrics.total_reward if self._last_metrics else 0.0

        return self._create_observation(
            action_result=action_result,
            action_type=action.action_type.value,
            reward=reward,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def get_metrics(self) -> RewardMetrics | None:
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
