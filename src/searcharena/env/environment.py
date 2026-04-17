"""SearchArena environment orchestration."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata

from ..models import ActionType, SearchAction, SearchEnvConfig, SearchObservation, SearchTask
from ..retrieval import DocumentCorpus
from ..rewards import BetaScheduler, RewardCalculator, RewardMetrics
from ..training.datasets import TaskSampler
from .dispatcher import ActionDispatcher, DispatchResult
from .observations import render_observation
from .policies import BudgetPolicy, TerminationInfo, TerminationPolicy
from .state import EpisodeState


class SearchEnvironment(Environment):
    """Multi-hop search environment with explicit episode state."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(
        self,
        config: SearchEnvConfig | None = None,
        corpus: DocumentCorpus | None = None,
        tasks: list[SearchTask] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SearchEnvConfig()
        self.corpus = corpus or DocumentCorpus(config=self.config.model_dump())
        self.tasks = tasks or []
        self._task_index = 0

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
        self.dispatcher = ActionDispatcher(corpus=self.corpus, config=self.config)
        self.budget_policy = BudgetPolicy(
            max_context_tokens=self.config.max_context_tokens,
            hard_budget_threshold=self.config.hard_budget_threshold,
        )
        self.termination_policy = TerminationPolicy(max_steps=self.config.max_steps)
        self.episode = EpisodeState.create()

    def set_corpus(self, corpus: DocumentCorpus) -> None:
        """Replace the retrieval corpus."""
        self.corpus = corpus
        self.dispatcher = ActionDispatcher(corpus=self.corpus, config=self.config)

    def set_tasks(self, tasks: list[SearchTask]) -> None:
        """Replace available tasks."""
        self.tasks = tasks
        self._task_index = 0

    def add_task(self, task: SearchTask) -> None:
        """Append a task."""
        self.tasks.append(task)

    def _get_next_task(self, seed: int | None = None) -> SearchTask | None:
        """Choose the next task, deterministically when seeded."""
        if not self.tasks:
            return None
        if seed is not None:
            sampler = TaskSampler(self.tasks, seed=seed)
            return sampler.sample(1)[0]

        task = self.tasks[self._task_index % len(self.tasks)]
        self._task_index += 1
        return task

    def _gold_chunk_ids(self) -> set[str]:
        """Return gold chunk ids for the active task."""
        if self.episode.current_task is None:
            return set()
        return {item.id for item in self.episode.current_task.supporting_items}

    def _configure_reward_beta(self, **kwargs: Any) -> None:
        reward_beta = kwargs.get("reward_beta")
        training_step = kwargs.get("training_step")

        if reward_beta is not None:
            self.reward_calculator.beta = float(reward_beta)
            return

        if self.beta_scheduler is not None and training_step is not None:
            self.reward_calculator.beta = self.beta_scheduler.get_beta(int(training_step))
            return

        self.reward_calculator.beta = self.config.beta

    def _create_observation(
        self,
        *,
        action_result: dict[str, Any] | None = None,
        action_type: str | None = None,
        reward: float = 0.0,
    ) -> SearchObservation:
        return render_observation(
            episode=self.episode,
            config=self.config,
            action_result=action_result,
            action_type=action_type,
            reward=reward,
        )

    def _finalize_episode(
        self,
        *,
        answer: str,
        supporting_chunk_ids: list[str] | None = None,
        action_result: dict[str, Any] | None = None,
        termination: TerminationInfo | None = None,
    ) -> tuple[dict[str, Any], float]:
        support_ids = list(dict.fromkeys(supporting_chunk_ids or []))
        gold_chunks = self._gold_chunk_ids()
        invalid_support_ids = [
            chunk_id
            for chunk_id in support_ids
            if chunk_id not in self.episode.context_chunks
        ]
        unsupported_support_ids = [
            chunk_id for chunk_id in support_ids if chunk_id not in gold_chunks
        ]

        self.episode.done = True
        self.episode.termination_reason = (
            termination.reason if termination else "answer"
        )
        self.episode.truncated = termination.truncated if termination else False
        self.episode.terminated = not self.episode.truncated

        if self.episode.current_task is None:
            final_result: dict[str, Any] = {
                "answer_submitted": answer,
                "supporting_chunk_ids_used": support_ids,
                "unsupported_supporting_chunk_ids": unsupported_support_ids,
                "invalid_supporting_chunk_ids": invalid_support_ids,
                "final_reward": 0.001,
            }
            self.episode.last_metrics = None
            reward = 0.001
        else:
            metrics = self.reward_calculator.calculate_reward(
                tracker=self.episode.tracker,
                gold_chunks=gold_chunks,
                gold_answer=self.episode.current_task.truth,
                predicted_answer=answer,
                context_texts=[
                    chunk.content for chunk in self.episode.context_chunks.values()
                ],
                steps_used=self.episode.state.step_count,
                max_steps=self.config.max_steps,
                tokens_used=self.episode.context_token_count,
                max_tokens=self.config.max_context_tokens,
                supporting_chunk_ids=support_ids,
                all_seen_texts=self.episode.seen_texts or None,
            )
            self.episode.last_metrics = metrics
            reward = metrics.total_reward
            final_result = {
                "answer_submitted": answer,
                "supporting_chunk_ids_used": support_ids,
                "unsupported_supporting_chunk_ids": unsupported_support_ids,
                "invalid_supporting_chunk_ids": invalid_support_ids,
                "final_reward": metrics.total_reward,
                "trajectory_recall": metrics.trajectory_recall,
                "output_recall": metrics.output_recall,
                "output_precision": metrics.output_precision,
                "f_beta": metrics.f_beta,
                "beta_used": metrics.beta,
                "answer_correct": metrics.answer_correct,
                "answer_found_in_context": metrics.answer_found_in_context,
                "answer_similarity": metrics.answer_similarity,
                "f_beta_reward": metrics.f_beta_reward,
                "trajectory_reward": metrics.trajectory_reward,
                "answer_reward": metrics.answer_reward,
                "turn_penalty": metrics.turn_penalty,
                "prune_penalty": metrics.prune_penalty,
                "pre_penalty_reward": metrics.pre_penalty_reward,
                "reward_floor": metrics.reward_floor,
                "evidence_precision": metrics.evidence_precision,
                "evidence_recall": metrics.evidence_recall,
                "evidence_reward": metrics.evidence_reward,
                "unsupported_support_count": metrics.unsupported_support_count,
            }

        merged_result = dict(action_result or {})
        merged_result.update(final_result)
        merged_result["termination_reason"] = self.episode.termination_reason
        merged_result["step_limit_reached"] = self.episode.termination_reason == "max_steps"

        if termination is not None:
            termination.final_reward_applied = True

        return merged_result, reward

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task: SearchTask | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> SearchObservation:
        """Reset to a fresh episode."""
        self.episode = EpisodeState.create(episode_id=episode_id, seed=seed)
        self._configure_reward_beta(**kwargs)

        if task is not None:
            self.episode.current_task = task
        elif task_id is not None:
            self.episode.current_task = next(
                (candidate for candidate in self.tasks if candidate.task_id == task_id),
                None,
            )
        else:
            self.episode.current_task = self._get_next_task(seed)

        if self.episode.current_task is None:
            self.episode.done = True
            self.episode.terminated = True
            self.episode.termination_reason = "no_tasks"
            return SearchObservation(
                question="No tasks available. Please add tasks to the environment.",
                done=True,
                reward=0.001,
                terminated=True,
                termination_reason="no_tasks",
            )

        return self._create_observation()

    def step(
        self,
        action: SearchAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SearchObservation:
        """Execute one environment step."""
        _ = timeout_s, kwargs

        if self.episode.done:
            return self._create_observation(
                action_result={"error": "Episode already finished"},
                action_type=action.action_type.value,
                reward=0.001,
            )

        self.episode.state.step_count += 1
        reward = 0.0

        if self.budget_policy.blocks(action.action_type, self.episode.context_token_count):
            dispatch_result = DispatchResult(
                action_result=self.budget_policy.blocked_result(),
                reward_delta=-0.1,
            )
        else:
            dispatch_result = self.dispatcher.dispatch(action, self.episode)

        reward += dispatch_result.reward_delta
        action_result = dispatch_result.action_result

        if dispatch_result.answer is not None:
            action_result, reward = self._finalize_episode(
                answer=dispatch_result.answer,
                supporting_chunk_ids=dispatch_result.supporting_chunk_ids,
                action_result=action_result,
                termination=TerminationInfo(reason="answer", truncated=False),
            )
        elif not self.episode.done:
            termination = self.termination_policy.after_step(self.episode.state.step_count)
            if termination is not None:
                action_result, reward = self._finalize_episode(
                    answer="",
                    supporting_chunk_ids=[],
                    action_result=action_result,
                    termination=termination,
                )

        return self._create_observation(
            action_result=action_result,
            action_type=action.action_type.value,
            reward=reward,
        )

    @property
    def state(self):
        """Expose the OpenEnv runtime state."""
        return self.episode.state

    def get_metrics(self) -> RewardMetrics | None:
        """Return metrics from the last finished episode."""
        return self.episode.last_metrics

    def get_metadata(self) -> EnvironmentMetadata:
        """Return static environment metadata."""
        return EnvironmentMetadata(
            name="SearchArena",
            description="Multi-hop document retrieval environment for search agents.",
            version="0.1.0",
        )