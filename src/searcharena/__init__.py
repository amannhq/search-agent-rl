"""SearchArena - RL environment for multi-hop document retrieval."""

from .client import SearchEnv
from .env import (
    ActionDispatcher,
    BudgetPolicy,
    DispatchResult,
    EpisodeState,
    SearchEnvironment,
    TerminationInfo,
    TerminationPolicy,
    render_observation,
)
from .models import (
    ActionType,
    AnswerActionPayload,
    Chunk,
    ChunkSummary,
    PruneActionPayload,
    ReadActionPayload,
    SearchAction,
    SearchActionPayload,
    SearchEnvConfig,
    SearchObservation,
    SearchTask,
)
from .retrieval import (
    BM25Index,
    DocumentCorpus,
)
from .rewards import (
    BetaScheduler,
    RewardCalculator,
    RewardMetrics,
    TrajectoryTracker,
)
from .tasks import (
    create_sample_corpus,
    create_sample_tasks,
    get_directory_statistics,
    get_sample_statistics,
    get_sample_tasks,
    get_sample_tasks_by_domain,
    get_sample_tasks_by_level,
    load_tasks_by_level,
    load_tasks_from_directory,
    load_verified_tasks,
)
from .training import (
    CurriculumScheduler,
    DatasetBuilder,
    DifficultyLevel,
    EpisodeBuffer,
    EpisodeMetrics,
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    MetricsLogger,
    PromptBuilder,
    SYSTEM_PROMPTS,
    TaskSampler,
    TrainingConfig,
    TrainingMetrics,
)

__all__ = [
    # Client
    "SearchEnv",
    # Models
    "ActionType",
    "AnswerActionPayload",
    "Chunk",
    "ChunkSummary",
    "PruneActionPayload",
    "ReadActionPayload",
    "SearchAction",
    "SearchActionPayload",
    "SearchEnvConfig",
    "SearchObservation",
    "SearchTask",
    # Engine
    "ActionDispatcher",
    "BudgetPolicy",
    "DispatchResult",
    "EpisodeState",
    "TerminationInfo",
    "TerminationPolicy",
    "SearchEnvironment",
    "create_sample_corpus",
    "create_sample_tasks",
    "render_observation",
    # Retrieval
    "BM25Index",
    "DocumentCorpus",
    # Rewards
    "BetaScheduler",
    "RewardCalculator",
    "RewardMetrics",
    "TrajectoryTracker",
    # Tasks
    "get_directory_statistics",
    "get_sample_statistics",
    "get_sample_tasks",
    "get_sample_tasks_by_domain",
    "get_sample_tasks_by_level",
    "load_tasks_by_level",
    "load_tasks_from_directory",
    "load_verified_tasks",
    # Training
    "CurriculumScheduler",
    "DatasetBuilder",
    "DifficultyLevel",
    "EpisodeBuffer",
    "EpisodeMetrics",
    "EvaluationConfig",
    "EvaluationResult",
    "Evaluator",
    "MetricsLogger",
    "PromptBuilder",
    "SYSTEM_PROMPTS",
    "TaskSampler",
    "TrainingConfig",
    "TrainingMetrics",
]
