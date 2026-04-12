"""SearchArena - RL environment for multi-hop document retrieval."""

from .client import SearchEnv
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
from .engine import (
    BM25Index,
    BetaScheduler,
    DocumentCorpus,
    RewardCalculator,
    RewardMetrics,
    SearchEnvironment,
    TrajectoryTracker,
    create_sample_corpus,
    create_sample_tasks,
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
    "SearchEnvironment",
    "create_sample_corpus",
    "create_sample_tasks",
    # Retrieval
    "BM25Index",
    "DocumentCorpus",
    # Rewards
    "BetaScheduler",
    "RewardCalculator",
    "RewardMetrics",
    "TrajectoryTracker",
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
