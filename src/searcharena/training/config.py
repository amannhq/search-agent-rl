"""Training and evaluation configs."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class TrainingConfig(BaseModel):
    """Configuration for RL training runs."""

    # Run identification
    run_name: str = Field(default="search-training", description="Name for this training run")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Training duration
    total_steps: int = Field(default=10000, description="Total training steps")
    episodes_per_step: int = Field(default=4, description="Episodes to collect per step")
    eval_frequency: int = Field(default=500, description="Steps between evaluations")
    checkpoint_frequency: int = Field(default=1000, description="Steps between checkpoints")

    # Learning parameters
    learning_rate: float = Field(default=1e-5, description="Learning rate")
    optimizer: OptimizerType = Field(default=OptimizerType.ADAMW, description="Optimizer type")
    weight_decay: float = Field(default=0.01, description="Weight decay for regularization")
    max_grad_norm: float = Field(default=1.0, description="Gradient clipping threshold")
    warmup_steps: int = Field(default=100, description="Learning rate warmup steps")

    # Batch settings
    batch_size: int = Field(default=8, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=4, description="Steps for gradient accumulation")

    # GRPO/PPO specific
    kl_coef: float = Field(default=0.1, description="KL divergence coefficient")
    clip_range: float = Field(default=0.2, description="PPO clip range")
    value_coef: float = Field(default=0.5, description="Value loss coefficient")
    entropy_coef: float = Field(default=0.01, description="Entropy bonus coefficient")

    # Curriculum settings
    use_curriculum: bool = Field(default=True, description="Enable curriculum learning")
    curriculum_warmup_steps: int = Field(default=1000, description="Steps before increasing difficulty")

    # Logging
    log_frequency: int = Field(default=10, description="Steps between logging")
    use_wandb: bool = Field(default=False, description="Enable Weights & Biases logging")
    wandb_project: str = Field(default="searcharena", description="W&B project name")

    # Paths
    output_dir: str = Field(default="./outputs", description="Output directory for checkpoints")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Create from dictionary."""
        return cls(**data)


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    # Evaluation settings
    num_episodes: int = Field(default=100, description="Episodes to evaluate")
    max_parallel: int = Field(default=4, description="Max parallel evaluations")

    # Task selection
    difficulties: list[str] = Field(
        default_factory=lambda: ["easy", "medium", "hard"],
        description="Difficulties to evaluate on"
    )
    domains: list[str] = Field(
        default_factory=list,
        description="Specific domains to evaluate (empty = all)"
    )

    # Metrics settings
    compute_per_difficulty: bool = Field(
        default=True,
        description="Compute metrics per difficulty level"
    )
    compute_per_domain: bool = Field(
        default=True,
        description="Compute metrics per domain"
    )

    # Output
    save_trajectories: bool = Field(
        default=False,
        description="Save full episode trajectories"
    )
    output_format: str = Field(
        default="json",
        description="Output format (json, csv)"
    )
