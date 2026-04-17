"""Inference configuration from environment variables."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv


_DOTENV_PATH = find_dotenv(usecwd=True)
if _DOTENV_PATH:
    load_dotenv(_DOTENV_PATH, override=False)


@dataclass
class InferenceConfig:
    """Configuration for inference runs."""

    # Environment identification
    benchmark: str = field(default_factory=lambda: os.getenv("SEARCH_ENV_BENCHMARK", "search_env"))
    env_base_url: str = field(default_factory=lambda: os.getenv("ENV_BASE_URL", ""))
    local_image_name: str = field(default_factory=lambda: os.getenv("LOCAL_IMAGE_NAME", ""))

    # LLM API settings (injected by submission system)
    api_base_url: str = field(default_factory=lambda: os.getenv("API_BASE_URL", "https://api.openai.com/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY") or os.getenv("HF_TOKEN", ""))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))

    # Task settings
    task_ids: tuple[str, ...] = field(default_factory=lambda: _parse_task_ids())

    # Search/Read parameters
    num_episodes: int = field(default_factory=lambda: int(os.getenv("NUM_EPISODES", "1") or "1"))
    search_top_k: int = field(default_factory=lambda: int(os.getenv("SEARCH_TOP_K", "5")))
    read_top_k: int = field(default_factory=lambda: int(os.getenv("READ_TOP_K", "2")))

    # LLM parameters
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.2")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_COMPLETION_TOKENS", "350")))
    max_retries: int = 4

    # Budget thresholds
    char_limit: int = 800
    soft_budget_threshold: float = 0.75
    hard_budget_threshold: float = 0.95
    prune_target_threshold: float = 0.60

    def validate(self) -> None:
        """Validate required configuration."""
        if not self.api_key:
            raise RuntimeError("API_KEY environment variable is required but not set")
        if not self.api_base_url:
            raise RuntimeError("API_BASE_URL environment variable is required but not set")


def _parse_task_ids() -> tuple[str, ...]:
    """Parse task IDs from environment variable."""
    default_tasks = (
        "sample_tech_instagram_001,sample_tech_whatsapp_001,"
        "sample_tech_facebook_acquisitions_001,sample_science_curie_001,"
        "sample_history_berlin_wall_001"
    )
    raw = os.getenv("SEARCH_ENV_TASKS", default_tasks)
    return tuple(t.strip() for t in raw.split(",") if t.strip())


# Stopwords for answer extraction
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "by", "did", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "to", "was",
    "were", "what", "when", "where", "which", "who", "much",
    "compare", "compared",
})

# Compiled regex patterns
RE_WHITESPACE = re.compile(r"\s+")
RE_WORD = re.compile(r"\w+")
RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
RE_DIGIT = re.compile(r"\d")
