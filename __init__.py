# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Search RL Environment - Multi-hop document retrieval for training search agents."""

try:
    from .client import SearchEnv
    from .models import (
        ActionType,
        AnswerActionPayload,
        AnswerResult,
        Chunk,
        ChunkSummary,
        PruneActionPayload,
        PruneResult,
        ReadActionPayload,
        ReadResult,
        SearchAction,
        SearchActionPayload,
        SearchEnvConfig,
        SearchObservation,
        SearchResult,
        SearchTask,
    )
except ImportError:
    pass

__all__ = [
    # Client
    "SearchEnv",
    # Action types
    "ActionType",
    "SearchAction",
    "SearchActionPayload",
    "ReadActionPayload",
    "PruneActionPayload",
    "AnswerActionPayload",
    # Results
    "SearchResult",
    "ReadResult",
    "PruneResult",
    "AnswerResult",
    # Models
    "Chunk",
    "ChunkSummary",
    "SearchObservation",
    "SearchTask",
    "SearchEnvConfig",
]
