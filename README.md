---
title: Search RL Environment
emoji: 🔍
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Search RL Environment

A reinforcement learning environment for training agents to perform **multi-hop document retrieval** tasks with explicit context management. Based on the Context-1 paper's F-beta reward curriculum.

## Overview

Agents must:
1. **Search** - Issue queries to find relevant documents
2. **Read** - Add document chunks to context (consumes token budget)
3. **Prune** - Remove irrelevant chunks to manage token budget
4. **Answer** - Submit final answer based on retrieved evidence

The environment rewards agents for:
- Finding relevant documents (F-beta score emphasizing recall)
- Exploring broadly (trajectory recall)
- Providing correct, evidence-backed answers
- Efficient use of limited token budget

## Quick Start

```python
from client import SearchEnv
from models import SearchAction

# Create environment from Docker image
env = SearchEnv.from_docker_image("search_env:latest")

try:
    # Reset to get a question
    obs = env.reset()
    print(f"Question: {obs.question}")
    print(f"Token budget: {obs.context_token_budget}")

    # Search for relevant documents
    action = SearchAction.make_search("Facebook acquisition Instagram")
    obs = env.step(action)
    results = obs.action_result.get("results", [])
    print(f"Found {len(results)} results")

    # Read top result into context
    if results:
        action = SearchAction.make_read([results[0]["chunk_id"]])
        obs = env.step(action)
        print(f"Context now has {obs.context_token_count} tokens")

    # Submit answer
    action = SearchAction.make_answer("2010", supporting_chunk_ids=[...])
    obs = env.step(action)
    print(f"Reward: {obs.reward}")
    print(f"Answer correct: {obs.action_result.get('answer_correct')}")

finally:
    env.close()
```

## Action Space

| Action | Description | Payload |
|--------|-------------|---------|
| `search` | Query the document corpus | `query: str`, `top_k: int` |
| `read` | Add chunks to context | `chunk_ids: List[str]` |
| `prune` | Remove chunks from context | `chunk_ids: List[str]` |
| `answer` | Submit final answer (ends episode) | `answer: str`, `supporting_chunk_ids: List[str]` |

## Observation Space

```python
SearchObservation:
    question: str                    # The question to answer
    context_chunks: List[ChunkSummary]  # Chunks currently in context
    context_token_count: int         # Current tokens used
    context_token_budget: int        # Maximum tokens allowed (default: 32768)
    budget_usage_percent: float      # Percentage of budget used
    budget_warning: Optional[str]    # Warning when near limit
    action_result: Dict              # Result of last action
    step_count: int                  # Current step number
    max_steps: int                   # Maximum steps allowed (default: 20)
    queries_issued: List[str]        # Search history
    done: bool                       # Episode finished?
    reward: float                    # Reward (only non-zero on answer)
```

## Reward Function

The reward combines multiple components:

| Component | Weight | Description |
|-----------|--------|-------------|
| **F-beta Score** | 0.7 | Precision/recall of gold chunks in context (β=4 emphasizes recall) |
| **Trajectory Recall** | 0.3 | Credit for finding gold chunks, even if later pruned |
| **Answer Bonus** | 1.0 | Bonus if answer is correct AND evidence is in context |

Penalties:
- **Turn penalty**: Gradual penalty for using many steps
- **Prune penalty**: Penalty for excessive consecutive pruning

## Tasks

The environment includes multi-hop questions requiring document retrieval:

| Difficulty | Description | Example |
|------------|-------------|---------|
| **Easy** | Single-hop, one document needed | "When was Instagram launched?" |
| **Medium** | Two-hop, requires connecting facts | "Where was Facebook's founder born?" |
| **Hard** | Multi-constraint, requires reasoning | Complex comparison/temporal questions |

## Building & Running

### Docker

```bash
# Build the image
docker build -t search_env:latest .

# Run the server
docker run -p 8000:8000 search_env:latest
```

### Local Development

```bash
# Install dependencies
uv sync

# Run server
uvicorn server.app:app --reload

# Run inference baseline
python inference.py
```

### Validation

```bash
# Validate OpenEnv spec compliance
openenv validate

# Run tests
pytest tests/
```

## Configuration

Environment behavior can be configured via `SearchEnvConfig`:

```python
SearchEnvConfig(
    max_steps=20,              # Maximum actions per episode
    max_context_tokens=32768,  # Token budget
    beta=4.0,                  # F-beta parameter (>1 favors recall)
    f_beta_weight=0.7,         # Weight for F-beta component
    trajectory_reward_weight=0.3,  # Weight for exploration credit
    search_top_k=10,           # Default search results
)
```

## Project Structure

```
search_env/
├── openenv.yaml           # OpenEnv manifest
├── models.py              # Action/Observation Pydantic models
├── client.py              # SearchEnv client for connecting to server
├── inference.py           # Baseline agent implementation
├── Dockerfile             # Container definition
├── server/
│   ├── app.py             # FastAPI server
│   ├── environment.py     # Core RL environment logic
│   ├── rewards.py         # Reward calculation (F-beta, penalties)
│   └── retrieval.py       # BM25 search index
└── tests/
    └── test_environment.py  # Environment tests
```

## API Endpoints

When deployed, the server exposes:

- `POST /reset` - Start new episode
- `POST /step` - Execute action
- `GET /health` - Health check
- `GET /docs` - OpenAPI documentation
- `WS /ws` - WebSocket for low-latency interaction

## References

- Based on the Context-1 paper's reward curriculum for iterative retrieval
- Uses F-beta scoring to balance precision and recall
- Implements trajectory reward for exploration credit
