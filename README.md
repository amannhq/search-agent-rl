---
title: Search RL Environment
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Search RL Environment

RL environment for multi-hop document retrieval with explicit context management. Uses F-beta reward curriculum from the Context-1 paper.

## What it does

Agent searches a corpus, reads documents into a limited context window, prunes irrelevant content, and submits an answer. Reward is based on finding the right documents and answering correctly.

**Actions:** `search`, `read`, `prune`, `answer`

**Live demo:** https://aman045-openenv-search-rl.hf.space

## Quick test

```bash
curl -X POST https://aman045-openenv-search-rl.hf.space/reset

curl -X POST https://aman045-openenv-search-rl.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"search","search":{"query":"Instagram","top_k":3}}}'
```

## Usage

```python
from searcharena import SearchEnv, SearchAction

env = SearchEnv.from_docker_image("searcharena:latest")

obs = env.reset()
print(obs.question)

# search -> read -> answer
obs = env.step(SearchAction.make_search("Facebook acquisition Instagram"))
results = obs.action_result["results"]

obs = env.step(SearchAction.make_read([results[0]["chunk_id"]]))
obs = env.step(SearchAction.make_answer("2012"))

print(f"Reward: {obs.reward}")

env.close()
```

## Reward

| Component | Weight | What it measures |
|-----------|--------|------------------|
| F-beta | 0.7 | Gold chunks in final context (β=4, recall-heavy) |
| Trajectory | 0.3 | Gold chunks seen at any point |
| Answer bonus | 1.0 | Correct answer with evidence |

Penalties for excessive steps and repeated pruning.

## Tasks

Three difficulty levels:
- **Easy**: Single fact lookup
- **Medium**: Two-hop reasoning
- **Hard**: Multi-constraint, cross-document

## Run locally

```bash
uv sync
uvicorn searcharena.server.app:app --reload
```

Or with Docker:

```bash
docker build -t searcharena .
docker run -p 8000:8000 searcharena
```

## Project structure

```
src/searcharena/      # Canonical runtime package
  env/                # Environment orchestration and state
  retrieval/          # Retrieval backends and corpus layer
  rewards/            # Reward calculation and tracking
  server/             # FastAPI app factory and routers
  tasks/              # Bundled tasks and loading helpers
  training/           # Training utilities
docs/                 # Architecture and design notes
tools/                # Offline data generation and utilities
```

## Config

```python
SearchEnvConfig(
    max_steps=20,
    max_context_tokens=32768,
    beta=4.0,
    f_beta_weight=0.7,
    trajectory_reward_weight=0.3,
)
```

## API

- `POST /reset` - New episode
- `POST /step` - Execute action
- `GET /health` - Health check
