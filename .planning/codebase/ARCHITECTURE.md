# Architecture

## High-Level Shape

This repository is an OpenEnv-compatible retrieval environment with four main layers:

1. Domain models in [searcharena/models.py](/Users/aman/Documents/search-env/searcharena/models.py:1)
2. Runtime environment logic in [searcharena/engine](/Users/aman/Documents/search-env/searcharena/engine)
3. HTTP/OpenEnv server adapters in [server](/Users/aman/Documents/search-env/server)
4. Training, evaluation, and inference helpers in [searcharena/training](/Users/aman/Documents/search-env/searcharena/training) and [inference.py](/Users/aman/Documents/search-env/inference.py:1)

## Runtime Flow

- `SearchEnvironment` owns episode state and action dispatch in [searcharena/engine/core.py](/Users/aman/Documents/search-env/searcharena/engine/core.py:1)
- Retrieval is handled by `DocumentCorpus` and `BM25Index` in [searcharena/engine/retrieval.py](/Users/aman/Documents/search-env/searcharena/engine/retrieval.py:1)
- Reward shaping is encapsulated in [searcharena/engine/rewards.py](/Users/aman/Documents/search-env/searcharena/engine/rewards.py:1)
- Action-specific mutations are delegated to helper functions in [searcharena/engine/handlers.py](/Users/aman/Documents/search-env/searcharena/engine/handlers.py:1)
- Observation rendering is centralized in [searcharena/engine/observations.py](/Users/aman/Documents/search-env/searcharena/engine/observations.py:1)

## Data Model Boundary

- Actions, observations, chunks, tasks, and config are all Pydantic models in [searcharena/models.py](/Users/aman/Documents/search-env/searcharena/models.py:1)
- The environment and server pass these models around rather than raw dicts wherever possible
- The server converts observations to JSON with `model_dump()` at the API boundary in [server/app.py](/Users/aman/Documents/search-env/server/app.py:118)

## Server Boundary

- [server/environment.py](/Users/aman/Documents/search-env/server/environment.py:1) is a thin OpenEnv wrapper over `searcharena.engine.SearchEnvironment`
- [server/app.py](/Users/aman/Documents/search-env/server/app.py:1) adds a FastAPI façade and manages a singleton runtime instance
- This means the codebase currently supports one in-memory environment per process in the HTTP path

## Training / Evaluation Architecture

- Prompt generation lives in [searcharena/training/prompts.py](/Users/aman/Documents/search-env/searcharena/training/prompts.py:1)
- Sampling/buffers live in [searcharena/training/datasets.py](/Users/aman/Documents/search-env/searcharena/training/datasets.py:1)
- Curriculum logic lives in [searcharena/training/curriculum.py](/Users/aman/Documents/search-env/searcharena/training/curriculum.py:1)
- Metrics and evaluation aggregation live in [searcharena/training/metrics.py](/Users/aman/Documents/search-env/searcharena/training/metrics.py:1) and [searcharena/training/evaluation.py](/Users/aman/Documents/search-env/searcharena/training/evaluation.py:1)

## Data / Benchmark Support

- Sample benchmark tasks are loaded through [searcharena/engine/factory.py](/Users/aman/Documents/search-env/searcharena/engine/factory.py:1) and [sample/__init__.py](/Users/aman/Documents/search-env/sample/__init__.py:1)
- Broader dataset generation and verification tooling lives under [data/generator](/Users/aman/Documents/search-env/data/generator)

## Current Architectural Intent

- Keep the environment runtime mostly pure Python and easily testable
- Keep `server/` thin so runtime logic remains reusable in tests, training, and alternative frontends
- Use local retrieval and reward logic as the MVP baseline, with room for more sophisticated search later
