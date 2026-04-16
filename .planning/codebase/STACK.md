# Stack

## Runtime

- Language: Python 3.11+ from [pyproject.toml](/Users/aman/Documents/search-env/pyproject.toml:1)
- Packaging/build: `setuptools` + `wheel` from [pyproject.toml](/Users/aman/Documents/search-env/pyproject.toml:1)
- Dependency/tooling workflow: `uv` lockfile and local `.venv`
- Deployment target: OpenEnv-compatible FastAPI service described in [openenv.yaml](/Users/aman/Documents/search-env/openenv.yaml:1)
- Container/runtime mode: Docker / Hugging Face Space from [README.md](/Users/aman/Documents/search-env/README.md:1)

## Core Libraries

- `openenv-core[core]` for environment interfaces and client/server types in [pyproject.toml](/Users/aman/Documents/search-env/pyproject.toml:6)
- `pydantic` for action, observation, task, and config models in [searcharena/models.py](/Users/aman/Documents/search-env/searcharena/models.py:1)
- `fastapi` and `uvicorn` for the HTTP server in [server/app.py](/Users/aman/Documents/search-env/server/app.py:1)
- `openai` for the baseline inference agent in [inference.py](/Users/aman/Documents/search-env/inference.py:1)
- Standard library heavy implementation in `searcharena/engine/*` and `searcharena/training/*`

## Optional / Domain-Specific Dependencies

- `anthropic` for data generation workflows in [pyproject.toml](/Users/aman/Documents/search-env/pyproject.toml:15)
- `chromadb` for indexing/generation pipelines in [pyproject.toml](/Users/aman/Documents/search-env/pyproject.toml:15)
- `tiktoken` for token-aware generation utilities in [pyproject.toml](/Users/aman/Documents/search-env/pyproject.toml:15)
- `requests`, `httpx`, `pandas`, `python-docx`, `rich` for domain-specific data tooling in [pyproject.toml](/Users/aman/Documents/search-env/pyproject.toml:15)

## Project Modules

- Public package exports live in [searcharena/__init__.py](/Users/aman/Documents/search-env/searcharena/__init__.py:1)
- Environment logic lives under [searcharena/engine](/Users/aman/Documents/search-env/searcharena/engine)
- Training utilities live under [searcharena/training](/Users/aman/Documents/search-env/searcharena/training)
- OpenEnv server wrapper lives under [server](/Users/aman/Documents/search-env/server)
- Root compatibility shims exist in [client.py](/Users/aman/Documents/search-env/client.py:1) and [models.py](/Users/aman/Documents/search-env/models.py:1)

## Data / Assets

- Sample tasks live in [sample](/Users/aman/Documents/search-env/sample)
- Generator code lives in [data/generator](/Users/aman/Documents/search-env/data/generator)
- Benchmark and design artifacts are currently stored as markdown and images at repo root, including [DESIGN.md](/Users/aman/Documents/search-env/DESIGN.md:1) and [IMPLEMENTATION_PLAN.md](/Users/aman/Documents/search-env/IMPLEMENTATION_PLAN.md:1)

## Test Stack

- `pytest` and `pytest-cov` in [pyproject.toml](/Users/aman/Documents/search-env/pyproject.toml:12)
- Shared fixtures in [tests/conftest.py](/Users/aman/Documents/search-env/tests/conftest.py:1)
- Engine, retrieval, reward, task, and training coverage under [tests](/Users/aman/Documents/search-env/tests)

## Notes

- The repo is structured as a brownfield OpenEnv environment project rather than a generic FastAPI app.
- The current stack is deliberately lightweight in the runtime path: retrieval is local BM25 and reward logic is pure Python.
