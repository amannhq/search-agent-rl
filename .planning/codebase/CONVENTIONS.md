# Conventions

## Module Design

- Shared domain contracts are centralized in [searcharena/models.py](/Users/aman/Documents/search-env/searcharena/models.py:1)
- Runtime logic is split into focused helper modules under [searcharena/engine](/Users/aman/Documents/search-env/searcharena/engine)
- Server code is expected to stay thin and mostly delegate to the engine, as documented in [server/environment.py](/Users/aman/Documents/search-env/server/environment.py:1)

## Data Modeling

- Pydantic `BaseModel` is the standard for config, actions, observations, chunks, and tasks
- Dataclasses are used for lighter-weight internal metrics structures such as `RewardMetrics` in [searcharena/engine/rewards.py](/Users/aman/Documents/search-env/searcharena/engine/rewards.py:1)
- Enums are used for constrained value sets, e.g. `ActionType` and `OptimizerType`

## Naming

- Environment-facing verbs are explicit: `search`, `read`, `prune`, `answer`
- Helper functions are named `handle_*` for action mutation paths in [searcharena/engine/handlers.py](/Users/aman/Documents/search-env/searcharena/engine/handlers.py:1)
- Factory helpers use `create_*` naming in [searcharena/engine/factory.py](/Users/aman/Documents/search-env/searcharena/engine/factory.py:1)
- Training utilities use noun-heavy class names like `TaskSampler`, `EpisodeBuffer`, `CurriculumScheduler`

## Typing Style

- The codebase uses modern Python typing (`list[str]`, `dict[str, Any]`, `X | None`)
- Most runtime modules lean on explicit return types and type-annotated attributes
- `from __future__ import annotations` is used broadly across modules

## Documentation Style

- Modules generally start with a short top-level docstring
- Public methods include concise docstrings and usually document Args / Returns when the method is important
- README usage examples stay close to the public API

## Testing Style

- Pytest is the default test runner
- Shared fixtures live in [tests/conftest.py](/Users/aman/Documents/search-env/tests/conftest.py:1)
- Tests emphasize behavior and edge cases over implementation details, especially in [tests/test_environment.py](/Users/aman/Documents/search-env/tests/test_environment.py:1) and [tests/test_edge_cases.py](/Users/aman/Documents/search-env/tests/test_edge_cases.py:1)

## Architectural Conventions

- Keep the public import surface stable through [searcharena/__init__.py](/Users/aman/Documents/search-env/searcharena/__init__.py:1)
- Treat `server/` as an adapter layer, not the home of core logic
- Sample tasks/corpus creation are first-class fixtures for tests and local development

## Notable Local Convention Risks

- The repo currently has both `searcharena` package exports and `search_env` packaging aliases in [pyproject.toml](/Users/aman/Documents/search-env/pyproject.toml:1)
- There are root re-export shims, which suggests packaging/submission compatibility is part of the design and should be preserved carefully during refactors
