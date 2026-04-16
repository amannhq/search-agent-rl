# Testing

## Test Framework

- Test runner: `pytest` from [pyproject.toml](/Users/aman/Documents/search-env/pyproject.toml:1)
- Shared fixtures are defined in [tests/conftest.py](/Users/aman/Documents/search-env/tests/conftest.py:1)
- Project root is inserted into `sys.path` in the fixture module to make local imports work consistently

## Covered Areas

- [tests/test_environment.py](/Users/aman/Documents/search-env/tests/test_environment.py:1) covers reset/step flow and budget behavior
- [tests/test_edge_cases.py](/Users/aman/Documents/search-env/tests/test_edge_cases.py:1) covers lifecycle and invalid-input handling
- [tests/test_retrieval.py](/Users/aman/Documents/search-env/tests/test_retrieval.py:1) covers BM25/corpus behaviors
- [tests/test_rewards.py](/Users/aman/Documents/search-env/tests/test_rewards.py:1) covers reward calculations and beta scheduling
- [tests/test_tasks.py](/Users/aman/Documents/search-env/tests/test_tasks.py:1) covers task/sample integrity
- [tests/test_training.py](/Users/aman/Documents/search-env/tests/test_training.py:1) covers curriculum, prompts, buffers, and config serialization

## Fixture Model

- `corpus()` returns a sample corpus from [searcharena/engine/factory.py](/Users/aman/Documents/search-env/searcharena/engine/factory.py:1)
- `tasks()` returns sample tasks from the `sample` module
- `env()` and `env_with_config()` build a configured `SearchEnvironment` for tests
- `custom_task()` creates a hand-rolled task model for focused behavior checks

## Test Style

- Most tests are small unit/integration hybrids around package-level APIs
- Tests call the real environment methods rather than mocking internal helpers
- Assertions focus on observable outputs such as `obs.done`, `action_result`, reward bounds, and context token counts

## Coverage Strengths

- Environment behavior has dedicated edge-case coverage
- Retrieval and reward math have standalone tests
- Training helper objects are covered separately from the runtime environment

## Coverage Gaps

- No server-level FastAPI endpoint tests are visible under [tests](/Users/aman/Documents/search-env/tests)
- No explicit tests for multi-session behavior despite concurrent-session flags in runtime/server code
- Packaging / installability mismatches are not directly tested
- The inference script is not covered by the pytest suite

## Practical Guidance

- Start with targeted test files for runtime fixes: `tests/test_environment.py`, `tests/test_edge_cases.py`, `tests/test_rewards.py`
- Keep using the sample corpus/tasks path unless a change specifically targets generation or external data integration
