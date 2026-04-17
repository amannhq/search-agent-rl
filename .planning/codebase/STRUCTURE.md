# Structure

## Top-Level Layout

- [searcharena](/Users/aman/Documents/search-env/searcharena) — main Python package
- [server](/Users/aman/Documents/search-env/server) — FastAPI and OpenEnv wrapper layer
- [tests](/Users/aman/Documents/search-env/tests) — pytest coverage
- [sample](/Users/aman/Documents/search-env/sample) — sample task JSON data
- [data/generator](/Users/aman/Documents/search-env/data/generator) — offline data generation/indexing flows
- Root scripts/docs — [inference.py](/Users/aman/Documents/search-env/inference.py:1), [validate-submission.sh](/Users/aman/Documents/search-env/validate-submission.sh:1), [README.md](/Users/aman/Documents/search-env/README.md:1)

## Main Package

- [searcharena/__init__.py](/Users/aman/Documents/search-env/searcharena/__init__.py:1) re-exports the public surface
- [searcharena/models.py](/Users/aman/Documents/search-env/searcharena/models.py:1) defines shared contracts
- [searcharena/client.py](/Users/aman/Documents/search-env/searcharena/client.py:1) implements the OpenEnv client wrapper

## Engine Package

- [searcharena/engine/__init__.py](/Users/aman/Documents/search-env/searcharena/engine/__init__.py:1) exposes runtime classes
- [searcharena/engine/core.py](/Users/aman/Documents/search-env/searcharena/engine/core.py:1) is the main environment module today
- [searcharena/engine/retrieval.py](/Users/aman/Documents/search-env/searcharena/engine/retrieval.py:1) contains BM25 and corpus management
- [searcharena/engine/rewards.py](/Users/aman/Documents/search-env/searcharena/engine/rewards.py:1) contains reward logic
- [searcharena/engine/handlers.py](/Users/aman/Documents/search-env/searcharena/engine/handlers.py:1) contains per-action helper functions
- [searcharena/engine/observations.py](/Users/aman/Documents/search-env/searcharena/engine/observations.py:1) renders observations
- [searcharena/engine/factory.py](/Users/aman/Documents/search-env/searcharena/engine/factory.py:1) creates sample corpora/tasks

## Training Package

- [searcharena/training/config.py](/Users/aman/Documents/search-env/searcharena/training/config.py:1)
- [searcharena/training/prompts.py](/Users/aman/Documents/search-env/searcharena/training/prompts.py:1)
- [searcharena/training/datasets.py](/Users/aman/Documents/search-env/searcharena/training/datasets.py:1)
- [searcharena/training/metrics.py](/Users/aman/Documents/search-env/searcharena/training/metrics.py:1)
- [searcharena/training/evaluation.py](/Users/aman/Documents/search-env/searcharena/training/evaluation.py:1)
- [searcharena/training/curriculum.py](/Users/aman/Documents/search-env/searcharena/training/curriculum.py:1)

## Server Package

- [server/app.py](/Users/aman/Documents/search-env/server/app.py:1) is the actual HTTP app
- [server/environment.py](/Users/aman/Documents/search-env/server/environment.py:1) adapts the engine runtime to OpenEnv metadata and types

## Root Compatibility Files

- [client.py](/Users/aman/Documents/search-env/client.py:1) re-exports `searcharena.client`
- [models.py](/Users/aman/Documents/search-env/models.py:1) re-exports `searcharena.models`
- These appear to exist to satisfy package layout / submission expectations

## Naming Patterns

- Runtime code uses descriptive modules rather than deep subpackages
- Tests mirror package areas: `test_environment`, `test_retrieval`, `test_rewards`, `test_training`
- Data generator subtrees are organized by domain (`web`, `sec`, `patents`, `epstein`) under [data/generator/domains](/Users/aman/Documents/search-env/data/generator/domains)
