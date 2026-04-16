# Concerns

## Runtime Lifecycle Risks

- `SearchEnvironment` in [searcharena/engine/core.py](/Users/aman/Documents/search-env/searcharena/engine/core.py:1) currently mixes episode lifecycle, dispatch, budget enforcement, and reward finalization in one class
- Recent local todo notes indicate likely bugs around step-limit handling and reset behavior for empty task sets
- The reward finalization path appears duplicated across timeout and explicit answer handling, which raises drift risk

## Concurrency / State Risks

- `SUPPORTS_CONCURRENT_SESSIONS` is declared in both [searcharena/engine/core.py](/Users/aman/Documents/search-env/searcharena/engine/core.py:1) and [server/environment.py](/Users/aman/Documents/search-env/server/environment.py:1)
- The FastAPI layer uses a single global `_env` instance in [server/app.py](/Users/aman/Documents/search-env/server/app.py:41)
- The runtime stores mutable episode state directly on the environment instance, which may not match the advertised concurrency model

## Packaging Risks

- [pyproject.toml](/Users/aman/Documents/search-env/pyproject.toml:1) declares packages as `search_env` / `search_env.server`, while the actual main package directory is `searcharena`
- Root shim files [client.py](/Users/aman/Documents/search-env/client.py:1) and [models.py](/Users/aman/Documents/search-env/models.py:1) help bridge this, but the arrangement is fragile and worth validating before deeper refactors

## Server Behavior Risks

- `/reset` in [server/app.py](/Users/aman/Documents/search-env/server/app.py:108) always returns `done=False` in the HTTP response wrapper even if the underlying observation is terminal
- The API layer currently trusts in-memory state, which is fine for local dev but risky for true multi-user serving

## Testing / Verification Risks

- The suite is strong on unit behavior but weak on end-to-end server validation
- No visible tests cover the singleton server runtime, metadata endpoint behavior, or packaging/install paths
- The inference script and submission flow have operational complexity that is not exercised in pytest

## Brownfield Cleanup Candidates

- Clarify the authoritative runtime module and avoid accidental duplication in the engine package
- Tighten lifecycle regression tests before refactoring the engine layout
- Decide whether concurrency is truly supported or should be disabled/documented as single-session only
- Reconcile package naming (`searcharena` vs `search_env`) before release automation or larger structural work
