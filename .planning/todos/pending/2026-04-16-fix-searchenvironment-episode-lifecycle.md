---
created: 2026-04-16T17:50:28.588Z
title: Fix SearchEnvironment episode lifecycle
area: general
files:
  - searcharena/engine/core.py:165
  - searcharena/engine/core.py:198
  - searcharena/engine/core.py:203
  - searcharena/engine/core.py:288
  - searcharena/engine/handlers.py:141
  - server/environment.py:24
  - tests/test_environment.py
  - tests/test_edge_cases.py
---

## Problem

`SearchEnvironment` has a few lifecycle and API consistency issues that should be fixed together rather than patched one at a time.

- `max_steps` is off by one in `searcharena/engine/core.py`: `step_count` is incremented before the limit check, so the final allowed step becomes a forced timeout instead of executing the requested action.
- `reset()` returns `done=True` when there are no tasks, but it does not set the environment's internal `_done` flag, leaving observation state and runtime state out of sync.
- The environment advertises `SUPPORTS_CONCURRENT_SESSIONS = True`, but the implementation stores all episode state on a single mutable instance (`_state`, `_current_task`, `_tracker`, `_context_chunks`, `_done`), which looks unsafe unless the outer runtime creates strict per-session isolation.
- Final-answer handling is duplicated between timeout finalization and explicit `answer()` handling, which increases drift risk.
- `supporting_chunk_ids` is accepted but not used in reward computation or validation, so the public API promises evidence tracking that the implementation does not actually enforce.

These issues surfaced while reviewing `searcharena/engine/core.py` for improvement work. Two of them were verified directly in local execution: the step-limit off-by-one behavior and the empty-task reset state mismatch.

## Solution

Refactor episode-finalization and state management in one pass.

- Add regression tests for:
  - exact `max_steps` behavior
  - `reset()` with no tasks
  - any intended concurrent-session semantics
- Extract a shared helper for episode finalization so timeout and explicit answer use the same reward path.
- Fix step ordering so exactly `max_steps` actions are allowed.
- Make the no-task reset path internally terminal, not just observationally terminal.
- Decide whether `SUPPORTS_CONCURRENT_SESSIONS` is truly valid; if not, disable it or move episode state behind real per-session isolation.
- Either wire `supporting_chunk_ids` into validation/reward logic or remove it from the action contract.
