---
created: 2026-04-16T18:10:05.992Z
title: Improve OpenEnv SearchEnvironment design
area: general
files:
  - searcharena/engine/core.py:1
  - server/environment.py:1
  - server/app.py:1
  - tests/test_environment.py:1
  - tests/test_edge_cases.py:1
---

## Problem

`SearchEnvironment` currently mixes OpenEnv lifecycle control, mutable episode state, action dispatch, budget enforcement, and reward finalization in one place. That makes the implementation harder to extend and has already led to behavior drift around terminal episodes.

- The final allowed step is easy to mishandle because step counting and forced termination are tightly coupled.
- `reset()` can return a terminal observation for an empty task list while leaving internal runtime state inconsistent.
- The runtime contract around concurrent sessions is unclear because the environment keeps all episode data on a single mutable instance.
- Final reward construction is duplicated across explicit answer submission and forced episode termination.

These issues matter most in `searcharena/engine/core.py`, where the OpenEnv implementation should stay predictable and easy to evolve.

## Solution

Refactor the environment around clearer lifecycle helpers and validate it against OpenEnv expectations.

- Add shared episode-finalization helpers so timeout and explicit answer follow the same reward path.
- Make reset and terminal-state handling internally consistent, not just observationally correct.
- Align concurrency signaling with how the server actually instantiates environments.
- Keep action dispatch simple and make terminal metadata explicit so tests and clients can reason about step-limit endings.
- Review whether additional OpenEnv-focused patterns or skills can help with follow-on implementation work.
