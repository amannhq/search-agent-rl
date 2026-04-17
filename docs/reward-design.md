# Search RL Environment Implementation Plan

## Goal

Turn the current search environment branch into a clean, reviewable series of commits, then use that foundation to improve the RL environment and reward mechanism in ways that better reflect long-horizon search behavior.

---

## Current Worktree Snapshot

### Changed files that are worth committing

- `IMPLEMENTATION_PLAN.md`
- `__init__.py`
- `client.py`
- `models.py`
- `server/__init__.py`
- `server/app.py`
- `server/environment.py`
- `server/retrieval.py`
- `server/rewards.py`
- `DESIGN.md`
- `CONTEXT_1_PAPER_SUMMARY.md`

### File to keep out unless we explicitly want it

- `links.txt`

Reason: it reads like a personal research scratchpad, not product or implementation documentation.

---

## Recommended Technical Roadmap

These are the next reward and environment improvements worth implementing after the current branch lands.

### 1. Make turn penalties match the actual episode horizon

Problem:
- `max_steps` is short, but the turn penalty only starts much later, so the penalty may never activate.

Implementation:
- Tie `turn_penalty_start` and `turn_penalty_end` to fractions of `max_steps`
- Add tests for short and long horizons

Success criteria:
- Penalty activates for long-winded trajectories under the configured horizon
- Same reward config scales sensibly across different `max_steps`

### 2. Move from chunk-level labels to fact-level supervision

Problem:
- Exact chunk overlap under-rewards valid retrieval when the same fact appears in multiple chunks or documents.

Implementation:
- Extend `SearchTask` with fact groups or equivalent supporting IDs
- Update reward computation to score fact coverage, not only chunk coverage

Success criteria:
- Retrieving an alternate supporting chunk still gets credit
- Recall is less brittle across chunking strategies

### 3. Add ranking-aware reward

Problem:
- The environment currently rewards set membership, not result ordering.

Implementation:
- Add a ranking-sensitive term such as discounted gain over final kept chunks
- Prefer a simple deterministic metric first

Success criteria:
- Keeping the same evidence but ordering stronger chunks earlier improves reward
- Low-quality ranking is distinguishable from high-quality ranking

### 4. Score evidence selection explicitly

Problem:
- `supporting_chunk_ids` are accepted by the answer action but are not used to reward good citation behavior.

Implementation:
- Reward correct supporting chunk selection
- Penalize unsupported or noisy support sets

Success criteria:
- Correctly cited answers outrank unsupported answers with the same text
- Over-citing irrelevant chunks reduces score

### 5. Add novelty rewards and wasted-action penalties

Problem:
- Duplicate queries, invalid prunes, and zero-gain reads are weakly discouraged today.

Implementation:
- Penalize repeated or low-yield actions
- Reward retrieval novelty or marginal recall gain

Success criteria:
- Training signal favors new information over repeated search loops
- Invalid tool use becomes measurably less attractive

### 6. Add abstention and conflict tasks

Problem:
- The current task shape mostly assumes the answer exists and is recoverable.

Implementation:
- Add no-answer tasks
- Add conflicting-evidence tasks
- Add tasks where the correct action is to stop and abstain

Success criteria:
- The agent can learn not only to retrieve, but also when to conclude evidence is insufficient

---

## Commit Strategy For The Current Branch

Target: 5 commits.

This is the most human-looking split for the worktree as it exists today. It reads like a normal implementation arc: API first, retrieval next, environment loop, reward system, then documentation.

### Commit 1

Message:
- `model the search environment API`

Intent:
- Replace the old echo-style API surface with typed search-environment actions, observations, tasks, and client helpers.

Files:
- `__init__.py`
- `client.py`
- `models.py`

Notes:
- In `models.py`, stage the baseline API and environment model hunks first.
- Leave reward-tuning fields such as `beta_used`, `answer_found_in_context`, and beta-schedule config for later commits if you want a cleaner progression.

### Commit 2

Message:
- `add a small BM25 retrieval layer`

Intent:
- Introduce the corpus manager, chunking behavior, and BM25 retrieval primitives the environment depends on.

Files:
- `server/retrieval.py`
- `server/__init__.py`

Notes:
- This commit should feel self-contained and infrastructure-oriented.

### Commit 3

Message:
- `wire up the search environment loop`

Intent:
- Add the actual search/read/prune/answer loop, token budget handling, and sample corpus/tasks.

Files:
- `server/environment.py`
- `server/app.py`
- `server/__init__.py`

Notes:
- Stage the main loop, action handlers, sample data, and server factory wiring here.
- If possible, leave beta-schedule-specific plumbing for the reward follow-up commit.

### Commit 4

Message:
- `track retrieval rewards and episode metrics`

Intent:
- Introduce reward metrics, trajectory tracking, F-beta computation, and final answer scoring.

Files:
- `server/rewards.py`
- reward-related hunks in `models.py`
- reward plumbing hunks in `server/environment.py`
- `server/__init__.py`

Notes:
- This commit should establish the reward framework even if the paper-alignment details come one commit later.

### Commit 5

Message:
- `align the reward with context-1`

Intent:
- Tune the reward behavior to more closely match the Context-1 paper.

Files:
- remaining reward-related hunks in `server/rewards.py`
- remaining reward/config/result hunks in `models.py`
- remaining reward wiring in `server/environment.py`
- `DESIGN.md`
- `CONTEXT_1_PAPER_SUMMARY.md`

Notes:
- This is where the answer-in-context bonus, beta defaults, and beta-schedule wiring belong.
- If you want docs in their own commit, split this into two commits and use the optional sixth commit below.

### Optional Commit 6

Message:
- `document the search env design`

Intent:
- Land the implementation notes and research summary separately.

Files:
- `DESIGN.md`
- `CONTEXT_1_PAPER_SUMMARY.md`
- `IMPLEMENTATION_PLAN.md`

Notes:
- Use this if you want the code history to stay tighter and keep docs as a final cleanup pass.

---

## Practical Staging Notes

Because `models.py` and `server/environment.py` span several concerns, the cleanest history will require partial staging.

Recommended approach:
- Use normal file-based staging where a file is clearly owned by one commit
- Use patch-based staging only for `models.py` and `server/environment.py`

Files that are naturally one-commit files:
- `client.py`
- `__init__.py`
- `server/retrieval.py`
- `server/rewards.py`
- `server/app.py`
- `DESIGN.md`
- `CONTEXT_1_PAPER_SUMMARY.md`

Files that likely need hunk splitting:
- `models.py`
- `server/environment.py`
- `server/__init__.py`

---

## Validation Before Pushing

For each commit, prefer one quick validation step:

- API/model commit: import and schema sanity checks
- Retrieval commit: corpus add/search smoke test
- Environment commit: sample episode walkthrough
- Reward commit: deterministic reward calculation smoke test
- Docs commit: no extra validation needed beyond proofreading

Final pre-push checks:
- `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile server/rewards.py server/environment.py models.py`
- Full environment smoke test once `openenv` is available locally

---

## Suggested Next Step

Start with Commit 1 and keep `links.txt` out of the staging area.

If we want to execute the plan from the terminal, the next action should be:
- review the staged subset for Commit 1
- create the commit
- then repeat for Commit 2
