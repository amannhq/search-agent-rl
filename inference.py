"""
Inference script for the Search RL Environment.

Submission system injects: API_BASE_URL, API_KEY, MODEL_NAME
Environment connection: LOCAL_IMAGE_NAME or ENV_BASE_URL
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)

try:
    from .models import SearchAction
except ImportError:
    from models import SearchAction


def _load_search_env():
    try:
        from . import SearchEnv
        return SearchEnv
    except ImportError:
        import importlib.util
        root = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "search_env", root / "__init__.py", submodule_search_locations=[str(root)]
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["search_env"] = module
            spec.loader.exec_module(module)
            return module.SearchEnv
        raise ImportError("Cannot load search_env package")


SearchEnv = _load_search_env()


# ---------------------------------------------------------------------------
# Configuration - simple module-level constants (like OpsArena)
# ---------------------------------------------------------------------------

BENCHMARK = os.getenv("SEARCH_ENV_BENCHMARK", "search_env")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")

# CRITICAL: These are injected by submission system
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

NUM_EPISODES = int(os.getenv("NUM_EPISODES", "1") or "1")
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
READ_TOP_K = int(os.getenv("READ_TOP_K", "2"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "350"))
MAX_RETRIES = 4
CHAR_LIMIT = 800
SOFT_BUDGET_THRESHOLD = 0.75
HARD_BUDGET_THRESHOLD = 0.95
PRUNE_TARGET_THRESHOLD = 0.60

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "did", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "to", "was",
    "were", "what", "when", "where", "which", "who", "much",
    "compare", "compared",
}

_RE_WHITESPACE = re.compile(r"\s+")
_RE_WORD = re.compile(r"\w+")
_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_RE_DIGIT = re.compile(r"\d")


def clean(text: str) -> str:
    return _RE_WHITESPACE.sub(" ", text).strip()


def truncate(text: str, limit: int = 200) -> str:
    text = clean(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Mandatory stdout format for submission
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={clean(action)[:120]} "
        f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# OpenAI tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for evidence",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read chunks into context",
            "parameters": {
                "type": "object",
                "required": ["chunk_ids"],
                "additionalProperties": False,
                "properties": {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prune",
            "description": "Remove chunks from context",
            "parameters": {
                "type": "object",
                "required": ["chunk_ids"],
                "additionalProperties": False,
                "properties": {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Submit final answer",
            "parameters": {
                "type": "object",
                "required": ["answer"],
                "additionalProperties": False,
                "properties": {
                    "answer": {"type": "string"},
                    "supporting_chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    },
]

SYSTEM = f"""You are a retrieval agent. Call exactly one tool per turn.

Available: search, read, prune, answer

Strategy:
1. Search for relevant evidence
2. Read promising results into context
3. Prune low-relevance chunks when budget > 75%
4. Answer when context supports it

Budget thresholds:
- 75%: Consider pruning low-relevance chunks
- 95%: Must prune or answer (search/read blocked)

Default top_k: {SEARCH_TOP_K}"""


# ---------------------------------------------------------------------------
# ActionBuilder: converts LLM tool calls to env actions, with auto-fallback
# ---------------------------------------------------------------------------

class ActionBuilder:
    def __init__(self, observation: Any) -> None:
        self.obs = observation
        self.result: dict[str, Any] = observation.action_result or {}
        self.context = observation.context_chunks or []
        self.context_ids: set[str] = {c.chunk_id for c in self.context}
        self.budget_pct: float = observation.budget_usage_percent / 100.0

    def search(self, query: str, top_k: int | None = None) -> tuple[SearchAction, str]:
        k = top_k if top_k is not None else SEARCH_TOP_K
        return SearchAction.make_search(query, k), f"search('{truncate(query, 40)}', k={k})"

    def read(self, chunk_ids: list[str]) -> tuple[SearchAction, str]:
        return SearchAction.make_read(chunk_ids), f"read({len(chunk_ids)} chunks)"

    def prune(self, chunk_ids: list[str]) -> tuple[SearchAction, str]:
        return SearchAction.make_prune(chunk_ids), f"prune({len(chunk_ids)} chunks)"

    def answer(self, text: str, support_ids: list[str] | None = None) -> tuple[SearchAction, str]:
        ids = support_ids if support_ids else list(self.context_ids)
        return SearchAction.make_answer(text, ids), f"answer('{truncate(text, 30)}')"

    def from_tool_call(self, tool_call: Any) -> tuple[SearchAction, str, str]:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")

        if name == "search":
            query = clean(args.get("query", "")) or self.obs.question
            action, desc = self.search(query, args.get("top_k"))
            return action, desc, tool_call.id

        if name == "read":
            ids = [str(c) for c in args.get("chunk_ids", []) if c]
            if not ids:
                raise ValueError("read requires chunk_ids")
            return *self.read(ids), tool_call.id

        if name == "prune":
            ids = [str(c) for c in args.get("chunk_ids", []) if c]
            if not ids:
                raise ValueError("prune requires chunk_ids")
            return *self.prune(ids), tool_call.id

        if name == "answer":
            text = clean(args.get("answer", ""))
            if not text:
                raise ValueError("answer requires text")
            support = [s for s in args.get("supporting_chunk_ids", []) if s in self.context_ids]
            return *self.answer(text, support or None), tool_call.id

        raise ValueError(f"Unknown tool: {name}")

    def auto(self) -> tuple[SearchAction, str]:
        """Heuristic fallback when LLM is unavailable."""
        if self._should_prune():
            return self._prune_lowest()
        if self._has_unread_results():
            return self._read_top()
        if self.context:
            return self.answer(self._extract_answer())
        return self.search(self.obs.question)

    def _should_prune(self) -> bool:
        if len(self.context) < 2:
            return False
        return (
            self.budget_pct >= HARD_BUDGET_THRESHOLD
            or (self.budget_pct >= SOFT_BUDGET_THRESHOLD and len(self.context) > 3)
        )

    def _prune_lowest(self) -> tuple[SearchAction, str]:
        sorted_chunks = sorted(self.context, key=lambda c: getattr(c, "score", 0))
        target = int(self.obs.context_token_count * (1 - PRUNE_TARGET_THRESHOLD / self.budget_pct))
        to_prune, freed = [], 0
        for chunk in sorted_chunks:
            if freed >= target:
                break
            to_prune.append(chunk.chunk_id)
            freed += chunk.token_count
        if not to_prune:
            to_prune = [sorted_chunks[0].chunk_id]
        return self.prune(to_prune)

    def _has_unread_results(self) -> bool:
        results = self.result.get("results", [])
        return bool(results) and any(r.get("chunk_id") not in self.context_ids for r in results)

    def _read_top(self) -> tuple[SearchAction, str]:
        results = self.result.get("results", [])
        ids = [
            r["chunk_id"]
            for r in results[: READ_TOP_K]
            if r.get("chunk_id") and r["chunk_id"] not in self.context_ids
        ]
        return self.read(ids) if ids else self.answer(self._extract_answer())

    def _extract_answer(self) -> str:
        q_words = {w for w in _RE_WORD.findall(self.obs.question.lower()) if w not in STOPWORDS}
        texts: list[str] = []
        for chunk in self.result.get("chunks", []):
            texts.append(clean(chunk.get("content", "")))
        for chunk in self.context:
            texts.append(clean(getattr(chunk, "snippet", "")))

        scored: list[tuple[int, str]] = []
        for text in texts:
            for sent in _RE_SENTENCE_SPLIT.split(text):
                sent = clean(sent)
                if not sent:
                    continue
                words = set(_RE_WORD.findall(sent.lower()))
                sc = len(words & q_words) + (1 if _RE_DIGIT.search(sent) else 0)
                if sc > 0:
                    scored.append((sc, sent))

        if scored:
            scored.sort(key=lambda x: (-x[0], len(x[1])))
            return " ".join(s for _, s in scored[:3])
        return texts[0] if texts else f"Based on retrieved evidence: {self.obs.question}"


# ---------------------------------------------------------------------------
# LLM message helpers
# ---------------------------------------------------------------------------

def build_state(obs: Any, step: int, max_steps: int, full_context: dict[str, str]) -> str:
    context_items = []
    for chunk in (obs.context_chunks or [])[:3]:
        text = full_context.get(chunk.chunk_id) or getattr(chunk, "snippet", "")
        context_items.append({
            "id": chunk.chunk_id,
            "title": truncate(chunk.title, 40),
            "tokens": chunk.token_count,
            "text": truncate(text, CHAR_LIMIT),
        })

    pct = obs.budget_usage_percent
    if pct >= HARD_BUDGET_THRESHOLD * 100:
        status = "CRITICAL - must prune or answer"
    elif pct >= SOFT_BUDGET_THRESHOLD * 100:
        status = "high - consider pruning"
    else:
        status = "ok"

    state = {
        "step": step,
        "remaining": max_steps - step,
        "budget": {"used": obs.context_token_count, "limit": obs.context_token_budget, "percent": round(pct, 1), "status": status},
        "context": context_items,
        "last_action": _summarize_action(obs),
    }
    return f"Question: {obs.question}\n\nState:\n{json.dumps(state, indent=2)}"


def _summarize_action(obs: Any) -> dict[str, Any]:
    at = obs.action_type or "none"
    r = obs.action_result or {}
    if at == "search":
        return {"type": "search", "query": r.get("query"), "results": [
            {"id": x.get("chunk_id"), "title": truncate(x.get("title", ""), 40), "snippet": truncate(x.get("snippet", ""), 100)}
            for x in r.get("results", [])[:4]
        ]}
    if at == "read":
        return {"type": "read", "tokens_added": r.get("tokens_added"), "chunks": [
            {"id": c.get("chunk_id"), "text": truncate(c.get("content", ""), 150)} for c in r.get("chunks", [])[:2]
        ]}
    if at == "prune":
        return {"type": "prune", "removed": r.get("chunks_removed"), "freed": r.get("tokens_freed")}
    if at == "answer":
        return {"type": "answer", "reward": r.get("final_reward"), "correct": r.get("answer_correct")}
    return {"type": at}


def build_tool_result(tool_id: str, obs: Any, step: int, max_steps: int) -> ChatCompletionMessageParam:
    return {
        "role": "tool",
        "tool_call_id": tool_id,
        "content": json.dumps({
            "step": step + 1, "remaining": max_steps - step - 1, "done": obs.done,
            "reward": obs.reward or 0.0,
            "budget": {"used": obs.context_token_count, "percent": round(obs.budget_usage_percent, 1)},
            "last_action": _summarize_action(obs),
        }),
    }


# ---------------------------------------------------------------------------
# LLM call with retries. Always attempts the call (so the proxy sees it).
# Falls back to heuristic only after all retries are exhausted.
# ---------------------------------------------------------------------------

async def call_llm(
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
    attempt_extra_tokens: int = 0,
) -> Any:
    """Make a single LLM call. Raises on failure."""
    return await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=min(TEMPERATURE, 0.1),
        max_tokens=MAX_TOKENS + attempt_extra_tokens,
    )


async def get_action(
    client: AsyncOpenAI,
    obs: Any,
    messages: list[ChatCompletionMessageParam],
) -> tuple[SearchAction, str, ChatCompletionMessageParam | None, str | None]:
    builder = ActionBuilder(obs)

    for attempt in range(MAX_RETRIES):
        try:
            completion = await call_llm(client, messages, attempt_extra_tokens=attempt * 64)
            message = completion.choices[0].message
            tool_calls = list(message.tool_calls or [])

            if not tool_calls:
                if completion.choices[0].finish_reason == "length" and attempt < MAX_RETRIES - 1:
                    continue
                raise ValueError("No tool call returned")

            action, action_str, tool_id = builder.from_tool_call(tool_calls[0])

            std_calls = [tc for tc in tool_calls if isinstance(tc, ChatCompletionMessageToolCall)]
            assistant_msg: ChatCompletionMessageParam = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in std_calls
                ],
            }
            return action, action_str, assistant_msg, tool_id

        except RateLimitError:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
            continue

        except Exception as exc:
            print(f"LLM error (attempt {attempt + 1}/{MAX_RETRIES}): {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if attempt < MAX_RETRIES - 1:
                continue
            break

    # All retries exhausted — use heuristic fallback
    action, action_str = builder.auto()
    return action, action_str, None, None


def update_context_cache(obs: Any, cache: dict[str, str]) -> dict[str, str]:
    if obs.action_type == "read":
        for chunk in (obs.action_result or {}).get("chunks", []):
            if chunk.get("chunk_id") and chunk.get("content"):
                cache[chunk["chunk_id"]] = chunk["content"]
    current_ids = {c.chunk_id for c in (obs.context_chunks or [])}
    return {k: v for k, v in cache.items() if k in current_ids}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(env: Any, client: AsyncOpenAI) -> tuple[bool, int, float, list[float]]:
    result = await env.reset()
    obs = result.observation
    max_steps = int(os.getenv("MAX_STEPS", "") or obs.max_steps or 20)

    log_start(task=clean(obs.question)[:60], env=BENCHMARK, model=MODEL_NAME)

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_state(obs, 1, max_steps, {})},
    ]

    full_context: dict[str, str] = {}
    rewards: list[float] = []
    total_reward = 0.0
    success = False
    steps = 0

    try:
        for step in range(1, max_steps + 1):
            if result.done:
                break

            try:
                action, action_str, assistant_msg, tool_id = await get_action(client, obs, messages)
            except Exception:
                builder = ActionBuilder(obs)
                action, action_str = builder.auto()
                assistant_msg, tool_id = None, None

            try:
                result = await env.step(action)
            except Exception as e:
                log_step(step=step, action=action_str, reward=0.0, done=True, error=str(e))
                rewards.append(0.0)
                steps = step
                break

            obs = result.observation
            full_context = update_context_cache(obs, full_context)
            steps = step
            step_reward = result.reward or 0.0
            rewards.append(step_reward)

            log_step(step=step, action=action_str, reward=step_reward, done=result.done,
                     error=(obs.action_result or {}).get("error"))

            if result.done:
                ar = obs.action_result or {}
                total_reward = float(ar.get("final_reward", 0) or 0)
                success = bool(ar.get("answer_found_in_context") or total_reward > 0)
                break

            if assistant_msg and tool_id:
                messages.append(assistant_msg)
                messages.append(build_tool_result(tool_id, obs, step, max_steps))
            else:
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_state(obs, step + 1, max_steps, full_context)},
                ]
    except Exception:
        pass

    score = max(0.0, min(1.0, total_reward))
    log_end(success=success, steps=steps, score=score, rewards=rewards)
    return success, steps, score, rewards


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def create_env() -> Any:
    if LOCAL_IMAGE_NAME:
        env_keys = ["MAX_STEPS", "MAX_CONTEXT_TOKENS", "SEARCH_TOP_K"]
        env_vars = {k: v for k in env_keys if (v := os.getenv(k))}
        return await SearchEnv.from_docker_image(LOCAL_IMAGE_NAME, env_vars=env_vars)
    if ENV_BASE_URL:
        env = SearchEnv(base_url=ENV_BASE_URL)
        await env.connect()
        return env
    raise RuntimeError("Set LOCAL_IMAGE_NAME or ENV_BASE_URL")


async def main() -> None:
    # Validate required env vars for submission
    if not API_KEY:
        raise RuntimeError("API_KEY environment variable is required but not set")
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL environment variable is required but not set")

    print(f"Config: base_url={API_BASE_URL} model={MODEL_NAME} api_key={'set' if API_KEY else 'MISSING'}", file=sys.stderr, flush=True)

    env = None
    try:
        client = AsyncOpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
        env = await create_env()
        for _ in range(NUM_EPISODES):
            try:
                await run_episode(env, client)
            except Exception:
                log_end(success=False, steps=0, score=0.0, rewards=[])
    except Exception as e:
        print(f"Fatal error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        log_start(task="error", env=BENCHMARK, model=MODEL_NAME)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        raise
    finally:
        if env:
            try:
                await env.close()
            except Exception:
                pass


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
