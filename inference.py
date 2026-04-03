"""
Inference script for the Search RL Environment.

Required: OPENAI_API_KEY, (LOCAL_IMAGE_NAME or ENV_BASE_URL)
Optional: MODEL_NAME, OPENAI_BASE_URL, NUM_EPISODES, MAX_STEPS, SEARCH_TOP_K,
          READ_TOP_K, TEMPERATURE, LOG_FILE, SEARCH_BACKEND, SERPER_API_KEY
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

try:
    from .log_utils import append_log_record as sync_log
    from .log_utils import append_log_record_async as async_log
    from .models import SearchAction
except ImportError:
    from log_utils import append_log_record as sync_log
    from log_utils import append_log_record_async as async_log
    from models import SearchAction

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


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


@dataclass
class Config:
    openai_base_url: str = os.getenv(
        "OPENAI_BASE_URL", "https://router.huggingface.co/v1"
    )
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    local_image_name: str = os.getenv("LOCAL_IMAGE_NAME", "")
    env_base_url: str = os.getenv("ENV_BASE_URL", "")
    num_episodes: int = int(os.getenv("NUM_EPISODES", "1") or "1")
    search_top_k: int = int(os.getenv("SEARCH_TOP_K", "5"))
    read_top_k: int = int(os.getenv("READ_TOP_K", "2"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("MAX_COMPLETION_TOKENS", "350"))
    max_retries: int = 4
    log_file: str = os.getenv("LOG_FILE", "").strip()
    char_limit: int = 800
    soft_budget_threshold: float = 0.75
    hard_budget_threshold: float = 0.95
    prune_target_threshold: float = 0.60

    @property
    def reasoning_effort(self) -> Optional[str]:
        return "low" if "gpt-oss" in self.model_name.lower() else None


CFG = Config()
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "did",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "much",
    "compare",
    "compared",
}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, limit: int = 200) -> str:
    text = clean(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class Logger:
    def __init__(self):
        self._tasks: Set[asyncio.Task] = set()
        self._lock: Optional[asyncio.Lock] = None
        self.episode = 0
        self.task = ""
        self.backend = ""

    def _record(self, event: str, **data) -> Dict[str, Any]:
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "episode": self.episode,
            "task": self.task[:80],
            "backend": self.backend,
            "model": CFG.model_name,
            **data,
        }

    def _write(self, record: Dict[str, Any]):
        if not CFG.log_file:
            return
        try:
            loop = asyncio.get_running_loop()
            t = loop.create_task(self._async_write(record))
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)
        except RuntimeError:
            sync_log(CFG.log_file, record)

    async def _async_write(self, record: Dict[str, Any]):
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            await async_log(CFG.log_file, record)

    async def flush(self):
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def start(self, task: str, backend: str, max_steps: int, budget: int):
        print(f"[START] {clean(task)[:60]} | backend={backend}", flush=True)
        self._write(self._record("start", max_steps=max_steps, budget=budget))

    def step(
        self,
        step: int,
        action: str,
        done: bool,
        tokens: int,
        budget_pct: float,
        error: Optional[str] = None,
    ):
        err = f" | error={clean(error)}" if error else ""
        done_str = " [DONE]" if done else ""
        print(f"[{step:2d}] {truncate(action, 70)}{done_str}{err}", flush=True)
        self._write(
            self._record(
                "step",
                step=step,
                action=action,
                done=done,
                tokens=tokens,
                budget_pct=round(budget_pct, 1),
                error=error,
            )
        )

    def budget(self, used: int, limit: int, pct: float):
        bar = "=" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"    [{bar}] {used:,}/{limit:,} ({pct:.1f}%)", flush=True)

    def end(self, success: bool, steps: int, reward: float):
        print(
            f"[END] success={success} steps={steps} reward={reward:.3f}\n", flush=True
        )
        self._write(self._record("end", success=success, steps=steps, reward=reward))

    def reward(self, step: int, metrics: Dict[str, Any]):
        # Print reward breakdown to console
        final = metrics.get("final_reward", 0) or 0
        f_beta_r = metrics.get("f_beta_reward", 0) or 0
        traj_r = metrics.get("trajectory_reward", 0) or 0
        answer_r = metrics.get("answer_reward", 0) or 0
        turn_p = metrics.get("turn_penalty", 0) or 0
        prune_p = metrics.get("prune_penalty", 0) or 0

        print(f"[REWARD] final={final:.3f}", flush=True)
        print(
            f"    components: f_beta={f_beta_r:.3f} traj={traj_r:.3f} answer={answer_r:.3f}",
            flush=True,
        )
        if turn_p > 0 or prune_p > 0:
            print(f"    penalties:  turn={turn_p:.3f} prune={prune_p:.3f}", flush=True)

        self._write(
            self._record(
                "reward",
                step=step,
                metrics={
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metrics.items()
                    if v is not None
                },
            )
        )


LOG = Logger()

TOOLS = [
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

Default top_k: {CFG.search_top_k}"""


class ActionBuilder:
    def __init__(self, observation):
        self.obs = observation
        self.result = observation.action_result or {}
        self.context = observation.context_chunks or []
        self.context_ids = {c.chunk_id for c in self.context}
        self.issued_queries = {
            clean(q).lower() for q in (observation.queries_issued or [])
        }
        self.budget_pct = observation.budget_usage_percent / 100.0

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[SearchAction, str]:
        k = top_k if top_k is not None else CFG.search_top_k
        return SearchAction.make_search(
            query, k
        ), f"search('{truncate(query, 40)}', k={k})"

    def read(self, chunk_ids: List[str]) -> Tuple[SearchAction, str]:
        return SearchAction.make_read(chunk_ids), f"read({len(chunk_ids)} chunks)"

    def prune(self, chunk_ids: List[str]) -> Tuple[SearchAction, str]:
        return SearchAction.make_prune(chunk_ids), f"prune({len(chunk_ids)} chunks)"

    def answer(
        self, text: str, support_ids: Optional[List[str]] = None
    ) -> Tuple[SearchAction, str]:
        ids = support_ids if support_ids else list(self.context_ids)
        return SearchAction.make_answer(text, ids), f"answer('{truncate(text, 30)}')"

    def from_tool_call(self, tool_call) -> Tuple[SearchAction, str, str]:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")

        if name == "search":
            query = clean(args.get("query", "")) or self.obs.question
            top_k = args.get("top_k")
            action, desc = self.search(query, top_k)
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
            support = [
                s for s in args.get("supporting_chunk_ids", []) if s in self.context_ids
            ]
            return *self.answer(text, support if support else None), tool_call.id

        raise ValueError(f"Unknown tool: {name}")

    def auto(self) -> Tuple[SearchAction, str]:
        if self._should_prune():
            return self._prune_lowest_relevance()

        if self._has_unread_results():
            return self._read_top_results()

        if self.context:
            return self._answer_from_context()

        return self.search(self.obs.question)

    def _should_prune(self) -> bool:
        if len(self.context) < 2:
            return False
        if self.budget_pct >= CFG.hard_budget_threshold:
            return True
        if self.budget_pct >= CFG.soft_budget_threshold and len(self.context) > 3:
            return True
        return False

    def _prune_lowest_relevance(self) -> Tuple[SearchAction, str]:
        sorted_chunks = sorted(self.context, key=lambda c: getattr(c, "score", 0))
        tokens_to_free = int(
            self.obs.context_token_count
            * (1 - CFG.prune_target_threshold / self.budget_pct)
        )

        chunks_to_prune = []
        tokens_freed = 0
        for chunk in sorted_chunks:
            if tokens_freed >= tokens_to_free:
                break
            chunks_to_prune.append(chunk.chunk_id)
            tokens_freed += chunk.token_count

        if not chunks_to_prune:
            chunks_to_prune = [sorted_chunks[0].chunk_id]

        return self.prune(chunks_to_prune)

    def _has_unread_results(self) -> bool:
        results = self.result.get("results", [])
        return bool(results) and any(
            r.get("chunk_id") not in self.context_ids for r in results
        )

    def _read_top_results(self) -> Tuple[SearchAction, str]:
        results = self.result.get("results", [])
        ids = [
            r["chunk_id"]
            for r in results[: CFG.read_top_k]
            if r.get("chunk_id") and r["chunk_id"] not in self.context_ids
        ]
        return self.read(ids) if ids else self._answer_from_context()

    def _answer_from_context(self) -> Tuple[SearchAction, str]:
        return self.answer(self._extract_answer())

    def _extract_answer(self) -> str:
        question_words = {
            w
            for w in re.findall(r"\w+", self.obs.question.lower())
            if w not in STOPWORDS
        }

        texts = []
        for chunk in self.result.get("chunks", []):
            texts.append(clean(chunk.get("content", "")))
        for chunk in self.context:
            texts.append(clean(getattr(chunk, "snippet", "")))

        scored_sentences = []
        for text in texts:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                sentence = clean(sentence)
                if not sentence:
                    continue
                words = set(re.findall(r"\w+", sentence.lower()))
                score = len(words & question_words)
                if re.search(r"\d", sentence):
                    score += 1
                if score > 0:
                    scored_sentences.append((score, sentence))

        if scored_sentences:
            scored_sentences.sort(key=lambda x: (-x[0], len(x[1])))
            return " ".join(s for _, s in scored_sentences[:3])

        if texts:
            return texts[0]

        return f"Based on retrieved evidence: {self.obs.question}"


def build_state(obs, step: int, max_steps: int, full_context: Dict[str, str]) -> str:
    context_items = []
    for chunk in (obs.context_chunks or [])[:3]:
        text = full_context.get(chunk.chunk_id) or getattr(chunk, "snippet", "")
        context_items.append(
            {
                "id": chunk.chunk_id,
                "title": truncate(chunk.title, 40),
                "tokens": chunk.token_count,
                "text": truncate(text, CFG.char_limit),
            }
        )

    budget_pct = obs.budget_usage_percent
    budget_status = "ok"
    if budget_pct >= CFG.hard_budget_threshold * 100:
        budget_status = "CRITICAL - must prune or answer"
    elif budget_pct >= CFG.soft_budget_threshold * 100:
        budget_status = "high - consider pruning"

    state = {
        "step": step,
        "remaining": max_steps - step,
        "budget": {
            "used": obs.context_token_count,
            "limit": obs.context_token_budget,
            "percent": round(budget_pct, 1),
            "status": budget_status,
        },
        "context": context_items,
        "last_action": _summarize_action(obs),
    }
    return f"Question: {obs.question}\n\nState:\n{json.dumps(state, indent=2)}"


def _summarize_action(obs) -> Dict[str, Any]:
    action_type = obs.action_type or "none"
    result = obs.action_result or {}

    if action_type == "search":
        return {
            "type": "search",
            "query": result.get("query"),
            "results": [
                {
                    "id": r.get("chunk_id"),
                    "title": truncate(r.get("title", ""), 40),
                    "snippet": truncate(r.get("snippet", ""), 100),
                }
                for r in result.get("results", [])[:4]
            ],
        }

    if action_type == "read":
        return {
            "type": "read",
            "tokens_added": result.get("tokens_added"),
            "chunks": [
                {"id": c.get("chunk_id"), "text": truncate(c.get("content", ""), 150)}
                for c in result.get("chunks", [])[:2]
            ],
        }

    if action_type == "prune":
        return {
            "type": "prune",
            "removed": result.get("chunks_removed"),
            "freed": result.get("tokens_freed"),
        }

    if action_type == "answer":
        return {
            "type": "answer",
            "reward": result.get("final_reward"),
            "correct": result.get("answer_correct"),
        }

    return {"type": action_type}


def build_tool_result(tool_id: str, obs, step: int, max_steps: int) -> Dict:
    return {
        "role": "tool",
        "tool_call_id": tool_id,
        "content": json.dumps(
            {
                "step": step + 1,
                "remaining": max_steps - step - 1,
                "done": obs.done,
                "reward": obs.reward or 0.0,
                "budget": {
                    "used": obs.context_token_count,
                    "percent": round(obs.budget_usage_percent, 1),
                },
                "last_action": _summarize_action(obs),
            }
        ),
    }


async def get_action(
    client: AsyncOpenAI, obs, step: int, max_steps: int, messages: List[Dict]
) -> Tuple[SearchAction, str, Optional[Dict], Optional[str]]:
    builder = ActionBuilder(obs)

    for attempt in range(CFG.max_retries):
        try:
            request = {
                "model": CFG.model_name,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": min(CFG.temperature, 0.1),
                "max_tokens": CFG.max_tokens + (attempt * 64),
                "stream": False,
            }
            if CFG.reasoning_effort:
                request["reasoning_effort"] = CFG.reasoning_effort

            completion = await client.chat.completions.create(**request)
            message = completion.choices[0].message
            tool_calls = list(message.tool_calls or [])

            if not tool_calls:
                if (
                    completion.choices[0].finish_reason == "length"
                    and attempt < CFG.max_retries - 1
                ):
                    continue
                raise ValueError("No tool call")

            action, action_str, tool_id = builder.from_tool_call(tool_calls[0])
            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            return action, action_str, assistant_msg, tool_id

        except RateLimitError:
            delay = min(2.0 * (attempt + 1), 8.0)
            if attempt < CFG.max_retries - 1:
                await asyncio.sleep(delay)
            continue

        except Exception as e:
            if "JSON" in str(e) and attempt < CFG.max_retries - 1:
                continue
            break

    action, action_str = builder.auto()
    return action, action_str, None, None


def update_context_cache(obs, cache: Dict[str, str]) -> Dict[str, str]:
    if obs.action_type == "read":
        for chunk in (obs.action_result or {}).get("chunks", []):
            if chunk.get("chunk_id") and chunk.get("content"):
                cache[chunk["chunk_id"]] = chunk["content"]

    current_ids = {c.chunk_id for c in (obs.context_chunks or [])}
    return {k: v for k, v in cache.items() if k in current_ids}


async def run_episode(
    env, client: AsyncOpenAI, episode: int
) -> Tuple[bool, int, float]:
    result = await env.reset()
    obs = result.observation
    max_steps = int(os.getenv("MAX_STEPS", "") or obs.max_steps or 20)

    LOG.episode = episode
    LOG.task = obs.question
    LOG.backend = getattr(obs, "search_backend", "unknown")
    LOG.start(obs.question, LOG.backend, max_steps, obs.context_token_budget)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_state(obs, 1, max_steps, {})},
    ]

    full_context: Dict[str, str] = {}
    total_reward = 0.0
    success = False
    steps = 0

    try:
        for step in range(1, max_steps + 1):
            if result.done:
                break

            action, action_str, assistant_msg, tool_id = await get_action(
                client, obs, step, max_steps, messages
            )

            result = await env.step(action)
            obs = result.observation
            full_context = update_context_cache(obs, full_context)

            steps = step
            error = (obs.action_result or {}).get("error")

            LOG.step(
                step,
                action_str,
                result.done,
                obs.context_token_count,
                obs.budget_usage_percent,
                error,
            )

            if obs.action_type in ("read", "prune"):
                LOG.budget(
                    obs.context_token_count,
                    obs.context_token_budget,
                    obs.budget_usage_percent,
                )

            if result.done:
                answer_result = obs.action_result or {}
                # Get the final reward from the answer result
                total_reward = float(answer_result.get("final_reward", 0) or 0)
                success = bool(
                    answer_result.get("answer_found_in_context") or total_reward > 0
                )
                LOG.reward(
                    step,
                    {
                        "final_reward": answer_result.get("final_reward"),
                        "pre_penalty_reward": answer_result.get("pre_penalty_reward"),
                        "f_beta": answer_result.get("f_beta"),
                        "f_beta_reward": answer_result.get("f_beta_reward"),
                        "trajectory_recall": answer_result.get("trajectory_recall"),
                        "trajectory_reward": answer_result.get("trajectory_reward"),
                        "output_recall": answer_result.get("output_recall"),
                        "output_precision": answer_result.get("output_precision"),
                        "answer_correct": answer_result.get("answer_correct"),
                        "answer_found_in_context": answer_result.get(
                            "answer_found_in_context"
                        ),
                        "answer_reward": answer_result.get("answer_reward"),
                        "turn_penalty": answer_result.get("turn_penalty"),
                        "prune_penalty": answer_result.get("prune_penalty"),
                        "beta_used": answer_result.get("beta_used"),
                    },
                )
                break

            if assistant_msg and tool_id:
                messages.append(assistant_msg)
                messages.append(build_tool_result(tool_id, obs, step, max_steps))
            else:
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": build_state(obs, step + 1, max_steps, full_context),
                    },
                ]

    finally:
        await LOG.flush()
        LOG.end(success, steps, total_reward)

    return success, steps, total_reward


async def create_env():
    if CFG.env_base_url:
        env = SearchEnv(base_url=CFG.env_base_url)
        await env.connect()
        return env

    if not CFG.local_image_name:
        raise RuntimeError("Set LOCAL_IMAGE_NAME or ENV_BASE_URL")

    env_keys = [
        "LOG_FILE",
        "MAX_STEPS",
        "MAX_CONTEXT_TOKENS",
        "SEARCH_TOP_K",
        "SEARCH_BACKEND",
        "SERPER_API_KEY",
    ]
    env_vars = {k: v for k in env_keys if (v := os.getenv(k))}
    return await SearchEnv.from_docker_image(CFG.local_image_name, env_vars=env_vars)


async def main():
    if not CFG.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY required")

    env = None
    try:
        async with AsyncOpenAI(
            base_url=CFG.openai_base_url, api_key=CFG.openai_api_key
        ) as client:
            env = await create_env()
            for ep in range(1, CFG.num_episodes + 1):
                await run_episode(env, client, ep)
    finally:
        await LOG.flush()
        if env:
            try:
                await env.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())


def cli():
    asyncio.run(main())
