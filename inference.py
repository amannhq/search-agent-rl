"""
Inference script for the Search RL Environment.

Required: API_BASE_URL, MODEL_NAME, HF_TOKEN, (LOCAL_IMAGE_NAME or ENV_BASE_URL)
Optional: NUM_EPISODES, MAX_STEPS, SEARCH_TOP_K,
          READ_TOP_K, TEMPERATURE, LOG_FILE
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

try:
    from .models import SearchAction
except ImportError:
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
    openai_base_url: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model_name: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
    openai_api_key: str = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
    local_image_name: str = os.getenv("LOCAL_IMAGE_NAME") or ""
    env_base_url: str = os.getenv("ENV_BASE_URL") or ""
    benchmark: str = os.getenv("SEARCH_ENV_BENCHMARK", "search_env")
    num_episodes: int = int(os.getenv("NUM_EPISODES", "1") or "1")
    search_top_k: int = int(os.getenv("SEARCH_TOP_K", "5"))
    read_top_k: int = int(os.getenv("READ_TOP_K", "2"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("MAX_COMPLETION_TOKENS", "350"))
    max_retries: int = 4
    char_limit: int = 800
    soft_budget_threshold: float = 0.75
    hard_budget_threshold: float = 0.95
    prune_target_threshold: float = 0.60

    @property
    def reasoning_effort(self) -> Optional[str]:
        return "low" if "gpt-oss" in self.model_name.lower() else None


CFG = Config()

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


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = clean(action)[:120]
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


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
        return SearchAction.make_search(query, k), f"search('{truncate(query, 40)}', k={k})"

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
            w for w in _RE_WORD.findall(self.obs.question.lower())
            if w not in STOPWORDS
        }

        texts = []
        for chunk in self.result.get("chunks", []):
            texts.append(clean(chunk.get("content", "")))
        for chunk in self.context:
            texts.append(clean(getattr(chunk, "snippet", "")))

        scored_sentences = []
        for text in texts:
            for sentence in _RE_SENTENCE_SPLIT.split(text):
                sentence = clean(sentence)
                if not sentence:
                    continue
                words = set(_RE_WORD.findall(sentence.lower()))
                score = len(words & question_words)
                if _RE_DIGIT.search(sentence):
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
            request: Dict[str, Any] = {
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

        except Exception:
            if attempt < CFG.max_retries - 1:
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
    env, client: AsyncOpenAI,
) -> Tuple[bool, int, float, List[float]]:
    result = await env.reset()
    obs = result.observation
    max_steps = int(os.getenv("MAX_STEPS", "") or obs.max_steps or 20)

    task_name = clean(obs.question)[:60]
    log_start(task=task_name, env=CFG.benchmark, model=CFG.model_name)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_state(obs, 1, max_steps, {})},
    ]

    full_context: Dict[str, str] = {}
    rewards: List[float] = []
    total_reward = 0.0
    success = False
    steps = 0

    try:
        for step in range(1, max_steps + 1):
            if result.done:
                break

            try:
                action, action_str, assistant_msg, tool_id = await get_action(
                    client, obs, step, max_steps, messages
                )
            except Exception:
                builder = ActionBuilder(obs)
                action, action_str = builder.auto()
                assistant_msg, tool_id = None, None

            try:
                result = await env.step(action)
            except Exception as e:
                # Env step failed — log and break
                log_step(step=step, action=action_str, reward=0.0, done=True, error=str(e))
                rewards.append(0.0)
                steps = step
                break

            obs = result.observation
            full_context = update_context_cache(obs, full_context)

            steps = step
            step_reward = result.reward or 0.0
            rewards.append(step_reward)
            error = (obs.action_result or {}).get("error")

            log_step(
                step=step,
                action=action_str,
                reward=step_reward,
                done=result.done,
                error=error,
            )

            if result.done:
                answer_result = obs.action_result or {}
                total_reward = float(answer_result.get("final_reward", 0) or 0)
                success = bool(
                    answer_result.get("answer_found_in_context") or total_reward > 0
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

    except Exception as e:
        # Catch-all for unexpected errors within the episode loop
        print(f"[DEBUG] Episode error: {e}", flush=True)

    # Compute score clamped to [0, 1]
    score = max(0.0, min(1.0, total_reward))

    log_end(success=success, steps=steps, score=score, rewards=rewards)
    return success, steps, score, rewards


async def create_env():
    """Create environment, preferring LOCAL_IMAGE_NAME over ENV_BASE_URL."""
    if CFG.local_image_name:
        env_keys = ["MAX_STEPS", "MAX_CONTEXT_TOKENS", "SEARCH_TOP_K"]
        env_vars = {k: v for k in env_keys if (v := os.getenv(k))}
        return await SearchEnv.from_docker_image(CFG.local_image_name, env_vars=env_vars)

    if CFG.env_base_url:
        env = SearchEnv(base_url=CFG.env_base_url)
        await env.connect()
        return env

    raise RuntimeError("Set LOCAL_IMAGE_NAME or ENV_BASE_URL")


async def main():
    if not CFG.openai_api_key:
        print("[DEBUG] Warning: HF_TOKEN / API key not set, LLM calls will fail", flush=True)

    env = None
    try:
        client = AsyncOpenAI(
            base_url=CFG.openai_base_url, api_key=CFG.openai_api_key or "no-key"
        )
        env = await create_env()

        for ep in range(1, CFG.num_episodes + 1):
            try:
                await run_episode(env, client)
            except Exception as e:
                print(f"[DEBUG] Episode {ep} failed: {e}", flush=True)
                log_end(success=False, steps=0, score=0.0, rewards=[])

    except Exception as e:
        # If env creation or client setup fails, still produce valid output
        print(f"[DEBUG] Fatal error: {e}", flush=True)
        log_start(task="error", env=CFG.benchmark, model=CFG.model_name)
        log_end(success=False, steps=0, score=0.0, rewards=[])
    finally:
        if env:
            try:
                await env.close()
            except Exception:
                pass


def cli():
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
