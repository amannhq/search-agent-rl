"""LLM client wrapper with retry logic."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
)

from ..models import SearchAction
from .action import ActionBuilder
from .config import InferenceConfig
from .policy import HeuristicPolicy
from .tools import get_tools


async def call_llm(
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
    config: InferenceConfig,
    attempt_extra_tokens: int = 0,
) -> Any:
    """Make a single LLM call."""
    return await client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        tools=get_tools(),
        tool_choice="auto",
        temperature=min(config.temperature, 0.1),
        max_tokens=config.max_tokens + attempt_extra_tokens,
    )


async def get_action(
    client: AsyncOpenAI,
    obs: Any,
    messages: list[ChatCompletionMessageParam],
    config: InferenceConfig,
) -> tuple[SearchAction, str, ChatCompletionMessageParam | None, str | None]:
    """
    Get next action from LLM with retry and fallback.

    Returns:
        Tuple of (action, description, assistant_message, tool_id)
        assistant_message and tool_id are None if using fallback
    """
    builder = ActionBuilder(obs, search_top_k=config.search_top_k)

    for attempt in range(config.max_retries):
        try:
            completion = await call_llm(
                client,
                messages,
                config,
                attempt_extra_tokens=attempt * 64,
            )
            message = completion.choices[0].message
            tool_calls = list(message.tool_calls or [])

            if not tool_calls:
                if (
                    completion.choices[0].finish_reason == "length"
                    and attempt < config.max_retries - 1
                ):
                    continue
                raise ValueError("No tool call returned")

            action, action_str, tool_id = builder.from_tool_call(tool_calls[0])

            std_calls = [
                tc for tc in tool_calls if isinstance(tc, ChatCompletionMessageToolCall)
            ]
            assistant_msg: ChatCompletionMessageParam = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in std_calls
                ],
            }
            return action, action_str, assistant_msg, tool_id

        except RateLimitError:
            if attempt < config.max_retries - 1:
                await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
            continue

        except Exception as exc:
            print(
                f"LLM error (attempt {attempt + 1}/{config.max_retries}): "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < config.max_retries - 1:
                continue
            break

    # All retries exhausted — use heuristic fallback
    policy = HeuristicPolicy(
        builder,
        soft_budget_threshold=config.soft_budget_threshold,
        hard_budget_threshold=config.hard_budget_threshold,
        prune_target_threshold=config.prune_target_threshold,
        read_top_k=config.read_top_k,
    )
    action, action_str = policy.get_action()
    return action, action_str, None, None
