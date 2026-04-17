"""Episode runner for inference."""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .action import ActionBuilder, clean
from .config import InferenceConfig
from .llm import get_action
from .logging import log_start, log_step, log_end
from .policy import HeuristicPolicy
from .state import build_state, build_tool_result, update_context_cache
from .tools import get_system_prompt


async def run_episode(
    env: Any,
    client: AsyncOpenAI,
    config: InferenceConfig,
    task_id: str | None = None,
) -> tuple[bool, int, float, list[float]]:
    """
    Run a single episode.

    Returns:
        Tuple of (success, steps, score, rewards)
    """
    result = await env.reset(task_id=task_id) if task_id else await env.reset()
    obs = result.observation
    max_steps = int(os.getenv("MAX_STEPS", "") or obs.max_steps or 20)

    log_start(
        task=clean(obs.question)[:60],
        env=config.benchmark,
        model=config.model_name,
    )

    system_prompt = get_system_prompt(config.search_top_k)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_state(
            obs, 1, max_steps, {},
            char_limit=config.char_limit,
            soft_budget_threshold=config.soft_budget_threshold,
            hard_budget_threshold=config.hard_budget_threshold,
        )},
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
                action, action_str, assistant_msg, tool_id = await get_action(
                    client, obs, messages, config
                )
            except Exception:
                builder = ActionBuilder(obs, search_top_k=config.search_top_k)
                policy = HeuristicPolicy(
                    builder,
                    soft_budget_threshold=config.soft_budget_threshold,
                    hard_budget_threshold=config.hard_budget_threshold,
                    prune_target_threshold=config.prune_target_threshold,
                    read_top_k=config.read_top_k,
                )
                action, action_str = policy.get_action()
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

            log_step(
                step=step,
                action=action_str,
                reward=step_reward,
                done=result.done,
                error=(obs.action_result or {}).get("error"),
            )

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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_state(
                        obs, step + 1, max_steps, full_context,
                        char_limit=config.char_limit,
                        soft_budget_threshold=config.soft_budget_threshold,
                        hard_budget_threshold=config.hard_budget_threshold,
                    )},
                ]
    except Exception:
        pass

    # Score must be strictly between 0 and 1 (not 0.0, not 1.0)
    score = max(0.001, min(0.999, total_reward))
    log_end(success=success, steps=steps, score=score, rewards=rewards)
    return success, steps, score, rewards
