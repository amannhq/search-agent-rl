"""Tool definitions and system prompts for LLM agent."""

from __future__ import annotations

from openai.types.chat import ChatCompletionToolParam


def get_tools() -> list[ChatCompletionToolParam]:
    """Get tool definitions for the search agent."""
    return [
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


def get_system_prompt(search_top_k: int = 5) -> str:
    """Get system prompt for the search agent."""
    return f"""You are a retrieval agent. Call exactly one tool per turn.

Available: search, read, prune, answer

Strategy:
1. Search for relevant evidence
2. Read promising results into context
3. Prune low-relevance chunks when budget > 75%
4. Answer when context supports it

Budget thresholds:
- 75%: Consider pruning low-relevance chunks
- 95%: Must prune or answer (search/read blocked)

Default top_k: {search_top_k}"""
