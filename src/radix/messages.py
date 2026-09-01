from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Message types: chat message helpers plus ToolCall, Usage and ChatResult.
# A chat message is an OpenAI-format dict: {"role", "content", plus
# "tool_call_id"/"tool_calls" for function-calling messages}.
Message = dict[str, Any]


@dataclass
class ToolCall:
    """A tool call requested by the model.

    Fields:
        id: Identifier for the call (used to match tool results).
        name: Name of the tool to invoke.
        raw_arguments: Unparsed JSON string of the arguments as sent by the
            model; empty when the model provided none.
    """

    id: str
    name: str
    raw_arguments: str = ""


@dataclass
class Usage:
    """Token accounting for one completion, as reported by the server.

    Fields:
        prompt_tokens: Tokens used by the request (messages + tools).
        completion_tokens: Tokens used by the reply.
        total_tokens: prompt_tokens + completion_tokens.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResult:
    """The outcome of one chat completion.

    Fields:
        content: The text the model produced ("" when it only called tools).
        tool_calls: Every tool call the model requested, in order.
        usage: Token accounting for the completion, or None when the server
            did not report one.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None


def system_message(content: str) -> Message:
    """Build a system message.

    Args:
        content: The system instruction text.

    Returns:
        A Message dict with role "system".
    """
    return {"role": "system", "content": content}


def user_message(content: str) -> Message:
    """Build a user message.

    Args:
        content: The user's text.

    Returns:
        A Message dict with role "user".
    """
    return {"role": "user", "content": content}


def assistant_message(content: str) -> Message:
    """Build a plain assistant message (no tool calls).

    Args:
        content: The assistant's text.

    Returns:
        A Message dict with role "assistant".
    """
    return {"role": "assistant", "content": content}


def tool_message(tool_call_id: str, content: str) -> Message:
    """Build the result message for one tool call.

    Args:
        tool_call_id: The id of the ToolCall this result answers.
        content: The tool's output text.

    Returns:
        A Message dict with role "tool".
    """
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def assistant_tool_call_message(content: str, tool_calls: list[ToolCall]) -> Message:
    """Build an assistant message announcing one or more tool calls.

    Args:
        content: Accompanying assistant text, or "" for none.
        tool_calls: The calls the model wants to make.

    Returns:
        A Message dict with role "assistant" and a "tool_calls" list.
    """
    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.raw_arguments},
            }
            for tc in tool_calls
        ],
    }
