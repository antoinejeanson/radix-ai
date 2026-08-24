from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    raw_arguments: str = ""


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def system_message(content: str) -> Message:
    return {"role": "system", "content": content}


def user_message(content: str) -> Message:
    return {"role": "user", "content": content}


def assistant_message(content: str) -> Message:
    return {"role": "assistant", "content": content}


def tool_message(tool_call_id: str, content: str) -> Message:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def assistant_tool_call_message(content: str, tool_calls: list[ToolCall]) -> Message:
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
