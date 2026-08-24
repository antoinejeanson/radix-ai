from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .messages import (
    Message,
    ToolCall,
    assistant_message,
    assistant_tool_call_message,
    system_message,
    tool_message,
    user_message,
)
from .permissions import CliPermissionGate, PermissionGate
from .tool import Tool

if TYPE_CHECKING:
    from .client import Client
    from .context import ContextManager

MAX_TOOL_OUTPUT_CHARS = 16000


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [output truncated]"


def _describe_call(call: ToolCall) -> str:
    args = " ".join(call.raw_arguments.split())
    if len(args) > 120:
        args = args[:117] + "..."
    return f"{call.name} {args}".strip()


class Agent:
    """A specialized agent: a system prompt plus a set of tools.

    Sub-agents are stateless: every `run()` starts with a fresh context, so
    only the final answer is handed back to whoever delegated the task.
    Set `stateful=True` (as the coordinator does) to keep conversation
    history between runs.
    """

    def __init__(
        self,
        name: str,
        *,
        description: str = "",
        system_prompt: str = "",
        tools: list[Tool] | None = None,
        client: Client | None = None,
        context: ContextManager | None = None,
        permission_gate: PermissionGate | None = None,
        max_tool_rounds: int = 8,
        stateful: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = list(tools or [])
        self.client = client
        self.context = context
        self.permission_gate = permission_gate or CliPermissionGate()
        self.max_tool_rounds = max_tool_rounds
        self.stateful = stateful
        self.history: list[Message] = []
        self._tools_by_name = {t.name: t for t in self.tools}

    def reset(self) -> None:
        self.history.clear()

    def run(
        self,
        task: str,
        *,
        on_delta: Callable[[str], None] | None = None,
        on_activity: Callable[[str], None] | None = None,
    ) -> str:
        if self.client is None:
            raise RuntimeError(
                f"agent '{self.name}' has no client; pass it to an Assistant or set agent.client"
            )
        working: list[Message] = []
        if self.system_prompt:
            working.append(system_message(self.system_prompt))
        if self.stateful:
            working.extend(self.history)
        working.append(user_message(task))

        answer = self._loop(working, on_delta=on_delta, on_activity=on_activity)

        if self.stateful:
            self.history.append(user_message(task))
            self.history.append(assistant_message(answer))
        return answer

    def _loop(
        self,
        messages: list[Message],
        *,
        on_delta: Callable[[str], None] | None,
        on_activity: Callable[[str], None] | None,
    ) -> str:
        tool_schemas = [t.schema() for t in self.tools] or None
        for _ in range(self.max_tool_rounds):
            prepared = self.context.prepare(messages) if self.context else messages
            stream = self.client.chat_stream(prepared, tools=tool_schemas)
            for delta in stream:
                if on_delta:
                    on_delta(delta)
            result = stream.result
            if not result.tool_calls:
                return result.content
            messages.append(assistant_tool_call_message(result.content, result.tool_calls))
            for call in result.tool_calls:
                if on_activity:
                    on_activity(_describe_call(call))
                output = self._execute(call)
                messages.append(tool_message(call.id, output))

        messages.append(
            user_message("You have used all your tool rounds. Give your final answer now, without calling any tools.")
        )
        stream = self.client.chat_stream(messages, tools=None)
        for delta in stream:
            if on_delta:
                on_delta(delta)
        return stream.result.content

    def _execute(self, call: ToolCall) -> str:
        tool = self._tools_by_name.get(call.name)
        if tool is None:
            available = ", ".join(self._tools_by_name) or "(none)"
            return f"Error: unknown tool '{call.name}'. Available tools: {available}"
        try:
            arguments: Any = json.loads(call.raw_arguments) if call.raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return f"Error: tool arguments are not valid JSON: {exc}"
        if not isinstance(arguments, dict):
            return "Error: tool arguments must be a JSON object"
        if not self.permission_gate.check(tool, arguments):
            return "Permission denied by the user. Do not retry the same call; ask the user how to proceed instead."
        try:
            output = tool.run(**arguments)
        except TypeError as exc:
            return f"Error: invalid arguments for tool '{call.name}': {exc}"
        except Exception as exc:
            return f"Error while running tool '{call.name}': {exc}"
        return _truncate(output, MAX_TOOL_OUTPUT_CHARS)
