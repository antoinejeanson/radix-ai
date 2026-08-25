from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any

from .events import Events
from .messages import (
    ChatResult,
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
DEFAULT_MAX_DELEGATION_DEPTH = 2


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [output truncated]"


def _describe_call(call: ToolCall) -> str:
    args = " ".join(call.raw_arguments.split())
    if len(args) > 120:
        args = args[:117] + "..."
    return f"{call.name} {args}".strip()


def _safe_name(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_") or "agent"


def delegation_tool(agent: Agent, parent: Agent) -> Tool:
    """Build an `ask_<name>` tool that runs `agent` as a sub-agent of `parent`.

    The child runs at `parent`'s depth + 1, so depth limits propagate down the
    delegation chain. The parent's active events flow into the child, so a UI
    sees everything the child does.
    """

    def ask(task: str) -> str:
        events = parent._events
        return agent.run(task, events=events, depth=parent._depth + 1)

    return Tool(
        name=f"ask_{_safe_name(agent.name)}",
        description=(
            f"Delegate a task to the '{agent.name}' sub-agent. {agent.description} "
            "The sub-agent cannot see this conversation, so the task must be fully self-contained."
        ).strip(),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Self-contained description of the task to delegate.",
                }
            },
            "required": ["task"],
        },
        fn=ask,
    )


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
        subagents: list[Agent] | None = None,
        client: Client | None = None,
        context: ContextManager | None = None,
        permission_gate: PermissionGate | None = None,
max_tool_rounds: int = 8,
        max_delegation_depth: int | None = None,
        stateful: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = list(tools or [])
        self.subagents = list(subagents or [])
        names = [a.name for a in self.subagents]
        if len(names) != len(set(names)):
            raise ValueError("sub-agent names must be unique")
        self.client = client
        self.context = context
        self.permission_gate = permission_gate or CliPermissionGate()
        self.max_tool_rounds = max_tool_rounds
        self.max_delegation_depth = (
            max_delegation_depth
            if max_delegation_depth is not None
            else DEFAULT_MAX_DELEGATION_DEPTH
        )
        self.stateful = stateful
        self.history: list[Message] = []
        self._events: Events | None = None
        self._depth = 0
        self._run_tools: list[Tool] = []
        self._run_tools_by_name: dict[str, Tool] = {}

    def reset(self) -> None:
        self.history.clear()

    def delegation_tools(self, depth: int = 0) -> list[Tool]:
        """The `ask_<name>` tools this agent may call at the given depth.

        Agents at or past the maximum delegation depth get none, which bounds
        recursion even when sub-agent graphs contain cycles.
        """
        if depth + 1 > self.max_delegation_depth:
            return []
        return [delegation_tool(child, self) for child in self.subagents]

    def run(self, task: str, *, events: Events | None = None, depth: int = 0) -> str:
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

        self._events = events
        self._depth = depth
        self._run_tools = self.tools + self.delegation_tools(depth)
        self._run_tools_by_name = {t.name: t for t in self._run_tools}
        try:
            answer = self._loop(working, events)
        finally:
            self._events = None
            self._depth = 0
            self._run_tools = []
            self._run_tools_by_name = {}

        if self.stateful:
            self.history.append(user_message(task))
            self.history.append(assistant_message(answer))
        return answer

    def _chat_round(
        self, messages: list[Message], tools: list[dict[str, Any]] | None, events: Events | None
    ) -> ChatResult:
        if events and events.on_start:
            events.on_start(self.name)
        started = time.monotonic()
        produced_text = False
        stream = self.client.chat_stream(messages, tools=tools)
        for delta in stream:
            produced_text = True
            if events and events.on_delta:
                events.on_delta(self.name, delta)
        result = stream.result
        if events and events.on_stop:
            events.on_stop(self.name, time.monotonic() - started, produced_text, result.usage)
        return result

    def _loop(self, messages: list[Message], events: Events | None) -> str:
        tool_schemas = [t.schema() for t in self._run_tools] or None
        for _ in range(self.max_tool_rounds):
            prepared = self.context.prepare(messages) if self.context else messages
            result = self._chat_round(prepared, tool_schemas, events)
            if not result.tool_calls:
                return result.content
            messages.append(assistant_tool_call_message(result.content, result.tool_calls))
            for call in result.tool_calls:
                if events and events.on_activity:
                    events.on_activity(self.name, _describe_call(call))
                output = self._execute(call)
                if events and events.on_tool_output:
                    events.on_tool_output(self.name, call.name, output)
                messages.append(tool_message(call.id, output))

        messages.append(
            user_message("You have used all your tool rounds. Give your final answer now, without calling any tools.")
        )
        return self._chat_round(messages, None, events).content

    def _execute(self, call: ToolCall) -> str:
        tool = self._run_tools_by_name.get(call.name)
        if tool is None:
            available = ", ".join(self._run_tools_by_name) or "(none)"
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
