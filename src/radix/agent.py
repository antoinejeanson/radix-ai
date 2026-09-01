from __future__ import annotations

import json
import time
from collections.abc import Callable
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

# Agent: a system prompt plus a set of tools; runs the streaming tool-call
# loop, compacts context, and enforces permission gates on every call.
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 16000
DEFAULT_MAX_TOOL_ROUNDS = 8


def _truncate(text: str, limit: int) -> str:
    """Cut `text` to `limit` characters, appending a marker when cut.

    Args:
        text: The tool output to truncate.
        limit: Maximum number of characters to keep.

    Returns:
        The output, possibly shortened with a trailing marker.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [output truncated]"


def _describe_call(call: ToolCall) -> str:
    """One-line human-readable summary of a tool call, for events.

    Args:
        call: The tool call to summarize.

    Returns:
        Something like "edit_file path=foo.py old_string=..." (truncated).
    """
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
        tool_gates: dict[str, PermissionGate] | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
        stateful: bool = False,
        pre_tool_hook: Callable[[Tool, dict[str, Any]], None] | None = None,
    ) -> None:
        """Create an agent.

        Args:
            name: Identifier shown in events and used to build the
                coordinator's `ask_<name>` delegation tool.
            description: What the agent is good at and when to delegate to
                it (e.g. "edits code and runs tests"). Becomes part of the
                `ask_<name>` tool description shown to the coordinator.
            system_prompt: Instructions defining the agent's role and
                behavior. An empty string sends no system prompt.
            tools: Tools the agent may call. Every call must pass the
                permission gate first.
            client: Model client used for completions. When the agent is
                bound to an Assistant, the assistant's client is used when
                this is None.
            context: ContextManager that keeps the conversation within a
                token budget via compaction. None means no compaction: the
                full history is always sent.
            permission_gate: Decides whether a tool call may run. Defaults
                to a CliPermissionGate that prompts on the terminal. When
                bound to an Assistant, the assistant's gate is inherited
                unless an explicit gate is passed here.
            tool_gates: Per-tool overrides of `permission_gate`, keyed by
                tool name. An Assistant validates the names and reports
                unknown ones.
            max_tool_rounds: At most this many model/tool rounds per run;
                afterwards the agent is forced to answer without tools.
            max_tool_output_chars: Longest tool output kept in the
                conversation; anything longer is truncated with a marker.
            stateful: When True, `run()` appends the task and its answer to
                `history` so the next run continues the conversation. The
                coordinator uses True; sub-agents default to False.
            pre_tool_hook: Optional callback invoked right before each tool
                call with the Tool and its parsed arguments. The Assistant
                uses it to snapshot files for undo.
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = list(tools or [])
        self.client = client
        self.context = context
        self.permission_gate = permission_gate or CliPermissionGate()
        self.tool_gates = tool_gates or {}
        self._permission_gate_explicit = permission_gate is not None
        self._tool_gates_explicit = tool_gates is not None
        self.max_tool_rounds = max_tool_rounds
        self.max_tool_output_chars = max_tool_output_chars
        self.stateful = stateful
        self.pre_tool_hook = pre_tool_hook
        self.history: list[Message] = []
        self._tools_by_name = {t.name: t for t in self.tools}
        self._events: Events | None = None

    def reset(self) -> None:
        """Forget the saved conversation history (only relevant for
        stateful agents; has no effect on the tool set or gates)."""
        self.history.clear()

    def run(self, task: str, *, events: Events | None = None) -> str:
        """Run one task and return the final answer.

        The task is sent as a user message. With `stateful=True`, the task
        and the answer are appended to `history` for the next run.

        Args:
            task: The instruction to execute, as plain text.
            events: Optional observer callbacks for streaming live progress.

        Returns:
            The agent's final answer after all tool rounds.

        Raises:
            RuntimeError: if the agent has no client (it was neither passed
                here nor bound by an Assistant).
        """
        if self.client is None:
            raise RuntimeError(
                f"agent '{self.name}' has no client; pass it to an Assistant "
                "or set agent.client"
            )
        working: list[Message] = []
        if self.system_prompt:
            working.append(system_message(self.system_prompt))
        if self.stateful:
            working.extend(self.history)
        working.append(user_message(task))

        self._events = events
        try:
            answer = self._loop(working, events)
        finally:
            self._events = None

        if self.stateful:
            self.history.append(user_message(task))
            self.history.append(assistant_message(answer))
        return answer

    def _chat_round(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        events: Events | None,
    ) -> ChatResult:
        """Run one streaming completion round.

        Args:
            messages: Messages to send (already compacted if applicable).
            tools: OpenAI tool schemas offered to the model, or None.
            events: Observer callbacks notified as the stream produces
                deltas and when the round ends.

        Returns:
            The completed ChatResult with content, tool calls and usage.
        """
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
            events.on_stop(
                self.name, time.monotonic() - started, produced_text, result.usage
            )
        return result

    def _loop(self, messages: list[Message], events: Events | None) -> str:
        """Drive the tool-call loop until the model answers without tools.

        Compacts via `context` before every round and executes each tool
        call, appending call and output messages to `messages`.

        Args:
            messages: Working conversation; starts with the system prompt
                (if any) and ends with the user task. Grows in place with
                tool calls and their outputs.
            events: Observer callbacks for live progress.

        Returns:
            The agent's final answer.
        """
        tool_schemas = [t.schema() for t in self.tools] or None
        for _ in range(self.max_tool_rounds):
            prepared = self.context.prepare(messages) if self.context else messages
            result = self._chat_round(prepared, tool_schemas, events)
            if not result.tool_calls:
                return result.content
            messages.append(
                assistant_tool_call_message(result.content, result.tool_calls)
            )
            for call in result.tool_calls:
                if events and events.on_activity:
                    events.on_activity(self.name, _describe_call(call))
                output = self._execute(call)
                if events and events.on_tool_output:
                    events.on_tool_output(self.name, call.name, output)
                messages.append(tool_message(call.id, output))

        messages.append(
            user_message(
                "You have used all your tool rounds. Give your final answer now, "
                "without calling any tools."
            )
        )
        return self._chat_round(messages, None, events).content

    def _execute(self, call: ToolCall) -> str:
        """Run one tool call, guarding permission and errors.

        Args:
            call: The tool call requested by the model.

        Returns:
            The tool output as a string, or an error message when the tool
            is unknown, the arguments are invalid, permission is denied,
            or the tool itself raises.
        """
        tool = self._tools_by_name.get(call.name)
        if tool is None:
            available = ", ".join(self._tools_by_name) or "(none)"
            return f"Error: unknown tool '{call.name}'. Available tools: {available}"
        try:
            arguments: Any = (
                json.loads(call.raw_arguments) if call.raw_arguments.strip() else {}
            )
        except json.JSONDecodeError as exc:
            return f"Error: tool arguments are not valid JSON: {exc}"
        if not isinstance(arguments, dict):
            return "Error: tool arguments must be a JSON object"
        gate = self.tool_gates.get(tool.name, self.permission_gate)
        if not gate.check(tool, arguments):
            return (
                "Permission denied by the user. Do not retry the same call; "
                "ask the user how to proceed instead."
            )
        if self.pre_tool_hook is not None:
            self.pre_tool_hook(tool, arguments)
        try:
            output = tool.run(**arguments)
        except TypeError as exc:
            return f"Error: invalid arguments for tool '{call.name}': {exc}"
        except Exception as exc:
            return f"Error while running tool '{call.name}': {exc}"
        return _truncate(output, self.max_tool_output_chars)
