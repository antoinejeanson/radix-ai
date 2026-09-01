from __future__ import annotations

from typing import Any

from .agent import Agent
from .client import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL, Client
from .context import ContextManager
from .coordinator import Coordinator
from .events import Events
from .permissions import CliPermissionGate, PermissionGate
from .tool import Tool
from .undo import UndoLog, UndoResult


class Assistant:
    """Your Radix assistant: assistant as code.

    Everything is defined here, in Python: the model endpoint, the
    coordinator, its direct tools and its sub-agents. No config files,
    no hidden state.

    Defaults target a local llama.cpp server (`llama-server`); any
    OpenAI-compatible endpoint works via `base_url`.
    """

    _SNAPSHOT_TOOLS = frozenset({"edit_file", "write_file"})

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        agents: list[Agent] | None = None,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        max_context_tokens: int = 8192,
        reserve_output_tokens: int = 2048,
        permission_gate: PermissionGate | None = None,
        client: Client | None = None,
    ) -> None:
        self.client = client or Client(model=model, base_url=base_url, api_key=api_key)
        self.permission_gate = permission_gate or CliPermissionGate()
        self.undo_log = UndoLog()
        self.context = ContextManager(
            self.client,
            max_context_tokens=max_context_tokens,
            reserve_output_tokens=reserve_output_tokens,
        )
        agents = list(agents or [])
        for agent in agents:
            self._bind(agent)
        self.coordinator = Coordinator(
            agents=agents,
            tools=list(tools or []),
            system_prompt=system_prompt,
            client=self.client,
            context=self.context,
            permission_gate=self.permission_gate,
        )
        self.coordinator.pre_tool_hook = self._snapshot_tool_call

    def _bind(self, agent: Agent) -> None:
        if agent.client is None:
            agent.client = self.client
        if agent.context is None:
            agent.context = self.context
        agent.permission_gate = self.permission_gate
        agent.pre_tool_hook = self._snapshot_tool_call

    def _snapshot_tool_call(self, tool: Tool, arguments: dict[str, Any]) -> None:
        if tool.name in self._SNAPSHOT_TOOLS and "path" in arguments:
            self.undo_log.snapshot(str(arguments["path"]))

    def chat(self, text: str, *, events: Events | None = None) -> str:
        self.undo_log.begin_turn(len(self.coordinator.history))
        return self.coordinator.run(text, events=events)

    def undo(self, turns: int = 1) -> UndoResult:
        """Revert the last `turns` turn(s): restore changed files and rewind
        the conversation. Returns what was restored."""
        result = self.undo_log.undo(turns)
        if result.history_depth is not None:
            del self.coordinator.history[result.history_depth:]
        return result

    def reset(self) -> None:
        self.coordinator.reset()
        self.undo_log.clear()

    def run(self) -> None:
        """Start the interactive CLI REPL."""
        from .repl import Repl

        Repl(self).run()
