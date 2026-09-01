from __future__ import annotations

from typing import Any

from .agent import DEFAULT_MAX_TOOL_OUTPUT_CHARS, DEFAULT_MAX_TOOL_ROUNDS, Agent
from .client import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL, Client
from .context import (
    DEFAULT_KEEP_RECENT,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RESERVE_OUTPUT_TOKENS,
    FALLBACK_SUMMARY,
    SUMMARY_PROMPT,
    TRANSCRIPT_CHAR_LIMIT,
    ContextManager,
)
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
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        reserve_output_tokens: int = DEFAULT_RESERVE_OUTPUT_TOKENS,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        summary_prompt: str = SUMMARY_PROMPT,
        fallback_summary: str = FALLBACK_SUMMARY,
        transcript_char_limit: int = TRANSCRIPT_CHAR_LIMIT,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
        permission_gate: PermissionGate | None = None,
        tool_gates: dict[str, PermissionGate] | None = None,
        client: Client | None = None,
    ) -> None:
        self.client = client or Client(model=model, base_url=base_url, api_key=api_key)
        self.permission_gate = permission_gate or CliPermissionGate()
        self.tool_gates = tool_gates or {}
        self.undo_log = UndoLog()
        self.context = ContextManager(
            self.client,
            max_context_tokens=max_context_tokens,
            reserve_output_tokens=reserve_output_tokens,
            keep_recent=keep_recent,
            summary_prompt=summary_prompt,
            fallback_summary=fallback_summary,
            transcript_char_limit=transcript_char_limit,
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
            tool_gates=self.tool_gates,
            max_tool_rounds=max_tool_rounds,
            max_tool_output_chars=max_tool_output_chars,
        )
        self._validate_tool_gates()
        self.coordinator.pre_tool_hook = self._snapshot_tool_call

    def _bind(self, agent: Agent) -> None:
        if agent.client is None:
            agent.client = self.client
        if agent.context is None:
            agent.context = self.context
        if not agent._permission_gate_explicit:
            agent.permission_gate = self.permission_gate
        if not agent._tool_gates_explicit:
            agent.tool_gates = self.tool_gates
        agent.pre_tool_hook = self._snapshot_tool_call

    def _validate_tool_gates(self) -> None:
        known = {t.name for t in self.coordinator.tools}
        for agent in self.coordinator.agents:
            known |= {t.name for t in agent.tools}
        unknown = sorted(set(self.tool_gates) - known)
        if unknown:
            raise ValueError(
                "tool_gates reference unknown tool(s): " + ", ".join(unknown) + ". "
                "Known tools: " + ", ".join(sorted(known)) + "."
            )

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
