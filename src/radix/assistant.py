from __future__ import annotations

from typing import Any

from .agent import DEFAULT_MAX_TOOL_ROUNDS, Agent
from .client import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL, Client
from .context import (
    DEFAULT_KEEP_RECENT,
    DEFAULT_KEEP_RECENT_TURNS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RESERVE_OUTPUT_TOKENS,
    FALLBACK_SUMMARY,
    SUMMARY_PROMPT,
    ContextManager,
)
from .coordinator import Coordinator
from .events import Events
from .permissions import CliPermissionGate, PermissionGate
from .tool import Tool
from .undo import UndoLog, UndoResult


# Assistant: the top-level "assistant as code" object; wires the model
# client, context manager, undo log, coordinator and its sub-agents.
class Assistant:
    """Your Radix assistant: assistant as code.

    Everything is defined here, in Python: the model endpoint, the
    coordinator, its direct tools and its sub-agents. No config files,
    no hidden state.

    Defaults target a local llama.cpp server (`llama-server`); any
    OpenAI-compatible endpoint works via `base_url`.
    """

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
        keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS,
        summary_prompt: str = SUMMARY_PROMPT,
        fallback_summary: str = FALLBACK_SUMMARY,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        permission_gate: PermissionGate | None = None,
        tool_gates: dict[str, PermissionGate] | None = None,
        client: Client | None = None,
    ) -> None:
        """Create the assistant: model client, context manager, undo log,
        coordinator and its sub-agents.

        Args:
            model: Model id sent to the API; the default "radix" matches a
                llama.cpp server loaded with any model.
            base_url: OpenAI-compatible endpoint; the default is a local
                llama-server (`http://localhost:8080/v1`).
            api_key: Key for the endpoint. llama.cpp accepts anything;
                hosted APIs require a real one.
            agents: Sub-agents the coordinator can delegate to; each becomes
                an `ask_<name>` tool. Sub-agent names must be unique.
            tools: Tools the coordinator may call directly. Sub-agents only
                have access to their own tools.
            system_prompt: Coordinator system prompt; defaults to
                DEFAULT_COORDINATOR_PROMPT when None.
            max_context_tokens: Token budget for the conversation. Messages
                are re-attempted against this budget, and when the
                conversation exceeds budget - reserve_output_tokens the
                oldest messages are summarized.
            reserve_output_tokens: Tokens reserved for the model's answer;
                the conversation budget is
                max_context_tokens - reserve_output_tokens.
            keep_recent: Number of most recent messages kept verbatim when
                the conversation is compacted.
            keep_recent_turns: Number of most recent complete turns kept
                verbatim when compacted (a turn runs from a user message
                through the following assistant message, including any tool
                calls and results).
            summary_prompt: System prompt used when summarizing the old
                conversation during compaction.
            fallback_summary: Summary message used when summarization fails
                or returns nothing.
            max_tool_rounds: Maximum model/tool rounds for the coordinator
                per user message.
            permission_gate: Gate for the coordinator and for any agent that
                does not pass an explicit gate of its own. Defaults to a
                CliPermissionGate that prompts.
            tool_gates: Per-tool gate overrides, keyed by tool name, applied
                to the coordinator and bound agents (unless an agent passed
                its own explicit tool_gates). Unknown tool names raise a
                ValueError.
            client: Pre-built Client to share (e.g. one with a custom
                `openai_client`); a new Client is built from model, base_url
                and api_key when None.
        """
        self.client = client or Client(model=model, base_url=base_url, api_key=api_key)
        self.permission_gate = permission_gate or CliPermissionGate()
        self.tool_gates = tool_gates or {}
        self.undo_log = UndoLog()
        self.context = ContextManager(
            self.client,
            max_context_tokens=max_context_tokens,
            reserve_output_tokens=reserve_output_tokens,
            keep_recent=keep_recent,
            keep_recent_turns=keep_recent_turns,
            summary_prompt=summary_prompt,
            fallback_summary=fallback_summary,
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
        )
        self._validate_tool_gates()
        self.coordinator.pre_tool_hook = self._snapshot_tool_call

    def _bind(self, agent: Agent) -> None:
        """Wire a sub-agent to the assistant's shared resources.

        Fills in the client and context manager when the agent does not
        have its own, and inherits the permission gate unless the agent
        passed one explicitly. Also installs the undo snapshot hook.

        Args:
            agent: The sub-agent to bind. Modified in place.
        """
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
        """Check that every name in `self.tool_gates` is an actual tool.

        Raises:
            ValueError: listing the unknown gate names and all known tools.
        """
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
        """Pre-tool hook: snapshot files a tool is about to mutate.

        Whether a tool snapshots is the tool's own declaration (the
        `snapshot` flag, set via `@tool(snapshot=True)`), not a hardcoded
        name list, so custom file-mutating tools opt in by setting the flag.

        Args:
            tool: The tool about to run.
            arguments: Its parsed arguments; `path` names the file to
                snapshot when present.
        """
        if tool.snapshot and "path" in arguments:
            self.undo_log.snapshot(str(arguments["path"]))

    def chat(self, text: str, *, events: Events | None = None) -> str:
        """Send one user message to the coordinator and return its answer.

        The turn is recorded in the undo log so `undo()` can revert it.

        Args:
            text: The user's message.
            events: Optional observer callbacks for streaming live progress.

        Returns:
            The coordinator's final answer.
        """
        self.undo_log.begin_turn(len(self.coordinator.history))
        return self.coordinator.run(text, events=events)

    def undo(self, turns: int = 1) -> UndoResult:
        """Revert the last `turns` turn(s): restore changed files and rewind
        the conversation.

        Args:
            turns: Number of turns to revert. 1 undoes the last turn, 0 or a
                negative value is a no-op.

        Returns:
            UndoResult with the restored and failed file paths, and how far
            the conversation history should be rewound (None if nothing was
            undone).
        """
        result = self.undo_log.undo(turns)
        if result.history_depth is not None:
            del self.coordinator.history[result.history_depth :]
        return result

    def compact(self) -> bool:
        """Manually compact the coordinator's saved conversation now.

        Summarizes the oldest messages in the coordinator's history right
        away — even when still within the token budget — so subsequent
        turns send less context. Sub-agents are stateless and keep no
        history, so only the coordinator is affected. The compaction
        cannot be undone: summarized messages are gone from the history.

        Returns:
            True when the coordinator's history was rewritten to a
            smaller list.
        """
        return self.coordinator.compact()

    def reset(self) -> None:
        """Reset the assistant: forget the conversation and clear the undo
        log. The model client, tools and gates stay configured."""
        self.coordinator.reset()
        self.undo_log.clear()

    def run(self) -> None:
        """Start the interactive CLI REPL (blocks until the user exits)."""
        from .repl import Repl

        Repl(self).run()
