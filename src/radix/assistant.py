from __future__ import annotations

from .agent import DEFAULT_MAX_DELEGATION_DEPTH, Agent
from .client import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL, Client
from .context import ContextManager
from .coordinator import Coordinator
from .events import Events
from .permissions import CliPermissionGate, PermissionGate
from .tool import Tool


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
        max_context_tokens: int = 8192,
        reserve_output_tokens: int = 2048,
        max_delegation_depth: int = DEFAULT_MAX_DELEGATION_DEPTH,
        permission_gate: PermissionGate | None = None,
        client: Client | None = None,
    ) -> None:
        self.client = client or Client(model=model, base_url=base_url, api_key=api_key)
        self.permission_gate = permission_gate or CliPermissionGate()
        self.max_delegation_depth = max_delegation_depth
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
            max_delegation_depth=max_delegation_depth,
        )

    def _bind(self, agent: Agent) -> None:
        """Recursively bind a client, context and gate to an agent and its
        whole sub-agent tree (cycles are traversed once thanks to a visited
        set)."""
        visited: set[int] = set()
        stack = [agent]
        while stack:
            current = stack.pop()
            if id(current) in visited:
                continue
            visited.add(id(current))
            if current.client is None:
                current.client = self.client
            if current.context is None:
                current.context = self.context
            current.permission_gate = self.permission_gate
            current.max_delegation_depth = self.max_delegation_depth
            stack.extend(current.subagents)

    def chat(self, text: str, *, events: Events | None = None) -> str:
        return self.coordinator.run(text, events=events)

    def reset(self) -> None:
        self.coordinator.reset()

    def run(self) -> None:
        """Start the interactive CLI REPL."""
        from .repl import Repl

        Repl(self).run()
