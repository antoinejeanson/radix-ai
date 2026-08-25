from __future__ import annotations

from .agent import Agent
from .tool import Tool

DEFAULT_COORDINATOR_PROMPT = """You are Radix, a concise AI assistant.

Answer simple questions directly. Delegate specialized or multi-step work to your sub-agents with the ask_<name> tools, then integrate their results into a clear answer.

When delegating, write a fully self-contained task description: sub-agents cannot see this conversation and have no memory of earlier tasks.

If you have no suitable tool or sub-agent for a request, say so honestly."""


class Coordinator(Agent):
    """The root agent the user talks to (a stateful Agent).

    Every sub-agent is exposed as an `ask_<name>` tool, so delegation happens
    through ordinary tool calls. Sub-agents run with isolated context and only
    their final answer enters the coordinator's conversation. Sub-agents may
    carry their own sub-agents, down to the assistant-wide
    `max_delegation_depth` (single source: `agent.DEFAULT_MAX_DELEGATION_DEPTH`).
    """

    def __init__(
        self,
        *,
        agents: list[Agent] | None = None,
        name: str = "coordinator",
        description: str = "Coordinates the conversation and delegates to sub-agents.",
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        subagents = list(agents or [])
        self.agents = subagents
        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt or DEFAULT_COORDINATOR_PROMPT,
            subagents=subagents,
            tools=list(tools or []),
            stateful=True,
            **kwargs,
        )
