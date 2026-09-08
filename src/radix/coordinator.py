from __future__ import annotations

import re

from .agent import Agent
from .tool import Tool

# Coordinator: the stateful agent the user talks to; each sub-agent is
# exposed to it as an ask_<name> tool so delegation is an ordinary tool call.
DEFAULT_COORDINATOR_PROMPT = (
    "You are Radix, a concise AI assistant.\n\n"
    "Answer simple questions directly. Delegate specialized or multi-step work "
    "to your sub-agents with the ask_<name> tools, then integrate their results "
    "into a clear answer.\n\n"
    "When delegating, write a fully self-contained task description: sub-agents "
    "cannot see this conversation and have no memory of earlier tasks.\n\n"
    "If you have no suitable tool or sub-agent for a request, say so honestly."
)


def safe_name(name: str) -> str:
    """Make a name safe for a tool name (alphanumerics and underscore).

    Args:
        name: The sub-agent name.

    Returns:
        A tool-name-safe string, or "agent" when nothing remains.
    """
    return re.sub(r"\W+", "_", name).strip("_") or "agent"


def delegation_tool(agent: Agent, parent: Agent | None = None) -> Tool:
    """Build the `ask_<name>` delegation tool for a sub-agent.

    Args:
        agent: The sub-agent to delegate to. Its description becomes part of
            the tool description shown to the model.
        parent: The agent that will call this tool (the coordinator). Its
            event observers are forwarded to the sub-agent so hosts see the
            delegation live; None disables forwarding.

    Returns:
        A Tool whose `run(task)` executes `agent.run(task)` and returns the
        sub-agent's final answer.
    """

    def ask(task: str) -> str:
        events = parent._events if parent is not None else None
        return agent.run(task, events=events)

    return Tool(
        name=f"ask_{safe_name(agent.name)}",
        description=(
            f"Delegate a task to the '{agent.name}' sub-agent. {agent.description} "
            "The sub-agent cannot see this conversation, so the task must be "
            "fully self-contained."
        ).strip(),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Self-contained description of the task to delegate."
                    ),
                }
            },
            "required": ["task"],
        },
        fn=ask,
    )


class Coordinator(Agent):
    """The agent the user talks to.

    Every sub-agent is exposed as an `ask_<name>` tool, so delegation happens
    through ordinary tool calls. Sub-agents run with isolated context and only
    their final answer enters the coordinator's conversation.
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
        """Create the coordinator: the stateful agent the user talks to.

        Each sub-agent becomes an `ask_<name>` tool, so delegation happens
        through ordinary tool calls. Sub-agent names must be unique or a
        ValueError is raised.

        Args:
            agents: Sub-agents exposed as delegation tools to the
                coordinator.
            name: Coordinator name, used in events and tool names.
            description: What the coordinator is; appears in its own
                `ask_<name>` tool when nested under another coordinator.
            system_prompt: Coordinator instructions; defaults to
                DEFAULT_COORDINATOR_PROMPT when None.
            tools: Additional tools beyond the delegation tools.
            **kwargs: Remaining Agent parameters (client, context,
                permission_gate, tool_gates, max_tool_rounds, ...), passed
                through to Agent.
        """
        self.agents = list(agents or [])
        names = [a.name for a in self.agents]
        if len(names) != len(set(names)):
            raise ValueError("sub-agent names must be unique")
        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt or DEFAULT_COORDINATOR_PROMPT,
            tools=[delegation_tool(a, parent=self) for a in self.agents]
            + list(tools or []),
            stateful=True,
            **kwargs,
        )
