from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .messages import Usage


@dataclass
class Events:
    """Observers for agent execution, used by the REPL to show what happens.

    on_start(agent_name): a model round begins (the agent is "thinking").
    on_delta(agent_name, text): a chunk of streamed text arrived.
    on_activity(agent_name, description): a tool call is about to run.
    on_tool_output(agent_name, tool_name, output): a tool finished running.
    on_stop(agent_name, elapsed_seconds, produced_text, usage): a model round
        ended; `usage` is the token accounting for the round, when the server
        reports one.

    Events from sub-agents carry the sub-agent's name, so hosts can tell
    coordinator and sub-agent output apart.
    """

    on_start: Callable[[str], None] | None = None
    on_delta: Callable[[str, str], None] | None = None
    on_activity: Callable[[str, str], None] | None = None
    on_tool_output: Callable[[str, str, str], None] | None = None
    on_stop: Callable[[str, float, bool, "Usage | None"], None] | None = None
