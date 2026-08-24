from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Events:
    """Observers for agent execution, used by the REPL to show what happens.

    on_start(agent_name): a model round begins (the agent is "thinking").
    on_delta(agent_name, text): a chunk of streamed text arrived.
    on_activity(agent_name, description): a tool call is about to run.
    on_stop(agent_name, elapsed_seconds, produced_text): a model round ended.

    Events from sub-agents carry the sub-agent's name, so hosts can tell
    coordinator and sub-agent output apart.
    """

    on_start: Callable[[str], None] | None = None
    on_delta: Callable[[str, str], None] | None = None
    on_activity: Callable[[str, str], None] | None = None
    on_stop: Callable[[str, float, bool], None] | None = None
