"""Radix: modular AI agent for small, memory-limited LLMs."""

from .agent import DEFAULT_MAX_TOOL_OUTPUT_CHARS, Agent
from .assistant import Assistant
from .client import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL, Client
from .context import FALLBACK_SUMMARY, SUMMARY_PROMPT, TRANSCRIPT_CHAR_LIMIT
from .coordinator import Coordinator
from .events import Events
from .messages import ChatResult, Message, ToolCall, Usage
from .permissions import (
    AutoApproveGate,
    CliPermissionGate,
    DenyGate,
    LlmAdvisoryGate,
    LlmAutoSafetyGate,
    PermissionGate,
)
from .safety import SAFETY_SYSTEM_PROMPT, LlmSafetyChecker, SafetyVerdict
from .tool import Tool, tool
from .undo import UndoLog, UndoResult

__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy so `import radix` does not pull in the REPL's console dependencies.
    if name == "MAX_OUTPUT_LINES":
        from .repl import MAX_OUTPUT_LINES

        return MAX_OUTPUT_LINES
    raise AttributeError(f"module 'radix' has no attribute {name!r}")


__all__ = [
    "DEFAULT_API_KEY",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_TOOL_OUTPUT_CHARS",
    "DEFAULT_MODEL",
    "FALLBACK_SUMMARY",
    "MAX_OUTPUT_LINES",
    "SAFETY_SYSTEM_PROMPT",
    "SUMMARY_PROMPT",
    "TRANSCRIPT_CHAR_LIMIT",
    "Agent",
    "Assistant",
    "AutoApproveGate",
    "ChatResult",
    "CliPermissionGate",
    "Client",
    "Coordinator",
    "DenyGate",
    "Events",
    "LlmAdvisoryGate",
    "LlmAutoSafetyGate",
    "LlmSafetyChecker",
    "Message",
    "PermissionGate",
    "SafetyVerdict",
    "Tool",
    "ToolCall",
    "UndoLog",
    "UndoResult",
    "Usage",
    "tool",
]
