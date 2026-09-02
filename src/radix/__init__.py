"""Radix: modular AI agent for small, memory-limited LLMs."""

from .agent import Agent
from .assistant import Assistant
from .client import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL, Client
from .context import (
    DEFAULT_KEEP_RECENT,
    DEFAULT_KEEP_RECENT_TURNS,
    FALLBACK_SUMMARY,
    SUMMARY_PROMPT,
)
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

# Public package surface: re-exports the classes, tools and constants that
# users import from `radix` (the REPL stays lazy-loaded to keep imports light).
__version__ = "0.1.0"


__all__ = [
    "DEFAULT_API_KEY",
    "DEFAULT_BASE_URL",
    "DEFAULT_KEEP_RECENT",
    "DEFAULT_KEEP_RECENT_TURNS",
    "DEFAULT_MODEL",
    "FALLBACK_SUMMARY",
    "SAFETY_SYSTEM_PROMPT",
    "SUMMARY_PROMPT",
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
