"""Radix: modular AI agent for small, memory-limited LLMs."""

from .agent import Agent
from .assistant import Assistant
from .client import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL, Client
from .coordinator import Coordinator
from .events import Events
from .messages import ChatResult, Message, ToolCall, Usage
from .permissions import AutoApproveGate, CliPermissionGate, DenyGate, PermissionGate
from .tool import Tool, tool

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_API_KEY",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "Agent",
    "Assistant",
    "AutoApproveGate",
    "ChatResult",
    "CliPermissionGate",
    "Client",
    "Coordinator",
    "DenyGate",
    "Events",
    "Message",
    "PermissionGate",
    "Tool",
    "ToolCall",
    "Usage",
    "tool",
]
