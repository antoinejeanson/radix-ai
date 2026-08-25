from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from .tool import Tool


@runtime_checkable
class PermissionGate(Protocol):
    """Decides whether a tool call may run.

    Gates receive the tool and its parsed arguments, which leaves room for
    finer-grained policies later (per-tool allowlists, command prefixes, ...)
    without changing the interface.
    """

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool: ...


class CliPermissionGate:
    """Default gate: prompts the user on every tool call requiring permission."""

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        printer: Callable[[str], None] = print,
    ) -> None:
        self._input = input_fn
        self._print = printer

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        if not tool.ask_permission:
            return True
        self._print(f"! Permission request: '{tool.name}' wants to run with arguments:")
        self._print("  " + json.dumps(arguments, ensure_ascii=False, indent=2).replace("\n", "\n  "))
        answer = self._input("Allow this call? [y/N] ")
        return answer.strip().lower() in ("y", "yes")


class AutoApproveGate:
    """Approves everything. Use only for tests or fully trusted setups."""

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        return True


class DenyGate:
    """Denies every call."""

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        return False
