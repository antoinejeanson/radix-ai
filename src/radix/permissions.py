from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .safety import LlmSafetyChecker, SafetyVerdict
from .tool import Tool

if TYPE_CHECKING:
    from .client import Client


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
        self._print(f"! Permission request: '{tool.name}' wants to run with arguments:")
        self._print("  " + json.dumps(arguments, ensure_ascii=False, indent=2).replace("\n", "\n  "))
        answer = self._input("Allow this call? [y/N] ")
        return answer.strip().lower() in ("y", "yes")


class AutoApproveGate:
    """Approves everything. Use only for tests or fully trusted setups.

    After `ask_permission` was removed, this is the way to opt out of
    prompting: `Assistant(permission_gate=AutoApproveGate(), ...)`. Composes
    with `tool_gates` when only some tools should still be checked.
    """

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        return True


class DenyGate:
    """Denies every call."""

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        return False


def _print_verdict(printer: Callable[[str], None], verdict: SafetyVerdict) -> None:
    label = "SAFE" if verdict.safe else "DANGEROUS"
    line = f"• safety: {label}"
    if verdict.reason:
        line += f" — {verdict.reason}"
    printer(line)


class LlmAutoSafetyGate:
    """LLM-judged gate: auto-approves safe calls, denies dangerous ones.

    No user prompt: the model's verdict is the decision. Fails closed —
    a failed or unparseable check is denied, never auto-approved.
    """

    def __init__(
        self,
        client: Client,
        *,
        checker: LlmSafetyChecker | None = None,
        printer: Callable[[str], None] = print,
    ) -> None:
        self._checker = checker or LlmSafetyChecker(client)
        self._print = printer

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        verdict = self._checker.check(tool, arguments)
        _print_verdict(self._print, verdict)
        return verdict.safe


class LlmAdvisoryGate:
    """Shows the LLM safety verdict, then defers to a base gate (user decides).

    Defaults to `CliPermissionGate`, so the verdict is extra context and
    the human keeps the final word. Pass any other gate as `base_gate`.
    """

    def __init__(
        self,
        client: Client,
        *,
        base_gate: PermissionGate | None = None,
        checker: LlmSafetyChecker | None = None,
        printer: Callable[[str], None] = print,
    ) -> None:
        self._checker = checker or LlmSafetyChecker(client)
        self._base_gate = base_gate or CliPermissionGate(printer=printer)
        self._print = printer

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        verdict = self._checker.check(tool, arguments)
        _print_verdict(self._print, verdict)
        return self._base_gate.check(tool, arguments)
