from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .safety import LlmSafetyChecker, SafetyVerdict
from .tool import Tool

if TYPE_CHECKING:
    from .client import Client


# Permission gates: decide whether a tool call may run — CLI prompt,
# auto-approve, deny, or LLM-judged (with optional human confirmation).
@runtime_checkable
class PermissionGate(Protocol):
    """Decides whether a tool call may run.

    Gates receive the tool and its parsed arguments, which leaves room for
    finer-grained policies later (per-tool allowlists, command prefixes, ...)
    without changing the interface.
    """

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        """Decide whether a tool call may run.

        Args:
            tool: The tool being called.
            arguments: Its parsed arguments.

        Returns:
            True to allow the call, False to deny it.
        """
        ...


class CliPermissionGate:
    """Default gate: prompts the user on every tool call requiring permission."""

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        printer: Callable[[str], None] = print,
    ) -> None:
        """Create a terminal-prompting gate.

        Args:
            input_fn: Reads the user's answer (default `builtins.input`).
            printer: Prints the permission request (default `builtins.print`).
        """
        self._input = input_fn
        self._print = printer

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        """Ask the user to allow or deny the call.

        Args:
            tool: The tool being called.
            arguments: Its parsed arguments (displayed in the prompt).

        Returns:
            True on an explicit "y"/"yes" answer, False otherwise.
        """
        self._print(f"! Permission request: '{tool.name}' wants to run with arguments:")
        self._print(
            "  "
            + json.dumps(arguments, ensure_ascii=False, indent=2).replace("\n", "\n  ")
        )
        answer = self._input("Allow this call? [y/N] ")
        return answer.strip().lower() in ("y", "yes")


class AutoApproveGate:
    """Approves everything. Use only for tests or fully trusted setups.

    After `ask_permission` was removed, this is the way to opt out of
    prompting: `Assistant(permission_gate=AutoApproveGate(), ...)`. Composes
    with `tool_gates` when only some tools should still be checked.
    """

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        """Approve every call (used by tests and trusted setups).

        Args:
            tool: The tool being called (ignored).
            arguments: Its parsed arguments (ignored).

        Returns:
            Always True.
        """
        return True


class DenyGate:
    """Denies every call."""

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        """Deny every call.

        Args:
            tool: The tool being called (ignored).
            arguments: Its parsed arguments (ignored).

        Returns:
            Always False.
        """
        return False


def _print_verdict(printer: Callable[[str], None], verdict: SafetyVerdict) -> None:
    """Print one human-readable verdict line.

    Args:
        printer: The callable used to print.
        verdict: The verdict to print.
    """
    label = "SAFE" if verdict.safe else "DANGEROUS"
    line = f"• safety: {label}"
    if verdict.reason:
        line += f" — {verdict.reason}"
    printer(line)


class LlmAutoSafetyGate:
    """LLM-judged gate: auto-approves safe calls, denies dangerous ones.

    No user prompt by default: the model's verdict is the decision. Fails
    closed — a failed or unparseable check is denied, never auto-approved.
    With `confirm_unsafe=True`, a non-safe verdict (including a failed
    check) instead asks the user for a final [y/N] before denying.
    """

    def __init__(
        self,
        client: Client,
        *,
        checker: LlmSafetyChecker | None = None,
        confirm_unsafe: bool = False,
        input_fn: Callable[[str], str] = input,
        printer: Callable[[str], None] = print,
    ) -> None:
        """Create an LLM-judged gate.

        Args:
            client: Client used to ask the safety model for verdicts.
            checker: Reusable LlmSafetyChecker; one is built from `client`
                when None.
            confirm_unsafe: When True, a non-safe verdict (including a
                failed check) asks the user for a final [y/N] before
                denying. When False, non-safe calls are denied silently.
            input_fn: Reads the confirmation answer when `confirm_unsafe`
                is True (default `builtins.input`).
            printer: Prints the verdict line (default `builtins.print`).
        """
        self._checker = checker or LlmSafetyChecker(client)
        self._confirm_unsafe = confirm_unsafe
        self._input = input_fn
        self._print = printer

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        """Judge the call with the LLM, denying non-safe verdicts.

        Args:
            tool: The tool being called.
            arguments: Its parsed arguments, sent to the safety model.

        Returns:
            True when the verdict is safe (or the user overrides with
            `confirm_unsafe`), False otherwise.
        """
        verdict = self._checker.check(tool, arguments)
        _print_verdict(self._print, verdict)
        if verdict.safe:
            return True
        if not self._confirm_unsafe:
            return False
        answer = self._input("The model flagged this call. Run anyway? [y/N] ")
        return answer.strip().lower() in ("y", "yes")


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
        """Create an advisory gate.

        Args:
            client: Client used to ask the safety model for verdicts.
            base_gate: The gate that makes the final decision; defaults to
                a CliPermissionGate (the human keeps the final word).
            checker: Reusable LlmSafetyChecker; one is built from `client`
                when None.
            printer: Prints the verdict line (default `builtins.print`).
        """
        self._checker = checker or LlmSafetyChecker(client)
        self._base_gate = base_gate or CliPermissionGate(printer=printer)
        self._print = printer

    def check(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        """Show the LLM verdict, then let the base gate decide.

        Args:
            tool: The tool being called.
            arguments: Its parsed arguments.

        Returns:
            The base gate's decision.
        """
        verdict = self._checker.check(tool, arguments)
        _print_verdict(self._print, verdict)
        return self._base_gate.check(tool, arguments)
