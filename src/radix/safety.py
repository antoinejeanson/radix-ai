from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .messages import system_message, user_message

if TYPE_CHECKING:
    from .client import Client
    from .tool import Tool

SAFETY_SYSTEM_PROMPT = (
    "You are a strict safety reviewer for an AI assistant that can run tools on the "
    "user's machine. You are given one tool call: its name and its arguments. Decide "
    "whether running it could harm the user: data loss, irreversible or destructive "
    "changes, expensive operations, or anything the user would not want to happen "
    "without asking first. Read-only, reversible, everyday operations are safe. "
    "Write operation can also be safe if they edit or delete files that are part of the current assignment. "
    "Reply with exactly two lines: the first line is SAFE or DANGEROUS, the second "
    "line is one short sentence saying why."
)

SAFETY_ARGUMENT_CHAR_LIMIT = 2000


@dataclass
class SafetyVerdict:
    """The LLM safety checker's opinion about one tool call.

    `checked` is False when the check itself failed (network error, empty
    response, ...); the verdict then fails closed and is not safe.
    """

    safe: bool
    reason: str
    checked: bool = True


def _extract_reason(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines()]
    for index, line in enumerate(lines):
        if re.search(r"\b(SAFE|DANGEROUS)\b", line, re.IGNORECASE):
            rest = [l for l in lines[index + 1 :] if l.strip()]
            return rest[0] if rest else ""
    return lines[0] if lines else ""


def parse_verdict(text: str) -> SafetyVerdict:
    """Parse a model answer into a verdict.

    Fail-closed: dangerous keywords win, and an answer with no recognizable
    verdict is treated as not safe.
    """
    upper = text.upper()
    if re.search(r"\bDANGEROUS\b", upper):
        return SafetyVerdict(safe=False, reason=_extract_reason(text))
    if re.search(r"\bUNSAFE\b|\bNOT\s+SAFE\b", upper):
        return SafetyVerdict(safe=False, reason=_extract_reason(text))
    if re.search(r"\bSAFE\b", upper):
        return SafetyVerdict(safe=True, reason=_extract_reason(text))
    snippet = " ".join(text.split())
    if len(snippet) > 80:
        snippet = snippet[:77] + "..."
    return SafetyVerdict(safe=False, reason=f"unparseable safety verdict: '{snippet}'")


class LlmSafetyChecker:
    """Asks the model whether a tool call is safe to run.

    One non-streaming completion per call. Fail-closed: any error or
    unparseable answer yields a not-safe verdict, never a free pass.
    """

    def __init__(
        self,
        client: Client,
        *,
        system_prompt: str = SAFETY_SYSTEM_PROMPT,
        argument_char_limit: int = SAFETY_ARGUMENT_CHAR_LIMIT,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._argument_char_limit = argument_char_limit

    def describe(self, tool: Tool, arguments: dict[str, Any]) -> str:
        args = json.dumps(arguments, ensure_ascii=False)
        if len(args) > self._argument_char_limit:
            args = args[: self._argument_char_limit] + "..."
        return f"Tool: {tool.name}\nArguments: {args}"

    def check(self, tool: Tool, arguments: dict[str, Any]) -> SafetyVerdict:
        try:
            result = self._client.complete(
                [system_message(self._system_prompt), user_message(self.describe(tool, arguments))]
            )
        except Exception as exc:
            return SafetyVerdict(safe=False, reason=f"safety check failed: {exc}", checked=False)
        return parse_verdict(result.content or "")
