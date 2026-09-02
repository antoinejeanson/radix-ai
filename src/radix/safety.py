from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .messages import system_message, user_message

if TYPE_CHECKING:
    from .client import Client
    from .tool import Tool

# Safety: an LLM safety checker that classifies one tool call as SAFE or
# DANGEROUS for the permission gates, failing closed on any error. The
# reviewer is told the current working directory (CWD) on every call — the
# scope within which destructive work is normal — so "is this an attack or
# just part of the task?" is judged against the actual project.
SAFETY_SYSTEM_PROMPT = (
    "You are a strict safety reviewer for an AI assistant that can run tools on the "
    "user's machine.\n"
    "A separate context line gives the current working directory (CWD): the project "
    "the user is working on. Tool calls are expected to operate inside that "
    "directory.\n\n"
    "Decide whether running the given tool call could harm the user: irreversible or "
    "destructive changes to anything outside the working directory, sending private "
    "or project data to the Internet, expensive operations, or anything the user "
    "would not want to happen without asking first.\n\n"
    "SAFE unless there are other risks:\n"
    "- Read-only and everyday operations (ls, grep, find, git status, read_file).\n"
    "- Destructive operations on paths INSIDE the working directory: deleting or "
    "overwriting files that are part of the current task, build artifacts, temp or "
    "generated files. Deleting project files inside the CWD is a normal part of "
    "coding, not an attack.\n"
    "- Plain read-only web fetches (GET) of documentation or public pages.\n\n"
    "DANGEROUS:\n"
    "- Destructive operations on any path OUTSIDE the working directory: parent "
    "folders, '..', '/', system directories (/etc, /usr, /var), the home directory, "
    "and files of other projects.\n"
    "- Destructive operations at whole-project or whole-system scale even inside the "
    "CWD: deleting the repository root, wiping the main database, 'rm -rf /', 'dd', "
    "'mkfs'.\n"
    "- Sending local data to the Internet: uploading file contents, POST/PUT/PATCH "
    "bodies that contain local data, 'curl'/'wget'/'scp'/'ftp' commands that push "
    "data to a server, pasting into unknown web services, or crafting URLs that "
    "encode private data. When the assistant runs locally for privacy, nothing "
    "should leave the machine carelessly, even to seemingly benign hosts.\n"
    "- Expensive or unsafe shell commands (network data transfers, 'dd', 'mkfs', "
    "and similar).\n\n"
    "Reply with exactly two lines: the first line is SAFE or DANGEROUS, the second "
    "line is one short sentence saying why."
)


@dataclass
class SafetyVerdict:
    """The LLM safety checker's opinion about one tool call.

    `checked` is False when the check itself failed (network error, empty
    response, ...); the verdict then fails closed and is not safe.

    Fields:
        safe: Whether the call is deemed safe to run.
        reason: Short human-readable justification.
        checked: True when the model actually produced a verdict; False
            when the check failed.
    """

    safe: bool
    reason: str
    checked: bool = True


def _extract_reason(text: str) -> str:
    """Extract the justification line that follows the verdict word.

    Args:
        text: The model's raw answer.

    Returns:
        The first non-empty line after the SAFE/DANGEROUS line, or the
        first line (or "") when nothing follows.
    """
    lines = [line.strip() for line in text.strip().splitlines()]
    for index, line in enumerate(lines):
        if re.search(r"\b(SAFE|DANGEROUS)\b", line, re.IGNORECASE):
            rest = [r for r in lines[index + 1 :] if r.strip()]
            return rest[0] if rest else ""
    return lines[0] if lines else ""


def parse_verdict(text: str) -> SafetyVerdict:
    """Parse a model answer into a verdict.

    Fail-closed: dangerous keywords win, and an answer with no recognizable
    verdict is treated as not safe.

    Args:
        text: The model's answer.

    Returns:
        A SafetyVerdict with the parsed outcome and reason.
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

    One non-streaming completion per call. The system prompt is prefixed
    with the current working directory, so the reviewer can judge whether a
    destructive call targets the project being worked on (safe) or something
    outside it (dangerous). Fail-closed: any error or unparseable answer
    yields a not-safe verdict, never a free pass.
    """

    def __init__(
        self,
        client: Client,
        *,
        system_prompt: str = SAFETY_SYSTEM_PROMPT,
        cwd: str | None = None,
    ) -> None:
        """Create the safety checker.

        Args:
            client: Client used for the safety completion.
            system_prompt: Instructions for the reviewing model; defaults
                to SAFETY_SYSTEM_PROMPT. The checking directory is always
                prepended as a context line.
            cwd: Directory the reviewer treats as the working scope; the
                process's current working directory when None (resolved at
                construction time).
        """
        self._client = client
        self._cwd = cwd if cwd is not None else os.getcwd()
        self._system_prompt = system_prompt

    def describe(self, tool: Tool, arguments: dict[str, Any]) -> str:
        """Format one tool call as the text handed to the safety model.

        Args:
            tool: The tool being called.
            arguments: Its parsed arguments, serialized as JSON.

        Returns:
            A "Tool: <name>\nArguments: <json>" block.
        """
        args = json.dumps(arguments, ensure_ascii=False)
        return f"Tool: {tool.name}\nArguments: {args}"

    def check(self, tool: Tool, arguments: dict[str, Any]) -> SafetyVerdict:
        """Ask the model whether the call is safe to run.

        Args:
            tool: The tool being called.
            arguments: Its parsed arguments.

        Returns:
            A SafetyVerdict; `checked` is False when the completion failed,
            and the verdict is then not safe.
        """
        try:
            system_text = (
                f"Current working directory: {self._cwd}\n\n{self._system_prompt}"
            )
            result = self._client.complete(
                [
                    system_message(system_text),
                    user_message(self.describe(tool, arguments)),
                ]
            )
        except Exception as exc:
            return SafetyVerdict(
                safe=False, reason=f"safety check failed: {exc}", checked=False
            )
        return parse_verdict(result.content or "")
