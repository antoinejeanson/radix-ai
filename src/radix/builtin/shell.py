from __future__ import annotations

import subprocess

from ..tool import tool

MAX_OUTPUT_CHARS = 16000
TIMEOUT_SECONDS = 120


@tool(ask_permission=True)
def run_shell(command: str) -> str:
    """Run a shell command on the user's machine and return its output."""
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {TIMEOUT_SECONDS} seconds: {command}"
    parts = []
    if proc.stdout:
        parts.append(proc.stdout)
    if proc.stderr:
        parts.append("stderr:\n" + proc.stderr)
    output = "\n".join(parts).strip() or "(no output)"
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
    return f"exit code: {proc.returncode}\n{output}"
