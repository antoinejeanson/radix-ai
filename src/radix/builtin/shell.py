from __future__ import annotations

import subprocess

from ..tool import tool

# Built-in shell tool: run_shell executes commands on the user's machine,
# with a timeout, captured output, and an output size cap.
MAX_OUTPUT_CHARS = 16000
TIMEOUT_SECONDS = 120


@tool
def run_shell(command: str) -> str:
    """Run a shell command on the user's machine and return its output.

    Args:
        command: The command line to run, executed via the shell. Times
            out after TIMEOUT_SECONDS; output is capped at
            MAX_OUTPUT_CHARS characters.

    Returns:
        The exit code and merged stdout/stderr, or an error message.
    """
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
