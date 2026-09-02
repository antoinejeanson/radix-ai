from __future__ import annotations

import subprocess

from ..tool import tool

# Built-in shell tool: run_shell executes commands on the user's machine,
# with a timeout and a line-limited output cap, both configurable by the
# agent.
DEFAULT_MAX_OUTPUT_LINES = 500
DEFAULT_TIMEOUT_SECONDS = 120


@tool
def run_shell(
    command: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_lines: int = DEFAULT_MAX_OUTPUT_LINES,
) -> str:
    """Run a shell command on the user's machine and return its output.

    The output is capped at `max_output_lines` lines (default 500); when
    the command produces more, a footer reports how many lines were cut so
    the agent can re-run a more focused command. The command times out
    after `timeout` seconds (default 120).

    Args:
        command: The command line to run, executed via the shell.
        timeout: Maximum seconds before the command is killed; 0 means no
            timeout.
        max_output_lines: Maximum lines of output returned, after which
            the output is cut with a footer; 0 means unlimited.

    Returns:
        The exit code and merged stdout/stderr, or an error message.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout if timeout > 0 else None,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout} seconds: {command}"
    parts = []
    if proc.stdout:
        parts.append(proc.stdout)
    if proc.stderr:
        parts.append("stderr:\n" + proc.stderr)
    output = "\n".join(parts).strip() or "(no output)"
    lines = output.splitlines()
    if max_output_lines > 0 and len(lines) > max_output_lines:
        cut = len(lines) - max_output_lines
        output = "\n".join(lines[:max_output_lines])
        output += f"\n... [output truncated — {cut} more lines]"
    return f"exit code: {proc.returncode}\n{output}"
