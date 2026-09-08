from __future__ import annotations

import os
import re

from ..tool import tool

# Built-in read-only search tool: grep finds lines matching a regex across
# the files under a path. Read-only and dependency-free; it skips binary
# files, files over a size cap, and hidden directories (like .git).
MAX_FILE_BYTES = 1_000_000  # skip files larger than this


@tool
def grep(pattern: str, path: str = ".", max_results: int = 50) -> str:
    """Search files for lines matching a regular expression.

    Recursively searches the files under `path` (or just that file) and
    returns matching lines as `file:line: text`. Read-only: it never
    modifies anything. Binary files, files over ~1 MB, and hidden
    directories (e.g. .git) are skipped.

    Args:
        pattern: A regular expression (Python `re` syntax) to search for.
        path: A file or directory to search; `~` is expanded. Defaults to
            the current directory.
        max_results: Maximum number of matching lines to return (default
            50); further matches are cut with a footer.

    Returns:
        Matching lines as `file:line: text`, or "no matches" / an error.
    """
    path = os.path.expanduser(path)
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regex: {exc}"
    if not os.path.exists(path):
        return f"Error: no such file or directory: {path}"

    files: list[str] = []
    if os.path.isfile(path):
        files.append(path)
    else:
        for root, dirs, names in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in names:
                if name.startswith("."):
                    continue
                files.append(os.path.join(root, name))

    results: list[str] = []
    truncated = 0
    for file in files:
        try:
            if os.path.getsize(file) > MAX_FILE_BYTES:
                continue
            with open(file, "rb") as f:
                data = f.read()
        except OSError:
            continue
        if b"\x00" in data[:8192]:
            continue
        text = data.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                if len(results) < max_results:
                    results.append(f"{file}:{lineno}: {line}")
                else:
                    truncated += 1
    if not results:
        return "no matches"
    out = "\n".join(results)
    if truncated:
        out += f"\n... [{truncated} more matches]"
    return out
