from __future__ import annotations

import os

from ..tool import tool

MAX_CONTENT_CHARS = 16000


@tool
def read_file(path: str) -> str:
    """Read a text file from the local filesystem and return its contents."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: no such file or directory: {path}"
    if os.path.isdir(path):
        return f"Error: {path} is a directory, not a file"
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read(MAX_CONTENT_CHARS + 1)
    except UnicodeDecodeError:
        return f"Error: {path} does not look like a UTF-8 text file"
    except OSError as exc:
        return f"Error: could not read {path}: {exc}"
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "\n... [content truncated]"
    return content
