from __future__ import annotations

import difflib
import os

from ..tool import tool
from ..undo import _write_atomic

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


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Edit a text file by replacing one exact snippet with another.

    `old_string` must match a single, unique stretch of the file exactly,
    including whitespace and newlines. Copy it verbatim from read_file output.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: no such file or directory: {path}"
    if os.path.isdir(path):
        return f"Error: {path} is a directory, not a file"
    if old_string == new_string:
        return "Error: old_string and new_string are identical, nothing to change"
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return f"Error: {path} does not look like a UTF-8 text file"
    except OSError as exc:
        return f"Error: could not read {path}: {exc}"
    count = content.count(old_string)
    if count == 0:
        return (
            f"Error: old_string not found in {path}. Copy the exact text from "
            "read_file output, including whitespace and newlines."
        )
    if count > 1:
        return (
            f"Error: old_string matches {count} places in {path}. "
            "Include more surrounding lines to make it unique."
        )
    new_content = content.replace(old_string, new_string, 1)
    try:
        _write_atomic(path, new_content)
    except OSError as exc:
        return f"Error: could not write {path}: {exc}"
    label = os.path.basename(path)
    diff = "\n".join(
        difflib.unified_diff(
            content.splitlines(),
            new_content.splitlines(),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            lineterm="",
        )
    )
    return f"Edited {path}.\n{diff}"


@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a text file with the given content."""
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        return f"Error: {path} is a directory, not a file"
    parent = os.path.dirname(path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            return f"Error: could not create directories for {path}: {exc}"
    try:
        _write_atomic(path, content)
    except OSError as exc:
        return f"Error: could not write {path}: {exc}"
    return f"Wrote {len(content)} chars to {path}."
