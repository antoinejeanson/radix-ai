from __future__ import annotations

import difflib
import os

from ..tool import tool
from ..undo import _write_atomic

# Built-in file tools: read_file (line-limited reads with offset/limit),
# edit_file (unique-snippet replace with a diff), write_file (atomic
# create/overwrite).
DEFAULT_MAX_READ_LINES = 1000


@tool
def read_file(path: str, offset: int = 0, limit: int = DEFAULT_MAX_READ_LINES) -> str:
    """Read a text file from the local filesystem and return its contents.

    The file is read line by line: `offset` skips lines, and `limit`
    caps how many lines are returned (default 1000). Use both to page
    through large files. When the file has more lines than returned, a
    header and footer report the range shown and how many lines remain,
    so the agent knows the file is bigger and can page back.

    Args:
        path: File to read; `~` is expanded.
        offset: Number of lines to skip from the start (0 starts at the
            first line).
        limit: Maximum number of lines to return, -1 for all the lines
            after `offset`.

    Returns:
        The file contents of the requested line range. When lines remain,
        a header and footer describe the range shown and how many lines
        remain. Errors start with "Error: ".
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: no such file or directory: {path}"
    if os.path.isdir(path):
        return f"Error: {path} is a directory, not a file"
    if offset < 0:
        offset = 0
    # Stream the file line by line instead of readlines(): the file iterator
    # reads in chunks and yields one line at a time, so a large file is not
    # held in memory all at once. We still pass over every line to count the
    # total (needed for the range header), but only the requested window is
    # kept.
    try:
        with open(path, encoding="utf-8") as f:
            total = 0
            selected: list[str] = []
            for line in f:
                if total >= offset and (limit < 0 or total < offset + limit):
                    selected.append(line)
                total += 1
    except UnicodeDecodeError:
        return f"Error: {path} does not look like a UTF-8 text file"
    except OSError as exc:
        return f"Error: could not read {path}: {exc}"
    content = "".join(selected)
    if offset + len(selected) < total:
        if content and not content.endswith("\n"):
            content += "\n"
        return (
            f"[Showing lines {offset + 1}-{offset + len(selected)} of {total}]\n"
            f"{content}[{total - offset - len(selected)} more lines remain]"
        )
    return content


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Edit a text file by replacing one exact snippet with another.

    `old_string` must match a single, unique stretch of the file exactly,
    including whitespace and newlines. Copy it verbatim from read_file output.

    Args:
        path: File to edit; `~` is expanded.
        old_string: The exact snippet to replace.
        new_string: The replacement text.

    Returns:
        The file path plus a unified diff, or an error message.
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
    """Create or overwrite a text file with the given content. Parent
    directories are created as needed.

    Args:
        path: Target file path; `~` is expanded.
        content: The full text to write.

    Returns:
        A confirmation with the character count, or an error message.
    """
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
