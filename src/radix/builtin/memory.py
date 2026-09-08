from __future__ import annotations

import os
from datetime import datetime, timezone

from ..tool import tool

# Built-in memory tools: remember and recall, backed by a single persistent
# markdown file. This is the agent's long-term memory — small, local, and
# dependency-free. The file path is a module-level default so hosts and
# tests can point it elsewhere (e.g. at a temp file).
NOTES_PATH = ".radix/notes.md"
MAX_RECALL_LINES = 50


@tool
def remember(text: str) -> str:
    """Remember a fact, decision, or preference for future sessions.

    Appends the text to the persistent notes file, so it survives across
    runs. Use it to store things worth keeping long-term: project facts,
    user preferences, decisions and their rationale.

    Args:
        text: The note to store, as a single line of plain text.

    Returns:
        A confirmation with the notes file path.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    parent = os.path.dirname(NOTES_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(NOTES_PATH, "a", encoding="utf-8") as f:
        f.write(f"- [{stamp}] {text}\n")
    return f"Remembered (in {NOTES_PATH})."


@tool
def recall(query: str) -> str:
    """Search the persistent notes for a keyword or phrase.

    A case-insensitive substring search over the notes file. Use it to find
    something you remembered earlier.

    Args:
        query: The keyword or phrase to look for.

    Returns:
        The matching note lines, or "no matching notes" when there are none
        or the notes file does not exist yet.
    """
    if not os.path.exists(NOTES_PATH):
        return "no matching notes"
    with open(NOTES_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    needle = query.lower()
    matches = [ln.rstrip("\n") for ln in lines if needle in ln.lower()]
    if not matches:
        return "no matching notes"
    if len(matches) > MAX_RECALL_LINES:
        shown = matches[:MAX_RECALL_LINES]
        more = len(matches) - MAX_RECALL_LINES
        return "\n".join(shown) + f"\n... [{more} more matches]"
    return "\n".join(matches)
