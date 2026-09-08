"""Built-in tools.

Which tools are sensitive is decided by the permission gate, not by the
tools themselves.
"""

# Built-in tools bundle: file editing, shell command execution, and web
# fetching — everything a quick assistant needs out of the box.
from .ask import ask_question
from .files import edit_file, read_file, write_file
from .memory import recall, remember
from .search import grep
from .shell import run_shell
from .web import fetch_url

__all__ = [
    "ask_question",
    "edit_file",
    "fetch_url",
    "grep",
    "read_file",
    "recall",
    "remember",
    "run_shell",
    "write_file",
]
