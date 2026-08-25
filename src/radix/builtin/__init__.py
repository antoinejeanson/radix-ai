"""Built-in tools.

Terminal and Internet tools are sensitive and always ask for permission.
"""

from .files import edit_file, read_file, write_file
from .shell import run_shell
from .web import fetch_url

__all__ = ["edit_file", "fetch_url", "read_file", "run_shell", "write_file"]
