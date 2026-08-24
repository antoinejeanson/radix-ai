"""Built-in tools.

Terminal and Internet tools are sensitive and always ask for permission.
"""

from .files import read_file
from .shell import run_shell
from .web import fetch_url

__all__ = ["fetch_url", "read_file", "run_shell"]
