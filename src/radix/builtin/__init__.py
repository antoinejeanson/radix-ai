"""Built-in tools. All of them are sensitive and always ask for permission."""

from .shell import run_shell
from .web import fetch_url

__all__ = ["fetch_url", "run_shell"]
