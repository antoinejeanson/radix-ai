from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

if TYPE_CHECKING:
    from .assistant import Assistant

HELP_TEXT = """commands:
  /help   show this help
  /reset  forget the conversation history
  /exit   quit (Ctrl+D also works)"""


class Repl:
    """Minimalist CLI REPL: streamed answers rendered as markdown."""

    def __init__(self, assistant: Assistant, *, console: Console | None = None) -> None:
        self.assistant = assistant
        self.console = console or Console()
        self._session: PromptSession[str] = PromptSession(history=InMemoryHistory())
        self._live: Live | None = None
        self._buffer = ""

    def run(self) -> None:
        self.console.print("[bold]Radix[/bold] — type /help for help, Ctrl+D to exit")
        while True:
            try:
                text = self._session.prompt("you> ")
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                if self._command(text):
                    break
                continue
            try:
                self.assistant.chat(text, on_delta=self._on_delta, on_activity=self._on_activity)
            except KeyboardInterrupt:
                self._flush_stream()
                self.console.print("[dim]interrupted[/dim]")
            except Exception as exc:
                self._flush_stream()
                self.console.print(f"[red]error:[/red] {exc}")
            else:
                self._flush_stream(markdown=True)
            self.console.print()

    def _command(self, text: str) -> bool:
        command = text.split()[0]
        if command in ("/exit", "/quit"):
            return True
        if command == "/help":
            self.console.print(HELP_TEXT)
        elif command == "/reset":
            self.assistant.reset()
            self.console.print("[dim]conversation reset[/dim]")
        else:
            self.console.print(f"[dim]unknown command: {command} — try /help[/dim]")
        return False

    def _on_delta(self, delta: str) -> None:
        self._buffer += delta
        if self._live is None:
            self._live = Live(
                Text(""), console=self.console, refresh_per_second=15, vertical_overflow="visible"
            )
            self._live.start()
        self._live.update(Text(self._buffer))

    def _on_activity(self, text: str) -> None:
        self._flush_stream()
        self.console.print(f"[dim]• {text}[/dim]")

    def _flush_stream(self, markdown: bool = False) -> None:
        if self._live is not None:
            content = self._buffer.strip()
            if content and markdown:
                self._live.update(Markdown(content))
            elif content:
                self._live.update(Text(content))
            else:
                self._live.update(Text(""))
            self._live.stop()
            self._live = None
        self._buffer = ""
