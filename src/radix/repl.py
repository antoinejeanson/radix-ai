from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from .context import estimate_tokens
from .events import Events

if TYPE_CHECKING:
    from .assistant import Assistant
    from .messages import Usage

HELP_TEXT = """commands:
  /help   show this help
  /undo   revert the last turn: restore changed files, rewind the conversation
  /undo N revert the last N turns
  /undo all  revert everything (files and conversation)
  /exit   quit (Ctrl+D also works)"""

MAX_OUTPUT_LINES = 40


def _diff_text(output: str) -> Text:
    text = Text()
    for line in output.splitlines():
        if line.startswith(("---", "+++")):
            text.append(line + "\n", style="bold")
        elif line.startswith("@@"):
            text.append(line + "\n", style="cyan")
        elif line.startswith("-"):
            text.append(line + "\n", style="red")
        elif line.startswith("+"):
            text.append(line + "\n", style="green")
        else:
            text.append(line + "\n", style="dim")
    text.rstrip()
    return text


def _format_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


class Repl:
    """Minimalist CLI REPL.

    Shows everything that happens: thinking spinners with elapsed time for
    the coordinator and sub-agents, every tool call and its output (edits
    as colored diffs), and sub-agent answers with their total call time and
    token use.
    """

    def __init__(
        self,
        assistant: Assistant,
        *,
        max_output_lines: int = MAX_OUTPUT_LINES,
        console: Console | None = None,
    ) -> None:
        self.assistant = assistant
        self.max_output_lines = max_output_lines
        self.console = console or Console()
        self._session: PromptSession[str] = PromptSession(history=InMemoryHistory())
        self._live: Live | None = None
        self._buffer = ""
        self._agent: str | None = None
        self._run_start: dict[str, float] = {}
        self._tokens: dict[str, int] = {}
        self._estimated: dict[str, bool] = {}
        self.events = Events(
            on_start=self._on_start,
            on_delta=self._on_delta,
            on_activity=self._on_activity,
            on_tool_output=self._on_tool_output,
            on_stop=self._on_stop,
        )

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
                self.assistant.chat(text, events=self.events)
            except KeyboardInterrupt:
                self._cleanup_stream()
                self.console.print("[dim]interrupted[/dim]")
            except Exception as exc:
                self._cleanup_stream()
                self.console.print(f"[red]error:[/red] {exc}")
            else:
                self._cleanup_stream()
            self.console.print()

    def _command(self, text: str) -> bool:
        words = text.split()
        command = words[0]
        if command in ("/exit", "/quit"):
            return True
        if command == "/help":
            self.console.print(HELP_TEXT)
        elif command == "/undo":
            self._undo(words[1:])
        else:
            self.console.print(f"[dim]unknown command: {command} — try /help[/dim]")
        return False

    def _undo(self, args: list[str]) -> None:
        if not args:
            turns = 1
        elif args[0] == "all":
            turns = len(self.assistant.undo_log)
        else:
            try:
                turns = int(args[0])
            except ValueError:
                self.console.print("[dim]usage: /undo [N|all][/dim]")
                return
        before = len(self.assistant.coordinator.history)
        result = self.assistant.undo(turns)
        if result.restored:
            self.console.print(f"[dim]restored:[/dim] {', '.join(result.restored)}")
        if len(self.assistant.coordinator.history) < before:
            self.console.print("[dim]conversation rewound[/dim]")
        if result.failed:
            self.console.print(f"[red]could not restore:[/red] {', '.join(result.failed)}")
        if not result.restored and result.history_depth is None:
            self.console.print("[dim]nothing to undo[/dim]")

    def _is_root(self) -> bool:
        return self._agent == self.assistant.coordinator.name

    def _on_start(self, name: str) -> None:
        self._cleanup_stream()
        self._agent = name
        self._live = Live(
            Spinner("dots", text=Text(f"{name} is thinking", style="dim")),
            console=self.console,
            refresh_per_second=10,
            vertical_overflow="visible",
        )
        self._live.start()

    def _on_delta(self, name: str, delta: str) -> None:
        if self._agent != name:
            self._on_start(name)
        self._buffer += delta
        style = None if self._is_root() else "dim"
        self._live.update(Text(self._buffer, style=style))

    def _on_activity(self, name: str, text: str) -> None:
        self._cleanup_stream()
        if name == self.assistant.coordinator.name:
            tool_name = text.split()[0].split("{")[0] if text.split() else ""
            if tool_name.startswith("ask_"):
                target = tool_name[4:]
                self._run_start[target] = time.monotonic()
                self._tokens.pop(target, None)
                self._estimated.pop(target, None)
        self.console.print(f"[dim]• {name}: {text}[/dim]")

    def _on_tool_output(self, name: str, tool_name: str, output: str) -> None:
        self._cleanup_stream()
        if tool_name.startswith("ask_"):
            return
        lines = output.splitlines()
        capped = "\n".join(lines[: self.max_output_lines])
        if len(lines) > self.max_output_lines:
            capped += "\n[dim]... [output truncated][/dim]"
        if tool_name == "edit_file" and not output.startswith("Error"):
            self.console.print(_diff_text(capped))
        else:
            self.console.print(Text(capped, style="dim"))

    def _on_stop(
        self, name: str, elapsed: float, produced_text: bool, usage: "Usage | None" = None
    ) -> None:
        if self._agent != name or self._live is None:
            return
        content = self._buffer.strip()
        if usage is not None:
            self._tokens[name] = self._tokens.get(name, 0) + usage.total_tokens
        elif content:
            self._tokens[name] = self._tokens.get(name, 0) + estimate_tokens(content)
            self._estimated[name] = True
        if not produced_text or not content:
            self._live.update(Text(f"• {name} thought for {elapsed:.1f}s", style="dim"))
            self._live.stop()
            self._live = None
            self._buffer = ""
            return
        if self._is_root():
            self._live.update(Markdown(content))
        else:
            total = time.monotonic() - self._run_start.get(name, time.monotonic() - elapsed)
            subtitle = f"{total:.1f}s"
            tokens = self._tokens.get(name, 0)
            if tokens:
                prefix = "~" if self._estimated.get(name) else ""
                subtitle += f" · {prefix}{_format_tokens(tokens)} tok"
            self._live.update(
                Panel(
                    Markdown(content),
                    title=name,
                    subtitle=subtitle,
                    border_style="dim",
                    expand=False,
                )
            )
        self._live.stop()
        self._live = None
        self._buffer = ""

    def _cleanup_stream(self) -> None:
        if self._live is not None:
            content = self._buffer.strip()
            self._live.update(Text(content) if content else Text(""))
            self._live.stop()
            self._live = None
        self._buffer = ""
        self._agent = None
