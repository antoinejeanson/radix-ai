from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from .context import estimate_tokens
from .coordinator import safe_name
from .events import Events

if TYPE_CHECKING:
    from .agent import Agent
    from .assistant import Assistant
    from .messages import Message, Usage

# Repl: the interactive CLI — prompt loop, slash commands (/help, /compact,
# /undo, /exit), and live streaming views of everything the agents do.
HELP_TEXT = """commands:
  /help    show this help
  /compact summarize the old conversation now, reducing the context
  /undo    revert the last turn: restore changed files, rewind the conversation
  /undo N  revert the last N turns
  /undo all  revert everything (files and conversation)
  /transcript [path]  write the conversation to a JSONL file
  /exit    quit (Ctrl+D also works)"""


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

    Shows everything the coordinator does: thinking spinners with elapsed
    time, every tool call and its output (edits as colored diffs). Sub-agent
    work is shown more tersely: their tool calls are listed, but their tool
    results and any context line stay hidden by default (see the constructor
    options), and each delegation ends with the sub-agent's answer in a panel
    with its total call time and token use. A persistent status bar under the
    input line shows the current context usage.
    """

    def __init__(
        self,
        assistant: Assistant,
        *,
        console: Console | None = None,
        show_subagent_context: bool = False,
        show_subagent_tool_outputs: bool = False,
    ) -> None:
        """Create the REPL.

        Args:
            assistant: The Assistant to chat with; must already be fully
                configured and bound.
            console: Rich console to render on; a new one is created when
                None.
            show_subagent_context: When True, show a context line while a
                sub-agent is working — an estimate of the sub-agent's own
                (fresh) conversation, not the coordinator's. When False,
                no context line is rendered during sub-agent work.
            show_subagent_tool_outputs: When True, print every sub-agent
                tool result as it runs (edits as colored diffs). When
                False, sub-agent tool results stay hidden; each delegation
                still ends with the sub-agent's final answer panel.
        """
        self.assistant = assistant
        self.console = console or Console()
        self.show_subagent_context = show_subagent_context
        self.show_subagent_tool_outputs = show_subagent_tool_outputs
        self._session: PromptSession[str] = PromptSession(history=InMemoryHistory())
        self._live: Live | None = None
        self._buffer = ""
        self._agent: str | None = None
        self._run_start: dict[str, float] = {}
        self._tokens: dict[str, int] = {}
        self._estimated: dict[str, bool] = {}
        self._status_override: int | None = None
        self._status_message_count = 0
        self._sub_ctx: dict[str, dict[str, int]] = {}
        # Maps each delegation tool name (ask_<safe_name>) to the sub-agent's
        # real name, so tracking keyed by the real name works even when the
        # name needs sanitizing (e.g. hyphens).
        self._delegation_by_tool: dict[str, str] = {
            f"ask_{safe_name(a.name)}": a.name
            for a in self.assistant.coordinator.agents
        }
        self.events = Events(
            on_start=self._on_start,
            on_delta=self._on_delta,
            on_activity=self._on_activity,
            on_tool_output=self._on_tool_output,
            on_stop=self._on_stop,
        )

    def run(self) -> None:
        """Run the prompt loop until the user exits.

        Accepts messages, slash commands (/help, /compact,
        /undo [N|all], /exit) and Ctrl+D. Blocks, and returns when the
        user quits.
        """
        self.console.print("[bold]Radix[/bold] — type /help for help, Ctrl+D to exit")
        while True:
            try:
                text = self._session.prompt("you> ", bottom_toolbar=self._status_bar)
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
                self._status_override = self.assistant.context.message_tokens(
                    self._status_messages() + [{"role": "user", "content": text}]
                )
                self._status_message_count = len(self._status_messages()) + 1
                self.assistant.chat(text, events=self.events)
            except KeyboardInterrupt:
                self._cleanup_stream()
                self.console.print("[dim]interrupted[/dim]")
            except Exception as exc:
                self._cleanup_stream()
                self.console.print(f"[red]error:[/red] {exc}")
            finally:
                self._cleanup_stream()
                self._status_override = None
            self.console.print()

    def _status_messages(self) -> list[Message]:
        """What the next model round would send: system prompt plus history.

        Returns:
            The coordinator's system prompt (if any) followed by its saved
            conversation history.
        """
        messages: list[Message] = []
        if self.assistant.coordinator.system_prompt:
            messages.append(
                {"role": "system", "content": self.assistant.coordinator.system_prompt}
            )
        messages.extend(self.assistant.coordinator.history)
        return messages

    def _bar_text(self, name: str | None = None) -> str | None:
        """The status line: current context use versus the full window.

        Between turns it is exact, computed from the coordinator's memory
        (system prompt plus history, which now includes prior tool calls,
        their results, and delegation results). During a chat turn it
        tracks a running estimate that adds the coordinator's own tool
        outputs as they stream in.

        For a sub-agent (pass its name, or the active `_agent`), returns the
        sub-agent's own estimate instead — a fresh context seeded with its
        system prompt and task, grown by each tool call and result — or None
        when `show_subagent_context` is off or no estimate is available yet.

        Args:
            name: The sub-agent to describe; the active streaming agent when
                None. The coordinator bar is always returned when this is
                not a sub-agent.

        Returns:
            One formatted line, e.g. "context: 3.4k / 8.2k tok · 21 messages
            · 41%" or "explorer context: ..." — or None for a sub-agent that
            should not show a bar.
        """
        active = name if name is not None else self._agent
        if active is not None and active != self.assistant.coordinator.name:
            return self._subagent_bar(active)
        used = self._status_override
        if used is None:
            used = self.assistant.context.message_tokens(self._status_messages())
            message_count = len(self._status_messages())
        else:
            message_count = self._status_message_count
        budget = self.assistant.context.max_context_tokens
        pct = round(100 * used / budget) if budget else 0
        return (
            f"context: {_format_tokens(used)} / {_format_tokens(budget)} tok"
            f" · {message_count} messages · {pct}%"
        )

    def _subagent_bar(self, name: str) -> str | None:
        """The context line for one sub-agent's in-flight (fresh) context.

        Args:
            name: The sub-agent's name.

        Returns:
            The formatted sub-agent context line, or None when
            `show_subagent_context` is off or no estimate has been recorded.
        """
        if not self.show_subagent_context:
            return None
        info = self._sub_ctx.get(name)
        if info is None:
            return None
        budget = self.assistant.context.max_context_tokens
        used = info["tokens"]
        pct = round(100 * used / budget) if budget else 0
        return (
            f"{name} context: {_format_tokens(used)} / {_format_tokens(budget)} tok"
            f" · {info['messages']} messages · {pct}%"
        )

    def _agent_by_name(self, name: str) -> Agent | None:
        """Find a registered sub-agent by name.

        Args:
            name: The agent's name.

        Returns:
            The Agent object, or None when no sub-agent has that name.
        """
        for agent in self.assistant.coordinator.agents:
            if agent.name == name:
                return agent
        return None

    def _delegation_task(self, arguments: str) -> str:
        """Extract the plain task text from a delegation tool's arguments.

        Args:
            arguments: The raw tool arguments JSON.

        Returns:
            The `task` value, or the raw arguments when it cannot be parsed.
        """
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
        return parsed.get("task", arguments) if isinstance(parsed, dict) else arguments

    def _subagent_baseline(self, name: str, arguments: str) -> dict[str, int]:
        """The sub-agent's starting context for a delegation.

        A stateless sub-agent begins each run with a fresh conversation: its
        system prompt (when it has one) plus the delegated task.

        Args:
            name: The sub-agent's name.
            arguments: The raw `task` argument of the delegation call.

        Returns:
            A `{"tokens", "messages"}` summary of that starting context.
        """
        agent = self._agent_by_name(name)
        messages: list[Message] = []
        if agent is not None and agent.system_prompt:
            messages.append({"role": "system", "content": agent.system_prompt})
        messages.append({"role": "user", "content": self._delegation_task(arguments)})
        return {
            "tokens": self.assistant.context.message_tokens(messages),
            "messages": len(messages),
        }

    def _status_bar(self) -> list[tuple[str, str]]:
        """The persistent bottom toolbar: the coordinator's context use.

        Recomputed whenever the prompt redraws, so it stays current after
        chat turns, /compact and /undo. The prompt is only shown between
        turns, so this always reflects the saved conversation exactly.

        Returns:
            One formatted line, e.g. "context: 1.2k / 6.1k tok · 14 messages
            · 19%".
        """
        bar = self._bar_text()
        return [("dim", bar)] if bar is not None else []

    def _command(self, text: str) -> bool:
        """Handle one slash command.

        Args:
            text: The full input line, starting with "/".

        Returns:
            True when the command quits the REPL, False otherwise.
        """
        words = text.split()
        command = words[0]
        if command in ("/exit", "/quit"):
            return True
        if command == "/help":
            self.console.print(HELP_TEXT)
        elif command == "/compact":
            self._compact()
        elif command == "/undo":
            self._undo(words[1:])
        elif command == "/transcript":
            self._transcript(words[1:])
        else:
            self.console.print(f"[dim]unknown command: {command} — try /help[/dim]")
        return False

    def _undo(self, args: list[str]) -> None:
        """Implement `/undo [N|all]`.

        Args:
            args: The words after "/undo".
        """
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
            self.console.print(
                f"[red]could not restore:[/red] {', '.join(result.failed)}"
            )
        if not result.restored and result.history_depth is None:
            self.console.print("[dim]nothing to undo[/dim]")

    def _transcript(self, args: list[str]) -> None:
        """Implement `/transcript [path]`: write the conversation to JSONL.

        Args:
            args: The words after "/transcript"; the first is the path.
        """
        path = args[0] if args else "radix-transcript.jsonl"
        count = self.assistant.export_transcript(path)
        self.console.print(f"[dim]wrote {count} messages to {path}[/dim]")

    def _compact(self) -> None:
        """Implement `/compact`: summarize the old conversation now.

        Delegates to the coordinator's context manager so the very next
        model round sends less context.
        """
        before = len(self.assistant.coordinator.history)
        if self.assistant.compact():
            after = len(self.assistant.coordinator.history)
            self.console.print(
                f"[dim]context compacted: {before} messages → {after}[/dim]"
            )
        else:
            self.console.print(
                "[dim]nothing to compact — the conversation is too short[/dim]"
            )

    def _is_root(self) -> bool:
        """Whether the currently streaming agent is the coordinator.

        Returns:
            True when the active agent is the coordinator.
        """
        return self._agent == self.assistant.coordinator.name

    def _with_bar(self, body):
        """Wrap a live renderable so the context bar stays visible under it.

        The bar is omitted entirely when the active agent should not show one
        (sub-agents with `show_subagent_context` disabled).

        Args:
            body: The renderable shown above the status line.

        Returns:
            A Group of the body plus the dim context bar, or just `body`.
        """
        bar = self._bar_text()
        if bar is None:
            return body
        return Group(body, Text(bar, style="dim"))

    def _on_start(self, name: str) -> None:
        """Event: a model round begins for `name`; show a thinking spinner.

        Args:
            name: The agent starting to think.
        """
        self._cleanup_stream()
        self._agent = name
        self._live = Live(
            self._with_bar(
                Spinner("dots", text=Text(f"{name} is thinking", style="dim"))
            ),
            console=self.console,
            refresh_per_second=10,
            vertical_overflow="visible",
        )
        self._live.start()

    def _on_delta(self, name: str, delta: str) -> None:
        """Event: streamed text arrived; append it to the live view.

        Args:
            name: The agent producing the text.
            delta: The new chunk of text.
        """
        if self._agent != name:
            self._on_start(name)
        self._buffer += delta
        style = None if self._is_root() else "dim"
        self._live.update(self._with_bar(Text(self._buffer, style=style)))

    def _on_activity(self, name: str, text: str, raw_arguments: str = "") -> None:
        """Event: a tool call is about to run; print it, start the
        stopwatch when a sub-agent is being delegated to, and add the
        tool-call message to the running context estimate.

        Args:
            name: The agent calling the tool.
            text: One-line description of the call.
            raw_arguments: The full serialized arguments of the call, used
                to keep the context-bar estimate honest.
        """
        self._cleanup_stream()
        if name == self.assistant.coordinator.name:
            if self._status_override is not None:
                self._status_override += 4 + estimate_tokens(raw_arguments)
                self._status_message_count += 1
            tool_name = text.split()[0].split("{")[0] if text.split() else ""
            if tool_name.startswith("ask_"):
                # Resolve the sub-agent's real name from the (sanitized)
                # delegation tool name, so tracking keyed by the real name
                # matches the events the sub-agent emits under that name.
                target = self._delegation_by_tool.get(tool_name, tool_name[4:])
                self._run_start[target] = time.monotonic()
                self._tokens.pop(target, None)
                self._estimated.pop(target, None)
                self._sub_ctx[target] = self._subagent_baseline(target, raw_arguments)
        elif self.show_subagent_context:
            info = self._sub_ctx.setdefault(name, {"tokens": 0, "messages": 0})
            info["tokens"] += 4 + estimate_tokens(raw_arguments)
            info["messages"] += 1
        self.console.print(f"[dim]• {name}: {text}[/dim]")

    def _on_tool_output(self, name: str, tool_name: str, output: str) -> None:
        """Event: a tool finished; print its output, styled (diffs in color
        for edit_file), then the refreshed context line.

        Sub-agent tool results are silent by default — the delegation ends
        with the sub-agent's answer panel. With `show_subagent_tool_outputs`
        they print like coordinator results, and with
        `show_subagent_context` the line shown is the sub-agent's own.

        Args:
            name: The agent that ran the tool.
            tool_name: The tool's name.
            output: The tool's output text.
        """
        self._cleanup_stream()
        if (
            self._status_override is not None
            and name == self.assistant.coordinator.name
        ):
            self._status_override += estimate_tokens(output)
            self._status_message_count += 1
        if tool_name.startswith("ask_"):
            self.console.print(f"[dim]{self._bar_text()}[/dim]")
            return
        if name != self.assistant.coordinator.name:
            if self.show_subagent_context:
                info = self._sub_ctx.setdefault(name, {"tokens": 0, "messages": 0})
                info["tokens"] += estimate_tokens(output)
                info["messages"] += 1
            if not self.show_subagent_tool_outputs:
                return
            if tool_name == "edit_file" and not output.startswith("Error"):
                self.console.print(_diff_text(output))
            else:
                self.console.print(Text(output, style="dim"))
            bar = self._bar_text(name)
            if bar is not None:
                self.console.print(f"[dim]{bar}[/dim]")
            return
        if tool_name == "edit_file" and not output.startswith("Error"):
            self.console.print(_diff_text(output))
        else:
            self.console.print(Text(output, style="dim"))
        self.console.print(f"[dim]{self._bar_text()}[/dim]")

    def _on_stop(
        self,
        name: str,
        elapsed: float,
        produced_text: bool,
        usage: "Usage | None" = None,
    ) -> None:
        """Event: a model round ended; render the final answer (markdown
        for the coordinator, a panel with time/tokens for sub-agents).

        Args:
            name: The agent that finished.
            elapsed: Duration of the round in seconds.
            produced_text: Whether any text streamed during the round.
            usage: Token accounting from the server, when reported.
        """
        if self._agent != name or self._live is None:
            return
        content = self._buffer.strip()
        if usage is not None:
            self._tokens[name] = self._tokens.get(name, 0) + usage.total_tokens
        elif content:
            self._tokens[name] = self._tokens.get(name, 0) + estimate_tokens(content)
            self._estimated[name] = True
        if not produced_text or not content:
            self._live.update(
                self._with_bar(
                    Text(f"• {name} thought for {elapsed:.1f}s", style="dim")
                )
            )
            self._live.stop()
            self._live = None
            self._buffer = ""
            return
        if self._is_root():
            self._live.update(self._with_bar(Markdown(content)))
        else:
            total = time.monotonic() - self._run_start.get(
                name, time.monotonic() - elapsed
            )
            subtitle = f"{total:.1f}s"
            tokens = self._tokens.get(name, 0)
            if tokens:
                prefix = "~" if self._estimated.get(name) else ""
                subtitle += f" · {prefix}{_format_tokens(tokens)} tok"
            self._live.update(
                self._with_bar(
                    Panel(
                        Markdown(content),
                        title=name,
                        subtitle=subtitle,
                        border_style="dim",
                        expand=False,
                    )
                )
            )
        self._live.stop()
        self._live = None
        self._buffer = ""

    def _cleanup_stream(self) -> None:
        """Stop the live view and reset the stream state (also used to
        recover from interrupts)."""
        if self._live is not None:
            content = self._buffer.strip()
            self._live.update(Text(content) if content else Text(""))
            self._live.stop()
            self._live = None
        self._buffer = ""
        self._agent = None
