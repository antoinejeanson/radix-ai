import json
import time
from io import StringIO

from rich.console import Console

from fakes import ScriptedClient
from radix import Agent, Assistant, AutoApproveGate, Usage
from radix.builtin import write_file
from radix.messages import ChatResult, ToolCall
from radix.repl import Repl


# Tests for the Repl: slash commands, live event rendering, undo flow.
def make_assistant(results, tools=None):
    client = ScriptedClient(results)
    return Assistant(
        client=client,
        agents=[Agent("coder")],
        tools=tools,
        permission_gate=AutoApproveGate(),
    )


def make_repl():
    assistant = make_assistant([])
    io = StringIO()
    console = Console(file=io, force_terminal=False, width=80)
    return Repl(assistant, console=console), io


def run_repl(assistant, inputs):
    console = Console(file=open("/dev/null", "w"), force_terminal=False)
    repl = Repl(assistant, console=console)
    repl._session = type(
        "S", (), {"prompt": staticmethod(lambda *_a, **_k: inputs.pop(0))}
    )()
    repl.run()
    return repl


def run_repl_captured(assistant, inputs):
    io = StringIO()
    console = Console(file=io, force_terminal=False, width=80)
    repl = Repl(assistant, console=console)
    repl._session = type(
        "S", (), {"prompt": staticmethod(lambda *_a, **_k: inputs.pop(0))}
    )()
    repl.run()
    return repl, io


def test_repl_chat_and_commands():
    assistant = make_assistant(
        [ChatResult(content="answer one"), ChatResult(content="answer two")]
    )
    run_repl(assistant, ["/help", "first", "/undo all", "second", "/exit"])
    assert len(assistant.coordinator.history) == 2
    assert assistant.coordinator.history[0]["content"] == "second"


def test_repl_undo_restores_files_and_rewinds(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("before")
    assistant = make_assistant(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        raw_arguments=json.dumps(
                            {"path": str(path), "content": "after"}
                        ),
                    )
                ]
            ),
            ChatResult(content="done"),
        ],
        tools=[write_file],
    )
    repl, io = run_repl_captured(assistant, ["change it", "/undo", "/exit"])
    assert path.read_text() == "before"
    assert assistant.coordinator.history == []
    out = io.getvalue()
    assert "restored:" in out
    assert "conversation rewound" in out


def test_repl_undo_rewinds_conversation_only():
    assistant = make_assistant([ChatResult(content="hi")])
    run_repl_captured(assistant, ["hello", "/undo", "/exit"])
    assert assistant.coordinator.history == []


def test_repl_undo_nothing_to_undo():
    assistant = make_assistant([])
    _, io = run_repl_captured(assistant, ["/undo", "/exit"])
    assert "nothing to undo" in io.getvalue()


def test_repl_undo_bad_argument():
    assistant = make_assistant([])
    _, io = run_repl_captured(assistant, ["/undo x", "/exit"])
    assert "usage: /undo [N|all]" in io.getvalue()


def test_repl_undo_all_reverts_everything(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("v0")
    edit = lambda content: ChatResult(  # noqa: E731
        tool_calls=[
            ToolCall(
                id="c",
                name="write_file",
                raw_arguments=json.dumps({"path": str(path), "content": content}),
            )
        ]
    )
    assistant = make_assistant(
        [edit("v1"), ChatResult(content="one"), edit("v2"), ChatResult(content="two")],
        tools=[write_file],
    )
    run_repl(assistant, ["t1", "t2", "/undo all", "/exit"])
    assert path.read_text() == "v0"
    assert assistant.coordinator.history == []
    assert len(assistant.undo_log) == 0


def test_repl_survives_errors():
    class BrokenClient:
        def chat_stream(self, *a, **k):
            raise ConnectionError("server down")

    assistant = make_assistant([])
    assistant.coordinator.client = BrokenClient()
    run_repl(assistant, ["hello", "/exit"])


def test_repl_renders_thinking_tool_subagent_and_answer():
    repl, io = make_repl()
    ev = repl.events
    ev.on_start("coordinator")
    ev.on_stop("coordinator", 1.234, False)
    ev.on_activity("coordinator", 'ask_coder {"task": "write hello"}')
    ev.on_start("coder")
    ev.on_delta("coder", "print('hi')")
    ev.on_stop("coder", 2.0, True)
    ev.on_start("coordinator")
    ev.on_delta("coordinator", "**done**")
    ev.on_stop("coordinator", 0.5, True)
    out = io.getvalue()
    assert "coordinator thought for 1.2s" in out
    assert "• coordinator: ask_coder" in out
    assert "print('hi')" in out
    assert "0.0s · ~3 tok" in out
    assert "done" in out


def test_repl_subagent_panel_shows_total_time_and_reported_tokens():
    repl, io = make_repl()
    ev = repl.events
    ev.on_activity("coordinator", 'ask_coder {"task": "t"}')
    repl._run_start["coder"] = time.monotonic() - 5.0
    ev.on_start("coder")
    ev.on_delta("coder", "result")
    ev.on_stop(
        "coder",
        0.1,
        True,
        Usage(prompt_tokens=40, completion_tokens=10, total_tokens=50),
    )
    out = io.getvalue()
    assert "5.0s" in out
    assert "· 50 tok" in out
    assert "~" not in out


def test_repl_resets_subagent_stats_between_delegations():
    repl, io = make_repl()
    ev = repl.events
    ev.on_activity("coordinator", 'ask_coder {"task": "t"}')
    ev.on_start("coder")
    ev.on_delta("coder", "first")
    ev.on_stop("coder", 0.1, True, Usage(total_tokens=50))
    ev.on_activity("coordinator", 'ask_coder {"task": "t2"}')
    ev.on_start("coder")
    ev.on_delta("coder", "second")
    ev.on_stop("coder", 0.1, True, Usage(total_tokens=30))
    out = io.getvalue()
    assert "· 50 tok" in out
    assert "· 30 tok" in out
    assert "· 80 tok" not in out


def test_repl_shows_subagent_tool_calls():
    repl, io = make_repl()
    ev = repl.events
    ev.on_start("coder")
    ev.on_stop("coder", 0.1, False)
    ev.on_activity("coder", 'read_file {"path": "main.py"}')
    out = io.getvalue()
    assert "coder thought for 0.1s" in out
    assert "• coder: read_file" in out


def test_repl_eof_exits():
    def raise_eof(*_a, **_k):
        raise EOFError

    assistant = make_assistant([])
    console = Console(file=open("/dev/null", "w"), force_terminal=False)
    repl = Repl(assistant, console=console)
    repl._session = type("S", (), {"prompt": staticmethod(raise_eof)})()
    repl.run()


def test_repl_renders_edit_diff():
    repl, io = make_repl()
    ev = repl.events
    diff = (
        "Edited /tmp/app.py.\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def greet():\n"
        "-    print('hi')\n"
        "+    print('hello')"
    )
    ev.on_activity("coder", 'edit_file {"path": "/tmp/app.py"}')
    ev.on_tool_output("coder", "edit_file", diff)
    out = io.getvalue()
    assert "• coder: edit_file" in out
    assert "--- a/app.py" in out
    assert "-    print('hi')" in out
    assert "+    print('hello')" in out


def test_repl_renders_plain_tool_output_capped():
    repl, io = make_repl()
    ev = repl.events
    output = "\n".join(f"line {i}" for i in range(60))
    ev.on_tool_output("coder", "read_file", output)
    out = io.getvalue()
    assert "line 0" in out
    assert "line 39" in out
    assert "line 40" not in out
    assert "[output truncated]" in out


def test_repl_max_output_lines_configurable():
    assistant = make_assistant([])
    io = StringIO()
    console = Console(file=io, force_terminal=False, width=80)
    repl = Repl(assistant, console=console, max_output_lines=2)
    repl.events.on_tool_output(
        "coder", "read_file", "\n".join(f"line {i}" for i in range(10))
    )
    out = io.getvalue()
    assert "line 0" in out
    assert "line 1" in out
    assert "line 2" not in out
    assert "[output truncated]" in out


def test_repl_renders_edit_error_as_plain_output():
    repl, io = make_repl()
    ev = repl.events
    ev.on_tool_output(
        "coder", "edit_file", "Error: old_string not found in /tmp/app.py."
    )
    out = io.getvalue()
    assert "old_string not found" in out


def test_repl_hides_delegation_tool_output():
    repl, io = make_repl()
    ev = repl.events
    ev.on_tool_output("coordinator", "ask_coder", "the sub-agent's answer")
    assert "the sub-agent's answer" not in io.getvalue()
