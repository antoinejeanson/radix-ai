import json
import re
import time
from io import StringIO

from rich.console import Console

from fakes import ScriptedClient
from radix import Agent, Assistant, AutoApproveGate, Usage
from radix.builtin import write_file
from radix.messages import ChatResult, ToolCall
from radix.repl import Repl, _format_tokens


# Tests for the Repl: slash commands, live event rendering, undo flow.
def make_assistant(results, tools=None):
    client = ScriptedClient(results)
    return Assistant(
        client=client,
        agents=[Agent("coder")],
        tools=tools,
        permission_gate=AutoApproveGate(),
    )


def make_repl(**options):
    assistant = make_assistant([])
    io = StringIO()
    console = Console(file=io, force_terminal=False, width=80)
    return Repl(assistant, console=console, **options), io


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


def make_recording_repl(assistant, inputs):
    captured = {"toolbar": None}
    io = StringIO()
    console = Console(file=io, force_terminal=False, width=80)
    repl = Repl(assistant, console=console)

    def prompt(*_a, **_k):
        captured["toolbar"] = _k.get("bottom_toolbar")
        return inputs.pop(0)

    repl._session = type("S", (), {"prompt": staticmethod(prompt)})()
    return repl, captured


def bar_text(captured):
    toolbar = captured["toolbar"]
    assert toolbar is not None, "bottom_toolbar was never requested"
    return "".join(part for _, part in toolbar())


def status_messages(assistant):
    return [
        {"role": "system", "content": assistant.coordinator.system_prompt},
        *assistant.coordinator.history,
    ]


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


def test_repl_compact_command():
    assistant = make_assistant(
        [
            ChatResult(content="one"),
            ChatResult(content="two"),
            ChatResult(content="three"),
            ChatResult(content="MANUAL SUMMARY"),
        ]
    )
    repl, io = run_repl_captured(
        assistant, ["first", "second", "third", "/compact", "/exit"]
    )

    history = assistant.coordinator.history
    assert len(history) == 5
    assert history[0]["role"] == "system"
    assert "MANUAL SUMMARY" in history[0]["content"]
    out = io.getvalue()
    assert "context compacted: 6 messages → 5" in out


def test_repl_compact_nothing_to_compact():
    assistant = make_assistant([])
    _, io = run_repl_captured(assistant, ["/compact", "/exit"])
    assert "nothing to compact" in io.getvalue()


def test_repl_transcript_command(tmp_path):
    assistant = make_assistant([ChatResult(content="one"), ChatResult(content="two")])
    path = str(tmp_path / "t.jsonl")
    _, io = run_repl_captured(
        assistant, ["first", "second", f"/transcript {path}", "/exit"]
    )
    expected = len(assistant.coordinator.history)
    # (the full path may wrap at the console width, so match the prefix)
    assert f"wrote {expected} messages" in io.getvalue()
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert len(lines) == expected
    assert all(json.loads(line)["role"] in ("user", "assistant") for line in lines)


def test_repl_transcript_default_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assistant = make_assistant([ChatResult(content="one")])
    _, io = run_repl_captured(assistant, ["first", "/transcript", "/exit"])
    assert "wrote 2 messages to radix-transcript.jsonl" in io.getvalue()
    assert (tmp_path / "radix-transcript.jsonl").exists()


def test_repl_status_bar_shows_context():
    assistant = make_assistant([ChatResult(content="answer one")])
    repl, captured = make_recording_repl(assistant, ["first", "/exit"])
    repl.run()

    messages = status_messages(assistant)
    used = assistant.context.message_tokens(messages)
    budget = assistant.context.max_context_tokens
    bar = bar_text(captured)
    assert f"context: {_format_tokens(used)} / {_format_tokens(budget)} tok" in bar
    assert f"· {len(messages)} messages" in bar
    assert f"· {round(100 * used / budget)}%" in bar


def test_repl_status_bar_grows_with_history():
    single = make_assistant([ChatResult(content="one")])
    repl_one, captured_one = make_recording_repl(single, ["first", "/exit"])
    repl_one.run()
    used_one = single.context.message_tokens(status_messages(single))

    assistant = make_assistant([ChatResult(content="one"), ChatResult(content="two")])
    repl, captured = make_recording_repl(assistant, ["first", "second", "/exit"])
    repl.run()
    used_two = assistant.context.message_tokens(status_messages(assistant))

    assert used_two > used_one
    assert f"context: {_format_tokens(used_two)}" in bar_text(captured)


def test_repl_status_bar_after_undo_all():
    assistant = make_assistant([ChatResult(content="hi")])
    repl, captured = make_recording_repl(assistant, ["hello", "/undo all", "/exit"])
    repl.run()

    assert assistant.coordinator.history == []
    messages = [{"role": "system", "content": assistant.coordinator.system_prompt}]
    used = assistant.context.message_tokens(messages)
    bar = bar_text(captured)
    assert f"context: {_format_tokens(used)}" in bar
    pct = round(100 * used / assistant.context.max_context_tokens)
    assert f"· 1 messages · {pct}%" in bar


def test_repl_status_bar_after_compact():
    assistant = make_assistant(
        [
            ChatResult(content="one"),
            ChatResult(content="two"),
            ChatResult(content="three"),
        ]
    )
    repl, captured = make_recording_repl(
        assistant, ["first", "second", "third", "/compact", "/exit"]
    )
    repl.run()

    history = assistant.coordinator.history
    assert len(history) == 5
    assert history[0]["role"] == "system"
    used = assistant.context.message_tokens(status_messages(assistant))
    bar = bar_text(captured)
    assert f"context: {_format_tokens(used)}" in bar
    assert f"· {len(status_messages(assistant))} messages" in bar


def _parse_tokens(text):
    if text.endswith("k"):
        return int(float(text[:-1]) * 1000)
    return int(text)


def test_repl_status_bar_grows_across_tool_rounds(tmp_path):
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
        [edit("v1"), edit("v2"), ChatResult(content="final")],
        tools=[write_file],
    )
    io = StringIO()
    console = Console(file=io, force_terminal=False, width=80)
    repl = Repl(assistant, console=console)
    inputs = ["change it", "/exit"]
    repl._session = type(
        "S", (), {"prompt": staticmethod(lambda *_a, **_k: inputs.pop(0))}
    )()
    repl.run()
    out = io.getvalue()

    bars = [
        _parse_tokens(used)
        for used in re.findall(r"context: ([\d.]+k|\d+) / [\d.]+k tok", out)
    ]
    assert len(bars) >= 3
    assert any(b < n for b, n in zip(bars, bars[1:]))


def test_repl_status_bar_persists_tool_work_between_turns(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("v0")
    write = lambda content: ChatResult(  # noqa: E731
        tool_calls=[
            ToolCall(
                id="c",
                name="write_file",
                raw_arguments=json.dumps({"path": str(path), "content": content}),
            )
        ]
    )
    assistant = make_assistant(
        [write("A" * 800), ChatResult(content="written"), ChatResult(content="second")],
        tools=[write_file],
    )
    bars = []
    io = StringIO()
    console = Console(file=io, force_terminal=False, width=80)
    repl = Repl(assistant, console=console)

    def prompt(*_a, **_k):
        bars.append("".join(part for _, part in _k["bottom_toolbar"]()))
        return inputs.pop(0)

    inputs = ["first", "second", "/exit"]
    repl._session = type("S", (), {"prompt": staticmethod(prompt)})()
    repl.run()

    assert "write_file" in json.dumps(assistant.coordinator.history)
    used_before = _parse_tokens(
        re.findall(r"context: ([\d.]+k|\d+) / [\d.]+k tok", bars[0])[0]
    )
    used_after_t1 = _parse_tokens(
        re.findall(r"context: ([\d.]+k|\d+) / [\d.]+k tok", bars[1])[0]
    )
    used_after_t2 = _parse_tokens(
        re.findall(r"context: ([\d.]+k|\d+) / [\d.]+k tok", bars[2])[0]
    )
    # Turn 1's tool output (~800 chars ≈ 200 tokens) stays in the bar.
    assert used_after_t1 - used_before >= 150
    # No snap-back: the second prompt's bar still shows the tool work.
    assert used_after_t2 >= used_after_t1


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


def test_repl_subagent_tracking_uses_real_name_for_sanitized_names():
    # A sub-agent whose name needs sanitizing (hyphen): the REPL must track
    # it under its REAL name, matching the events the sub-agent emits.
    client = ScriptedClient([])
    assistant = Assistant(
        client=client,
        agents=[Agent("code-review", system_prompt="be brief")],
        permission_gate=AutoApproveGate(),
    )
    io = StringIO()
    console = Console(file=io, force_terminal=False, width=80)
    repl = Repl(assistant, console=console)
    ev = repl.events
    # The delegation tool is ask_code_review (sanitized from "code-review").
    ev.on_activity(
        "coordinator", 'ask_code_review {"task": "do work"}', '{"task": "do work"}'
    )
    # Tracking must be keyed by the real name, not the sanitized one.
    assert "code-review" in repl._run_start
    assert "code-review" in repl._sub_ctx
    assert "code_review" not in repl._run_start
    # The baseline includes the agent's system prompt (prompt + task = 2).
    assert repl._sub_ctx["code-review"]["messages"] == 2
    # The sub-agent emits events under its real name; the panel uses the
    # recorded start time and reported tokens.
    repl._run_start["code-review"] = time.monotonic() - 5.0
    ev.on_start("code-review")
    ev.on_delta("code-review", "result")
    ev.on_stop("code-review", 0.1, True, Usage(total_tokens=50))
    out = io.getvalue()
    assert "5.0s" in out
    assert "· 50 tok" in out


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
    repl, io = make_repl(show_subagent_tool_outputs=True)
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


def test_repl_renders_plain_tool_output_uncapped():
    repl, io = make_repl(show_subagent_tool_outputs=True)
    ev = repl.events
    output = "\n".join(f"line {i}" for i in range(60))
    ev.on_tool_output("coder", "read_file", output)
    out = io.getvalue()
    assert "line 0" in out
    assert "line 59" in out
    assert "[output truncated]" not in out


def test_repl_renders_edit_error_as_plain_output():
    repl, io = make_repl(show_subagent_tool_outputs=True)
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


def test_repl_subagent_tool_output_hidden_by_default():
    repl, io = make_repl()
    ev = repl.events
    ev.on_start("coder")
    ev.on_stop("coder", 0.1, False)
    ev.on_activity("coder", 'read_file {"path": "main.py"}')
    ev.on_tool_output("coder", "read_file", "secret file content")
    out = io.getvalue()
    assert "• coder: read_file" in out
    assert "secret file content" not in out


def test_repl_subagent_tool_output_shown_when_enabled():
    repl, io = make_repl(show_subagent_tool_outputs=True)
    ev = repl.events
    ev.on_tool_output("coder", "read_file", "file content")
    assert "file content" in io.getvalue()


def test_repl_no_context_bar_for_subagents_by_default():
    repl, io = make_repl()
    ev = repl.events
    ev.on_activity("coordinator", 'ask_coder {"task": "t"}')
    ev.on_start("coder")
    ev.on_delta("coder", "thinking")
    ev.on_tool_output("coder", "read_file", "out")
    ev.on_stop("coder", 0.2, True)
    out = io.getvalue()
    assert "context:" not in out


def test_repl_shows_subagent_context_bar_when_enabled():
    repl, io = make_repl(show_subagent_context=True, show_subagent_tool_outputs=True)
    ev = repl.events
    ev.on_activity("coordinator", 'ask_coder {"task": "t"}')
    ev.on_start("coder")
    ev.on_delta("coder", "thinking")
    ev.on_tool_output("coder", "read_file", "a" * 200)
    ev.on_stop("coder", 0.2, True)
    out = io.getvalue()
    assert "coder context: " in out
    assert "coordinator context: " not in out


def test_repl_subagent_context_is_seeded_then_grows():
    repl, io = make_repl(show_subagent_context=True, show_subagent_tool_outputs=True)
    ev = repl.events
    parse = lambda text: _parse_tokens(  # noqa: E731
        re.findall(r"coder context: ([\d.]+k|\d+) /", text)[0]
    )
    ev.on_activity("coordinator", 'ask_coder {"task": "hello"}')
    before = parse(repl._bar_text("coder"))
    ev.on_activity("coder", 'read_file {"path": "a.py"}')
    ev.on_tool_output("coder", "read_file", "b" * 400)
    ev.on_stop("coder", 0.2, True)
    after = parse(repl._bar_text("coder"))
    assert after > before
