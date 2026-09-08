import pytest

from fakes import FakeStream, ScriptedClient
from radix import Agent, AutoApproveGate, DenyGate, Events, Usage, tool
from radix.context import ContextManager
from radix.messages import ChatResult, ToolCall


# Tests for Agent: the tool-call loop, permission gates, hooks and
# stateful history.
@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@tool
def shell(command: str) -> str:
    """Run a command."""
    return f"ran: {command}"


def make_agent(results, tools=None, **kwargs):
    client = ScriptedClient(results)
    kwargs.setdefault("permission_gate", AutoApproveGate())
    agent = Agent(
        "tester",
        system_prompt="be brief",
        tools=tools or [add],
        client=client,
        **kwargs,
    )
    return agent, client


def test_plain_answer():
    agent, client = make_agent([ChatResult(content="2+2=4")])
    assert agent.run("what is 2+2?") == "2+2=4"
    messages = client.stream_calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[1] == {"role": "user", "content": "what is 2+2?"}


def test_tool_call_loop():
    agent, client = make_agent(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="add", raw_arguments='{"a": 2, "b": 3}')
                ]
            ),
            ChatResult(content="5"),
        ]
    )
    assert agent.run("add 2 and 3") == "5"
    second = client.stream_calls[1]["messages"]
    assert second[2]["role"] == "assistant"
    assert second[2]["tool_calls"][0]["function"] == {
        "name": "add",
        "arguments": '{"a": 2, "b": 3}',
    }
    assert second[3] == {"role": "tool", "tool_call_id": "c1", "content": "5"}


def test_events_callbacks():
    usage = Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    agent, _ = make_agent(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="add", raw_arguments='{"a": 1, "b": 1}')
                ]
            ),
            ChatResult(content="done", usage=usage),
        ]
    )
    starts: list[str] = []
    deltas: list[tuple[str, str]] = []
    activity: list[tuple[str, str]] = []
    outputs: list[tuple[str, str, str]] = []
    stops: list[tuple[str, bool, Usage | None]] = []
    events = Events(
        on_start=starts.append,
        on_delta=lambda name, text: deltas.append((name, text)),
        on_activity=lambda name, text, raw="": activity.append((name, text)),
        on_tool_output=lambda name, tool_name, output: outputs.append(
            (name, tool_name, output)
        ),
        on_stop=lambda name, elapsed, produced, usage: stops.append(
            (name, produced, usage)
        ),
    )
    agent.run("go", events=events)
    assert starts == ["tester", "tester"]
    assert deltas == [("tester", "done")]
    assert activity == [("tester", 'add {"a": 1, "b": 1}')]
    assert outputs == [("tester", "add", "2")]
    assert stops == [("tester", False, None), ("tester", True, usage)]
    assert agent._events is None


def test_unknown_tool():
    agent, client = make_agent(
        [
            ChatResult(tool_calls=[ToolCall(id="c1", name="nope", raw_arguments="{}")]),
            ChatResult(content="sorry"),
        ]
    )
    assert agent.run("x") == "sorry"
    tool_msg = client.stream_calls[1]["messages"][3]
    assert "unknown tool 'nope'" in tool_msg["content"]


def test_invalid_json_arguments():
    agent, client = make_agent(
        [
            ChatResult(
                tool_calls=[ToolCall(id="c1", name="add", raw_arguments="not json")]
            ),
            ChatResult(content="fixed"),
        ]
    )
    assert agent.run("x") == "fixed"
    assert "not valid JSON" in client.stream_calls[1]["messages"][3]["content"]


def test_bad_arguments_type():
    agent, client = make_agent(
        [
            ChatResult(
                tool_calls=[ToolCall(id="c1", name="add", raw_arguments='{"a": "x"}')]
            ),
            ChatResult(content="fixed"),
        ]
    )
    assert agent.run("x") == "fixed"
    assert "invalid arguments" in client.stream_calls[1]["messages"][3]["content"]


def test_permission_denied_is_reported_to_model():
    agent, client = make_agent(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="shell", raw_arguments='{"command": "ls"}')
                ]
            ),
            ChatResult(content="ok"),
        ],
        tools=[shell],
        permission_gate=DenyGate(),
    )
    assert agent.run("list files") == "ok"
    assert "Permission denied" in client.stream_calls[1]["messages"][3]["content"]


def test_gated_tool_runs_when_approved():
    agent, client = make_agent(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="shell", raw_arguments='{"command": "ls"}')
                ]
            ),
            ChatResult(content="done"),
        ],
        tools=[shell],
        permission_gate=AutoApproveGate(),
    )
    assert agent.run("list files") == "done"
    assert client.stream_calls[1]["messages"][3]["content"] == "ran: ls"


def test_gated_tool_is_denied_when_gate_says_no():
    agent, client = make_agent(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="shell", raw_arguments='{"command": "ls"}')
                ]
            ),
            ChatResult(content="ok"),
        ],
        tools=[shell],
        permission_gate=DenyGate(),
    )
    assert agent.run("list files") == "ok"
    assert "Permission denied" in client.stream_calls[1]["messages"][3]["content"]


def test_tool_gate_denies_even_with_permissive_global_gate():
    agent, client = make_agent(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c1", name="shell", raw_arguments='{"command": "rm -rf /"}'
                    )
                ]
            ),
            ChatResult(content="ok"),
        ],
        tools=[shell],
        permission_gate=AutoApproveGate(),
        tool_gates={"shell": DenyGate()},
    )
    assert agent.run("clean up") == "ok"
    assert "Permission denied" in client.stream_calls[1]["messages"][3]["content"]


def test_tool_gate_replaces_global_gate():
    agent, client = make_agent(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="shell", raw_arguments='{"command": "ls"}')
                ]
            ),
            ChatResult(content="done"),
        ],
        tools=[shell],
        permission_gate=DenyGate(),
        tool_gates={"shell": AutoApproveGate()},
    )
    assert agent.run("list files") == "done"
    assert client.stream_calls[1]["messages"][3]["content"] == "ran: ls"


def test_max_tool_rounds_forces_final_answer():
    loop_result = ChatResult(
        tool_calls=[ToolCall(id="c1", name="add", raw_arguments='{"a": 1, "b": 1}')]
    )
    agent, client = make_agent(
        [loop_result, loop_result, ChatResult(content="final")], max_tool_rounds=2
    )
    assert agent.run("loop forever") == "final"
    last = client.stream_calls[-1]
    assert last["tools"] is None
    assert "final answer" in last["messages"][-1]["content"]


def test_run_without_client_raises():
    agent = Agent("lonely")
    with pytest.raises(RuntimeError, match="no client"):
        agent.run("hello")


def make_stateful_agent(results, keep_recent=2):
    client = ScriptedClient(results)
    context = ContextManager(
        client,
        max_context_tokens=100000,
        reserve_output_tokens=0,
        keep_recent=keep_recent,
    )
    agent = Agent(
        "tester",
        system_prompt="be brief",
        client=client,
        context=context,
        stateful=True,
        permission_gate=AutoApproveGate(),
    )
    return agent, client


def test_compact_rewrites_history():
    agent, client = make_stateful_agent(
        [
            ChatResult(content="one"),
            ChatResult(content="two"),
            ChatResult(content="MANUAL SUMMARY"),
            ChatResult(content="three"),
        ]
    )
    agent.run("first")
    agent.run("second")

    assert agent.compact() is True

    assert len(agent.history) == 3
    assert agent.history[0]["role"] == "system"
    assert "MANUAL SUMMARY" in agent.history[0]["content"]
    assert agent.history[1:] == [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "two"},
    ]

    agent.run("third")
    sent = client.stream_calls[2]["messages"]
    assert sent[0] == {"role": "system", "content": "be brief"}
    assert sent[1]["role"] == "system"
    assert "MANUAL SUMMARY" in sent[1]["content"]


def test_compact_noop_when_history_is_short():
    agent, _ = make_stateful_agent([ChatResult(content="one")])
    agent.run("first")

    assert agent.compact() is False

    assert agent.history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
    ]


def test_stateful_history_keeps_tool_work():
    client = ScriptedClient(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="add", raw_arguments='{"a": 2, "b": 3}')
                ]
            ),
            ChatResult(content="five"),
            ChatResult(content="again answer"),
        ]
    )
    context = ContextManager(client, max_context_tokens=100000, reserve_output_tokens=0)
    agent = Agent(
        "tester",
        system_prompt="be brief",
        tools=[add],
        client=client,
        context=context,
        stateful=True,
        permission_gate=AutoApproveGate(),
    )
    agent.run("add 2 and 3")

    roles = [m["role"] for m in agent.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert agent.history[0]["content"] == "add 2 and 3"
    assert agent.history[1]["tool_calls"][0]["function"]["name"] == "add"
    assert agent.history[2]["content"] == "5"
    assert agent.history[3]["content"] == "five"

    # The next run re-sends the remembered tool work verbatim.
    agent.run("again")
    sent = client.stream_calls[2]["messages"]
    sent_roles = [m["role"] for m in sent]
    assert sent_roles == ["system", "user", "assistant", "tool", "assistant", "user"]


def test_compact_noop_without_context_or_state():
    agent, _ = make_agent([ChatResult(content="hi")])
    agent.run("hello")

    assert agent.compact() is False


def test_stateful_no_system_prompt_keeps_summary():
    # Regression: a stateful agent with NO system prompt used to drop the
    # compaction summary from its history, because _remember assumed the
    # leading system-role message was the (absent) system prompt.
    client = ScriptedClient(
        [
            ChatResult(content="SUMMARY TEXT"),  # consumed by compact's _summarize
            ChatResult(content="final answer"),  # consumed by the chat round
        ]
    )
    context = ContextManager(
        client,
        max_context_tokens=40,  # tiny budget -> compaction fires
        reserve_output_tokens=0,
        keep_recent=2,
        keep_recent_turns=1,
    )
    agent = Agent(
        "tester",
        client=client,
        context=context,
        stateful=True,
        permission_gate=AutoApproveGate(),
    )
    agent.history = [
        {"role": "user", "content": "q1 " + "x" * 30},
        {"role": "assistant", "content": "a1 " + "y" * 30},
        {"role": "user", "content": "q2 " + "z" * 30},
        {"role": "assistant", "content": "a2 " + "w" * 30},
    ]
    assert agent.run("new task") == "final answer"

    summary_msgs = [
        m
        for m in agent.history
        if m.get("role") == "system" and "SUMMARY TEXT" in (m.get("content") or "")
    ]
    assert summary_msgs, f"summary dropped from history: {agent.history}"
    # The final answer is still appended after the summary.
    assert agent.history[-1] == {"role": "assistant", "content": "final answer"}


def test_stateful_system_prompt_keeps_summary_and_no_dup():
    # With a system prompt, the summary is kept AND the system prompt is
    # not duplicated into the history.
    client = ScriptedClient(
        [
            ChatResult(content="SUMMARY TEXT"),
            ChatResult(content="final answer"),
        ]
    )
    context = ContextManager(
        client,
        max_context_tokens=40,
        reserve_output_tokens=0,
        keep_recent=2,
        keep_recent_turns=1,
    )
    agent = Agent(
        "tester",
        system_prompt="be brief",
        client=client,
        context=context,
        stateful=True,
        permission_gate=AutoApproveGate(),
    )
    agent.history = [
        {"role": "user", "content": "q1 " + "x" * 30},
        {"role": "assistant", "content": "a1 " + "y" * 30},
        {"role": "user", "content": "q2 " + "z" * 30},
        {"role": "assistant", "content": "a2 " + "w" * 30},
    ]
    assert agent.run("new task") == "final answer"

    # No system prompt leaked into the persisted history.
    assert not any(m.get("content") == "be brief" for m in agent.history), agent.history
    # The summary survived compaction.
    assert any("SUMMARY TEXT" in (m.get("content") or "") for m in agent.history), (
        agent.history
    )


class _CompactingClient:
    """Like ScriptedClient but `complete` returns a fixed summary instead of
    popping a scripted result, so the test is robust to however many times
    compaction fires during the run."""

    def __init__(self, chat_results, summary="SUMMARY"):
        self._chat = list(chat_results)
        self._summary = summary
        self.stream_calls = []
        self.complete_calls = []

    def chat_stream(self, messages, tools=None):
        self.stream_calls.append({"messages": list(messages), "tools": tools})
        return FakeStream(self._chat.pop(0))

    def complete(self, messages, tools=None):
        self.complete_calls.append({"messages": list(messages), "tools": tools})
        return ChatResult(content=self._summary)


def _exhausted_agent(client, budget, keep_recent):
    context = ContextManager(
        client,
        max_context_tokens=budget,
        reserve_output_tokens=0,
        keep_recent=keep_recent,
        keep_recent_turns=1,
        token_estimator=lambda s: len(s),  # deterministic token counts
    )
    agent = Agent(
        "tester",
        tools=[add],
        client=client,
        context=context,
        stateful=True,
        max_tool_rounds=1,
        permission_gate=AutoApproveGate(),
    )
    agent.history = [
        {"role": "user", "content": "q1 " + "x" * 30},
        {"role": "assistant", "content": "a1 " + "y" * 30},
    ]
    return agent, context


def test_rounds_exhausted_keeps_last_tool_round():
    # Regression: when max_tool_rounds is exhausted AND compaction fires, the
    # last tool call and its result used to be missing from the persisted
    # history (last_prepared was captured before they were appended).
    client = _CompactingClient(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="add", raw_arguments='{"a": 2, "b": 3}')
                ]
            ),
            ChatResult(content="final answer"),
        ]
    )
    agent, context = _exhausted_agent(client, budget=220, keep_recent=4)
    assert agent.run("add 2 and 3") == "final answer"

    # The final round must have been compacted (a summary was requested).
    assert client.complete_calls, "expected the final round to be compacted"
    # The last tool call and its result must be in the persisted history.
    assert any(m.get("tool_calls") for m in agent.history), agent.history
    assert any(
        m.get("role") == "tool" and m.get("content") == "5" for m in agent.history
    ), agent.history
    # And the final answer is the last message.
    assert agent.history[-1] == {"role": "assistant", "content": "final answer"}
    # The final round stayed within the token budget.
    final_messages = client.stream_calls[-1]["messages"]
    assert context.message_tokens(final_messages) <= context.budget
