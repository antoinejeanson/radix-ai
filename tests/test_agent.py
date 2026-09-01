import pytest

from fakes import ScriptedClient
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


@tool
def verbose() -> str:
    """Return a long string."""
    return "x" * 50


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
        on_activity=lambda name, text: activity.append((name, text)),
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


def test_max_tool_output_chars_truncates_tool_output():
    agent, client = make_agent(
        [
            ChatResult(
                tool_calls=[ToolCall(id="c1", name="verbose", raw_arguments="{}")]
            ),
            ChatResult(content="done"),
        ],
        tools=[verbose],
        max_tool_output_chars=10,
    )
    assert agent.run("go") == "done"
    output = client.stream_calls[1]["messages"][3]["content"]
    assert output.startswith("x" * 10)
    assert len(output) < 50
    assert "[output truncated]" in output


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


def test_compact_noop_without_context_or_state():
    agent, _ = make_agent([ChatResult(content="hi")])
    agent.run("hello")

    assert agent.compact() is False
