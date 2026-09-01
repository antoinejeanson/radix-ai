import pytest

from fakes import ScriptedClient
from radix import Agent, AutoApproveGate, Coordinator, Events
from radix.messages import ChatResult, ToolCall


def make_setup(results):
    client = ScriptedClient(results)
    coder = Agent(
        "coder",
        description="Writes code.",
        system_prompt="you write code",
        client=client,
    )
    coordinator = Coordinator(
        agents=[coder],
        client=client,
        permission_gate=AutoApproveGate(),
    )
    return coordinator, coder, client


def test_delegation_tools_generated():
    coordinator, _, _ = make_setup([])
    assert [t.name for t in coordinator.tools] == ["ask_coder"]
    schema = coordinator.tools[0].schema()
    assert schema["function"]["parameters"]["required"] == ["task"]
    assert "Writes code." in schema["function"]["description"]
    assert coordinator.stateful is True


def test_duplicate_agent_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        Coordinator(agents=[Agent("a"), Agent("a")])


def test_delegation_flow_keeps_coordinator_context_clean():
    coordinator, _, client = make_setup(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="ask_coder",
                        raw_arguments='{"task": "write hello world"}',
                    )
                ]
            ),
            ChatResult(content="print('hello')"),
            ChatResult(content="Here is the code: print('hello')"),
        ]
    )
    answer = coordinator.run("write hello world in python")
    assert answer == "Here is the code: print('hello')"

    sub_messages = client.stream_calls[1]["messages"]
    assert sub_messages[0]["content"] == "you write code"
    assert sub_messages[1]["content"] == "write hello world"

    final_messages = client.stream_calls[2]["messages"]
    tool_result = final_messages[-1]
    assert tool_result["role"] == "tool"
    assert tool_result["content"] == "print('hello')"


def test_history_only_user_and_final_answer():
    coordinator, _, _ = make_setup(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="ask_coder", raw_arguments='{"task": "t"}')
                ]
            ),
            ChatResult(content="sub answer"),
            ChatResult(content="final answer"),
        ]
    )
    coordinator.run("first question")
    assert [m["role"] for m in coordinator.history] == ["user", "assistant"]
    assert coordinator.history[0]["content"] == "first question"
    assert coordinator.history[1]["content"] == "final answer"

    coordinator.reset()
    assert coordinator.history == []


def test_events_propagate_from_subagent():
    coordinator, _, _ = make_setup(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="ask_coder", raw_arguments='{"task": "t"}')
                ]
            ),
            ChatResult(content="sub answer"),
            ChatResult(content="final answer"),
        ]
    )
    starts: list[str] = []
    deltas: list[tuple[str, str]] = []
    events = Events(
        on_start=starts.append,
        on_delta=lambda name, text: deltas.append((name, text)),
    )
    coordinator.run("do the thing", events=events)
    assert starts == ["coordinator", "coder", "coordinator"]
    assert ("coder", "sub answer") in deltas
    assert ("coordinator", "final answer") in deltas
    assert coordinator._events is None


def test_history_sent_on_second_turn():
    coordinator, _, client = make_setup(
        [ChatResult(content="hi"), ChatResult(content="bye")]
    )
    coordinator.run("hello")
    coordinator.run("goodbye")
    messages = client.stream_calls[1]["messages"]
    assert messages[1] == {"role": "user", "content": "hello"}
    assert messages[2] == {"role": "assistant", "content": "hi"}
    assert messages[3] == {"role": "user", "content": "goodbye"}
