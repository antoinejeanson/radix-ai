from fakes import ScriptedClient
from radix import (
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    Agent,
    Assistant,
    AutoApproveGate,
    DenyGate,
    tool,
)
from radix.messages import ChatResult, ToolCall


@tool
def fetch_url(url: str) -> str:
    """Fetch a web page."""
    return f"fetched: {url}"


def test_assistant_binds_subagents():
    client = ScriptedClient([ChatResult(content="hi")])
    coder = Agent("coder", tools=[])
    assistant = Assistant(
        client=client, agents=[coder], permission_gate=AutoApproveGate()
    )

    assert coder.client is client
    assert coder.context is assistant.context
    assert coder.permission_gate is assistant.permission_gate
    assert coder.pre_tool_hook is not None
    assert coder.max_tool_output_chars == DEFAULT_MAX_TOOL_OUTPUT_CHARS
    assert assistant.coordinator.pre_tool_hook is not None
    assert assistant.coordinator.client is client
    assert [t.name for t in assistant.coordinator.tools] == ["ask_coder"]


def test_assistant_chat_end_to_end():
    client = ScriptedClient(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="ask_coder", raw_arguments='{"task": "t"}')
                ]
            ),
            ChatResult(content="sub says hi"),
            ChatResult(content="hi from coordinator"),
        ]
    )
    coder = Agent("coder", system_prompt="coder prompt")
    assistant = Assistant(
        client=client, agents=[coder], permission_gate=AutoApproveGate()
    )
    assert assistant.chat("hello") == "hi from coordinator"
    assert assistant.coordinator.history[1]["content"] == "hi from coordinator"
    assistant.reset()
    assert assistant.coordinator.history == []


def test_assistant_knobs_propagate():
    client = ScriptedClient([ChatResult(content="hi")])
    coder = Agent("coder", tools=[])
    assistant = Assistant(
        client=client,
        agents=[coder],
        permission_gate=AutoApproveGate(),
        keep_recent=7,
        summary_prompt="sp",
        fallback_summary="fs",
        transcript_char_limit=123,
        max_tool_rounds=3,
        max_tool_output_chars=555,
    )
    assert assistant.context.keep_recent == 7
    assert assistant.context.summary_prompt == "sp"
    assert assistant.context.fallback_summary == "fs"
    assert assistant.context.transcript_char_limit == 123
    assert assistant.coordinator.max_tool_rounds == 3
    assert assistant.coordinator.max_tool_output_chars == 555
    assert coder.max_tool_output_chars == DEFAULT_MAX_TOOL_OUTPUT_CHARS


def test_subagent_keeps_explicit_max_tool_output_chars():
    client = ScriptedClient([ChatResult(content="hi")])
    coder = Agent("coder", tools=[], max_tool_output_chars=77)
    assistant = Assistant(
        client=client,
        agents=[coder],
        permission_gate=AutoApproveGate(),
        max_tool_output_chars=555,
    )
    assert coder.max_tool_output_chars == 77
    assert assistant.coordinator.max_tool_output_chars == 555


def test_tool_gates_propagate_to_coordinator_and_subagents():
    client = ScriptedClient([ChatResult(content="hi")])
    coder = Agent("coder", tools=[])
    gates = {"fetch_url": DenyGate()}
    assistant = Assistant(
        client=client, agents=[coder], tools=[fetch_url], tool_gates=gates
    )
    assert assistant.coordinator.tool_gates is gates
    assert coder.tool_gates is gates


def test_subagent_explicit_permission_gate_is_kept():
    client = ScriptedClient([ChatResult(content="hi")])
    coder = Agent("coder", tools=[], permission_gate=DenyGate())
    assistant = Assistant(
        client=client, agents=[coder], permission_gate=AutoApproveGate()
    )
    assert coder.permission_gate is not assistant.permission_gate
    assert isinstance(coder.permission_gate, DenyGate)


def test_subagent_explicit_tool_gates_are_kept():
    client = ScriptedClient([ChatResult(content="hi")])
    own = {"fetch_url": DenyGate()}
    coder = Agent("coder", tools=[fetch_url], tool_gates=own)
    Assistant(
        client=client,
        agents=[coder],
        tools=[fetch_url],
        tool_gates={"fetch_url": AutoApproveGate()},
    )
    assert coder.tool_gates is own


def test_subagent_explicit_gates_are_enforced():
    client = ScriptedClient(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c1", name="ask_researcher", raw_arguments='{"task": "t"}'
                    )
                ]
            ),
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c2", name="fetch_url", raw_arguments='{"url": "http://x"}'
                    )
                ]
            ),
            ChatResult(content="sub answer"),
            ChatResult(content="coordinator answer"),
        ]
    )
    researcher = Agent(
        "researcher", tools=[fetch_url], tool_gates={"fetch_url": DenyGate()}
    )
    assistant = Assistant(
        client=client,
        agents=[researcher],
        permission_gate=AutoApproveGate(),
        tool_gates={"fetch_url": AutoApproveGate()},
    )
    assert assistant.chat("research") == "coordinator answer"
    sub_round = client.stream_calls[2]
    assert "Permission denied" in sub_round["messages"][2]["content"]


def test_tool_gates_reject_unknown_tool_names():
    client = ScriptedClient([ChatResult(content="hi")])
    try:
        Assistant(client=client, tool_gates={"definitely_not_a_tool": DenyGate()})
    except ValueError as exc:
        assert "definitely_not_a_tool" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown tool gate name")


def test_tool_gates_apply_to_subagent_tools():
    client = ScriptedClient(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c1", name="ask_researcher", raw_arguments='{"task": "t"}'
                    )
                ]
            ),
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c2", name="fetch_url", raw_arguments='{"url": "http://x"}'
                    )
                ]
            ),
            ChatResult(content="sub answer"),
            ChatResult(content="coordinator answer"),
        ]
    )
    researcher = Agent("researcher", tools=[fetch_url])
    assistant = Assistant(
        client=client,
        agents=[researcher],
        permission_gate=AutoApproveGate(),
        tool_gates={"fetch_url": DenyGate()},
    )
    assert assistant.chat("research") == "coordinator answer"
    sub_round = client.stream_calls[2]
    assert "Permission denied" in sub_round["messages"][2]["content"]
