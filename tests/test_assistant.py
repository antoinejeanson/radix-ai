from fakes import ScriptedClient
from radix import DEFAULT_MAX_TOOL_OUTPUT_CHARS
from radix import Agent, Assistant, AutoApproveGate
from radix.messages import ChatResult, ToolCall


def test_assistant_binds_subagents():
    client = ScriptedClient([ChatResult(content="hi")])
    coder = Agent("coder", tools=[])
    assistant = Assistant(client=client, agents=[coder], permission_gate=AutoApproveGate())

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
            ChatResult(tool_calls=[ToolCall(id="c1", name="ask_coder", raw_arguments='{"task": "t"}')]),
            ChatResult(content="sub says hi"),
            ChatResult(content="hi from coordinator"),
        ]
    )
    coder = Agent("coder", system_prompt="coder prompt")
    assistant = Assistant(client=client, agents=[coder], permission_gate=AutoApproveGate())
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
        client=client, agents=[coder], permission_gate=AutoApproveGate(), max_tool_output_chars=555
    )
    assert coder.max_tool_output_chars == 77
    assert assistant.coordinator.max_tool_output_chars == 555
