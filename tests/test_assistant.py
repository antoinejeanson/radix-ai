from fakes import ScriptedClient
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
