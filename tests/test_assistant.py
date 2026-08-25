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
    assert assistant.coordinator.client is client
    assert [t.name for t in assistant.coordinator.delegation_tools(0)] == ["ask_coder"]


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


def test_assistant_binds_nested_and_cyclic_subagents():
    client = ScriptedClient([ChatResult(content="hi")])
    leaf = Agent("leaf", tools=[])
    helper = Agent("helper", tools=[], subagents=[leaf])
    coder = Agent("coder", tools=[], subagents=[helper])
    helper.subagents.append(coder)  # cycle: helper -> coder -> helper
    assistant = Assistant(agents=[coder], client=client, permission_gate=AutoApproveGate())

    for agent in (coder, helper, leaf):
        assert agent.client is client
        assert agent.context is assistant.context
        assert agent.permission_gate is assistant.permission_gate
        assert agent.max_delegation_depth == 2


def test_assistant_max_delegation_depth_threads_down():
    client = ScriptedClient([ChatResult(content="hi")])
    nested = Agent("nested", tools=[])
    coder = Agent("coder", tools=[], subagents=[nested])
    assistant = Assistant(
        client=client,
        agents=[coder],
        permission_gate=AutoApproveGate(),
        max_delegation_depth=7,
    )
    assert coder.max_delegation_depth == 7
    assert nested.max_delegation_depth == 7
    assert assistant.coordinator.max_delegation_depth == 7
