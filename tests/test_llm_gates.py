from fakes import ScriptedClient
from radix import Agent, AutoApproveGate, CliPermissionGate, DenyGate, LlmAdvisoryGate, LlmAutoSafetyGate, tool
from radix.messages import ChatResult, ToolCall


@tool
def shell(command: str) -> str:
    """Run a command."""
    return f"ran: {command}"


@tool
def harmless(text: str) -> str:
    """Do nothing that requires permission."""
    return text


def gate_for(gate_cls, content: str, **kwargs):
    client = ScriptedClient([ChatResult(content=content)])
    lines: list[str] = []
    printer = kwargs.pop("printer", lines.append)
    gate = gate_cls(client, printer=printer, **kwargs)
    return gate, client, lines


def test_auto_gate_checks_every_tool():
    gate, client, _ = gate_for(LlmAutoSafetyGate, "SAFE\nread-only")
    assert gate.check(harmless, {"text": "hi"}) is True
    assert len(client.complete_calls) == 1


def test_auto_gate_approves_safe_call_without_prompt():
    prompts: list[str] = []
    gate, _, lines = gate_for(LlmAutoSafetyGate, "SAFE\nread-only listing", printer=lambda s: prompts.append(s))
    assert gate.check(shell, {"command": "ls"}) is True
    assert any("SAFE" in line and "read-only listing" in line for line in prompts)


def test_auto_gate_denies_dangerous_call():
    gate, _, lines = gate_for(LlmAutoSafetyGate, "DANGEROUS\nrecursively deletes files")
    assert gate.check(shell, {"command": "rm -rf /"}) is False
    assert any("DANGEROUS" in line for line in lines)


def test_auto_gate_denies_when_check_fails():
    class ExplodingClient:
        def complete(self, messages, tools=None):
            raise RuntimeError("connection refused")

    lines: list[str] = []
    gate = LlmAutoSafetyGate(ExplodingClient(), printer=lines.append)
    assert gate.check(shell, {"command": "ls"}) is False
    assert any("safety check failed" in line for line in lines)


def test_advisory_gate_shows_verdict_and_defers_to_user():
    prompts: list[str] = []
    client = ScriptedClient([ChatResult(content="SAFE\nread-only listing")])
    gate = LlmAdvisoryGate(
        client,
        base_gate=CliPermissionGate(input_fn=lambda p: prompts.append(p) or "y", printer=print),
        printer=lambda s: prompts.append("verdict:" + s),
    )
    assert gate.check(shell, {"command": "ls"}) is True
    assert any(line.startswith("verdict:• safety: SAFE") for line in prompts)
    assert "Allow this call? [y/N] " in prompts


def test_advisory_gate_user_can_reject_safe_call():
    client = ScriptedClient([ChatResult(content="SAFE\nread-only listing")])
    gate = LlmAdvisoryGate(
        client,
        base_gate=CliPermissionGate(input_fn=lambda p: "n", printer=lambda s: None),
        printer=lambda s: None,
    )
    assert gate.check(shell, {"command": "ls"}) is False


def test_advisory_gate_user_can_approve_dangerous_call():
    client = ScriptedClient([ChatResult(content="DANGEROUS\nwipes the disk")])
    gate = LlmAdvisoryGate(
        client,
        base_gate=CliPermissionGate(input_fn=lambda p: "y", printer=lambda s: None),
        printer=lambda s: None,
    )
    assert gate.check(shell, {"command": "rm -rf /"}) is True


def test_advisory_gate_checks_every_tool():
    gate, client, _ = gate_for(
        LlmAdvisoryGate,
        "DANGEROUS\nsubverts",
        base_gate=AutoApproveGate(),
    )
    assert gate.check(harmless, {"text": "hi"}) is True
    assert len(client.complete_calls) == 1


def test_agent_runs_tool_when_safety_check_says_safe():
    client = ScriptedClient(
        [
            ChatResult(tool_calls=[ToolCall(id="c1", name="shell", raw_arguments='{"command": "ls"}')]),
            ChatResult(content="SAFE\nread-only listing"),
            ChatResult(content="done"),
        ]
    )
    agent = Agent(
        "tester", system_prompt="be brief", tools=[shell], client=client,
        permission_gate=LlmAutoSafetyGate(client, printer=lambda s: None),
    )
    assert agent.run("list files") == "done"
    assert client.stream_calls[1]["messages"][3]["content"] == "ran: ls"


def test_agent_denies_tool_when_safety_check_says_dangerous():
    client = ScriptedClient(
        [
            ChatResult(tool_calls=[ToolCall(id="c1", name="shell", raw_arguments='{"command": "rm -rf /"}')]),
            ChatResult(content="DANGEROUS\nrecursively deletes files"),
            ChatResult(content="ok"),
        ]
    )
    agent = Agent(
        "tester", system_prompt="be brief", tools=[shell], client=client,
        permission_gate=LlmAutoSafetyGate(client, printer=lambda s: None),
    )
    assert agent.run("clean up") == "ok"
    assert "Permission denied" in client.stream_calls[1]["messages"][3]["content"]


def test_tool_gate_llm_safety_replaces_global_gate():
    client = ScriptedClient(
        [
            ChatResult(tool_calls=[ToolCall(id="c1", name="shell", raw_arguments='{"command": "ls"}')]),
            ChatResult(content="SAFE\nread-only listing"),
            ChatResult(content="done"),
        ]
    )
    agent = Agent(
        "tester", system_prompt="be brief", tools=[shell], client=client,
        permission_gate=DenyGate(),
        tool_gates={"shell": LlmAutoSafetyGate(client, printer=lambda s: None)},
    )
    assert agent.run("list files") == "done"
    assert client.stream_calls[1]["messages"][3]["content"] == "ran: ls"


def test_tool_gate_llm_safety_denies_call():
    client = ScriptedClient(
        [
            ChatResult(tool_calls=[ToolCall(id="c1", name="shell", raw_arguments='{"command": "rm -rf /"}')]),
            ChatResult(content="DANGEROUS\nwipes the disk"),
            ChatResult(content="ok"),
        ]
    )
    agent = Agent(
        "tester", system_prompt="be brief", tools=[shell], client=client,
        permission_gate=AutoApproveGate(),
        tool_gates={"shell": LlmAutoSafetyGate(client, printer=lambda s: None)},
    )
    assert agent.run("clean up") == "ok"
    assert "Permission denied" in client.stream_calls[1]["messages"][3]["content"]
