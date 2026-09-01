from radix import AutoApproveGate, CliPermissionGate, DenyGate, tool


@tool
def run(command: str) -> str:
    """Run a command."""
    return command


@tool
def read(path: str) -> str:
    """Read a file."""
    return path


def test_cli_gate_prompts_on_every_tool():
    prompts: list[str] = []
    printed: list[str] = []
    gate = CliPermissionGate(
        input_fn=lambda p: prompts.append(p) or "y",
        printer=printed.append,
    )
    assert gate.check(read, {"path": "/tmp/x"}) is True
    assert prompts == ["Allow this call? [y/N] "]
    assert any("read" in line for line in printed)
    assert any("/tmp/x" in line for line in printed)


def test_cli_gate_prompts_on_dangerous_tool():
    prompts: list[str] = []
    printed: list[str] = []
    gate = CliPermissionGate(
        input_fn=lambda p: prompts.append(p) or "y",
        printer=printed.append,
    )
    assert gate.check(run, {"command": "ls"}) is True
    assert prompts == ["Allow this call? [y/N] "]
    assert any("run" in line for line in printed)
    assert any("ls" in line for line in printed)


def test_cli_gate_rejects_on_no():
    gate = CliPermissionGate(input_fn=lambda p: "n", printer=lambda s: None)
    assert gate.check(run, {"command": "ls"}) is False
    gate = CliPermissionGate(input_fn=lambda p: "", printer=lambda s: None)
    assert gate.check(run, {"command": "ls"}) is False


def test_auto_approve_and_deny_gates():
    assert AutoApproveGate().check(run, {}) is True
    assert DenyGate().check(read, {}) is False
