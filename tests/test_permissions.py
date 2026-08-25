from radix import AutoApproveGate, CliPermissionGate, DenyGate, tool


@tool(ask_permission=True)
def dangerous(command: str) -> str:
    """Run something dangerous."""
    return command


@tool
def harmless(text: str) -> str:
    """Do nothing that requires permission."""
    return text


def test_cli_gate_auto_approves_non_ask_permission():
    def explode(prompt: str) -> str:
        raise AssertionError("should not prompt for non-ask_permission tools")

    gate = CliPermissionGate(input_fn=explode)
    assert gate.check(harmless, {"text": "hi"}) is True


def test_cli_gate_prompts_on_ask_permission_yes():
    prompts: list[str] = []
    printed: list[str] = []
    gate = CliPermissionGate(
        input_fn=lambda p: prompts.append(p) or "y",
        printer=printed.append,
    )
    assert gate.check(dangerous, {"command": "ls"}) is True
    assert prompts == ["Allow this call? [y/N] "]
    assert any("dangerous" in line for line in printed)
    assert any("ls" in line for line in printed)


def test_cli_gate_prompts_on_ask_permission_no():
    gate = CliPermissionGate(input_fn=lambda p: "n", printer=lambda s: None)
    assert gate.check(dangerous, {"command": "ls"}) is False
    gate = CliPermissionGate(input_fn=lambda p: "", printer=lambda s: None)
    assert gate.check(dangerous, {"command": "ls"}) is False


def test_auto_approve_and_deny_gates():
    assert AutoApproveGate().check(dangerous, {}) is True
    assert DenyGate().check(harmless, {}) is False
