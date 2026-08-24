from rich.console import Console

from fakes import ScriptedClient
from radix import Agent, Assistant, AutoApproveGate
from radix.messages import ChatResult
from radix.repl import Repl


def make_assistant(results):
    client = ScriptedClient(results)
    return Assistant(client=client, agents=[Agent("coder")], permission_gate=AutoApproveGate())


def run_repl(assistant, inputs):
    console = Console(file=open("/dev/null", "w"), force_terminal=False)
    repl = Repl(assistant, console=console)
    repl._session = type("S", (), {"prompt": staticmethod(lambda *_a, **_k: inputs.pop(0))})()
    repl.run()
    return repl


def test_repl_chat_and_commands():
    assistant = make_assistant([ChatResult(content="answer one"), ChatResult(content="answer two")])
    run_repl(assistant, ["/help", "first", "/reset", "second", "/exit"])
    assert len(assistant.coordinator.history) == 2
    assert assistant.coordinator.history[0]["content"] == "second"


def test_repl_survives_errors():
    class BrokenClient:
        def chat_stream(self, *a, **k):
            raise ConnectionError("server down")

    assistant = make_assistant([])
    assistant.coordinator.client = BrokenClient()
    run_repl(assistant, ["hello", "/exit"])


def test_repl_eof_exits():
    def raise_eof(*_a, **_k):
        raise EOFError

    assistant = make_assistant([])
    console = Console(file=open("/dev/null", "w"), force_terminal=False)
    repl = Repl(assistant, console=console)
    repl._session = type("S", (), {"prompt": staticmethod(raise_eof)})()
    repl.run()
