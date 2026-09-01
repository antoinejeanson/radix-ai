from fakes import ScriptedClient
from radix import LlmSafetyChecker, tool
from radix.messages import ChatResult


@tool
def run(command: str) -> str:
    """Run a command."""
    return command


def make_checker(content: str, **kwargs):
    client = ScriptedClient([ChatResult(content=content)])
    return LlmSafetyChecker(client, **kwargs), client


def test_safe_verdict():
    checker, _ = make_checker("SAFE\nread-only listing command")
    verdict = checker.check(run, {"command": "ls"})
    assert verdict.safe is True
    assert verdict.checked is True
    assert verdict.reason == "read-only listing command"


def test_dangerous_verdict():
    checker, _ = make_checker("DANGEROUS\nrecursively deletes files")
    verdict = checker.check(run, {"command": "rm -rf /"})
    assert verdict.safe is False
    assert verdict.reason == "recursively deletes files"


def test_lowercase_verdict():
    checker, _ = make_checker("safe\nit only lists files")
    assert checker.check(run, {"command": "ls"}).safe is True


def test_not_safe_is_dangerous():
    checker, _ = make_checker("not safe\nthis deletes the database")
    assert checker.check(run, {"command": "drop db"}).safe is False


def test_dangerous_wins_over_safe():
    checker, _ = make_checker("DANGEROUS\nbut overall a safe command")
    assert checker.check(run, {"command": "x"}).safe is False


def test_safety_word_does_not_count():
    checker, _ = make_checker("SAFETY first, then we decide")
    verdict = checker.check(run, {"command": "x"})
    assert verdict.safe is False
    assert "unparseable" in verdict.reason


def test_unparseable_fails_closed():
    checker, _ = make_checker("I am not sure what to say here.")
    verdict = checker.check(run, {"command": "x"})
    assert verdict.safe is False
    assert "unparseable" in verdict.reason


def test_empty_answer_fails_closed():
    checker, _ = make_checker("")
    verdict = checker.check(run, {"command": "x"})
    assert verdict.safe is False
    assert "unparseable" in verdict.reason


def test_client_error_fails_closed():
    class ExplodingClient:
        def complete(self, messages, tools=None):
            raise RuntimeError("connection refused")

    checker = LlmSafetyChecker(ExplodingClient())
    verdict = checker.check(run, {"command": "ls"})
    assert verdict.safe is False
    assert verdict.checked is False
    assert "safety check failed" in verdict.reason


def test_describe_contains_tool_and_arguments():
    checker, _ = make_checker("SAFE\nok")
    assert checker.describe(run, {"command": "ls -la"}) == 'Tool: run\nArguments: {"command": "ls -la"}'


def test_describe_truncates_long_arguments():
    checker, _ = make_checker("SAFE\nok", argument_char_limit=20)
    description = checker.describe(run, {"command": "x" * 100})
    assert "..." in description
    assert len(description) < 60


def test_checker_sends_prompt_and_description():
    checker, client = make_checker("SAFE\nok")
    checker.check(run, {"command": "ls"})
    call = client.complete_calls[0]
    assert call["messages"][0]["role"] == "system"
    assert "safety reviewer" in call["messages"][0]["content"]
    assert call["messages"][1]["role"] == "user"
    assert "Tool: run" in call["messages"][1]["content"]
    assert 'ls' in call["messages"][1]["content"]
