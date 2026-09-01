from fakes import ScriptedClient
from radix.context import ContextManager, estimate_tokens
from radix.messages import ChatResult, assistant_message, system_message, user_message


def build_messages(n_body=8, size=100):
    messages = [system_message("sys " + "x" * size)]
    for i in range(n_body):
        role_msg = (
            user_message(f"msg{i} " + "y" * size)
            if i % 2 == 0
            else assistant_message(f"ans{i} " + "z" * size)
        )
        messages.append(role_msg)
    return messages


def make_manager(results, budget=150):
    client = ScriptedClient(results)
    cm = ContextManager(
        client, max_context_tokens=budget, reserve_output_tokens=0, keep_recent=2
    )
    return cm, client


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("x" * 400) == 100


def test_prepare_under_budget_is_unchanged():
    cm, client = make_manager([], budget=100000)
    messages = build_messages()
    assert cm.prepare(messages) == messages
    assert client.complete_calls == []


def test_prepare_compacts_with_summary():
    cm, client = make_manager([ChatResult(content="SUMMARY OF OLD")])
    messages = build_messages()
    assert cm.message_tokens(messages) > cm.budget

    out = cm.prepare(messages)

    assert len(client.complete_calls) == 1
    assert out[0] == messages[0]
    assert out[1]["role"] == "system"
    assert out[1]["content"] == "[Summary of the earlier conversation]\nSUMMARY OF OLD"
    assert out[2:] == messages[-2:]
    assert cm.message_tokens(out) <= cm.budget


def test_prepare_fallback_when_summary_fails():
    cm, _ = make_manager([])
    out = cm.prepare(build_messages())
    assert "dropped to fit the context window" in out[1]["content"]
    assert out[2:] == build_messages()[-2:]


def test_prepare_keeps_everything_when_body_too_small_to_compact():
    cm, _ = make_manager([], budget=10)
    messages = [system_message("s"), user_message("u")]
    assert cm.prepare(messages) == messages


def test_custom_summary_prompt_and_transcript_limit():
    client = ScriptedClient([ChatResult(content="S")])
    cm = ContextManager(
        client,
        max_context_tokens=150,
        reserve_output_tokens=0,
        keep_recent=2,
        summary_prompt="my custom prompt",
        transcript_char_limit=10,
    )
    out = cm.prepare(build_messages())
    call = client.complete_calls[0]["messages"]
    assert call[0] == {"role": "system", "content": "my custom prompt"}
    assert len(call[1]["content"]) <= 10
    assert out[1]["content"] == "[Summary of the earlier conversation]\nS"


def test_custom_fallback_summary():
    cm, _ = make_manager([], budget=150)
    cm.fallback_summary = "custom fallback"
    out = cm.prepare(build_messages())
    assert "custom fallback" in out[1]["content"]
