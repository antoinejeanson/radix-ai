import types

import httpx2
import pytest
from openai import APIStatusError

from radix import DEFAULT_BASE_URL, Client, Usage
from radix.client import ChatStream
from radix.messages import ChatResult


class FakeFunction:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class FakeToolCallDelta:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = FakeFunction(name, arguments)


class FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChunk:
    def __init__(self, delta=None, usage=None):
        self.choices = [types.SimpleNamespace(delta=delta)] if delta is not None else []
        self.usage = usage


class FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = FakeFunction(name, arguments)


class FakeCompletions:
    def __init__(self, chunks=None, message=None, usage=None):
        self.chunks = chunks or []
        self.message = message
        self.usage = usage
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return iter(self.chunks)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=self.message)], usage=self.usage
        )


def make_client(completions, model="test-model"):
    api = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    return Client(model=model, openai_client=api)


def text(s):
    return FakeChunk(FakeDelta(content=s))


def toolc(index, id=None, name=None, arguments=None):
    return FakeChunk(FakeDelta(tool_calls=[FakeToolCallDelta(index, id, name, arguments)]))


def test_default_base_url_targets_llama_cpp():
    assert DEFAULT_BASE_URL == "http://localhost:8080/v1"


def test_streaming_content():
    completions = FakeCompletions(chunks=[text("Hel"), text("lo"), FakeChunk()])
    client = make_client(completions)
    stream = client.chat_stream([{"role": "user", "content": "hi"}])
    assert list(stream) == ["Hel", "lo"]
    assert stream.result == ChatResult(content="Hello", tool_calls=[])
    assert completions.kwargs["model"] == "test-model"
    assert completions.kwargs["stream"] is True


def test_streaming_passes_tools():
    completions = FakeCompletions(chunks=[text("ok")])
    client = make_client(completions)
    tools = [{"type": "function", "function": {"name": "t"}}]
    stream = client.chat_stream([{"role": "user", "content": "hi"}], tools=tools)
    list(stream)
    assert completions.kwargs["tools"] == tools


def test_streaming_accumulates_tool_call_fragments():
    chunks = [
        text("thinking "),
        toolc(0, id="call_a", name="ad", arguments='{"a"'),
        toolc(0, name="d", arguments=": 1}"),
        toolc(1, id="call_b", name="search", arguments='{"q": "x"}'),
    ]
    client = make_client(FakeCompletions(chunks=chunks))
    stream = client.chat_stream([{"role": "user", "content": "hi"}])
    assert list(stream) == ["thinking "]
    result = stream.result
    assert result.content == "thinking "
    assert [(tc.id, tc.name, tc.raw_arguments) for tc in result.tool_calls] == [
        ("call_a", "add", '{"a": 1}'),
        ("call_b", "search", '{"q": "x"}'),
    ]


def test_stream_result_before_iteration_raises():
    client = make_client(FakeCompletions(chunks=[text("x")]))
    stream = client.chat_stream([{"role": "user", "content": "hi"}])
    with pytest.raises(RuntimeError):
        stream.result


def test_complete_parses_message_and_tool_calls():
    message = types.SimpleNamespace(
        content="done",
        tool_calls=[FakeToolCall("c1", "add", '{"a": 1}')],
    )
    client = make_client(FakeCompletions(message=message))
    result = client.complete([{"role": "user", "content": "hi"}])
    assert result.content == "done"
    assert result.tool_calls[0].id == "c1"
    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].raw_arguments == '{"a": 1}'


def test_stream_requests_include_usage():
    completions = FakeCompletions(chunks=[text("ok")])
    client = make_client(completions)
    list(client.chat_stream([{"role": "user", "content": "hi"}]))
    assert completions.kwargs["stream_options"] == {"include_usage": True}


def test_stream_captures_usage_chunk():
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunks = [text("hi"), FakeChunk(usage=usage)]
    client = make_client(FakeCompletions(chunks=chunks))
    stream = client.chat_stream([{"role": "user", "content": "hi"}])
    assert list(stream) == ["hi"]
    assert stream.result.usage == Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_stream_retries_when_stream_options_rejected():
    class RejectingCompletions(FakeCompletions):
        def create(self, **kwargs):
            if "stream_options" in kwargs:
                response = httpx2.Response(400, request=httpx2.Request("POST", "http://test"))
                raise APIStatusError("stream_options is not supported", response=response, body=None)
            return super().create(**kwargs)

    completions = RejectingCompletions(chunks=[text("ok")])
    client = make_client(completions)
    stream = client.chat_stream([{"role": "user", "content": "hi"}])
    assert list(stream) == ["ok"]
    assert "stream_options" not in completions.kwargs
    assert stream.result.usage is None


def test_stream_raises_other_400_errors():
    class AlwaysRejecting(FakeCompletions):
        def create(self, **kwargs):
            response = httpx2.Response(400, request=httpx2.Request("POST", "http://test"))
            raise APIStatusError("something else is wrong", response=response, body=None)

    client = make_client(AlwaysRejecting(chunks=[text("ok")]))
    stream = client.chat_stream([{"role": "user", "content": "hi"}])
    with pytest.raises(APIStatusError):
        list(stream)


def test_complete_captures_usage():
    usage = types.SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)
    message = types.SimpleNamespace(content="ok", tool_calls=None)
    client = make_client(FakeCompletions(message=message, usage=usage))
    result = client.complete([{"role": "user", "content": "hi"}])
    assert result.usage == Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7)


def test_complete_without_usage():
    message = types.SimpleNamespace(content="ok", tool_calls=None)
    client = make_client(FakeCompletions(message=message))
    assert client.complete([{"role": "user", "content": "hi"}]).usage is None
