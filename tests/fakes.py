from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from radix.messages import ChatResult


class FakeStream:
    def __init__(self, result: ChatResult) -> None:
        self._result = result
        self._deltas = [result.content] if result.content else []

    def __iter__(self) -> Iterator[str]:
        yield from self._deltas

    @property
    def result(self) -> ChatResult:
        return self._result


class ScriptedClient:
    """Test double for radix.Client: returns pre-scripted ChatResults in order."""

    def __init__(self, results: list[ChatResult]) -> None:
        self.results = list(results)
        self.stream_calls: list[dict[str, Any]] = []
        self.complete_calls: list[dict[str, Any]] = []

    def chat_stream(self, messages, tools=None) -> FakeStream:
        self.stream_calls.append({"messages": list(messages), "tools": tools})
        return FakeStream(self.results.pop(0))

    def complete(self, messages, tools=None) -> ChatResult:
        self.complete_calls.append({"messages": list(messages), "tools": tools})
        return self.results.pop(0)
