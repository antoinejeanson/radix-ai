from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import APIStatusError, OpenAI

from .messages import ChatResult, Message, ToolCall, Usage

DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_API_KEY = "radix"
DEFAULT_MODEL = "radix"


def _to_usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        prompt = raw.get("prompt_tokens", 0)
        completion = raw.get("completion_tokens", 0)
        total = raw.get("total_tokens", 0)
    else:
        prompt = getattr(raw, "prompt_tokens", 0)
        completion = getattr(raw, "completion_tokens", 0)
        total = getattr(raw, "total_tokens", 0)
    return Usage(
        prompt_tokens=int(prompt or 0),
        completion_tokens=int(completion or 0),
        total_tokens=int(total or 0),
    )


class ChatStream:
    """Streams text deltas from a chat completion.

    Iterate to receive text deltas as they arrive. After the iterator is
    exhausted, `result` holds the complete response, including any tool calls.
    """

    def __init__(
        self,
        api: OpenAI,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> None:
        self._api = api
        self._model = model
        self._messages = messages
        self._tools = tools
        self._result: ChatResult | None = None

    def _open(self, kwargs: dict[str, Any]) -> Iterator[Any]:
        try:
            return self._api.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            if exc.status_code == 400 and "stream_options" in str(exc):
                kwargs.pop("stream_options", None)
                return self._api.chat.completions.create(**kwargs)
            raise

    def __iter__(self) -> Iterator[str]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self._tools:
            kwargs["tools"] = self._tools

        content_parts: list[str] = []
        call_slots: dict[int, dict[str, str]] = {}
        raw_usage: Any = None

        for chunk in self._open(kwargs):
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                raw_usage = chunk_usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
                yield delta.content
            for tc in delta.tool_calls or []:
                slot = call_slots.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

        tool_calls = [
            ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                raw_arguments=slot["arguments"],
            )
            for index, slot in sorted(call_slots.items())
            if slot["name"]
        ]
        self._result = ChatResult(
            content="".join(content_parts), tool_calls=tool_calls, usage=_to_usage(raw_usage)
        )

    @property
    def result(self) -> ChatResult:
        if self._result is None:
            raise RuntimeError("stream has not been consumed yet")
        return self._result


class Client:
    """Thin wrapper over any OpenAI-compatible chat completions API.

    Defaults point at a local llama.cpp server (`llama-server`), but any
    OpenAI-compatible endpoint works (vLLM, Ollama, etc.).
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        timeout: float = 600.0,
        openai_client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self._api = openai_client or OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> ChatResult:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = self._api.chat.completions.create(**kwargs)
        message = response.choices[0].message
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                raw_arguments=tc.function.arguments or "",
            )
            for tc in (message.tool_calls or [])
        ]
        return ChatResult(
            content=message.content or "",
            tool_calls=tool_calls,
            usage=_to_usage(getattr(response, "usage", None)),
        )

    def chat_stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> ChatStream:
        return ChatStream(self._api, model=self.model, messages=messages, tools=tools)
