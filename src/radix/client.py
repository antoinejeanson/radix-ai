from __future__ import annotations

import random
import time
from collections.abc import Iterator
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI

from .messages import ChatResult, Message, ToolCall, Usage

# Client: thin wrapper over any OpenAI-compatible chat completions API, with
# streaming (ChatStream) and non-streaming (complete) requests.
DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_API_KEY = "radix"
DEFAULT_MODEL = "radix"


def _to_usage(raw: Any) -> Usage | None:
    """Normalize a server usage object or dict into a `Usage`.

    Args:
        raw: Usage payload from the API client (an object with
            prompt_tokens/completion_tokens/total_tokens attributes), a
            matching dict, or None.

    Returns:
        A Usage, or None when `raw` is None.
    """
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


# HTTP status codes worth retrying: request timeouts and server-side
# failures. Other 4xx (bad request, auth, not found) are not retryable.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException) -> bool:
    """Whether a request error is a transient failure worth retrying.

    Args:
        exc: The exception raised by the API client.

    Returns:
        True for connection errors and retryable status codes, else False.
    """
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS
    return False


def _backoff_delay(base: float, attempt: int) -> float:
    """Exponential backoff with a little jitter, in seconds.

    Args:
        base: The base delay in seconds.
        attempt: The zero-based retry attempt number.

    Returns:
        The delay to sleep before the next attempt.
    """
    return base * (2**attempt) + random.uniform(0, 0.1)


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
        retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        """Stream a completion. Prefer `Client.chat_stream()` over building
        this directly.

        Args:
            api: The openai client that performs the network request.
            model: Model id to use for the completion.
            messages: The conversation, in OpenAI chat format.
            tools: Tool schemas offered to the model, or None.
            retries: How many times to retry transient failures before the
                first chunk is yielded.
            backoff_base: Base delay (seconds) for the exponential backoff
                between retries.
        """
        self._api = api
        self._model = model
        self._messages = messages
        self._tools = tools
        self._retries = retries
        self._backoff_base = backoff_base
        self._result: ChatResult | None = None

    def _open(self, kwargs: dict[str, Any]) -> Iterator[Any]:
        """Open the streaming response, retrying without `stream_options`
        when the server rejects them (some OpenAI-compatible endpoints do).

        Args:
            kwargs: The completion request arguments.

        Yields:
            Response chunks from the API.

        Raises:
            APIStatusError: for any non-400 error, or a 400 unrelated to
                `stream_options`.
        """
        try:
            return self._api.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            if exc.status_code == 400 and "stream_options" in str(exc):
                kwargs.pop("stream_options", None)
                return self._api.chat.completions.create(**kwargs)
            raise

    def __iter__(self) -> Iterator[str]:
        """Iterate to receive text deltas as they stream in.

        After the iterator is exhausted, read `.result` for the complete
        response, tool calls and usage.

        Yields:
            Each text delta as a string.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self._tools:
            kwargs["tools"] = self._tools

        # Retry the whole open-and-iterate, but only while nothing has been
        # yielded yet: once tokens are flowing the request can't be resumed,
        # so a mid-stream failure is re-raised instead.
        attempt = 0
        while True:
            content_parts: list[str] = []
            call_slots: dict[int, dict[str, str]] = {}
            raw_usage: Any = None
            yielded = False
            try:
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
                        yielded = True
                        yield delta.content
                    for tc in delta.tool_calls or []:
                        slot = call_slots.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                slot["name"] += tc.function.name
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments
                break
            except (APIConnectionError, APIStatusError) as exc:
                if yielded or not _is_retryable(exc) or attempt >= self._retries:
                    raise
                time.sleep(_backoff_delay(self._backoff_base, attempt))
                attempt += 1

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
            content="".join(content_parts),
            tool_calls=tool_calls,
            usage=_to_usage(raw_usage),
        )

    @property
    def result(self) -> ChatResult:
        """The complete response once the stream has been consumed.

        Returns:
            The ChatResult with content, tool calls and usage.

        Raises:
            RuntimeError: if accessed before the stream was fully iterated.
        """
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
        retries: int = 3,
        backoff_base: float = 0.5,
        openai_client: OpenAI | None = None,
    ) -> None:
        """Connect to an OpenAI-compatible chat completions API.

        Args:
            model: Model id sent with every request.
            base_url: API base URL. The default targets a local llama.cpp
                server (`llama-server`); any OpenAI-compatible endpoint
                works (vLLM, Ollama, OpenRouter, ...).
            api_key: API key for the endpoint. llama.cpp accepts anything;
                hosted APIs require a real one.
            timeout: HTTP timeout in seconds for the underlying client.
            retries: How many times to retry a request after a transient
                failure (connection error, 408/429/5xx). 0 disables retry.
            backoff_base: Base delay (seconds) for the exponential backoff
                between retries.
            openai_client: Pre-built openai client to use instead of
                constructing one from base_url, api_key and timeout.
        """
        self.model = model
        self.retries = retries
        self.backoff_base = backoff_base
        self._api = openai_client or OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout
        )

    def _create_with_retry(self, kwargs: dict[str, Any]):
        """Call the completions API, retrying transient failures.

        Args:
            kwargs: The completion request arguments.

        Returns:
            The API response.

        Raises:
            The last error when retries are exhausted, or any non-retryable
                error immediately.
        """
        attempt = 0
        while True:
            try:
                return self._api.chat.completions.create(**kwargs)
            except (APIConnectionError, APIStatusError) as exc:
                if not _is_retryable(exc) or attempt >= self.retries:
                    raise
                time.sleep(_backoff_delay(self.backoff_base, attempt))
                attempt += 1

    def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> ChatResult:
        """Run one non-streaming chat completion and return the full result.

        Args:
            messages: The conversation, in OpenAI chat format.
            tools: Tool schemas offered to the model, or None.

        Returns:
            The complete ChatResult with content, tool calls and usage.
        """
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = self._create_with_retry(kwargs)
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
        """Start a streaming chat completion.

        Args:
            messages: The conversation, in OpenAI chat format.
            tools: Tool schemas offered to the model, or None.

        Returns:
            A ChatStream: iterate for text deltas, then read `.result`.
        """
        return ChatStream(
            self._api,
            model=self.model,
            messages=messages,
            tools=tools,
            retries=self.retries,
            backoff_base=self.backoff_base,
        )
