from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .messages import Message

if TYPE_CHECKING:
    from .client import Client

SUMMARY_PROMPT = (
    "Summarize the conversation transcript below concisely. Keep all facts, decisions, "
    "results and pending tasks. Reply with the summary only."
)
FALLBACK_SUMMARY = "(earlier messages were dropped to fit the context window)"
TRANSCRIPT_CHAR_LIMIT = 12000
DEFAULT_MAX_CONTEXT_TOKENS = 8192
DEFAULT_RESERVE_OUTPUT_TOKENS = 2048
DEFAULT_KEEP_RECENT = 4


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars per token) without a tokenizer dependency."""
    return max(1, (len(text) + 3) // 4)


class ContextManager:
    """Keeps message lists within a token budget for small context windows.

    When a conversation exceeds the budget, older messages are summarized by
    the model into a single message. If summarization fails, they are dropped.
    The system prompt and the most recent messages are always kept.
    """

    def __init__(
        self,
        client: Client,
        *,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        reserve_output_tokens: int = DEFAULT_RESERVE_OUTPUT_TOKENS,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        summary_prompt: str = SUMMARY_PROMPT,
        fallback_summary: str = FALLBACK_SUMMARY,
        transcript_char_limit: int = TRANSCRIPT_CHAR_LIMIT,
        token_estimator: Callable[[str], int] | None = None,
    ) -> None:
        self.client = client
        self.budget = max_context_tokens - reserve_output_tokens
        self.keep_recent = keep_recent
        self.summary_prompt = summary_prompt
        self.fallback_summary = fallback_summary
        self.transcript_char_limit = transcript_char_limit
        self._estimate = token_estimator or estimate_tokens

    def message_tokens(self, messages: list[Message]) -> int:
        total = 0
        for message in messages:
            total += 4
            for value in message.values():
                if isinstance(value, str):
                    total += self._estimate(value)
        return total

    def prepare(self, messages: list[Message]) -> list[Message]:
        if self.message_tokens(messages) <= self.budget:
            return messages
        return self._compact(messages)

    def _compact(self, messages: list[Message]) -> list[Message]:
        system = messages[0] if messages and messages[0].get("role") == "system" else None
        body = messages[1:] if system else list(messages)
        if len(body) <= self.keep_recent:
            return messages

        old, recent = body[: -self.keep_recent], body[-self.keep_recent :]
        summary = self._summarize(old)
        summary_message: Message = {
            "role": "system",
            "content": f"[Summary of the earlier conversation]\n{summary}",
        }

        def build(kept: list[Message]) -> list[Message]:
            prefix = [system] if system else []
            return prefix + [summary_message] + kept

        recent = list(recent)
        result = build(recent)
        while self.message_tokens(result) > self.budget and len(recent) > 1:
            recent.pop(0)
            result = build(recent)
        return result

    def _summarize(self, messages: list[Message]) -> str:
        lines = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                lines.append(f"{message.get('role')}: {content}")
        if not lines:
            return self.fallback_summary
        transcript = "\n".join(lines)[: self.transcript_char_limit]
        try:
            result = self.client.complete(
                [
                    {"role": "system", "content": self.summary_prompt},
                    {"role": "user", "content": transcript},
                ]
            )
            return result.content.strip() or self.fallback_summary
        except Exception:
            return self.fallback_summary
