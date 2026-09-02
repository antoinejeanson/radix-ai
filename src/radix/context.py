from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .messages import Message

if TYPE_CHECKING:
    from .client import Client

# ContextManager: keeps a conversation within a token budget by summarizing
# (compacting) the oldest messages when it overflows.
SUMMARY_PROMPT = (
    "Summarize the conversation transcript below concisely. Keep all facts, decisions, "
    "results and pending tasks. Reply with the summary only."
)
FALLBACK_SUMMARY = "(earlier messages were dropped to fit the context window)"
DEFAULT_MAX_CONTEXT_TOKENS = 8192
DEFAULT_RESERVE_OUTPUT_TOKENS = 2048
DEFAULT_KEEP_RECENT = 4


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars per token) without a tokenizer
    dependency.

    Args:
        text: The text to estimate.

    Returns:
        An approximate token count, at least 1.
    """
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
        token_estimator: Callable[[str], int] | None = None,
    ) -> None:
        """Create a context manager with a token budget.

        Args:
            client: Client used to summarize old messages during compaction.
            max_context_tokens: Maximum tokens the full conversation may
                use; compaction kicks in beyond budget - reserve_output_tokens.
            reserve_output_tokens: Tokens reserved for the model's answer;
                the conversation budget is
                max_context_tokens - reserve_output_tokens.
            keep_recent: Number of most recent messages kept verbatim when
                compacting; everything older is summarized.
            summary_prompt: System prompt for the summarization completion.
            fallback_summary: Message used when summarization fails or the
                model returns nothing usable.
            token_estimator: Callable mapping text to an estimated token
                count; defaults to `estimate_tokens` (~4 chars/token).
        """
        self.client = client
        self.budget = max_context_tokens - reserve_output_tokens
        self.keep_recent = keep_recent
        self.summary_prompt = summary_prompt
        self.fallback_summary = fallback_summary
        self._estimate = token_estimator or estimate_tokens

    def message_tokens(self, messages: list[Message]) -> int:
        """Estimate how many tokens a message list uses.

        Args:
            messages: Message list in OpenAI chat format.

        Returns:
            Estimated token count: 4 per message plus the estimate of every
            string value in it.
        """
        total = 0
        for message in messages:
            total += 4
            for value in message.values():
                if isinstance(value, str):
                    total += self._estimate(value)
        return total

    def prepare(self, messages: list[Message]) -> list[Message]:
        """Return a message list within budget, compacting when needed.

        Args:
            messages: The conversation to check.

        Returns:
            The same list when within budget, else `compact`'s result.
        """
        if self.message_tokens(messages) <= self.budget:
            return messages
        return self.compact(messages)

    def compact(self, messages: list[Message]) -> list[Message]:
        """Compact a conversation eagerly, summarizing the oldest messages.

        This is `prepare` without the budget check: the oldest messages are
        summarized even when the list is still within budget. The leading
        system prompt (if any) and the `keep_recent` newest messages stay
        verbatim, and the result is trimmed to fit the budget when the
        summary is not enough.

        Args:
            messages: The conversation to compact.

        Returns:
            A new, smaller message list — or the same list when there is
            nothing to compact.
        """
        system = (
            messages[0] if messages and messages[0].get("role") == "system" else None
        )
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
        """Ask the model to summarize a stretch of conversation.

        Args:
            messages: Messages to summarize (roles plus text content).

        Returns:
            The model's summary, or `fallback_summary` when there is nothing
            to summarize or the completion fails.
        """
        lines = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                lines.append(f"{message.get('role')}: {content}")
        if not lines:
            return self.fallback_summary
        transcript = "\n".join(lines)
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
