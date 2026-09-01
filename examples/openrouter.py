"""Radix assistant against OpenRouter's hosted OpenAI-compatible API.

Radix works with any OpenAI-compatible endpoint, not just local llama.cpp
servers. This example points the same assistant at OpenRouter.

You need an API key:

    export OPENROUTER_API_KEY=sk-or-...

Then run this file:

    uv run python examples/openrouter.py

OpenRouter also recommends sending HTTP-Referer and X-Title headers for
attribution; pass them by handing `Client` an `openai_client` built with
`OpenAI(..., default_headers={...})` if you want that.
"""

import os

from radix import Assistant, Client
from radix.builtin import edit_file, fetch_url, read_file, run_shell, write_file

api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit(
        "set OPENROUTER_API_KEY (or OPENAI_API_KEY) before running this example"
    )

client = Client(
    # Any OpenRouter model id, e.g. "anthropic/claude-3.5-sonnet".
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

assistant = Assistant(
    client=client,
    # Hosted models have large context windows; 64k keeps sessions long
    # without blowing up summarization cost.
    max_context_tokens=65536,
    tools=[read_file, edit_file, write_file, run_shell, fetch_url],
)

if __name__ == "__main__":
    assistant.run()
