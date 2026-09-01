"""Example Radix assistant.

Start a llama.cpp server first, e.g.:

    llama-server -m your-model.gguf -c 8192 --port 8080

Then run this file:

    uv run python examples/assistant.py
"""

from radix import (
    Agent,
    Assistant,
    AutoApproveGate,
    Client,
    LlmAdvisoryGate,
    LlmAutoSafetyGate,
)
from radix.builtin import edit_file, fetch_url, read_file, run_shell, write_file

# Trusted machine: never prompt for file tools. The tool_gates below
# re-introduce checks exactly where they matter: shell and the web.
# Remove `permission_gate` to prompt on every tool call.
trusted = AutoApproveGate()

# Radix defaults point at a local llama.cpp server; an explicit client lets
# the LLM safety gates use the same model to review tool calls.
client = Client(model="radix", base_url="http://localhost:8080/v1")

coder = Agent(
    name="coder",
    description="Writes, reviews and debugs code, and inspects the local machine.",
    system_prompt=(
        "You are a concise senior software engineer. Solve the task step by step. "
        "Use read_file to inspect files, edit_file to change existing files, "
        "write_file to create new ones, and run_shell to run commands. "
        "Always explain what a command does before running it."
    ),
    tools=[read_file, edit_file, write_file, run_shell],
)

researcher = Agent(
    name="researcher",
    description="Fetches and summarizes information from the web.",
    system_prompt=(
        "You are a research assistant. Use fetch_url to read pages, then answer "
        "with a short, factual summary citing the URLs you used."
    ),
    tools=[fetch_url],
)

assistant = Assistant(
    client=client,
    agents=[coder, researcher],
    permission_gate=trusted,
    max_context_tokens=32767,  # match your llama.cpp -c value
    tool_gates={
        # LLM judges: read-only commands run, destructive ones are denied.
        "run_shell": LlmAutoSafetyGate(client),
        # LLM advises, the user keeps the final say.
        "fetch_url": LlmAdvisoryGate(client),
    },
)

if __name__ == "__main__":
    assistant.run()
