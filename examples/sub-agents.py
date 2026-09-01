"""Radix assistant with sub-agents: delegation in practice.

Start a llama.cpp server first, e.g.:

    llama-server -m your-model.gguf -c 16384 --port 8080

Then run this file:

    uv run python examples/sub-agents.py
"""

from radix import Agent, Assistant, Client
from radix.builtin import edit_file, fetch_url, read_file, run_shell, write_file

client = Client(model="radix", base_url="http://localhost:8080/v1")

# Sub-agents keep the coordinator's context clean: each ask_<name> call runs
# the agent with a fresh, isolated context, and only its final answer comes
# back. The description becomes the ask_<name> tool's description, so the
# coordinator knows when to delegate.
coder = Agent(
    name="coder",
    description="Writes, reviews and debugs code on the user's machine.",
    system_prompt=(
        "You are a concise senior software engineer. Inspect files with "
        "read_file, change them with edit_file and write_file, and run "
        "commands with run_shell. Explain what a command does before running it."
    ),
    tools=[read_file, edit_file, write_file, run_shell],
)

researcher = Agent(
    name="researcher",
    description="Fetches and summarizes information from the web.",
    system_prompt=(
        "You are a research assistant. Use fetch_url to read pages, then "
        "answer with a short, factual summary citing the URLs you used."
    ),
    tools=[fetch_url],
)

assistant = Assistant(
    client=client,
    agents=[coder, researcher],
    max_context_tokens=16384,  # match your llama.cpp -c value
)

if __name__ == "__main__":
    assistant.run()
