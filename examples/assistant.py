"""Example Radix assistant.

Start a llama.cpp server first, e.g.:

    llama-server -m your-model.gguf -c 8192 --port 8080

Then run this file:

    uv run python examples/assistant.py
"""

from radix import Agent, Assistant
from radix.builtin import edit_file, fetch_url, read_file, run_shell, write_file

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

# Agents can call other agents: the coder hands off documentation questions
# to the researcher, which may in turn delegate further. Depth is bounded by
# Assistant(max_delegation_depth=...), so such cycles cannot run forever.
coder.subagents = [researcher]

assistant = Assistant(
    model="radix",  # llama.cpp accepts any name unless --alias is set
    base_url="http://localhost:8080/v1",  # llama.cpp default
    agents=[coder, researcher],
    max_context_tokens=8192,  # match your llama.cpp -c value
)

if __name__ == "__main__":
    assistant.run()
