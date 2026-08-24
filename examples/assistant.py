"""Example Radix assistant.

Start a llama.cpp server first, e.g.:

    llama-server -m your-model.gguf -c 8192 --port 8080

Then run this file:

    uv run python examples/assistant.py
"""

from radix import Agent, Assistant
from radix.builtin import fetch_url, run_shell

coder = Agent(
    name="coder",
    description="Writes, reviews and debugs code, and inspects the local machine.",
    system_prompt=(
        "You are a concise senior software engineer. Solve the task step by step, "
        "using the run_shell tool when you need to inspect files or run commands. "
        "Always explain what a command does before running it."
    ),
    tools=[run_shell],
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
    model="radix",  # llama.cpp accepts any name unless --alias is set
    base_url="http://localhost:8080/v1",  # llama.cpp default
    agents=[coder, researcher],
    max_context_tokens=8192,  # match your llama.cpp -c value
)

if __name__ == "__main__":
    assistant.run()
