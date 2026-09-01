"""Minimal Radix assistant: the quickstart, as a file.

Start a llama.cpp server first, e.g.:

    llama-server -m your-model.gguf -c 8192 --port 8080

Then run this file:

    uv run python examples/assistant.py
"""

from radix import Assistant
from radix.builtin import edit_file, fetch_url, read_file, run_shell, write_file

assistant = Assistant(
    model="radix",  # llama.cpp accepts any name unless --alias is set
    base_url="http://localhost:8080/v1",  # llama.cpp default
    max_context_tokens=8192,  # match your llama.cpp -c value
    tools=[read_file, edit_file, write_file, run_shell, fetch_url],
)

if __name__ == "__main__":
    assistant.run()
