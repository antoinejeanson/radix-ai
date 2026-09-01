"""radix-code: an agentic coding assistant.

Start a llama.cpp server with a matching context first, e.g.:

    llama-server -m your-model.gguf -c 16384 --port 8080

Then run this file:

    uv run python examples/radix-code.py
"""

from radix import Assistant, AutoApproveGate, Client, LlmAutoSafetyGate
from radix.builtin import edit_file, read_file, run_shell, write_file

client = Client(model="radix", base_url="http://localhost:8080/v1")

# Sane defaults for agentic coding:
# - The default 8k context is tight for code; 16k fits a small GGUF while
#   leaving room for files, diffs and tool output.
# - More recent turns survive compaction, and transcripts summarize less
#   aggressively, because coding sessions are iterative.
# - More tool rounds because coding is multi-step.
# - The coordinator is stateful, so tools live on it directly: a coding
#   session builds on its own earlier turns (sub-agents start fresh every
#   task, which is great for isolation but wrong for one long session).

radix_code = Assistant(
    client=client,
    max_context_tokens=16384,  # match your llama.cpp -c value
    reserve_output_tokens=3072,
    keep_recent=6,
    transcript_char_limit=16000,
    max_tool_rounds=16,
    max_tool_output_chars=24000,
    system_prompt=(
        "You are radix-code, a senior software engineer working on the user's machine. "
        "Work step by step: inspect the project with read_file and run_shell before "
        "changing anything, always say what a command does before running it, make "
        "minimal surgical edits, then verify with tests or a quick command and report "
        "what you changed and what you verified. Follow the project's existing "
        "conventions. Never invent files, commands or test results."
    ),
    # Trusted machine: file reads and edits never prompt. The shell is the
    # one thing that matters, so the LLM reviews every command: read-only
    # commands run on their own, while dangerous ones ask before running.
    permission_gate=AutoApproveGate(),
    tool_gates={
        "run_shell": LlmAutoSafetyGate(client, confirm_unsafe=True),
    },
    tools=[read_file, edit_file, write_file, run_shell],
)

if __name__ == "__main__":
    radix_code.run()
