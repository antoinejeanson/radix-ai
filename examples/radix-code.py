"""radix-code: an agentic coding assistant.

Start a llama.cpp server with a matching context first, e.g.:

    llama-server -m your-model.gguf -c 16384 --port 8080

Then run this file:

    uv run python examples/radix-code.py
"""

from radix import Agent, Assistant, AutoApproveGate, Client, LlmAutoSafetyGate
from radix.builtin import ask_question, edit_file, read_file, run_shell, write_file

client = Client(model="radix", base_url="http://localhost:8080/v1")

# The explorer is a sub-agent that handles repo-wide exploration.  Its
# stateless runs start with a fresh context every time, so the coordinator's
# conversation history stays lean: only the explorer's final answer comes back.
# The coordinator should delegate file-reading and shell-inspection to it
# rather than calling read_file / run_shell directly — that keeps repo
# exploration out of the coordinator's context budget.
explorer = Agent(
    name="explorer",
    description=(
        "Reads files and inspects the project structure via the shell. "
        "Use it whenever you need to understand the codebase, locate symbols, "
        "or run diagnostic commands."
    ),
    system_prompt=(
        "You are the explorer agent. Your job is to read files and inspect "
        "the project structure. Use read_file to read files and run_shell to "
        "run read-only shell commands (ls, grep, find, git log, etc.). "
        "Summarise your findings concisely — the coordinator only receives "
        "your final answer, not the raw tool output."
    ),
    tools=[read_file, run_shell],
)

# Sane defaults for agentic coding:
# - The default 8k context is tight for code; 16k fits a small GGUF while
#   leaving room for files, diffs and tool output.
# - More recent turns survive compaction, because coding sessions are
#   iterative; keep a few whole turns verbatim.
# - More tool rounds because coding is multi-step.
# - The coordinator is stateful, so tools live on it directly: a coding
#   session builds on its own earlier turns (sub-agents start fresh every
#   task, which is great for isolation but wrong for one long session).

radix_code = Assistant(
    client=client,
    agents=[explorer],
    max_context_tokens=32767,  # match your llama.cpp -c value
    reserve_output_tokens=3072,
    keep_recent=6,
    keep_recent_turns=3,
    max_tool_rounds=16,
    system_prompt=(
        "You are radix-code, a senior software engineer working on the user's "
        "machine.\n\n"
        "Your memory is limited. Do NOT pollute your own context with repo "
        "exploration. Instead of calling read_file or run_shell yourself to "
        "inspect the codebase, delegate that work to the explorer agent using "
        "the ask_explorer tool. The explorer runs with a fresh context every "
        "time and only its final answer comes back to you.\n\n"
        "Work step by step: ask the explorer to inspect the project, always say "
        "what a command does before asking the explorer to run it, make minimal "
        "surgical edits, then verify with tests or a quick command and report "
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
    tools=[read_file, run_shell, edit_file, write_file, ask_question],
)

if __name__ == "__main__":
    radix_code.run()
