"""benchmark-code: a coding assistant for automated evaluation.

This assistant is designed for benchmarking small LLMs on coding tasks. It:
- Reads task prompts from stdin (one per line) or from a single CLI argument
- Auto-approves all tool calls (no prompts — for use in sandboxes)
- Outputs structured JSON results: response, token usage, tool calls per task

Usage:

    # Single task via CLI argument:
    uv run python examples/benchmark-code.py "Fix the bug in src/radix/agent.py"

    # Multiple tasks via stdin (one task per line):
    echo -e "Task 1\\nTask 2" | uv run python examples/benchmark-code.py

    # Multiple tasks from a file:
    uv run python examples/benchmark-code.py < tasks.txt

    # With a custom model:
    uv run python examples/benchmark-code.py --model my-model \
        --base-url http://localhost:8080/v1 \
        "Implement merge sort in src/radix/sorting.py"

    # With a custom context size:
    uv run python examples/benchmark-code.py \
        --max-context-tokens 16384 "Write a parser for JSON"

Start a model server first, e.g.:

    llama-server -m your-model.gguf -c 32768 --port 8080

Sandboxed with Podman (all tool calls are auto-approved, so this is safe
to run where edits and shell commands stay inside the container):

    podman build -t radix-sandbox .
    mkdir -p ~/work

    # Single task:
    podman run --rm \
        --network=host \
        -v ~/work:/work \
        -w /work \
        radix-sandbox \
        python /app/examples/benchmark-code.py "Fix the bug in src/radix/agent.py"

    # Tasks from stdin, results to stdout (no -t, to keep it scriptable):
    echo -e "Task 1\\nTask 2" | podman run --rm -i \
        --network=host \
        -v ~/work:/work \
        -w /work \
        radix-sandbox \
        python /app/examples/benchmark-code.py > results.json

    # Tasks from a file, mounted into the container:
    podman run --rm -i \
        --network=host \
        -v ~/work:/work \
        -v "$PWD/tasks.txt:/tasks.txt:ro" \
        -w /work \
        radix-sandbox \
        python /app/examples/benchmark-code.py < tasks.txt \
        > ~/work/results.json

`--network=host` reaches the llama.cpp server on `localhost:8080` directly
(works when the server binds only loopback). For network isolation instead,
restart llama-server with `--host 0.0.0.0` and pass a gateway base URL:

    podman run --rm -i \
        --network=pasta \
        --add-host host.containers.internal:host-gateway \
        -v ~/work:/work \
        -w /work \
        radix-sandbox \
        python /app/examples/benchmark-code.py \
        --base-url http://host.containers.internal:8080/v1 \
        < tasks.txt > results.json
"""

import argparse
import json
import sys

from radix import Agent, Assistant, AutoApproveGate, Client, Events, Usage
from radix.builtin import ask_question, edit_file, read_file, run_shell, write_file

# Default configuration for benchmarking.
# These values are chosen to be reasonable for small GGUF models while
# leaving enough room for tool output and conversation history.

client = Client(
    model="radix",
    base_url="http://localhost:8080/v1",
)

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

# Coding assistant with all tool calls auto-approved for sandboxed use.
benchmark_code = Assistant(
    client=client,
    agents=[explorer],
    max_context_tokens=32767,
    reserve_output_tokens=3072,
    keep_recent=6,
    keep_recent_turns=3,
    max_tool_rounds=16,
    system_prompt=(
        "You are a senior software engineer working on the user's machine.\n\n"
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
    # All tool calls pass by default — this assistant runs in a sandbox.
    permission_gate=AutoApproveGate(),
    tools=[read_file, run_shell, edit_file, write_file, ask_question],
)


def run_task(task: str) -> dict:
    """Run a single benchmark task and return structured results.

    `Assistant.chat()` returns only the final answer string, so tool calls
    and token usage are collected through the `Events` observers: every tool
    call fires `on_activity` (for the coordinator and for any delegated
    sub-agent) and every model round fires `on_stop` with its token usage.

    Args:
        task: The task prompt to send to the assistant.

    Returns:
        A dict with keys: task, response, tool_calls, usage, tokens_total.
    """
    tool_calls_info: list[dict] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def on_activity(name: str, text: str, raw_arguments: str = "") -> None:
        tool_calls_info.append(
            {"agent": name, "call": text, "arguments": raw_arguments}
        )

    def on_stop(
        name: str,
        elapsed: float,
        produced_text: bool,
        usage: Usage | None = None,
    ) -> None:
        if usage is not None:
            usage_totals["prompt_tokens"] += usage.prompt_tokens or 0
            usage_totals["completion_tokens"] += usage.completion_tokens or 0
            usage_totals["total_tokens"] += usage.total_tokens or 0

    events = Events(on_activity=on_activity, on_stop=on_stop)
    response = benchmark_code.chat(task, events=events)

    return {
        "task": task,
        "response": response,
        "tool_calls": tool_calls_info,
        "usage": usage_totals,
        "tokens_total": usage_totals["total_tokens"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a coding assistant on one or more tasks."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="A single task prompt. If omitted, reads tasks from stdin (one per line).",
    )
    parser.add_argument(
        "--model",
        default="radix",
        help="Model name (default: radix).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="API base URL (default: http://localhost:8080/v1).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=32767,
        help="Max context tokens (default: 32767).",
    )
    args = parser.parse_args()

    # Rebuild client with any CLI overrides
    benchmark_code.client = Client(
        model=args.model,
        base_url=args.base_url,
    )

    # Read tasks from stdin if no CLI prompt given
    if args.prompt:
        tasks = [args.prompt]
    else:
        tasks = [line.rstrip("\n") for line in sys.stdin if line.strip()]

    if not tasks:
        print(
            "Error: no tasks provided. Use a CLI argument or pipe tasks via stdin.",
            file=sys.stderr,
        )
        sys.exit(1)

    results = []
    for task in tasks:
        result = run_task(task)
        results.append(result)

    # Print a JSON array of results to stdout
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
