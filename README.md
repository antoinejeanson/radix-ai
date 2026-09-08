# Radix

**A Python framework for building your own AI assistant on small, local LLMs.**

Built with Python 3 and uv. Compatible with Python 3.13+.

> **Radix is not an assistant. It's what assistants are built with.**
>
> You have a workflow. Radix lets you compose an assistant for exactly that
> workflow — your agents, your tools, your prompts, in plain Python. No config
> files, no YAML, no JSON, no hidden state. Your whole assistant is a Python
> file you can read, diff, and version-control.

```
your workflow
     │
     ▼
┌──────────────────────────────────────────────────┐
│                    Radix                         │
│                                                  │
│   Assistant(model=..., agents=[...], tools=[...])│
│                                                  │
│   ┌───────────┐   ask_*   ┌──────────────────┐   │
│   │coordinator│ ─────────►│ your sub-agents  │   │
│   └───────────┘ (optional)└──────────────────┘   │
│        │                                         │
│        └── your tools: Python functions          │
└──────────────────────────────────────────────────┘
```

Radix believes your 3B model running on a laptop deserves a great assistant
too. Every token is precious, every context window is sacred real estate — so
Radix keeps contexts clean, shows you everything it does, and never touches
your terminal or the Internet without asking first.

## Why "Radix"?

*Radix* is Latin for **root**.

A Radix assistant grows like a root system: the coordinator is the taproot,
and each sub-agent a branch root reaching into its own patch of soil — an
isolated context, a set of specialized tools. And the tree growing out of
those roots is *your* project, drawing up the nutrients — tokens, context,
model horsepower — to do your actual work.

Deep roots, small footprint. Just like the models we love.

---

## Features

- **Assistant as code** — the whole assistant is defined in Python. What you see in the file *is* the assistant.
- **Your workflow, your agents** — compose specialized agents and tools around how *you* work, then hand the result to anyone: it's one `uv run` away.
- **Delegation without dogma** — Radix supports delegating to specialized sub-agents to keep the main context clean, but never enforces it. A one-agent assistant with no tools is a perfectly good Radix assistant.
- **Minimalist CLI REPL** — no web UI, no daemon, no dashboard. A prompt, a model, and you.
- **Sane, secure defaults** — every tool call goes through a permission gate, and the default gate asks for explicit approval on every call. Trusted setups opt out with `AutoApproveGate`.
- **llama.cpp first** — defaults point at `http://localhost:8080/v1`. Anything with an OpenAI-compatible API works too (vLLM, Ollama, OpenRouter, ...).
- **Made for small models** — designed for locally hosted LLMs (< 35B params, < 32k context) and their precious token budgets. Also perfectly happy with bigger models.

## What a Radix assistant looks like at work

Every step is shown: who's thinking, every tool call, every result — file
edits as colored diffs, sub-agent answers in panels with their total call time
and token use.

```
you> change the greeting in app.py to hello

• coordinator thought for 0.2s
• coordinator: ask_coder {"task": "In app.py, change the printed greeting ..."}
• coder thought for 0.1s
• coder: read_file {"path": "app.py"}
def greet():
    print("hi")
• coder: edit_file {"path": "app.py", "old_string": "print(\"hi\")", ...}
Edited app.py.
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def greet():
-    print("hi")
+    print("hello")
╭──────────────────────────────── coder ─────────────────────────────────╮
│ Changed the greeting to hello in app.py.                               │
╰───────────────────────────── 2.3s · 91 tok ────────────────────────────╯

Done — app.py now says hello.
```

Token counts come from the server when it reports them (OpenAI-style
`include_usage`); otherwise Radix shows a `~` estimate.

---

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```sh
uv add radix-ai --from git+https://github.com/antoinejeanson/radix-ai.git
```

Or use it straight from a clone (see [Development](#development)).

## Quickstart: your first assistant

**1. Start a model server.** Radix defaults match llama.cpp:

```sh
llama-server -m your-model.gguf -c 8192 --port 8080
```

**2. The smallest possible assistant.** Create `assistant.py`. No agents, no
tools — just you and the model:

```python
from radix import Assistant

assistant = Assistant(
    model="radix",  # llama.cpp accepts any name unless --alias is set
    base_url="http://localhost:8080/v1",  # llama.cpp default
    max_context_tokens=8192,  # match your llama.cpp -c value
)

if __name__ == "__main__":
    assistant.run()
```

**3. Run it.**

```sh
uv run python assistant.py
```

Then grow it: give the coordinator tools for quick jobs, and add specialized
sub-agents when tasks deserve their own context and expertise. That's the
whole framework — everything below is just composition.

Inside the REPL: `/help` for commands, `/compact` to summarize the old
conversation right away (reducing the context without waiting for the budget),
`/undo` to revert the last turn (files and conversation), `/transcript [path]`
to write the conversation to a JSONL file, `Ctrl+D` to exit. A
status line always shows the current context usage
(e.g. `context: 3.4k / 8.2k tok · 21 messages · 41%`): under the input while
waiting, and through the whole turn — under the live views and after every
tool call, so you can watch the context grow in real time. The percentage is
against `max_context_tokens` (the full window), and the count includes the
tool calls and results the coordinator remembers.

Sub-agents work quietly: their tool calls are listed, but their raw tool
results stay hidden — each delegation just ends with the sub-agent's answer
in a panel (with its total time). The context line is only the coordinator's;
it is not shown while a sub-agent is working underneath it. The `Repl`
constructor opts into more detail when you want it:

- `Repl(..., show_subagent_tool_outputs=True)` prints every sub-agent tool
  result as it runs (edits as colored diffs).
- `Repl(..., show_subagent_context=True)` shows a context line for the
  working sub-agent — an estimate of *its own* fresh conversation, labeled
  with the agent's name — instead of hiding the line during sub-agent work.

---

## Building your assistant

### Tools

A tool is a typed Python function behind the `@tool` decorator. The name comes
from the function, the description from the docstring, and the JSON schema from
the type hints:

```python
from radix import tool


@tool
def word_count(path: str) -> str:
    """Count the words in a text file."""
    with open(path) as f:
        return str(len(f.read().split()))
```

Tools can go on the coordinator (`Assistant(tools=[...])`) or on any
sub-agent. Every tool call goes through the assistant's permission gate; the
gate sees the tool and its arguments and decides, showing the user the
arguments and asking for approval when the policy says so:

```python
from radix import AutoApproveGate, Assistant

assistant = Assistant(permission_gate=AutoApproveGate())  # trusted: never prompt
```

### Sub-agents and delegation (optional)

When tasks get specialized, split them off. A sub-agent is just a system
prompt plus tools; each becomes an `ask_<name>` tool on the coordinator:

```python
from radix import Agent, Assistant
from radix.builtin import fetch_url

researcher = Agent(
    name="researcher",
    description="Fetches and summarizes information from the web.",
    system_prompt="Use fetch_url to read pages, then answer with a short factual summary.",
    tools=[fetch_url],
)

assistant = Assistant(agents=[researcher], tools=[...])
```

Sub-agents run with a fresh, isolated context every time, and only their
final answer enters the coordinator's memory — that's how Radix keeps small
context windows usable over long conversations.

But delegation is a tool in your toolbox, not a rule: plenty of assistants are
one agent with a few tools, and that's exactly what Radix supports too.

### Built-in tools

| Tool        | What it does                                                          |
| ----------- | --------------------------------------------------------------------- |
| `read_file` | Read a text file's lines (UTF-8) with `offset`/`limit` paging (default 1000 lines); when lines remain it reports the range shown and how many are left so the agent can page further. |
| `edit_file` | Replace one exact, unique snippet — returns a unified diff.           |
| `write_file`| Create or overwrite a file, creating parent directories as needed.    |
| `grep` | Search file contents with a regex; returns `file:line: text`, skipping binaries and hidden directories (read-only). |
| `run_shell` | Run a shell command (default 120s timeout, default 500-line output cap, both agent-configurable; a footer reports cut lines). |
| `fetch_url` | Fetch a web page (default 30s timeout, default 500-line body cap, both agent-configurable; a footer reports cut lines). |
| `remember` | Append a timestamped note to `.radix/notes.md` (persistent across sessions). |
| `recall` | Search the notes for a substring and return the matching lines. |

Whether any of these prompt for approval is the permission gate's decision,
not the tool's: with the default gate every call asks; with `AutoApproveGate`
nothing does; with `tool_gates` you pick per tool.

`edit_file` is deliberately fussy for small models: it refuses to edit when
`old_string` matches zero or multiple places, so a confused model can't
silently rewrite the wrong line. The diff it returns lets the model — and you
— verify the change.

Every change made by `edit_file` and `write_file` is tracked per turn, so
`/undo` in the REPL restores the affected files to their pre-turn state (and
deletes files the assistant created). `/undo N` reverts the last N turns,
`/undo all` reverts everything. Changes made through `run_shell` or your own
tools are not tracked. When the conversation runs long, use `/compact` (or
`assistant.compact()`) to summarize the oldest turns immediately; compaction
happens automatically at the token budget, but the command does it on demand.
Note that summarized turns are gone: `/undo` cannot rewind past them.

### Permissions

The default `CliPermissionGate` prompts on every tool call. Gates are a
one-method protocol, so policies are just Python:

```python
class OnlyLocalhostGate:
    def check(self, tool, arguments):
        if tool.name == "fetch_url":
            return "localhost" in arguments.get("url", "")
        return True


assistant = Assistant(agents=[...], permission_gate=OnlyLocalhostGate())
```

Per-tool allowlists, command prefixes, working-directory jails — the gate sees
the tool and its parsed arguments, so the sky's the limit.

There is no per-tool "ask me" flag: `AutoApproveGate` is the documented opt-out
for fully trusted setups, and `tool_gates` (below) re-introduces prompting for
just the tools you care about:

```python
assistant = Assistant(
    permission_gate=AutoApproveGate(),  # trust everything by default
    tool_gates={"run_shell": CliPermissionGate()},  # ...but always ask for shell
)
```

#### LLM safety gates

Two ready-made gates ask the model whether a permission-gated call is safe
before anything happens. Both take the assistant's `Client` and work with any
tool, not just `run_shell`:

```python
client = Client(model="radix", base_url="http://localhost:8080/v1")

# SAFE verdicts auto-approve, DANGEROUS (or a failed check) denies: no prompts.
assistant = Assistant(client=client, permission_gate=LlmAutoSafetyGate(client))

# Same, but a DANGEROUS verdict asks "Run anyway? [y/N]" before denying:
# the model flags, the human stays in charge.
assistant = Assistant(
    client=client, permission_gate=LlmAutoSafetyGate(client, confirm_unsafe=True)
)

# Shows the model's verdict, then the user gets the usual [y/N] prompt.
assistant = Assistant(client=client, permission_gate=LlmAdvisoryGate(client))
```

Both fail closed: a network error or an unparseable answer never auto-approves
— the auto gate denies, the advisory gate falls back to the user prompt.

Each check prefixes the system prompt with the current working directory
(`Current working directory: ...`), so the reviewer judges the call against
the project actually being worked on. The policy (`SAFETY_SYSTEM_PROMPT`, and
see `LlmSafetyChecker(..., cwd=...)` to pin a scope) considers:

- **Safe:** read-only and everyday operations; destructive edits *inside* the
  working directory (deleting build artifacts, temp files, task-owned files);
  plain read-only (GET) web fetches of docs and public pages.
- **Dangerous:** destructive operations on paths *outside* the working
  directory (parents, `..`, `/`, system dirs, the home directory, other
  projects); whole-project or whole-system destruction even in-CWD (deleting
  the repo root, wiping the main database, `rm -rf /`, `dd`, `mkfs`); and
  **sending local data to the Internet** — uploads, POST/PUT/PATCH bodies with
  local contents, `curl`/`wget`/`scp`/`ftp` pushing data to a server, or URLs
  that encode private data. When the assistant runs locally for privacy,
  things like `fetch_url` must not leak data to external hosts casually.

#### Per-tool gates

A single gate applies to every tool, but you can hand specific tools their own
gate with `tool_gates` — useful when one tool deserves a harder or softer
policy than the rest. A per-tool gate **replaces** the global one for that
tool; everything else keeps the `permission_gate`:

```python
assistant = Assistant(
    client=client,
    tools=[run_shell, fetch_url],
    permission_gate=CliPermissionGate(),  # default for everything else
    tool_gates={
        "run_shell": LlmAutoSafetyGate(client),  # `ls` runs, `rm -rf /` auto-denied
        "fetch_url": LlmAdvisoryGate(client),  # LLM advises, you decide
    },
)
```

The map applies to sub-agents too, and names are checked against the tools
that actually exist — a typo like `"run_shll"` raises a `ValueError` instead
of silently bypassing the gate. `ask_<name>` delegation tools are valid
targets.

An `Agent` always overrides the assistant's gates with its own when you pass
`permission_gate=` or `tool_gates=` explicitly to the agent; otherwise the
assistant's gates are inherited.

### Context management

Small context windows are Radix's home turf. `Assistant(max_context_tokens=...)`
sets the budget; when a conversation outgrows it, older messages are
summarized by the model into a single compact message (and dropped if even
that fails). The system prompt and the most recent messages always survive.

A stateful agent (the coordinator) remembers everything it did: each turn's
task, every tool call with its arguments, every tool result, and the final
answer — plus, for delegations, the `ask_<name>` call and the sub-agent's
answer (never the sub-agent's internal tool work, which stays private and
stateless). That is what makes the next turn able to quote earlier tool
results, and it is also what makes context grow faster — compaction is the
safety valve.

Compaction keeps whole turns verbatim: `keep_recent_turns` (default 1) keeps
that many complete turns (a turn runs from a user message through the
following assistant message, including any tool calls and results), never
fewer than `keep_recent` trailing messages.

Every limit is a plain constructor kwarg with a sensible default. On
`Assistant`: `max_context_tokens`, `reserve_output_tokens`, `keep_recent`,
`keep_recent_turns`, `summary_prompt`, `fallback_summary`, and
`max_tool_rounds` for the coordinator. On `Agent`, for a specific sub-agent:
`max_tool_rounds`. The constants (`SUMMARY_PROMPT`, `FALLBACK_SUMMARY`,
`DEFAULT_KEEP_RECENT_TURNS`, ...) are exported from `radix` if you want to
tune relative to them.

---

## Example assistants

Radix ships example assistants you can run as-is or steal parts from:

- **`examples/assistant.py`** — the smallest useful assistant: the
  coordinator with all the built-in tools and the default permission gate
  (every call prompts). The right file to copy when you start your own.
- **`examples/radix-code.py`** — an agentic coding assistant. It trades the
  small-context defaults for a 16k context, a system prompt built around
  inspect-edit-verify, more tool rounds per turn, and a coordinator that
  keeps its session state. It also shows the permission-gate combo in
  practice: file tools never prompt, while the LLM safety gate reviews
  every shell command and asks you before dangerous ones run.
- **`examples/sub-agents.py`** — delegation in practice: a `coder` and a
  `researcher` sub-agent. Each delegated task runs in a fresh, isolated
  context and only the final answer reaches the coordinator.
- **`examples/openrouter.py`** — the same assistant pointed at OpenRouter's
  hosted API (`https://openrouter.ai/api/v1`) instead of a local server,
  with the model id and API key read from the environment.

More ready-made assistants are on the way. And if you build one you like,
that's the intended happy ending — an assistant is just a Python file.

---

## How it works

```
radix/
├── assistant.py    # Assistant: wires client, context, coordinator, REPL
├── coordinator.py  # Coordinator + ask_<name> delegation tools
├── agent.py        # The tool-calling loop every agent runs
├── client.py       # OpenAI-compatible streaming client (llama.cpp defaults)
├── context.py      # Token budgeting + summarization
├── events.py       # The observer hooks the REPL renders
├── repl.py         # The CLI: spinners, diffs, panels, tokens
├── permissions.py  # Permission gates
├── safety.py       # LLM safety checker used by the LLM permission gates
├── tool.py         # @tool decorator + schema generation
├── messages.py     # Message & result types
├── undo.py         # Per-turn file snapshots for /undo
└── builtin/        # read_file, edit_file, write_file, grep, remember, recall, run_shell, fetch_url
```

- **Streaming everywhere.** Answers stream token by token; tool calls stream too.
- **Everything is an event.** `on_start`, `on_delta`, `on_activity`,
  `on_tool_output`, `on_stop` — the REPL is just one consumer. Build your own
  UI, logger, or test harness on the same hooks.
- **Tools are ordinary calls.** The model proposes, the permission gate
  decides, the function runs, the result goes back to the model — and to your
  screen.
- **No hidden state.** What you see in `assistant.py` is the entire assistant.

---

## Development

```sh
git clone https://github.com/antoinejeanson/radix-ai.git
cd radix-ai
uv sync          # install dependencies
uv run pytest    # run the test suite
```

The test suite covers the tool decorator, permission gates, the agent loop,
delegation, context compaction, the streaming client, and the REPL rendering
— all with scripted fake models, no server required.

Ideas for hacking: new built-in tools, new permission policies, a web UI on top
of `Events`, multi-model setups where the researcher uses a bigger model than
the coder... it's all just Python.

---

*Radix: because even a model that fits in 2GB of RAM deserves a competent assistant — and you deserve one that's yours.*
