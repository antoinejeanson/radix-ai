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
- **Sane, secure defaults** — terminal and Internet tools ask for explicit permission on every call. Reading and editing files doesn't.
- **llama.cpp first** — defaults point at `http://localhost:8080/v1`. Anything with an OpenAI-compatible API works too (vLLM, Ollama, ...).
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
    model="radix",                        # llama.cpp accepts any name unless --alias is set
    base_url="http://localhost:8080/v1",  # llama.cpp default
    max_context_tokens=8192,              # match your llama.cpp -c value
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

Inside the REPL: `/help` for commands, `/reset` to forget the conversation,
`Ctrl+D` to exit.

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
sub-agent. Mark anything dangerous with `ask_permission=True` — the user approves
every single call, with the arguments shown:

```python
@tool(ask_permission=True)
def deploy(target: str) -> str:
    """Deploy the app to the given target."""
    ...
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

**Sub-agents can call sub-agents.** Any agent may carry its own `subagents`
and get an `ask_<name>` tool for each — the coordinator is just the first
agent in the chain. A `coder` agent can hand documentation questions to a
`researcher` that delegates further down the line:

```python
research_helper = Agent(name="helper", ...)
researcher = Agent(name="researcher", subagents=[research_helper], ...)
coder = Agent(name="coder", ...)
coder.subagents = [researcher]      # coder can call researcher

assistant = Assistant(agents=[coder, researcher])
```

Mutual references (`coder` ↔ `researcher`) work too: assign `subagents`
after construction, as above. Recursion can't run away —
`Assistant(max_delegation_depth=2)` bounds how many levels of agents a single
call may traverse (the coordinator is level 0); any agent already at the cap
gets no delegation tools, so cycles terminate naturally. So by default, the
coordinator can hand off to a sub-agent, that sub-agent can hand off to one
more — but nothing deeper.

But delegation is a tool in your toolbox, not a rule: plenty of assistants are
one agent with a few tools, and that's exactly what Radix supports too.

### Built-in tools

| Tool        | ask_permission | What it does                                                        |
| ----------- | --------- | ------------------------------------------------------------------- |
| `read_file` | no        | Read a text file (UTF-8, truncated to keep context small).          |
| `edit_file` | no        | Replace one exact, unique snippet — returns a unified diff.         |
| `write_file`| no        | Create or overwrite a file, creating parent directories as needed.  |
| `run_shell` | **yes**   | Run a shell command (120s timeout, output truncated).               |
| `fetch_url` | **yes**   | Fetch a web page.                                                   |

`edit_file` is deliberately fussy for small models: it refuses to edit when
`old_string` matches zero or multiple places, so a confused model can't
silently rewrite the wrong line. The diff it returns lets the model — and you
— verify the change.

### Permissions

The default `CliPermissionGate` prompts on every call that requires permission. Gates are a
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

### Context management

Small context windows are Radix's home turf. `Assistant(max_context_tokens=...)`
sets the budget; when a conversation outgrows it, older messages are
summarized by the model into a single compact message (and dropped if even
that fails). The system prompt and the most recent messages always survive.

---

## Example assistants

Radix ships example assistants you can run as-is or steal parts from:

- **`examples/assistant.py`** — a general-purpose demo with all the built-in
  tools: a `coder` (read, edit, write, shell) and a `researcher` (web). It
  exists mostly to exercise every feature... which is why it's about to grow
  up: it will become **`radix-code`**, an assistant specialized for coding.

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
├── tool.py         # @tool decorator + schema generation
├── messages.py     # Message & result types
└── builtin/        # read_file, edit_file, write_file, run_shell, fetch_url
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
