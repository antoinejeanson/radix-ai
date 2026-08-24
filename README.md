# Radix
Modular AI agent for small, memory-limited LLMs.
Built with Python 3 and uv.
Compatible with Python 3.13+.

# Core concepts
- Minimalist CLI REPL user interface
- Sane, secure default settings (No terminal or Internet access without explicit permission by default)
- Compatible with most self-hosting tools that expose an OpenAI-compatible API (Designed for llama.cpp but works with any OpenAI-compatible API)
- Infinitely extensible with Python-based plugins: agents and tools are Python modules
- The user always talks to a coordinator, tasks are delegated to specialized sub-agents to avoid filling up coordinator context
- AI assistant as code: your whole assistant is defined in Python files. No config files, no YAML, no JSON, no hidden state. Everything is explicit and version-controlled
- Made for small, locally hosted LLMs (< 35B parameters) with limited context windows (< 32k tokens). Also works well with larger models.

# Getting started - Create your own Radix assistant
Install Radix.
Create a folder for your assistant.
Import Radix.
Run.
Infinitely customize your assistant by adding new tools and agents.
Hack Radix itself to add new features and capabilities.
