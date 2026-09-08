from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Tool: dataclass plus decorator turning typed Python functions into tools
# with OpenAI function-calling schemas derived from their type hints.
_PRIMITIVES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _type_to_schema(tp: Any) -> dict[str, Any]:
    if tp in _PRIMITIVES:
        return {"type": _PRIMITIVES[tp]}
    origin = typing.get_origin(tp)
    if origin in (list, tuple, set):
        schema: dict[str, Any] = {"type": "array"}
        args = typing.get_args(tp)
        if args:
            schema["items"] = _type_to_schema(args[0])
        return schema
    if origin is dict:
        return {"type": "object"}
    if origin in (typing.Union, types.UnionType):
        alternatives = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(alternatives) == 1:
            return _type_to_schema(alternatives[0])
    return {"type": "string"}


def _is_optional(tp: Any) -> bool:
    origin = typing.get_origin(tp)
    return origin in (typing.Union, types.UnionType) and type(None) in typing.get_args(
        tp
    )


def schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    """Derive the OpenAI parameter JSON schema from a function's signature.

    Type hints decide the schema; parameters without a default and not
    Optional become `required`. Unsupported types fall back to "string".

    Args:
        fn: The function to inspect (must be introspectable; avoid
            `*args`/`**kwargs` and names without hints).

    Returns:
        A JSON schema object with "type", "properties" and, when needed,
        "required".
    """
    hints = typing.get_type_hints(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        tp = hints.get(name, str)
        properties[name] = _type_to_schema(tp)
        if param.default is inspect.Parameter.empty and not _is_optional(tp):
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


@dataclass
class Tool:
    """A capability an agent can invoke.

    Every tool call goes through the agent's permission gate before running;
    which tools are sensitive is the gate's decision, not the tool's.

    Fields:
        name: Unique tool name shown to the model.
        description: What the tool does and when to use it.
        parameters: OpenAI JSON schema describing the accepted arguments.
        fn: The implementation, called with keyword arguments parsed from
            the model's call. Its return value (stringified if needed)
            becomes the tool output message.
        snapshot: When True, the assistant snapshots the `path` argument
            (if present) into its undo log before the tool runs, so a later
            /undo can restore the file. Set on file-mutating tools.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    snapshot: bool = False

    def run(self, **kwargs: Any) -> str:
        """Call the tool with keyword arguments.

        Args:
            **kwargs: Arguments matching the declared parameters.

        Returns:
            The tool's output as a string.
        """
        result = self.fn(**kwargs)
        return result if isinstance(result, str) else str(result)

    def schema(self) -> dict[str, Any]:
        """The OpenAI function-calling schema for this tool.

        Returns:
            A dict with "type": "function" and the function name,
            description and parameter schema.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    snapshot: bool = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator turning a typed Python function into a Tool.

    The tool name defaults to the function name, the description to its
    docstring, and the parameter JSON schema is derived from the type hints.

    Args:
        fn: The function to wrap; omit when using `@tool(...)` with
            arguments.
        name: Tool name; defaults to the function name.
        description: Tool description; defaults to the function's docstring.
        snapshot: When True, the assistant snapshots the tool's `path`
            argument into its undo log before the tool runs. Set on
            file-mutating tools so /undo can restore them.

    Returns:
        A Tool, or the decorator itself when called as `@tool(...)`.
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or f.__name__,
            description=description or (inspect.getdoc(f) or ""),
            parameters=schema_from_signature(f),
            fn=f,
            snapshot=snapshot,
        )

    if fn is None:
        return wrap
    return wrap(fn)
