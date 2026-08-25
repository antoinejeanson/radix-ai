from typing import Optional

from radix import Tool, tool


def test_tool_decorator_defaults():
    @tool
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    assert isinstance(add, Tool)
    assert add.name == "add"
    assert add.description == "Add two numbers."
    assert add.ask_permission is False
    assert add.parameters == {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }
    assert add.run(a=2, b=3) == "5"


def test_tool_decorator_kwargs_and_ask_permission():
    @tool(name="custom", description="Custom desc", ask_permission=True)
    def something(x: str) -> str:
        return x

    assert something.name == "custom"
    assert something.description == "Custom desc"
    assert something.ask_permission is True


def test_tool_optional_and_container_types():
    @tool
    def search(query: str, tags: list[str], limit: Optional[int] = None) -> str:
        return query

    props = search.parameters["properties"]
    assert props["query"] == {"type": "string"}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["limit"] == {"type": "integer"}
    assert search.parameters["required"] == ["query", "tags"]


def test_tool_schema_openai_shape():
    @tool
    def noop(text: str) -> str:
        """Do nothing."""
        return text

    schema = noop.schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "noop"
    assert schema["function"]["description"] == "Do nothing."
    assert schema["function"]["parameters"]["required"] == ["text"]


def test_tool_run_converts_non_str_result():
    @tool
    def answer() -> str:
        return 42  # type: ignore[return-value]

    assert answer.run() == "42"
