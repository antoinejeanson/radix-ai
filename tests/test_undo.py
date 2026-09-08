import json
import os

from fakes import ScriptedClient
from radix import Agent, Assistant, AutoApproveGate, tool
from radix.builtin import edit_file, write_file
from radix.messages import ChatResult, ToolCall
from radix.undo import UndoLog


# Tests for UndoLog and Assistant.undo: snapshots, file restore, history
# rewind, and the reset flow.
def test_undo_restores_edited_file(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("before")
    log = UndoLog()
    log.begin_turn(0)
    log.snapshot(str(path))
    path.write_text("after")
    result = log.undo()
    assert result.restored == [str(path)]
    assert result.history_depth == 0
    assert path.read_text() == "before"
    assert len(log) == 0


def test_undo_deletes_created_file(tmp_path):
    path = tmp_path / "new.txt"
    log = UndoLog()
    log.begin_turn(0)
    log.snapshot(str(path))
    path.write_text("created")
    result = log.undo()
    assert result.restored == [str(path)]
    assert not path.exists()


def test_undo_skips_unchanged_file(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("same")
    log = UndoLog()
    log.begin_turn(0)
    log.snapshot(str(path))
    result = log.undo()
    assert result.restored == []
    assert path.read_text() == "same"


def test_snapshot_earliest_state_wins_within_turn(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("v0")
    log = UndoLog()
    log.begin_turn(0)
    log.snapshot(str(path))
    path.write_text("v1")
    log.snapshot(str(path))
    path.write_text("v2")
    log.snapshot(str(path))
    log.undo()
    assert path.read_text() == "v0"


def test_undo_single_turn_of_two(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("v0")
    log = UndoLog()
    log.begin_turn(0)
    log.snapshot(str(path))
    path.write_text("v1")
    log.begin_turn(2)
    log.snapshot(str(path))
    path.write_text("v2")
    result = log.undo(1)
    assert path.read_text() == "v1"
    assert result.history_depth == 2
    assert len(log) == 1


def test_undo_multiple_turns_keeps_oldest_state(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("v0")
    log = UndoLog()
    log.begin_turn(0)
    log.snapshot(str(path))
    path.write_text("v1")
    log.begin_turn(2)
    log.snapshot(str(path))
    path.write_text("v2")
    result = log.undo(2)
    assert path.read_text() == "v0"
    assert result.history_depth == 0
    assert len(log) == 0


def test_undo_zero_or_empty_log():
    pathless = UndoLog()
    assert pathless.undo().restored == []
    assert pathless.undo().history_depth is None
    log = UndoLog()
    log.begin_turn(0)
    assert log.undo(0).restored == []
    assert len(log) == 1


def test_snapshot_without_turn_is_noop(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("x")
    log = UndoLog()
    log.snapshot(str(path))
    assert log.undo().restored == []
    assert path.read_text() == "x"


def test_snapshot_skips_binary_file(tmp_path):
    path = tmp_path / "bin.dat"
    path.write_bytes(b"\xff\xfe\x00")
    log = UndoLog()
    log.begin_turn(0)
    log.snapshot(str(path))
    assert log.undo().restored == []
    assert path.read_bytes() == b"\xff\xfe\x00"


def test_clear(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("v0")
    log = UndoLog()
    log.begin_turn(0)
    log.snapshot(str(path))
    path.write_text("v1")
    log.clear()
    assert len(log) == 0
    assert log.undo().restored == []
    assert path.read_text() == "v1"


def test_assistant_undo_direct_tool(tmp_path):
    path = tmp_path / "app.py"
    original = "def greet():\n    print('hi')\n"
    path.write_text(original)
    client = ScriptedClient(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="edit_file",
                        raw_arguments=json.dumps(
                            {
                                "path": str(path),
                                "old_string": "print('hi')",
                                "new_string": "print('hello')",
                            }
                        ),
                    )
                ]
            ),
            ChatResult(content="done"),
        ]
    )
    assistant = Assistant(
        client=client, tools=[edit_file], permission_gate=AutoApproveGate()
    )
    assistant.chat("change the greeting")
    assert "print('hello')" in path.read_text()
    result = assistant.undo()
    assert result.restored == [str(path)]
    assert path.read_text() == original
    assert assistant.coordinator.history == []


def test_assistant_undo_subagent_delegation(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("before")
    client = ScriptedClient(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="ask_coder", raw_arguments='{"task": "t"}')
                ]
            ),
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="write_file",
                        raw_arguments=json.dumps(
                            {"path": str(path), "content": "after"}
                        ),
                    )
                ]
            ),
            ChatResult(content="done editing"),
            ChatResult(content="all done"),
        ]
    )
    coder = Agent("coder", tools=[write_file])
    assistant = Assistant(
        client=client, agents=[coder], permission_gate=AutoApproveGate()
    )
    assistant.chat("edit it")
    assert path.read_text() == "after"
    result = assistant.undo()
    assert result.restored == [str(path)]
    assert path.read_text() == "before"
    assert assistant.coordinator.history == []


def test_assistant_undo_all(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("v0")
    edit = lambda content: ChatResult(  # noqa: E731
        tool_calls=[
            ToolCall(
                id="c",
                name="write_file",
                raw_arguments=json.dumps({"path": str(path), "content": content}),
            )
        ]
    )
    client = ScriptedClient(
        [edit("v1"), ChatResult(content="one"), edit("v2"), ChatResult(content="two")]
    )
    assistant = Assistant(
        client=client, tools=[write_file], permission_gate=AutoApproveGate()
    )
    assistant.chat("turn one")
    assistant.chat("turn two")
    result = assistant.undo(2)
    assert result.restored == [str(path)]
    assert path.read_text() == "v0"
    assert assistant.coordinator.history == []
    assert len(assistant.undo_log) == 0


def test_assistant_undo_after_failed_edit(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("same")
    client = ScriptedClient(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="edit_file",
                        raw_arguments=json.dumps(
                            {"path": str(path), "old_string": "nope", "new_string": "x"}
                        ),
                    )
                ]
            ),
            ChatResult(content="could not find it"),
        ]
    )
    assistant = Assistant(
        client=client, tools=[edit_file], permission_gate=AutoApproveGate()
    )
    assistant.chat("edit it")
    result = assistant.undo()
    assert result.restored == []
    assert path.read_text() == "same"
    assert assistant.coordinator.history == []


def test_assistant_undo_after_interrupted_turn(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("v0")
    client = ScriptedClient(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        raw_arguments=json.dumps({"path": str(path), "content": "v1"}),
                    )
                ]
            )
        ]
    )
    assistant = Assistant(
        client=client, tools=[write_file], permission_gate=AutoApproveGate()
    )
    try:
        assistant.chat("edit it")
    except IndexError:
        pass  # scripted client runs out of results: the turn is interrupted
    assert path.read_text() == "v1"
    assert assistant.coordinator.history == []
    result = assistant.undo()
    assert result.restored == [str(path)]
    assert path.read_text() == "v0"


def test_assistant_reset_clears_undo_log(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("v0")
    client = ScriptedClient(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        raw_arguments=json.dumps({"path": str(path), "content": "v1"}),
                    )
                ]
            ),
            ChatResult(content="done"),
        ]
    )
    assistant = Assistant(
        client=client, tools=[write_file], permission_gate=AutoApproveGate()
    )
    assistant.chat("edit it")
    assistant.reset()
    assert assistant.coordinator.history == []
    assert len(assistant.undo_log) == 0
    assert assistant.undo().restored == []
    assert path.read_text() == "v1"


def _overwrite_tool(snapshot: bool):
    # A custom file-mutating tool; `snapshot` controls undo opt-in.
    @tool(snapshot=snapshot)
    def overwrite(path: str, content: str) -> str:
        """Overwrite a file with the given content."""
        with open(path, "w") as f:
            f.write(content)
        return f"wrote {path}"

    return overwrite


def _overwrite_client(path, content):
    return ScriptedClient(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="overwrite",
                        raw_arguments=json.dumps(
                            {"path": str(path), "content": content}
                        ),
                    )
                ]
            ),
            ChatResult(content="done"),
        ]
    )


def test_custom_snapshot_tool_is_undone(tmp_path):
    # A custom file-mutating tool opts into undo via @tool(snapshot=True).
    path = tmp_path / "c.txt"
    path.write_text("original")
    assistant = Assistant(
        client=_overwrite_client(path, "changed"),
        tools=[_overwrite_tool(snapshot=True)],
        permission_gate=AutoApproveGate(),
    )
    assistant.chat("change it")
    assert path.read_text() == "changed"
    result = assistant.undo()
    assert result.restored == [str(path)]
    assert path.read_text() == "original"


def test_non_snapshot_tool_is_not_undone(tmp_path):
    # A tool without snapshot=True does not get its file snapshotted.
    path = tmp_path / "c.txt"
    path.write_text("original")
    assistant = Assistant(
        client=_overwrite_client(path, "changed"),
        tools=[_overwrite_tool(snapshot=False)],
        permission_gate=AutoApproveGate(),
    )
    assistant.chat("change it")
    assert path.read_text() == "changed"
    result = assistant.undo()
    assert result.restored == []
    assert path.read_text() == "changed"  # not restored


def _mode(path):
    return os.stat(path).st_mode & 0o777


def test_write_atomic_preserves_existing_mode(tmp_path):
    from radix.undo import _write_atomic

    path = tmp_path / "a.txt"
    path.write_text("before")
    os.chmod(path, 0o640)
    _write_atomic(str(path), "after")
    assert path.read_text() == "after"
    assert _mode(path) == 0o640


def test_write_atomic_preserves_0600(tmp_path):
    from radix.undo import _write_atomic

    path = tmp_path / "secret.txt"
    path.write_text("before")
    os.chmod(path, 0o600)
    _write_atomic(str(path), "after")
    assert _mode(path) == 0o600


def test_write_atomic_new_file_uses_umask_default(tmp_path):
    from radix.undo import _write_atomic

    path = tmp_path / "new.txt"
    old = os.umask(0o022)
    try:
        _write_atomic(str(path), "hello")
    finally:
        os.umask(old)
    assert path.read_text() == "hello"
    assert _mode(path) == 0o644


def test_write_atomic_new_file_respects_umask(tmp_path):
    from radix.undo import _write_atomic

    path = tmp_path / "new.txt"
    old = os.umask(0o077)
    try:
        _write_atomic(str(path), "hello")
    finally:
        os.umask(old)
    assert _mode(path) == 0o600


def test_edit_file_preserves_mode(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello world")
    os.chmod(path, 0o640)
    out = edit_file.run(path=str(path), old_string="world", new_string="radix")
    assert not out.startswith("Error")
    assert path.read_text() == "hello radix"
    assert _mode(path) == 0o640


def test_write_file_new_mode_and_edit_preserves(tmp_path):
    path = tmp_path / "b.txt"
    old = os.umask(0o022)
    try:
        out = write_file.run(path=str(path), content="v0")
    finally:
        os.umask(old)
    assert not out.startswith("Error")
    assert _mode(path) == 0o644
    os.chmod(path, 0o604)
    out = edit_file.run(path=str(path), old_string="v0", new_string="v1")
    assert not out.startswith("Error")
    assert _mode(path) == 0o604


def test_undo_restore_preserves_mode(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("v0")
    os.chmod(path, 0o640)
    log = UndoLog()
    log.begin_turn(0)
    log.snapshot(str(path))
    path.write_text("v1")
    result = log.undo()
    assert result.restored == [str(path)]
    assert path.read_text() == "v0"
    assert _mode(path) == 0o640


def test_undo_restore_deleted_file_default_mode(tmp_path):
    # A file that existed, was snapshotted, then deleted during the turn:
    # undo recreates it, so it gets the default (umask) mode.
    path = tmp_path / "d.txt"
    path.write_text("v0")
    old = os.umask(0o022)
    try:
        log = UndoLog()
        log.begin_turn(0)
        log.snapshot(str(path))  # exists with content
        path.unlink()  # deleted during the turn
        result = log.undo()
    finally:
        os.umask(old)
    assert result.restored == [str(path)]
    assert path.read_text() == "v0"
    assert _mode(path) == 0o644
