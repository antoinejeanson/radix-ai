# Tests for the persistent memory tools (remember / recall).

import radix.builtin.memory as memory
from radix.builtin import recall, remember


def _point_notes_at(tmp_path, monkeypatch):
    notes = tmp_path / "notes.md"
    monkeypatch.setattr(memory, "NOTES_PATH", str(notes))
    return notes


def test_remember_appends_a_line(tmp_path, monkeypatch):
    notes = _point_notes_at(tmp_path, monkeypatch)
    out = remember.run(text="the deploy key is in ~/.ssh")
    assert "Remembered" in out
    content = notes.read_text()
    assert "the deploy key is in ~/.ssh" in content
    assert content.startswith("- [")


def test_remember_appends_not_overwrites(tmp_path, monkeypatch):
    notes = _point_notes_at(tmp_path, monkeypatch)
    remember.run(text="first note")
    remember.run(text="second note")
    content = notes.read_text()
    assert "first note" in content
    assert "second note" in content
    assert len(content.splitlines()) == 2


def test_recall_finds_a_note(tmp_path, monkeypatch):
    _point_notes_at(tmp_path, monkeypatch)
    remember.run(text="prefer tabs over spaces")
    assert "prefer tabs over spaces" in recall.run(query="tabs")


def test_recall_is_case_insensitive(tmp_path, monkeypatch):
    _point_notes_at(tmp_path, monkeypatch)
    remember.run(text="Use PostgreSQL for the DB")
    assert "PostgreSQL for the DB" in recall.run(query="postgresql")


def test_recall_absent_term(tmp_path, monkeypatch):
    _point_notes_at(tmp_path, monkeypatch)
    remember.run(text="something else")
    assert recall.run(query="zebra") == "no matching notes"


def test_recall_missing_file(tmp_path, monkeypatch):
    _point_notes_at(tmp_path, monkeypatch)  # file is never created
    assert recall.run(query="anything") == "no matching notes"


def test_notes_file_is_the_source_of_truth(tmp_path, monkeypatch):
    # The note lives in the file, not in the Tool object, so it survives
    # across "sessions" (fresh agents) by construction.
    notes = _point_notes_at(tmp_path, monkeypatch)
    remember.run(text="decision: use uv for deps")
    assert "use uv for deps" in recall.run(query="uv")
    assert "use uv for deps" in notes.read_text()


def test_recall_caps_matches_with_footer(tmp_path, monkeypatch):
    _point_notes_at(tmp_path, monkeypatch)
    for i in range(60):
        remember.run(text=f"note {i} about x")
    out = recall.run(query="about x")
    lines = out.splitlines()
    assert len(lines) == memory.MAX_RECALL_LINES + 1
    assert lines[-1] == f"... [{60 - memory.MAX_RECALL_LINES} more matches]"


def test_remember_schema_only_exposes_text():
    assert list(remember.parameters["properties"]) == ["text"]
    assert list(recall.parameters["properties"]) == ["query"]
