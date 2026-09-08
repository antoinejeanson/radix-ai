# Tests for the read-only grep search tool.

from radix.builtin import grep


def test_grep_finds_match_across_nested_files(tmp_path):
    (tmp_path / "a.py").write_text("def alpha():\n    pass\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.py").write_text("x = 1\nbeta = 'needle'\n")
    out = grep.run(pattern="needle", path=str(tmp_path))
    assert "b.py" in out
    assert "needle" in out
    # format is file:line: text
    assert any(line.split(":")[1].strip() == "2" for line in out.splitlines())


def test_grep_on_a_single_file(tmp_path):
    p = tmp_path / "one.txt"
    p.write_text("hello\nworld\n")
    out = grep.run(pattern="world", path=str(p))
    assert "one.txt" in out
    assert "world" in out


def test_grep_no_matches(tmp_path):
    (tmp_path / "a.txt").write_text("nothing here\n")
    assert grep.run(pattern="zebra", path=str(tmp_path)) == "no matches"


def test_grep_respects_max_results(tmp_path):
    (tmp_path / "many.txt").write_text("\n".join(f"line {i} match" for i in range(10)))
    out = grep.run(pattern="match", path=str(tmp_path), max_results=3)
    lines = out.splitlines()
    assert len(lines) == 4  # 3 matches + footer
    assert lines[-1] == "... [7 more matches]"


def test_grep_skips_binary_files(tmp_path):
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02needle\x00")
    (tmp_path / "ok.txt").write_text("needle in text\n")
    out = grep.run(pattern="needle", path=str(tmp_path))
    assert "ok.txt" in out
    assert "bin.dat" not in out


def test_grep_skips_hidden_dirs(tmp_path):
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "secret.txt").write_text("needle in git\n")
    (tmp_path / "visible.txt").write_text("needle in visible\n")
    out = grep.run(pattern="needle", path=str(tmp_path))
    assert "visible.txt" in out
    assert ".git" not in out


def test_grep_invalid_regex(tmp_path):
    out = grep.run(pattern="([unclosed", path=str(tmp_path))
    assert out.startswith("Error: invalid regex")


def test_grep_missing_path(tmp_path):
    out = grep.run(pattern="x", path=str(tmp_path / "nope"))
    assert out.startswith("Error: no such file")


def test_grep_schema():
    props = grep.parameters["properties"]
    assert set(props) == {"pattern", "path", "max_results"}
    assert grep.parameters["required"] == ["pattern"]
