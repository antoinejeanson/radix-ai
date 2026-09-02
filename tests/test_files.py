from radix.builtin import edit_file, read_file, write_file


# Tests for the built-in file tools.
def test_read_file(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_text("hello radix")
    assert read_file.run(path=str(path)) == "hello radix"


def test_read_file_missing():
    out = read_file.run(path="/nonexistent/nope.txt")
    assert out.startswith("Error")


def test_read_file_directory(tmp_path):
    out = read_file.run(path=str(tmp_path))
    assert "directory" in out


def test_read_file_binary(tmp_path):
    path = tmp_path / "bin.dat"
    path.write_bytes(b"\xff\xfe\x00\x01")
    out = read_file.run(path=str(path))
    assert "UTF-8" in out


def test_read_file_default_line_cap_reports_remaining(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("\n".join(f"line {i}" for i in range(1500)))
    out = read_file.run(path=str(path))
    assert out.startswith("[Showing lines 1-1000 of 1500]")
    assert "line 999" in out
    assert "line 1000" not in out
    assert "[500 more lines remain]" in out


def test_read_file_offset_pages_past_the_cap(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("\n".join(f"line {i}" for i in range(1500)))
    out = read_file.run(path=str(path), offset=1000)
    assert out == "line 1000\n" + "\n".join(f"line {i}" for i in range(1001, 1500))


def test_read_file_all_lines_with_negative_limit(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("\n".join(f"line {i}" for i in range(1500)))
    out = read_file.run(path=str(path), limit=-1)
    assert out == "\n".join(f"line {i}" for i in range(1500))


def test_read_file_offset_and_limit(tmp_path):
    path = tmp_path / "n.txt"
    path.write_text("\n".join(f"line {i}" for i in range(10)))
    assert read_file.run(path=str(path), offset=3, limit=7) == (
        "line 3\n" + "\n".join(f"line {i}" for i in range(4, 10))
    )


def test_read_file_partial_window_reports_remaining(tmp_path):
    path = tmp_path / "n.txt"
    path.write_text("\n".join(f"line {i}" for i in range(10)))
    out = read_file.run(path=str(path), offset=3, limit=2)
    assert out == "[Showing lines 4-5 of 10]\nline 3\nline 4\n[5 more lines remain]"


def test_read_file_limit_without_remaining_is_plain(tmp_path):
    path = tmp_path / "n.txt"
    path.write_text("\n".join(f"line {i}" for i in range(5)))
    assert read_file.run(path=str(path), limit=5) == (
        "line 0\nline 1\nline 2\nline 3\nline 4"
    )


def test_read_file_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    assert read_file.run(path=str(path)) == ""


def test_edit_file_replaces_snippet(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("def greet():\n    print('hi')\n")
    out = edit_file.run(
        path=str(path), old_string="print('hi')", new_string="print('hello')"
    )
    assert out.startswith(f"Edited {path}.")
    assert "-    print('hi')" in out
    assert "+    print('hello')" in out
    assert path.read_text() == "def greet():\n    print('hello')\n"


def test_edit_file_multiline(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("a = 1\nb = 2\nc = 3\n")
    edit_file.run(
        path=str(path), old_string="a = 1\nb = 2", new_string="a = 10\nb = 20"
    )
    assert path.read_text() == "a = 10\nb = 20\nc = 3\n"


def test_edit_file_not_found(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("hello")
    out = edit_file.run(path=str(path), old_string="nope", new_string="x")
    assert out.startswith("Error")
    assert path.read_text() == "hello"


def test_edit_file_ambiguous(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("x = 1\nx = 1\n")
    out = edit_file.run(path=str(path), old_string="x = 1", new_string="x = 2")
    assert out.startswith("Error")
    assert path.read_text() == "x = 1\nx = 1\n"


def test_edit_file_noop(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("hello")
    out = edit_file.run(path=str(path), old_string="hello", new_string="hello")
    assert out.startswith("Error")


def test_edit_file_missing_file(tmp_path):
    out = edit_file.run(path=str(tmp_path / "nope.txt"), old_string="a", new_string="b")
    assert out.startswith("Error")


def test_edit_file_directory(tmp_path):
    out = edit_file.run(path=str(tmp_path), old_string="a", new_string="b")
    assert "directory" in out


def test_edit_file_binary(tmp_path):
    path = tmp_path / "bin.dat"
    path.write_bytes(b"\xff\xfe\x00\x01")
    out = edit_file.run(path=str(path), old_string="a", new_string="b")
    assert "UTF-8" in out


def test_write_file_creates_with_parents(tmp_path):
    path = tmp_path / "a" / "b" / "new.txt"
    out = write_file.run(path=str(path), content="hello radix")
    assert out == f"Wrote 11 chars to {path}."
    assert path.read_text() == "hello radix"


def test_write_file_overwrites(tmp_path):
    path = tmp_path / "new.txt"
    path.write_text("old")
    write_file.run(path=str(path), content="new")
    assert path.read_text() == "new"


def test_write_file_directory(tmp_path):
    out = write_file.run(path=str(tmp_path), content="x")
    assert "directory" in out
