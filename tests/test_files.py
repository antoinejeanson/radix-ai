from radix.builtin import edit_file, read_file, write_file


def test_read_file(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_text("hello radix")
    assert read_file.run(path=str(path)) == "hello radix"


def test_read_file_not_sensitive():
    assert read_file.sensitive is False


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


def test_read_file_truncation(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("x" * 20000)
    out = read_file.run(path=str(path))
    assert out.endswith("[content truncated]")
    assert len(out) < 20000


def test_edit_file_not_sensitive():
    assert edit_file.sensitive is False


def test_write_file_not_sensitive():
    assert write_file.sensitive is False


def test_edit_file_replaces_snippet(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("def greet():\n    print('hi')\n")
    out = edit_file.run(path=str(path), old_string="print('hi')", new_string="print('hello')")
    assert out.startswith(f"Edited {path}.")
    assert "-    print('hi')" in out
    assert "+    print('hello')" in out
    assert path.read_text() == "def greet():\n    print('hello')\n"


def test_edit_file_multiline(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("a = 1\nb = 2\nc = 3\n")
    edit_file.run(path=str(path), old_string="a = 1\nb = 2", new_string="a = 10\nb = 20")
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
