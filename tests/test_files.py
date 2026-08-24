from radix.builtin import read_file


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
