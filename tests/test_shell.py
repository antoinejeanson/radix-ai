from radix.builtin import run_shell


# Tests for the built-in run_shell tool: line cap, timeout, and the
# agent-configurable parameters.
def test_run_shell_echo():
    out = run_shell.run(command="echo hello")
    assert out.startswith("exit code: 0")
    assert "hello" in out


def test_run_shell_merges_stderr():
    out = run_shell.run(command="echo out; echo err >&2")
    assert "exit code: 0" in out
    assert "out" in out
    assert "stderr:" in out
    assert "err" in out


def test_run_shell_line_cap_reports_cut(tmp_path):
    script = tmp_path / "many.sh"
    script.write_text("seq 1 600\n")
    out = run_shell.run(command=f"sh {script}", max_output_lines=500)
    assert "exit code: 0" in out
    assert "... [output truncated — 100 more lines]" in out


def test_run_shell_no_line_cap_when_unlimited(tmp_path):
    script = tmp_path / "many.sh"
    script.write_text("seq 1 10\n")
    out = run_shell.run(command=f"sh {script}", max_output_lines=0)
    assert "exit code: 0" in out
    assert "[output truncated" not in out


def test_run_shell_timeout():
    out = run_shell.run(command="sleep 5", timeout=0.01)
    assert out.startswith("Error: command timed out")
    assert "after 0.01 seconds" in out


def test_run_shell_no_timeout_when_zero():
    out = run_shell.run(command="sleep 0.01", timeout=0)
    assert out.startswith("exit code: 0")
