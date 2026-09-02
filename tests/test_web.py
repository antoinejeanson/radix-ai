import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from radix.builtin import fetch_url


# Tests for the built-in fetch_url tool: line cap and agent-configurable
# parameters, using an in-process HTTP server.
class _Handler(BaseHTTPRequestHandler):
    body = ""

    def do_GET(self):
        payload = self.body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    handler = _Handler
    handler.body = "\n".join(f"line {i}" for i in range(600))
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/"
    httpd.shutdown()
    thread.join()


def test_fetch_url_returns_body(server):
    out = fetch_url.run(url=server)
    assert "line 0" in out
    assert "line 499" in out
    assert "line 500" not in out


def test_fetch_url_line_cap_reports_cut(server):
    out = fetch_url.run(url=server, max_body_lines=500)
    assert "line 0" in out
    assert "line 499" in out
    assert "line 500" not in out
    assert "... [body truncated — 100 more lines]" in out


def test_fetch_url_no_line_cap_when_unlimited(server):
    out = fetch_url.run(url=server, max_body_lines=0)
    assert "line 599" in out
    assert "[body truncated" not in out


def test_fetch_url_error():
    out = fetch_url.run(url="http://127.0.0.1:1/nope")
    assert out.startswith("Error: could not fetch")
