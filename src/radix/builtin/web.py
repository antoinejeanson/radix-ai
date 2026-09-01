from __future__ import annotations

import urllib.request

from ..tool import tool

# Built-in web tool: fetch_url retrieves HTTP(S) pages as text, with a
# timeout and a body size cap.
MAX_BODY_CHARS = 16000
TIMEOUT_SECONDS = 30


@tool
def fetch_url(url: str) -> str:
    """Fetch a URL over HTTP(S) and return the response body as text.

    Args:
        url: The URL to fetch. Times out after TIMEOUT_SECONDS; the body is
            capped at MAX_BODY_CHARS characters and decoded as UTF-8 with
            replacement for invalid bytes.

    Returns:
        The response body, or an error message.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "radix"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(MAX_BODY_CHARS + 1).decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Error: could not fetch {url}: {exc}"
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n... [body truncated]"
    return body
