from __future__ import annotations

import urllib.request

from ..tool import tool

# Built-in web tool: fetch_url retrieves HTTP(S) pages as text, with a
# line-limited output cap and a timeout, both configurable by the agent.
DEFAULT_MAX_BODY_LINES = 500
DEFAULT_TIMEOUT_SECONDS = 30


@tool
def fetch_url(
    url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_body_lines: int = DEFAULT_MAX_BODY_LINES,
) -> str:
    """Fetch a URL over HTTP(S) and return the response body as text.

    The body is capped at `max_body_lines` lines (default 500); when the
    page has more, a footer reports how many lines were cut so the agent
    can fetch a more targeted URL or endpoint. The request times out
    after `timeout` seconds (default 30). The body is decoded as UTF-8
    with replacement for invalid bytes.

    Args:
        url: The URL to fetch.
        timeout: Maximum seconds before the request is aborted; 0 means
            no timeout.
        max_body_lines: Maximum lines of body text returned, after which
            the output is cut with a footer; 0 means unlimited.

    Returns:
        The response body of the requested line window, or an error
        message.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "radix"})
    try:
        with urllib.request.urlopen(
            request, timeout=timeout if timeout > 0 else None
        ) as response:
            body = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Error: could not fetch {url}: {exc}"
    lines = body.splitlines()
    if max_body_lines > 0 and len(lines) > max_body_lines:
        cut = len(lines) - max_body_lines
        body = "\n".join(lines[:max_body_lines])
        body += f"\n... [body truncated — {cut} more lines]"
    return body
