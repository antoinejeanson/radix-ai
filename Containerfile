# radix sandbox: run a radix example (e.g. radix-code) with file edits,
# shell commands and web access confined to the container.
#
# Build:     podman build -t radix-sandbox .
# Run:       see examples/radix-code.py docstring
FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --no-cache-dir /app

WORKDIR /work