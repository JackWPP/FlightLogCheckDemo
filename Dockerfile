# syntax=docker/dockerfile:1.7
# Flight Log Check Demo — production image
# Build:  docker build -t flight-log-check-demo:latest .
# Run:    docker compose up -d
#
# Base image note: we use the official `astral-sh/uv` Python image so uv is
# already on PATH. This avoids a slow/failed `pip install uv` step that hits
# PyPI from networks where pypi.org is blocked or throttled (e.g. mainland
# China, some corporate networks). The project deps themselves are pulled via
# UV_INDEX_URL → Tsinghua mirror below for the same reason.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8003 \
    # Tsinghua mirror — swap to https://mirrors.aliyun.com/pypi/simple/ or
    # https://pypi.org/simple/ if you prefer. This affects `uv sync` only;
    # `uv` itself comes from the base image.
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

# --- 1. Dependency layer (cached unless pyproject/uv.lock change) ----------------
COPY pyproject.toml uv.lock ./
# --frozen honors uv.lock exactly; --no-dev skips pytest etc.; --no-install-project
# defers installing the editable package until we copy the source.
RUN uv sync --python 3.12 --frozen --no-dev --no-install-project

# --- 2. Application layer --------------------------------------------------------
COPY src ./src
COPY static ./static
COPY assets ./assets
COPY docs ./docs
COPY fields.yaml ./
COPY outputs/demo_sample ./outputs/demo_sample

# Now install the project itself (editable, so 'formcheck' is importable).
RUN uv sync --python 3.12 --frozen --no-dev

# Pre-create runtime output dirs (mounted volumes overlay these at run time).
RUN mkdir -p /app/out /app/outputs/runtime

# --- 3. Runtime config -----------------------------------------------------------
# NOTE on user / permissions:
# The container intentionally runs as root so the bind-mounted host dirs
# (./out, ./outputs/runtime — owned by the host's deploy user, e.g. uid 1004)
# are writable by uvicorn. If the host dir is owned by some other uid than
# 10001, the previous `USER appuser` setup would fail with PermissionError
# at runtime.
#
# To harden for production, switch back to a non-root user and add an
# entrypoint that chowns /app/out and /app/outputs/runtime at startup:
#   RUN useradd --create-home --shell /bin/bash --uid 10001 appuser \
#       && chown -R appuser:appuser /app
#   USER root                              # entrypoint needs root to chown
#   RUN apt-get update -qq && apt-get install -y --no-install-recommends gosu \
#       && rm -rf /var/lib/apt/lists/*
#   COPY deploy/entrypoint.sh /entrypoint.sh
#   RUN chmod +x /entrypoint.sh
#   ENTRYPOINT ["/entrypoint.sh"]
#   USER appuser
# And in entrypoint.sh:
#   #!/bin/sh
#   set -e
#   mkdir -p /app/out /app/outputs/runtime
#   chown -R appuser:appuser /app/out /app/outputs/runtime
#   exec gosu appuser "$@"

EXPOSE 8003

# Workers=1 keeps the in-process SQLite task worker simple for the V2 demo.
# --proxy-headers so X-Forwarded-* from nginx reaches FastAPI.
# --timeout-keep-alive 120 because PP-OCRv6 + VLM review can take 30-90s.
CMD ["uv", "run", "--python", "3.12", "--no-dev", \
     "uvicorn", "formcheck.app:app", \
     "--host", "0.0.0.0", "--port", "8003", \
     "--workers", "1", \
     "--proxy-headers", \
     "--timeout-keep-alive", "120"]
