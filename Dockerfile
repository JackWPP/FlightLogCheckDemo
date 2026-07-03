# syntax=docker/dockerfile:1.7
# Flight Log Check Demo — production image
# Build:  docker build -t flight-log-check-demo:latest .
# Run:    docker compose up -d

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8003

# Install uv (dependency manager pinned by uv.lock). pip is already present in the slim image.
RUN pip install --no-cache-dir "uv>=0.4"

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
# Drop privileges.
RUN useradd --create-home --shell /bin/bash --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8003

# Workers=2 is comfortable for a demo; raise if you front it with many users.
# --proxy-headers so X-Forwarded-* from nginx reaches FastAPI.
# --timeout-keep-alive 120 because PP-OCRv6 + VLM review can take 30-90s.
CMD ["uv", "run", "--python", "3.12", "--no-dev", \
     "uvicorn", "formcheck.app:app", \
     "--host", "0.0.0.0", "--port", "8003", \
     "--workers", "2", \
     "--proxy-headers", \
     "--timeout-keep-alive", "120"]