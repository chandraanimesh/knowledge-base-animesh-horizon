# Multi-stage production Dockerfile for Enterprise Adaptive RAG Backend
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PYTHONPATH="/app/src:${PYTHONPATH}"

WORKDIR /app

# Install minimal OS build dependencies and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official Astral image for fast, reproducible dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency manifests first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies into system Python environment
RUN uv pip install --system -e .

# Copy application source code and migrations
COPY alembic.ini ./
COPY migrations ./migrations
COPY scripts ./scripts
COPY src ./src

# Create storage directories and set permissions
RUN mkdir -p /app/storage/artifacts && chmod +x /app/scripts/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/scripts/entrypoint.sh"]

