# syntax=docker/dockerfile:1

### ---- Base: install uv ----
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /uvx /usr/local/bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1
WORKDIR /app

### ---- Dependency layer (cached separately from source for fast rebuilds) ----
FROM base AS deps
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project --no-dev || uv sync --no-install-project --no-dev

### ---- Final image ----
FROM base AS final
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev || uv sync --no-dev

# artifacts/ and data/ are expected to be mounted or trained into at runtime
RUN mkdir -p artifacts data

EXPOSE 8000

CMD ["uvicorn", "student_success.main:app", "--host", "0.0.0.0", "--port", "8000"]
