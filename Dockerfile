# ================================
# Stage 1: Builder
# ================================
FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_HTTP_TIMEOUT=300

COPY pyproject.toml uv.lock /app/

# Cache mount survives failed builds - resumes downloads
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY app/ /app/app/


# ================================
# Stage 2: Production
# ================================
FROM python:3.12-slim AS production

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app   /app/app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4" "--reload"]