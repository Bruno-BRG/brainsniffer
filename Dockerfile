FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Validate and consume uv.lock; do not substitute an unlocked CPU-only torch wheel.
# src/ and README.md are present for the non-editable Hatch package build.
RUN uv sync --locked --no-dev --no-editable --python /usr/local/bin/python --no-cache

COPY dash_app.py ./dash_app.py
COPY assets ./assets
COPY data ./data
COPY models ./models
COPY reports ./reports
COPY docs ./docs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/healthz', timeout=3)"

CMD ["gunicorn", "--bind", "0.0.0.0:8501", "--workers", "1", "--threads", "4", "--timeout", "120", "dash_app:server"]
