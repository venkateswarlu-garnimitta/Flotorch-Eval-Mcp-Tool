# Flotorch Evaluation MCP Server - Hugging Face Spaces
# Uses PORT=7860 (HF Spaces default)

FROM python:3.11-slim-bookworm AS builder
WORKDIR /build
COPY requirements.txt pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.11-slim-bookworm

# Install git (required by ragas/GitPython)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -s /bin/bash appuser

ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_ARENA_MAX=2 \
    DEEPEVAL_TELEMETRY_OPT_OUT=1 \
    PORT=7860

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Make /app writable for appuser (deepeval creates .deepeval here)
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/.well-known/flotorch-mcp')" || exit 1

CMD ["python", "-u", "-m", "flotorch_eval_mcp"]
