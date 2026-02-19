# Flotorch Evaluation MCP Server - Production Dockerfile
# Supports: Hugging Face Spaces, AWS EC2, and generic container deployments

FROM python:3.11-slim-bookworm AS builder

# Build stage: install dependencies and package
WORKDIR /build

COPY requirements.txt pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

# Production stage
FROM python:3.11-slim-bookworm

# Install git (required by ragas/GitPython)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (UID 1000 required for Hugging Face Spaces)
RUN useradd -m -u 1000 -s /bin/bash appuser

ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DEEPEVAL_TELEMETRY_OPT_OUT=1 \
    PORT=8080

WORKDIR /app

# Install runtime dependencies and package from builder wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Make /app writable for appuser (deepeval creates .deepeval here)
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (override with PORT env for Hugging Face: 7860)
EXPOSE 8080

# Health check (uses PORT env)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request, os; port=int(os.environ.get('PORT','8080')); urllib.request.urlopen(f'http://127.0.0.1:{port}/.well-known/flotorch-mcp')" || exit 1

# Run the MCP server
CMD ["python", "-u", "-m", "flotorch_eval_mcp"]
