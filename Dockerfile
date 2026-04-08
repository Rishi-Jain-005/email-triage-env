# Email Triage OpenEnv — Docker image
# Build:  docker build -t email-triage-env:latest .
# Run:    docker run -p 8000:8000 email-triage-env:latest

FROM python:3.11-slim

# ── system dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── application code ─────────────────────────────────────────────────────────
COPY . .

# ── environment variables ────────────────────────────────────────────────────
ENV PORT=8000
ENV WORKERS=4
ENV MAX_CONCURRENT_ENVS=100
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── expose port ──────────────────────────────────────────────────────────────
EXPOSE 8000

# ── health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── start server ─────────────────────────────────────────────────────────────
CMD uvicorn server.app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers ${WORKERS}