FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps: gcc for any C extensions, curl for the HEALTHCHECK probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies layer — cached as long as requirements.txt is unchanged
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code — layer invalidates only on code changes
COPY app/ app/

EXPOSE 8000

# Readiness probe — /ready returns 503 until model is loaded
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/ready || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
