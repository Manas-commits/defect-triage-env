# Dockerfile — Manufacturing Defect Triage RL Environment
# Base: python:3.11-slim
# Exposes port 7860 (Hugging Face Spaces standard)

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install OS-level build dependencies (needed by scipy)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose Hugging Face Spaces port
EXPOSE 7860

# Health check so HF Spaces (and docker) can verify the container is up
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Launch FastAPI via uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]