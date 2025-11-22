# Multi-stage build for smaller images
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY config.example.yaml ./

# Ensure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Create non-root user
RUN useradd -m -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app

USER mcpuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default to HTTP transport
CMD ["python", "src/math_server/server.py", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
