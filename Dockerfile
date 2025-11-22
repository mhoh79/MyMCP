# Multi-stage build for smaller images
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

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

# Health check (port determined by CMD or environment)
# Default health check for port 8000, override via HEALTHCHECK_PORT env var
ENV HEALTHCHECK_PORT=8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${HEALTHCHECK_PORT}/health || exit 1

# Default to HTTP transport
CMD ["python", "src/math_server/server.py", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
