# Add Docker Deployment Support

## Overview
Create Docker containers and configuration for production deployment beyond GitHub Codespaces.

## Dependencies
- Issue #30 (HTTP/HTTPS Transport Infrastructure)
- Issue #31 (Configuration System)

## Implementation Tasks
- [ ] Create `Dockerfile` for containerized deployment
- [ ] Use multi-stage build for smaller image size
- [ ] Create `docker-compose.yml` for both servers
- [ ] Configure ports 8000 and 8001 mapping
- [ ] Mount config file as volume
- [ ] Add health checks in docker-compose
- [ ] Create `.dockerignore` file
- [ ] Create `nginx.conf` example for reverse proxy
- [ ] Document Docker deployment in README
- [ ] Add HTTPS termination example with nginx

## Acceptance Criteria
- Dockerfile builds successfully
- docker-compose starts both servers
- Health checks work in containers
- nginx reverse proxy example included
- Docker deployment documented
- Production-ready configuration provided

## Dockerfile

```dockerfile
# Multi-stage build for smaller images
FROM python:3.11-slim as builder

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
```

## Docker Compose

```yaml
version: '3.8'

services:
  math-server:
    build: .
    container_name: mcp-math-server
    ports:
      - "8000:8000"
    volumes:
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - MCP_AUTH_ENABLED=true
      - MCP_API_KEY=${MCP_API_KEY}
      - MCP_RATE_LIMIT_ENABLED=true
    command: python src/math_server/server.py --transport http --host 0.0.0.0 --port 8000 --config config.yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    networks:
      - mcp-network

  stats-server:
    build: .
    container_name: mcp-stats-server
    ports:
      - "8001:8001"
    volumes:
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - MCP_AUTH_ENABLED=true
      - MCP_API_KEY=${MCP_API_KEY}
      - MCP_RATE_LIMIT_ENABLED=true
    command: python src/stats_server/server.py --transport http --host 0.0.0.0 --port 8001 --config config.yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    networks:
      - mcp-network

  nginx:
    image: nginx:alpine
    container_name: mcp-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - math-server
      - stats-server
    restart: unless-stopped
    networks:
      - mcp-network

networks:
  mcp-network:
    driver: bridge
```

## Nginx Configuration

```nginx
events {
    worker_connections 1024;
}

http {
    upstream math_server {
        server math-server:8000;
    }

    upstream stats_server {
        server stats-server:8001;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=60r/m;

    server {
        listen 80;
        server_name mcp.example.com;

        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name mcp.example.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Math Server
        location /math/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://math_server/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # SSE specific
            proxy_buffering off;
            proxy_cache off;
            proxy_read_timeout 86400s;
        }

        # Stats Server
        location /stats/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://stats_server/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # SSE specific
            proxy_buffering off;
            proxy_cache off;
            proxy_read_timeout 86400s;
        }
    }
}
```

## .dockerignore

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
.env
.venv
venv/
.git
.github
.vscode
*.md
.dockerignore
Dockerfile
docker-compose.yml
.pids
tests/
*.log
```

## Usage

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

## Environment Variables

Create `.env` file:
```env
MCP_API_KEY=your-secure-api-key-here
MCP_AUTH_ENABLED=true
MCP_RATE_LIMIT_ENABLED=true
```

## Production Checklist
- [ ] Set strong API key in environment
- [ ] Enable authentication in config
- [ ] Configure CORS origins restrictively
- [ ] Set up SSL certificates for nginx
- [ ] Configure log rotation
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure backup strategy
- [ ] Set resource limits (CPU, memory)
- [ ] Enable container security scanning
- [ ] Document incident response procedures

## Labels
enhancement, docker, deployment
