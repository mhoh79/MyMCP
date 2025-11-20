# Update Documentation and Configuration

## Overview
Comprehensive documentation update covering HTTP transport usage, Codespaces setup, configuration options, and troubleshooting.

## Dependencies
- All previous issues (#30-36)

## Implementation Tasks
- [ ] Update `README.md` with new "HTTP/HTTPS Transport" section
- [ ] Add quick start guide for HTTP mode
- [ ] Document connection examples for local and Codespaces
- [ ] Document all `config.yaml` options with descriptions
- [ ] Add GitHub Codespaces setup instructions
- [ ] Document public URL format (`https://<codespace>-<port>.app.github.dev`)
- [ ] Add security best practices section (API key management)
- [ ] Create troubleshooting section (connection issues, CORS errors, authentication failures)
- [ ] Add architecture diagram showing stdio vs HTTP transport
- [ ] Update `mcp-config.json` to fix paths (`math_server`, `stats_server`)
- [ ] Add example client code snippets for connecting via HTTP

## Acceptance Criteria
- README has complete HTTP transport documentation
- All configuration options documented
- Codespaces setup is clear and tested
- Troubleshooting guide covers common issues
- Architecture diagram illustrates both transport modes
- mcp-config.json points to correct server files

## Documentation Sections

### 1. Quick Start - HTTP Mode
```bash
# Install dependencies
pip install -r requirements.txt

# Copy example config
cp config.example.yaml config.yaml

# Start servers
./start-http-servers.sh

# Test connection
curl http://localhost:8000/health
```

### 2. Configuration Reference
Document all config.yaml options:
- `server.math.host` - Math server bind address
- `server.math.port` - Math server port (default: 8000)
- `server.stats.host` - Stats server bind address
- `server.stats.port` - Stats server port (default: 8001)
- `server.cors_origins` - Allowed CORS origins (array)
- `authentication.enabled` - Enable API key auth (boolean)
- `authentication.api_key` - API key for authentication (string)
- `rate_limiting.enabled` - Enable rate limiting (boolean)
- `rate_limiting.requests_per_minute` - Rate limit threshold (integer)
- `logging.level` - Log level (DEBUG, INFO, WARNING, ERROR)

### 3. GitHub Codespaces Setup
Step-by-step guide:
1. Create new Codespace from repository
2. Wait for automatic setup to complete
3. Check terminal for server URLs
4. Access public URLs in browser or client
5. Configure MCP client with Codespaces URLs

### 4. Security Best Practices
- Never commit `config.yaml` with real API keys
- Use environment variables for production secrets
- Rotate API keys regularly
- Enable authentication in production
- Use HTTPS in production (not HTTP)
- Configure CORS origins restrictively
- Monitor `/metrics` for unusual activity

### 5. Troubleshooting Guide

**Problem**: Connection refused  
**Solution**: Check server is running, verify port, check firewall

**Problem**: CORS errors in browser  
**Solution**: Add origin to `cors_origins` in config

**Problem**: 401 Unauthorized  
**Solution**: Check API key is correct, verify auth is enabled

**Problem**: 429 Rate limit exceeded  
**Solution**: Reduce request rate or disable rate limiting

**Problem**: SSE connection drops  
**Solution**: Check network stability, verify keepalive settings

### 6. Architecture Diagram
```
┌─────────────────────────────────────────────┐
│           MCP Server (Dual Mode)            │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │ STDIO Mode   │      │ HTTP Mode       │ │
│  │              │      │                 │ │
│  │ stdin/stdout │      │ FastAPI + SSE   │ │
│  │ JSON-RPC     │      │ /sse endpoint   │ │
│  │              │      │ /messages POST  │ │
│  │ Claude       │      │ Web clients     │ │
│  │ Desktop      │      │ Codespaces      │ │
│  └──────────────┘      └─────────────────┘ │
│                                             │
│         ┌───────────────────┐               │
│         │   MCP Tools       │               │
│         │   (24 tools)      │               │
│         └───────────────────┘               │
└─────────────────────────────────────────────┘
```

### 7. Client Connection Examples

**Python Client**:
```python
import httpx
from sse_client import EventSource

# Connect to SSE stream
async with httpx.AsyncClient() as client:
    headers = {"Authorization": "Bearer your-api-key"}
    
    # Listen to events
    async with EventSource(
        "http://localhost:8000/sse",
        headers=headers
    ) as source:
        async for event in source:
            print(event.data)
    
    # Call tool
    response = await client.post(
        "http://localhost:8000/messages",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "fibonacci",
                "arguments": {"n": 10}
            },
            "id": 1
        },
        headers=headers
    )
```

**cURL Examples**:
```bash
# Health check
curl http://localhost:8000/health

# Call tool (with auth)
curl -X POST http://localhost:8000/messages \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "is_prime",
      "arguments": {"number": 17}
    },
    "id": 1
  }'
```

## Labels
documentation
