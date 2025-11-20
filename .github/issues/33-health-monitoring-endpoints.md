# Add Health Check and Monitoring Endpoints

## Overview
Implement standard health check endpoints for monitoring server status, readiness, and basic metrics.

## Dependencies
- Issue #30 (HTTP/HTTPS Transport Infrastructure)

## Implementation Tasks
- [ ] Add `/health` GET endpoint to both servers returning JSON with status "ok", uptime seconds, server name
- [ ] Add `/ready` GET endpoint checking MCP server initialization state
- [ ] Add `/metrics` GET endpoint with basic statistics: total_requests, active_connections, tools_available
- [ ] Track request counts and connection stats in FastAPI app state
- [ ] Implement graceful shutdown handler to cleanup SSE connections
- [ ] Close all active SSE streams on server shutdown
- [ ] Log shutdown process appropriately
- [ ] Test health endpoints return proper HTTP status codes

## Acceptance Criteria
- `/health` always returns 200 with uptime information
- `/ready` returns 200 when server is ready, 503 otherwise
- `/metrics` returns basic operational statistics
- Graceful shutdown closes connections cleanly
- All endpoints return properly formatted JSON

## Endpoint Specifications

### GET /health
**Purpose**: Basic liveness check  
**Returns**: Always 200 OK
```json
{
  "status": "ok",
  "server": "math-server",
  "uptime_seconds": 3600,
  "timestamp": "2025-11-20T12:00:00Z"
}
```

### GET /ready
**Purpose**: Readiness check for load balancers  
**Returns**: 200 if ready, 503 if not ready
```json
{
  "status": "ready",
  "mcp_initialized": true,
  "tools_count": 24
}
```

### GET /metrics
**Purpose**: Basic operational metrics  
**Returns**: 200 with statistics
```json
{
  "total_requests": 1523,
  "active_connections": 3,
  "tools_available": 24,
  "uptime_seconds": 3600,
  "requests_per_minute": 42.3
}
```

## Graceful Shutdown
- Register signal handlers (SIGTERM, SIGINT)
- Close all active SSE connections
- Wait for in-flight requests to complete (max 30s)
- Log shutdown completion
- Exit cleanly

## Use Cases
- Kubernetes liveness/readiness probes
- Load balancer health checks
- Monitoring dashboard integration
- Alerting systems

## Labels
enhancement, monitoring, health-checks
